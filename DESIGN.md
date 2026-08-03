# Geoplumb design

Demand-driven geospatial compute engine. A distinct layer from geodukt (batch ETL to files) and fluvius (event streams): geoplumb computes at request time. With nubis it is the planned PDAL replacement.

## Data plane: pull only

The data plane is pull, permanently. A pull names a window (bbox plus target ground resolution). It flows sink to source, each transform rewriting it upstream via `plan`: a kernel widens by its halo, a reproject inverse-projects the bbox and rescales resolution by the local scale factor. Chunks flow back down, each transform producing exactly its requested grid via `compute`.

- Batch is a degenerate pull: enumerate the chunk grid over an extent (`materialize`, or the pyramid example's tile loop) and pull everything.
- Live is invalidation plus re-pull: `Engine::invalidate` walks a dirty window downstream (widened by each element's `spread`, projected across CRS changes), drops overlapping cache entries, and publishes `Invalidation` events for drivers to re-render.
- Event-stream processing (per-event latency, geofencing) is out of scope, that is fluvius.

## Caps negotiation

Adapted from glass2glass's CSP design (`src/caps.rs`, `src/solver.rs`, MPL-2.0). Elements declare a `Constraint`:

- `Produces(CapsSet)` — source shape
- `Identity(CapsSet)` — pass-through, optionally narrowing
- `Derived { input, passthrough, output }` — output derived from input, declarative: a `FieldMask` names the fields copied across, the override pattern pins retargeted fields (reproject pins `crs`)

`CapsSet` is preference-ordered alternatives of field patterns (dtype, bands, crs, resolution range, chunk px). The solver alternates forward sweeps (narrow each output link through the node's constraint) and backward sweeps (consumers narrow producers, coupling backward through `Derived` only on passthrough fields) until the links stop changing: a fanin couples its input links through the shared consumer, so one sweep each way is no longer complete. Fixation then assigns concrete caps source-first by backtracking search, because a diamond can leave branch preference orders that disagree, where greedy per-link fixation picks a jointly impossible combination. The greedy choice is tried first, so chains and plain fan-out fixate as before. Failures are structured and name the link.

A fanin declares one `Constraint` that applies to every input link and the output link alike, so all pins negotiate to matching caps.

An empty link is not always a failure. The graph carries a registry of `Adapter`s: a template constraint with the element's retargeted fields left free, plus a factory. On a failing link the solver checks each adapter generically, `template.output_set(offer) ∩ demand`, and splices the built element when the bridge is nonempty, so the solver never knows which caps field an adapter fixes. `Graph::new` registers reproject (bridging crs, the target being whatever the demanding side prefers, at a fanin the first parent's), `register_adapter` adds more. Spliced nodes are appended after their consumers, so the engine and solver walk `topo_order` rather than index order.

Divergences from g2g, deliberate:

- Resolution stays a range on fixated caps and is resolved per pull by ladder snapping, because in a pull engine resolution is request-time, not negotiation-time.
- Field coupling is declarative (masks) rather than closure-based. Geo caps fields are independent enough that the closure machinery and passthrough probing are unnecessary.

## Chunks and snapping

Requests snap to a per-node grid: a power-of-two resolution ladder anchored at the node's origin (`GridSpec`), tiled at the negotiated chunk size. Cache keys are `(node, level, ix, iy)`. A request finer than the ladder base clamps to level 0 and the driver upsamples, coarser requests snap to the finest level that is at least as coarse.

Chunks are self-describing (`RasterChunk`: bands, resolved bbox, resolution, crs) because responses are addressed by request, not stream order, and snapping may widen the window. `Chunk` is an enum, vector and point cloud variants are reserved.

Each node's grid derives from its parent's through `output_grid`: identity for most transforms, a reproject anchors the canonical origin for known CRS (web mercator, wgs84) and estimates base resolution from the local scale. A fanin takes its finest input grid by default, and its elements sample inputs bilinearly onto the output grid, exact when an input shares the output alignment.

## Runtime

One map per engine is both cache and coalescing table: a chunk entry is `Ready` (cached, LRU by tick, byte-budgeted), `Spilled` (on disk only), or `Pending` (in flight, with waiters). Concurrent pulls of one chunk share the computation. A cancelled computer's drop guard removes its `Pending` entry and wakes waiters, one of which takes over, so cancellation never wedges a chunk. Errors are not cached.

The disk tier (`Engine::with_disk_cache`) writes every computed chunk through to a flat binary file and marks the entry spilled once the file is safely down, so memory eviction demotes to `Spilled` instead of dropping. A spilled hit reloads through the same pending machinery, coalescing concurrent readers. The disk budget counts cold bytes only and evicts files LRU. The store is a fresh subdir per engine, removed on drop, so files never outlive the caps and elements they were computed under, which is what makes persistence safe without an element identity hash.

Transforms' `compute` is synchronous by contract, the engine offloads it to tokio's blocking pool when a runtime is present and runs inline otherwise. Source `read` is async for future ranged IO.

Engine caps are immutable per instance (solve happens in `Engine::new`), so cache keys need no caps fingerprint until re-solve exists.

## Elements

Thin wrappers over sibling crates: terrano-core for kernels and GeoTIFF IO, projicio-core for CRS transforms. The engine repo carries orchestration, not math.

Three sources. `RasterSrc` holds the whole dataset resident and serves ladder levels by block-averaged decimation. `CogSrc` reads windowed over terrano's `CogReader`: each pull fetches only the tiles it touches at the file overview nearest the requested ladder level, block-averaging the remainder when the file pyramid is shallower than the request. Its transport is the `RangeRead` seam, with a local file impl in terrano and a blocking-reqwest `HttpRange` here for remote files. `StacSrc` searches a STAC api once at open, keeps the matched items' cog assets as lazily-opened `HttpRange` readers (s3 hrefs rewritten to their public https form), and serves pulls by mosaicking items most-recent-first, so a whole collection behaves as one raster with no local data.

Two fanin elements. `Mosaic` takes the first input, wiring order, that has a value at each output pixel. `Combine` samples both inputs onto the output grid and runs terrano's binary op band by band.

## Known limits

- The disk tier lives and dies with one engine instance. Cross-process reuse needs an element identity hash so a graph edit cannot serve stale files.
- `CogSrc` and `StacSrc` are single-band and serialize range reads per file behind a mutex. `StacSrc` uses only the first search page, keeps only items sharing the most recent item's crs, and assumes items share the grid alignment (mismatches land on the nearest pixel). No time axis on pulls, `datetime` filters at open.
- Invalidation spread uses the coarsest cached level per node, over-invalidating slightly, never stale.
