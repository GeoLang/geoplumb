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

`CapsSet` is preference-ordered alternatives of field patterns (dtype, bands, crs, resolution range, chunk px). The solver runs a forward sweep (narrow each output link), a backward sweep (consumers narrow producers, coupling backward through `Derived` only on passthrough fields), then fixates source-first so children fixate against concrete parent caps. Failures are structured and name the link.

Divergences from g2g, deliberate:

- Resolution stays a range on fixated caps and is resolved per pull by ladder snapping, because in a pull engine resolution is request-time, not negotiation-time.
- Field coupling is declarative (masks) rather than closure-based. Geo caps fields are independent enough that the closure machinery and passthrough probing are unnecessary.
- Single-input nodes only, so the constraint graph is a forest and one sweep each way is complete. Fan-in (mosaic, multi-input algebra) brings back g2g's backtracking fixation.

## Chunks and snapping

Requests snap to a per-node grid: a power-of-two resolution ladder anchored at the node's origin (`GridSpec`), tiled at the negotiated chunk size. Cache keys are `(node, level, ix, iy)`. A request finer than the ladder base clamps to level 0 and the driver upsamples, coarser requests snap to the finest level that is at least as coarse.

Chunks are self-describing (`RasterChunk`: bands, resolved bbox, resolution, crs) because responses are addressed by request, not stream order, and snapping may widen the window. `Chunk` is an enum, vector and point cloud variants are reserved.

Each node's grid derives from its parent's through `output_grid`: identity for most transforms, a reproject anchors the canonical origin for known CRS (web mercator, wgs84) and estimates base resolution from the local scale.

## Runtime

One map per engine is both cache and coalescing table: a chunk entry is `Ready` (cached, LRU by tick, byte-budgeted) or `Pending` (in flight, with waiters). Concurrent pulls of one chunk share the computation. A cancelled computer's drop guard removes its `Pending` entry and wakes waiters, one of which takes over, so cancellation never wedges a chunk. Errors are not cached.

Transforms' `compute` is synchronous by contract, the engine offloads it to tokio's blocking pool when a runtime is present and runs inline otherwise. Source `read` is async for future ranged IO.

Engine caps are immutable per instance (solve happens in `Engine::new`), so cache keys need no caps fingerprint until re-solve exists.

## Elements

Thin wrappers over sibling crates: terrano-core for kernels and GeoTIFF IO, projicio-core for CRS transforms. The engine repo carries orchestration, not math.

Two sources. `RasterSrc` holds the whole dataset resident and serves ladder levels by block-averaged decimation. `CogSrc` reads windowed over terrano's `CogReader`: each pull fetches only the tiles it touches at the file overview nearest the requested ladder level, block-averaging the remainder when the file pyramid is shallower than the request. Its transport is the `RangeRead` seam, with a local file impl in terrano and a blocking-reqwest `HttpRange` here for remote files.

## Known limits

- Fan-in is not modeled: no mosaic of multiple sources, no two-input algebra.
- Reproject auto-plug (splicing on CRS mismatch) is deferred, graphs wire reproject explicitly.
- The cache is in-memory only, a disk tier belongs behind the same map.
- `CogSrc` covers the subset terrano writes (uncompressed single-band f64) and serializes range reads per source behind a mutex. No time axis.
- Invalidation spread uses the coarsest cached level per node, over-invalidating slightly, never stale.
