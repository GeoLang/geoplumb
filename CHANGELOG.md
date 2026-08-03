# Changelog

## 2026-08-03

- disk cache tier behind the same entry map: `Engine::with_disk_cache` writes chunks through to a per-engine spill dir, memory eviction demotes entries to disk, spilled hits reload through the coalescing machinery, disk budget and invalidation delete files, drop removes the dir
- auto-plug generalized to an adapter registry on the graph: elements declare what they bridge via a template constraint and the solver checks `output_set(offer) ∩ demand` with no field knowledge, reproject registered by default, `register_adapter` for user elements
- reproject auto-plug: a link empty only because of crs gets a reproject spliced in during the solve (mixed-crs mosaic, crs-demanding consumer), spliced nodes break index-as-topo-order so the solver, engine construction, and invalidation now walk an explicit topo order
- fan-in: `Fanin` nodes with several parents, solver reworked to iterated sweeps plus backtracking fixation (a diamond with disagreeing branch preferences now converges), `Mosaic` (first-wins stitching) and `Combine` (two-input per-cell algebra) elements, invalidation walks through fanin nodes, bilinear fallback now picks the nearest present neighbor instead of an arbitrary one
- windowed cog source: `CogSrc` over terrano's `CogReader` with per-pull overview selection and block-averaged decimation past the file pyramid, `HttpRange` transport for remote files via http range requests, tests proving equality with the in-memory source, overview byte savings, nan padding outside the file, and an end-to-end http pull
- initial engine: caps solver adapted from glass2glass (declarative field masks instead of closures), window-native pull with chunk snapping on a power-of-two ladder, per-node in-memory LRU cache unified with in-flight coalescing, cancellation-safe pending guards, downstream invalidation with halo/projection spread and subscriber events
- elements: in-memory GeoTIFF source with block-averaged ladder levels, reproject via projicio, hillshade and slope via terrano with seam-free halo planning, map algebra and reclassify, xyz tile adapter, png and geotiff encoders
- drivers: axum tile server example and batch pyramid example over one shared graph, output verified byte-identical between the two
- tests: negotiation fixation and failure naming, chunk-seam equality against whole-raster hillshade, pull coalescing, cancellation recovery, cache invalidation semantics, batch materialization, reprojected tile accuracy
