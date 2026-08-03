# Changelog

## 2026-08-03

- windowed cog source: `CogSrc` over terrano's `CogReader` with per-pull overview selection and block-averaged decimation past the file pyramid, `HttpRange` transport for remote files via http range requests, tests proving equality with the in-memory source, overview byte savings, nan padding outside the file, and an end-to-end http pull
- initial engine: caps solver adapted from glass2glass (declarative field masks instead of closures), window-native pull with chunk snapping on a power-of-two ladder, per-node in-memory LRU cache unified with in-flight coalescing, cancellation-safe pending guards, downstream invalidation with halo/projection spread and subscriber events
- elements: in-memory GeoTIFF source with block-averaged ladder levels, reproject via projicio, hillshade and slope via terrano with seam-free halo planning, map algebra and reclassify, xyz tile adapter, png and geotiff encoders
- drivers: axum tile server example and batch pyramid example over one shared graph, output verified byte-identical between the two
- tests: negotiation fixation and failure naming, chunk-seam equality against whole-raster hillshade, pull coalescing, cancellation recovery, cache invalidation semantics, batch materialization, reprojected tile accuracy
