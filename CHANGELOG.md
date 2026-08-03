# Changelog

## 2026-08-03

- initial engine: caps solver adapted from glass2glass (declarative field masks instead of closures), window-native pull with chunk snapping on a power-of-two ladder, per-node in-memory LRU cache unified with in-flight coalescing, cancellation-safe pending guards, downstream invalidation with halo/projection spread and subscriber events
- elements: in-memory GeoTIFF source with block-averaged ladder levels, reproject via projicio, hillshade and slope via terrano with seam-free halo planning, map algebra and reclassify, xyz tile adapter, png and geotiff encoders
- drivers: axum tile server example and batch pyramid example over one shared graph, output verified byte-identical between the two
- tests: negotiation fixation and failure naming, chunk-seam equality against whole-raster hillshade, pull coalescing, cancellation recovery, cache invalidation semantics, batch materialization, reprojected tile accuracy
