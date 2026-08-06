# Geoplumb

[![CI](https://github.com/GeoLang/geoplumb/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/geoplumb/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

A pull-based geospatial compute pipeline for the GeoLang GIS stack.

Build a dag of sources and transforms, negotiate caps once, then pull windows: a demand (bbox plus resolution) flows sink to source, each element rewriting it on the way up (kernel halo, inverse projection), and chunks flow back, cached and coalesced per node. Nothing is computed until something asks, and only the asked-for window is computed.

- **Caps negotiation** — every element declares a constraint over its link caps (dtype, bands, CRS, resolution range, chunk size), per chunk kind (raster, point cloud, vector or tensor). A solver adapted from [glass2glass](https://github.com/Glass2GlassHQ/glass2glass) fixates the graph or fails naming the incompatible link.
- **Window-native pull** — requests snap to a per-node chunk grid on a power-of-two resolution ladder, so results are cacheable and concurrent pulls of one chunk share a single computation.
- **Seam-free kernels** — a transform widens its upstream request by its halo and crops it back, so hillshade at chunk borders equals whole-raster hillshade.
- **Batch = pull** — materializing a pyramid is a driver loop over the chunk grid, the same graph the live server pulls.
- **Live = invalidate + re-pull** — declare a window dirty and the engine drops overlapping cache downstream (halo-spread, projected across CRS changes) and publishes events for re-rendering.

Elements in v1: in-memory GeoTIFF source, windowed COG source (only the tiles a pull touches are fetched, locally or over HTTP range requests, from the file overview nearest the request, single- or multi-band), STAC collection source (items searched lazily per pulled window, paged to the end of the api's results, and combined band by band as a most-recent-first mosaic or a mean, median, min or max temporal composite over the searched interval, streamed the same way, see `examples/stac_tiles.rs` for live Copernicus DEM hillshade), LAS point cloud source with per-level voxel thinning and IDW gridding to raster (nubis), GeoJSON vector source with per-level simplification, per-fragment filter, schema map and boundary clip, and rasterization by constant or property (topoi), windowed tensors (raster bands to CHW f32 channels and back, 3x3 convolution per channel), reproject for rasters and vectors (projicio, auto-plugged wherever a link disagrees only on CRS), hillshade, slope, map algebra and reclassify (terrano), band math over an expression on the input bands (NDVI and friends, parsed at construction), mosaic and two-input algebra over fan-in nodes, an XYZ tile adapter, PNG and GeoTIFF encoders.

## Quick start

```sh
# hillshade XYZ tile server over a synthetic DEM (or GEOPLUMB_DEM=dem.tif)
cargo run --release -p geoplumb --example tile_server

# batch: materialize the tile pyramid from the same graph
cargo run --release -p geoplumb --example pyramid -- 12 tiles
```

## Serving

`geoplumb-server` is the HTTP face of the engine: one engine per layer, layers defined in a TOML file named by `GEOPLUMB_LAYERS` (there is no default, it refuses to start without one). A layer is a source, either a STAC collection or a local COG, plus an ordered pipeline of `hillshade` and `bandmath` ops, and every layer ends reprojected to web mercator and encoded as grayscale PNG. `GET /tiles/{layer}/{z}/{x}/{y}.png` renders a tile, `?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z` renders it at that interval instead of the source's own `datetime`, `GET /layers` lists what is served with each collection's temporal extent, and `GET /health` is the healthcheck. `PORT` defaults to 3000, `GEOPLUMB_CACHE_BYTES` to 256 MiB per layer, and setting `GEOPLUMB_DISK_CACHE` to a directory adds the disk tier (eight times the memory budget per layer). Endpoints are public, matching the rest of the public tile path.

```sh
cp crates/geoplumb-server/examples/layers.toml /tmp/layers.toml   # then edit it
GEOPLUMB_LAYERS=/tmp/layers.toml cargo run --release -p geoplumb-server
curl localhost:3000/layers
```

```toml
# the smallest useful layer file: hillshade over a public collection
[[layer]]
name = "cop-dem-hillshade"
source = { kind = "stac", api = "https://earth-search.aws.element84.com/v1", collection = "cop-dem-glo-30", assets = ["data"], bbox = [7.0, 46.3, 8.0, 46.9] }

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0
```

```rust
use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::{Crs, Engine, Graph, WindowReq};

let mut g = Graph::new();
let src = g.add_source(Box::new(RasterSrc::from_geotiff(&bytes)?));
let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
let engine = Engine::new(g, 256 << 20)?;   // caps solve happens here

// time is None unless the pull asks for an instant
let chunk = engine.pull(rp, WindowReq { bbox, resolution, time }).await?;
```

## Build

```sh
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Pure Rust, no GDAL. Depends on sibling GeoLang crates `terrano-core` (raster kernels) and `projicio-core` (CRS transforms), tracked at master.

See [DESIGN.md](DESIGN.md) for the architecture and [CHANGELOG.md](CHANGELOG.md) for history.

## License

AGPL-3.0-or-later. `src/caps.rs` and `src/solver.rs` are adapted from glass2glass and remain MPL-2.0.
