# Geoplumb

[![CI](https://github.com/GeoLang/geoplumb/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/geoplumb/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

A pull-based geospatial compute pipeline for the GeoLang GIS stack.

Build a dag of sources and transforms, negotiate caps once, then pull windows: a demand (bbox plus resolution) flows sink to source, each element rewriting it on the way up (kernel halo, inverse projection), and chunks flow back, cached and coalesced per node. Nothing is computed until something asks, and only the asked-for window is computed.

- **Caps negotiation** — every element declares a constraint over its link caps, per chunk kind (raster, point cloud, vector or tensor). A raster or tensor pattern carries dtype, bands or channels, CRS, resolution range and chunk size; the point cloud and vector patterns carry only CRS, resolution and chunk size. A solver adapted from [glass2glass](https://github.com/Glass2GlassHQ/glass2glass) fixates the graph or fails naming the incompatible link.
- **Window-native pull** — requests snap to a per-node chunk grid on a power-of-two resolution ladder, so results are cacheable and concurrent pulls of one chunk share a single computation.
- **Seam-free kernels** — a transform widens its upstream request by its halo and crops it back, so hillshade at chunk borders equals whole-raster hillshade.
- **Batch = pull** — materializing a pyramid is a driver loop over the chunk grid, the same graph the live server pulls.
- **Live = invalidate + re-pull** — declare a window dirty and the engine drops overlapping cache downstream (halo-spread, projected across CRS changes) and publishes events for re-rendering.

Elements in v1 split into what a served layer file can name and what only Rust code against the crate can reach.

Reachable from a layer file: the windowed COG source (only the tiles a pull touches are fetched, locally or over HTTP range requests, from the file overview nearest the request, single- or multi-band), the STAC collection source (items searched lazily per pulled window, paged to the end of the api's results, and combined band by band as a most-recent-first mosaic or a mean, median, min, max, percentile, standard deviation or count temporal composite over the searched interval, see `examples/stac_tiles.rs` for live Copernicus DEM hillshade), the GeoJSON vector source (a feature collection read whole at startup, simplified per level and clipped to the pulled window), the LAS point cloud source (a cloud read whole at startup and thinned by voxel decimation per level, paired with the `idw` op that grids its points into a raster band), and the ten ops `hillshade`, `slope`, `aspect`, `bandmath`, `focal`, `mask`, `reclassify`, `unary` and `convolve`, which is every single-input raster transform the crate ships, plus `rasterize`, which burns a constant or a numeric property per feature, and the three vector ops `vec_filter`, `vec_schema` and `vec_clip`, which run over features before a `rasterize` turns them into pixels. A layer may name several inputs instead of one source, each with its own chain, joined by a `mosaic` or a two-input `combine` and followed by a chain over the join. Reproject to web mercator and PNG encoding are appended to every layer, not named. The PNG stretch range comes from the last op that fixes one, or from the layer's own `gray = { min, max }`, which a layer ending in `reclassify` or `rasterize` has to name because class numbers and burned values carry no range of their own, as does a fan-in layer with nothing after the join to fix one. Mean, min, max, standard deviation and count fold item by item, so a pull's peak memory is one wave however deep the stack; median and percentile instead need every value at a pixel at once and hold a horizontal strip's whole stack under a fixed value budget.

Library-only, meaning no layer file can name them: the in-memory GeoTIFF source, windowed tensors past the raster to tensor, 3x3 convolution, tensor to raster chain `convolve` builds out of them, explicit reproject for rasters and vectors (projicio, auto-plugged wherever a link disagrees only on CRS), zonal statistics and time-series pull drivers, and the XYZ tile adapter.

## Quick start

```sh
# hillshade XYZ tile server over a synthetic DEM (or GEOPLUMB_DEM=dem.tif)
cargo run --release -p geoplumb --example tile_server

# batch: materialize the tile pyramid from the same graph
cargo run --release -p geoplumb --example pyramid -- 12 tiles
```

## Serving

`geoplumb-server` is the HTTP face of the engine: one engine per layer, layers defined in a TOML file named by `GEOPLUMB_LAYERS` (there is no default, it refuses to start without one). A layer is a source, either a STAC collection, a local COG, a GeoJSON file or a LAS point cloud, plus an ordered pipeline of `hillshade`, `bandmath`, `slope`, `aspect`, `focal`, `mask`, `reclassify`, `unary`, `convolve`, `rasterize`, `vec_filter`, `vec_schema`, `vec_clip` and `idw` ops, and every layer ends reprojected to web mercator. `GET /tiles/{layer}/{z}/{x}/{y}.png` renders a tile as grayscale PNG over the range the ops imply or the layer's own `gray = { min, max }`, and `GET /tiles/{layer}/{z}/{x}/{y}.tif` renders the same tile as a GeoTIFF carrying every band at full f64 precision with its web mercator georeferencing. `?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z` renders it at that interval instead of the source's own `datetime`, `POST /zonal/{layer}` returns zonal statistics and `POST /zonal/{layer}/series` returns a time series, four of them running at once and any more refused 503, each with four minutes to finish before it answers 504. `GET /layers` lists what is served with each collection's temporal extent, and `GET /health` is the healthcheck. `PORT` defaults to 3000, `GEOPLUMB_CACHE_BYTES` to 256 MiB per layer, and setting `GEOPLUMB_DISK_CACHE` to a directory adds the disk tier (eight times the memory budget per layer). Endpoints are public, matching the rest of the public tile path.

```sh
cp crates/geoplumb-server/examples/layers.toml /tmp/layers.toml   # then edit it
GEOPLUMB_LAYERS=/tmp/layers.toml cargo run --release -p geoplumb-server
curl localhost:3000/layers
```

The `ndvi` layer in that example file is a Sentinel-2 median composite with no
cloud masking, which is not what you would deploy. Masking would need
`QualityMask` over the SCL band, and `OpConfig` has no variant for it, so a
layer file cannot ask for it. Build the graph in Rust when you need masked
optical imagery.

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

Pure Rust, no GDAL. Depends on sibling GeoLang crates `terrano-core` (raster kernels), `projicio-core` (CRS transforms), `nubis-core` (point clouds) and `topoi-core` (vector geometry), all tracked at master.

See [DESIGN.md](DESIGN.md) for the architecture and [CHANGELOG.md](CHANGELOG.md) for history.

## License

AGPL-3.0-or-later. `src/caps.rs` and `src/solver.rs` are adapted from glass2glass and remain MPL-2.0.
