# Geoplumb

[![CI](https://github.com/GeoLang/geoplumb/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoLang/geoplumb/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

A pull-based geospatial compute pipeline for the GeoLang GIS stack.

You build a graph of sources and transforms, the engine negotiates caps once, and then you pull windows. A request (bbox plus resolution) travels from the sink to the source, and each element rewrites it on the way, adding a kernel halo or inverse-projecting it. Chunks come back cached and coalesced per node. Only the requested window is computed.

- **Caps negotiation.** Every element declares a constraint over its link caps, per chunk kind: raster, point cloud, vector or tensor. A raster or tensor pattern carries dtype, bands or channels, CRS, resolution range and chunk size. The point cloud and vector patterns carry only CRS, resolution and chunk size. A solver adapted from [glass2glass](https://github.com/Glass2GlassHQ/glass2glass) fixes the graph or fails naming the incompatible link.
- **Window pulls.** Requests snap to a per-node chunk grid on a power-of-two resolution ladder, so results are cacheable and concurrent pulls of one chunk share one computation.
- **Seam-free kernels.** A transform widens its upstream request by its halo and crops the result, so hillshade at chunk borders equals whole-raster hillshade.
- **Batch is a pull.** Materializing a pyramid is a loop over the chunk grid on the same graph the server pulls.
- **Live updates.** `Engine::invalidate` drops the cache that overlaps a dirty window downstream, widened by halos and projected across CRS changes, and sends an `Invalidation` to every `Engine::subscribe` receiver.

## Elements

A served layer file can name these:

- Sources: `stac` (a STAC collection), `cog` (a local COG file), `gdal` (any raster GDAL opens, in a build with the `gdal` feature), `geojson`, `gpkg` (a GeoPackage feature table) and `las`.
- Raster ops: `hillshade`, `slope`, `aspect`, `bandmath`, `focal`, `mask`, `reclassify`, `unary` and `convolve`.
- Vector ops: `vec_filter`, `vec_schema` and `vec_clip`, then `rasterize`, which burns a constant or a numeric property per feature.
- Point op: `idw`, which grids LAS points into one raster band.
- Fan-in: several `input` entries, each with its own op chain, joined by `mosaic` or a two-input `combine`, with an optional chain after the join.

Source behaviour:

- `stac` searches items per pulled window and follows `next` links to the end of the results. The `composite` key picks how the items combine band by band: `latest` (the default, newest item first), `mean`, `median`, `min`, `max`, `percentile`, `stddev` or `count`. `mean`, `min`, `max`, `stddev` and `count` fold item by item, so peak memory is one batch of parallel item reads however many the search returns. `median` and `percentile` need every value at a pixel at once, so they read the window in horizontal strips under a fixed value budget. See `examples/stac_tiles.rs` for Copernicus DEM hillshade.
- `cog` fetches only the tiles a pull touches, from the file overview nearest the request, single- or multi-band.
- `gdal` reads only the window a pull touches, from the file overview nearest the request, every band as f64. It refuses a rotated or sheared geotransform and a CRS without an EPSG code at startup. GDAL's own `AVERAGE` overviews count NaN as a value unless the file declares NaN as nodata.
- `geojson` reads a feature collection whole at startup, then simplifies it per level and clips it to the pulled window.
- `gpkg` reads a GeoPackage feature table tile by tile through the file's rtree index, decoding only the rows a pull touches, then simplifies and clips them like `geojson`: `source = { kind = "gpkg", path = "parcels.gpkg", table = "parcels", crs = 4326 }`. `table` defaults to the first feature table and `crs` to the table's EPSG srs. A file without the rtree is refused.
- `las` reads a point cloud whole at startup and thins it by voxel decimation per level.

Every layer ends reprojected to web mercator and encoded as PNG. Neither step is named in the file. The PNG stretch range comes from the last op that fixes one, or from the layer's own `gray = { min, max }`. A layer has to name `gray` when it ends in `reclassify`, `rasterize` or `idw`, since class numbers, burned values and interpolated heights carry no range. A fan-in layer with no range-fixing op after the join has to name it too.

Only Rust code against the crate can reach these: the in-memory GeoTIFF source (`RasterSrc::from_geotiff`), COG reads over HTTP range requests, the tensor elements (`ToTensor`, `TensorConv`, `ToRaster`, which `convolve` chains), explicit `Reproject` and `VecReproject` (also inserted automatically on any link that differs only in CRS), the zonal statistics and time-series drivers, and the XYZ tile adapter in `geoplumb::tile`.

## Quick start

```sh
# hillshade XYZ tile server over a synthetic DEM (or GEOPLUMB_DEM=dem.tif)
cargo run --release -p geoplumb --example tile_server

# batch: write the tile pyramid to ./tiles up to zoom 12 from the same graph
cargo run --release -p geoplumb --example pyramid -- 12 tiles
```

## Serving

`geoplumb-server` serves one engine per layer. Layers come from the TOML file named by `GEOPLUMB_LAYERS`. There is no default, and the server refuses to start without one.

```sh
cp crates/geoplumb-server/examples/layers.toml /tmp/layers.toml   # then edit it
GEOPLUMB_LAYERS=/tmp/layers.toml cargo run --release -p geoplumb-server
curl localhost:3000/layers
```

| Route | Returns |
|---|---|
| `GET /tiles/{layer}/{z}/{x}/{y}.png` | grayscale PNG over the layer's stretch range |
| `GET /tiles/{layer}/{z}/{x}/{y}.tif` | GeoTIFF with every band as f64 and web mercator georeferencing |
| `POST /zonal/{layer}` | count, sum, min, max and mean per zone |
| `POST /zonal/{layer}/series` | the same per time step |
| `GET /layers` | each layer's source kind, collection, default datetime and temporal extent |
| `GET /health` | `{"status":"ok"}` |

Tiles take `?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z` to render at that interval instead of the source's own `datetime`. A zonal body is a GeoJSON FeatureCollection of zones in lon/lat plus `resolution` (web mercator metres) and optional `bbox` and `t`. A series body takes `steps`, a list of intervals, in place of `t`. Four reductions run at once, a fifth is refused with 503, and one that takes over four minutes answers 504.

| Variable | Default | Sets |
|---|---|---|
| `GEOPLUMB_LAYERS` | none, required | layer TOML path |
| `PORT` | `3000` | listen port |
| `GEOPLUMB_CACHE_BYTES` | 256 MiB | memory cache per layer |
| `GEOPLUMB_DISK_CACHE` | unset | directory for a disk cache tier of eight times the memory budget per layer |

Endpoints are public, the same as the rest of the public tile path. Tagged releases publish `ghcr.io/geolang/geoplumb`.

The `ndvi` layer in the example file is a Sentinel-2 median composite with no cloud masking, which is not what you would deploy. Add a `mask` op over the SCL band ahead of the `bandmath` and list `scl` among the search assets, as the `s2-red-clear` layer in the same file does.

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

A server built with the `gdal` feature also takes a local raster in any format GDAL reads. `band` is 1-based and optional, and every band is served when it is absent. Startup prints the GDAL driver each `gdal` layer opened with. A build without the feature refuses a layer file naming `gdal` and says so.

```toml
[[layer]]
name = "dem"
source = { kind = "gdal", path = "/data/dem.tif", band = 1 }
```

## Library

```rust
use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::{Crs, Engine, Graph, WindowReq};

let mut g = Graph::new();
let src = g.add_source(Box::new(RasterSrc::from_geotiff(&bytes)?));
let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
let engine = Engine::new(g, 256 << 20)?;   // caps solve happens here

// time is None unless the pull asks for an interval
let chunk = engine.pull(rp, WindowReq { bbox, resolution, time }).await?;
```

## Build

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all --all-features
```

The default build is pure Rust with no GDAL. The `gdal` feature, on `geoplumb` and on `geoplumb-server`, adds the `gdal` source and links the system libgdal, so `--all-features` needs it installed (`apt-get install libgdal-dev` on Debian and Ubuntu, GDAL 3.4 to 3.12). Build the server with it by `cargo build --release -p geoplumb-server --features gdal`, or the image by `docker build --build-arg GDAL=1 .`, which installs bookworm's `libgdal32` at runtime. Depends on the sibling GeoLang crates `terrano-core` (raster kernels), `projicio-core` (CRS transforms), `nubis-core` (point clouds) and `topoi-core` (vector geometry), all as git dependencies on master.

See [DESIGN.md](DESIGN.md) for the architecture and [CHANGELOG.md](CHANGELOG.md) for history.

## License

AGPL-3.0-or-later. `crates/geoplumb/src/caps.rs` and `crates/geoplumb/src/solver.rs` are adapted from glass2glass and remain MPL-2.0.
