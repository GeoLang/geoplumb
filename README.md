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

- Sources: `stac` (a STAC collection), `cog` (a local COG file), `gdal` (any raster GDAL opens, in a build with the `gdal` feature), `geojson`, `gpkg` (a GeoPackage feature table), `las`, `copc` (a cloud-optimized point cloud, local or over HTTP), `netcdf` (one variable of a NetCDF classic or 64-bit offset file) and `grib` (one simple-packed GRIB2 field on a regular lat/lon grid).
- Raster ops: `hillshade`, `slope`, `aspect`, `bandmath`, `focal`, `mask`, `reclassify`, `unary` and `convolve`.
- Vector ops: `vec_filter`, `vec_schema` and `vec_clip`, and `rasterize`, which burns a constant or a numeric property per feature into a raster.
- Point op: `idw`, which grids LAS or COPC points into one raster band.
- Fan-in: several `input` entries, each with its own op chain, joined by `mosaic` or a two-input `combine`, with an optional chain after the join.

Source behaviour:

- `stac` searches items per pulled window and follows `next` links to the end of the results. The `composite` key picks how the items combine band by band: `latest` (the default, newest item first), `mean`, `median`, `min`, `max`, `percentile`, `stddev` or `count`. `mean`, `min`, `max`, `stddev` and `count` fold item by item, so peak memory is one batch of parallel item reads however many the search returns. `median` and `percentile` need every value at a pixel at once, so they read the window in horizontal strips under a fixed value budget. See `examples/stac_tiles.rs` for Copernicus DEM hillshade.
- `cog` fetches only the tiles a pull touches, from the file overview nearest the request, single- or multi-band.
- `gdal` reads only the window a pull touches, from the file overview nearest the request, every band as f64. It refuses a rotated or sheared geotransform and a CRS without an EPSG code at startup. GDAL's own `AVERAGE` overviews count NaN as a value unless the file declares NaN as nodata.
- `geojson` reads a feature collection whole at startup, then simplifies it per level and clips it to the pulled window.
- `gpkg` reads a GeoPackage feature table tile by tile through the file's rtree index, decoding only the rows a pull touches, then simplifies and clips them like `geojson`: `source = { kind = "gpkg", path = "parcels.gpkg", table = "parcels", crs = 4326 }`. `table` defaults to the first feature table and `crs` to the table's EPSG srs. A file without the rtree is refused.
- `las` reads a point cloud whole at startup and thins it by voxel decimation per level.
- `copc` fetches only the octree nodes a pull touches, from the root level down to the one its resolution needs, unthinned: `source = { kind = "copc", url = "https://example.com/cloud.copc.laz", crs = 3857 }`, or `path` in place of `url` for a local file. Like `las` it takes the `crs` its coordinates are in and needs an `idw`.
- `netcdf` serves one variable of a NetCDF classic or 64-bit offset file as one f64 band: `source = { kind = "netcdf", path = "sst.nc", variable = "sst", datetime = "2024-01-01T00:00:00Z/2024-01-04T00:00:00Z", select = { zlev = 0 } }`. Its last two dimensions are y and x, georeferenced from their coordinate variables: latitude and longitude are EPSG:4326, projection x and y need a `crs`. Spacing has to be regular and cells square. Rows are flipped north-up and 0 to 360 longitudes rotated to -180 to 180. `scale_factor` and `add_offset` are applied and `_FillValue` and `missing_value` become nodata. A time dimension (units like `hours since 1900-01-01`, standard or proleptic Gregorian calendar) makes the layer answer `t`: a pull takes the latest step inside its interval, `datetime` when it names none, and an interval holding no step serves nodata. `datetime` is required with a time dimension, and every other extra dimension needs an index in `select`. NetCDF-4 (HDF5) files are refused.
- `grib` serves one field of a GRIB2 file: `source = { kind = "grib", path = "gsm.grib2", message = 5 }`, `message` 1-based and counting each field of a multi-field message, the numbering ecCodes uses with multi-field support on. Only the regular lat/lon grid (template 3.0) with square cells and simple packing (template 5.0) are read, and anything else fails at startup naming the template. Bitmap holes are nodata.

`cog`, `gdal`, `stac`, `netcdf` and `grib` take `value_kind = "categorical"` for class-coded rasters such as land cover, `source = { kind = "cog", path = "landcover.tif", value_kind = "categorical" }`. Coarse levels then take the most frequent code of each block, ties going to the smaller code, and reprojection and tile resampling take the nearest pixel, so every tile value is one of the file's codes or nodata. The default, `continuous`, averages blocks and resamples bilinearly. A file's own overviews are read as written, so an overview built by averaging still blends codes. `reclassify` and `rasterize` output is categorical whatever their input, and the terrain ops, `focal`, `bandmath`, `unary`, `convolve` and `idw` output is continuous. A `combine` samples each input by its own kind and its output is continuous, so it can join a categorical input with a continuous one. A `mosaic` needs every input on one kind, and a layer mixing them fails at startup naming `value_kind`. A categorical `stac` source takes the `latest`, `min`, `max` or `count` composite, `count` output being continuous. The layer file refuses `mean`, `median`, `percentile` and `stddev` over categorical values, since they blend codes.

Every layer ends reprojected to web mercator and encoded as PNG. Neither step is named in the file. A layer whose chain ends in features, a `geojson` or `gpkg` source with no `rasterize`, is a vector layer instead: it is served only as Mapbox Vector Tiles and names no `gray`. The PNG stretch range comes from the last op that fixes one, or from the layer's own `gray = { min, max }`. A layer has to name `gray` when it ends in `reclassify`, `rasterize` or `idw`, since class numbers, burned values and interpolated heights carry no range. A fan-in layer with no range-fixing op after the join has to name it too.

Only Rust code against the crate can reach these: the in-memory GeoTIFF source (`RasterSrc::from_geotiff`), COG reads over HTTP range requests, the tensor elements (`ToTensor`, `TensorConv`, `ToRaster`, which `convolve` chains), explicit `Reproject`, which takes raster and vector links alike (also inserted automatically on any link that differs only in CRS), the zonal statistics and time-series drivers, and the XYZ tile adapter in `geoplumb::tile`.

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
| `GET /tiles/{layer}/{z}/{x}/{y}.mvt` | a vector layer's features as a Mapbox Vector Tile, one MVT layer named after the layer |
| `POST /zonal/{layer}` | count, sum, min, max and mean per zone, raster layers only |
| `POST /zonal/{layer}/series` | the same per time step |
| `GET /layers` | each layer's `kind` (`raster` or `vector`), source kind, collection, default datetime and temporal extent |
| `GET /health` | `{"status":"ok"}` |

A raster layer answers `.mvt` with 400, and a vector layer answers PNG and `.tif` with 400. An MVT tile holds the features in the tile plus a 64 unit buffer of its 4096 extent, and a feature the engine's chunk seams cut apart comes back as one MVT feature. The MVT id is the feature's position among a geojson file's features that have a geometry, counting from 0, or its primary key in a gpkg table. A geometry collection goes out as one MVT feature per geometry type, all with the same id. Null properties are left out, and arrays and objects go out as JSON text. Tiles take `?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z` to render at that interval instead of the source's own `datetime`. A zonal body is a GeoJSON FeatureCollection of zones in lon/lat plus `resolution` (web mercator metres) and optional `bbox` and `t`. A series body takes `steps`, a list of intervals, in place of `t`. Four reductions run at once, a fifth is refused with 503, and one that takes over four minutes answers 504.

| Variable | Default | Sets |
|---|---|---|
| `GEOPLUMB_LAYERS` | none, required | layer TOML path |
| `PORT` | `3000` | listen port |
| `GEOPLUMB_CACHE_BYTES` | 256 MiB | memory cache per layer |
| `GEOPLUMB_DISK_CACHE` | unset | directory for a disk cache tier of eight times the memory budget per layer, kept across restarts in a subdirectory per layer graph, so an edited layer starts empty and its old subdirectory stays until you delete it |

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

The default build is pure Rust with no GDAL. The `gdal` feature, on `geoplumb` and on `geoplumb-server`, adds the `gdal` source and links the system libgdal, so `--all-features` needs it installed (`apt-get install libgdal-dev` on Debian and Ubuntu, GDAL 3.4 to 3.12). Build the server with it by `cargo build --release -p geoplumb-server --features gdal`, or the image by `docker build --build-arg GDAL=1 .`, which installs bookworm's `libgdal32` at runtime. The `wgpu` feature, on `geoplumb` only, adds the GPU tensor path (`Graph::with_gpu`, `Upload`, `Download`, `WgpuConvolve`) over wgpu, which needs Vulkan, Metal or DX12 at runtime. Its tests skip without a GPU adapter unless `GEOPLUMB_REQUIRE_GPU=1` is set, and mesa's lavapipe (`apt-get install mesa-vulkan-drivers`) is enough to run them. Depends on the sibling GeoLang crates `terrano-core` (raster kernels), `projicio-core` (CRS transforms), `nubis-core` (point clouds) and `topoi-core` (vector geometry), all as git dependencies on master.

See [DESIGN.md](DESIGN.md) for the architecture and [CHANGELOG.md](CHANGELOG.md) for history.

## License

AGPL-3.0-or-later. `crates/geoplumb/src/caps.rs` and `crates/geoplumb/src/solver.rs` are adapted from glass2glass and remain MPL-2.0.
