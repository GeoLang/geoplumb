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

- Sources: `stac` (a STAC collection), `cog` (a local COG file), `gdal` (any raster GDAL opens, in a build with the `gdal` feature), `geojson`, `gpkg` (a GeoPackage feature table), `shapefile`, `csv` (points in two coordinate columns), `las`, `copc` (a cloud-optimized point cloud, local or over HTTP), `netcdf` (one variable of a NetCDF classic or 64-bit offset file), `zarr` (one array of a Zarr v2 or v3 store, local or over HTTP) and `grib` (one simple-packed GRIB2 field on a regular lat/lon grid).
- Raster ops: `hillshade`, `slope`, `aspect`, `bandmath`, `focal`, `mask`, `reclassify`, `unary`, `convolve`, `viewshed`, `cast` and `color_relief`. `viewshed` takes `lon`, `lat`, `height` and `radius` and marks each cell an observer at `lon`, `lat` with an eye `height` above the ground can see 1 and each hidden one 0, with nodata past `radius`, which is in the source CRS units. `cast` takes `dtype`, one of `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32` and `f64`, and for an integer `dtype` an optional `nodata`. `color_relief` colours a one-band raster as `gdaldem color-relief -alpha` does, see below.
- Raster to vector ops: `contours`, lines every `interval` elevation units counted from `base` (0 when absent), one feature per level with an `elevation` property, and `polygonize`, one feature per class code of a categorical raster with a `value` property. A layer ending in either is a vector layer, and a `rasterize` after them burns the features back into a raster.
- Vector ops: `vec_filter`, `vec_schema` and `vec_clip`, and `rasterize`, which burns a constant or a numeric property per feature into a raster.
- Raster clip: `clip_raster`, which turns every pixel whose centre lies outside the polygons of another layer of the file into nodata, as `gdalwarp -cutline` does without `-crop_to_cutline`. See below.
- Point op: `idw`, which grids LAS or COPC points into one raster band.
- Hydrology ops: `flow_direction`, `flow_accumulation`, `watershed` and `stream_order`, each over a one-band DEM. See below.
- Fan-in: several `input` entries, each with its own op chain, joined by `mosaic` or a two-input `combine`, with an optional chain after the join.

Source behaviour:

- `stac` searches items per pulled window and follows `next` links to the end of the results. The `composite` key picks how the items combine band by band: `latest` (the default, newest item first), `mean`, `median`, `min`, `max`, `percentile`, `stddev` or `count`. `mean`, `min`, `max`, `stddev` and `count` fold item by item, so peak memory is one batch of parallel item reads however many the search returns. `median` and `percentile` need every value at a pixel at once, so they read the window in horizontal strips under a fixed value budget. The `assets` may differ in pixel size, like Sentinel-2's 10 m `red` and 20 m `scl`. The layer's grid is then the finest asset's, and each item's coarser assets are resampled onto it before the composite by each asset's value kind. `item_mask = { asset = "scl", valid_values = [4.0, 5.0, 6.0] }` masks each item before the composite. The named asset is read with each item on the layer's grid, nearest, and where its code is nodata or not in `valid_values` that item has no value in any band. Every composite skips those values, `latest` taking the newest item with a value, and a pixel masked in every item is nodata. The mask asset is not a band of the layer, so `median` and the other blending composites stay allowed over the `assets`. A `mask` op, by contrast, masks the composite. See `examples/stac_tiles.rs` for Copernicus DEM hillshade.
- `cog` fetches only the tiles a pull touches, from the file overview nearest the request, single- or multi-band, in the file's sample format.
- `gdal` reads only the window a pull touches, from the file overview nearest the request, each band in its own sample format. A band GDAL stores as signed byte, 64-bit integer or complex is read as f64. It refuses a rotated or sheared geotransform and a CRS without an EPSG code at startup. GDAL's own `AVERAGE` overviews count NaN as a value unless the file declares NaN as nodata.
- `geojson` reads a feature collection whole at startup, then simplifies it per level and clips it to the pulled window.
- `gpkg` reads a GeoPackage feature table tile by tile through the file's rtree index, decoding only the rows a pull touches, then simplifies and clips them like `geojson`: `source = { kind = "gpkg", path = "parcels.gpkg", table = "parcels", crs = 4326 }`. `table` defaults to the first feature table and `crs` to the table's EPSG srs. A file without the rtree is refused.
- `shapefile` reads a shapefile whole at startup and serves it like `geojson`: `source = { kind = "shapefile", path = "parcels.shp" }`. The `.shx` and `.dbf` beside the `.shp` are required, with uppercase extensions when the `.shp` has one. The crs is the EPSG code the `.prj` names, and a missing `.prj` or one with no EPSG authority, as ESRI writes them, needs `crs = 32633` in the source. Multi-part polygons become multipolygons, each hole going to the outer ring before it. Attributes keep their dBase type: text, logical, date as `YYYY-MM-DD`, and a numeric column reads as integers when every value in it is whole, otherwise as floats. Text is decoded the way GDAL picks the encoding: the first line of a `.cpg` beside the `.dbf`, else the language driver id in the `.dbf` header (87, GDAL's default, is ISO-8859-1), else UTF-8. An encoding geoplumb has no decoder for, such as the DOS code page CP437, fails startup naming it, and text that is not UTF-8 in a file declaring nothing fails too. A multipatch record fails startup.
- `csv` reads a CSV of points whole at startup and serves it like `geojson`: `source = { kind = "csv", path = "sites.csv", x = "lon", y = "lat", crs = 4326 }`. `x` and `y` name the coordinate columns, matched exactly, and `crs` is required since CSV carries none. A coordinate that is not a finite number fails startup naming its line. Every other column is a property: a cell that parses as an integer is an integer, then a finite float, otherwise text, and an empty cell is empty text.
- `las` reads a point cloud whole at startup and thins it by voxel decimation per level.
- `copc` fetches only the octree nodes a pull touches, from the root level down to the one its resolution needs, unthinned: `source = { kind = "copc", url = "https://example.com/cloud.copc.laz", crs = 3857 }`, or `path` in place of `url` for a local file. Like `las` it takes the `crs` its coordinates are in and needs an `idw`.
- `netcdf` serves one variable of a NetCDF classic or 64-bit offset file as one f64 band: `source = { kind = "netcdf", path = "sst.nc", variable = "sst", datetime = "2024-01-01T00:00:00Z/2024-01-04T00:00:00Z", select = { zlev = 0 } }`. Its last two dimensions are y and x, georeferenced from their coordinate variables: latitude and longitude are EPSG:4326, projection x and y need a `crs`. Spacing has to be regular and cells square. Rows are flipped north-up and 0 to 360 longitudes rotated to -180 to 180. `scale_factor` and `add_offset` are applied and `_FillValue` and `missing_value` become nodata. A time dimension (units like `hours since 1900-01-01`, standard or proleptic Gregorian calendar) makes the layer answer `t`: a pull takes the latest step inside its interval, `datetime` when it names none, and an interval holding no step serves nodata. `datetime` is required with a time dimension, and every other extra dimension needs an index in `select`. NetCDF-4 (HDF5) files are refused.
- `zarr` serves one array of a Zarr store as one band: `source = { kind = "zarr", path = "era5.zarr", array = "t2m", datetime = "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z", select = { level = 3 } }`, or `url = "https://host/era5.zarr"` in place of `path`. Zarr v3 arrays and v2 arrays that convert to v3 by a metadata change are read, holding 8 to 64 bit integers, float32 or float64. A pull reads only the chunks its window touches. The last two dimensions are y and x. Their coordinate arrays sit beside the array in its group and are named by `x_coordinate` and `y_coordinate`, else by the array's `dimension_names` (v3) or `_ARRAY_DIMENSIONS` attribute (v2). Coordinates must be regularly spaced with square cells. The crs is the layer's `crs`, else EPSG:4326 when the coordinates are latitude and longitude, else the layer fails at startup. Time and `select` work as for `netcdf`. `_FillValue` and `missing_value` mark nodata, a float `_FillValue` also as the base64 text xarray writes for v3, and so does a v2 array's `fill_value` but not a v3 one, which is xarray's rule. `scale_factor` and `add_offset` are applied and make the band f64. Without them the band keeps the array's sample format, except int64 and uint64, which are f64. A pull over many base cells reads all of them, since Zarr has no overviews here, so coarse zooms over a large array are slow. Chunks may be raw or compressed with zstd, gzip or zlib, with crc32c checksums, transposed or sharded. blosc is not read and fails at startup naming the codec. Over HTTP a chunk is one GET, a missing chunk (404) reads as fill, and a sharded array reads its index and inner chunks by range request.
- `grib` serves one field of a GRIB2 file: `source = { kind = "grib", path = "gsm.grib2", message = 5 }`, `message` 1-based and counting each field of a multi-field message, the numbering ecCodes uses with multi-field support on. Only the regular lat/lon grid (template 3.0) with square cells and simple packing (template 5.0) are read, and anything else fails at startup naming the template. Bitmap holes are nodata.

`cog`, `gdal`, `stac`, `netcdf`, `zarr` and `grib` take `value_kind = "categorical"` for class-coded rasters such as land cover, `source = { kind = "cog", path = "landcover.tif", value_kind = "categorical" }`. Coarse levels then take the most frequent code of each block, ties going to the smaller code, and reprojection and tile resampling take the nearest pixel, so every tile value is one of the file's codes or nodata. The default, `continuous`, averages blocks and resamples bilinearly. `value_kind = "peak"` is for measurements whose largest values lie on lines one cell wide, such as contributing area along a channel. Coarse levels then take the largest value of each block, and reprojection and tile resampling take the largest pixel under each output pixel, or the nearest pixel when sampling up, so the line keeps its value at every zoom. A file's own overviews are read as written, so an overview built by averaging still blends codes. `reclassify`, `rasterize`, `viewshed`, `flow_direction`, `watershed` and `stream_order` output is categorical whatever their input, `polygonize` takes only categorical input and `contours`, `viewshed` and the hydrology ops only continuous, and the terrain ops, `focal`, `bandmath`, `unary`, `convolve`, `idw` and `color_relief` output is continuous, except that `unary` `add`, `sqrt`, `log` and `multiply` by a positive factor keep a peak input peak. `flow_accumulation` output is peak. A `stac` source also takes a kind per asset, `value_kind = { scl = "categorical" }`, for a search that stacks measurements with a class-coded band such as Sentinel-2's SCL. Each band is then resampled by its own kind, and an asset left out is continuous. A `combine` samples each input by its own kinds and its output is continuous, so it can join a categorical input with a continuous one. A `mosaic` needs every input on the same kind band for band, and a layer mixing them fails at startup naming `value_kinds`. A `stac` source with a categorical asset takes the `latest`, `min`, `max` or `count` composite, `count` output being continuous. The layer file refuses `mean`, `median`, `percentile` and `stddev` over categorical values, since they blend codes. An `item_mask` asset is always read as categorical and does not count here, since it is not a band of the layer.

Each band keeps the sample format its source stores. A `cog` band has the file's format and nodata. Integer cells past the image edge or in a file tile with no bytes hold the nodata value, or 0 when the file declares none, as GDAL fills them. An integer file with a transparency mask and no nodata is refused at startup unless the source names one, `source = { kind = "cog", path = "landcover.tif", nodata = 0 }`. Cells equal to that value and masked cells are then nodata. A `stac` band has its asset's format under `latest`, `min` and `max` and is f64 under the other composites. A `stac` integer asset with no nodata is refused at startup, since cells outside an item would read as 0 and hide older items. `viewshed` outputs u8 with nodata 255, and `color_relief` four u8 bands with no nodata. `reclassify` outputs the smallest integer type that holds every class value with its top value left free as nodata, or f64 when a class value is fractional. `convolve` back to raster is f32, and the other computing ops, `netcdf` and `grib` are f64. A `zarr` band has the array's format with its single `_FillValue` as the integer nodata, and is f64 when packed or 64-bit integer. A `mosaic` whose inputs differ widens the narrower input where no value changes and otherwise fails at startup. A `cast` op converts every band to its `dtype`: a value rounds half away from zero, saturates at the type's range, and moves one step off `nodata` when it lands on it, down or up at the type's minimum, as GDAL's warp does. A `cast` to an integer type without `nodata` is refused when the band has nodata cells.

Every layer ends reprojected to web mercator and encoded as PNG. Neither step is named in the file. A layer whose chain ends in features, a `geojson`, `gpkg`, `shapefile` or `csv` source with no `rasterize` or a chain ending in `contours` or `polygonize`, is a vector layer instead: it is served only as Mapbox Vector Tiles and names no `gray`. The PNG stretch range comes from the last op that fixes one, or from the layer's own `gray = { min, max }`. A layer has to name `gray` when it ends in `reclassify`, `rasterize`, `idw` or a hydrology op, since class numbers, burned values, interpolated heights, direction codes, areas, basin labels and stream orders carry no range. `viewshed` stretches over 0 to 1. A fan-in layer with no range-fixing op after the join has to name it too.

A layer can colour its PNG from a gdaldem colour file instead. `ramp = "relief.txt"` renders the layer's one band through the file's stops in place of the gray stretch, leaving `.tif` tiles, exports and zonal statistics on the values. A `color_relief` op turns the band into four u8 bands, red, green, blue and alpha, which the PNG serves as RGBA and `.tif` tiles and exports write as bands. It has to be the layer's last op, and such a layer names neither `gray` nor `ramp`.

```toml
[[layer.op]]
kind = "color_relief"
color_file = "relief.txt"
selection = "nearest"           # or "exact", "interpolate" when absent
range = { min = 0.0, max = 4800.0 }
```

The colour file is the one `gdaldem color-relief` reads: one stop per line, a value then red, green and blue from 0 to 255 and an optional alpha, or a colour name and an optional alpha, split by spaces, tabs, commas or colons. Lines starting with `#` or `/` are comments. The colour names are gdaldem's sixteen, `white` to `indigo`. A value between two stops interpolates linearly, and values past the first or last stop take its colour. `selection = "exact"` colours only values equal to a stop and leaves the rest transparent, and `"nearest"` takes the closer stop. `nv` gives the colour of nodata cells, which are transparent without it, where gdaldem colours them by their value. `50%` stops sit at that fraction of `range`, which `color_relief` has to name when the file has one, as gdaldem would read the raster's statistics. A `ramp` takes no percent stops and always interpolates. The parser lives in jung-core (`ColorFile`, `ValueRamp`) and is stricter than gdaldem's: a colour channel must be a whole number from 0 to 255, a value must be a number, a line with a value and no colour is an error, and a GMT `.cpt` palette is refused. Equal stops are not moved apart, so a value exactly at a repeated stop takes the first one's colour. With `color_relief`, the warp to web mercator blends channels bilinearly, so pixels beside transparent ones fade toward transparent black, which `ramp` avoids by colouring after the warp.

The hydrology ops take `crs = "EPSG:<code>"`, a projected CRS the DEM is reprojected into, and `region = [min_x, min_y, max_x, max_y]` in that CRS. A CRS in longitude and latitude is refused at startup. The region is computed whole, once per zoom level, and every tile is cut out of that result, with nodata outside the region. The region border is the outlet, so flow from outside it is not counted and a basin crossing it is cut there. Depressions and flats are filled before flow is routed, so flow crosses them. `flow_direction` gives ESRI D8 codes, 1 east, 2 south-east, 4 south and so on to 128 north-east, and 0 where no neighbour is lower. `flow_accumulation` gives the contributing area in CRS units squared, the cell included. `watershed` gives one label per outlet. `stream_order` takes `threshold`, an area in CRS units squared, and gives the Strahler order of the cells whose contributing area is at least that, with nodata elsewhere. Ops after a hydrology op run in its CRS. A tile or export at a zoom level where the region is over the engine's pixel cap answers 400 naming the finest resolution that fits. The cap grows with `GEOPLUMB_CACHE_BYTES`.

```toml
[[layer.op]]
kind = "stream_order"
crs = "EPSG:32632"
region = [381990.0, 5145210.0, 419010.0, 5183190.0]
threshold = 1000350.0
```

`clip_raster` takes `cutline`, the name of a layer in the same file whose chain ends in polygons, such as a `geojson` or `gpkg` layer or one ending in `polygonize`. The output keeps the raster's grid, bands, sample format and value kind. A pixel stays when its centre lies inside a polygon of the cutline layer and outside its holes, the rule `rasterize` burns by, and every other pixel of every band becomes nodata. The cutline layer is reprojected into the raster's CRS, and each tile reads only the features that reach it. At zoom levels coarser than the cutline's vertex spacing its polygons are simplified first, as they are for `rasterize`. Its features have to be polygons or multipolygons, and it cannot have a `clip_raster` of its own. Nodata is the band's own, NaN in a float band and the sentinel in an integer band that has one. An integer band without a sentinel needs `nodata`, which becomes its sentinel, and a kept pixel equal to it moves one step off it, as in `cast`. `nodata` on a band that already marks absent cells fails at startup. A clipped layer carries no colour table into exports.

```toml
[[layer.op]]
kind = "clip_raster"
cutline = "parcels"
nodata = 0
```

Only Rust code against the crate can reach these: the in-memory GeoTIFF source (`RasterSrc::from_geotiff`), COG reads over HTTP range requests, the tensor elements (`ToTensor`, `TensorConv`, `ToRaster`, which `convolve` chains), explicit `Reproject`, which takes raster and vector links alike (also inserted automatically on any link that differs only in CRS), the zonal statistics and time-series drivers, and the XYZ tile adapter in `geoplumb::tile`.

## Quick start

```sh
# hillshade XYZ tile server over a synthetic DEM (or GEOPLUMB_DEM=dem.tif)
cargo run --release -p geoplumb --example tile_server

# batch: write the tile pyramid to ./tiles up to zoom 12 from the same graph
cargo run --release -p geoplumb --example pyramid -- 12 tiles

# 100 m contours over a 900 px square of a DEM COG from pixel 2520, 1080, as GeoJSON
cargo run --release -p geoplumb --example raster_vector_geojson -- dem.tif 2520 1080 900 900 contours.geojson contours 100

# one polygon set per class over the same kind of window of a land cover COG
cargo run --release -p geoplumb --example raster_vector_geojson -- landcover.tif 18000 18000 1024 1024 classes.geojson polygonize
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
| `GET /tiles/{layer}/{z}/{x}/{y}.png` | grayscale PNG over the layer's stretch range, or RGBA through its `ramp` or `color_relief` colours |
| `GET /tiles/{layer}/{z}/{x}/{y}.tif` | one-tile GeoTIFF with every band and web mercator georeferencing, in the layer's sample format |
| `GET /tiles/{layer}/{z}/{x}/{y}.mvt` | a vector layer's features as a Mapbox Vector Tile, one MVT layer named after the layer |
| `POST /zonal/{layer}` | count, sum, min, max and mean per zone, raster layers only |
| `POST /zonal/{layer}/series` | the same per time step |
| `GET /export/{layer}?bbox=&crs=&resolution=` | a cloud optimized GeoTIFF of the layer over that window, in the layer's sample format |
| `GET /layers` | each layer's `kind` (`raster` or `vector`), source kind, collection, default datetime and temporal extent |
| `GET /health` | `{"status":"ok"}` |

A raster layer answers `.mvt` with 400, and a vector layer answers PNG and `.tif` with 400. An MVT tile holds the features in the tile plus a 64 unit buffer of its 4096 extent, and a feature the engine's chunk seams cut apart comes back as one MVT feature. The MVT id is the feature's position among the features that have a geometry in a geojson file, a shapefile or a csv, counting from 0, or its primary key in a gpkg table. A geometry collection goes out as one MVT feature per geometry type, all with the same id. Null properties are left out, and arrays and objects go out as JSON text. Tiles take `?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z` to render at that interval instead of the source's own `datetime`. A zonal body is a GeoJSON FeatureCollection of zones in lon/lat plus `resolution` (web mercator metres) and optional `bbox` and `t`. A series body takes `steps`, a list of intervals, in place of `t`. Four reductions run at once, a fifth is refused with 503, and one that takes over four minutes answers 504.

An export takes `bbox` and `resolution` in the units of `crs`, like gdalwarp's `-te` and `-tr`, and `crs` as `EPSG:<code>`, the authority in any case, with no default. The grid starts at the bbox's west and north edges, and its pixel counts are the bbox's width and height over `resolution`, rounded and at least one, as gdalwarp counts them. Overviews are pulled at their own resolution, not averaged, so a categorical layer keeps its codes at every level. An integer layer writes its nodata as GDAL_NODATA, and a float layer marks nodata with NaN. A layer of bands of different formats is written as f64. A one-band layer whose samples come unchanged from a `cog` with a colour table carries that table into the export and the `.tif` tile, and a layer with a computing op does not. `t` works as on tiles. The file is written under `GEOPLUMB_EXPORT_DIR` before it is streamed, up to about 2.9 GB at the cap, and without that variable an export answers 500. Two exports run at once, a third is refused with 503, one over four minutes answers 504 and stops after the tile it is pulling, as does one whose client disconnects, and one past 16384 x 16384 values (counted over whole 512 px tiles and every band) is refused with 400.

```sh
curl -o dem.tif 'localhost:3000/export/dem?bbox=380000,5170000,400000,5190000&crs=EPSG:32632&resolution=20'
```

| Variable | Default | Sets |
|---|---|---|
| `GEOPLUMB_LAYERS` | none, required | layer TOML path |
| `PORT` | `3000` | listen port |
| `GEOPLUMB_CACHE_BYTES` | 256 MiB | memory cache per layer |
| `GEOPLUMB_DISK_CACHE` | unset | directory for a disk cache tier of eight times the memory budget per layer, kept across restarts in a subdirectory per layer graph, so an edited layer starts empty and its old subdirectory stays until you delete it |
| `GEOPLUMB_EXPORT_DIR` | unset, required for `/export` | directory export files are written in before they are streamed, unnamed and gone when the response ends. Put it on disk, not tmpfs |

Endpoints are public, the same as the rest of the public tile path. Tagged releases publish `ghcr.io/geolang/geoplumb`.

The `ndvi` layer in the example file is a cloud-free median. Its search assets are `red` and `nir`, its composite is `median`, and its `item_mask` keeps SCL classes 4, 5 and 6, so each item counts only where its own SCL shows vegetation, bare ground or water. A pixel under cloud, cloud shadow or snow in one item takes the median of the items clear there, and a pixel no item sees clear is transparent. SCL is not a band of the layer, so the `bandmath` reads red as `b0` and nir as `b1`.

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

## Command line

`geoplumb` writes a layer from the same TOML file to disk instead of serving it. It is the second binary of the `geoplumb-server` crate.

```sh
cargo install --path crates/geoplumb-server --bin geoplumb
geoplumb run <layers.toml> <layer> --bbox minx,miny,maxx,maxy --crs EPSG:<code> --resolution <r> [--t <start>/<end>] -o <out>
geoplumb pyramid <layers.toml> <layer> --bbox west,south,east,north [--min-zoom <z>] --max-zoom <z> [--t <start>/<end>] -o <dir>
geoplumb info <file.tif>
```

`run` builds the one layer it names. `--bbox`, `--crs`, `--resolution` and `--t` mean what they mean on `GET /export/{layer}`, and the grid is sized the same way. A raster layer writes a COG to a `.tif` or `.tiff` path, in the layout the export route writes. A vector layer writes GeoJSON to a `.geojson` path or a GeoPackage to a `.gpkg` path: the features in the bbox, clipped to it and merged back where chunk seams cut them. GeoJSON coordinates are lon/lat, so a `.geojson` output takes `--crs EPSG:4326` only. A `.gpkg` path takes any `--crs` with well-known text in crs-definitions and gets one feature table named after the layer, with a `fid` key, a `geom` column declared as the one geometry type every feature shares or `GEOMETRY`, a column per property typed `INTEGER`, `REAL`, `BOOLEAN` or `TEXT` from the values it takes, the extent in `gpkg_contents`, and the rtree index and triggers GDAL writes for GeoPackage 1.4. A property named `fid` or `geom` in any case is refused. Any other extension is an error. The file is written under a temporary name in the output's directory and renamed when it is complete, so a failed run leaves no file. There is no time limit and no pixel cap. The COG is a classic TIFF, so a run whose compressed tiles pass 4 GiB fails after its last tile.

`pyramid` writes every xyz tile covering a lon/lat bbox from `--min-zoom`, 0 by default, to `--max-zoom`, as `<dir>/z/x/y.png` for a raster layer and `<dir>/z/x/y.mvt` for a vector layer. Each file holds the bytes `GET /tiles/{layer}/{z}/{x}/{y}` serves for that tile, so any static file server serves the directory as the layer.

`info` prints one `key: value` line each for size, band count, sample format, bits, CRS, geotransform (GDAL order), compression, predictor and nodata, then the overview sizes as a list. It reads what terrano's `CogReader` reads: classic or BigTIFF, tiled or striped, uncompressed, LZW, deflate, ZSTD, JPEG or WebP, pixel- or band-interleaved.

```
size: 600 x 400
bands: 1
sample format: u16
bits: 16
crs: EPSG:32632
geotransform: 380000, 20, 0, 5190000, 0, -20
compression: deflate
predictor: none
nodata: 0
overviews:
- 300 x 200
- 150 x 100
```

Routine GDAL jobs, each with the `geoplumb` command that does the same over this layer file:

```toml
[[layer]]
name = "dem"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer]]
name = "contours"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "contours"
interval = 10.0

[[layer]]
name = "s2-red-median"
source = { kind = "stac", api = "https://earth-search.aws.element84.com/v1", collection = "sentinel-2-l2a", assets = ["red"], bbox = [7.0, 46.3, 8.0, 46.9], composite = "median" }
```

```sh
# gdalwarp -t_srs EPSG:32632 -te 380000 5170000 400000 5190000 -tr 20 20 -r bilinear -of COG dem.tif dem-utm.tif
geoplumb run layers.toml dem --bbox 380000,5170000,400000,5190000 --crs EPSG:32632 --resolution 20 -o dem-utm.tif

# gdal_contour -i 10 -a elevation dem.tif contours.geojson, over a lon/lat window
geoplumb run layers.toml contours --bbox 7.2,46.5,7.6,46.8 --crs EPSG:4326 --resolution 0.0003 -o contours.geojson

# the same contours in UTM as a GeoPackage, gdal_contour -f GPKG on the warped dem
geoplumb run layers.toml contours --bbox 380000,5170000,400000,5190000 --crs EPSG:32632 --resolution 20 -o contours.gpkg

# a summer median of Sentinel-2 red over the same UTM window, pulled from the STAC API window by window
geoplumb run layers.toml s2-red-median --bbox 380000,5170000,400000,5190000 --crs EPSG:32632 --resolution 10 --t 2025-06-01T00:00:00Z/2025-09-01T00:00:00Z -o red-median.tif

# gdal2tiles.py -z 8-12 over a lon/lat window, png tiles stretched over the layer's gray range
geoplumb pyramid layers.toml dem --bbox 7.2,46.5,7.6,46.8 --min-zoom 8 --max-zoom 12 -o tiles

# gdalinfo dem-utm.tif
geoplumb info dem-utm.tif
```

The Docker image holds `geoplumb-server` only.

## Compared with GDAL

[docs/gdal-parity.md](docs/gdal-parity.md) lists each operator beside the GDAL command it was checked against, the public input, the measure and the result. The scripts in [proofs/](proofs/README.md) rerun every comparison against your build, and `proofs/bench.sh` times hillshade, warp, contours and COG export against GDAL.

A cloud-free Sentinel-2 median written as a COG is one command over [examples/sentinel2-median.toml](examples/sentinel2-median.toml). GDAL 3.12 has no STAC source that keeps items apart, so the same output takes a STAC search of its own, a `gdalwarp` and a `gdal raster calc` per item and band, and a `gdal raster mosaic --pixel-function median` per band, see `proofs/sentinel2-median.sh`.

```sh
geoplumb run examples/sentinel2-median.toml sentinel2-median --bbox 410100,5170020,420000,5179920 --crs EPSG:32632 --resolution 10 -o median.tif
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
