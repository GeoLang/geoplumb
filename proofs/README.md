# proofs

Scripts that compare the current geoplumb with GDAL on public data. [docs/gdal-parity.md](../docs/gdal-parity.md) holds the results.

Needs bash, curl, GDAL 3.12 or later with its Python bindings and numpy, and GNU time for `bench.sh`. The warp check in `cli.sh` passes against gdalwarp 3.13 and fails against 3.12, see the reprojection row of the parity table.

```sh
cargo build --release -p geoplumb-server --bin geoplumb
export GEOPLUMB_BIN=$PWD/target/release/geoplumb
export GEOPLUMB_PROOF_DATA=$HOME/geoplumb-proof-data
proofs/fetch.sh
proofs/cutline.sh
```

`fetch.sh` downloads Copernicus GLO-30 N46 E007, ESA WorldCover 2021 N45E006 and Natural Earth 1:110m countries, about 150 MB, and checks each against the sha256 the published results were measured on. `sentinel2-median.sh` reads Earth Search over HTTP and needs no fetch.

Each script prints one `RESULT <input>: ... PASS` or `FAIL` line per input and ends with `PROOF PASS`, `PROOF FAIL` or `INCONCLUSIVE`. It exits 0 only on `PROOF PASS`. `INCONCLUSIVE` means an input did not have the property the comparison depends on, such as a GDAL file that is not striped.

| Script | Compares |
|---|---|
| `geotiff-read.sh` | `geoplumb info` and `geoplumb run` against GDAL's read of 31 GeoTIFFs `gdal_translate` writes: LZW, striped, multi-band, ZSTD, BigTIFF, JPEG and WebP |
| `cli.sh` | `geoplumb info` against `gdalinfo`, `geoplumb run` to UTM 32N against `gdalwarp -r bilinear -et 0` |
| `export-cog.sh` | a categorical COG in UTM 32N against `gdalwarp -r near` and GDAL's COG validator |
| `raster-vector.sh` | `contours` against `gdal_contour` |
| `polygonize.sh` | `polygonize` against GDAL's polygonize |
| `viewshed.sh` | `viewshed` against `gdal_viewshed` |
| `native-dtype.sh` | byte and float32 exports against `gdal_translate` and `gdalwarp -r near` |
| `cutline.sh` | `clip_raster` against `gdalwarp -cutline` |
| `color-relief.sh` | `color_relief` against `gdaldem color-relief -alpha` |
| `vector-io.sh` | shapefile and CSV sources and `.gpkg` output against `ogr2ogr` and OGR's read |
| `sentinel2-median.sh` | `examples/sentinel2-median.toml` against the closest GDAL 3.12 pipeline, output and wall clock |
| `bench.sh` | wall clock and peak memory of hillshade, warp, contours and COG export against GDAL, median of three runs, as a markdown table |

`color-relief.sh` needs a geoplumb with the `color_relief` op and `vector-io.sh` one with shapefile, CSV and `.gpkg` support. On a build without them the run fails at the layer file.
