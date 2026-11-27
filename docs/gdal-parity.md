# geoplumb against GDAL

Each row is one geoplumb operator compared with the GDAL command that does the same job, on public data. The script in the last column reruns the comparison against your own build, see [proofs/README.md](../proofs/README.md). The results come from one run of every script through `geoplumb run`, against GDAL 3.12.4 and PROJ 9.8.1, unless the row names another version.

| Operator | GDAL command | Input | Measure | Result | Script |
|---|---|---|---|---|---|
| GeoTIFF read, LZW | `gdal_translate -of COG -co COMPRESS=LZW`, with and without `-co PREDICTOR=2` or `3` | ESA WorldCover N45E006 (byte), Copernicus GLO-30 N46E007 (float32) | `geoplumb info` against GDAL, `geoplumb run` over the file's own grid against GDAL's read of every cell | 4 files, info matches, 0 cells differ | `geotiff-read.sh` |
| GeoTIFF read, striped | `gdal_translate` defaults: raw, deflate predictor 3, LZW predictor 2, three pixel-interleaved bands | Copernicus GLO-30 N46E007, a 2400 x 2401 WorldCover window | as above | 5 files, info matches, 0 cells differ | `geotiff-read.sh` |
| GeoTIFF read, multi-band | band-interleaved raw, LZW and tiled deflate, band- and pixel-interleaved deflate predictor 3 | as above | as above | 5 files, info matches, 0 cells differ | `geotiff-read.sh` |
| GeoTIFF read, ZSTD | `-co COMPRESS=ZSTD`: a COG with overviews, strips with predictor 2 and 3, band- and pixel-interleaved | as above | as above | 5 files, info matches, 0 cells differ | `geotiff-read.sh` |
| GeoTIFF read, BigTIFF | `-co BIGTIFF=YES`: a ZSTD COG, raw strips, band-interleaved LZW strips, pixel-interleaved deflate tiles | as above | as above | 4 files, info matches, 0 cells differ | `geotiff-read.sh` |
| GeoTIFF read, JPEG and WebP | `-co COMPRESS=JPEG` (YCbCr COG and tiles, RGB, band and grey strips) and `-co COMPRESS=WEBP` (lossless 3 and 4 band, lossy) | Copernicus GLO-30 N46E007 scaled to bytes | as above | 8 files, info matches, 0 cells differ | `geotiff-read.sh` |
| `geoplumb info` | `gdalinfo` | GDAL LZW COG of Copernicus GLO-30 N46E007 | size, bands, compression, overviews | matches, 3 overviews | `cli.sh` |
| reprojection | `gdalwarp -t_srs EPSG:32632 -tr 30 30 -r bilinear -et 0` | as above, 20 km in UTM 32N | every pixel within 1 cm | against gdalwarp 3.13.3: 444889 pixels, max 0.67 mm. Against gdalwarp 3.12.4: median 11 cm, max 6.9 m, so the script fails on 3.12 | `cli.sh` |
| COG export, categorical | `gdalwarp -t_srs EPSG:32632 -tr 20 20 -r near` | ESA WorldCover N45E006, 20 km around 7.5 E 46.5 N | valid COG, class codes at every level, base level agreement at least 99% | valid COG, 0 non-class values at both levels, 99.394% agree | `export-cog.sh` |
| contours | `gdal_contour -i 100` | Copernicus GLO-30 N46E007, 900 x 900 pixels pulled in 256 px chunks | length per level within 1%, no line ends inside the window | 33 levels, 0.000% length difference, 176 open and 209 closed lines on both, 0 line ends inside | `raster-vector.sh` |
| polygonize | `gdal.Polygonize`, 4-connected, the call `gdal_polygonize` makes | ESA WorldCover N45E006, 1024 x 1024 pixels pulled in 256 px chunks | per class equal polygon and hole counts, no area in the symmetric difference | all 9 classes equal in polygon count, hole count and area, symmetric difference at most 1e-6 cells | `polygonize.sh` |
| viewshed | `gdal_viewshed -oz 2 -md 10000 -cc 0` | Copernicus GLO-30 N46E007 warped to UTM 32N at 30 m, observer at 7.65 E 46.645 N | at least 95% of in-disc cells agree, quadrant exports equal the whole | 343838 of 345917 in-disc cells agree (99.399%), quadrants equal the whole | `viewshed.sh` |
| native sample format | `gdal_translate -srcwin`, `gdalwarp -r near` | a 2400 px WorldCover window (byte), a 900 px Copernicus window (float32) | byte equal with the palette, float32 bit-equal | byte equal to both with the palette, float32 bit-equal | `native-dtype.sh` |
| `clip_raster` | `gdalwarp -cutline` | Copernicus GLO-30 N46E007 and a 2400 x 2401 WorldCover window, each with a 40 vertex star holding a 25 vertex hole, and the Copernicus tile in UTM 32N with the star in WGS84 | every cell equal on the source grid, cells kept by one tool only within one pixel of the cutline in UTM | 0 cells differ on both source grids, in UTM 231 of 9866637 cells kept by one tool only, all within one pixel | `cutline.sh` |
| `color_relief` | `gdaldem color-relief -alpha`, plain, `-nearest_color_entry`, `-exact_color_entry` | Copernicus GLO-30 N46E007 with absolute and percent stops, a 2400 x 2401 WorldCover window with one colour per class | all four bands equal in every cell | 0 cells differ in all four cases | `color-relief.sh` |
| vector I/O | `ogr2ogr` to shapefile, OGR's read of the GeoPackage | Natural Earth 1:110m countries in WGS84 and clipped to central Europe in UTM 32N, a CSV of one point per country | every feature and attribute, geometry within 1e-9 of its area, `validate_gpkg.py` clean | 177 and 16 features, symmetric difference at most 5.0e-17 of a feature's area, CSV types kept | `vector-io.sh` |
| STAC median composite | `gdalwarp` per item, `gdal raster calc` mask on SCL, `gdal raster mosaic --pixel-function median`, `gdal raster stack` | Earth Search Sentinel-2 L2A red, green and blue over 9.9 km around Interlaken, August 2025, SCL 4, 5 and 6 kept | validity and values agree on 99.9% of cells, wall clock | 19 items, every cell equal in all three bands, geoplumb 31.5 s, GDAL 93.2 s | `sentinel2-median.sh` |

GDAL 3.12 has no single command for the last row. The STACIT driver mosaics a search into one raster with the newest item on top, with no per-item mask and no median, so the GDAL side searches the STAC API itself, cuts each item's bands onto the output grid with `gdalwarp`, masks each band with `gdal raster calc` and takes the median per band with `gdal raster mosaic --pixel-function median`.

## Speed

`bench.sh`, median of three runs over Copernicus GLO-30 N46E007 (3600 x 3600 float32) on a 48 thread machine:

| operation | GDAL command | geoplumb wall s | GDAL wall s | geoplumb peak MB | GDAL peak MB |
| --- | --- | ---: | ---: | ---: | ---: |
| hillshade | `gdaldem hillshade -az 315 -alt 45` | 10.34 | 0.37 | 309 | 154 |
| warp | `gdalwarp -r bilinear -et 0`, UTM 32N at 30 m | 8.47 | 2.49 | 289 | 188 |
| contours | `gdal_contour -i 100` | 1.40 | 4.63 | 2197 | 189 |
| COG | `gdal_translate -of COG` | 5.71 | 0.94 | 125 | 155 |

`geoplumb run` writes every raster as a deflate COG with overviews. gdaldem and gdalwarp write a GeoTIFF without overviews, and `gdal_translate -of COG` writes an LZW COG with overviews.
