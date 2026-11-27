#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/geotiff-read.sh
# writes geotiffs with gdal_translate in every layout geoplumb reads, then checks `geoplumb info` and `geoplumb run`
# against gdal's read of each:
# lzw: worldcover lon 7 to 8 and lat 46 to 47 (byte) and copernicus glo-30 n46 e007 (float32) as gdal cogs, lzw with
# gdal defaults and lzw with predictor 2 and 3;
# striped: copernicus float32 raw and deflate with predictor 3, a 2400 x 2401 worldcover window raw and lzw with
# predictor 2, and three distinct byte bands of that window pixel-interleaved;
# multiband: three distinct byte bands of the worldcover window band-interleaved striped raw, striped lzw with
# predictor 2 and tiled deflate, and three distinct copernicus float32 bands deflate with predictor 3, band- and
# pixel-interleaved;
# zstd: copernicus as a cog with overviews and as strips with predictor 3, the worldcover window as byte strips with
# predictor 2, and three distinct bands of it band- and pixel-interleaved;
# bigtiff: copernicus as a zstd cog with overviews and as raw strips, three distinct worldcover bands as
# band-interleaved lzw strips with predictor 2 and pixel-interleaved deflate tiles;
# jpeg and webp: copernicus scaled to bytes, three distinct bands as a jpeg cog (ycbcr), jpeg ycbcr tiles, jpeg rgb
# strips, jpeg band-interleaved strips, one grey band as jpeg strips, webp lossless with three and four bands and webp
# lossy at level 75.
# thresholds:
# gdal reports every input with the layout its name gives (lzw and its predictor, one block one image wide, zstd,
# tiff version 43, jpeg or webp, band or pixel interleave), at least one striped input has a short last strip, and
# no two bands of a multi-band input are equal;
# geoplumb info gives gdal's size, band count, sample format and compression;
# geoplumb run over the file's own extent, crs and pixel size gives gdal's grid and dtype;
# lossy webp inputs have gdal's nodata cells and differ from gdal's read by at most one level in any cell, every
# other input equals gdal's read in every cell of every band;
# no input is as close to its read shifted one pixel east as to the read itself
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem" "$worldcover"

names=()
write() {
  local name=$1
  shift
  gdal_translate -q "$@" "$work/$name.tif" || { echo "FAIL: gdal_translate $* $name.tif"; exit 1; }
  names+=("$name")
}

# lon 7 to 8 and lat 46 to 47, the copernicus tile's extent
gdal_translate -q -srcwin 12000 12000 12000 12000 "$worldcover" "$work/worldcover_crop.tif" || { echo "FAIL: gdal_translate crop"; exit 1; }
write lzw_worldcover_default -of COG -co COMPRESS=LZW "$work/worldcover_crop.tif"
write lzw_worldcover_predictor2 -of COG -co COMPRESS=LZW -co PREDICTOR=2 "$work/worldcover_crop.tif"
write lzw_copdem_default -of COG -co COMPRESS=LZW "$dem"
write lzw_copdem_predictor3 -of COG -co COMPRESS=LZW -co PREDICTOR=3 "$dem"

window="-srcwin 18000 18000 2400 2401"
worldcover_bands="$window -b 1 -b 1 -b 1 -scale_2 0 100 100 0 -scale_3 0 100 50 0"
write striped_copdem_raw "$dem"
write striped_copdem_deflate_predictor3 -co COMPRESS=DEFLATE -co PREDICTOR=3 "$dem"
# shellcheck disable=SC2086
write striped_worldcover_raw $window "$worldcover"
# shellcheck disable=SC2086
write striped_worldcover_lzw_predictor2 $window -co COMPRESS=LZW -co PREDICTOR=2 "$worldcover"
# gdal_translate copies INTERLEAVE=BAND from a single-band cog source when writing uncompressed
# shellcheck disable=SC2086
write striped_worldcover_three_bands $worldcover_bands -co INTERLEAVE=PIXEL "$worldcover"

copdem_bands="-ot Float32 -b 1 -b 1 -b 1 -scale_2 0 5000 5000 0 -scale_3 0 5000 0 2500 -co COMPRESS=DEFLATE -co PREDICTOR=3"
# shellcheck disable=SC2086
write multiband_worldcover_band_raw $worldcover_bands "$worldcover"
# shellcheck disable=SC2086
write multiband_worldcover_band_lzw_predictor2 $worldcover_bands -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 "$worldcover"
# shellcheck disable=SC2086
write multiband_worldcover_band_tiled_deflate $worldcover_bands -co INTERLEAVE=BAND -co TILED=YES -co COMPRESS=DEFLATE "$worldcover"
# shellcheck disable=SC2086
write multiband_copdem_band_deflate_predictor3 $copdem_bands -co INTERLEAVE=BAND "$dem"
# shellcheck disable=SC2086
write multiband_copdem_pixel_deflate_predictor3 $copdem_bands -co INTERLEAVE=PIXEL "$dem"

write zstd_copdem_cog -of COG -co COMPRESS=ZSTD "$dem"
write zstd_copdem_strips_predictor3 -co COMPRESS=ZSTD -co PREDICTOR=3 "$dem"
# shellcheck disable=SC2086
write zstd_worldcover_strips_predictor2 $window -co COMPRESS=ZSTD -co PREDICTOR=2 "$worldcover"
# shellcheck disable=SC2086
write zstd_worldcover_band_predictor2 $worldcover_bands -co COMPRESS=ZSTD -co PREDICTOR=2 -co INTERLEAVE=BAND "$worldcover"
# shellcheck disable=SC2086
write zstd_worldcover_pixel_predictor2 $worldcover_bands -co COMPRESS=ZSTD -co PREDICTOR=2 -co INTERLEAVE=PIXEL "$worldcover"

write bigtiff_copdem_cog -of COG -co BIGTIFF=YES -co COMPRESS=ZSTD "$dem"
write bigtiff_copdem_strips -co BIGTIFF=YES "$dem"
# shellcheck disable=SC2086
write bigtiff_worldcover_band_lzw $worldcover_bands -co BIGTIFF=YES -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 "$worldcover"
# shellcheck disable=SC2086
write bigtiff_worldcover_pixel_tiled_deflate $worldcover_bands -co BIGTIFF=YES -co INTERLEAVE=PIXEL -co TILED=YES -co COMPRESS=DEFLATE "$worldcover"

three="-ot Byte -b 1 -b 1 -b 1 -scale_1 300 4600 0 255 -scale_2 300 4600 255 0 -scale_3 300 2500 0 255"
grey="-ot Byte -b 1 -scale 300 4600 0 255"
four="$three -b 1 -scale_4 300 3500 255 0"
# shellcheck disable=SC2086
write jpeg_cog_ycbcr -of COG -co COMPRESS=JPEG $three "$dem"
# shellcheck disable=SC2086
write jpeg_tiles_ycbcr -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co TILED=YES $three "$dem"
# shellcheck disable=SC2086
write jpeg_strips_rgb -co COMPRESS=JPEG -co PHOTOMETRIC=RGB $three "$dem"
# shellcheck disable=SC2086
write jpeg_strips_band -co COMPRESS=JPEG -co INTERLEAVE=BAND $three "$dem"
# shellcheck disable=SC2086
write jpeg_strips_grey -co COMPRESS=JPEG $grey "$dem"
# shellcheck disable=SC2086
write webp_lossless_three -co COMPRESS=WEBP -co WEBP_LOSSLESS=YES -co TILED=YES $three "$dem"
# shellcheck disable=SC2086
write webp_lossless_four -co COMPRESS=WEBP -co WEBP_LOSSLESS=YES -co TILED=YES $four "$dem"
# shellcheck disable=SC2086
write webp_lossy_three -co COMPRESS=WEBP -co WEBP_LEVEL=75 -co TILED=YES $three "$dem"

for name in "${names[@]}"; do
  file=$work/$name.tif
  read -r west south east north pixel epsg < <(file_grid "$file")
  [ -n "${epsg:-}" ] || { echo "INCONCLUSIVE: no square-pixel grid or epsg for $name"; exit 1; }
  value_kind=""
  case $name in *worldcover*) value_kind=', value_kind = "categorical"' ;; esac
  printf '[[layer]]\nname = "%s"\nsource = { kind = "cog", path = "%s"%s }\n' "$name" "$file" "$value_kind" >"$work/$name.toml"
  "$geoplumb" info "$file" >"$work/$name.info" 2>&1 || { cat "$work/$name.info"; echo "FAIL: geoplumb info on $name"; exit 1; }
  "$geoplumb" run "$work/$name.toml" "$name" --bbox "$west,$south,$east,$north" --crs "EPSG:$epsg" --resolution "$pixel" \
    -o "$work/$name.run.tif" >"$work/$name.run.log" 2>&1 || { cat "$work/$name.run.log"; echo "FAIL: geoplumb run on $name"; exit 1; }
done

python3 -I - "$work" "${names[@]}" <<'EOF'
import re
import sys
from pathlib import Path

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = Path(sys.argv[1])
names = sys.argv[2:]
# geoplumb info names formats as terrano's SampleFormat::name and compression in lower case
gdal_formats = {"Byte": "u8", "Int16": "i16", "UInt16": "u16", "Int32": "i32", "UInt32": "u32", "Float32": "f32", "Float64": "f64"}
# 512 rows of the 12000 px lzw crop in float64 take 49 MB per array
compared_rows = 512
failures = []
inconclusive = []
short_last_strip = False


def rows_with_nan(dataset, row, rows):
    bands = []
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        values = band.ReadAsArray(0, row, dataset.RasterXSize, rows).astype(numpy.float64)
        if band.GetNoDataValue() is not None:
            values[values == band.GetNoDataValue()] = numpy.nan
        bands.append(values)
    return numpy.stack(bands)


def finite_max(values, current):
    finite = values[~numpy.isnan(values)]
    return max(current, float(finite.max())) if finite.size else current


def compare(run, source):
    largest = shifted = 0.0
    differing = nodata_mismatch = 0
    equal_bands = {(a, b) for a in range(source.RasterCount) for b in range(a + 1, source.RasterCount)}
    for row in range(0, source.RasterYSize, compared_rows):
        rows = min(compared_rows, source.RasterYSize - row)
        ours, theirs = rows_with_nan(run, row, rows), rows_with_nan(source, row, rows)
        missing_ours, missing_theirs = numpy.isnan(ours), numpy.isnan(theirs)
        differing += int((~((ours == theirs) | (missing_ours & missing_theirs))).sum())
        nodata_mismatch += int((missing_ours ^ missing_theirs).sum())
        largest = finite_max(numpy.abs(ours - theirs), largest)
        shifted = finite_max(numpy.abs(ours[:, :, :-1] - theirs[:, :, 1:]), shifted)
        equal_bands = {(a, b) for a, b in equal_bands if numpy.array_equal(theirs[a], theirs[b], equal_nan=True)}
    return {"largest": largest, "shifted": shifted, "differing": differing, "nodata_mismatch": nodata_mismatch, "equal_bands": equal_bands}


for name in names:
    failures_before = len(failures)
    family = name.split("_")[0]
    source = gdal.Open(str(work / f"{name}.tif"))
    band = source.GetRasterBand(1)
    block_width, block_rows = band.GetBlockSize()
    width, height = source.RasterXSize, source.RasterYSize
    structure = source.GetMetadata("IMAGE_STRUCTURE")
    compression = structure.get("COMPRESSION", "none").lower()
    if compression == "ycbcr jpeg":
        compression = "jpeg"
    interleave = structure.get("INTERLEAVE")
    predictor = structure.get("PREDICTOR", "1")
    print(f"{name}: {width} x {height}, {source.RasterCount} bands of {gdal.GetDataTypeName(band.DataType)}, "
          f"blocks {block_width} x {block_rows}, compression {structure.get('COMPRESSION', 'none')}, predictor {predictor}, "
          f"interleave {interleave}, overviews {band.GetOverviewCount()}")

    if family == "lzw":
        wanted_predictor = name[-1] if "predictor" in name else "1"
        if compression != "lzw" or predictor != wanted_predictor:
            inconclusive.append(f"{name} is {compression} with predictor {predictor}, not lzw with predictor {wanted_predictor}")
    if family == "striped":
        if block_width != width or block_rows >= height:
            inconclusive.append(f"{name} is not striped: blocks {block_width} x {block_rows}")
        if height % block_rows:
            short_last_strip = True
    if family in ("zstd", "jpeg", "webp") and compression != family:
        inconclusive.append(f"{name} is {compression}, not {family}")
    if family == "bigtiff":
        with open(work / f"{name}.tif", "rb") as file:
            version = int.from_bytes(file.read(4)[2:4], "little")
        if version != 43:
            inconclusive.append(f"{name} is tiff version {version}, not bigtiff")
    if "_band_" in name or "_pixel_" in name:
        wanted_interleave = "BAND" if "_band_" in name else "PIXEL"
        if interleave != wanted_interleave or source.RasterCount != 3:
            inconclusive.append(f"{name} is not three {wanted_interleave.lower()}-interleaved bands")

    info = (work / f"{name}.info").read_text()
    expected = {
        "size": f"{width} x {height}",
        "bands": str(source.RasterCount),
        "sample format": gdal_formats[gdal.GetDataTypeName(band.DataType)],
        "compression": compression,
    }
    for key, value in expected.items():
        found = re.search(rf"^{key}: (.*)$", info, re.M)
        if not found or found.group(1).strip().lower() != value:
            failures.append(f"{name} info {key}: geoplumb {found.group(1) if found else None!r}, gdal {value!r}")
    info_matches = len(failures) == failures_before

    run = gdal.Open(str(work / f"{name}.run.tif"))
    if (run.RasterXSize, run.RasterYSize, run.RasterCount) != (width, height, source.RasterCount):
        failures.append(f"{name} run is {run.RasterXSize} x {run.RasterYSize} x {run.RasterCount}, the input {width} x {height} x {source.RasterCount}")
        print(f"RESULT {name}: run grid differs from the input FAIL")
        continue
    if any(abs(a - b) > 1e-9 for a, b in zip(run.GetGeoTransform(), source.GetGeoTransform())):
        failures.append(f"{name} run geotransform {run.GetGeoTransform()}, the input {source.GetGeoTransform()}")
    if run.GetRasterBand(1).DataType != band.DataType:
        failures.append(f"{name} run is {gdal.GetDataTypeName(run.GetRasterBand(1).DataType)}, the input {gdal.GetDataTypeName(band.DataType)}")

    measured = compare(run, source)
    cells = width * height * source.RasterCount
    if family == "webp" and "lossy" in name:
        if measured["nodata_mismatch"]:
            failures.append(f"{name} has nodata in {measured['nodata_mismatch']} cells where gdal's read has a value, or the reverse")
        if measured["largest"] > 1:
            failures.append(f"{name} differs from gdal's read by up to {measured['largest']:g} levels, past one")
    elif measured["differing"]:
        failures.append(f"{name} differs from gdal's read in {measured['differing']} cells")
    if measured["shifted"] <= measured["largest"]:
        inconclusive.append(f"{name} is as close to its read shifted one pixel east ({measured['shifted']:g}) as to the read itself")
    if measured["equal_bands"]:
        inconclusive.append(f"{name} has two equal bands, so a band swap goes unseen")
    verdict = "PASS" if len(failures) == failures_before else "FAIL"
    print(f"RESULT {name}: info {'matches' if info_matches else 'differs from'} gdal, run against gdal's read "
          f"{measured['differing']} of {cells} cells differ, largest difference {measured['largest']:g} {verdict}")

if not short_last_strip:
    inconclusive.append("no striped input has a short last strip")
for failure in failures:
    print("FAIL:", failure)
for reason in inconclusive:
    print("INCONCLUSIVE:", reason)
sys.exit(3 if failures else 4 if inconclusive else 0)
EOF
status=$?

case $status in
  0) echo "PROOF PASS"; exit 0 ;;
  3) echo "PROOF FAIL: see the FAIL lines above"; exit 1 ;;
  4) echo "INCONCLUSIVE: see the INCONCLUSIVE lines above"; exit 1 ;;
  *) echo "UNEXPECTED: the checks exited $status"; exit 1 ;;
esac
