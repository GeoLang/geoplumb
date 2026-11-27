#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/native-dtype.sh
# exports with geoplumb run and checks each against gdal: a 2400 px worldcover window on its own grid as byte with
# nodata 0, equal to gdal_translate -srcwin and gdalwarp -r near, with the source palette; the same window without
# source overviews holding only class codes at every overview level, against a continuous control that blends them;
# worldcover in utm 32n at 20 m as byte against gdalwarp -r near, printed and not gated; a 900 px copernicus glo-30
# window as float32 bit-equal to gdal_translate -srcwin
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$worldcover" "$dem"
# design 8.9 check 1: 7.5 to 7.7 e and 46.3 to 46.5 n in worldcover pixels
worldcover_column=18000
worldcover_row=18000
worldcover_side=2400
# raster-vector.sh's window, lon 7.70 to 7.95 and lat 46.45 to 46.70
copernicus_column=2520
copernicus_row=1080
copernicus_side=900

# gdalinfo -json prints the geotransform to 16 decimals, which drops digits of worldcover's 1/12000 degree pixel
read -r worldcover_min_x worldcover_min_y worldcover_max_x worldcover_max_y worldcover_pixel \
  copernicus_min_x copernicus_min_y copernicus_max_x copernicus_max_y copernicus_pixel < <(python3 -I -c '
import sys

from osgeo import gdal

gdal.UseExceptions()


def window(path, column, row, side):
    origin_x, pixel_x, _, origin_y, _, pixel_y = gdal.Open(path).GetGeoTransform()
    if pixel_x != -pixel_y:
        sys.exit(f"{path} has {pixel_x} by {-pixel_y} pixels, and run takes one square pixel size")
    column, row, side = int(column), int(row), int(side)
    return [origin_x + column * pixel_x, origin_y - (row + side) * pixel_x,
            origin_x + (column + side) * pixel_x, origin_y - row * pixel_x, pixel_x]


print(*map(repr, window(*sys.argv[1:5]) + window(*sys.argv[5:9])))
' "$worldcover" $worldcover_column $worldcover_row $worldcover_side "$dem" $copernicus_column $copernicus_row $copernicus_side)
if [ -z "${copernicus_pixel:-}" ]; then echo "INCONCLUSIVE: no on-grid export window, see the line above"; exit 1; fi
echo "worldcover window $worldcover_min_x $worldcover_min_y $worldcover_max_x $worldcover_max_y at $worldcover_pixel"
echo "copernicus window $copernicus_min_x $copernicus_min_y $copernicus_max_x $copernicus_max_y at $copernicus_pixel"

# esa ships worldcover with overviews down to 562 px, and CogSrc serves every export level from them without decimating
worldcover_flat=$work/worldcover-flat.tif
gdal_translate -q -of COG -co OVERVIEWS=NONE -co COMPRESS=DEFLATE \
  -srcwin $worldcover_column $worldcover_row $worldcover_side $worldcover_side \
  "$worldcover" "$worldcover_flat" || { echo "FAIL: gdal_translate worldcover without overviews"; exit 1; }

# export-cog.sh's window: the utm 32n point under 7.5 e 46.5 n, snapped to the 20 m grid, then 10 km either way
read -r utm_min_x utm_min_y utm_max_x utm_max_y < <(python3 -I -c '
from osgeo import osr
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
utm = osr.SpatialReference()
utm.ImportFromEPSG(32632)
x, y, _ = osr.CoordinateTransformation(wgs84, utm).TransformPoint(7.5, 46.5)
x, y = round(x / 20) * 20, round(y / 20) * 20
print(x - 10000, y - 10000, x + 10000, y + 10000)
')
if [ -z "${utm_max_y:-}" ]; then echo "FAIL: no utm window"; exit 1; fi
echo "utm window in EPSG:32632: $utm_min_x $utm_min_y $utm_max_x $utm_max_y"

cat >"$work/worldcover.toml" <<EOF
[[layer]]
name = "worldcover"
gray = { min = 0.0, max = 100.0 }
source = { kind = "cog", path = "$worldcover", value_kind = "categorical" }
EOF
cat >"$work/worldcover-flat.toml" <<EOF
[[layer]]
name = "worldcover-flat"
gray = { min = 0.0, max = 100.0 }
source = { kind = "cog", path = "$worldcover_flat", value_kind = "categorical" }

[[layer]]
name = "worldcover-flat-continuous"
gray = { min = 0.0, max = 100.0 }
source = { kind = "cog", path = "$worldcover_flat" }
EOF
cat >"$work/copernicus.toml" <<EOF
[[layer]]
name = "copernicus"
gray = { min = 0.0, max = 4500.0 }
source = { kind = "cog", path = "$dem" }
EOF
cat "$work/worldcover.toml" "$work/worldcover-flat.toml" "$work/copernicus.toml" >"$work/layers.toml"

export_layer() {
  local layer=$1 name=$2 bbox=$3 crs=$4 resolution=$5
  "$geoplumb" run "$work/layers.toml" "$layer" --bbox "$bbox" --crs "$crs" --resolution "$resolution" -o "$work/$name.tif" \
    || { echo "FAIL: $name export"; exit 1; }
}

worldcover_bbox="$worldcover_min_x,$worldcover_min_y,$worldcover_max_x,$worldcover_max_y"
export_layer worldcover export-worldcover "$worldcover_bbox" EPSG:4326 "$worldcover_pixel"
export_layer worldcover-flat export-worldcover-flat "$worldcover_bbox" EPSG:4326 "$worldcover_pixel"
export_layer worldcover-flat-continuous export-worldcover-flat-continuous "$worldcover_bbox" EPSG:4326 "$worldcover_pixel"
export_layer worldcover export-worldcover-utm "$utm_min_x,$utm_min_y,$utm_max_x,$utm_max_y" EPSG:32632 20
export_layer copernicus export-copernicus "$copernicus_min_x,$copernicus_min_y,$copernicus_max_x,$copernicus_max_y" EPSG:4326 "$copernicus_pixel"

gdal_translate -q -srcwin $worldcover_column $worldcover_row $worldcover_side $worldcover_side \
  "$worldcover" "$work/reference-worldcover-translate.tif" || { echo "FAIL: gdal_translate worldcover"; exit 1; }
gdal_translate -q -srcwin $((worldcover_column + 1)) $worldcover_row $worldcover_side $worldcover_side \
  "$worldcover" "$work/reference-worldcover-shifted.tif" || { echo "FAIL: gdal_translate shifted worldcover"; exit 1; }
gdalwarp -q -te "$worldcover_min_x" "$worldcover_min_y" "$worldcover_max_x" "$worldcover_max_y" \
  -tr "$worldcover_pixel" "$worldcover_pixel" -r near "$worldcover" "$work/reference-worldcover-warp.tif" || { echo "FAIL: gdalwarp worldcover"; exit 1; }
gdalwarp -q -t_srs EPSG:32632 -te "$utm_min_x" "$utm_min_y" "$utm_max_x" "$utm_max_y" -tr 20 20 -r near \
  "$worldcover" "$work/reference-worldcover-utm.tif" || { echo "FAIL: gdalwarp utm"; exit 1; }
gdal_translate -q -srcwin $copernicus_column $copernicus_row $copernicus_side $copernicus_side \
  "$dem" "$work/reference-copernicus-translate.tif" || { echo "FAIL: gdal_translate copernicus"; exit 1; }
gdal_translate -q -srcwin $((copernicus_column + 1)) $copernicus_row $copernicus_side $copernicus_side \
  "$dem" "$work/reference-copernicus-shifted.tif" || { echo "FAIL: gdal_translate shifted copernicus"; exit 1; }

python3 -I - "$work" "$worldcover" <<'EOF'
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = Path(sys.argv[1])
worldcover_source = sys.argv[2]
# esa worldcover v200 legend codes and 0, the file's nodata
worldcover_codes = numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=numpy.uint8)

failures = []
inconclusive = []


def gdal_info(path):
    return json.loads(subprocess.run(["gdalinfo", "-json", str(path)], check=True, capture_output=True).stdout)


def levels(path):
    # gdal's python bindings free a dataset nothing names, and its bands with it
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(1)
    return [band.ReadAsArray()] + [band.GetOverview(index).ReadAsArray() for index in range(band.GetOverviewCount())]


def base_with_nan(path):
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    if nodata is not None:
        values[values == nodata] = numpy.nan
    return values


def palette(band):
    return band.get("colorInterpretation"), band.get("colorTable")


def off_code_count(values):
    return int((~numpy.isin(values, worldcover_codes)).sum())


print("check 1: worldcover on its own grid")
failures_before = len(failures)
source_band = gdal_info(worldcover_source)["bands"][0]
export_info = gdal_info(work / "export-worldcover.tif")
export_band = export_info["bands"][0]
print(f"export: type {export_band['type']}, nodata {export_band.get('noDataValue')}, "
      f"geotransform {export_info['geoTransform']}, size {export_info['size']}")
print(f"gdal_translate window geotransform {gdal_info(work / 'reference-worldcover-translate.tif')['geoTransform']}")
if export_band["type"] != "Byte":
    failures.append(f"the worldcover export is {export_band['type']}, not Byte")
if export_band.get("noDataValue") != 0:
    failures.append(f"the worldcover export's nodata is {export_band.get('noDataValue')}, not 0")

export_levels = levels(work / "export-worldcover.tif")
base = export_levels[0]
for reference_name in ["reference-worldcover-translate.tif", "reference-worldcover-warp.tif"]:
    reference = levels(work / reference_name)[0]
    equal = base.dtype == numpy.uint8 and reference.dtype == numpy.uint8 and numpy.array_equal(base, reference)
    print(f"base {base.shape} {base.dtype} against {reference_name} {reference.shape} {reference.dtype}: "
          f"{'equal' if equal else 'differs'}")
    if not equal:
        failures.append(f"the worldcover base level differs from {reference_name}")
if numpy.array_equal(base, levels(work / "reference-worldcover-shifted.tif")[0]):
    inconclusive.append("the worldcover base equals the window shifted one pixel east, so equality shows nothing")

export_levels = levels(work / "export-worldcover-flat.tif")
if len(export_levels) < 2:
    failures.append("the worldcover export without source overviews has no overviews")
for level, values in enumerate(export_levels[1:], start=1):
    off_code = off_code_count(values)
    print(f"level {level} without source overviews: {values.shape[1]} x {values.shape[0]}, {off_code} values neither a class code nor 0")
    if off_code:
        failures.append(f"worldcover level {level} without source overviews holds {off_code} values that are neither a class code nor 0")
continuous_off_code = sum(off_code_count(values) for values in levels(work / "export-worldcover-flat-continuous.tif")[1:])
print(f"continuous control: {continuous_off_code} overview values neither a class code nor 0")
if not continuous_off_code:
    inconclusive.append("the continuous control blends no codes in its overviews, so the class check shows nothing")

factors = [round(base.shape[1] / values.shape[1]) for values in export_levels[1:]]
if factors:
    shutil.copy(work / "reference-worldcover-translate.tif", work / "reference-worldcover-mode.tif")
    subprocess.run(["gdaladdo", "-q", "-r", "mode", str(work / "reference-worldcover-mode.tif"), *map(str, factors)], check=True)
    for level, (values, reference) in enumerate(zip(export_levels[1:], levels(work / "reference-worldcover-mode.tif")[1:]), start=1):
        if values.shape != reference.shape:
            print(f"level {level} is {values.shape}, gdaladdo -r mode's is {reference.shape}, not compared")
            continue
        print(f"level {level} against gdaladdo -r mode, not gated: {100 * float((values == reference).mean()):.3f}% of pixels agree")

source_palette = palette(source_band)
export_palette = palette(export_band)
print(f"source colour interpretation {source_palette[0]}, {len((source_palette[1] or {}).get('entries', []))} entries")
print(f"export colour interpretation {export_palette[0]}, {len((export_palette[1] or {}).get('entries', []))} entries")
if source_palette[1] is None:
    inconclusive.append("the worldcover source has no colour table, so the palette check compares nothing")
if export_palette != source_palette:
    failures.append("the worldcover export's colour table is not the source's")
print(f"RESULT worldcover 2400 x 2400 window on its own grid: byte export {'equal' if len(failures) == failures_before else 'not equal'} "
      f"to gdal_translate -srcwin and gdalwarp -r near with the source palette, {continuous_off_code} off-code overview values "
      f"in the continuous control {'PASS' if len(failures) == failures_before else 'FAIL'}")

print("check 2: worldcover off its grid, EPSG:32632 at 20 m")
failures_before = len(failures)
export_band = gdal_info(work / "export-worldcover-utm.tif")["bands"][0]
print(f"export {export_band['type']}")
if export_band["type"] != "Byte":
    failures.append(f"the utm export is {export_band['type']}, not Byte")
export_levels = levels(work / "export-worldcover-utm.tif")
reference = levels(work / "reference-worldcover-utm.tif")[0]
utm_agreement = "not compared, the grids differ"
if reference.shape == export_levels[0].shape:
    either = (export_levels[0] != 0) | (reference != 0)
    agree = int(((export_levels[0] == reference) & either).sum())
    utm_agreement = f"{agree} of {int(either.sum())} pixels agree, {100 * agree / max(int(either.sum()), 1):.3f}%"
print(f"RESULT worldcover in EPSG:32632 at 20 m: {export_band['type']} export, against gdalwarp -r near (not gated) {utm_agreement} "
      f"{'PASS' if len(failures) == failures_before else 'FAIL'}")


def bit_equal(values, reference):
    if values.shape != reference.shape or values.dtype != numpy.float32 or reference.dtype != numpy.float32:
        return False
    missing = numpy.isnan(reference)
    return bool(numpy.array_equal(numpy.isnan(values), missing)
                and numpy.array_equal(values[~missing].view(numpy.uint32), reference[~missing].view(numpy.uint32)))


print("check 3: copernicus glo-30 on its own grid")
failures_before = len(failures)
export_info = gdal_info(work / "export-copernicus.tif")
export_band = export_info["bands"][0]
print(f"export: type {export_band['type']}, geotransform {export_info['geoTransform']}, size {export_info['size']}")
print(f"gdal_translate window geotransform {gdal_info(work / 'reference-copernicus-translate.tif')['geoTransform']}")
if export_band["type"] != "Float32":
    failures.append(f"the copernicus export is {export_band['type']}, not Float32")
export_base = levels(work / "export-copernicus.tif")[0]
reference = base_with_nan(work / "reference-copernicus-translate.tif")
print(f"reference {reference.shape} {reference.dtype}, {int(numpy.isnan(reference).sum())} nan, "
      f"{numpy.nanmin(reference):.2f} to {numpy.nanmax(reference):.2f} m")
export_equal = bit_equal(export_base, reference)
if not export_equal:
    failures.append("the copernicus export is not bit-equal to gdal_translate -srcwin")
print(f"RESULT copernicus glo-30 900 x 900 window on its own grid: {export_band['type']} export "
      f"{'bit-equal' if export_equal else 'not bit-equal'} to gdal_translate -srcwin {'PASS' if len(failures) == failures_before else 'FAIL'}")
if bit_equal(export_base, base_with_nan(work / "reference-copernicus-shifted.tif")):
    inconclusive.append("the copernicus export equals the window shifted one pixel east, so equality shows nothing")

for failure in failures:
    print("FAIL:", failure)
for reason in inconclusive:
    print("INCONCLUSIVE:", reason)
if failures:
    sys.exit(3)
if inconclusive:
    sys.exit(4)
sys.exit(0)
EOF
status=$?

case $status in
  0) echo "PROOF PASS"; exit 0 ;;
  3) echo "PROOF FAIL: see the FAIL lines above"; exit 1 ;;
  4) echo "INCONCLUSIVE: see the INCONCLUSIVE lines above"; exit 1 ;;
  *) echo "UNEXPECTED: the checks exited $status"; exit 1 ;;
esac
