#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/color-relief.sh
# colours rasters with a color_relief layer and with
# gdaldem color-relief -alpha on the same colour file: copernicus glo-30 n46 e007 through absolute stops, the same
# stops with -nearest_color_entry, and percent stops with the band's exact min and max as the range, and a
# 2400 x 2401 esa worldcover window through one colour per class with -exact_color_entry.
# thresholds, fixed before the first run:
# every input has at least four distinct colours in gdal's output, so the colouring shows;
# geoplumb run gives gdal's grid and four byte bands, each equal to gdaldem's in every cell
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem" "$worldcover"

cp "$dem" "$work/dem.tif"
gdal_translate -q -srcwin 18000 18000 2400 2401 -of COG -co COMPRESS=LZW "$worldcover" "$work/worldcover.tif" \
  || { echo "FAIL: gdal_translate"; exit 1; }
cat >"$work/elevation.txt" <<'COLORS'
# elevation stops, metres
nv 0 0 0 0
400 20 90 30
1000 120 170 60
1800 230 220 120
2600 160 110 60
3400 140 140 140
4400 255 255 255
COLORS
cat >"$work/percent.txt" <<'COLORS'
0% 0 0 120
20% 0 160 200
45% 90 200 90
70% 220 200 80
100% 250 250 250 200
COLORS
cat >"$work/classes.txt" <<'COLORS'
nv 0 0 0 0
10 0 100 0
20 255 187 34
30 255 255 76
40 240 150 255
50 250 0 0
60 180 180 180
70 240 240 240
80 0 100 200
90 0 150 160
95 0 207 117
100 250 230 160
COLORS
read -r dem_min dem_max < <(python3 -I -c '
import sys
from osgeo import gdal
gdal.UseExceptions()
dataset = gdal.Open(sys.argv[1])
print(*map(repr, dataset.GetRasterBand(1).ComputeRasterMinMax(False)))
' "$work/dem.tif")

# name, raster, colour file, geoplumb selection, gdaldem flag
cases="dem_interpolate:dem:elevation:interpolate:
dem_nearest:dem:elevation:nearest:-nearest_color_entry
dem_percent:dem:percent:interpolate:
worldcover_exact:worldcover:classes:exact:-exact_color_entry"
names=""
while IFS=: read -r name raster colors selection flag; do
  names="$names $name"
  file=$work/$raster.tif
  read -r west south east north pixel _ < <(file_grid "$file")
  [ -n "${pixel:-}" ] || { echo "INCONCLUSIVE: no square-pixel grid for $name"; exit 1; }
  range=""
  [ "$colors" = percent ] && range=$'\n'"range = { min = $dem_min, max = $dem_max }"
  cat >"$work/$name.toml" <<TOML
[[layer]]
name = "$name"
source = { kind = "cog", path = "$file" }

[[layer.op]]
kind = "color_relief"
color_file = "$work/$colors.txt"
selection = "$selection"$range
TOML
  run_args=("$work/$name.toml" "$name" --bbox "$west,$south,$east,$north" --crs EPSG:4326 --resolution "$pixel")
  "$geoplumb" run "${run_args[@]}" -o "$work/$name.run.tif" >"$work/$name.run.log" 2>&1 \
    || { cat "$work/$name.run.log"; echo "FAIL: geoplumb run on $name"; exit 1; }
  # shellcheck disable=SC2086
  gdaldem color-relief -q -alpha $flag "$file" "$work/$colors.txt" "$work/$name.gdal.tif" \
    || { echo "FAIL: gdaldem on $name"; exit 1; }
done <<<"$cases"

# shellcheck disable=SC2086
python3 -I - "$work" $names <<'EOF'
import sys
from pathlib import Path

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = Path(sys.argv[1])
failures = []
inconclusive = []
for name in sys.argv[2:]:
    failures_before = len(failures)
    ours_dataset = gdal.Open(str(work / f"{name}.run.tif"))
    theirs_dataset = gdal.Open(str(work / f"{name}.gdal.tif"))
    theirs = theirs_dataset.ReadAsArray()
    colours = len(numpy.unique(theirs.reshape(theirs.shape[0], -1), axis=1).T)
    if colours < 4:
        inconclusive.append(f"{name}: gdal's output holds {colours} colours")
    if ours_dataset.RasterCount != 4 or ours_dataset.GetRasterBand(1).DataType != gdal.GDT_Byte:
        failures.append(f"{name}: {ours_dataset.RasterCount} bands of {gdal.GetDataTypeName(ours_dataset.GetRasterBand(1).DataType)}")
        continue
    ours = ours_dataset.ReadAsArray()
    if ours.shape != theirs.shape:
        failures.append(f"{name}: geoplumb {ours.shape}, gdal {theirs.shape}")
        continue
    if any(abs(a - b) > 1e-9 for a, b in zip(ours_dataset.GetGeoTransform(), theirs_dataset.GetGeoTransform())):
        failures.append(f"{name}: geotransform {ours_dataset.GetGeoTransform()}, gdal {theirs_dataset.GetGeoTransform()}")
    differing = (ours != theirs).any(axis=0)
    largest = int(numpy.abs(ours.astype(int) - theirs.astype(int)).max())
    if differing.any():
        row, col = numpy.argwhere(differing)[0]
        failures.append(f"{name}: {int(differing.sum())} cells differ, first at row {row} col {col}: geoplumb {ours[:, row, col]}, gdal {theirs[:, row, col]}")
    print(f"RESULT {name}: {colours} colours in gdal's output, {int(differing.sum())} of {differing.size} cells differ, "
          f"largest difference {largest} {'PASS' if len(failures) == failures_before else 'FAIL'}")

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
