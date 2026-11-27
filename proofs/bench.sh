#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/bench.sh
set -euo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem"
[ -x /usr/bin/time ] || { echo "/usr/bin/time is missing, bench.sh reads its -v report"; exit 1; }
runs=3

read -r west south east north pixel _ < <(file_grid "$dem")
# the tile's footprint in utm 32n from points along its edges, snapped outward to the 30 m grid
read -r utm_west utm_south utm_east utm_north < <(python3 -I -c '
import math
import sys
from osgeo import osr
west, south, east, north = map(float, sys.argv[1:5])
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
utm = osr.SpatialReference()
utm.ImportFromEPSG(32632)
transform = osr.CoordinateTransformation(wgs84, utm)
steps = [i / 64 for i in range(65)]
edges = [(west + (east - west) * s, y) for s in steps for y in (south, north)] + [(x, south + (north - south) * s) for s in steps for x in (west, east)]
points = [transform.TransformPoint(x, y)[:2] for x, y in edges]
xs, ys = [p[0] for p in points], [p[1] for p in points]
print(math.floor(min(xs) / 30) * 30, math.floor(min(ys) / 30) * 30, math.ceil(max(xs) / 30) * 30, math.ceil(max(ys) / 30) * 30)
' "$west" "$south" "$east" "$north")
[ -n "${utm_north:-}" ] || { echo "no utm footprint for $dem"; exit 1; }

cat >"$work/layers.toml" <<EOF
[[layer]]
name = "dem"
source = { kind = "cog", path = "$dem" }

[[layer]]
name = "hillshade"
source = { kind = "cog", path = "$dem" }

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0

[[layer]]
name = "contours"
source = { kind = "cog", path = "$dem" }

[[layer.op]]
kind = "contours"
interval = 100.0
EOF

own_grid=(--bbox "$west,$south,$east,$north" --crs EPSG:4326 --resolution "$pixel")
utm_grid=(--bbox "$utm_west,$utm_south,$utm_east,$utm_north" --crs EPSG:32632 --resolution 30)

measure() {
  local label=$1 output=$2
  shift 2
  local run
  for run in $(seq "$runs"); do
    rm -f "$output"
    /usr/bin/time -v -o "$work/$label.$run.time" "$@" >"$work/$label.$run.log" 2>&1 \
      || { cat "$work/$label.$run.log"; echo "$label failed"; exit 1; }
  done
}

measure geoplumb-hillshade "$work/out.tif" "$geoplumb" run "$work/layers.toml" hillshade "${own_grid[@]}" -o "$work/out.tif"
measure gdal-hillshade "$work/out.tif" gdaldem hillshade -q -az 315 -alt 45 "$dem" "$work/out.tif"
measure geoplumb-warp "$work/out.tif" "$geoplumb" run "$work/layers.toml" dem "${utm_grid[@]}" -o "$work/out.tif"
measure gdal-warp "$work/out.tif" gdalwarp -q -t_srs EPSG:32632 -te "$utm_west" "$utm_south" "$utm_east" "$utm_north" -tr 30 30 \
  -r bilinear -ovr NONE -et 0 "$dem" "$work/out.tif"
measure geoplumb-contours "$work/out.geojson" "$geoplumb" run "$work/layers.toml" contours "${own_grid[@]}" -o "$work/out.geojson"
measure gdal-contours "$work/out.geojson" gdal_contour -q -i 100 -a elevation -f GeoJSON "$dem" "$work/out.geojson"
measure geoplumb-cog "$work/out.tif" "$geoplumb" run "$work/layers.toml" dem "${own_grid[@]}" -o "$work/out.tif"
measure gdal-cog "$work/out.tif" gdal_translate -q -of COG "$dem" "$work/out.tif"

python3 -I - "$work" "$runs" <<'EOF'
import re
import statistics
import sys
from pathlib import Path

work = Path(sys.argv[1])
runs = int(sys.argv[2])
operations = [
    ("hillshade", "gdaldem hillshade -az 315 -alt 45"),
    ("warp", "gdalwarp -r bilinear -et 0, utm 32n at 30 m"),
    ("contours", "gdal_contour -i 100"),
    ("cog", "gdal_translate -of COG"),
]


def seconds(text):
    # gnu time prints elapsed wall clock as h:mm:ss or m:ss
    clock = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text).group(1)
    total = 0.0
    for part in clock.split(":"):
        total = total * 60 + float(part)
    return total


def peak_megabytes(text):
    return int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1)) / 1024


def medians(label):
    reports = [(work / f"{label}.{run}.time").read_text() for run in range(1, runs + 1)]
    return statistics.median(map(seconds, reports)), statistics.median(map(peak_megabytes, reports))


print(f"median of {runs} runs over Copernicus GLO-30 N46 E007 (3600 x 3600 float32)")
print()
print("| operation | GDAL command | geoplumb wall s | GDAL wall s | geoplumb peak MB | GDAL peak MB |")
print("| --- | --- | ---: | ---: | ---: | ---: |")
for operation, command in operations:
    ours_seconds, ours_megabytes = medians(f"geoplumb-{operation}")
    theirs_seconds, theirs_megabytes = medians(f"gdal-{operation}")
    print(f"| {operation} | `{command}` | {ours_seconds:.2f} | {theirs_seconds:.2f} | {ours_megabytes:.0f} | {theirs_megabytes:.0f} |")
print()
print("geoplumb run writes every raster as a deflate COG with overviews. gdaldem and gdalwarp write a GeoTIFF without "
      "overviews, gdal_translate -of COG writes an LZW COG with overviews.")
EOF
