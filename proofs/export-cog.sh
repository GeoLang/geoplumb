#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/export-cog.sh
# exports esa worldcover around 7.5 e 46.5 n, 20 km square, in utm 32n at 20 m
# through geoplumb run, which writes the cog GET /export serves, then checks with gdal:
# a valid cog under --full-check, the requested crs, origin and pixel size, overviews,
# only class codes or nan at every level, and the base level against gdalwarp -r near
# on at least 99% of pixels. a continuous layer over the same file is the
# control that has to blend codes, so the class check is shown able to fail
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$worldcover"

cat >"$work/layers.toml" <<EOF
[[layer]]
name = "worldcover"
gray = { min = 0.0, max = 100.0 }
source = { kind = "cog", path = "$worldcover", value_kind = "categorical" }

[[layer]]
name = "worldcover-continuous"
gray = { min = 0.0, max = 100.0 }
source = { kind = "cog", path = "$worldcover" }
EOF

# the utm 32n point under 7.5 e 46.5 n, snapped to the 20 m grid, then 10 km either way
read -r min_x min_y max_x max_y < <(python3 -I -c '
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
if [ -z "${max_y:-}" ]; then echo "FAIL: no utm window"; exit 1; fi
echo "window in EPSG:32632: $min_x $min_y $max_x $max_y"

for layer in worldcover worldcover-continuous; do
  "$geoplumb" run "$work/layers.toml" "$layer" --bbox "$min_x,$min_y,$max_x,$max_y" --crs EPSG:32632 --resolution 20 \
    -o "$work/$layer.tif" || { echo "FAIL: $layer export"; exit 1; }
done

python3 -I -m osgeo_utils.samples.validate_cloud_optimized_geotiff --full-check=yes "$work/worldcover.tif" \
  || { echo "FAIL: not a valid cloud optimized geotiff"; exit 1; }
gdalinfo -json "$work/worldcover.tif" >"$work/info.json" || { echo "FAIL: gdalinfo"; exit 1; }
gdalwarp -q -t_srs EPSG:32632 -te "$min_x" "$min_y" "$max_x" "$max_y" -tr 20 20 -r near \
  "$worldcover" "$work/gdalwarp.tif" || { echo "FAIL: gdalwarp"; exit 1; }
gdalwarp -q -ovr NONE -t_srs EPSG:32632 -te "$min_x" "$min_y" "$max_x" "$max_y" -tr 20 20 -r near \
  "$worldcover" "$work/gdalwarp-full-resolution.tif" || { echo "FAIL: gdalwarp -ovr NONE"; exit 1; }

python3 -I - "$work" "$min_x" "$min_y" "$max_x" "$max_y" <<'EOF'
import json
import sys

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = sys.argv[1]
min_x, min_y, max_x, max_y = map(float, sys.argv[2:6])
classes = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=float)
failures = []

with open(f"{work}/info.json") as file:
    info = json.load(file)
geotransform = info["geoTransform"]
overview_sizes = [overview["size"] for overview in info["bands"][0].get("overviews", [])]
print("gdalinfo size", info["size"], "geotransform", geotransform, "overviews", overview_sizes)
if 'ID["EPSG",32632]' not in info["coordinateSystem"]["wkt"]:
    failures.append("gdalinfo does not show EPSG:32632")
if geotransform != [min_x, 20.0, 0.0, max_y, 0.0, -20.0]:
    failures.append(f"geotransform {geotransform} is not origin {min_x},{max_y} at 20 m")
if info["size"] != [1000, 1000]:
    failures.append(f"size {info['size']} is not 1000 x 1000")
if not overview_sizes:
    failures.append("gdalinfo shows no overviews")


def levels(path):
    # gdal's python bindings free a dataset nothing names, and its bands with it
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    arrays = [band.ReadAsArray().astype(float)]
    for index in range(band.GetOverviewCount()):
        arrays.append(band.GetOverview(index).ReadAsArray().astype(float))
    # geoplumb exports a byte source as byte with the source's nodata, 0 for worldcover
    nodata = band.GetNoDataValue()
    if nodata is not None:
        for values in arrays:
            values[values == nodata] = numpy.nan
    return arrays


def off_class(values):
    finite = values[numpy.isfinite(values)]
    return int((~numpy.isin(finite, classes)).sum()), int(finite.size)


categorical = levels(f"{work}/worldcover.tif")
categorical_blended = 0
for level, values in enumerate(categorical):
    blended, valid = off_class(values)
    categorical_blended += blended
    print(f"categorical level {level}: {values.shape[1]} x {values.shape[0]}, {valid} valid pixels, {blended} not a class code")
    if blended:
        failures.append(f"categorical level {level} holds {blended} values that are not class codes")
    if not valid:
        failures.append(f"categorical level {level} holds no valid pixel")

control_blended = 0
for level, values in enumerate(levels(f"{work}/worldcover-continuous.tif")):
    blended, valid = off_class(values)
    control_blended += blended
    print(f"continuous control level {level}: {valid} valid pixels, {blended} not a class code")
if not control_blended:
    failures.append("the continuous control blends no codes, so the class check shows nothing")


def agreement(reference_path):
    dataset = gdal.Open(reference_path)
    band = dataset.GetRasterBand(1)
    reference = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        reference[reference == nodata] = numpy.nan
    base = categorical[0]
    either = numpy.isfinite(base) | numpy.isfinite(reference)
    same = (base == reference) | (numpy.isnan(base) & numpy.isnan(reference))
    return int((same & either).sum()), int(either.sum())


agree, total = agreement(f"{work}/gdalwarp.tif")
share = agree / total if total else 0.0
print(f"base level against gdalwarp -r near: {agree} of {total} pixels agree, {100 * share:.3f}%")
if share < 0.99:
    failures.append(f"base level agrees with gdalwarp on {100 * share:.3f}% of pixels, under 99%")
gated_share = share
agree, total = agreement(f"{work}/gdalwarp-full-resolution.tif")
share = agree / total if total else 0.0
print(f"base level against gdalwarp -r near -ovr NONE, not gated: {agree} of {total} pixels agree, {100 * share:.3f}%")

print(f"RESULT worldcover in EPSG:32632 at 20 m: valid cog, {categorical_blended} non-class values at {len(categorical)} levels "
      f"(continuous control {control_blended}), {100 * gated_share:.3f}% agree with gdalwarp -r near {'PASS' if not failures else 'FAIL'}")
for failure in failures:
    print("FAIL:", failure)
sys.exit(1 if failures else 0)
EOF
status=$?
if [ $status -eq 0 ]; then echo "PROOF PASS"; exit 0; fi
echo "PROOF FAIL"
exit 1
