#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/cutline.sh
# clips with a clip_raster layer and with gdalwarp -cutline
# on the raster's own grid: copernicus glo-30 n46 e007 and a 2400 x 2401 esa worldcover window, each by a 40 vertex
# star with a 25 vertex hole built over its extent, and the copernicus tile warped to utm 32n at 30 m by the copernicus
# star in wgs84, which both tools reproject. natural earth 1:110m switzerland contains the whole copernicus tile, so it
# shows no clip.
# thresholds, fixed before the first run:
# on every input gdal keeps between 10% and 90% of the cells, so the clip shows;
# on the two inputs whose cutline is in the raster's crs, geoplumb run equals gdalwarp's output in every cell;
# on the utm input, every cell both keep is equal, at most 0.01% of cells are kept by one tool only, and the centre of
# every such cell lies within one pixel of the cutline's boundary
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem" "$worldcover"

gdal_translate -q -srcwin 18000 18000 2400 2401 -of COG -co COMPRESS=LZW "$worldcover" "$work/worldcover.tif" \
  || { echo "FAIL: gdal_translate"; exit 1; }
gdalwarp -q -t_srs EPSG:32632 -tr 30 30 -tap -r bilinear -of COG -co COMPRESS=LZW "$dem" "$work/dem_utm.tif" \
  || { echo "FAIL: gdalwarp to utm"; exit 1; }
cp "$dem" "$work/dem.tif"

for star in worldcover dem; do
python3 -I - "$work/$star.tif" "$work/${star}_star.geojson" <<'EOF'
import json
import math
import sys

from osgeo import gdal

dataset = gdal.Open(sys.argv[1])
origin_x, pixel_x, _, origin_y, _, pixel_y = dataset.GetGeoTransform()
width = dataset.RasterXSize * pixel_x
height = -dataset.RasterYSize * pixel_y
centre_x = origin_x + width / 2
centre_y = origin_y - height / 2


def ring(count, radius, wobble, turn):
    points = []
    for index in range(count):
        angle = turn + 2 * math.pi * index / count
        scale = radius * (1 + wobble * math.sin(7 * angle) * (1 if index % 2 else 0.4))
        points.append([centre_x + scale * width * math.cos(angle), centre_y + scale * height * math.sin(angle)])
    points.append(points[0])
    return points


outer = ring(40, 0.42, 0.18, 0.1)
hole = list(reversed(ring(25, 0.15, 0.25, 0.7)))
collection = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [outer, hole]}}]}
open(sys.argv[2], "w").write(json.dumps(collection))
EOF
[ -s "$work/${star}_star.geojson" ] || { echo "FAIL: no $star star cutline"; exit 1; }
done

names="dem worldcover dem_utm"
for name in $names; do
  file=$work/$name.tif
  cutline=$work/dem_star.geojson
  [ "$name" = worldcover ] && cutline=$work/worldcover_star.geojson
  read -r west south east north pixel epsg < <(file_grid "$file")
  [ -n "${epsg:-}" ] || { echo "INCONCLUSIVE: no square-pixel grid or epsg for $name"; exit 1; }
  value_kind=""
  [ "$name" = worldcover ] && value_kind=', value_kind = "categorical"'
  cat >"$work/$name.toml" <<TOML
[[layer]]
name = "cutline"
source = { kind = "geojson", path = "$cutline" }

[[layer]]
name = "clipped"
source = { kind = "cog", path = "$file"$value_kind }

[[layer.op]]
kind = "clip_raster"
cutline = "cutline"
TOML
  run_args=("$work/$name.toml" clipped --bbox "$west,$south,$east,$north" --crs "EPSG:$epsg" --resolution "$pixel")
  "$geoplumb" run "${run_args[@]}" -o "$work/$name.run.tif" >"$work/$name.run.log" 2>&1 \
    || { cat "$work/$name.run.log"; echo "FAIL: geoplumb run on $name"; exit 1; }
  nodata=nan
  [ "$name" = worldcover ] && nodata=0
  gdalwarp -q -cutline "$cutline" -te "$west" "$south" "$east" "$north" -tr "$pixel" "$pixel" -r near \
    -dstnodata "$nodata" "$file" "$work/$name.gdal.tif" || { echo "FAIL: gdalwarp -cutline on $name"; exit 1; }
done

python3 -I - "$work" <<'EOF'
import json
import sys
from pathlib import Path

import numpy
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
work = Path(sys.argv[1])
failures = []
inconclusive = []


def with_nan(dataset):
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray().astype(numpy.float64)
    if band.GetNoDataValue() is not None:
        values[values == band.GetNoDataValue()] = numpy.nan
    return values


def check(name):
    ours_dataset = gdal.Open(str(work / f"{name}.run.tif"))
    theirs_dataset = gdal.Open(str(work / f"{name}.gdal.tif"))
    ours, theirs = with_nan(ours_dataset), with_nan(theirs_dataset)
    if ours.shape != theirs.shape:
        failures.append(f"{name}: geoplumb is {ours.shape}, gdal {theirs.shape}")
        return "grids differ"
    if any(abs(a - b) > 1e-9 for a, b in zip(ours_dataset.GetGeoTransform(), theirs_dataset.GetGeoTransform())):
        failures.append(f"{name}: geotransform {ours_dataset.GetGeoTransform()}, gdal {theirs_dataset.GetGeoTransform()}")
    kept_ours, kept_theirs = ~numpy.isnan(ours), ~numpy.isnan(theirs)
    kept_fraction = kept_theirs.mean()
    both = kept_ours & kept_theirs
    one_only = kept_ours ^ kept_theirs
    unequal = int((ours[both] != theirs[both]).sum())
    summary = (f"{ours.shape[1]} x {ours.shape[0]}, gdal keeps {kept_fraction:.2%}, geoplumb keeps {kept_ours.mean():.2%}, "
               f"{int(one_only.sum())} of {one_only.size} cells kept by one tool only, {unequal} kept cells unequal")
    if not 0.1 <= kept_fraction <= 0.9:
        inconclusive.append(f"{name}: gdal keeps {kept_fraction:.2%} of cells, so the clip does not show")
    if unequal:
        failures.append(f"{name}: {unequal} cells kept by both differ in value")
    if name != "dem_utm":
        if one_only.any():
            failures.append(f"{name}: {int(one_only.sum())} cells kept by one tool only")
        return summary
    if one_only.sum() > 0.0001 * one_only.size:
        failures.append(f"{name}: {int(one_only.sum())} cells kept by one tool only, past 0.01% of {one_only.size}")
        return summary
    origin_x, pixel_x, _, origin_y, _, pixel_y = ours_dataset.GetGeoTransform()
    cutline = ogr.CreateGeometryFromJson(json.dumps(json.load(open(work / "dem_star.geojson"))["features"][0]["geometry"]))
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32632)
    cutline.Transform(osr.CoordinateTransformation(wgs84, utm))
    boundary = cutline.Boundary()
    far = 0
    for row, col in zip(*numpy.nonzero(one_only)):
        centre = ogr.Geometry(ogr.wkbPoint)
        centre.AddPoint_2D(origin_x + (col + 0.5) * pixel_x, origin_y + (row + 0.5) * pixel_y)
        if centre.Distance(boundary) > abs(pixel_x):
            far += 1
    if far:
        failures.append(f"{name}: {far} one-tool cells lie farther than one pixel from the cutline boundary")
    return f"{summary}, {far} of them farther than one pixel from the cutline boundary"


for name in ["dem", "worldcover", "dem_utm"]:
    failures_before = len(failures)
    summary = check(name)
    print(f"RESULT {name}: {summary} {'PASS' if len(failures) == failures_before else 'FAIL'}")


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
