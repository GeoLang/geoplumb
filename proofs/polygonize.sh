#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/polygonize.sh
# writes geoplumb polygonize over a 1024 by 1024 pixel window of esa worldcover
# with geoplumb run, pulled through 256 px chunks and dissolved per class, and compares it
# with gdal's polygonize over the same pixels, both 4-connected: per class the number of
# polygons and holes must match and the symmetric difference must have no
# area, so regions cut by chunk seams come back whole
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
# lon 7.500 to 7.585 and lat 46.415 to 46.500
column=18000
row=18000
side=1024
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$worldcover"

gdal_translate -q -srcwin $column $row $side $side "$worldcover" "$work/window.tif" || exit 2
read -r west south east north pixel _ < <(file_grid "$work/window.tif")
[ -n "${pixel:-}" ] || { echo "INCONCLUSIVE: no square-pixel grid for the window"; exit 1; }
cat >"$work/layers.toml" <<EOF
[[layer]]
name = "polygons"
source = { kind = "cog", path = "$worldcover", value_kind = "categorical" }

[[layer.op]]
kind = "polygonize"
EOF
"$geoplumb" run "$work/layers.toml" polygons --bbox "$west,$south,$east,$north" --crs EPSG:4326 --resolution "$pixel" \
  -o "$work/geoplumb.geojson" || { echo "geoplumb polygonize failed"; exit 2; }

python3 -I - "$work" "$pixel" <<'EOF'
import json
import sys
from collections import defaultdict

from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
work = sys.argv[1]
cell_area = float(sys.argv[2]) ** 2

dataset = gdal.Open(f"{work}/window.tif")
band = dataset.GetRasterBand(1)
memory = ogr.GetDriverByName("Memory").CreateDataSource("gdal")
spatial_reference = osr.SpatialReference()
spatial_reference.ImportFromEPSG(4326)
layer = memory.CreateLayer("gdal", spatial_reference, ogr.wkbPolygon)
layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
gdal.Polygonize(band, band.GetMaskBand(), layer, 0, [])


def empty():
    return {"polygons": 0, "holes": 0, "invalid": 0, "geometry": ogr.Geometry(ogr.wkbMultiPolygon)}


gdal_classes = defaultdict(empty)
for feature in layer:
    polygon = feature.GetGeometryRef()
    entry = gdal_classes[feature.GetField("value")]
    entry["polygons"] += 1
    entry["holes"] += polygon.GetGeometryCount() - 1
    entry["invalid"] += not polygon.IsValid()
    entry["geometry"].AddGeometry(polygon)

ours_classes = defaultdict(empty)
for feature in json.load(open(f"{work}/geoplumb.geojson"))["features"]:
    geometry = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
    polygons = [geometry] if geometry.GetGeometryType() == ogr.wkbPolygon else [geometry.GetGeometryRef(i) for i in range(geometry.GetGeometryCount())]
    entry = ours_classes[feature["properties"]["value"]]
    for polygon in polygons:
        entry["polygons"] += 1
        entry["holes"] += polygon.GetGeometryCount() - 1
        entry["invalid"] += not polygon.IsValid()
        entry["geometry"].AddGeometry(polygon)

print(f"window {dataset.RasterXSize}x{dataset.RasterYSize} px")
failed = False
for value in sorted(set(gdal_classes) | set(ours_classes)):
    ours, theirs = ours_classes[value], gdal_classes[value]
    ours_area = ours["geometry"].GetArea() / cell_area
    theirs_area = theirs["geometry"].GetArea() / cell_area
    difference = ours["geometry"].MakeValid().SymDifference(theirs["geometry"].MakeValid()).GetArea() / cell_area
    problems = []
    if ours["polygons"] != theirs["polygons"]:
        problems.append("polygon count")
    if ours["holes"] != theirs["holes"]:
        problems.append("hole count")
    if ours["invalid"]:
        problems.append("invalid polygons")
    # f64 noise at degree coordinates measured about 1e-6 cells per class, and seam drift from i32 snapping reached 0.03
    if difference > 1e-4:
        problems.append("geometry")
    failed |= bool(problems)
    print(
        f"RESULT worldcover 1024 x 1024 window class {value:3d}: geoplumb {ours['polygons']:6d} polygons {ours['holes']:5d} holes {ours['invalid']:4d} invalid {ours_area:9.1f} cells, "
        f"gdal {theirs['polygons']:6d} polygons {theirs['holes']:5d} holes {theirs['invalid']:4d} invalid {theirs_area:9.1f} cells, "
        f"symmetric difference {difference:.6f} cells"
        + (f" FAIL {', '.join(problems)}" if problems else " PASS")
    )
sys.exit(3 if failed else 0)
EOF
status=$?
if [ $status -eq 0 ]; then echo "PROOF PASS"; exit 0; fi
echo "PROOF FAIL, check exited $status"
exit 1
