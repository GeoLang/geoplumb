#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/viewshed.sh
# copernicus glo-30 warped to utm 32n at 30 m around an observer at 7.65 e 46.645 n,
# a viewshed layer written by geoplumb run over the 10 km disc, then checked: the four quadrant exports put together equal the whole disc
# export bit for bit, and at least 95% of the cells inside the disc agree with
# gdal_viewshed run with no curvature correction, cells within 1.5 cells of the
# radius left out since the two tools round the disc edge differently
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem"
radius=10000
height=2

# the utm 32n pixel center nearest 7.65 e 46.645 n on a 30 m grid, and its lon and lat
read -r x y lon lat < <(python3 -I -c '
from osgeo import osr
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
utm = osr.SpatialReference()
utm.ImportFromEPSG(32632)
x, y, _ = osr.CoordinateTransformation(wgs84, utm).TransformPoint(7.65, 46.645)
x, y = (x // 30) * 30 + 15, (y // 30) * 30 + 15
lon, lat, _ = osr.CoordinateTransformation(utm, wgs84).TransformPoint(x, y)
print(x, y, repr(lon), repr(lat))
')
if [ -z "${lat:-}" ]; then echo "FAIL: no observer"; exit 1; fi
echo "observer at $x $y in EPSG:32632 ($lon, $lat)"

half=$((radius + 5000))
gdalwarp -q -t_srs EPSG:32632 -tr 30 30 -te $((${x%.*} - 15 - half)) $((${y%.*} - 15 - half)) $((${x%.*} - 15 + half)) $((${y%.*} - 15 + half)) \
  -r bilinear -of COG -co COMPRESS=DEFLATE "$dem" "$work/dem.tif" || { echo "FAIL: gdalwarp"; exit 1; }

cat >"$work/layers.toml" <<EOF
[[layer]]
name = "viewshed"
source = { kind = "cog", path = "$work/dem.tif" }

[[layer.op]]
kind = "viewshed"
lon = $lon
lat = $lat
height = $height.0
radius = $radius.0
EOF

# the disc's bbox on the 30 m grid, then its four quadrants split at the observer's pixel edge
# the radius rounded up to whole 30 m cells on each side of the observer's cell
reach=$(((radius + 29) / 30 * 30))
min_x=$((${x%.*} - 15 - reach))
min_y=$((${y%.*} - 15 - reach))
max_x=$((${x%.*} + 15 + reach))
max_y=$((${y%.*} + 15 + reach))
mid_x=$((${x%.*} - 15))
mid_y=$((${y%.*} - 15))
export_bbox() {
  local name=$1 bbox=$2
  "$geoplumb" run "$work/layers.toml" viewshed --bbox "$bbox" --crs EPSG:32632 --resolution 30 -o "$work/$name.tif" \
    || { echo "FAIL: $name export"; exit 1; }
}
export_bbox whole "$min_x,$min_y,$max_x,$max_y"
export_bbox south-west "$min_x,$min_y,$mid_x,$mid_y"
export_bbox south-east "$mid_x,$min_y,$max_x,$mid_y"
export_bbox north-west "$min_x,$mid_y,$mid_x,$max_y"
export_bbox north-east "$mid_x,$mid_y,$max_x,$max_y"

gdal_viewshed -q -ox "$x" -oy "$y" -oz $height -tz 0 -md $radius -cc 0 -vv 1 -iv 0 -ov 2 "$work/dem.tif" "$work/gdal-full.tif" || { echo "FAIL: gdal_viewshed"; exit 1; }
gdal_translate -q -projwin "$min_x" "$max_y" "$max_x" "$min_y" "$work/gdal-full.tif" "$work/gdal.tif" || { echo "FAIL: gdal_translate"; exit 1; }

python3 -I - "$work" "$x" "$y" "$min_x" "$max_y" $radius "$mid_x" "$mid_y" <<'EOF'
import sys

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = sys.argv[1]
x, y, min_x, max_y, radius, mid_x, mid_y = map(float, sys.argv[2:9])
failures = []


def base(path):
    # gdal's python bindings free a dataset nothing names, and its bands with it
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None and not numpy.isnan(nodata):
        values[values == nodata] = numpy.nan
    return values, dataset.GetGeoTransform()


whole, transform = base(f"{work}/whole.tif")
print("whole export", whole.shape, "geotransform", transform)
if transform[0] != min_x or transform[3] != max_y or transform[1] != 30.0:
    failures.append(f"whole export geotransform {transform} is not origin {min_x},{max_y} at 30 m")

rows, columns = whole.shape
half_rows, half_columns = round((max_y - mid_y) / 30.0), round((mid_x - min_x) / 30.0)
pieces = {
    "north-west": (slice(0, half_rows), slice(0, half_columns)),
    "north-east": (slice(0, half_rows), slice(half_columns, columns)),
    "south-west": (slice(half_rows, rows), slice(0, half_columns)),
    "south-east": (slice(half_rows, rows), slice(half_columns, columns)),
}
for name, (row_range, column_range) in pieces.items():
    piece, _ = base(f"{work}/{name}.tif")
    expected = whole[row_range, column_range]
    same = (piece == expected) | (numpy.isnan(piece) & numpy.isnan(expected))
    if piece.shape != expected.shape or not same.all():
        failures.append(f"{name} quadrant differs from the whole export in {int((~same).sum()) if piece.shape == expected.shape else 'shape'} cells")
    else:
        print(f"{name} quadrant equals the whole export bit for bit")

theirs, gdal_transform = base(f"{work}/gdal.tif")
if theirs.shape != whole.shape:
    failures.append(f"gdal window {theirs.shape} is not the export's {whole.shape}")
else:
    centers_x = transform[0] + (numpy.arange(columns) + 0.5) * 30.0
    centers_y = transform[3] - (numpy.arange(rows) + 0.5) * 30.0
    distance = numpy.hypot(centers_x[None, :] - x, centers_y[:, None] - y)
    judged = (distance < radius - 45.0) & numpy.isfinite(whole) & (theirs != 2)
    agree = int(((whole == theirs) & judged).sum())
    total = int(judged.sum())
    share = agree / total if total else 0.0
    visible_ours = int(((whole == 1) & judged).sum())
    visible_theirs = int(((theirs == 1) & judged).sum())
    print(f"inside the disc: {agree} of {total} cells agree with gdal_viewshed, {100 * share:.3f}%")
    print(f"visible cells: geoplumb {visible_ours}, gdal {visible_theirs}")
    print(f"geoplumb visible where gdal hidden: {int(((whole == 1) & (theirs == 0) & judged).sum())}, the reverse: {int(((whole == 0) & (theirs == 1) & judged).sum())}")
    if share < 0.95:
        failures.append(f"{100 * share:.3f}% agreement with gdal_viewshed is under 95%")
    if not 0.05 < visible_ours / total < 0.95:
        failures.append("the visible share is not between 5% and 95%, so agreement says little")
    quadrants = "differ from" if any("quadrant" in failure for failure in failures) else "equal"
    print(f"RESULT copernicus glo-30 in EPSG:32632 at 30 m, 10 km disc: quadrant exports {quadrants} the whole export, "
          f"{agree} of {total} in-disc cells ({100 * share:.3f}%) agree with gdal_viewshed -cc 0, "
          f"{100 * visible_ours / total if total else 0.0:.1f}% visible {'PASS' if not failures else 'FAIL'}")

for failure in failures:
    print("FAIL:", failure)
sys.exit(1 if failures else 0)
EOF
status=$?
if [ $status -eq 0 ]; then echo "PROOF PASS"; exit 0; fi
echo "PROOF FAIL"
exit 1
