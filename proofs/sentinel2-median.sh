#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> proofs/sentinel2-median.sh
# the job in examples/sentinel2-median.toml two ways over a 9.9 km square in utm 32n at 10 m: one geoplumb run, and the
# closest gdal 3.12 equivalent. gdal has no stac search that keeps items apart: the STACIT driver mosaics the items of
# a search into one raster, most recent on top, with no per-item mask and no median. so the gdal side lists the items
# with the earth search api itself, then per item gdalwarp cuts red, green, blue and scl onto the output grid and
# gdal raster calc keeps a band only where scl is 4, 5 or 6, and per band gdal raster mosaic --output-format GTiff --pixel-function median
# (gdal 3.12) takes the median across items, and gdal raster stack joins the three bands into a cog.
# thresholds, fixed before the first run:
# the search lists at least three items, and at least 10% of the cells of every band are valid in gdal's output;
# geoplumb's output has gdal's grid and three bands;
# in every band at least 99.9% of the cells are valid in both or nodata in both, and at least 99.9% of the cells valid
# in both differ by under 0.001;
# both wall clock times are printed, neither is gated
set -uo pipefail

geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
layers=$(dirname "$0")/../examples/sentinel2-median.toml
layer=sentinel2-median
api=https://earth-search.aws.element84.com/v1
datetime=2025-08-01T00:00:00Z/2025-09-01T00:00:00Z
# multiples of 60 m, so the 10 m bands and the 20 m scl of every utm 32n item lie on this grid
west=410100
south=5170020
east=420000
north=5179920
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# gdal requests a directory listing for every /vsicurl/ file it opens unless this is EMPTY_DIR
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

geoplumb_start=$EPOCHREALTIME
"$geoplumb" run "$layers" "$layer" --bbox "$west,$south,$east,$north" --crs EPSG:32632 --resolution 10 -o "$work/geoplumb.tif" \
  || { echo "FAIL: geoplumb run"; exit 1; }
geoplumb_end=$EPOCHREALTIME

gdal_start=$EPOCHREALTIME
python3 -I - "$api" "$datetime" "$west" "$south" "$east" "$north" >"$work/items.txt" <<'EOF' || { echo "FAIL: stac search"; exit 1; }
import json
import sys
import urllib.request

from osgeo import osr

api, datetime = sys.argv[1:3]
west, south, east, north = map(float, sys.argv[3:7])
wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
utm = osr.SpatialReference()
utm.ImportFromEPSG(32632)
transform = osr.CoordinateTransformation(utm, wgs84)
corners = [transform.TransformPoint(x, y)[:2] for x in (west, east) for y in (south, north)]
lons, lats = [c[0] for c in corners], [c[1] for c in corners]
url = (f"{api}/search?collections=sentinel-2-l2a&bbox={min(lons)},{min(lats)},{max(lons)},{max(lats)}"
       f"&datetime={datetime}&limit=100")
while url:
    request = urllib.request.Request(url, headers={"Accept": "application/geo+json"})
    with urllib.request.urlopen(request) as response:
        page = json.load(response)
    for item in page["features"]:
        # geoplumb's stac source skips an item missing any asset it reads
        if all(asset in item["assets"] for asset in ("red", "green", "blue", "scl")):
            print(item["id"], *(item["assets"][asset]["href"] for asset in ("red", "green", "blue", "scl")))
    url = next((link["href"] for link in page.get("links", []) if link["rel"] == "next" and page["features"]), None)
EOF
item_count=$(wc -l <"$work/items.txt")
echo "earth search lists $item_count items"

while read -r id red green blue scl <&3; do
  for pair in "red $red" "green $green" "blue $blue" "scl $scl"; do
    read -r band href <<<"$pair"
    gdalwarp -q -overwrite -t_srs EPSG:32632 -te $west $south $east $north -tr 10 10 -r near -dstnodata 0 \
      "/vsicurl/$href" "$work/$id.$band.tif" || { echo "FAIL: gdalwarp $id $band"; exit 1; }
  done
  for band in red green blue; do
    gdal raster calc --input "R=$work/$id.$band.tif" --input "S=$work/$id.scl.tif" --propagate-nodata \
      --calc "(S == 4 || S == 5 || S == 6) ? R : -1" --output-data-type Float32 --nodata=-1 --output-format GTiff \
      --output "$work/$id.$band.masked.tif" || { echo "FAIL: gdal raster calc $id $band"; exit 1; }
  done
done 3<"$work/items.txt"

for band in red green blue; do
  gdal raster mosaic --output-format GTiff --pixel-function median --src-nodata=-1 --dst-nodata=-1 "$work"/*."$band".masked.tif \
    "$work/median.$band.tif" || { echo "FAIL: gdal raster mosaic $band"; exit 1; }
done
gdal raster stack --output-format COG "$work/median.red.tif" "$work/median.green.tif" "$work/median.blue.tif" "$work/gdal.tif" \
  || { echo "FAIL: gdal raster stack"; exit 1; }
gdal_end=$EPOCHREALTIME

python3 -I - "$work" "$item_count" "$geoplumb_start" "$geoplumb_end" "$gdal_start" "$gdal_end" <<'EOF'
import sys
from pathlib import Path

import numpy
from osgeo import gdal

gdal.UseExceptions()
work = Path(sys.argv[1])
item_count = int(sys.argv[2])
geoplumb_start, geoplumb_end, gdal_start, gdal_end = map(float, sys.argv[3:7])
failures = []
inconclusive = []
print(f"wall clock: geoplumb run {geoplumb_end - geoplumb_start:.1f} s, gdal {gdal_end - gdal_start:.1f} s over {item_count} items")
if item_count < 3:
    inconclusive.append(f"the search lists {item_count} items, so a median says little")


def with_nan(band):
    values = band.ReadAsArray().astype(numpy.float64)
    if band.GetNoDataValue() is not None:
        values[values == band.GetNoDataValue()] = numpy.nan
    return values


ours, theirs = gdal.Open(str(work / "geoplumb.tif")), gdal.Open(str(work / "gdal.tif"))
if (ours.RasterXSize, ours.RasterYSize, ours.RasterCount) != (theirs.RasterXSize, theirs.RasterYSize, 3):
    failures.append(f"geoplumb is {ours.RasterXSize} x {ours.RasterYSize} x {ours.RasterCount}, "
                    f"gdal {theirs.RasterXSize} x {theirs.RasterYSize} x {theirs.RasterCount}")
elif any(abs(a - b) > 1e-6 for a, b in zip(ours.GetGeoTransform(), theirs.GetGeoTransform())):
    failures.append(f"geotransform {ours.GetGeoTransform()}, gdal {theirs.GetGeoTransform()}")
else:
    for index, name in enumerate(["red", "green", "blue"], start=1):
        failures_before = len(failures)
        a, b = with_nan(ours.GetRasterBand(index)), with_nan(theirs.GetRasterBand(index))
        valid_a, valid_b = ~numpy.isnan(a), ~numpy.isnan(b)
        same_validity = float((valid_a == valid_b).mean())
        both = valid_a & valid_b
        difference = numpy.abs(a[both] - b[both])
        close = float((difference < 0.001).mean()) if difference.size else 0.0
        largest = float(difference.max()) if difference.size else 0.0
        if valid_b.mean() < 0.1:
            inconclusive.append(f"{name}: {valid_b.mean():.1%} of gdal's cells are valid")
        if same_validity < 0.999:
            failures.append(f"{name}: validity agrees on {same_validity:.4%} of cells, under 99.9%")
        if close < 0.999:
            failures.append(f"{name}: {close:.4%} of the cells valid in both differ by under 0.001, under 99.9%")
        print(f"RESULT {name}: {item_count} items, valid in geoplumb {valid_a.mean():.2%}, in gdal {valid_b.mean():.2%}, "
              f"validity agrees on {same_validity:.4%} of cells, {close:.4%} of {int(both.sum())} cells valid in both "
              f"within 0.001, largest difference {largest:g} {'PASS' if len(failures) == failures_before else 'FAIL'}")

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
