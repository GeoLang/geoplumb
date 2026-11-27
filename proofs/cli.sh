#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/cli.sh
# the geoplumb command on a gdal-written lzw cog of copernicus glo-30 n46 e007:
# info against gdalinfo (size, bands, compression, overviews), and run reprojecting
# a 20 km window to utm 32n at 30 m against gdalwarp -r bilinear -et 0 on the same grid,
# passing when the grids match and every pixel is within 1 cm. gdalwarp's default
# -et 0.125 approximates the transform and puts 1.2% of pixels over 2 m off on steep slopes
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem"

gdal_translate -q -of COG -co COMPRESS=LZW "$dem" "$work/dem.tif" || exit 2

"$geoplumb" info "$work/dem.tif" >"$work/info.txt" || { cat "$work/info.txt"; echo "FAIL: geoplumb info"; exit 1; }
cat "$work/info.txt"
gdalinfo -json "$work/dem.tif" >"$work/gdalinfo.json" || exit 2

python3 -I - "$work/info.txt" "$work/gdalinfo.json" <<'EOF' || exit 1
import json, re, sys
text = open(sys.argv[1]).read()
info = json.load(open(sys.argv[2]))
band = info["bands"][0]
expected = {
    "size": f"{info['size'][0]} x {info['size'][1]}",
    "bands": str(len(info["bands"])),
    "compression": "lzw",
}
failures = []
for key, value in expected.items():
    found = re.search(rf"^{key}: (.*)$", text, re.M)
    if not found or value.lower() not in found.group(1).lower():
        failures.append(f"{key}: geoplumb {found.group(1) if found else None!r}, gdal {value!r}")
ours = re.findall(r"^- (\d+) x (\d+)$", text, re.M)
theirs = [(str(o["size"][0]), str(o["size"][1])) for o in band.get("overviews", [])]
if ours != theirs:
    failures.append(f"overviews: geoplumb {ours}, gdal {theirs}")
print(f"RESULT info: size, bands, compression and {len(theirs)} overviews against gdalinfo {'PASS' if not failures else 'FAIL'}")
for failure in failures:
    print("FAIL:", failure)
sys.exit(1 if failures else 0)
EOF

cat >"$work/layers.toml" <<EOF
[[layer]]
name = "dem"
source = { kind = "cog", path = "$work/dem.tif" }
EOF

bbox="380000,5160000,400000,5180000"
"$geoplumb" run "$work/layers.toml" dem --bbox "$bbox" --crs EPSG:32632 --resolution 30 -o "$work/geoplumb.tif" \
  || { echo "FAIL: geoplumb run"; exit 1; }
gdalwarp -q -t_srs EPSG:32632 -te 380000 5160000 400000 5180000 -tr 30 30 -r bilinear -ovr NONE -et 0 \
  "$work/dem.tif" "$work/gdal.tif" || exit 2

python3 -I - "$work/geoplumb.tif" "$work/gdal.tif" <<'EOF' || exit 1
import sys
import numpy
from osgeo import gdal
gdal.UseExceptions()
ours, theirs = gdal.Open(sys.argv[1]), gdal.Open(sys.argv[2])
failures = []
summary = "grids differ"
if (ours.RasterXSize, ours.RasterYSize) != (theirs.RasterXSize, theirs.RasterYSize):
    failures.append(f"size {ours.RasterXSize}x{ours.RasterYSize} against {theirs.RasterXSize}x{theirs.RasterYSize}")
if any(abs(a - b) > 1e-6 for a, b in zip(ours.GetGeoTransform(), theirs.GetGeoTransform())):
    failures.append(f"geotransform {ours.GetGeoTransform()} against {theirs.GetGeoTransform()}")
if not failures:
    a = ours.GetRasterBand(1).ReadAsArray().astype(float)
    b = theirs.GetRasterBand(1).ReadAsArray().astype(float)
    valid = numpy.isfinite(a) & numpy.isfinite(b)
    diff = numpy.abs(a - b)[valid]
    within = float((diff <= 0.01).mean())
    summary = f"{valid.sum()} pixels, median {numpy.median(diff):.5f} m, max {diff.max():.5f} m, within 1 cm {within:.5f}"
    if valid.mean() < 0.99:
        failures.append(f"only {valid.mean():.4f} of pixels valid in both")
    if within < 1.0:
        failures.append(f"{within:.5f} within 1 cm, not every pixel")
print(f"RESULT run: against gdalwarp -r bilinear -et 0, {summary} {'PASS' if not failures else 'FAIL'}")
for failure in failures:
    print("FAIL:", failure)
sys.exit(1 if failures else 0)
EOF

echo PROOF PASS
