#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/raster-vector.sh
# writes geoplumb contours over a 900 by 900 pixel window of a copernicus glo-30
# tile with geoplumb run, pulled through 256 px chunks, and compares them with gdal_contour
# over the same pixels: total length per 100 m level within 1%, and no line ends inside the
# window that gdal's lines do not end at too, so chunk seams leave no breaks.
# gdal_contour runs its lines on to the raster edge, so both sides are cut to
# the bbox of the outermost pixel centers before measuring
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
# bernese alps, lon 7.70 to 7.95 and lat 46.45 to 46.70
column=2520
row=1080
side=900
interval=100
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$dem"

gdal_translate -q -srcwin $column $row $side $side "$dem" "$work/window.tif" || exit 2
read -r west south east north pixel _ < <(file_grid "$work/window.tif")
[ -n "${pixel:-}" ] || { echo "INCONCLUSIVE: no square-pixel grid for the window"; exit 1; }
cat >"$work/layers.toml" <<EOF
[[layer]]
name = "contours"
source = { kind = "cog", path = "$dem" }

[[layer.op]]
kind = "contours"
interval = $interval.0
EOF
"$geoplumb" run "$work/layers.toml" contours --bbox "$west,$south,$east,$north" --crs EPSG:4326 --resolution "$pixel" \
  -o "$work/geoplumb.geojson" || { echo "geoplumb contours failed"; exit 2; }
gdal_contour -q -i $interval -a elevation -f GeoJSON "$work/window.tif" "$work/gdal.geojson" || exit 2
gdalinfo -json "$work/window.tif" >"$work/gdal-window.json" || exit 2

python3 -I - "$work" <<'EOF'
import json, math, sys
from collections import defaultdict

work = sys.argv[1]
info = json.load(open(f"{work}/gdal-window.json"))
x0, dx, _, y0, _, dy = info["geoTransform"]
width, height = info["size"]
resolution = dx
# the bbox of the outermost pixel centers, where both tools' lines end
gdal_window = {
    "min_x": x0 + 0.5 * dx,
    "max_x": x0 + (width - 0.5) * dx,
    "max_y": y0 + 0.5 * dy,
    "min_y": y0 + (height - 0.5) * dy,
}
edge_tolerance = resolution * 1e-3
closed_tolerance = resolution * 1e-6


def clip_segment(a, b, window):
    (x0, y0), (x1, y1) = a[:2], b[:2]
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in [
        (-dx, x0 - window["min_x"]),
        (dx, window["max_x"] - x0),
        (-dy, y0 - window["min_y"]),
        (dy, window["max_y"] - y0),
    ]:
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    if t0 > t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def clip(coords, window):
    parts = []
    for a, b in zip(coords, coords[1:]):
        cut = clip_segment(a, b, window)
        if cut is None or cut[0] == cut[1]:
            continue
        if parts and parts[-1][-1] == cut[0]:
            parts[-1].append(cut[1])
        else:
            parts.append([cut[0], cut[1]])
    # a ring that leaves the window continues from its last piece into its first across its start vertex
    if len(parts) > 1 and coords[0] == coords[-1] and parts[0][0] == parts[-1][-1]:
        parts[0] = parts.pop() + parts[0][1:]
    return parts


def lines(path, window):
    for feature in json.load(open(path))["features"]:
        geometry = feature["geometry"]
        level = round(feature["properties"]["elevation"])
        parts = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        for part in parts:
            for piece in clip(part, window):
                yield level, piece


def on_edge(point, window):
    x, y = point[0], point[1]
    return min(
        abs(x - window["min_x"]),
        abs(x - window["max_x"]),
        abs(y - window["min_y"]),
        abs(y - window["max_y"]),
    ) < edge_tolerance


def summary(path, window):
    length = defaultdict(float)
    open_lines = closed_lines = inner_ends = 0
    for level, coords in lines(path, window):
        length[level] += sum(math.dist(a[:2], b[:2]) for a, b in zip(coords, coords[1:]))
        if math.dist(coords[0][:2], coords[-1][:2]) < closed_tolerance and len(coords) > 2:
            closed_lines += 1
            continue
        open_lines += 1
        inner_ends += sum(not on_edge(end, window) for end in (coords[0], coords[-1]))
    return length, open_lines, closed_lines, inner_ends


ours = summary(f"{work}/geoplumb.geojson", gdal_window)
gdal = summary(f"{work}/gdal.geojson", gdal_window)
print(f"window {width}x{height} px")
failed = False
# a level shorter than this many cells in both is a summit sliver, printed but not judged
floor_cells = 50
judged_levels = 0
largest_difference = 0.0
for level in sorted(set(ours[0]) | set(gdal[0])):
    a, b = ours[0].get(level, 0.0) / resolution, gdal[0].get(level, 0.0) / resolution
    difference = abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0
    judged = max(a, b) >= floor_cells
    bad = judged and difference > 0.01
    failed |= bad
    if judged:
        judged_levels += 1
        largest_difference = max(largest_difference, difference)
    print(f"level {level:5d} m: geoplumb {a:10.1f} cells, gdal {b:10.1f} cells, {100 * difference:6.3f}%{'  FAIL' if bad else ''}{'' if judged else '  (below floor)'}")
for label, (_, open_lines, closed_lines, inner_ends) in [("geoplumb", ours), ("gdal", gdal)]:
    print(f"{label}: {open_lines} open lines, {closed_lines} closed, {inner_ends} line ends inside the window")
if ours[3] != gdal[3]:
    print(f"FAIL: geoplumb has {ours[3]} line ends inside the window against gdal's {gdal[3]}, a seam broke a line")
    failed = True
print(f"RESULT copernicus glo-30 900 x 900 window, 100 m contours: {judged_levels} levels at most {100 * largest_difference:.3f}% "
      f"length difference to gdal_contour, geoplumb {ours[1]} open and {ours[2]} closed lines, gdal {gdal[1]} open and "
      f"{gdal[2]} closed, {ours[3]} line ends inside {'FAIL' if failed else 'PASS'}")
sys.exit(3 if failed else 0)
EOF
status=$?
if [ $status -eq 0 ]; then echo "PROOF PASS"; exit 0; fi
echo "PROOF FAIL, check exited $status"
exit 1
