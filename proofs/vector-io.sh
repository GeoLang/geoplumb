#!/usr/bin/env bash
# usage: GEOPLUMB_BIN=<geoplumb> GEOPLUMB_PROOF_DATA=<directory> proofs/vector-io.sh
# reads natural earth 1:110m countries as a shapefile
# ogr2ogr wrote in wgs84 and one it reprojected to utm 32n over central europe, and a csv of one point per country,
# and writes each through `geoplumb run -o .gpkg`.
# thresholds, fixed before the first run:
# validate_gpkg.py reports no error on any output, and ogr reads each with the epsg code of --crs;
# each shapefile output holds every source feature, matched by NAME, with equal POP_EST, CONTINENT and ISO_A3, and
# geometry whose symmetric difference with ogr's read of the shapefile is under 1e-9 of its area;
# the csv output holds every row with x and y equal to the csv's, pop_est as an INTEGER column, gdp_md as REAL and
# name as TEXT, each value equal;
# a utm shapefile whose .prj names no epsg code is refused naming crs when the layer file names none
set -uo pipefail

source "$(dirname "$0")/inputs.sh"
geoplumb=${GEOPLUMB_BIN:?set GEOPLUMB_BIN to a geoplumb built with cargo build --release -p geoplumb-server --bin geoplumb}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
require_inputs "$countries"

fields="NAME,POP_EST,CONTINENT,ISO_A3"
ogr2ogr -q -select "$fields" "$work/countries.shp" "$countries" || { echo "FAIL: ogr2ogr wgs84 shapefile"; exit 1; }
ogr2ogr -q -select "$fields" -clipsrc 2 42 16 56 -t_srs EPSG:32632 "$work/europe_utm.shp" "$countries" \
  || { echo "FAIL: ogr2ogr utm shapefile"; exit 1; }
python3 -I - "$countries" "$work/points.csv" <<'EOF'
import csv
import json
import sys

from osgeo import ogr

collection = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["name", "x", "y", "pop_est", "gdp_md"])
    for feature in collection["features"]:
        point = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"])).PointOnSurface()
        properties = feature["properties"]
        writer.writerow([properties["NAME"], repr(point.GetX()), repr(point.GetY()), int(properties["POP_EST"]),
                         repr(float(properties["GDP_MD"]) + 0.25)])
EOF
[ -s "$work/points.csv" ] || { echo "FAIL: no csv written"; exit 1; }

cat >"$work/layers.toml" <<TOML
[[layer]]
name = "countries"
source = { kind = "shapefile", path = "$work/countries.shp", crs = 4326 }

[[layer]]
name = "europe_utm"
source = { kind = "shapefile", path = "$work/europe_utm.shp", crs = 32632 }

[[layer]]
name = "europe_utm_no_crs"
source = { kind = "shapefile", path = "$work/europe_utm.shp" }

[[layer]]
name = "points"
source = { kind = "csv", path = "$work/points.csv", x = "x", y = "y", crs = 4326 }
TOML
cat >"$work/geojson.toml" <<TOML
[[layer]]
name = "countries"
source = { kind = "geojson", path = "$countries" }
TOML

world=(--bbox=-180,-90,180,90 --crs EPSG:4326 --resolution 0.00001)
read -r utm_bbox < <(python3 -I -c '
import sys
from osgeo import ogr
ogr.UseExceptions()
dataset = ogr.Open(sys.argv[1])
west, east, south, north = dataset.GetLayer().GetExtent()
print(f"{west - 1000!r},{south - 1000!r},{east + 1000!r},{north + 1000!r}")
' "$work/europe_utm.shp")
utm=("--bbox=$utm_bbox" --crs EPSG:32632 --resolution 1)

"$geoplumb" run "$work/layers.toml" countries "${world[@]}" -o "$work/countries.gpkg" >"$work/countries.log" 2>&1 \
  || { cat "$work/countries.log"; echo "FAIL: run countries"; exit 1; }
"$geoplumb" run "$work/layers.toml" europe_utm "${utm[@]}" -o "$work/europe_utm.gpkg" >"$work/europe_utm.log" 2>&1 \
  || { cat "$work/europe_utm.log"; echo "FAIL: run europe_utm"; exit 1; }
"$geoplumb" run "$work/layers.toml" points "${world[@]}" -o "$work/points.gpkg" >"$work/points.log" 2>&1 \
  || { cat "$work/points.log"; echo "FAIL: run points"; exit 1; }
if "$geoplumb" run "$work/layers.toml" europe_utm_no_crs "${utm[@]}" -o "$work/no_crs.gpkg" >"$work/no_crs.log" 2>&1; then
  echo "FAIL: geoplumb read a shapefile whose .prj names no epsg code without a crs"
  exit 1
fi
echo "run on europe_utm_no_crs: $(grep -v '^$' "$work/no_crs.log" | tail -1)"
grep -q -i 'crs' "$work/no_crs.log" || { echo "FAIL: the no-crs refusal does not name crs"; exit 1; }

for output in countries europe_utm points; do
  if ! python3 -I -m osgeo_utils.samples.validate_gpkg "$work/$output.gpkg" >"$work/$output.validate" 2>&1; then
    cat "$work/$output.validate"
    echo "FAIL: validate_gpkg.py on $output.gpkg"
    exit 1
  fi
  echo "validate_gpkg.py on $output.gpkg: $(grep -c . "$work/$output.validate") lines of output"
done

python3 -I - "$work" <<'EOF'
import csv
import sys
from pathlib import Path

from osgeo import ogr, osr

ogr.UseExceptions()
work = Path(sys.argv[1])
failures = []


def epsg(layer):
    reference = layer.GetSpatialRef()
    return reference.GetAuthorityCode(None) if reference else None


def features_by_name(layer, name_field):
    return {feature.GetField(name_field): feature for feature in layer}


for output, source, code in [("countries", "countries.shp", "4326"), ("europe_utm", "europe_utm.shp", "32632")]:
    failures_before = len(failures)
    ours_dataset = ogr.Open(str(work / f"{output}.gpkg"))
    theirs_dataset = ogr.Open(str(work / source))
    ours, theirs = ours_dataset.GetLayer(0), theirs_dataset.GetLayer(0)
    if epsg(ours) != code:
        failures.append(f"{output}: epsg {epsg(ours)}, wanted {code}")
    ours_by_name = features_by_name(ours, "NAME")
    theirs_by_name = features_by_name(theirs, "NAME")
    print(f"{output}: geoplumb {len(ours_by_name)} features, ogr {len(theirs_by_name)}")
    missing = sorted(set(theirs_by_name) - set(ours_by_name))
    extra = sorted(set(ours_by_name) - set(theirs_by_name))
    if missing or extra:
        failures.append(f"{output}: missing {missing[:5]}, extra {extra[:5]}")
    worst = 0.0
    for name, theirs_feature in theirs_by_name.items():
        ours_feature = ours_by_name.get(name)
        if ours_feature is None:
            continue
        for field in ["POP_EST", "CONTINENT", "ISO_A3"]:
            if ours_feature.GetField(field) != theirs_feature.GetField(field):
                failures.append(f"{output} {name} {field}: geoplumb {ours_feature.GetField(field)!r}, ogr {theirs_feature.GetField(field)!r}")
        a, b = ours_feature.GetGeometryRef(), theirs_feature.GetGeometryRef()
        area = b.GetArea()
        ratio = a.SymDifference(b).GetArea() / area if area else 0.0
        worst = max(worst, ratio)
        if ratio >= 1e-9:
            failures.append(f"{output} {name}: symmetric difference {ratio:.3g} of the area")
    print(f"RESULT {output}: geoplumb {len(ours_by_name)} features against ogr {len(theirs_by_name)}, largest symmetric difference "
          f"{worst:.3g} of a feature's area {'PASS' if len(failures) == failures_before else 'FAIL'}")

failures_before = len(failures)
points_dataset = ogr.Open(str(work / "points.gpkg"))
ours = points_dataset.GetLayer(0)
if epsg(ours) != "4326":
    failures.append(f"points: epsg {epsg(ours)}, wanted 4326")
definition = ours.GetLayerDefn()
types = {definition.GetFieldDefn(i).GetName(): definition.GetFieldDefn(i).GetTypeName() for i in range(definition.GetFieldCount())}
print(f"points: field types {types}")
# ogr reads a geopackage INTEGER column as Integer64
for field, wanted in [("pop_est", {"Integer", "Integer64"}), ("gdp_md", {"Real"}), ("name", {"String"})]:
    if types.get(field) not in wanted:
        failures.append(f"points {field}: type {types.get(field)}, wanted one of {sorted(wanted)}")
rows = list(csv.DictReader(open(work / "points.csv")))
ours_by_name = features_by_name(ours, "name")
print(f"points: geoplumb {len(ours_by_name)} features, csv {len(rows)} rows")
for row in rows:
    feature = ours_by_name.get(row["name"])
    if feature is None:
        failures.append(f"points: {row['name']} missing")
        continue
    point = feature.GetGeometryRef()
    if (point.GetX(), point.GetY()) != (float(row["x"]), float(row["y"])):
        failures.append(f"points {row['name']}: {point.GetX()}, {point.GetY()} against {row['x']}, {row['y']}")
    if feature.GetField("pop_est") != int(row["pop_est"]) or feature.GetField("gdp_md") != float(row["gdp_md"]):
        failures.append(f"points {row['name']}: values {feature.GetField('pop_est')}, {feature.GetField('gdp_md')} against {row['pop_est']}, {row['gdp_md']}")
print(f"RESULT points: geoplumb {len(ours_by_name)} features against {len(rows)} csv rows, field types {types} "
      f"{'PASS' if len(failures) == failures_before else 'FAIL'}")

for failure in failures[:40]:
    print("FAIL:", failure)
sys.exit(3 if failures else 0)
EOF
status=$?

case $status in
  0) echo "PROOF PASS"; exit 0 ;;
  3) echo "PROOF FAIL: see the FAIL lines above"; exit 1 ;;
  *) echo "UNEXPECTED: the checks exited $status"; exit 1 ;;
esac
