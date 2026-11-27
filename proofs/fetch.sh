#!/usr/bin/env bash
# usage: GEOPLUMB_PROOF_DATA=<directory> proofs/fetch.sh
set -euo pipefail
source "$(dirname "$0")/inputs.sh"

fetch() {
  local url=$1 file=$2 sha256=$3
  if [ ! -s "$file" ]; then
    curl -sSfL -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  echo "$sha256  $file" | sha256sum -c --quiet || { echo "$file is not the file the parity results were measured on"; exit 1; }
}

mkdir -p "$GEOPLUMB_PROOF_DATA"
# copernicus glo-30 public, aws open data bucket copernicus-dem-30m
fetch https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_E007_00_DEM/Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif \
  "$dem" 426e3d1492d94e4f23d444843f852ff5be675da5c4c14e661b5642006864047d
# esa worldcover 2021 v200, aws open data bucket esa-worldcover in eu-central-1
fetch https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N45E006_Map.tif \
  "$worldcover" 9d4d371430b1aaec777228df59616cb503a9b62fe192c1de855036c43b07e75d
# natural earth 1:110m admin 0 countries at release v5.1.2 of the natural earth vector repository
fetch https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_admin_0_countries.geojson \
  "$countries" 6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f
echo "inputs in $GEOPLUMB_PROOF_DATA"
