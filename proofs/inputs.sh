: "${GEOPLUMB_PROOF_DATA:?set GEOPLUMB_PROOF_DATA to the directory proofs/fetch.sh downloads into}"
dem=$GEOPLUMB_PROOF_DATA/Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif
worldcover=$GEOPLUMB_PROOF_DATA/ESA_WorldCover_10m_2021_v200_N45E006_Map.tif
countries=$GEOPLUMB_PROOF_DATA/ne_110m_admin_0_countries.geojson

require_inputs() {
  local input
  for input in "$@"; do
    [ -s "$input" ] || { echo "INCONCLUSIVE: $input is missing, run proofs/fetch.sh"; exit 1; }
  done
}

file_grid() {
  # prints west south east north pixel epsg, the grid `geoplumb run` takes to write a raster onto its own pixels
  python3 -I -c '
import sys
from osgeo import gdal, osr
gdal.UseExceptions()
osr.UseExceptions()
dataset = gdal.Open(sys.argv[1])
origin_x, pixel_x, _, origin_y, _, pixel_y = dataset.GetGeoTransform()
if pixel_x != -pixel_y:
    sys.exit(f"{sys.argv[1]} has {pixel_x} by {-pixel_y} pixels, and run takes one square pixel size")
reference = osr.SpatialReference(wkt=dataset.GetProjection())
reference.AutoIdentifyEPSG()
print(*map(repr, [origin_x, origin_y + dataset.RasterYSize * pixel_y, origin_x + dataset.RasterXSize * pixel_x, origin_y, pixel_x]), reference.GetAuthorityCode(None))
' "$1"
}
