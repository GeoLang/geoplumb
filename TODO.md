# TODO

- ranged COG source: HTTP range reads with real overview selection, needs a windowed reader in terrano (its COG support is write-side only, the README's range-read claim is aspirational)
- fan-in: mosaic of multiple sources and two-input map algebra, brings back the backtracking fixation from the g2g solver
- reproject auto-plug on CRS mismatch instead of failing negotiation
- disk cache tier behind the same entry map
- STAC collection source (open Landsat/Sentinel COG buckets), the highest-leverage step toward Earth Engine-shaped workloads
- vector and point cloud chunk variants (nubis, the PDAL slot)
- geodukt executor swap onto the engine
