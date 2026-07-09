#!/usr/bin/env bash
# Locate existing crayfish-pipeline code and regional Hydrography90m / GeoFRESH assets on this Mac.
# Run each block, paste output back. Nothing here modifies anything.

echo "== 1) m21 analysis code =="
find ~ -type f -name "*.py" 2>/dev/null \
  | xargs grep -lEi "hydrography90m|geofresh|schoener|elapid|world of crayfish|snapp" 2>/dev/null \
  | grep -vE "/Library/|/\.Trash/" | head -50

echo "== 2) likely data files (network / predictors / occurrences) =="
find ~ -type f \( -iname "*hydrography90m*" -o -iname "*geofresh*" \
  -o -iname "*predictor*" -o -iname "*crayfish*" -o -iname "*woc*" \) 2>/dev/null \
  | grep -vE "/Library/|/\.Trash/" | head -80

echo "== 3) git repos =="
find ~ -type d -name ".git" 2>/dev/null | grep -vE "/Library/" | sed 's/\/\.git$//' | head -50

echo "== 4) large geospatial files (>5MB) =="
find ~ -type f \( -iname "*.gpkg" -o -iname "*.shp" -o -iname "*.tif" -o -iname "*.parquet" \) \
  -size +5M 2>/dev/null | grep -vE "/Library/" | head -80
