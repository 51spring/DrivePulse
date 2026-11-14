#!/usr/bin/env bash
# Generates the five 1-year datasets one after another (each uses all workers).
# Re-running continues unfinished datasets where they stopped.
#   bash sumo_sim/run_year_all.sh [python] ["area seed" ...]     (default: gangnam 3-5, pangyo 3-4)
set -e
cd "$(dirname "$0")/.."
PY="${1:-python}"
shift || true
SPECS=("$@")
[ ${#SPECS[@]} -eq 0 ] && SPECS=("gangnam 3" "gangnam 4" "gangnam 5" "pangyo 3" "pangyo 4")
mkdir -p out/year
for spec in "${SPECS[@]}"; do
  set -- $spec
  out="out/year/y_$1_s$2"
  if grep -q "완료" "$out.log" 2>/dev/null; then echo "skip $out (done)"; continue; fi
  echo "=== $out $(date '+%F %T')"
  "$PY" sumo_sim/generate_year.py --osm-area "$1" --seed "$2" --out "$out" >> "$out.log" 2>&1
  echo "=== $out done $(date '+%F %T')"
done
