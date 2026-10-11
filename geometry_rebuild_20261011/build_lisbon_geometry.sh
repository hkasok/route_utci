#!/usr/bin/env bash
# Rebuild Lisbon 1-6 geometry (unthinned crowns, segmented LoD1 buildings
# walled below the terrain mesh) into a staging input root. Four lanes.
set -uo pipefail
cd "$(dirname "$0")/.."
root=geometry_rebuild_20261011/input_new
mkdir -p "$root" geometry_rebuild_20261011/logs
lane() {
    for n in "$@"; do
        echo "== lisbon$n start $(date '+%F %T')"
        ( time nice python3 create_lisbon_laz_geometry.py --case "$n" --input-root "$root" --keep-intermediate ) \
            > geometry_rebuild_20261011/logs/geometry_lisbon$n.log 2>&1 \
            && echo "== lisbon$n done $(date '+%F %T')" || echo "== lisbon$n FAILED $(date '+%F %T')"
    done
}
lane 2 &
lane 3 1 &
lane 4 6 &
lane 5 &
wait
echo "== all geometry done $(date '+%F %T')"
