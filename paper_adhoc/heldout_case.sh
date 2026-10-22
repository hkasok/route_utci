#!/usr/bin/env bash
# One held-out Lisbon case (2nd or 3rd daytime walk), run exactly as the
# published lisbonN cases were on 11 Oct 2026 (geometry_rebuild_20261011/
# run_lisbon_base.sh): steps 2-3, sensor forcing, then step 4 -> 6.
#   usage: paper_adhoc/heldout_case.sh lisbonN_w2|lisbonN_w3
# Writes only input/<case> and run_output/<case>; never a published lisbonN.
# Called through paper_adhoc/gated.sh by paper_adhoc/run_heldout_walks.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
c=${1:?case name}
[[ "$c" =~ ^lisbon([1-6])_w([23])$ ]] || { echo "refusing case name '$c'"; exit 2; }
area=${BASH_REMATCH[1]}; w=${BASH_REMATCH[2]}
measurement=$([ "$w" = 2 ] && echo 2nd || echo 3rd)
fail() { echo "FAILED $c: $*"; exit 1; }
export USE_PEDESTRIAN_WIND=1          # as every published Lisbon run
# The machine is shared: cap library thread pools at the gated allowance.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
NICE=(nice -n 19)
OUT="$R/run_output/$c"
mkdir -p "$OUT"
echo "== $c ($measurement daytime walk of study area $area) $(date '+%F %T')"

# 0. inputs: walk + night walk, scene copied from input/lisbon$area
if [ ! -f "input/$c/case.json" ]; then
    extra=()
    [ "$c" = lisbon2_w3 ] && extra=(--assume-date 2023-08-13)   # layer has no date
    "${NICE[@]}" python3 prepare_lisbon_case_inputs.py --case "$area" \
        --measurement "$measurement" --case-suffix "_w$w" --skip-osm-download \
        "${extra[@]}" || fail "input preparation"
fi
# The pedestrian network depends only on the case bbox (identical to
# lisbon$area's); copy it so step 2 does not query Overpass again.
[ -d "$OUT/osm_paths" ] || cp -a "run_output/lisbon$area/osm_paths" "$OUT/" || fail "osm_paths copy"

# 1. provisional forcing: steps 2-3 need a weather file to exist
"${NICE[@]}" python3 weather_from_sensors.py --input-root "$R/input" --output-root "$R/run_output" \
    --case "$c" > "$OUT/weather_from_sensors_pass1.log" 2>&1 || fail "forcing pass 1"

# 2. steps 2-3 (ground + surface materials, potential-flow wind field)
INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$OUT" CASE_NAME=$c \
SKIP_05=1 SKIP_05A=1 SKIP_05B=1 SKIP_05FACET=1 SKIP_06=1 SKIP_07=1 SKIP_08=1 SKIP_09=1 \
SOLWEIG_COMPARE=0 "${NICE[@]}" bash start.sh 2 > "$OUT/steps23.log" 2>&1 || fail "steps 2-3"
grep -q "Pipeline complete" "$OUT/steps23.log" || fail "steps 2-3 did not complete"
mv "$OUT/stage_timings.log" "$OUT/stage_timings_steps23.log"

# 3. the paper's forcing: envelope fitted to THIS walk's pyranometer, inlet
#    wind inverted through this case's step-3 field
"${NICE[@]}" python3 weather_from_sensors.py --input-root "$R/input" --output-root "$R/run_output" \
    --case "$c" > "$OUT/weather_from_sensors.log" 2>&1 || fail "forcing pass 2"

# 4. step 4 -> 6 (as run_lisbon_step4.sh)
rm -f "$OUT/stage_timings.log"
FORCE_SVF=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$OUT" CASE_NAME=$c \
    "${NICE[@]}" bash start.sh 4 > "$OUT/variant_run.log" 2>&1 || fail "step 4"
grep -q "Pipeline complete" "$OUT/variant_run.log" || fail "step 4 did not complete"
echo "== $c done $(date '+%F %T')"
