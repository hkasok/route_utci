#!/usr/bin/env bash
# Lisbon base: step 4 -> 6 only (steps 2-3 and the forcing are unchanged),
# sequential on a quiet machine, then the comparison and the paper tables.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
export USE_PEDESTRIAN_WIND=1
fail() { echo "FAILED: $*"; exit 1; }
echo "== step 4 -> 6, sequential ($(date '+%F %T'))"
for c in lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6; do
    rm -f "run_output/$c/stage_timings.log"
    FORCE_SVF=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
    INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$R/run_output/$c" CASE_NAME=$c \
        bash start.sh 4 > "run_output/$c/variant_run.log" 2>&1 || fail "step 4 $c"
    grep -q "Pipeline complete" "run_output/$c/variant_run.log" || fail "step 4 $c did not complete"
    echo "  $c done ($(date '+%T'))"
done
echo "== comparison ($(date '+%F %T'))"
nice python3 compare_mrt_lisbon_data.py --case lisbon1 --case lisbon2 --case lisbon3 --case lisbon4 \
    --case lisbon5 --case lisbon6 --input-root "$R/input" --output-root "$R/run_output" \
    --aggregate-output-dir "$R/run_output/lisbon_sensor_validation" > run_output/compare_run.log 2>&1 \
    || fail "comparison"
nice python3 paper_validation_figures.py > geometry_rebuild_20261011/logs/validation_figures.log 2>&1 || fail "validation figures"
nice python3 paper_lisbon_route_table.py > geometry_rebuild_20261011/logs/lisbon_route_table.log 2>&1 || fail "route table"
cp run_output/lisbon1/validation/mrt_lisbon/route_1_day_experimental_radiometer_along_route_comparison.png \
    "/home/harshin/files/fastUTEC paper/along_route_radiometer.png"
PY=/media/harshin/data_drive/solweig/.venv/bin/python
for c in lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6; do
    $PY paper_solweig_compare.py --case $c > run_output/$c/solweig/compare.log 2>&1 || fail "solweig compare $c"
done
python3 paper_solweig_pool.py > geometry_rebuild_20261011/logs/solweig_pool.log 2>&1 || fail "solweig pool"
echo "== base step 4 done ($(date '+%F %T'))"
