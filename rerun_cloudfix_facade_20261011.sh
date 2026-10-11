#!/usr/bin/env bash
# rerun_cloudfix_facade_20261011.sh -- re-run Lisbon 4 and 6 in the five
# globe-sensitivity tests (Supplementary Table S6) after the cloud-inversion
# fix, from fresh copies of the corrected base outputs; then each test's
# six-case comparison and validation summary.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
B="$R/backups_cloudfix_20261011/facade_tests"
export USE_PEDESTRIAN_WIND=1
run_variant() {   # name, MRT extra args, EB extra args, skips, start step, convection scale
    local V=$1 MRT_X=$2 EB_X=$3 SKIPS=$4 STEP=$5 CONV=$6 T="$R/facade_tests/$1"
    for c in lisbon4 lisbon6; do
        if [ -d "$T/run_output/$c" ]; then mkdir -p "$B/$V"; mv "$T/run_output/$c" "$B/$V/$c"; fi
        rsync -a --exclude 'viz.bak*' --exclude 'route_jos3.bak*' "$R/run_output/$c/" "$T/run_output/$c/"
        echo "== $V / $c : start.sh $STEP ($(date '+%F %T'))"
        env $SKIPS SKIP_06=1 SKIP_07=1 SKIP_09=1 MRT_FACET_EXTRA_ARGS="$MRT_X" EB_EXTRA_ARGS="$EB_X" \
            GLOBE_FORCED_CONVECTION_SCALE=$CONV INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$T/run_output/$c" \
            nice bash start.sh "$STEP" > "$T/run_output/$c/variant_run.log" 2>&1
    done
    GLOBE_FORCED_CONVECTION_SCALE=$CONV nice python3 compare_mrt_lisbon_data.py --require-all --output-root "$T/run_output" \
        > "$T/run_output/compare_run.log" 2>&1
    mkdir -p "$T/fig"
    GLOBE_FORCED_CONVECTION_SCALE=$CONV nice python3 paper_validation_figures.py --root "$T" --paper-dir "$T/fig" \
        > "$T/fig/validation_figures.log" 2>&1
    echo "== $V done ($(date '+%F %T'))"
}
run_variant wall_m3 "--wall-temperature-offset-K -3" "" "SKIP_05=1 SKIP_05A=1 SKIP_05B=1" 4 1.0
run_variant wall_m6 "--wall-temperature-offset-K -6" "" "SKIP_05=1 SKIP_05A=1 SKIP_05B=1" 4 1.0
run_variant albedo_060 "" "--generic-facade-albedo 0.6" "SKIP_05=1 SKIP_05A=1" 4 1.0
run_variant conv_13 "" "" "" 6 1.3
run_variant conv_15 "" "" "" 6 1.5
echo "== all facade tests done ($(date '+%F %T'))"
