#!/usr/bin/env bash
# rerun_cloudfix_resume_20261011.sh -- finish the Lisbon 4/6 re-runs that the
# full disk interrupted (variants height_z1p2, abl_hemisphere_crowns; facade
# tests conv_13, conv_15). Copies leave out the SOLWEIG rasters and old viz
# backups, which no variant reads.
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"; cd "$R"
BK="$R/backups_cloudfix_20261011"
export USE_PEDESTRIAN_WIND=1
copy_base() {   # fresh lean copy of the corrected base case into $1
    rsync -a --exclude 'solweig' --exclude 'viz.bak*' --exclude 'route_jos3.bak*' "$R/run_output/$2/" "$1/"
}
set_aside() {   # move the old outputs to the backup once; drop partial copies
    local dir=$1 dest=$2
    [ -d "$dir" ] || return 0
    if [ -d "$dest" ]; then rm -rf "$dir"; else mkdir -p "$(dirname "$dest")"; mv "$dir" "$dest"; fi
}
compare6() {
    local root="$1" inroot="$2"; shift 2
    env "$@" nice python3 compare_mrt_lisbon_data.py --case lisbon1 --case lisbon2 --case lisbon3 --case lisbon4 --case lisbon5 --case lisbon6 \
        --input-root "$inroot" --output-root "$root/run_output" \
        --aggregate-output-dir "$root/run_output/lisbon_sensor_validation" > "$root/run_output/compare_run.log" 2>&1
    mkdir -p "$root/fig"
    env "$@" nice python3 paper_validation_figures.py --root "$root" --paper-dir "$root/fig" > "$root/fig/validation_figures.log" 2>&1
}
variant() {     # name, input root, step, env...
    local name=$1 inroot=$2 step=$3; shift 3
    local root="$R/variants/$name"
    for c in lisbon4 lisbon6; do
        set_aside "$root/run_output/$c" "$BK/variants/$name/$c"
        copy_base "$root/run_output/$c" "$c"; rm -f "$root/run_output/$c/stage_timings.log"
        echo "== $name / $c : start.sh $step ($(date '+%F %T'))"
        env "$@" SKIP_06=1 SKIP_07=1 SKIP_09=1 SOLWEIG_COMPARE=0 INPUT_CASE_DIR="$inroot/$c" \
            OUTPUT_CASE_DIR="$root/run_output/$c" CASE_NAME="$c" nice bash start.sh "$step" \
            > "$root/run_output/$c/variant_run.log" 2>&1
    done
    compare6 "$root" "$inroot"
    echo "== $name done ($(date '+%F %T'))"
}
facade() {      # name, MRT args, EB args, skips, step, convection scale
    local V=$1 MRT_X=$2 EB_X=$3 SKIPS=$4 STEP=$5 CONV=$6 T="$R/facade_tests/$1"
    for c in lisbon4 lisbon6; do
        set_aside "$T/run_output/$c" "$BK/facade_tests/$V/$c"
        copy_base "$T/run_output/$c" "$c"
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
variant height_z1p2 "$R/input" 4 Z_HEIGHT=1.2 FORCE_SVF=1
variant abl_hemisphere_crowns "$R/variants/abl_hemisphere_crowns/input_hemi" 4 FORCE_SVF=1 SKIP_05A=0
facade conv_13 "" "" "" 6 1.3
facade conv_15 "" "" "" 6 1.5
echo "== resume done ($(date '+%F %T'))"
