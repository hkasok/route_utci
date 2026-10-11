#!/usr/bin/env bash
# rerun_cloudfix_variants_20261011.sh -- re-run Lisbon 4 and 6 in every variant
# that contains them, from fresh copies of the corrected base outputs, then
# re-derive each variant's six-case comparison and pooled figures.
set -euo pipefail
cd "$(dirname "$0")"
B="$PWD/backups_cloudfix_20261011/variants"
export USE_PEDESTRIAN_WIND=1
mkdir -p "$B"
fresh() {   # move the old Lisbon 4/6 variant outputs aside
    for c in lisbon4 lisbon6; do
        d="variants/$1/run_output/$c"
        if [ -d "$d" ]; then mkdir -p "$B/$1"; [ -d "$B/$1/$c" ] && rm -rf "$d" || mv "$d" "$B/$1/$c"; fi
    done
}
compare6() {  # six-case comparison and pooled figures for one variant
    local root="$PWD/variants/$1" inroot="${2:-$PWD/input}"
    echo "== $1: six-case comparison ($(date '+%F %T'))"
    nice python3 compare_mrt_lisbon_data.py --case lisbon1 --case lisbon2 --case lisbon3 --case lisbon4 --case lisbon5 --case lisbon6 \
        --input-root "$inroot" --output-root "$root/run_output" \
        --aggregate-output-dir "$root/run_output/lisbon_sensor_validation" > "$root/run_output/compare_run.log" 2>&1
    mkdir -p "$root/fig"
    nice python3 paper_validation_figures.py --root "$root" --paper-dir "$root/fig" > "$root/fig/validation_figures.log" 2>&1 \
        || echo "validation figures FAILED for $1"
}
run() {  # name, then env assignments
    local name="$1"; shift
    fresh "$name"
    env "$@" nice ./paper_variant_runs.sh "$name" 4 lisbon4 lisbon6
    compare6 "$name" "$(env "$@" bash -c 'echo ${INPUT_ROOT:-$PWD/input}')"
}
run height_0p8 SENSOR_HEIGHT_M=0.8 SKIP_05=1 SKIP_05A=1 SKIP_05B=1
run height_1p2 SENSOR_HEIGHT_M=1.2 SKIP_05=1 SKIP_05A=1 SKIP_05B=1
run abl_surfaces_at_ta "MRT_FACET_EXTRA_ARGS=--facet-temperature-source air" SKIP_05=1 SKIP_05A=1 SKIP_05B=1
run abl_global_reflected "MRT_FACET_EXTRA_ARGS=--reflected-model global" SKIP_05=1 SKIP_05A=1 SKIP_05B=1
run abl_steady_state EB_EXTRA_ARGS=--steady-state SKIP_05=1 SKIP_05A=1
run abl_generic_paving "EB_EXTRA_ARGS=--material-json $PWD/variants_generic_paving_material.json" SKIP_05=1 SKIP_05A=1
run height_z0p8 Z_HEIGHT=0.8 FORCE_SVF=1
run height_z1p2 Z_HEIGHT=1.2 FORCE_SVF=1
run abl_hemisphere_crowns INPUT_ROOT="$PWD/variants/abl_hemisphere_crowns/input_hemi" FORCE_SVF=1 SKIP_05A=0
echo "== all variants done ($(date '+%F %T'))"
