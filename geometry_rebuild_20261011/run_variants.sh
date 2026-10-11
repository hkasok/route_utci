#!/usr/bin/env bash
# Lisbon variants and globe-sensitivity tests on the rebuilt geometry
# (11 Oct 2026). Run after run_lisbon_base.sh and BEFORE SOLWEIG, from lean
# copies of the base outputs (no solweig/). Lanes run in parallel but each
# block waits while the machine is busy (load >= 16).
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
L=geometry_rebuild_20261011/logs
mkdir -p "$L"
export USE_PEDESTRIAN_WIND=1
ALL=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do sleep 60; done; }
seed() {   # variant root, cases...
    local root=$1; shift
    for c in "$@"; do
        mkdir -p "$root/run_output"
        rsync -a --exclude solweig --exclude 'viz.bak*' --exclude 'route_jos3.bak*' \
            "run_output/$c/" "$root/run_output/$c/"
    done
}
variant() {   # name, start step, env..., -- cases...
    local name=$1 step=$2; shift 2
    local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local cases=("$@")
    seed "variants/$name" "${cases[@]}"
    wait_cpu
    echo "== $name start $(date '+%F %T')"
    env "${envs[@]}" nice ./paper_variant_runs.sh "$name" "$step" "${cases[@]}" \
        > "$L/variant_$name.log" 2>&1 && echo "== $name done $(date '+%F %T')" \
        || echo "== $name FAILED $(date '+%F %T') (see $L/variant_$name.log)"
}

laneA() {
    variant height_z0p8 4 Z_HEIGHT=0.8 FORCE_SVF=1 -- "${ALL[@]}"
    variant height_z1p2 4 Z_HEIGHT=1.2 FORCE_SVF=1 -- "${ALL[@]}"
}
laneB() {
    variant conv_rays48x36 4 "FACET_SELECT_EXTRA_ARGS=--n-lw-azimuth 48 --n-lw-elevation 36" SKIP_05=1 -- lisbon1
    variant conv_stride4 4 POINT_STRIDE=4 SKIP_05=1 -- lisbon1
    variant conv_range150 4 MAX_DISTANCE=150 SKIP_05=1 -- lisbon1
    variant conv_range600 4 MAX_DISTANCE=600 SKIP_05=1 -- lisbon1
    variant conv_fullscene 4 FACET_SELECT_EXTRA_ARGS=--select-all SKIP_05=1 -- lisbon1
    variant abl_surfaces_at_ta 4 "MRT_FACET_EXTRA_ARGS=--facet-temperature-source air" \
        SKIP_05=1 SKIP_05A=1 SKIP_05B=1 -- "${ALL[@]}"
    variant abl_global_reflected 4 "MRT_FACET_EXTRA_ARGS=--reflected-model global" \
        SKIP_05=1 SKIP_05A=1 SKIP_05B=1 -- "${ALL[@]}"
}
laneC() {
    variant abl_steady_state 4 EB_EXTRA_ARGS=--steady-state SKIP_05=1 SKIP_05A=1 -- "${ALL[@]}"
    # relative path: start.sh splits EB_EXTRA_ARGS on spaces and the repo path has one
    variant abl_generic_paving 4 "EB_EXTRA_ARGS=--material-json variants_generic_paving_material.json" \
        SKIP_05=1 SKIP_05A=1 -- "${ALL[@]}"
}
laneD() {
    # Hemisphere crowns: same buildings and terrain, crowns rebuilt with the
    # flat-based hemisphere model from the same (unthinned) returns. Run from
    # step 2, so the ground and surface materials see this vegetation mesh.
    local inp="$R/variants/abl_hemisphere_crowns/input_hemi"
    mkdir -p "$inp"
    for n in 1 2 3 4 5 6; do
        c=lisbon$n
        rm -rf "$inp/$c"
        rsync -a --exclude 'geometry.bak*' --exclude 'geometry' "input/$c/" "$inp/$c/"
        mkdir -p "$inp/$c/geometry"
        wait_cpu
        nice python3 create_lisbon_laz_geometry.py --input-root "$inp" --case "$n" \
            --crown-model hemisphere --overwrite > "$L/hemisphere_geometry_$c.log" 2>&1 \
            || { echo "== hemisphere geometry FAILED $c"; return; }
        for f in building_final.stl ground_and_water_final.stl; do
            cmp -s "$inp/$c/geometry/$f" "input/$c/geometry/$f" \
                || { echo "== hemisphere $c: $f differs from the base"; return; }
        done
        # case.json was rewritten by the builder with the base setup preserved;
        # keep the base forcing files exactly
        cp -p "input/$c/case.json" "$inp/$c/case.json"
    done
    seed variants/abl_hemisphere_crowns "${ALL[@]}"
    wait_cpu
    echo "== abl_hemisphere_crowns start $(date '+%F %T')"
    INPUT_ROOT="$inp" FORCE_SVF=1 nice ./paper_variant_runs.sh abl_hemisphere_crowns 2 \
        > "$L/variant_abl_hemisphere_crowns.log" 2>&1 \
        && echo "== abl_hemisphere_crowns done $(date '+%F %T')" \
        || echo "== abl_hemisphere_crowns FAILED $(date '+%F %T')"
}
laneE() {
    run_facade() {   # name, MRT extra, EB extra, skips, start step, convection scale
        local V=$1 MRT_X=$2 EB_X=$3 SKIPS=$4 STEP=$5 CONV=$6 T="$R/facade_tests/$1"
        mkdir -p "$T"
        [ -e "$T/input" ] || ln -s "$R/input" "$T/input"
        seed "$T" "${ALL[@]}"
        wait_cpu
        echo "== facade $V start $(date '+%F %T')"
        for c in "${ALL[@]}"; do
            env $SKIPS SKIP_06=1 SKIP_07=1 SKIP_09=1 SOLWEIG_COMPARE=0 MRT_FACET_EXTRA_ARGS="$MRT_X" \
                EB_EXTRA_ARGS="$EB_X" GLOBE_FORCED_CONVECTION_SCALE=$CONV CASE_NAME=$c \
                INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$T/run_output/$c" \
                nice bash start.sh "$STEP" > "$T/run_output/$c/variant_run.log" 2>&1 \
                || { echo "== facade $V FAILED on $c"; return; }
        done
        GLOBE_FORCED_CONVECTION_SCALE=$CONV nice python3 compare_mrt_lisbon_data.py --require-all \
            --input-root "$R/input" --output-root "$T/run_output" > "$T/run_output/compare_run.log" 2>&1 \
            || { echo "== facade $V comparison FAILED"; return; }
        mkdir -p "$T/fig"
        GLOBE_FORCED_CONVECTION_SCALE=$CONV nice python3 paper_validation_figures.py --root "$T" \
            --paper-dir "$T/fig" > "$T/fig/validation_figures.log" 2>&1
        echo "== facade $V done $(date '+%F %T')"
    }
    run_facade wall_m3 "--wall-temperature-offset-K -3" "" "SKIP_05=1 SKIP_05A=1 SKIP_05B=1" 4 1.0
    run_facade wall_m6 "--wall-temperature-offset-K -6" "" "SKIP_05=1 SKIP_05A=1 SKIP_05B=1" 4 1.0
    run_facade albedo_060 "" "--generic-facade-albedo 0.6" "SKIP_05=1 SKIP_05A=1" 4 1.0
    run_facade conv_13 "" "" "" 6 1.3
    run_facade conv_15 "" "" "" 6 1.5
}

laneA & laneB & laneC & laneD & laneE &
wait
echo "== all variants done $(date '+%F %T')"
