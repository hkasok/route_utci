#!/usr/bin/env bash
# Pre-submission review reruns (26 Oct 2026), each queued through gated.sh:
#  1. lisbon2_w3 with its published date (16 Aug 2023, Silva et al. 2024,
#     Table 3); then the held-out comparison and tables for all 12 walks.
#     The 13 Aug version is moved to variants/superseded_lisbon2_w3_13aug.
#  2. control_rerun: Lisbon 1-6 step 4 -> 6 again, unchanged (reference for 3-4)
#  3. forcing_measured_level: solar components scaled to the radiometer's
#     unshaded level (paper_adhoc/scale_solar_forcing.py), i.e. no clean-sky cap
#  4. wind_floor_0p5: street-to-roof wind floor 0.5 instead of 0.30
# Variants run on copies (variants/<name>/); published run_output/lisbonN and
# input/lisbonN are not touched.
#   launch: nohup setsid bash paper_adhoc/run_review_variants.sh > LOG 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
export USE_PEDESTRIAN_WIND=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
ALL=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)
G=paper_adhoc/gated.sh

# --- 1. held-out Lisbon 2 walk 3 on its published date ----------------------
AGG="$R/run_output/lisbon_heldout_validation"; HL="$AGG/logs"
SUP="$R/variants/superseded_lisbon2_w3_13aug"
if [ ! -d "$SUP" ]; then
    mkdir -p "$SUP"
    mv input/lisbon2_w3 "$SUP/input_lisbon2_w3"
    mv run_output/lisbon2_w3 "$SUP/run_output_lisbon2_w3"
fi
echo "== lisbon2_w3 (16 Aug) $(date '+%F %T')"
$G heldout_lisbon2_w3_16aug 6 bash paper_adhoc/heldout_case.sh lisbon2_w3 > "$HL/lisbon2_w3.log" 2>&1 \
    || { echo "lisbon2_w3 FAILED"; exit 1; }
args=(); for a in 1 2 3 4 5 6; do for w in 2 3; do args+=(--case "lisbon${a}_w$w"); done; done
$G heldout_compare 6 nice -n 19 python3 compare_mrt_lisbon_data.py "${args[@]}" \
    --input-root "$R/input" --output-root "$R/run_output" \
    --aggregate-output-dir "$AGG" > "$HL/compare.log" 2>&1 || { echo "held-out comparison FAILED"; exit 1; }
nice -n 19 python3 paper_adhoc/heldout_summary.py --out-dir "$AGG/summary" > "$HL/summary.log" 2>&1 \
    || echo "held-out summary FAILED"
echo "== held-out done $(date '+%F %T')"

# --- 2-4. Lisbon variants -----------------------------------------------------
make_input() {   # variant name -> variants/<name>/input_alt with copied cases
    local inp="$R/variants/$1/input_alt"
    for c in "${ALL[@]}"; do
        mkdir -p "$inp/$c"
        rsync -a --delete --exclude 'geometry.bak*' --exclude '*.bak*' \
            --exclude 'case.json.*' "input/$c/" "$inp/$c/"
    done
}
seed() {   # variant name -> lean copies of the base outputs
    for c in "${ALL[@]}"; do
        mkdir -p "variants/$1/run_output"
        rsync -a --exclude solweig --exclude 'viz.bak*' --exclude 'route_jos3.bak*' \
            "run_output/$c/" "variants/$1/run_output/$c/"
    done
}
run_variant() {   # name, env...
    local name=$1; shift
    seed "$name"
    echo "== $name start $(date '+%F %T')"
    $G "variant_$name" 6 env "$@" INPUT_ROOT="$R/variants/$name/input_alt" \
        nice -n 19 ./paper_variant_runs.sh "$name" 4 "${ALL[@]}" \
        > "variants/$name/variant.log" 2>&1 \
        && echo "== $name done $(date '+%F %T')" || echo "== $name FAILED (variants/$name/variant.log)"
}

make_input control_rerun
run_variant control_rerun FORCE_SVF=0

make_input forcing_measured_level
for c in "${ALL[@]}"; do
    python3 paper_adhoc/scale_solar_forcing.py --case-dir "variants/forcing_measured_level/input_alt/$c" \
        || { echo "scaling FAILED $c"; exit 1; }
done
run_variant forcing_measured_level FORCE_SVF=0

make_input wind_floor_0p5
for c in "${ALL[@]}"; do
    WIND_STREET_RATIO_FLOOR=0.5 nice -n 19 python3 weather_from_sensors.py \
        --input-root "$R/variants/wind_floor_0p5/input_alt" --output-root "$R/run_output" \
        --case "$c" > "variants/wind_floor_0p5/input_alt/$c/weather_from_sensors.log" 2>&1 \
        || { echo "forcing FAILED $c"; exit 1; }
done
run_variant wind_floor_0p5 FORCE_SVF=0
echo "== all review reruns done $(date '+%F %T')"
