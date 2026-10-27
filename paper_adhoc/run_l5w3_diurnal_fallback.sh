#!/usr/bin/env bash
# Variant: lisbon5_w3 with the diurnal air-temperature fallback (10 K range on
# the warmest window), as lisbon5 and lisbon5_w2 received. Published input/ and
# run_output/ are not touched.
set -uo pipefail
R="/media/harshin/data_drive/route based UTCI"
cd "$R"
V="$R/variants/l5w3_diurnal_fallback"; c=lisbon5_w3
mkdir -p "$V/input_alt" "$V/run_output"
rsync -a --delete "input/$c/" "$V/input_alt/$c/"
rsync -a --delete --exclude solweig --exclude 'viz.bak*' "run_output/$c/" "$V/run_output/$c/"
export USE_PEDESTRIAN_WIND=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
nice -n 19 python3 "$(dirname "$0")/forcing_with_fallback.py" --input-root "$V/input_alt" \
    --output-root "$V/run_output" --case "$c" > "$V/weather_from_sensors.log" 2>&1 || { echo "forcing FAILED"; exit 1; }
grep -n "reconstruct\|diurnal\|range" "$V/weather_from_sensors.log" | head
rm -f "$V/run_output/$c/stage_timings.log"
paper_adhoc/gated.sh l5w3_diurnal_fallback 6 env FORCE_SVF=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
    INPUT_CASE_DIR="$V/input_alt/$c" OUTPUT_CASE_DIR="$V/run_output/$c" CASE_NAME=$c \
    nice -n 19 bash start.sh 4 > "$V/variant_run.log" 2>&1
grep -q "Pipeline complete" "$V/variant_run.log" && echo "== done $(date '+%F %T')" || echo "== step 4 FAILED"
