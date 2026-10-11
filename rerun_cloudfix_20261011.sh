#!/usr/bin/env bash
# rerun_cloudfix_20261011.sh -- re-run Lisbon 4 and 6 after the cloud-inversion
# fix (weather_from_sensors.clearness_to_cloud_fraction). Base cases in place,
# then every variant that contains them, then the 6-case comparisons.
set -euo pipefail
cd "$(dirname "$0")"
B=backups_cloudfix_20261011
export USE_PEDESTRIAN_WIND=1          # as the 3 Oct runs of all six cases
mkdir -p "$B/run_output"
for c in lisbon4 lisbon6; do
    [ -d "$B/run_output/$c" ] || cp -a "run_output/$c" "$B/run_output/$c"
done
echo "== base runs ($(date '+%F %T'))"
for c in lisbon4 lisbon6; do
    out="run_output/$c"
    echo "== base / $c : start.sh 4 ($(date '+%F %T'))"
    SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 INPUT_CASE_DIR="$PWD/input/$c" OUTPUT_CASE_DIR="$PWD/$out" CASE_NAME="$c" \
        nice bash start.sh 4 > "$out/variant_run.log" 2>&1 \
        || { echo "FAILED: see $out/variant_run.log"; tail -30 "$out/variant_run.log"; exit 1; }
done
echo "== base comparison, six cases ($(date '+%F %T'))"
nice python3 compare_mrt_lisbon_data.py --case lisbon1 --case lisbon2 --case lisbon3 --case lisbon4 --case lisbon5 --case lisbon6 \
    --input-root "$PWD/input" --output-root "$PWD/run_output" \
    --aggregate-output-dir "$PWD/run_output/lisbon_sensor_validation" > run_output/compare_run.log 2>&1
echo "== base done ($(date '+%F %T'))"
