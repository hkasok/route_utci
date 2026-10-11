#!/usr/bin/env bash
# Lisbon base runs on the rebuilt geometry (11 Oct 2026):
#   steps 2-3 (ground + surface materials, potential-flow field) per case,
#   regenerate the sensor forcing (its free-stream wind depends on the
#   buildings and on the step-3 field), then step 4 -> 6 sequentially on a
#   quiet machine (these runs supply the timing table), then the comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
L=geometry_rebuild_20261011/logs
mkdir -p "$L"
export USE_PEDESTRIAN_WIND=1          # as every earlier Lisbon run
cases=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)
fail() { echo "FAILED: $*"; exit 1; }

echo "== steps 2-3 ($(date '+%F %T'))"
for c in "${cases[@]}"; do
    mkdir -p "run_output/$c"
    INPUT_CASE_DIR="$R/input/$c" OUTPUT_CASE_DIR="$R/run_output/$c" CASE_NAME=$c \
    SKIP_05=1 SKIP_05A=1 SKIP_05B=1 SKIP_05FACET=1 SKIP_06=1 SKIP_07=1 SKIP_08=1 SKIP_09=1 \
    SOLWEIG_COMPARE=0 nice bash start.sh 2 > "$L/steps23_$c.log" 2>&1 \
        || fail "steps 2-3 $c (see $L/steps23_$c.log)"
    grep -q "Pipeline complete" "$L/steps23_$c.log" || fail "steps 2-3 $c did not complete"
    mv "run_output/$c/stage_timings.log" "run_output/$c/stage_timings_steps23.log"
    echo "  $c steps 2-3 done ($(date '+%T'))"
done

echo "== sensor forcing ($(date '+%F %T'))"
args=(); for c in "${cases[@]}"; do args+=(--case "$c"); done
nice python3 weather_from_sensors.py --input-root "$R/input" --output-root "$R/run_output" "${args[@]}" \
    > "$L/weather_from_sensors.log" 2>&1 || fail "weather_from_sensors"
python3 - <<'EOF' || fail "solar forcing changed"
import pandas as pd, numpy as np
for n in range(1, 7):
    new = pd.read_csv(f"input/lisbon{n}/config/weather_from_sensors_solar_components.csv")
    old = pd.read_csv(f"backups_georebuild_20261011/input/lisbon{n}_config/weather_from_sensors_solar_components.csv")
    cols = [c for c in old.columns if c in new.columns]
    num = [c for c in cols if np.issubdtype(old[c].dtype, np.number)]
    d = float(np.nanmax(np.abs(new[num].to_numpy(float) - old[num].to_numpy(float))))
    print(f"  lisbon{n}: solar components max |new - old| = {d:.3g}")
    assert d < 1e-6, f"lisbon{n}: the solar forcing must not change with the geometry"
EOF

echo "== step 4 -> 6, sequential ($(date '+%F %T'))"
for c in "${cases[@]}"; do
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
echo "== base done ($(date '+%F %T'))"
