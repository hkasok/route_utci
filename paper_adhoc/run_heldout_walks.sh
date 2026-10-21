#!/usr/bin/env bash
# Held-out validation on the 2nd and 3rd daytime walks of the six Lisbon
# study areas (reviewer request). Each case is a NEW case lisbonN_w2/_w3 with
# lisbonN's scene copied (paper_adhoc/heldout_case.sh); the published lisbonN
# inputs and outputs are not touched. Every pipeline run is queued through
# paper_adhoc/gated.sh (serial, waits for 8 free threads). The comparison
# writes its aggregate to run_output/lisbon_heldout_validation, separate from
# the published run_output/lisbon_sensor_validation.
#   launch: nohup setsid bash paper_adhoc/run_heldout_walks.sh > LOG 2>&1 &
#   tabulate: python3 paper_adhoc/heldout_summary.py
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
AGG="$R/run_output/lisbon_heldout_validation"
L="$AGG/logs"
mkdir -p "$L"
CASES=(lisbon1_w2 lisbon1_w3 lisbon2_w2 lisbon2_w3 lisbon3_w2 lisbon3_w3
       lisbon4_w2 lisbon4_w3 lisbon5_w2 lisbon5_w3 lisbon6_w2 lisbon6_w3)
done_cases=(); failed=()
echo "== held-out walks queued $(date '+%F %T')"
for c in "${CASES[@]}"; do
    if paper_adhoc/gated.sh "heldout_$c" 8 bash paper_adhoc/heldout_case.sh "$c" > "$L/$c.log" 2>&1; then
        done_cases+=("$c"); echo "  $c done ($(date '+%F %T'))"
    else
        failed+=("$c"); echo "  $c FAILED ($(date '+%F %T')), see $L/$c.log"
    fi
done
[ ${#done_cases[@]} -gt 0 ] || { echo "no case completed"; exit 1; }
args=(); for c in "${done_cases[@]}"; do args+=(--case "$c"); done
echo "== comparison for ${#done_cases[@]} case(s) $(date '+%F %T')"
paper_adhoc/gated.sh heldout_compare 8 nice -n 19 python3 compare_mrt_lisbon_data.py "${args[@]}" \
    --input-root "$R/input" --output-root "$R/run_output" \
    --aggregate-output-dir "$AGG" > "$L/compare.log" 2>&1 || { echo "comparison FAILED, see $L/compare.log"; exit 1; }
# tables: persistent copy next to the aggregate, plus the session scratch default
nice -n 19 python3 paper_adhoc/heldout_summary.py --out-dir "$AGG/summary" > "$L/summary.log" 2>&1 \
    || echo "summary FAILED, see $L/summary.log"
nice -n 19 python3 paper_adhoc/heldout_summary.py >> "$L/summary.log" 2>&1 || true
echo "== held-out walks finished $(date '+%F %T'); failed: ${failed[*]:-none}"
