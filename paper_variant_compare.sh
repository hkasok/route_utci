#!/usr/bin/env bash
# paper_variant_compare.sh -- the comparison and pooled-figure steps of
# paper_variant_runs.sh on their own, for a variant whose case runs already
# exist under variants/<name>/run_output/.
#
#   usage: paper_variant_compare.sh <name> [case ...]
set -euo pipefail
cd "$(dirname "$0")"
name="${1:?variant name}"; shift
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)
root="$PWD/variants/$name"
input_root="$(readlink -f "$root/input")"
echo "== $name: comparison against the mobile measurements ($(date '+%F %T'))"
args=(); for case in "${cases[@]}"; do args+=(--case "$case"); done
python3 compare_mrt_lisbon_data.py "${args[@]}" --input-root "$input_root" \
    --output-root "$root/run_output" \
    --aggregate-output-dir "$root/run_output/lisbon_sensor_validation" \
    > "$root/run_output/compare_run.log" 2>&1 \
    || { echo "comparison FAILED"; tail -30 "$root/run_output/compare_run.log"; exit 1; }
if [ ${#cases[@]} -eq 6 ]; then
    mkdir -p "$root/fig"
    python3 paper_validation_figures.py --root "$root" --paper-dir "$root/fig" \
        > "$root/fig/validation_figures.log" 2>&1 || echo "validation figures FAILED (see $root/fig/validation_figures.log)"
fi
echo "== $name done ($(date '+%F %T'))"
