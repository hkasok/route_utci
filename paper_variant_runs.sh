#!/usr/bin/env bash
# paper_variant_runs.sh -- re-run the Lisbon validation cases under one
# variant of the pipeline and re-derive the comparison statistics.
#
# A variant is a name plus whatever environment start.sh understands
# (EB_EXTRA_ARGS, MRT_FACET_EXTRA_ARGS, FACET_SELECT_EXTRA_ARGS, POINT_STRIDE,
# MAX_DISTANCE, SENSOR_HEIGHT_M, FORCE_SVF, ...). Each case's existing results
# are copied into variants/<name>/run_output/<case> first, so the authoritative
# run_output/ is never touched, and only the requested stages are re-run on
# the copy. The comparison and the pooled validation figures are then written
# under variants/<name>/.
#
#   usage: paper_variant_runs.sh <name> <start_step> [case ...]
#
#   name        "base" re-runs IN PLACE (run_output/<case>); anything else is
#               a copy under variants/<name>/
#   start_step  passed to start.sh (4 = facet-thermal MRT onward)
#   case ...    default lisbon1..lisbon6
#
# Stages that only draw (06, 07), JOS-3 (09) and the SOLWEIG comparison are
# skipped unless the caller sets SKIP_06/SKIP_07/SKIP_09 to 0 explicitly.
set -euo pipefail
cd "$(dirname "$0")"

name="${1:?variant name}"; step="${2:?start step}"; shift 2
# INPUT_ROOT lets a variant run from an alternative input tree (e.g. a
# different crown geometry); it must hold the same case folders.
INPUT_ROOT="${INPUT_ROOT:-$PWD/input}"
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)

if [ "$name" = "base" ]; then
    root="$PWD"
else
    root="$PWD/variants/$name"
    mkdir -p "$root/run_output"
    [ -e "$root/input" ] || ln -s "$INPUT_ROOT" "$root/input"
fi

export SKIP_06="${SKIP_06:-1}" SKIP_07="${SKIP_07:-1}" SKIP_09="${SKIP_09:-1}" SOLWEIG_COMPARE=0
for case in "${cases[@]}"; do
    out="$root/run_output/$case"
    if [ "$name" != "base" ]; then
        if [ ! -d "$out" ]; then
            echo "== copying run_output/$case -> $out"
            cp -r "run_output/$case" "$out"
        fi
        rm -f "$out/stage_timings.log"
    fi
    echo "== $name / $case : start.sh $step  ($(date '+%F %T'))"
    INPUT_CASE_DIR="$INPUT_ROOT/$case" OUTPUT_CASE_DIR="$out" CASE_NAME="$case" \
        bash start.sh "$step" > "$out/variant_run.log" 2>&1 \
        || { echo "FAILED: see $out/variant_run.log"; tail -30 "$out/variant_run.log"; exit 1; }
done

echo "== comparison against the mobile measurements ($(date '+%F %T'))"
args=(); for case in "${cases[@]}"; do args+=(--case "$case"); done
python3 compare_mrt_lisbon_data.py "${args[@]}" --input-root "$INPUT_ROOT" \
    --output-root "$root/run_output" \
    --aggregate-output-dir "$root/run_output/lisbon_sensor_validation" \
    > "$root/run_output/compare_run.log" 2>&1 \
    || { echo "comparison FAILED"; tail -30 "$root/run_output/compare_run.log"; exit 1; }
if [ ${#cases[@]} -eq 6 ]; then
    mkdir -p "$root/fig"
    python3 paper_validation_figures.py --root "$root" --paper-dir "$root/fig" \
        > "$root/fig/validation_figures.log" 2>&1 || echo "validation figures FAILED (see $root/fig/validation_figures.log)"
fi
echo "== done ($(date '+%F %T'))"
