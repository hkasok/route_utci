#!/usr/bin/env bash
# SOLWEIG on the rebuilt geometry (11 Oct 2026), cases sequential on a quiet
# machine (GPU); wall-clock time of every step recorded for Supplement S13.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/media/harshin/data_drive/solweig/.venv/bin/python
T=geometry_rebuild_20261011/logs/solweig_timings.tsv
echo -e "case\tstep\tseconds" > "$T"
timed() { local c=$1 s=$2; shift 2; local t0=$(date +%s.%N); "$@"; local rc=$?
          echo -e "$c\t$s\t$(echo "$(date +%s.%N) - $t0" | bc)" >> "$T"; return $rc; }
for c in lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6; do
    echo "== $c ($(date +%T))"
    mkdir -p run_output/$c/solweig
    timed $c inputs python3 paper_solweig_inputs.py --case $c > run_output/$c/solweig/inputs.log 2>&1 || { echo "inputs FAILED $c"; exit 1; }
    for r in standing globe globe_k050 globe_k025; do
        timed $c run_$r $PY paper_solweig_run.py --case $c --runs $r > run_output/$c/solweig/run_$r.log 2>&1 \
            || { echo "run $r FAILED $c"; exit 1; }
    done
    timed $c compare $PY paper_solweig_compare.py --case $c > run_output/$c/solweig/compare.log 2>&1 || { echo "compare FAILED $c"; exit 1; }
done
python3 paper_solweig_pool.py > geometry_rebuild_20261011/logs/solweig_pool.log 2>&1 || { echo "pool FAILED"; exit 1; }
echo "== solweig done ($(date '+%F %T'))"
