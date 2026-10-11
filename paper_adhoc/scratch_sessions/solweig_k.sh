#!/usr/bin/env bash
# extra SOLWEIG globe runs (k = 0.475, 0.2375) for the cases whose forcing is unchanged
set -euo pipefail
cd "/media/harshin/data_drive/route based UTCI"
PY=/media/harshin/data_drive/solweig/.venv/bin/python
for c in lisbon1 lisbon2 lisbon3 lisbon5; do
  echo "== $c ($(date +%T))"
  nice $PY paper_solweig_run.py --case $c --runs globe_k050 globe_k025 > run_output/$c/solweig/run_k.log 2>&1
  nice $PY paper_solweig_compare.py --case $c > run_output/$c/solweig/compare.log 2>&1
  grep "rebuilt" run_output/$c/solweig/compare.log
  grep "globe .*solweig  \|globe .*trec  " run_output/$c/solweig/compare.log | head -4
done
echo "done ($(date +%T))"
