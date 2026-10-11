#!/usr/bin/env bash
set -euo pipefail
cd "/media/harshin/data_drive/route based UTCI"
PY=/media/harshin/data_drive/solweig/.venv/bin/python
for c in lisbon2 lisbon3 lisbon4 lisbon5 lisbon6; do
  echo "== $c ($(date +%T))"
  python3 paper_solweig_inputs.py --case $c > /dev/null
  $PY paper_solweig_run.py --case $c > run_output/$c/solweig/run.log 2>&1
  $PY paper_solweig_compare.py --case $c > run_output/$c/solweig/compare.log 2>&1
  grep "globe .*solweig  \|globe .*trec  " run_output/$c/solweig/compare.log | head -4
done
echo "done ($(date +%T))"
