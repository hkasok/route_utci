#!/usr/bin/env bash
# Phase A: corrected (metric) campus geometry with the current four routes;
# full-model sweep to calibrate and run the Route 1 shade search.
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"
L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/metricA
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$P"
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do echo "  waiting for CPU (load $(cut -d' ' -f1 /proc/loadavg)) $(date +%T)"; sleep 120; done; }
B=run_output_MMC_feet_20261006
if [ ! -d "$B" ]; then echo "backup -> $B ($(date +%T))"; cp -a run_output/MMC "$B"; fi
wait_cpu; echo "pipeline from step 2 ($(date +%T))"
INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$P/run_output/MMC" CASE_NAME=MMC \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 FORCE_SVF=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
  bash start.sh 2 > "$L/pipeline.log" 2>&1
grep -a "Route points\|thermal facets kept\|Pipeline complete" "$L/pipeline.log" | sed 's/\x1b\[[0-9;]*m//g'
wait_cpu; echo "fixed-clothing sweep ($(date +%T))"
python3 paper_jos3_figures.py --fixed-clothing --paper-dir "$L/fig" > "$L/sweep.log" 2>&1 || echo "sweep check failed -- see sweep.log"
grep "check route\|lowest" "$L/sweep.log" || true
wait_cpu; echo "route search ($(date +%T))"
python3 paper_route_search.py > "$L/search.log" 2>&1
tail -20 "$L/search.log"
echo "done ($(date +%T))"
