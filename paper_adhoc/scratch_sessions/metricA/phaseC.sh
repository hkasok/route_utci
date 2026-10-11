#!/usr/bin/env bash
# Phase C: final campus chain on the metric geometry with the re-selected Route 1.
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"
L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/metricA/C
mkdir -p "$L"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$P"
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do echo "  waiting for CPU $(date +%T)"; sleep 120; done; }
wait_cpu; echo "pipeline from step 4 ($(date +%T))"
INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$P/run_output/MMC" CASE_NAME=MMC \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 bash start.sh 4 > "$L/pipeline.log" 2>&1
grep -a "Route points\|Pipeline complete" "$L/pipeline.log" | sed 's/\x1b\[[0-9;]*m//g'
wait_cpu; echo "uniform grids ($(date +%T))"
python3 sensitivity.py > "$L/sensitivity.log" 2>&1
python3 sensitivity.py --ta-min 32.590 --ta-max 32.731 --e-min 29.5410 --e-max 29.5438 \
  --output-dir run_output/MMC/postprocessing/sensitivity_2sigma > "$L/sensitivity_2sigma.log" 2>&1
wait_cpu; echo "figures + sweeps ($(date +%T))"
python3 paper_jos3_figures.py > "$L/jos3_auto.log" 2>&1
python3 paper_jos3_figures.py --fixed-clothing > "$L/jos3_fixed.log" 2>&1
grep "check route\|lowest" "$L/jos3_fixed.log"
wait_cpu; echo "ranking uncertainty ($(date +%T))"
python3 paper_ranking_uncertainty.py --flip-rate 0.270 --correlation-length-m 20 \
  --load-offset-range -65 0 --workers 6 > "$L/ranking.log" 2>&1
wait_cpu; echo "Ta heterogeneity ($(date +%T))"
python3 paper_ta_heterogeneity.py > "$L/ta_het.log" 2>&1
wait_cpu; echo "global-reflection variant ($(date +%T))"
V="$P/variants/mmc_global_reflected/run_output/MMC"
rm -rf "$P/variants/mmc_global_reflected"; mkdir -p "$(dirname "$V")"; cp -a run_output/MMC "$V"
env INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$V" CASE_NAME=MMC MRT_FACET_EXTRA_ARGS="--reflected-model global" \
  USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_05=1 SKIP_05A=1 SKIP_05B=1 SKIP_06=1 SKIP_07=1 SKIP_08=1 SKIP_09=1 \
  SOLWEIG_COMPARE=0 bash start.sh 4 > "$L/global_reflected.log" 2>&1
echo "done ($(date +%T))"
