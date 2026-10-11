#!/usr/bin/env bash
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"; L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/metricA
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$P"
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do echo "  waiting for CPU $(date +%T)"; sleep 120; done; }
wait_cpu; echo "pipeline ($(date +%T))"
INPUT_CASE_DIR="$P/input/MMC_rsearch" OUTPUT_CASE_DIR="$P/run_output/MMC_rsearch" CASE_NAME=MMC_rsearch \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 bash start.sh 4 > $L/verify_pipeline.log 2>&1
wait_cpu; echo "sweep ($(date +%T))"
python3 /tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch/sweep_loose.py > $L/verify_sweep.log 2>&1
grep "check route" $L/verify_sweep.log; echo "done ($(date +%T))"
