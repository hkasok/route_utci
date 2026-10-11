#!/usr/bin/env bash
# Verify the shortlisted Route-1 candidates (routes 5, 6) with the full model.
set -uo pipefail
cd "$(dirname "$0")/.."
P="$PWD"; L=geometry_rebuild_20261011/logs/rsearch
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
INPUT_CASE_DIR="$P/input/MMC_rsearch" OUTPUT_CASE_DIR="$P/run_output/MMC_rsearch" CASE_NAME=MMC_rsearch \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
    nice bash start.sh 4 > $L/verify_pipeline.log 2>&1 || { echo "pipeline FAILED"; exit 1; }
echo "pipeline ok ($(date +%T))"
nice python3 geometry_rebuild_20261011/tools/sweep_loose.py > $L/verify_sweep.log 2>&1 || { echo "sweep FAILED"; exit 1; }
grep "check route" $L/verify_sweep.log; echo "verify done ($(date +%T))"
