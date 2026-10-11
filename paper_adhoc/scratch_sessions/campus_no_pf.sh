#!/usr/bin/env bash
# Re-run the FIU campus chain with the potential-flow wind removed: every facet's
# convection uses the hourly reference wind (USE_PEDESTRIAN_WIND=0), as in Lisbon.
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"
L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/campus_logs
mkdir -p "$L"; cd "$P"
B=run_output_MMC_pre_nopf_20261004
if [ ! -d "$B" ]; then echo "backup -> $B ($(date +%T))"; cp -a run_output/MMC "$B"; fi
echo "pipeline ($(date +%T))"
INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$P/run_output/MMC" CASE_NAME=MMC \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_05=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
  bash start.sh 4 > "$L/pipeline.log" 2>&1
grep -a "Surface convection wind\|Convection reference wind" "$L/pipeline.log" || true
echo "uniform grids ($(date +%T))"
python3 sensitivity.py > "$L/sensitivity.log" 2>&1
python3 sensitivity.py --ta-min 32.591 --ta-max 32.731 --e-min 29.5406 --e-max 29.5434 \
  --output-dir run_output/MMC/postprocessing/sensitivity_2sigma > "$L/sensitivity_2sigma.log" 2>&1
echo "figures + sweeps ($(date +%T))"
python3 paper_jos3_figures.py > "$L/jos3_auto.log" 2>&1
python3 paper_jos3_figures.py --fixed-clothing > "$L/jos3_fixed.log" 2>&1
echo "ranking uncertainty ($(date +%T))"
python3 paper_ranking_uncertainty.py --flip-rate 0.270 --correlation-length-m 20 \
  --load-offset-range -65 0 > "$L/ranking.log" 2>&1
echo "Ta heterogeneity ($(date +%T))"
python3 paper_ta_heterogeneity.py > "$L/ta_het.log" 2>&1
echo "done ($(date +%T))"
