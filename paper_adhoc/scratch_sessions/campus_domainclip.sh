#!/usr/bin/env bash
# Re-run the FIU campus chain after regenerating Route 1 inside the LiDAR domain.
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"
L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/campus_logs2
mkdir -p "$L"; cd "$P"
B=run_output_MMC_pre_domainclip_20261004
if [ ! -d "$B" ]; then echo "backup -> $B ($(date +%T))"; cp -a run_output/MMC "$B"; fi
CASE=(INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$P/run_output/MMC" CASE_NAME=MMC)
echo "pipeline from step 4 incl. preparation ($(date +%T))"
env "${CASE[@]}" USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
  bash start.sh 4 > "$L/pipeline.log" 2>&1
echo "uniform grids ($(date +%T))"
python3 sensitivity.py > "$L/sensitivity.log" 2>&1
python3 sensitivity.py --ta-min 32.590 --ta-max 32.728 --e-min 29.5410 --e-max 29.5438 \
  --output-dir run_output/MMC/postprocessing/sensitivity_2sigma > "$L/sensitivity_2sigma.log" 2>&1
echo "figures + sweeps ($(date +%T))"
python3 paper_jos3_figures.py > "$L/jos3_auto.log" 2>&1
python3 paper_jos3_figures.py --fixed-clothing > "$L/jos3_fixed.log" 2>&1
echo "ranking uncertainty ($(date +%T))"
python3 paper_ranking_uncertainty.py --flip-rate 0.270 --correlation-length-m 20 \
  --load-offset-range -65 0 > "$L/ranking.log" 2>&1
echo "Ta heterogeneity ($(date +%T))"
python3 paper_ta_heterogeneity.py > "$L/ta_het.log" 2>&1
echo "global-reflection variant for the deep-shade statement ($(date +%T))"
V="$P/variants/mmc_global_reflected/run_output/MMC"
rm -rf "$P/variants/mmc_global_reflected"; mkdir -p "$(dirname "$V")"
cp -a run_output/MMC "$V"
env INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$V" CASE_NAME=MMC \
  MRT_FACET_EXTRA_ARGS="--reflected-model global" USE_PEDESTRIAN_WIND=0 \
  SKIP_PEDESTRIAN_WIND=1 SKIP_05=1 SKIP_05A=1 SKIP_05B=1 SKIP_06=1 SKIP_07=1 SKIP_08=1 SKIP_09=1 \
  SOLWEIG_COMPARE=0 bash start.sh 4 > "$L/global_reflected.log" 2>&1
echo "done ($(date +%T))"
