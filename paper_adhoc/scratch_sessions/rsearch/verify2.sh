#!/usr/bin/env bash
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"; L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch
cd "$P"
INPUT_CASE_DIR="$P/input/MMC_rsearch" OUTPUT_CASE_DIR="$P/run_output/MMC_rsearch" CASE_NAME=MMC_rsearch \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 bash start.sh 4 > $L/pipeline2.log 2>&1
echo pipeline-ok
python3 paper_jos3_figures.py --case MMC_rsearch --fixed-clothing --sweep-spacing-m 1.0 --paper-dir $L/fig > $L/sweep2.log 2>&1
echo sweep-ok; grep "check route" $L/sweep2.log
