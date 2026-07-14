#!/bin/bash
# One-command launcher for the LAZ -> shading STL pipeline.
# Just run:  ./start_LAZ_to_stl
#
# Runs main.py in the background (via nohup) inside a memory-capped
# systemd scope, so if anything unexpectedly balloons in memory, only
# THIS process gets killed -- not your desktop session (see the GNOME
# Shell OOM crash from the DBSCAN run for why this matters).
#
# Edit the settings below if your input file, output folder, or memory
# cap need to change -- no need to remember the full command.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

INPUT_LAZ="/media/harshin/data_drive/route based UTCI/fl2021_miami_dade_J1413971/fl2021_miami_dade_J1413971tR0_C0.laz"
OUTPUT_DIR="./out_full"
MEMORY_MAX="60G"
GROUND_PLANAR_TIMEOUT_SEC=600
LOG_FILE="full_run_log.txt"

echo "Starting LAZ -> STL pipeline in the background."
echo "  Input:      $INPUT_LAZ"
echo "  Output dir: $OUTPUT_DIR"
echo "  Memory cap: $MEMORY_MAX"
echo "  Log file:   $LOG_FILE"
echo ""
echo "Watch progress with:   tail -f $LOG_FILE"
echo "Watch memory with:     watch -n 5 free -h"
echo ""

nohup systemd-run --scope --user -p MemoryMax="$MEMORY_MAX" -- python3 -u main.py \
    --input "$INPUT_LAZ" \
    --output-dir "$OUTPUT_DIR" \
    --ground-planar-timeout-sec "$GROUND_PLANAR_TIMEOUT_SEC" \
    > "$LOG_FILE" 2>&1 &

echo "Launched (background PID: $!). This terminal is free to use for other things."
