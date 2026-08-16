#!/usr/bin/env bash
#
# run_pipeline_ui.sh -- double-clickable launcher for the TREC-Route UI.
#
# Starts pipeline_ui.py (activating .venv if present) and opens the browser.
# If the server is already running it just opens the tab instead of starting
# a second copy.
#
# Put this file in the SAME folder as start.sh and pipeline_ui.py, then:
#     chmod +x run_pipeline_ui.sh

set -uo pipefail

# Always work from the folder this script lives in (handles spaces in path).
cd "$(dirname "$(readlink -f "$0")")" || exit 1

PORT="${PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

# --- already running? just open the tab ---------------------------------
if curl -s -m 2 -o /dev/null "$URL"; then
    echo "UI already running -> $URL"
    xdg-open "$URL" >/dev/null 2>&1 &
    sleep 1
    exit 0
fi

# --- activate a virtualenv if one is present ----------------------------
for venv in .venv venv env; do
    if [ -f "$venv/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$venv/bin/activate"
        echo "activated virtualenv: $venv"
        break
    fi
done

if [ ! -f pipeline_ui.py ]; then
    echo "ERROR: pipeline_ui.py not found in $(pwd)"
    echo "Put this launcher next to pipeline_ui.py and start.sh."
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "Starting TREC-Route Pipeline UI on $URL"
echo "Close this window (or press Ctrl-C) to stop the server."
echo

# pipeline_ui.py opens the browser itself.
python3 pipeline_ui.py --port "$PORT"

# Keep the terminal open if it exited unexpectedly, so errors stay visible.
status=$?
if [ $status -ne 0 ]; then
    echo
    echo "UI exited with status $status"
    read -r -p "Press Enter to close..."
fi
