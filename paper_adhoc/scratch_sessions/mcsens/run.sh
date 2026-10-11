#!/usr/bin/env bash
# Monte Carlo convergence (2000 draws) and correlation-length sensitivity (10, 40 m).
# Each run's outputs are copied to its own folder; the published 500-draw outputs
# are restored afterwards, and the paper directory is never written.
set -euo pipefail
P="/media/harshin/data_drive/route based UTCI"; L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/mcsens
export PYTHONUNBUFFERED=1
cd "$P"; RU=run_output/MMC/postprocessing/ranking_uncertainty; KEEP=run_output/MMC/postprocessing/ranking_uncertainty_500_published
[ -d "$KEEP" ] || cp -a "$RU" "$KEEP"
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do echo "  waiting for CPU $(date +%T)"; sleep 120; done; }
run() { name=$1; shift
  wait_cpu; echo "== $name ($(date +%T))"
  mkdir -p "$L/$name"
  python3 paper_ranking_uncertainty.py --flip-rate 0.270 --load-offset-range -65 0 --workers 12 --paper-dir "$L/$name" "$@" > "$L/$name/run.log" 2>&1
  rm -rf "run_output/MMC/postprocessing/ranking_uncertainty_$name"; cp -a "$RU" "run_output/MMC/postprocessing/ranking_uncertainty_$name"
  grep -E "^[0-9]+\.00|1-3|3-1|2-1" "$L/$name/run.log" | head -12
}
run corr10m --correlation-length-m 10
run corr40m --correlation-length-m 40
run n2000 --correlation-length-m 20 --draws 2000
rm -rf "$RU"; cp -a "$KEEP" "$RU"
echo "published 500-draw outputs restored; done ($(date +%T))"
