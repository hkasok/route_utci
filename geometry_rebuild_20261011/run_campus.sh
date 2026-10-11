#!/usr/bin/env bash
# Campus (MMC) on the rebuilt buildings (11 Oct 2026): full pipeline from
# step 2 (ground + surface materials keyed to the new faces), then the Route-1
# search re-run as a check, uniform-meteorology grids, JOS-3 figures and the
# global-reflection variant. The ranking Monte Carlo and the Ta-heterogeneity
# analysis need the new Lisbon validation and are run separately.
set -uo pipefail
cd "$(dirname "$0")/.."
P="$PWD"
L=geometry_rebuild_20261011/logs/campus
mkdir -p "$L"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
wait_cpu() { while [ "$(awk '{print int($1)}' /proc/loadavg)" -ge 16 ]; do sleep 60; done; }
fail() { echo "FAILED: $*"; exit 1; }

wait_cpu; echo "== pipeline from step 4 ($(date '+%F %T'))"
INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$P/run_output/MMC" CASE_NAME=MMC \
USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 FORCE_SVF=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
    nice bash start.sh 4 > "$L/pipeline.log" 2>&1 || fail "campus pipeline"
grep -q "Pipeline complete" "$L/pipeline.log" || fail "campus pipeline did not complete"

wait_cpu; echo "== Route-1 search on the new geometry (check) ($(date '+%T'))"
nice python3 paper_jos3_figures.py --fixed-clothing --paper-dir "$L/fig_search" > "$L/sweep_for_search.log" 2>&1 \
    || echo "  sweep for search failed (see log)"
nice python3 paper_route_search.py --out "$L/route_search" > "$L/route_search.log" 2>&1 \
    || echo "  route search failed (see log)"

wait_cpu; echo "== uniform-meteorology grids ($(date '+%T'))"
nice python3 sensitivity.py > "$L/sensitivity.log" 2>&1 || echo "  sensitivity failed"
nice python3 sensitivity.py --ta-min 32.590 --ta-max 32.731 --e-min 29.5410 --e-max 29.5438 \
    --output-dir run_output/MMC/postprocessing/sensitivity_2sigma > "$L/sensitivity_2sigma.log" 2>&1 \
    || echo "  sensitivity 2-sigma failed"

wait_cpu; echo "== JOS-3 figures and departure sweeps ($(date '+%T'))"
nice python3 paper_jos3_figures.py > "$L/jos3_auto.log" 2>&1 || echo "  jos3 auto failed"
nice python3 paper_jos3_figures.py --fixed-clothing > "$L/jos3_fixed.log" 2>&1 || echo "  jos3 fixed failed"
grep "check route\|lowest" "$L/jos3_fixed.log" || true

wait_cpu; echo "== global-reflection variant ($(date '+%T'))"
V="$P/variants/mmc_global_reflected/run_output/MMC"
mkdir -p "$(dirname "$V")"
rsync -a --exclude solweig "run_output/MMC/" "$V/"
env INPUT_CASE_DIR="$P/input/MMC" OUTPUT_CASE_DIR="$V" CASE_NAME=MMC MRT_FACET_EXTRA_ARGS="--reflected-model global" \
    USE_PEDESTRIAN_WIND=0 SKIP_PEDESTRIAN_WIND=1 SKIP_05=1 SKIP_05A=1 SKIP_05B=1 SKIP_06=1 SKIP_07=1 \
    SKIP_08=1 SKIP_09=1 SOLWEIG_COMPARE=0 nice bash start.sh 4 > "$L/global_reflected.log" 2>&1 \
    || echo "  global-reflection variant failed"
echo "== campus done ($(date '+%F %T'))"
