#!/usr/bin/env bash
# Regenerate every generated table and figure of the TREC-Route paper from the
# saved pipeline outputs into OUT (default: ./paper_artefacts_regen), without
# re-running the simulation and without touching the paper folder. Compare the
# result with the paper folder to confirm nothing has drifted.
#   usage: paper_adhoc/regenerate_paper_artefacts.sh [OUT]
# Not covered here (they write into run_output/ or take ~25 min): the full
# ranking Monte Carlo (run_ranking_mc.sh), the MC sensitivity table
# (run_ranking_mc_sensitivity.sh), the SOLWEIG pooling (paper_solweig_pool.py)
# and the air-temperature heterogeneity summary (paper_ta_heterogeneity.py).
# along_route_radiometer.png is a copy of run_output/lisbon1/validation/
# mrt_lisbon/route_1_day_experimental_radiometer_along_route_comparison.png.
set -u
cd "$(dirname "$0")/.." || exit 1
O=$(realpath -m "${1:-paper_artefacts_regen}"); L="$O/logs"; mkdir -p "$L"
export OMP_NUM_THREADS=2 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
run() { name=$1; shift; mkdir -p "$O/$name"; echo "== $name"; nice -n 19 "$@" > "$L/$name.log" 2>&1 || echo "   FAILED (see $L/$name.log)"; }
run validation     python3 paper_validation_figures.py --paper-dir "$O/validation"
run lisbon_routes  python3 paper_lisbon_route_table.py --out "$O/lisbon_routes/table_lisbon_routes.tex"
run convergence    python3 paper_convergence.py --paper-dir "$O/convergence"
run ranking        python3 paper_ranking_uncertainty.py --table-only --flip-model state --offset-model beam --correlation-length-m 5 --load-offset-range -64 0 --out-name ws_main --paper-dir "$O/ranking"
run route_measures python3 paper_route_measures_table.py --paper-dir "$O/route_measures"
run uniform        python3 paper_uniform_figure.py --out "$O/uniform/uniform_signed_error.png"
run crown          python3 paper_crown_figure.py --paper-dir "$O/crown"
run lisbon_map     python3 paper_lisbon_map.py --out-dir "$O/lisbon_map"
run fig_concept    python3 paper_adhoc/make_fig_concept.py "$O/fig_concept/fig_concept.tex"
run attribution    python3 paper_globe_attribution.py --paper-dir "$O/attribution" --height-run-root 0.8=variants/height_z0p8 1.2=variants/height_z1p2
D=run_output/MMC/postprocessing/departure_sweep
if [ -f $D/departure_sweep_fixedclothing.csv ] && [ -f $D/departure_sweep.csv ]; then
  run jos3_fixed python3 paper_jos3_figures.py --fixed-clothing --reuse-sweep --paper-dir "$O/jos3_fixed"
  run jos3_auto  python3 paper_jos3_figures.py --reuse-sweep --paper-dir "$O/jos3_auto"
else echo "departure-sweep CSVs missing: JOS-3 figures skipped (they would be recomputed into run_output)"; fi
echo "done: $O"
