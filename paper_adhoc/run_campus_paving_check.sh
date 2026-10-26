#!/usr/bin/env bash
# Campus material check (26 Oct 2026). input/MMC/config/osm_ground_materials.json
# embeds an earlier copy of the paving-stone and plaza properties (k 1.1,
# rho*c 1.85 MJ, 0.25-0.30 m), so the campus did not get the library's
# revised paving stone (k 1.9, 2.10 MJ, 0.60 m) that Lisbon uses. Two copies of
# the campus, step 4 -> 6 as the published run, each queued through gated.sh:
#   mmc_control       unchanged inputs (reproducibility reference)
#   mmc_paving_library  those two entries set to the library values
# Published input/MMC and run_output/MMC are not touched.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
for name in mmc_control mmc_paving_library; do
    V="$R/variants/$name"
    mkdir -p "$V/input_alt" "$V/run_output"
    rsync -a --delete --exclude '*.bak*' --exclude 'case.json.*' input/MMC/ "$V/input_alt/MMC/"
    rsync -a --exclude solweig --exclude postprocessing --exclude 'viz.bak*' \
        --exclude 'route_jos3.bak*' run_output/MMC/ "$V/run_output/MMC/"
    if [ "$name" = mmc_paving_library ]; then
        python3 - "$V/input_alt/MMC/config/osm_ground_materials.json" <<'EOF' || exit 1
import json, sys
sys.path.insert(0, ".")
import material_library as ml
path = sys.argv[1]
cfg = json.load(open(path))
for name in ("paving_stone_pedestrian", "pedestrian_plaza"):
    m = ml.MATERIAL_LIBRARY[name]
    e = cfg["materials"][name]
    e.update(albedo=m.albedo, emissivity=m.emissivity, thermal_conductivity=m.thermal_conductivity_WmK,
             k=m.thermal_conductivity_WmK, density=m.density_kgm3, specific_heat=m.specific_heat_JkgK,
             C=m.density_kgm3 * m.specific_heat_JkgK, depth=m.depth_m, n_layers=m.n_layers)
    print(name, e)
json.dump(cfg, open(path, "w"), indent=2)
EOF
    fi
    echo "== $name start $(date '+%F %T')"
    paper_adhoc/gated.sh "campus_$name" 6 env INPUT_CASE_DIR="$V/input_alt/MMC" \
        OUTPUT_CASE_DIR="$V/run_output/MMC" CASE_NAME=MMC USE_PEDESTRIAN_WIND=0 \
        SKIP_PEDESTRIAN_WIND=1 SKIP_06=1 SKIP_07=1 SOLWEIG_COMPARE=0 \
        nice -n 19 bash start.sh 4 > "$V/pipeline.log" 2>&1 \
        && grep -q "Pipeline complete" "$V/pipeline.log" \
        && echo "== $name done $(date '+%F %T')" || echo "== $name FAILED ($V/pipeline.log)"
done
