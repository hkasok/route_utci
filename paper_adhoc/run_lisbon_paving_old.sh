#!/usr/bin/env bash
# Variant: Lisbon 1-6 with the paving-stone and plaza classes at their earlier
# (pre-revision) properties, as the campus still carries (k 1.1, rho*c 1.85 MJ,
# slab 0.25/0.30 m; admittance ~12.2 instead of 17). Published outputs untouched.
set -uo pipefail
R="/media/harshin/data_drive/route based UTCI"; cd "$R"
name=lisbon_paving_old; V="$R/variants/$name"
ALL=(lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6)
mkdir -p "$V/run_output"
for c in "${ALL[@]}"; do
    rsync -a --exclude solweig --exclude 'viz.bak*' --exclude 'route_jos3.bak*' "run_output/$c/" "$V/run_output/$c/"
    python3 - "$V/run_output/$c/osm_ground_materials/ground_material_catalog.json" <<'PY' || exit 1
import json, sys
p = sys.argv[1]
old = json.load(open("run_output/MMC/osm_ground_materials/ground_material_catalog.json"))["materials"]
cat = json.load(open(p))
for k in ("paving_stone_pedestrian", "pedestrian_plaza"):
    if k in cat["materials"]:
        cat["materials"][k] = dict(old[k])
        print(p.split("/")[-3], k, cat["materials"][k]["k"], cat["materials"][k]["C"], cat["materials"][k]["depth"])
json.dump(cat, open(p, "w"), indent=2)
PY
done
export USE_PEDESTRIAN_WIND=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMBA_NUM_THREADS=6
paper_adhoc/gated.sh "variant_$name" 6 env FORCE_SVF=0 nice -n 19 ./paper_variant_runs.sh "$name" 4 "${ALL[@]}" \
    > "$V/variant.log" 2>&1 && echo "== done $(date '+%F %T')" || echo "== FAILED ($V/variant.log)"
