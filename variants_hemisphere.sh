#!/usr/bin/env bash
# Hemisphere-crown ablation: rebuild each Lisbon scene's vegetation with the
# flat-based hemisphere crown model, then run the whole facet-thermal path
# (sky view included) on the new geometry.
cd "$(dirname "$0")"
root="$PWD/variants/abl_hemisphere_crowns"; inp="$root/input_hemi"
mkdir -p "$inp"
for n in 1 2 3 4 5 6; do
  c="lisbon$n"
  if [ ! -f "$inp/$c/geometry/geometry_build.json" ] || ! grep -q hemisphere "$inp/$c/geometry/geometry_build.json"; then
    rm -rf "$inp/$c"; cp -r "input/$c" "$inp/$c"; rm -f "$inp/$c/geometry/"*.stl
    echo "== building hemisphere crowns for $c ($(date '+%T'))"
    python3 create_lisbon_laz_geometry.py --input-root "$inp" --case "$n" --crown-model hemisphere --overwrite \
      > "$inp/$c/geometry_build_hemisphere.log" 2>&1 || { echo "geometry FAILED for $c"; tail -20 "$inp/$c/geometry_build_hemisphere.log"; exit 1; }
  fi
done
INPUT_ROOT="$inp" FORCE_SVF=1 SKIP_05A=0 ./paper_variant_runs.sh abl_hemisphere_crowns 4
