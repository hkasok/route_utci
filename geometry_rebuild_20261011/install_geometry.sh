#!/usr/bin/env bash
# Install the rebuilt geometry (11 Oct 2026) and move the superseded outputs
# aside. Nothing is deleted: every replaced directory is renamed or moved into
# backups_georebuild_20261011/.
set -euo pipefail
cd "$(dirname "$0")/.."
B=backups_georebuild_20261011
STAGE=geometry_rebuild_20261011
TAG=pregeom_20261011
mkdir -p "$B/run_output" "$B/input"

echo "== Lisbon geometry"
for n in 1 2 3 4 5 6; do
    c=lisbon$n
    new="$STAGE/input_new/$c/geometry"
    for f in building_final.stl vegetation_final.stl ground_and_water_final.stl geometry_build.json; do
        [ -s "$new/$f" ] || { echo "missing $new/$f"; exit 1; }
    done
    # the terrain is rebuilt from the same returns: it must be byte-identical,
    # or the OSM ground-material map (keyed to terrain faces) would be invalid
    a=$(sha256sum < "$new/ground_and_water_final.stl"); b=$(sha256sum < "input/$c/geometry/ground_and_water_final.stl")
    [ "$a" = "$b" ] || { echo "$c: terrain differs from the installed one"; exit 1; }
    cp -a "input/$c/weather" "$B/input/${c}_weather"
    cp -a "input/$c/config" "$B/input/${c}_config"
    mv "input/$c/geometry" "input/$c/geometry.bak_$TAG"
    mkdir "input/$c/geometry"
    cp -p "$new"/{building_final.stl,vegetation_final.stl,ground_and_water_final.stl,geometry_build.json,building_final_lod1_report.json} \
        "input/$c/geometry/"
    echo "  $c installed"
done

echo "== campus (MMC) buildings; terrain and crowns unchanged"
mv input/MMC/geometry "input/MMC/geometry.bak_$TAG"
mkdir input/MMC/geometry
cp -p "input/MMC/geometry.bak_$TAG"/{vegetation_final.stl,ground_and_water_final.stl} input/MMC/geometry/
cp -p "$STAGE/mmc/building_final.stl" "$STAGE/mmc/building_final_lod1_report.json" input/MMC/geometry/
python3 - <<'EOF'
import json
p = "input/MMC/geometry/geometry_build.json"
old = json.load(open("input/MMC/geometry.bak_pregeom_20261011/geometry_build.json"))
old["rebuilt_buildings"] = "2026-10-11"
old["buildings_model"] = ("03_buildings_to_stl.py --lod1-model segmented --ground-stl "
                          "geometry/ground_and_water_final.stl (one roof per building part, "
                          "courtyards kept, outer walls below the terrain mesh)")
old["buildings_reason"] = ("the block model gave one roof per merged block at its 90th-percentile "
                           "height (median +3.3 m, MAE 5.3 m against the roof returns); "
                           "segmented: MAE 0.30 m")
old["stages"] = [s if not s.startswith("03_") else
                 "03_buildings_to_stl.py --lod1-model segmented --raster-res 0.5 --height-tolerance 3.0 "
                 "--min-hole-area 4 --min-part-area 6 --simplify-tolerance 0.75 --ground-stl ground_and_water_final.stl"
                 for s in old.get("stages", [])]
json.dump(old, open(p, "w"), indent=1)
print("  MMC geometry_build.json updated")
EOF

echo "== move superseded run outputs"
for c in lisbon1 lisbon2 lisbon3 lisbon4 lisbon5 lisbon6 MMC; do
    [ -d "run_output/$c" ] && mv "run_output/$c" "$B/run_output/$c"
done
for f in lisbon_sensor_validation impact_uniform solweig_lisbon_pooled.json compare_run.log; do
    [ -e "run_output/$f" ] && mv "run_output/$f" "$B/run_output/$f"
done
# The campus run reuses its cached OSM routing network (no re-download).
mkdir -p run_output/MMC
cp -a "$B/run_output/MMC/osm_paths" run_output/MMC/
for root in variants facade_tests; do
    for d in "$root"/*/; do
        [ -d "$d/run_output" ] && mv "$d/run_output" "${d%/}/run_output.$TAG"
    done
done
[ -d variants/abl_hemisphere_crowns/input_hemi ] && \
    mv variants/abl_hemisphere_crowns/input_hemi "variants/abl_hemisphere_crowns/input_hemi.$TAG"
echo "== installed $(date '+%F %T')"
