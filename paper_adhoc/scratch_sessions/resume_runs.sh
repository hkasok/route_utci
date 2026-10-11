#!/usr/bin/env bash
# Resume the T7 generic-paving and T1 route-height variant runs (3 chains in parallel).
set -uo pipefail
P="/media/harshin/data_drive/route based UTCI"
L=/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad
cd "$P"
rm -rf variants/abl_generic_paving/run_output/lisbon4 variants/height_z0p8/run_output/lisbon3

( EB_EXTRA_ARGS="--material-json /tmp/claude-1000/generic_paving_material.json" SKIP_05=1 SKIP_05A=1 \
    ./paper_variant_runs.sh abl_generic_paving 4 lisbon4 lisbon5 lisbon6 \
  && ./paper_variant_compare.sh abl_generic_paving ) > "$L/chain_paving.log" 2>&1 &
pa=$!
( Z_HEIGHT=0.8 FORCE_SVF=1 ./paper_variant_runs.sh height_z0p8 4 lisbon3 lisbon4 lisbon5 lisbon6 \
  && ./paper_variant_compare.sh height_z0p8 ) > "$L/chain_z0p8.log" 2>&1 &
pb=$!
( Z_HEIGHT=1.2 FORCE_SVF=1 ./paper_variant_runs.sh height_z1p2 4 ) > "$L/chain_z1p2.log" 2>&1 &
pc=$!
wait $pa; ra=$?; wait $pb; rb=$?; wait $pc; rc=$?
echo "paving=$ra z0p8=$rb z1p2=$rc ($(date '+%F %T'))"
