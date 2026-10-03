#!/usr/bin/env bash
# Three lanes of variant runs for the paper's convergence, height and ablation tables.
cd "$(dirname "$0")"
laneA() {
  SENSOR_HEIGHT_M=0.8 SKIP_05=1 SKIP_05A=1 SKIP_05B=1 ./paper_variant_runs.sh height_0p8 4
  SENSOR_HEIGHT_M=1.2 SKIP_05=1 SKIP_05A=1 SKIP_05B=1 ./paper_variant_runs.sh height_1p2 4
}
laneB() {
  FACET_SELECT_EXTRA_ARGS="--n-lw-azimuth 48 --n-lw-elevation 36" SKIP_05=1 ./paper_variant_runs.sh conv_rays48x36 4 lisbon1
  POINT_STRIDE=4 SKIP_05=1 ./paper_variant_runs.sh conv_stride4 4 lisbon1
  MAX_DISTANCE=150 SKIP_05=1 ./paper_variant_runs.sh conv_range150 4 lisbon1
  MAX_DISTANCE=600 SKIP_05=1 ./paper_variant_runs.sh conv_range600 4 lisbon1
  FACET_SELECT_EXTRA_ARGS="--select-all" SKIP_05=1 ./paper_variant_runs.sh conv_fullscene 4 lisbon1
}
laneC() {
  MRT_FACET_EXTRA_ARGS="--facet-temperature-source air" SKIP_05=1 SKIP_05A=1 SKIP_05B=1 ./paper_variant_runs.sh abl_surfaces_at_ta 4
  MRT_FACET_EXTRA_ARGS="--reflected-model global" SKIP_05=1 SKIP_05A=1 SKIP_05B=1 ./paper_variant_runs.sh abl_global_reflected 4
  EB_EXTRA_ARGS="--steady-state" SKIP_05=1 SKIP_05A=1 ./paper_variant_runs.sh abl_steady_state 4
  EB_EXTRA_ARGS="--material-json $PWD/variants_generic_paving_material.json" SKIP_05=1 SKIP_05A=1 ./paper_variant_runs.sh abl_generic_paving 4
}
laneA > variants/laneA.log 2>&1 &
laneB > variants/laneB.log 2>&1 &
laneC > variants/laneC.log 2>&1 &
wait
echo "all lanes done $(date)"
