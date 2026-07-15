#!/usr/bin/env bash
#
# start.sh -- run the whole route_utci pipeline in the correct order,
# EXCEPT the STL geometry build (LiDAR -> buildings/vegetation/ground).
# Those .stl files are assumed to already exist (produced by main.py /
# start_LAZ_to_stl.sh); this script consumes them.
#
# Stage order:
#   OSM   extract pedestrian network            -> pedestrian_network.graphml + path_polylines.pkl
#   05    MRT ray tracing (LEGACY surround)     -> mrt_out/          (path_xyz, times.csv, svf ...)
#   05a   select route-visible thermal facets   -> thermal_out/      (facets + LW view matrix)
#   05b   1D facet surface-energy balance        -> thermal_out/      (facet_T_matrix_K.npy ...)
#   05★   MRT ray tracing (FACET surface temps) -> mrt_facet_out/    (improved longwave)
#   06    visualize MRT network
#   07    visualize UTCI network
#   08    route thermal stress (UTCI-based)
#   09    route thermal stress (JOS-3)
#
# 06-09 are run against the IMPROVED (facet-thermal) MRT results by
# default. Set VIS_MRT_DIR=mrt_out to visualize the legacy results, or
# run this script twice with different OUT_ROOT values to compare.
#
# Usage:
#   ./start.sh                 # uses the defaults below
#   GEOM_DIR=out_full/02_final DATE=2025-07-06 ./start.sh
#   SKIP_OSM=1 ./start.sh      # reuse an existing OSM network
#
# Every stage is skippable via SKIP_<STAGE>=1 (OSM/05/05A/05B/05FACET/06/07/08/09)
# and re-running is safe: completed stages can be skipped individually.

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------
# Configuration (override any of these from the environment)
# ----------------------------------------------------------------------
PY="${PY:-python3}"

# Where the already-built STL geometry lives, and the canonical filenames
GEOM_DIR="${GEOM_DIR:-out_full/02_final}"
BUILDINGS_STL="${BUILDINGS_STL:-$GEOM_DIR/building_final.stl}"
VEGETATION_STL="${VEGETATION_STL:-$GEOM_DIR/vegetation_final.stl}"
GROUND_STL="${GROUND_STL:-$GEOM_DIR/ground_and_water_final.stl}"

# Output directories
OUT_ROOT="${OUT_ROOT:-run_output}"
OSM_DIR="${OSM_DIR:-$OUT_ROOT/osm_paths}"
MRT_DIR="${MRT_DIR:-$OUT_ROOT/mrt_out}"           # legacy-surround MRT
THERMAL_DIR="${THERMAL_DIR:-$OUT_ROOT/thermal_out}"   # 05a + 05b outputs
MRT_FACET_DIR="${MRT_FACET_DIR:-$OUT_ROOT/mrt_facet_out}"  # facet-thermal MRT
VIS_DIR="${VIS_DIR:-$OUT_ROOT/viz}"

# The MRT results the visualizations/route-stress stages consume.
# Default: the IMPROVED facet-thermal results.
VIS_MRT_DIR="${VIS_MRT_DIR:-$MRT_FACET_DIR}"
GRAPHML="${GRAPHML:-$OSM_DIR/pedestrian_network.graphml}"
POLYLINES="${POLYLINES:-$OSM_DIR/path_polylines.pkl}"

# Shared physical / sampling parameters. Values used in radiation MUST be
# consistent between 05, 05a, 05b (the scripts also hard-check counts).
DATE="${DATE:-2025-07-06}"
LAT="${LAT:-25.7560}"
LON="${LON:--80.3770}"
TZ="${TZ:-America/New_York}"
DT_MIN="${DT_MIN:-10}"
DS_PATH="${DS_PATH:-0.25}"
CLOUD="${CLOUD:-0.0}"
K_LAD_DIRECT="${K_LAD_DIRECT:-0.45}"
K_LAD_DIFFUSE="${K_LAD_DIFFUSE:-0.30}"

# 05a / 05b specific
POINT_STRIDE="${POINT_STRIDE:-8}"
MAX_DISTANCE="${MAX_DISTANCE:-300}"
SPINUP_DAYS="${SPINUP_DAYS:-2}"
WIND_SPEED="${WIND_SPEED:-1.5}"

# Downstream microclimate assumptions for 07/08/09
RH_PCT="${RH_PCT:-70}"
WIND_MS="${WIND_MS:-3.1}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
log()  { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
skip() { local v="SKIP_$1"; [ "${!v:-0}" = "1" ]; }

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required input not found: $1" >&2
        echo "       (set the matching path variable or build the STL first)" >&2
        exit 1
    fi
}

log "Checking pre-built STL geometry (NOT regenerated)"
require_file "$BUILDINGS_STL"
require_file "$VEGETATION_STL"
require_file "$GROUND_STL"
echo "  buildings : $BUILDINGS_STL"
echo "  vegetation: $VEGETATION_STL"
echo "  ground    : $GROUND_STL"
mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------------
# OSM pedestrian network
# ----------------------------------------------------------------------
if skip OSM || { [ -f "$GRAPHML" ] && [ -f "$POLYLINES" ] && [ "${FORCE_OSM:-0}" != 1 ]; }; then
    log "OSM network -- skipped (exists or SKIP_OSM=1)"
else
    log "Extracting OSM pedestrian network"
    "$PY" extract_osm_pedestrian_network.py --output-dir "$OSM_DIR"
fi
require_file "$POLYLINES"

# ----------------------------------------------------------------------
# 05  MRT ray tracing -- LEGACY surround (also produces path_xyz + times.csv
#     that 05a/05b depend on)
# ----------------------------------------------------------------------
if skip 05; then
    log "05 legacy MRT -- skipped"
else
    log "05  MRT ray tracing (legacy surround)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$POLYLINES" \
        --output-dir "$MRT_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE"
fi
require_file "$MRT_DIR/path_xyz.npy"
require_file "$MRT_DIR/times.csv"

# ----------------------------------------------------------------------
# 05a  Select route-visible thermal facets + LW view matrix
# ----------------------------------------------------------------------
if skip 05A; then
    log "05a facet selection -- skipped"
else
    log "05a  Selecting route-visible thermal facets"
    "$PY" 05a_thermal_facets_select.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        --point-stride "$POINT_STRIDE" --max-distance "$MAX_DISTANCE"
fi
require_file "$THERMAL_DIR/facets.npz"

# ----------------------------------------------------------------------
# 05b  1D facet surface-energy balance (same weather via times.csv)
# ----------------------------------------------------------------------
if skip 05B; then
    log "05b energy balance -- skipped"
else
    log "05b  Facet 1D surface-energy balance"
    "$PY" 05b_facet_energy_balance.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --facets-dir "$THERMAL_DIR" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        --spinup-days "$SPINUP_DAYS" --wind-speed "$WIND_SPEED" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE"
fi
require_file "$THERMAL_DIR/facet_T_matrix_K.npy"

# ----------------------------------------------------------------------
# 05★  MRT ray tracing again, now consuming the facet surface temperatures
# ----------------------------------------------------------------------
if skip 05FACET; then
    log "05 facet-thermal MRT -- skipped"
else
    log "05  MRT ray tracing (facet surface temperatures)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$POLYLINES" \
        --output-dir "$MRT_FACET_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --facet-thermal-dir "$THERMAL_DIR"
fi
require_file "$VIS_MRT_DIR/tmrt_matrix_C.npy"

echo
echo "Visualization / route-stress stages consume: $VIS_MRT_DIR"
echo "(set VIS_MRT_DIR=$MRT_DIR to use the legacy-surround results instead)"

# ----------------------------------------------------------------------
# 06  Visualize MRT network
# ----------------------------------------------------------------------
if skip 06; then
    log "06 MRT visualization -- skipped"
else
    log "06  Visualizing MRT network"
    "$PY" 06_visualize_mrt_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/mrt"
fi

# ----------------------------------------------------------------------
# 07  Visualize UTCI network
# ----------------------------------------------------------------------
if skip 07; then
    log "07 UTCI visualization -- skipped"
else
    log "07  Visualizing UTCI network"
    "$PY" 07_visualize_utci_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/utci" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS"
fi

# ----------------------------------------------------------------------
# 08  Route thermal stress (UTCI-based)
# ----------------------------------------------------------------------
if skip 08; then
    log "08 route thermal stress (UTCI) -- skipped"
else
    log "08  Route thermal stress (UTCI)"
    "$PY" 08_route_thermal_stress.py \
        --graphml "$GRAPHML" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_utci" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS"
fi

# ----------------------------------------------------------------------
# 09  Route thermal stress (JOS-3)
# ----------------------------------------------------------------------
if skip 09; then
    log "09 route thermal stress (JOS-3) -- skipped"
else
    log "09  Route thermal stress (JOS-3)"
    "$PY" 09_route_thermal_stress_jos3.py \
        --graphml "$GRAPHML" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_jos3" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS"
fi

log "Pipeline complete"
echo "  MRT (legacy)     : $MRT_DIR"
echo "  Facets + temps   : $THERMAL_DIR"
echo "  MRT (facet therm): $MRT_FACET_DIR"
echo "  Visualizations   : $VIS_DIR"
