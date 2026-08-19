#!/usr/bin/env bash
#
# start.sh -- run the route_utci pipeline, optionally starting from any step.
#
# ============================================================================
#  START FROM ANY EXECUTABLE STEP: ./start.sh <N>  (default N = 2)
# ============================================================================
#  N  STEP                              PRODUCES
#  1  Problem selection (browser UI)    input/<case> -> run_output/<case>
#  2  OSM data + ground materials      optional graph + classified terrain
#  3  Facet-thermal MRT (merged)        prep -> 05a -> 05b -> improved MRT
#  4  Optional radiation/microclimate   full surfaces -> 3-D Ta/velocity/MRT
#  5  Visualizations (stages 06, 07)    MRT + UTCI maps / animations
#  6  Route thermal stress (08-10)      UTCI + JOS-3 + optional comparison
#
#  Examples:
#     ./start.sh          # start at step 2 (OSM) -- assumes STL already built
#     INPUT_CASE_DIR=input/MMC OUTPUT_CASE_DIR=run_output/MMC ./start.sh 2
#     ./start.sh 3        # re-run facet-thermal MRT and everything after it
#     WITH_MICROCLIMATE=1 ./start.sh 4  # solve 3-D air fields + later stages
#     ./start.sh 5        # only (re)build visualizations and route stress
#     ./start.sh 6        # only re-run route stress (08, 09, optional compare)
#     ./start.sh 8        # legacy alias for current route-stress step 6
#     WITH_BASELINE=1 ./start.sh 3  # also compute legacy-surround MRT
#
#  Starting at step N runs N, N+1, ... to the end. Steps before N are assumed
#  already done; the script checks their outputs exist and stops with a clear
#  message if something required is missing. Step 1 is case selection in the
#  UI, not an executable shell stage. Geometry and route preparation remain
#  separate project utilities and are never launched by this workflow.
#
#  You can still force-skip an individual stage with SKIP_<NAME>=1
#  (OSM/OSM_MATERIALS/05/05A/05B/05FACET/MICROCLIMATE/
#  URBAN_RADIATION/MICROCLIMATE_05B/MICROCLIMATE_05FACET/06/07/08/09/10), e.g.
#  SKIP_07=1 ./start.sh 5. Stage 05 is preparation-only unless
#  WITH_BASELINE=1 requests the legacy-surround MRT result as well.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
Usage: bash start.sh [STEP]

Run STEP and every later TREC-Route pipeline step (default: STEP=2).
Step 1 is input/output problem-case selection in the browser UI.
Geometry and route generation are external case-preparation utilities.
  2  OSM data and ground materials (pedestrian graph only when case requires it)
  3  Facet-thermal MRT (05 --prep-only -> 05a -> 05b -> facet MRT)
  4  OPTIONAL full-surface radiation, 3-D air fields, and thermal recoupling
  5  MRT and UTCI visualizations (06, 07)
  6  Route stress and optional comparison (08, 09, 10)

Important controls:
  INPUT_CASE_DIR=DIR      self-contained input case (default: input/MMC)
  OUTPUT_CASE_DIR=DIR     selected result case (default: run_output/MMC)
  WITH_BASELINE=1        additionally compute legacy-surround MRT in mrt_out
  WITH_MICROCLIMATE=1    run optional full-surface radiation + 3-D Ta/velocity
  WITH_URBAN_RADIATION=0 disable full-surface radiation within that optional step
  URBAN_RADIATION_CONFIG=F  full-surface radiation configuration JSON
  MICROCLIMATE_CONFIG=F  diagnostic solver configuration JSON
  USE_MICROCLIMATE_FIELDS=auto|0|1  reuse an existing solved field downstream
  ROUTES_DIR=DIR          override the selected case's routes/ directory
  SVF_CACHE_DIR=DIR      exact-match static sky-view cache directory
  FORCE_SVF=1            rebuild the preparation SVF cache
  SKIP_<FLAG>=1          skip OSM, OSM_MATERIALS, 05, 05A, 05B,
                         05FACET, MICROCLIMATE, URBAN_RADIATION, MICROCLIMATE_05B,
                         MICROCLIMATE_05FACET, 06, 07, 08, 09, or 10

All existing model/environment overrides remain supported, including DATE,
DEPARTURE_HOUR, WALKING_SPEED_MS, WEATHER_CSV, SUBJECT_PROFILE,
LOCAL_ORIGIN_X/Y, PROJECT_CRS, VIS_MRT_DIR, and SOLWEIG_COMPARE.
EOF
    exit 0
fi

REQUESTED_STEP="${1:-2}"
START_STEP="$REQUESTED_STEP"
if [ "$START_STEP" = "8" ]; then
    echo "NOTE: legacy step 8 maps to current route-stress step 6."
    START_STEP=6
fi
case "$START_STEP" in 2|3|4|5|6) ;; *)
    echo "ERROR: executable step must be 2-6 (got '$START_STEP')." >&2
    echo "       Step 1 is problem selection in the browser UI." >&2
    exit 1 ;; esac

# ############################################################################
# #  CRITICAL INPUTS  --  the settings you are most likely to change.        #
# #  Every value can also be overridden from the environment, e.g.           #
# #     DATE=2025-08-01 DEPARTURE_HOUR=15.0 ./start.sh 3                      #
# ############################################################################

PY="${PY:-python3}"

# ---- STEP 1: PROBLEM CASE SELECTION ----------------------------------------
# The browser UI sets these two directories. Command-line runs may set them
# directly. All physical inputs are resolved from input/<case>/case.json and
# every generated product is rooted below the selected output case.
CASE_NAME="${CASE_NAME:-MMC}"
INPUT_CASE_DIR="${INPUT_CASE_DIR:-$PWD/input/$CASE_NAME}"
OUTPUT_CASE_DIR="${OUTPUT_CASE_DIR:-$PWD/run_output/$CASE_NAME}"
CASE_CONFIG="${CASE_CONFIG:-$INPUT_CASE_DIR/case.json}"
if [ ! -f "$CASE_CONFIG" ]; then
    echo "ERROR: selected input case has no case.json: $CASE_CONFIG" >&2
    echo "       Select a valid directory below $PWD/input in the UI." >&2
    exit 1
fi
if ! CASE_ASSIGNMENTS="$("$PY" case_config.py "$INPUT_CASE_DIR" --shell)"; then
    echo "ERROR: selected input case is invalid: $INPUT_CASE_DIR" >&2
    exit 1
fi
eval "$CASE_ASSIGNMENTS"

# ---- WHEN: day of the solar run and time of the walk -----------------------
DATE="${DATE:-$CASE_DATE}"
DEPARTURE_HOUR="${DEPARTURE_HOUR:-$CASE_DEPARTURE_HOUR}"
WALKING_SPEED_MS="${WALKING_SPEED_MS:-$CASE_WALKING_SPEED_MS}"
TIMING_MODE="${TIMING_MODE:-$CASE_TIMING_MODE}"
ROUTING_NETWORK_REQUIRED="${ROUTING_NETWORK_REQUIRED:-$CASE_ROUTING_NETWORK_REQUIRED}"
case "$ROUTING_NETWORK_REQUIRED" in 0|1) ;; *)
    echo "ERROR: ROUTING_NETWORK_REQUIRED must be 0 or 1" >&2; exit 1 ;; esac
# JOS-3 virtual subject for stage 09 (stage 8). Presets (cited in
# subject_profiles.py): healthy_adult, healthy_adult_female, child,
# elderly_male, elderly_female, obese_adult, acclimatized_adult.
# Empty = JOS-3 default healthy adult.
SUBJECT_PROFILE="${SUBJECT_PROFILE:-$CASE_SUBJECT_PROFILE}"

# ---- WHERE: site location (used for sun position) --------------------------
LAT="${LAT:-$CASE_LAT}"
LON="${LON:-$CASE_LON}"
TZ="${TZ:-$CASE_TZ}"
OSM_BBOX="${OSM_BBOX:-$CASE_OSM_BBOX}"

# ---- ROUTES + GEOREFERENCING ------------------------------------------------
# Route generation is a case-preparation activity outside the general UI.
ROUTES_DIR="${ROUTES_DIR:-$CASE_ROUTES_DIR}"
# Origin shift applied when the OSM network was built (must match
# extract_osm_pedestrian_network.py) so exported routes get TRUE lat/lon.
LOCAL_ORIGIN_X="${LOCAL_ORIGIN_X:-$CASE_LOCAL_ORIGIN_X}"
LOCAL_ORIGIN_Y="${LOCAL_ORIGIN_Y:-$CASE_LOCAL_ORIGIN_Y}"
PROJECT_CRS="${PROJECT_CRS:-$CASE_PROJECT_CRS}"

# ---- WEATHER: real time series (preferred) or parametric fallback ----------
# The weather CSV now drives the WHOLE pipeline (columns: hour|time,
# air_temp_C, rh_pct, wind_ms): the MRT stages (05/05*) read air temperature
# and RH from it -- RH feeds the humidity-dependent (Prata) sky longwave, and
# air temperature drives Tmrt -- so 05b (which inherits both via times.csv)
# and the route stages 08/09 all use one identical, real forcing. Previously
# only 08/09 used the CSV while the MRT ran on parametric weather, which made
# the MRT longwave inconsistent with the route stress and with any external
# (e.g. SOLWEIG) comparison. Point WEATHER_CSV at weather_miami_july06.csv to
# match the July 6 solar run exactly.
WEATHER_CSV="${WEATHER_CSV:-$CASE_WEATHER_CSV}"
RH_PCT="${RH_PCT:-$CASE_RH_PCT}"
WIND_MS="${WIND_MS:-$CASE_WIND_MS}"
CLOUD="${CLOUD:-$CASE_CLOUD}"
                                      #   (0 = clear sky; keep 05 and 05b equal)

# ---- PREBUILT GEOMETRY ------------------------------------------------------
# Geometry construction remains available in the project folder, but the UI
# consumes only these prepared case inputs.
GEOM_DIR="${GEOM_DIR:-$(dirname "$CASE_BUILDINGS_STL")}"
BUILDINGS_STL="${BUILDINGS_STL:-$CASE_BUILDINGS_STL}"
VEGETATION_STL="${VEGETATION_STL:-$CASE_VEGETATION_STL}"
GROUND_STL="${GROUND_STL:-$CASE_GROUND_STL}"

# ############################################################################
# #  SECONDARY SETTINGS  --  sensible defaults; change only if you know why. #
# ############################################################################

# Output directories
OUT_ROOT="${OUT_ROOT:-$OUTPUT_CASE_DIR}"
OSM_DIR="${OSM_DIR:-$OUT_ROOT/osm_paths}"
OSM_EDGE_FEATURES="${OSM_EDGE_FEATURES:-$OSM_DIR/pedestrian_edges.geojson}"
OSM_COMPLETE_CACHE="${OSM_COMPLETE_CACHE:-$CASE_OSM_COMPLETE_FILE}"
OSM_COMPLETE_INPUT="${OSM_COMPLETE_INPUT:-}"
OSM_GROUND_FEATURES="${OSM_GROUND_FEATURES:-$OSM_COMPLETE_CACHE}"
OSM_GROUND_MATERIALS_ENABLED="${OSM_GROUND_MATERIALS_ENABLED:-1}"
OSM_GROUND_CONFIG="${OSM_GROUND_CONFIG:-$CASE_OSM_GROUND_CONFIG}"
OSM_GROUND_MATERIAL_DIR="${OSM_GROUND_MATERIAL_DIR:-$OUT_ROOT/osm_ground_materials}"
OSM_GROUND_OVERRIDES="${OSM_GROUND_OVERRIDES:-}"
RADIANT_FLUX_CONFIG="${RADIANT_FLUX_CONFIG:-$CASE_RADIANT_FLUX_CONFIG}"
RADIATION_FORCING_CONFIG="${RADIATION_FORCING_CONFIG:-$CASE_RADIATION_FORCING_CONFIG}"
MRT_DIR="${MRT_DIR:-$OUT_ROOT/mrt_out}"                   # prep + optional legacy MRT
THERMAL_DIR="${THERMAL_DIR:-$OUT_ROOT/thermal_out}"       # 05a + 05b outputs
MRT_FACET_DIR="${MRT_FACET_DIR:-$OUT_ROOT/mrt_facet_out}" # facet-thermal MRT
MICROCLIMATE_DIR="${MICROCLIMATE_DIR:-$OUT_ROOT/microclimate}"
MICROCLIMATE_CONFIG="${MICROCLIMATE_CONFIG:-${CASE_MICROCLIMATE_CONFIG:-$PWD/microclimate_config.json}}"
WITH_MICROCLIMATE="${WITH_MICROCLIMATE:-0}"
URBAN_RADIATION_DIR="${URBAN_RADIATION_DIR:-$OUT_ROOT/urban_radiation}"
URBAN_RADIATION_CONFIG="${URBAN_RADIATION_CONFIG:-${CASE_URBAN_RADIATION_CONFIG:-$PWD/urban_radiation_config.json}}"
WITH_URBAN_RADIATION="${WITH_URBAN_RADIATION:-$WITH_MICROCLIMATE}"
USE_MICROCLIMATE_FIELDS="${USE_MICROCLIMATE_FIELDS:-auto}"
MICROCLIMATE_COUPLING_ITERATIONS="${MICROCLIMATE_COUPLING_ITERATIONS:-1}"
VIS_DIR="${VIS_DIR:-$OUT_ROOT/viz}"
SVF_CACHE_DIR="${SVF_CACHE_DIR:-$OUT_ROOT/svf_cache}"
WITH_BASELINE="${WITH_BASELINE:-0}"
SOLWEIG_COMPARE="${SOLWEIG_COMPARE:-0}"
SOLWEIG_COMPARE_OUTPUT_DIR="${SOLWEIG_COMPARE_OUTPUT_DIR:-$VIS_DIR/compare_solweig}"

case "$WITH_MICROCLIMATE" in 0|1) ;; *)
    echo "ERROR: WITH_MICROCLIMATE must be 0 or 1" >&2; exit 1 ;; esac
case "$WITH_URBAN_RADIATION" in 0|1) ;; *)
    echo "ERROR: WITH_URBAN_RADIATION must be 0 or 1" >&2; exit 1 ;; esac
case "$USE_MICROCLIMATE_FIELDS" in auto|0|1) ;; *)
    echo "ERROR: USE_MICROCLIMATE_FIELDS must be auto, 0, or 1" >&2; exit 1 ;; esac
case "$MICROCLIMATE_COUPLING_ITERATIONS" in
    ''|*[!0-9]*|0) echo "ERROR: MICROCLIMATE_COUPLING_ITERATIONS must be a positive integer" >&2; exit 1 ;;
esac

# Which MRT results the visualization / route stages consume
# (default: the IMPROVED facet-thermal results; set to $MRT_DIR for legacy)
VIS_MRT_DIR="${VIS_MRT_DIR:-$MRT_FACET_DIR}"
LEGACY_RADIANT_FLUX_ARG=()
if [ "$VIS_MRT_DIR" = "$MRT_DIR" ]; then
    # Only record a second (legacy-surround) contribution archive when the
    # user explicitly chooses that MRT result for downstream route analysis.
    LEGACY_RADIANT_FLUX_ARG=(--radiant-flux-config "$RADIANT_FLUX_CONFIG")
fi
GRAPHML="${GRAPHML:-$OSM_DIR/pedestrian_network.graphml}"
POLYLINES="${POLYLINES:-$OSM_DIR/path_polylines.pkl}"      # FULL network (from OSM)
# Derived from the authoritative CSV route files by generate_route.py. Stage
# 05 consumes this compatibility pickle; stages 08/09 consume ROUTES_DIR.
ROUTE_POLYLINES="${ROUTE_POLYLINES:-$ROUTES_DIR/route_polylines.pkl}"

# Radiation / sampling parameters -- MUST stay consistent across 05, 05a, 05b
DT_MIN="${DT_MIN:-10}"
DS_PATH="${DS_PATH:-0.25}"
K_LAD_DIRECT="${K_LAD_DIRECT:-0.45}"
K_LAD_DIFFUSE="${K_LAD_DIFFUSE:-0.30}"
# Clear-sky longwave emissivity model, shared by the MRT passes (05) and the
# surface energy balance (05b) so both see one identical sky. 'prata' is
# humidity-dependent (~0.89 in humid Miami vs the old constant 0.78) and
# raises downwelling longwave; 'constant' restores the old 0.78 behaviour.
CLEAR_SKY_MODEL="${CLEAR_SKY_MODEL:-prata}"
# Height above local ground at which MRT is sampled (pedestrian body height),
# meters. Passed to BOTH MRT passes (steps 3 and 6) so it stays consistent;
# it is baked into path_xyz.npy, which every downstream stage (05a/05b/08/09)
# reads, so this single value propagates through the whole pipeline.
# 1.1 m = ISO 7726 / UTCI standing-adult center-of-gravity convention.
Z_HEIGHT="${Z_HEIGHT:-1.1}"

# Facet pipeline (05a / 05b)
POINT_STRIDE="${POINT_STRIDE:-8}"
MAX_DISTANCE="${MAX_DISTANCE:-300}"
SPINUP_DAYS="${SPINUP_DAYS:-2}"
MAXIMUM_SPINUP_DAYS="${MAXIMUM_SPINUP_DAYS:-10}"
SPINUP_TOLERANCE_K="${SPINUP_TOLERANCE_K:-0.10}"
WIND_SPEED="${WIND_SPEED:-1.5}"       # near-surface wind for 05b convection
SURFACE_WIND_SOURCE="${SURFACE_WIND_SOURCE:-times_csv}"
SURFACE_CONVECTION_MODEL="${SURFACE_CONVECTION_MODEL:-mcadams}"
SURFACE_LATENT_HEAT_MODEL="${SURFACE_LATENT_HEAT_MODEL:-equilibrium}"

# Optional, case-specific atmospheric radiation forcing.  The numbered/UI
# workflow remains generic: the selected case manifest decides whether a
# forcing configuration is present.  An absent entry preserves clear-sky plus
# scalar CLOUD behavior.
RADIATION_FORCING_ARG=()
if [ -n "$RADIATION_FORCING_CONFIG" ]; then
    if [ ! -f "$RADIATION_FORCING_CONFIG" ]; then
        echo "ERROR: selected case radiation forcing config is missing: $RADIATION_FORCING_CONFIG" >&2
        exit 1
    fi
    RADIATION_FORCING_ARG=(--radiation-forcing-config "$RADIATION_FORCING_CONFIG")
fi

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log()  { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
skip() { local v="SKIP_$1"; [ "${!v:-0}" = "1" ]; }
active() { [ "$START_STEP" -le "$1" ]; }   # true if step N is at/after start

# Machine-readable progress events are emitted only for the browser UI.  A
# normal terminal run remains unchanged.  Percentages mark completed work
# within a top-level step; the UI also refines the expensive numerical loops
# from their real batch/time-step counters.
progress_event() {
    if [ "${TREC_PROGRESS:-0}" = "1" ]; then
        printf '[trec_progress] step=%s percent=%s state=%s message=%s\n' \
            "$1" "$2" "$3" "$4"
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required input not found: $1" >&2
        echo "       Check the selected case manifest or run from executable step ${2:-2}." >&2
        exit 1
    fi
}

require_routes() {
    if [ ! -d "$ROUTES_DIR" ] || ! compgen -G "$ROUTES_DIR/route_*.csv" >/dev/null; then
        echo "ERROR: no route input files found in: $ROUTES_DIR" >&2
        echo "       Routes are prepared case inputs, not a UI workflow step." >&2
        echo "       Populate the case routes/ folder using the documented contract." >&2
        exit 1
    fi
    require_file "$ROUTES_DIR/routes_index.json" 2
    require_file "$ROUTE_POLYLINES" 2
}

# Assemble the optional weather argument once
if [ -f "$WEATHER_CSV" ]; then
    WEATHER_ARG=(--weather-csv "$WEATHER_CSV" --require-weather-csv)
    WEATHER_NOTE="real series: $WEATHER_CSV"
else
    echo "FATAL: weather CSV not found: $WEATHER_CSV" >&2; exit 1
fi

log "route_utci pipeline -- starting at step $START_STEP"
cat <<EOF
  Input case       : $CASE_ID_FROM_MANIFEST ($INPUT_CASE_DIR)
  Site             : $CASE_SITE_NAME
  Output case      : $OUTPUT_CASE_DIR
  Date (solar)     : $DATE
  Departure hour   : $DEPARTURE_HOUR:00
  Route timing     : $TIMING_MODE
  Routing graph    : $([ "$ROUTING_NETWORK_REQUIRED" = 1 ] && echo required || echo not required)
  Walking speed    : $WALKING_SPEED_MS m/s (fallback when a route has no recorded schedule)
  MRT sample height: $Z_HEIGHT m above ground
  Location         : lat $LAT, lon $LON ($TZ)
  Weather          : $WEATHER_NOTE
  Geometry (STL)   : $GEOM_DIR
  Output root      : $OUT_ROOT
EOF
mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------------------
# STEP 1 is problem selection in the browser UI. Geometry is a prepared input;
# it is deliberately not an executable pipeline stage.
# ----------------------------------------------------------------------------
if active 2; then
    log "Checking selected case inputs"
    require_file "$BUILDINGS_STL" 2
    require_file "$VEGETATION_STL" 2
    require_file "$GROUND_STL" 2
    require_routes
    echo "  buildings : $BUILDINGS_STL"
    echo "  vegetation: $VEGETATION_STL"
    echo "  ground    : $GROUND_STL"
    echo "  routes    : $ROUTES_DIR"
fi

# ----------------------------------------------------------------------------
# STEP 2 -- OSM pedestrian network
# ----------------------------------------------------------------------------
read -r -a OSM_BBOX_VALUES <<< "$OSM_BBOX"
if [ "${#OSM_BBOX_VALUES[@]}" -ne 4 ]; then
    echo "ERROR: case OSM bbox must contain left bottom right top: $OSM_BBOX" >&2
    exit 1
fi
if active 2; then
    progress_event 2 0 running "Preparing OSM and material surfaces"
fi
if active 2 && [ "$ROUTING_NETWORK_REQUIRED" = 0 ]; then
    log "STEP 2  Pedestrian-network download not required for prepared experimental routes"
    echo "  Using authoritative case routes: $ROUTE_POLYLINES"
    echo "  Complete cached OSM features remain active for ground materials."
    progress_event 2 40 running "Prepared experimental routes ready; network download not required"
elif active 2 && ! skip OSM; then
    if [ -f "$GRAPHML" ] && [ -f "$POLYLINES" ] && [ "${FORCE_OSM:-0}" != 1 ]; then
        log "STEP 2  OSM network -- already present, skipped (FORCE_OSM=1 to rebuild)"
    else
        log "STEP 2  Extracting OSM pedestrian network"
        "$PY" extract_osm_pedestrian_network.py \
            --output-dir "$OSM_DIR" \
            --bbox "${OSM_BBOX_VALUES[@]}" \
            --target-crs "$PROJECT_CRS" \
            --local-origin-x "$LOCAL_ORIGIN_X" \
            --local-origin-y "$LOCAL_ORIGIN_Y"
    fi
    progress_event 2 20 running "Pedestrian network ready"
    progress_event 2 40 running "Pedestrian network ready; routes remain external inputs"
elif active 2; then
    progress_event 2 40 running "Routing-network work skipped"
fi
if active 3 && { ! skip 05 || ! skip 05FACET; }; then
    require_routes
fi

# ----------------------------------------------------------------------------
# PARALLEL OSM SURFACE BRANCH -- material subdivision only.
# This reads, but never modifies, the graph or external route-input files. It
# assigns materials to existing terrain face IDs.
# ----------------------------------------------------------------------------
GROUND_MATERIAL_ARG=()
if [ "$OSM_GROUND_MATERIALS_ENABLED" = "1" ]; then
    if active 2 && ! skip OSM_MATERIALS; then
        log "STEP 2  OSM ground-material subdivision (routing unchanged)"
        require_routes
        require_file "$OSM_GROUND_CONFIG" 2
        COMPLETE_INPUT_ARG=()
        [ -n "$OSM_COMPLETE_INPUT" ] && COMPLETE_INPUT_ARG=(--input-file "$OSM_COMPLETE_INPUT")
        FORCE_COMPLETE_ARG=()
        [ "${FORCE_OSM_COMPLETE:-0}" = "1" ] && FORCE_COMPLETE_ARG=(--force-download)
        "$PY" download_osm_complete_features.py \
            --ground-mesh "$GROUND_STL" \
            --config "$OSM_GROUND_CONFIG" \
            --output-file "$OSM_COMPLETE_CACHE" \
            --local-origin-x "$LOCAL_ORIGIN_X" \
            --local-origin-y "$LOCAL_ORIGIN_Y" \
            "${COMPLETE_INPUT_ARG[@]}" \
            "${FORCE_COMPLETE_ARG[@]}"
        progress_event 2 55 running "Complete OSM feature cache ready"
        require_file "$OSM_GROUND_FEATURES" 2
        OVERRIDE_ARG=()
        [ -n "$OSM_GROUND_OVERRIDES" ] && OVERRIDE_ARG=(--overrides "$OSM_GROUND_OVERRIDES")
        PROTECTED_ROUTE_ARTIFACTS=("$ROUTE_POLYLINES")
        if [ "$ROUTING_NETWORK_REQUIRED" = 1 ]; then
            require_file "$GRAPHML" 2
            PROTECTED_ROUTE_ARTIFACTS=("$GRAPHML" "$ROUTE_POLYLINES")
        fi
        ROUTE_HASH_BEFORE="$(sha256sum "${PROTECTED_ROUTE_ARTIFACTS[@]}")"
        "$PY" prepare_osm_ground_materials.py \
            --osm-features "$OSM_GROUND_FEATURES" \
            --osm-layer raw_complete_osm_features \
            --ground-mesh "$GROUND_STL" \
            --output-dir "$OSM_GROUND_MATERIAL_DIR" \
            --config "$OSM_GROUND_CONFIG" \
            --routes-pkl "$ROUTE_POLYLINES" \
            --buildings-mesh "$BUILDINGS_STL" \
            --vegetation-mesh "$VEGETATION_STL" \
            --local-origin-x "$LOCAL_ORIGIN_X" \
            --local-origin-y "$LOCAL_ORIGIN_Y" \
            "${OVERRIDE_ARG[@]}"
        progress_event 2 95 running "Terrain materials partitioned"
        ROUTE_HASH_AFTER="$(sha256sum "${PROTECTED_ROUTE_ARTIFACTS[@]}")"
        if [ "$ROUTE_HASH_BEFORE" != "$ROUTE_HASH_AFTER" ]; then
            echo "FATAL: OSM surface processing changed a protected routing artifact" >&2
            exit 1
        fi
        printf '%s\n' "$ROUTE_HASH_AFTER" > \
            "$OSM_GROUND_MATERIAL_DIR/route_artifact_integrity.sha256"
        echo "  Protected routing/route-selection artifacts unchanged"
    fi
    if active 2 && skip OSM_MATERIALS; then
        echo "  OSM ground-material subdivision skipped (SKIP_OSM_MATERIALS=1)"
    else
        require_file "$OSM_GROUND_MATERIAL_DIR/ground_face_materials.npz" 2
        require_file "$OSM_GROUND_MATERIAL_DIR/ground_material_catalog.json" 2
        GROUND_MATERIAL_ARG=(--ground-material-dir "$OSM_GROUND_MATERIAL_DIR")
    fi
else
    echo "  OSM ground materials disabled: uniform-ground backward-compatible mode"
fi
if active 2; then
    if skip OSM && skip OSM_MATERIALS; then
        progress_event 2 100 skipped "OSM network and material work skipped"
    else
        progress_event 2 100 done "OSM routes and ground materials ready"
    fi
fi

# ----------------------------------------------------------------------------
# STEP 3 -- merged facet-thermal MRT workflow.
# Stage 05 first prepares only path_xyz, times.csv and static SVF. Stages 05a
# and 05b consume those cheap byproducts, then the final stage-05 invocation
# reuses the exact-match SVF cache while computing the authoritative MRT.
# ----------------------------------------------------------------------------
SVF_FORCE_ARG=()
[ "${FORCE_SVF:-0}" = "1" ] && SVF_FORCE_ARG=(--force-svf)
if active 3 && ! skip 05; then
    progress_event 3 0 running "Loading MRT geometry and route receptors"
    log "STEP 3  Facet-thermal MRT -- preparation (stage 05 --prep-only)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$ROUTE_POLYLINES" \
        --output-dir "$MRT_DIR" \
        --prep-only --svf-cache "$SVF_CACHE_DIR" "${SVF_FORCE_ARG[@]}" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --z-height "$Z_HEIGHT" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
        "${RADIATION_FORCING_ARG[@]}" \
        "${WEATHER_ARG[@]}"
    progress_event 3 20 running "Route, forcing, and sky-view preparation complete"
elif active 3; then
    progress_event 3 20 running "MRT preparation skipped; using existing products"
fi
if active 3 && ! skip 05A; then
    require_file "$MRT_DIR/path_xyz.npy" 3
    require_file "$MRT_DIR/times.csv" 3
fi

# ----------------------------------------------------------------------------
# STEP 3b -- Facet selection + LW view matrix (stage 05a)
# ----------------------------------------------------------------------------
if active 3 && ! skip 05A; then
    log "STEP 3  Selecting route-visible thermal facets (stage 05a)"
    "$PY" 05a_thermal_facets_select.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        "${GROUND_MATERIAL_ARG[@]}" \
        --point-stride "$POINT_STRIDE" --max-distance "$MAX_DISTANCE"
    progress_event 3 45 running "Route-visible facets and view matrix complete"
elif active 3; then
    progress_event 3 45 running "Facet selection skipped; using existing products"
fi
if active 3 && ! skip 05B; then require_file "$THERMAL_DIR/facets.npz" 3; fi

# ----------------------------------------------------------------------------
# STEP 3c -- Facet 1D surface-energy balance (stage 05b).
# Same weather as preparation (reads times.csv); keep CLOUD equal.
# ----------------------------------------------------------------------------
if active 3 && ! skip 05B; then
    log "STEP 3  Facet 1D surface-energy balance (stage 05b)"
    "$PY" 05b_facet_energy_balance.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --facets-dir "$THERMAL_DIR" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        "${GROUND_MATERIAL_ARG[@]}" \
        --spinup-days "$SPINUP_DAYS" \
        --maximum-spinup-days "$MAXIMUM_SPINUP_DAYS" \
        --spinup-convergence-tolerance-k "$SPINUP_TOLERANCE_K" \
        --wind-speed "$WIND_SPEED" --wind-source "$SURFACE_WIND_SOURCE" \
        --convection-model "$SURFACE_CONVECTION_MODEL" \
        --latent-heat-model "$SURFACE_LATENT_HEAT_MODEL" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL"
    progress_event 3 68 running "Surface temperatures and radiosity complete"
elif active 3; then
    progress_event 3 68 running "Facet energy balance skipped; using existing products"
fi
if active 3 && ! skip 05FACET; then
    require_file "$THERMAL_DIR/facet_T_matrix_K.npy" 3
fi

# ----------------------------------------------------------------------------
# STEP 3d -- authoritative MRT consuming facet surface temperatures.
# ----------------------------------------------------------------------------
if active 3 && ! skip 05FACET; then
    progress_event 3 68 running "Loading facet-thermal MRT calculation"
    log "STEP 3  Facet-thermal MRT ray tracing (stage 05*)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$ROUTE_POLYLINES" \
        --output-dir "$MRT_FACET_DIR" \
        --svf-cache "$SVF_CACHE_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --z-height "$Z_HEIGHT" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
        --facet-thermal-dir "$THERMAL_DIR" \
        --radiant-flux-config "$RADIANT_FLUX_CONFIG" \
        "${RADIATION_FORCING_ARG[@]}" \
        "${WEATHER_ARG[@]}"
    progress_event 3 95 running "Facet-thermal MRT complete; preparing diagnostics"
elif active 3; then
    progress_event 3 95 running "Facet-thermal MRT calculation skipped"
fi
if active 3 && ! skip 05FACET && [ "$OSM_GROUND_MATERIALS_ENABLED" = "1" ]; then
    require_file "$MRT_FACET_DIR/radiant_flux_contributions.npz" 3
    log "STEP 3  Route-point ground-material diagnostics"
    "$PY" route_ground_material_diagnostics.py \
        --mrt-dir "$MRT_FACET_DIR" \
        --thermal-dir "$THERMAL_DIR" \
        --ground-mesh "$GROUND_STL" \
        --ground-material-dir "$OSM_GROUND_MATERIAL_DIR" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed "$WALKING_SPEED_MS" \
        --output "$OSM_GROUND_MATERIAL_DIR/route_point_ground_material_diagnostics.csv"
fi

# Optional baseline: retain the old full legacy-surround product without
# paying for it during normal runs. It shares the same static SVF cache.
if active 3 && ! skip 05 && [ "$WITH_BASELINE" = "1" ]; then
    progress_event 3 96 running "Computing opt-in legacy MRT baseline"
    log "STEP 3  Optional legacy-surround MRT baseline (WITH_BASELINE=1)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$ROUTE_POLYLINES" \
        --output-dir "$MRT_DIR" \
        --svf-cache "$SVF_CACHE_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --z-height "$Z_HEIGHT" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
        "${LEGACY_RADIANT_FLUX_ARG[@]}" \
        "${RADIATION_FORCING_ARG[@]}" \
        "${WEATHER_ARG[@]}"
fi
if active 3; then
    if skip 05 && skip 05A && skip 05B && skip 05FACET; then
        progress_event 3 100 skipped "Facet-thermal MRT workflow skipped"
    else
        progress_event 3 100 done "Facet-thermal MRT workflow complete"
    fi
fi
# ----------------------------------------------------------------------------
# STEP 4 -- OPTIONAL diagnostic microclimate enhancement.
#
# The baseline stage-3 solution remains the default. When enabled, each
# staggered coupling iteration solves a 3-D temperature/vector-velocity field
# from the current facet sensible heat, then reruns 05b with local facet Ta and
# speed. When enabled, 05d first solves radiation and 3-to-5-node conduction on
# every urban facet using an OSM-boundary-conforming ground mesh. The final MRT
# pass uses local receptor Ta. This is deliberately a
# separate UI step so the inexpensive established approximation remains usable.
# ----------------------------------------------------------------------------
if active 4; then
    if [ "$WITH_MICROCLIMATE" != "1" ]; then
        log "STEP 4  Optional microclimate enhancement -- not requested"
        echo "  The 3-D field is not recomputed. Auto mode may reuse an exact-context cache;"
        echo "  otherwise spatially uniform WeatherProvider Ta/wind remain active."
        echo "  Run WITH_MICROCLIMATE=1 bash start.sh 4 or use the UI step-4 button."
        progress_event 4 100 skipped "Optional microclimate enhancement not requested"
    elif skip MICROCLIMATE; then
        progress_event 4 100 skipped "Optional microclimate solver skipped"
    else
        require_file "$MICROCLIMATE_CONFIG" 4
        require_file "$THERMAL_DIR/facets.npz" 3
        require_file "$THERMAL_DIR/facet_T_matrix_K.npy" 3
        require_file "$MRT_DIR/path_xyz.npy" 3
        require_file "$MRT_DIR/times.csv" 3
        MICROCLIMATE_URBAN_ARG=()
        if [ "$WITH_URBAN_RADIATION" = "1" ] && ! skip URBAN_RADIATION; then
            require_file "$URBAN_RADIATION_CONFIG" 4
            require_file "$OSM_GROUND_MATERIAL_DIR/osm_ground_materials.gpkg" 2
            progress_event 4 2 running "Partitioning materials and solving full-surface radiation"
            log "STEP 4  Full-surface urban radiation and conduction (stage 05d)"
            "$PY" 05d_urban_radiation.py \
                --buildings-stl "$BUILDINGS_STL" \
                --vegetation-stl "$VEGETATION_STL" \
                --ground-stl "$GROUND_STL" \
                --ground-material-dir "$OSM_GROUND_MATERIAL_DIR" \
                --mrt-dir "$MRT_DIR" \
                --output-dir "$URBAN_RADIATION_DIR" \
                --config "$URBAN_RADIATION_CONFIG"
            MICROCLIMATE_URBAN_ARG=(--urban-radiation-dir "$URBAN_RADIATION_DIR")
            progress_event 4 40 running "Full-surface radiation and conduction complete"
        elif [ "$WITH_URBAN_RADIATION" = "1" ]; then
            echo "  Full-surface urban radiation skipped (SKIP_URBAN_RADIATION=1); using 05b facets."
        else
            echo "  Full-surface urban radiation disabled; using established 05b route-visible facets."
        fi
        coupling_iteration=1
        while [ "$coupling_iteration" -le "$MICROCLIMATE_COUPLING_ITERATIONS" ]; do
            progress_base=$(( (coupling_iteration - 1) * 80 / MICROCLIMATE_COUPLING_ITERATIONS ))
            progress_event 4 "$progress_base" running \
                "Solving diagnostic air field iteration $coupling_iteration/$MICROCLIMATE_COUPLING_ITERATIONS"
            log "STEP 4  Diagnostic 3-D microclimate iteration $coupling_iteration/$MICROCLIMATE_COUPLING_ITERATIONS"
            "$PY" 05c_microclimate_solver.py \
                --buildings-stl "$BUILDINGS_STL" \
                --ground-stl "$GROUND_STL" \
                --facets-dir "$THERMAL_DIR" \
                --mrt-dir "$MRT_DIR" \
                --output-dir "$MICROCLIMATE_DIR" \
                --config "$MICROCLIMATE_CONFIG" \
                "${MICROCLIMATE_URBAN_ARG[@]}"

            if ! skip MICROCLIMATE_05B; then
                log "STEP 4  Recoupling facet surface energy to local Ta/velocity"
                "$PY" 05b_facet_energy_balance.py \
                    --buildings-stl "$BUILDINGS_STL" \
                    --vegetation-stl "$VEGETATION_STL" \
                    --ground-stl "$GROUND_STL" \
                    --facets-dir "$THERMAL_DIR" --mrt-dir "$MRT_DIR" \
                    --output-dir "$THERMAL_DIR" \
                    --microclimate-dir "$MICROCLIMATE_DIR" \
                    "${GROUND_MATERIAL_ARG[@]}" \
                    --spinup-days "$SPINUP_DAYS" \
                    --maximum-spinup-days "$MAXIMUM_SPINUP_DAYS" \
                    --spinup-convergence-tolerance-k "$SPINUP_TOLERANCE_K" \
                    --wind-speed "$WIND_SPEED" --wind-source "$SURFACE_WIND_SOURCE" \
                    --convection-model "$SURFACE_CONVECTION_MODEL" \
                    --latent-heat-model "$SURFACE_LATENT_HEAT_MODEL" \
                    --cloud-cover-fraction "$CLOUD" \
                    --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
                    --clear-sky-emissivity "$CLEAR_SKY_MODEL"
            fi
            coupling_iteration=$((coupling_iteration + 1))
        done

        if ! skip MICROCLIMATE_05FACET; then
            progress_event 4 82 running "Regenerating MRT with solved local air field"
            log "STEP 4  Regenerating authoritative MRT with solved Ta/velocity"
            "$PY" 05_mrt_network_raytrace.py \
                --buildings-stl "$BUILDINGS_STL" \
                --vegetation-stl "$VEGETATION_STL" \
                --ground-stl "$GROUND_STL" \
                --polylines-pkl "$ROUTE_POLYLINES" \
                --output-dir "$MRT_FACET_DIR" \
                --svf-cache "$SVF_CACHE_DIR" \
                --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
                --z-height "$Z_HEIGHT" \
                --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
                --cloud-cover-fraction "$CLOUD" \
                --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
                --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
                --facet-thermal-dir "$THERMAL_DIR" \
                --microclimate-dir "$MICROCLIMATE_DIR" \
                --microclimate-receptor-height-m "$Z_HEIGHT" \
                --radiant-flux-config "$RADIANT_FLUX_CONFIG" \
                "${RADIATION_FORCING_ARG[@]}" "${WEATHER_ARG[@]}"
        fi
        progress_event 4 100 done "Solved microclimate field coupled to surfaces and MRT"
    fi
fi

DOWNSTREAM_MICROCLIMATE_ARG=()
DOWNSTREAM_URBAN_VALIDATION_ARG=()
if [ -f "$URBAN_RADIATION_DIR/urban_radiation_metadata.json" ]; then
    DOWNSTREAM_URBAN_VALIDATION_ARG=(--urban-radiation-dir "$URBAN_RADIATION_DIR")
fi
if [ "$USE_MICROCLIMATE_FIELDS" = "1" ] \
        || { [ "$USE_MICROCLIMATE_FIELDS" = "auto" ] \
             && [ -f "$MICROCLIMATE_DIR/microclimate_metadata.json" ]; }; then
    if "$PY" 05c_microclimate_solver.py \
            --buildings-stl "$BUILDINGS_STL" --ground-stl "$GROUND_STL" \
            --facets-dir "$THERMAL_DIR" --mrt-dir "$MRT_DIR" \
            --output-dir "$MICROCLIMATE_DIR" --config "$MICROCLIMATE_CONFIG" \
            "${DOWNSTREAM_URBAN_VALIDATION_ARG[@]}" \
            --validate-only; then
        DOWNSTREAM_MICROCLIMATE_ARG=(--microclimate-dir "$MICROCLIMATE_DIR")
        echo "Downstream air field: solved microclimate at $MICROCLIMATE_DIR"
    elif [ "$USE_MICROCLIMATE_FIELDS" = "1" ]; then
        echo "ERROR: USE_MICROCLIMATE_FIELDS=1 but the solved field is stale or invalid" >&2
        exit 1
    else
        echo "WARNING: cached microclimate field is stale; using uniform weather fallback"
    fi
else
    echo "Downstream air field: spatially uniform shared-weather fallback"
fi

if active 5 && { ! skip 06 || ! skip 07 || ! skip 08 || ! skip 09 \
        || { ! skip 10 && [ "$SOLWEIG_COMPARE" != "0" ]; }; }; then
    require_file "$VIS_MRT_DIR/tmrt_matrix_C.npy" 3
    echo
    echo "Visualization / route-stress stages consume: $VIS_MRT_DIR"
    echo "(set VIS_MRT_DIR=$MRT_DIR to use the legacy-surround results instead)"
fi

# ----------------------------------------------------------------------------
# STEP 5 -- Visualizations (stages 06 MRT, 07 UTCI)
# ----------------------------------------------------------------------------
if active 5; then
    progress_event 5 0 running "Preparing MRT visualizations"
fi
if active 5 && ! skip 06; then
    log "STEP 5  Visualizing MRT network (stage 06)"
    "$PY" 06_visualize_mrt_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/mrt"
fi
if active 5; then
    progress_event 5 45 running "MRT visualization complete; preparing UTCI"
fi
if active 5 && ! skip 07; then
    log "STEP 5  Visualizing UTCI network (stage 07)"
    "$PY" 07_visualize_utci_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/utci" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        --microclimate-receptor-height-m "$Z_HEIGHT" \
        "${DOWNSTREAM_MICROCLIMATE_ARG[@]}" \
        "${WEATHER_ARG[@]}"
fi
if active 5; then
    if skip 06 && skip 07; then
        progress_event 5 100 skipped "Visualization work skipped"
    else
        progress_event 5 100 done "MRT and UTCI visualizations complete"
    fi
fi

# ----------------------------------------------------------------------------
# STEP 6 -- Route thermal stress (08 UTCI, 09 JOS-3, optional comparison)
# ----------------------------------------------------------------------------
if active 6 && { ! skip 08 || ! skip 09 \
        || { ! skip 10 && [ "$SOLWEIG_COMPARE" != "0" ]; }; }; then
    require_routes
fi
if active 6 && { ! skip 08 || ! skip 09; }; then
    require_file "$VIS_MRT_DIR/tmrt_matrix_C.npy" 3
fi
if active 6; then
    progress_event 6 0 running "Preparing route-level UTCI analysis"
fi
if active 6 && ! skip 08; then
    log "STEP 6  Route thermal stress -- UTCI exposure (stage 08)"
    # Route CSV/JSON inputs are external to the numbered pipeline. Stage 05
    # ray-traces their derived polylines; this stage reads the same CSV files.
    "$PY" 08_route_thermal_stress.py \
        --routes-dir "$ROUTES_DIR" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_utci" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed-ms "$WALKING_SPEED_MS" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        --local-origin-x "$LOCAL_ORIGIN_X" --local-origin-y "$LOCAL_ORIGIN_Y" \
        --project-crs "$PROJECT_CRS" \
        --radiant-flux-config "$RADIANT_FLUX_CONFIG" \
        --microclimate-receptor-height-m "$Z_HEIGHT" \
        "${DOWNSTREAM_MICROCLIMATE_ARG[@]}" \
        "${WEATHER_ARG[@]}"
fi
if active 6; then
    progress_event 6 52 running "Route UTCI complete; preparing JOS-3"
fi
if active 6 && ! skip 09; then
    log "STEP 6  Route thermal stress -- JOS-3 core temperature (stage 09)"
    SUBJECT_ARG=()
    [ -n "$SUBJECT_PROFILE" ] && SUBJECT_ARG=(--subject-profile "$SUBJECT_PROFILE")
    "$PY" 09_route_thermal_stress_jos3.py \
        --routes-dir "$ROUTES_DIR" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_jos3" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed-ms "$WALKING_SPEED_MS" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        --local-origin-x "$LOCAL_ORIGIN_X" --local-origin-y "$LOCAL_ORIGIN_Y" \
        --project-crs "$PROJECT_CRS" \
        --microclimate-receptor-height-m "$Z_HEIGHT" \
        "${DOWNSTREAM_MICROCLIMATE_ARG[@]}" \
        "${SUBJECT_ARG[@]}" "${WEATHER_ARG[@]}"
fi
if active 6 && ! skip 10 && [ "$SOLWEIG_COMPARE" != "0" ]; then
    require_file "$VIS_DIR/route_utci/routes_points.csv" 6
    log "STEP 6  Optional SOLWEIG comparison (stage 10)"
    SOLWEIG_ARG=()
    [ "$SOLWEIG_COMPARE" != "1" ] && SOLWEIG_ARG=(--solweig "$SOLWEIG_COMPARE")
    "$PY" compare_mrt_solweig.py \
        --ours "$VIS_DIR/route_utci/routes_points.csv" \
        "${SOLWEIG_ARG[@]}" \
        --output-dir "$SOLWEIG_COMPARE_OUTPUT_DIR"
fi
if active 6; then
    if skip 08 && skip 09 && { skip 10 || [ "$SOLWEIG_COMPARE" = "0" ]; }; then
        progress_event 6 100 skipped "Route stress and comparison skipped"
    else
        progress_event 6 100 done "Route stress and optional comparison complete"
    fi
fi

log "Pipeline complete (started at step $START_STEP)"
echo "  MRT prep/baseline: $MRT_DIR (baseline enabled=$WITH_BASELINE)"
echo "  Route inputs      : $ROUTES_DIR"
echo "  Facets + temps   : $THERMAL_DIR"
echo "  Microclimate     : $MICROCLIMATE_DIR (requested=$WITH_MICROCLIMATE; use=$USE_MICROCLIMATE_FIELDS)"
echo "  Urban radiation  : $URBAN_RADIATION_DIR (requested=$WITH_URBAN_RADIATION)"
echo "  Ground materials : $OSM_GROUND_MATERIAL_DIR (enabled=$OSM_GROUND_MATERIALS_ENABLED)"
echo "  MRT (facet therm): $MRT_FACET_DIR"
echo "  Radiant flux      : $VIS_DIR/route_utci/radiant_flux_contributions"
echo "  Visualizations   : $VIS_DIR"
