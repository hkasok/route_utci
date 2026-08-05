#!/usr/bin/env bash
#
# start.sh -- run the route_utci pipeline, optionally starting from any step.
#
# ============================================================================
#  START FROM ANY STEP:   ./start.sh <N>       (default N = 2)
# ============================================================================
#  N  STEP                              PRODUCES
#  1  Geometry build (LAZ -> STL)       building/vegetation/ground_final.stl
#  2  OSM network + route selection     graphml, route_polylines, selected_routes
#  3  MRT ray tracing + SVF (stage 05)  path_xyz.npy, times.csv, svf_*.npy
#                                        (now ONLY along the selected routes)
#  4  Facet selection (stage 05a)       facets.npz, LW view matrix
#  5  Facet energy balance (stage 05b)  facet_T_matrix_K.npy (surface temps)
#  6  Facet-thermal MRT (stage 05*)     improved tmrt_matrix_C.npy
#  7  Visualizations (stages 06, 07)    MRT + UTCI maps / animations
#  8  Route thermal stress (08, 09)     UTCI exposure + JOS-3 core temp
#
#  Examples:
#     ./start.sh          # start at step 2 (OSM) -- assumes STL already built
#     ./start.sh 1        # full run including the LAZ -> STL geometry build
#     ./start.sh 3        # re-run everything from MRT onward (geometry+OSM kept)
#     ./start.sh 7        # only (re)build the visualizations and route stress
#     ./start.sh 8        # only re-run the route-stress stages (08, 09)
#
#  Starting at step N runs N, N+1, ... to the end. Steps before N are assumed
#  already done; the script checks their outputs exist and stops with a clear
#  message if something required is missing. Geometry (step 1) is NOT run
#  unless you explicitly start at step 1, because it is the slow LAZ pipeline.
#
#  You can still force-skip an individual stage with SKIP_<NAME>=1
#  (GEOM/OSM/05/05A/05B/05FACET/06/07/08/09), e.g. SKIP_07=1 ./start.sh 7.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

START_STEP="${1:-2}"
case "$START_STEP" in 1|2|3|4|5|6|7|8) ;; *)
    echo "ERROR: step must be 1-8 (got '$START_STEP'). See the table in this file." >&2
    exit 1 ;; esac

# ############################################################################
# #  CRITICAL INPUTS  --  the settings you are most likely to change.        #
# #  Every value can also be overridden from the environment, e.g.           #
# #     DATE=2025-08-01 DEPARTURE_HOUR=15.0 ./start.sh 3                      #
# ############################################################################

# ---- WHEN: day of the solar run and time of the walk -----------------------
DATE="${DATE:-2025-07-06}"            # day of year for solar ray tracing
                                      #   (July 6 = peak Miami summer sun)
DEPARTURE_HOUR="${DEPARTURE_HOUR:-13.0}"    # hour the walk BEGINS (0-24);
                                      #   13.0 = solar-afternoon heat, 8.0 = AM
WALKING_SPEED_MS="${WALKING_SPEED_MS:-1.3}" # walking pace, m/s
                                      #   1.3 ~= 4.7 km/h, average adult
# JOS-3 virtual subject for stage 09 (stage 8). Presets (cited in
# subject_profiles.py): healthy_adult, healthy_adult_female, child,
# elderly_male, elderly_female, obese_adult, acclimatized_adult.
# Empty = JOS-3 default healthy adult.
SUBJECT_PROFILE="${SUBJECT_PROFILE:-}"

# ---- WHERE: site location (used for sun position) --------------------------
LAT="${LAT:-25.7560}"                 # latitude  (FIU MMC campus, Miami)
LON="${LON:--80.3770}"                # longitude
TZ="${TZ:-America/New_York}"          # timezone

# ---- ROUTE ENDPOINTS + GEOREFERENCING (stage 08) ---------------------------
# Origin shift applied when the OSM network was built (must match
# extract_osm_pedestrian_network.py) so exported routes get TRUE lat/lon.
LOCAL_ORIGIN_X="${LOCAL_ORIGIN_X:-0.0}"
LOCAL_ORIGIN_Y="${LOCAL_ORIGIN_Y:-0.0}"
PROJECT_CRS="${PROJECT_CRS:-EPSG:6346}"   # NAD83(2011) UTM 17N (Miami)
# Optional explicit start/end as "lat lon" (leave empty = auto bbox corners).
# e.g. START_LATLON="25.7585 -80.3760"  END_LATLON="25.7532 -80.3735"
START_LATLON="${START_LATLON:-}"
END_LATLON="${END_LATLON:-}"
N_ROUTES="${N_ROUTES:-3}"             # number of edge-disjoint routes to compare

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
WEATHER_CSV="${WEATHER_CSV:-$PWD/weather.csv}"
RH_PCT="${RH_PCT:-70}"                # constant RH if no CSV / no rh column
WIND_MS="${WIND_MS:-3.1}"             # constant wind if no CSV / no wind col
CLOUD="${CLOUD:-0.0}"                 # cloud-cover fraction for the solar run
                                      #   (0 = clear sky; keep 05 and 05b equal)

# ---- GEOMETRY: pre-built STL inputs (NOT regenerated unless step 1) --------
GEOM_DIR="${GEOM_DIR:-out_full/02_final}"
BUILDINGS_STL="${BUILDINGS_STL:-$GEOM_DIR/building_final.stl}"
VEGETATION_STL="${VEGETATION_STL:-$GEOM_DIR/vegetation_final.stl}"
GROUND_STL="${GROUND_STL:-$GEOM_DIR/ground_and_water_final.stl}"
# Only used when starting at step 1 (the LAZ -> STL build). Defaults below
# come from the old start_LAZ_to_stl.sh launcher (now folded into this script):
INPUT_LAZ="${INPUT_LAZ:-/media/harshin/data_drive/route based UTCI/fl2021_miami_dade_J1413971/fl2021_miami_dade_J1413971tR0_C0.laz}"
GEOM_OUTPUT_DIR="${GEOM_OUTPUT_DIR:-./out_full}"
# Memory cap for the LAZ -> STL build. main.py can balloon on dense LiDAR, so
# it runs inside a memory-capped systemd scope (when available): if it runs
# away, only THIS build gets OOM-killed, not the whole desktop session.
GEOM_MEMORY_MAX="${GEOM_MEMORY_MAX:-60G}"
# Timeout for the ground planar-decimate pass in main.py (seconds).
GROUND_PLANAR_TIMEOUT_SEC="${GROUND_PLANAR_TIMEOUT_SEC:-600}"

# ############################################################################
# #  SECONDARY SETTINGS  --  sensible defaults; change only if you know why. #
# ############################################################################
PY="${PY:-python3}"

# Output directories
OUT_ROOT="${OUT_ROOT:-run_output}"
OSM_DIR="${OSM_DIR:-$OUT_ROOT/osm_paths}"
OSM_EDGE_FEATURES="${OSM_EDGE_FEATURES:-$OSM_DIR/pedestrian_edges.geojson}"
OSM_COMPLETE_CACHE="${OSM_COMPLETE_CACHE:-data/osm/fiu_mmc_complete_osm.gpkg}"
OSM_COMPLETE_INPUT="${OSM_COMPLETE_INPUT:-}"
OSM_GROUND_FEATURES="${OSM_GROUND_FEATURES:-$OSM_COMPLETE_CACHE}"
OSM_GROUND_MATERIALS_ENABLED="${OSM_GROUND_MATERIALS_ENABLED:-1}"
OSM_GROUND_CONFIG="${OSM_GROUND_CONFIG:-$PWD/osm_ground_materials.json}"
OSM_GROUND_MATERIAL_DIR="${OSM_GROUND_MATERIAL_DIR:-$OUT_ROOT/osm_ground_materials}"
OSM_GROUND_OVERRIDES="${OSM_GROUND_OVERRIDES:-}"
RADIANT_FLUX_CONFIG="${RADIANT_FLUX_CONFIG:-$PWD/radiant_flux_contribution_results.json}"
MRT_DIR="${MRT_DIR:-$OUT_ROOT/mrt_out}"                   # legacy-surround MRT
THERMAL_DIR="${THERMAL_DIR:-$OUT_ROOT/thermal_out}"       # 05a + 05b outputs
MRT_FACET_DIR="${MRT_FACET_DIR:-$OUT_ROOT/mrt_facet_out}" # facet-thermal MRT
VIS_DIR="${VIS_DIR:-$OUT_ROOT/viz}"

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
# Route selection (04_select_routes.py): the N start->end routes are chosen
# UP FRONT so MRT is ray-traced along only those routes, not the whole
# pedestrian network -- the big speedup. These feed stage 05 and 08/09.
ROUTE_POLYLINES="${ROUTE_POLYLINES:-$OSM_DIR/route_polylines.pkl}"   # -> MRT input
SELECTED_ROUTES="${SELECTED_ROUTES:-$OSM_DIR/selected_routes.pkl}"   # -> 08/09 input
ROUTE_SAMPLE_SPACING_M="${ROUTE_SAMPLE_SPACING_M:-1.0}"

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
WIND_SPEED="${WIND_SPEED:-1.5}"       # near-surface wind for 05b convection

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log()  { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
skip() { local v="SKIP_$1"; [ "${!v:-0}" = "1" ]; }
active() { [ "$START_STEP" -le "$1" ]; }   # true if step N is at/after start

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required input not found: $1" >&2
        echo "       Step $START_STEP assumes earlier steps already ran." >&2
        echo "       Re-run from an earlier step (e.g. ./start.sh ${2:-1}) or set the path." >&2
        exit 1
    fi
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
  Date (solar)     : $DATE
  Departure hour   : $DEPARTURE_HOUR:00
  Walking speed    : $WALKING_SPEED_MS m/s
  MRT sample height: $Z_HEIGHT m above ground
  Location         : lat $LAT, lon $LON ($TZ)
  Weather          : $WEATHER_NOTE
  Geometry (STL)   : $GEOM_DIR
  Output root      : $OUT_ROOT
EOF
mkdir -p "$OUT_ROOT"

# ----------------------------------------------------------------------------
# STEP 1 -- Geometry build (LAZ -> STL).  Slow; only if explicitly started
# here. Runs main.py in the foreground so the pipeline continues after it.
# ----------------------------------------------------------------------------
if active 1 && ! skip GEOM; then
    log "STEP 1  Geometry build (LAZ -> STL)"
    if [ -z "$INPUT_LAZ" ]; then
        echo "ERROR: starting at step 1 requires INPUT_LAZ=/path/to/file.laz" >&2
        echo "       Set INPUT_LAZ in the GEOMETRY section of this script, or:" >&2
        echo "       INPUT_LAZ=/data/site.laz ./start.sh 1" >&2
        exit 1
    fi
    require_file "$INPUT_LAZ" 1
    echo "  Input LAZ  : $INPUT_LAZ"
    echo "  Output dir : $GEOM_OUTPUT_DIR"

    # Run the LAZ -> STL build (main.py) in the FOREGROUND so the rest of the
    # pipeline continues after it. Wrap it in a memory-capped systemd scope
    # when available so a runaway-memory step is OOM-killed on its own instead
    # of taking down the desktop session (see the DBSCAN OOM incident).
    GEOM_CMD=("$PY" -u main.py
        --input "$INPUT_LAZ"
        --output-dir "$GEOM_OUTPUT_DIR"
        --ground-planar-timeout-sec "$GROUND_PLANAR_TIMEOUT_SEC")
    if command -v systemd-run >/dev/null 2>&1; then
        echo "  Memory cap : $GEOM_MEMORY_MAX (systemd-run --scope --user)"
        systemd-run --scope --user -p MemoryMax="$GEOM_MEMORY_MAX" -- "${GEOM_CMD[@]}"
    else
        echo "  Memory cap : (systemd-run unavailable -- running without a cap)"
        "${GEOM_CMD[@]}"
    fi
elif active 1; then
    log "STEP 1  Geometry build -- SKIP_GEOM=1, skipped"
fi

# From here on the STL geometry must exist (built above or pre-existing).
if active 2; then
    log "Checking pre-built STL geometry"
    require_file "$BUILDINGS_STL" 1
    require_file "$VEGETATION_STL" 1
    require_file "$GROUND_STL" 1
    echo "  buildings : $BUILDINGS_STL"
    echo "  vegetation: $VEGETATION_STL"
    echo "  ground    : $GROUND_STL"
fi

# ----------------------------------------------------------------------------
# STEP 2 -- OSM pedestrian network
# ----------------------------------------------------------------------------
if active 2 && ! skip OSM; then
    if [ -f "$GRAPHML" ] && [ -f "$POLYLINES" ] && [ "${FORCE_OSM:-0}" != 1 ]; then
        log "STEP 2  OSM network -- already present, skipped (FORCE_OSM=1 to rebuild)"
    else
        log "STEP 2  Extracting OSM pedestrian network"
        "$PY" extract_osm_pedestrian_network.py --output-dir "$OSM_DIR"
    fi

    # Select the N start->end routes NOW (cheap, graph-only) so the expensive
    # MRT stages ray-trace along only these routes instead of the entire
    # pedestrian network. Endpoints, origin shift and CRS mirror stage 08/09.
    log "STEP 2  Selecting $N_ROUTES start->end route(s) (stage 04)"
    ENDPOINT_ARG=()
    [ -n "$START_LATLON" ] && ENDPOINT_ARG+=(--start-latlon $START_LATLON)
    [ -n "$END_LATLON" ]   && ENDPOINT_ARG+=(--end-latlon $END_LATLON)
    "$PY" 04_select_routes.py \
        --graphml "$GRAPHML" \
        --output-dir "$OSM_DIR" \
        --n-routes "$N_ROUTES" \
        --route-sample-spacing-m "$ROUTE_SAMPLE_SPACING_M" \
        --local-origin-x "$LOCAL_ORIGIN_X" --local-origin-y "$LOCAL_ORIGIN_Y" \
        --project-crs "$PROJECT_CRS" \
        "${ENDPOINT_ARG[@]}"
fi
if active 3; then
    require_file "$ROUTE_POLYLINES" 2
    require_file "$SELECTED_ROUTES" 2
fi

# ----------------------------------------------------------------------------
# PARALLEL OSM SURFACE BRANCH -- material subdivision only.
# This runs AFTER route selection and reads, but never modifies, the graph or
# selected-route files. It assigns materials to existing terrain face IDs.
# ----------------------------------------------------------------------------
GROUND_MATERIAL_ARG=()
if [ "$OSM_GROUND_MATERIALS_ENABLED" = "1" ]; then
    if active 2; then
        log "STEP 2  OSM ground-material subdivision (routing unchanged)"
        require_file "$OSM_GROUND_CONFIG" 2
        COMPLETE_INPUT_ARG=()
        [ -n "$OSM_COMPLETE_INPUT" ] && COMPLETE_INPUT_ARG=(--input-file "$OSM_COMPLETE_INPUT")
        FORCE_COMPLETE_ARG=()
        [ "${FORCE_OSM_COMPLETE:-0}" = "1" ] && FORCE_COMPLETE_ARG=(--force-download)
        "$PY" download_osm_complete_features.py \
            --ground-mesh "$GROUND_STL" \
            --config "$OSM_GROUND_CONFIG" \
            --output-file "$OSM_COMPLETE_CACHE" \
            "${COMPLETE_INPUT_ARG[@]}" \
            "${FORCE_COMPLETE_ARG[@]}"
        require_file "$OSM_GROUND_FEATURES" 2
        OVERRIDE_ARG=()
        [ -n "$OSM_GROUND_OVERRIDES" ] && OVERRIDE_ARG=(--overrides "$OSM_GROUND_OVERRIDES")
        ROUTE_HASH_BEFORE="$(sha256sum "$GRAPHML" "$ROUTE_POLYLINES" "$SELECTED_ROUTES")"
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
        ROUTE_HASH_AFTER="$(sha256sum "$GRAPHML" "$ROUTE_POLYLINES" "$SELECTED_ROUTES")"
        if [ "$ROUTE_HASH_BEFORE" != "$ROUTE_HASH_AFTER" ]; then
            echo "FATAL: OSM surface processing changed a protected routing artifact" >&2
            exit 1
        fi
        printf '%s\n' "$ROUTE_HASH_AFTER" > \
            "$OSM_GROUND_MATERIAL_DIR/route_artifact_integrity.sha256"
        echo "  Protected route graph/selection hashes unchanged"
    fi
    require_file "$OSM_GROUND_MATERIAL_DIR/ground_face_materials.npz" 2
    require_file "$OSM_GROUND_MATERIAL_DIR/ground_material_catalog.json" 2
    GROUND_MATERIAL_ARG=(--ground-material-dir "$OSM_GROUND_MATERIAL_DIR")
else
    echo "  OSM ground materials disabled: uniform-ground backward-compatible mode"
fi

# ----------------------------------------------------------------------------
# STEP 3 -- MRT ray tracing + SVF (stage 05, legacy surround).
# Produces path_xyz + times.csv that 05a/05b depend on. SVF is computed ONCE
# here (it is geometry-only) and reused across all timesteps internally.
# ----------------------------------------------------------------------------
if active 3 && ! skip 05; then
    log "STEP 3  MRT ray tracing + SVF (stage 05)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$ROUTE_POLYLINES" \
        --output-dir "$MRT_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --z-height "$Z_HEIGHT" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
        "${LEGACY_RADIANT_FLUX_ARG[@]}" \
        "${WEATHER_ARG[@]}"
fi
if active 4; then
    require_file "$MRT_DIR/path_xyz.npy" 3
    require_file "$MRT_DIR/times.csv" 3
fi

# ----------------------------------------------------------------------------
# STEP 4 -- Facet selection + LW view matrix (stage 05a)
# ----------------------------------------------------------------------------
if active 4 && ! skip 05A; then
    log "STEP 4  Selecting route-visible thermal facets (stage 05a)"
    "$PY" 05a_thermal_facets_select.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        "${GROUND_MATERIAL_ARG[@]}" \
        --point-stride "$POINT_STRIDE" --max-distance "$MAX_DISTANCE"
fi
if active 5; then require_file "$THERMAL_DIR/facets.npz" 4; fi

# ----------------------------------------------------------------------------
# STEP 5 -- Facet 1D surface-energy balance (stage 05b).
# Same weather as stage 05 (reads times.csv); keep CLOUD equal to step 3.
# ----------------------------------------------------------------------------
if active 5 && ! skip 05B; then
    log "STEP 5  Facet 1D surface-energy balance (stage 05b)"
    "$PY" 05b_facet_energy_balance.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --facets-dir "$THERMAL_DIR" \
        --mrt-dir "$MRT_DIR" \
        --output-dir "$THERMAL_DIR" \
        "${GROUND_MATERIAL_ARG[@]}" \
        --spinup-days "$SPINUP_DAYS" --wind-speed "$WIND_SPEED" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL"
fi
if active 6; then require_file "$THERMAL_DIR/facet_T_matrix_K.npy" 5; fi

# ----------------------------------------------------------------------------
# STEP 6 -- MRT ray tracing again, consuming facet surface temperatures
# ----------------------------------------------------------------------------
if active 6 && ! skip 05FACET; then
    log "STEP 6  Facet-thermal MRT ray tracing (stage 05*)"
    "$PY" 05_mrt_network_raytrace.py \
        --buildings-stl "$BUILDINGS_STL" \
        --vegetation-stl "$VEGETATION_STL" \
        --ground-stl "$GROUND_STL" \
        --polylines-pkl "$ROUTE_POLYLINES" \
        --output-dir "$MRT_FACET_DIR" \
        --ds-path "$DS_PATH" --dt-min "$DT_MIN" --date "$DATE" \
        --z-height "$Z_HEIGHT" \
        --latitude "$LAT" --longitude "$LON" --timezone "$TZ" \
        --cloud-cover-fraction "$CLOUD" \
        --k-lad-direct "$K_LAD_DIRECT" --k-lad-diffuse "$K_LAD_DIFFUSE" \
        --clear-sky-emissivity "$CLEAR_SKY_MODEL" \
        --facet-thermal-dir "$THERMAL_DIR" \
        --radiant-flux-config "$RADIANT_FLUX_CONFIG" \
        "${WEATHER_ARG[@]}"
fi
if active 6 && [ "$OSM_GROUND_MATERIALS_ENABLED" = "1" ]; then
    require_file "$MRT_FACET_DIR/radiant_flux_contributions.npz" 6
    log "STEP 6  Route-point ground-material diagnostics"
    "$PY" route_ground_material_diagnostics.py \
        --mrt-dir "$MRT_FACET_DIR" \
        --thermal-dir "$THERMAL_DIR" \
        --ground-mesh "$GROUND_STL" \
        --ground-material-dir "$OSM_GROUND_MATERIAL_DIR" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed "$WALKING_SPEED_MS" \
        --output "$OSM_GROUND_MATERIAL_DIR/route_point_ground_material_diagnostics.csv"
fi
if active 7; then
    require_file "$VIS_MRT_DIR/tmrt_matrix_C.npy" 6
    echo
    echo "Visualization / route-stress stages consume: $VIS_MRT_DIR"
    echo "(set VIS_MRT_DIR=$MRT_DIR to use the legacy-surround results instead)"
fi

# ----------------------------------------------------------------------------
# STEP 7 -- Visualizations (stages 06 MRT, 07 UTCI)
# ----------------------------------------------------------------------------
if active 7 && ! skip 06; then
    log "STEP 7  Visualizing MRT network (stage 06)"
    "$PY" 06_visualize_mrt_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/mrt"
fi
if active 7 && ! skip 07; then
    log "STEP 7  Visualizing UTCI network (stage 07)"
    "$PY" 07_visualize_utci_network.py \
        --results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/utci" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        "${WEATHER_ARG[@]}"
fi

# ----------------------------------------------------------------------------
# STEP 8 -- Route thermal stress (08 UTCI exposure, 09 JOS-3 core temp)
# ----------------------------------------------------------------------------
if active 8; then require_file "$SELECTED_ROUTES" 2; fi
if active 8 && ! skip 08; then
    log "STEP 8  Route thermal stress -- UTCI exposure (stage 08)"
    # Routes were already chosen in step 2 (04_select_routes.py) and are the
    # exact routes MRT was computed along -- pass them via --routes-pkl so the
    # endpoints/count are consistent end to end (no re-selection here).
    "$PY" 08_route_thermal_stress.py \
        --routes-pkl "$SELECTED_ROUTES" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_utci" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed-ms "$WALKING_SPEED_MS" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        --local-origin-x "$LOCAL_ORIGIN_X" --local-origin-y "$LOCAL_ORIGIN_Y" \
        --project-crs "$PROJECT_CRS" \
        --radiant-flux-config "$RADIANT_FLUX_CONFIG" \
        "${WEATHER_ARG[@]}"
fi
if active 8 && ! skip 09; then
    log "STEP 8  Route thermal stress -- JOS-3 core temperature (stage 09)"
    SUBJECT_ARG=()
    [ -n "$SUBJECT_PROFILE" ] && SUBJECT_ARG=(--subject-profile "$SUBJECT_PROFILE")
    "$PY" 09_route_thermal_stress_jos3.py \
        --routes-pkl "$SELECTED_ROUTES" \
        --mrt-results-dir "$VIS_MRT_DIR" \
        --buildings-stl "$BUILDINGS_STL" \
        --output-dir "$VIS_DIR/route_jos3" \
        --departure-hour "$DEPARTURE_HOUR" \
        --walking-speed-ms "$WALKING_SPEED_MS" \
        --relative-humidity-pct "$RH_PCT" --wind-speed-ms "$WIND_MS" \
        "${SUBJECT_ARG[@]}" "${WEATHER_ARG[@]}"
fi

log "Pipeline complete (started at step $START_STEP)"
echo "  MRT (legacy)     : $MRT_DIR"
echo "  Facets + temps   : $THERMAL_DIR"
echo "  Ground materials : $OSM_GROUND_MATERIAL_DIR (enabled=$OSM_GROUND_MATERIALS_ENABLED)"
echo "  MRT (facet therm): $MRT_FACET_DIR"
echo "  Radiant flux      : $VIS_DIR/route_utci/radiant_flux_contributions"
echo "  Visualizations   : $VIS_DIR"
