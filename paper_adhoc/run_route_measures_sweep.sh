#!/usr/bin/env bash
# Route comfort/risk measures for the campus routes at every departure of the
# Fig. 9 sweep (07:00-19:00, 30 min), stages 08 and 09 as run by start.sh but
# with the clothing of the 13:00 walk (hot_minimal) worn at every departure,
# as in the fixed-clothing sweep. Output:
#   run_output/MMC/postprocessing/route_measures_sweep/dep_HHMM/{route_utci,route_jos3}
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=run_output/MMC/postprocessing/route_measures_sweep
mkdir -p "$OUT"
one() {
    local h=$1 tag
    tag=$(python3 -c "h=$h; print(f'{int(h):02d}{int(round((h%1)*60)):02d}')")
    local d="$OUT/dep_$tag"
    mkdir -p "$d"
    common=(--routes-dir input/MMC/routes --mrt-results-dir run_output/MMC/mrt_facet_out
            --departure-hour "$h" --walking-speed-ms 1.3
            --relative-humidity-pct 70.0 --wind-speed-ms 3.1
            --local-origin-x 0.0 --local-origin-y 0.0 --project-crs EPSG:6346
            --microclimate-receptor-height-m 1.1
            --weather-csv input/MMC/weather/weather.csv --require-weather-csv)
    nice python3 08_route_thermal_stress.py "${common[@]}" --output-dir "$d/route_utci" \
        > "$d/stage08.log" 2>&1 || { echo "08 failed at $h"; return 1; }
    nice python3 09_route_thermal_stress_jos3.py "${common[@]}" --output-dir "$d/route_jos3" \
        --clothing hot_minimal --clothing-climate temperate > "$d/stage09.log" 2>&1 \
        || { echo "09 failed at $h"; return 1; }
    echo "done $h"
}
export -f one; export OUT
python3 -c "print(' '.join(str(7+0.5*k) for k in range(25)))" | tr ' ' '\n' \
    | xargs -P 4 -I{} bash -c 'one {}'
echo "== sweep done"
