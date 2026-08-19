"""
08_route_thermal_stress.py -- load source-agnostic route input files and
compare their UTCI EXPOSURE as a person walks each route (encountering
different shade/sun at different times along the way).

SCOPE (important): this stage reports UTCI, an "equivalent temperature"
comfort index, spatially along each route. It does NOT compute a body
core-temperature rise. Core-temperature rise is a physiological state
that only the JOS-3 multi-node thermoregulation model (stage 09) computes
from a genuine time-stepped heat balance on the body; expressing a "core
rise" from MRT or UTCI would conflate three distinct quantities
(radiation, a feels-like index, and an actual body temperature). So:
  * stage 08 (here): UTCI along the route -- WHERE stress concentrates.
  * stage 09 (JOS-3): the one and only core-temperature-rise number.

UTCI is computed with pythermalcomfort (Brode et al. 2012 operational
polynomial) using the SAME call as stage 07, driven at each route point
by that point's ray-traced Tmrt at the walker's actual arrival time.

Routes are ranked by MEAN and PEAK UTCI along the route (lower = cooler).

Route generation is intentionally separate. Run generate_route.py first or
provide route CSV/JSON pairs satisfying the documented input contract.

Run:
    python3 08_route_thermal_stress.py \
        --routes-dir "input/MMC/routes" \
        --mrt-results-dir mrt_network_output/ \
        --output-dir route_stress_output/ \
        --buildings-stl input/MMC/geometry/building_final.stl \
        --departure-hour 8.0
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pythermalcomfort.models import utci

from weather_provider import add_weather_args, provider_from_args
from microclimate_field import EnvironmentField, add_microclimate_argument
from generate_route import load_routes_directory, route_arrival_schedule
from physical_checks import check_utci_inputs
from radiant_flux_contributions import (
    CONTRIBUTION_ARCHIVE, CONTRIBUTION_METADATA, LW_SOURCE_COLUMNS,
    PRIMARY_COLUMNS, SW_SOURCE_COLUMNS, TOTAL_COLUMNS,
    aggregate_plot_categories, load_contribution_config,
    plot_ranked_radiant_flux_contributions,
    plot_surface_longwave_classifications, route_contribution_summary,
    sample_route_contribution_matrices)
from thermal_common import SIGMA


# UTCI thermal-stress category boundaries (deg C) for reporting a route's
# exposure in physiologically meaningful terms (Brode et al. 2012).
UTCI_STRONG_STRESS_C = 32.0   # >= this = "strong heat stress" or worse


def remove_stale_route_outputs(directory, active_route_ids):
    """Remove generated per-route files left by an older, larger route set."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    active = {int(route_id) for route_id in active_route_ids}
    removed = []
    for path in directory.glob("route_*_*"):
        match = re.match(r"route_(\d+)_", path.name)
        if match and int(match.group(1)) not in active and path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def report_forcing(weather, environment, args, out_dir):
    """Print, and persist to disk, exactly where each UTCI driver came from.

    In the default fallback, air temperature, humidity and wind are spatially
    uniform along a route. The optional microclimate field instead supplies
    local Ta and vector-speed magnitude. Against a
    reference model that offset is indistinguishable from a radiation-scheme
    difference unless the forcing is recorded. Substituting RH 70% / wind
    3.1 m/s for a CSV's 60% / 3.7 m/s moves UTCI by about -1.5 degC with no
    other symptom, so the provenance is written next to the results.
    """
    import json

    prov = weather.provenance()
    h0 = float(args.departure_hour)
    ta0 = float(np.atleast_1d(weather.air_temp_c(h0))[0])
    rh0 = float(np.atleast_1d(weather.rh_pct(h0))[0])
    ws0 = float(np.atleast_1d(weather.wind_ms(h0))[0])

    print("\n" + "=" * 62)
    print("UTCI FORCING")
    print("=" * 62)
    print(f"  {weather.describe()}")
    print(f"  Air field: {environment.describe()}")
    for var, label in (("air_temp_C", "air temperature"),
                       ("rh_pct", "relative humidity"),
                       ("wind_ms", "wind speed")):
        print(f"    {label:<20s} <- {prov['source_' + var]}")
    print(f"  At departure hour {h0:g}: "
          f"Ta = {ta0:.2f} C, RH = {rh0:.1f} %, wind = {ws0:.2f} m/s")
    if not prov["all_from_csv"]:
        print("  " + "!" * 58)
        print("  ! WARNING: at least one UTCI driver is PARAMETRIC, not measured.")
        print("  ! Do not compare these UTCI values against another model.")
        print("  ! Re-run with --weather-csv ... --require-weather-csv")
        print("  " + "!" * 58)
    print("=" * 62)

    prov.update({"departure_hour": h0, "ta_at_departure_c": round(ta0, 3),
                 "rh_at_departure_pct": round(rh0, 2),
                 "wind_at_departure_ms": round(ws0, 3),
                 "walking_speed_ms": float(args.walking_speed_ms)})
    prov["spatiotemporal_air_field"] = environment.describe()
    (out_dir / "forcing_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Wrote {out_dir / 'forcing_provenance.json'}")


def export_routes_for_gis(results, out_dir, origin, project_crs):
    """Write each route's geometry in formats other software can validate
    against: GeoJSON (lat/lon, universal), a per-vertex CSV (lat/lon +
    projected + UTCI/Tmrt/arrival time along the route), and GPX tracks.

    Georeferencing is authoritative in each input route CSV. Conversion from
    the local frame to projected coordinates and WGS84 is performed once by
    generate_route.py rather than repeated during analysis.
    """
    import json
    features, csv_rows, gpx_tracks = [], [], []
    for r in results:
        route_input = r["route_input_frame"]
        xy_proj = route_input[["x_proj_m", "y_proj_m"]].to_numpy(dtype=float)
        lon = route_input["lon"].to_numpy(dtype=float)
        lat = route_input["lat"].to_numpy(dtype=float)
        rid = r["route_id"]

        features.append({
            "type": "Feature",
            "properties": {
                "route_id": rid,
                "length_m": round(r["length_m"], 1),
                "walk_duration_min": round(r["walk_duration_min"], 1),
                "mean_utci_c": round(r["mean_utci_c"], 2),
                "max_utci_c": round(r["max_utci_c"], 2),
                "mean_tmrt_c": round(r["mean_tmrt_c"], 2),
                "max_tmrt_c": round(r["max_tmrt_c"], 2),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(a), float(b)] for a, b in zip(lon, lat)],
            },
        })

        for k in range(len(lat)):
            csv_rows.append({
                "route_id": rid, "seq": k,
                "lat": round(float(lat[k]), 8), "lon": round(float(lon[k]), 8),
                "x_proj_m": round(float(xy_proj[k, 0]), 3),
                "y_proj_m": round(float(xy_proj[k, 1]), 3),
                "cumdist_m": round(float(r["cumdist_m"][k]), 2),
                "arrival_hour": round(float(r["arrival_hour"][k]), 4),
                "tmrt_c": round(float(r["tmrt_trace_c"][k]), 2),
                "utci_c": round(float(r["utci_trace_c"][k]), 2),
                # The three non-radiant UTCI drivers actually used at this
                # point. Exported so a model-vs-model comparison can verify
                # matched forcing directly instead of inferring it from the
                # UTCI-vs-Tmrt intercept after the fact.
                "ta_c": round(float(r["ta_trace_c"][k]), 3),
                "rh_pct": round(float(r["rh_trace_pct"][k]), 2),
                "wind_ms": round(float(r["wind_trace_ms"][k]), 3),
                "local_wind_ms": round(float(r["local_wind_trace_ms"][k]), 3),
                "wind_u_ms": round(float(r["wind_u_trace_ms"][k]), 3),
                "wind_v_ms": round(float(r["wind_v_trace_ms"][k]), 3),
                "wind_w_ms": round(float(r["wind_w_trace_ms"][k]), 3),
            })
        gpx_tracks.append((rid, lat, lon))

    fc = {"type": "FeatureCollection", "name": "route_utci_routes",
          "crs": {"type": "name",
                  "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": features}
    (out_dir / "routes.geojson").write_text(json.dumps(fc, indent=2))
    pd.DataFrame(csv_rows).to_csv(out_dir / "routes_points.csv", index=False)

    gpx = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="route_utci" '
           'xmlns="http://www.topografix.com/GPX/1/1">']
    for rid, lat, lon in gpx_tracks:
        gpx.append(f'  <trk><name>route_{rid}</name><trkseg>')
        for la, lo in zip(lat, lon):
            gpx.append(f'    <trkpt lat="{la:.8f}" lon="{lo:.8f}"></trkpt>')
        gpx.append('  </trkseg></trk>')
    gpx.append('</gpx>')
    (out_dir / "routes.gpx").write_text("\n".join(gpx))

    print("\nExported routes for external tools:")
    print(f"  {out_dir / 'routes.geojson'}   (lat/lon; QGIS/ArcGIS/geojson.io)")
    print(f"  {out_dir / 'routes_points.csv'} (per-vertex lat/lon + UTCI/Tmrt)")
    print(f"  {out_dir / 'routes.gpx'}        (Google Earth / GPS tools)")


# ============================================================
# Route loading
# ============================================================
def get_routes(args):
    """Load every validated route in numeric route-ID order."""
    routes = load_routes_directory(
        Path(args.routes_dir), expected_project_crs=args.project_crs,
        expected_origin=(args.local_origin_x, args.local_origin_y))
    print(f"Loading route inputs from {args.routes_dir} ...")
    print(f"  Loaded {len(routes)} validated route(s): "
          f"{[route['route_id'] for route in routes]}")
    # Existing plots use common start/end markers. The historical generated
    # routes share endpoints; source-agnostic routes may not, so the plotting
    # loops additionally mark every route below.
    return (routes, tuple(routes[0]["xy"][0]), tuple(routes[0]["xy"][-1]),
            "input_routes", "input_routes", "not_applicable")


def parse_args():
    p = argparse.ArgumentParser(description="Route thermal-stress comparison (UTCI)")
    p.add_argument("--routes-dir", default="input/MMC/routes",
                    help="Folder containing route_<id>.csv/json inputs "
                         "(default: 'input/MMC/routes')")
    p.add_argument("--mrt-results-dir", required=True, help="Output dir from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None)
    p.add_argument(
        "--radiant-flux-config", default=None,
        help=("Optional radiant_flux_contribution_results.json. If enabled, "
              "sample stage-05 absorbed-flux records at route arrival times "
              "and create ranked W m^-2 figures without reordering route data."),
    )

    # --- Georeferencing for exporting routes to other software ---
    p.add_argument("--local-origin-x", type=float, default=0.0,
                    help="Origin-shift X that was applied when building the "
                         "network (must match extract_osm_pedestrian_network.py "
                         "so exported routes get true coordinates).")
    p.add_argument("--local-origin-y", type=float, default=0.0,
                    help="Origin-shift Y (see --local-origin-x).")
    p.add_argument("--project-crs", default="EPSG:6346",
                    help="Projected CRS of the network/local frame "
                         "(default EPSG:6346 = NAD83(2011) UTM 17N, Miami). "
                         "Routes are exported in this CRS AND in lat/lon.")

    p.add_argument("--walking-speed-ms", type=float, default=1.3,
                    help="Average adult walking pace (default: 1.3 m/s ~= 4.7 km/h). "
                         "UTCI's reference activity is ~1.1 m/s; 1.3 better matches "
                         "a healthy adult crossing campus.")
    p.add_argument("--departure-hour", type=float, default=13.0,
                    help="Hour of day (0-24) the walk begins (default: 13.0, "
                         "solar-afternoon heat. Use 8.0 for a morning walk).")
    add_weather_args(p)
    add_microclimate_argument(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contribution_config = load_contribution_config(args.radiant_flux_config)

    weather = provider_from_args(args)
    environment = EnvironmentField(
        weather, args.microclimate_dir, args.microclimate_receptor_height_m)
    report_forcing(weather, environment, args, out_dir)

    # Origin shift (local frame -> projected CRS) for exporting routes to GIS.
    origin = np.array([args.local_origin_x, args.local_origin_y])

    routes, start_xy, end_xy, start_node, end_node, connectivity = get_routes(args)

    print("\nLoading MRT results for nearest-point lookup...")
    mrt_dir = Path(args.mrt_results_dir)
    mrt_xyz = np.load(mrt_dir / "path_xyz.npy")
    tmrt_matrix = np.load(mrt_dir / "tmrt_matrix_C.npy")
    times_df = pd.read_csv(mrt_dir / "times.csv", parse_dates=["time"])
    times = times_df["time"].tolist()
    time_hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in times])
    # unwrap in case times cross midnight boundary at the array edges
    mrt_tree = cKDTree(mrt_xyz[:, :2])
    contribution_matrices = None
    contribution_metadata = None
    direct_transmission_matrix = None
    if contribution_config["enabled"]:
        archive_path = mrt_dir / CONTRIBUTION_ARCHIVE
        metadata_path = mrt_dir / CONTRIBUTION_METADATA
        if not archive_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "radiant-flux results are enabled but stage-05 contribution "
                f"outputs are missing ({archive_path}, {metadata_path}); rerun stage 05")
        archive = np.load(archive_path)
        contribution_matrices = {key: archive[key] for key in archive.files}
        import json
        contribution_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_shape = (len(time_hours), len(mrt_xyz))
        for key, matrix in contribution_matrices.items():
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"contribution matrix {key} has {matrix.shape}, expected {expected_shape}")
        direct_transmission_matrix = np.load(
            mrt_dir / "direct_transmission_matrix.npy", mmap_mode="r")
        if direct_transmission_matrix.shape != expected_shape:
            raise ValueError("direct-transmission matrix does not match MRT grid")
        print(f"  Loaded absorbed radiant-flux contributions: {archive_path}")

    print("\nComputing UTCI along each route (at each point's arrival time)...")
    results = []
    contribution_route_frames = {}
    for i, route in enumerate(routes):
        route_id = route["route_id"]
        xy = route["xy"]
        n_pts = len(xy)

        # Experimental trajectories may carry device-derived per-point times.
        # Ordinary routes retain the historical distance/walking-speed timing.
        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumdist = np.concatenate(([0], np.cumsum(seg_lens)))
        arrival_hour, timing_source = route_arrival_schedule(
            route, args.departure_hour, args.walking_speed_ms)

        # nearest precomputed Tmrt sample point for each route point
        _, nearest_idx = mrt_tree.query(xy)

        # Local Tmrt at each point's actual arrival time
        h = arrival_hour % 24.0
        tmrt_trace = np.empty(n_pts)
        for j in range(n_pts):
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_trace[j] = np.interp(h[j], time_hours, tmrt_series, period=24.0)
        # Air temperature, RH and wind at each arrival time. The shared field
        # broadcasts WeatherProvider values by default or samples the optional
        # solved 3-D Ta/velocity. All three are returned together so UTCI's
        # exact values exported alongside it -- no chance of the reported
        # forcing drifting from the applied forcing.
        #
        # NOTE ON WIND: the UTCI polynomial is defined for wind at 10 m
        # reference height, so weather.csv's wind_ms must be a 10 m value.
        # A solved pedestrian-level vector is converted back to its 10 m
        # equivalent by EnvironmentField using the configured roughness.
        # This matches SOLWEIG, whose compute_utci_grid() documents its
        # `wind` argument as "Wind speed at 10m height". If your CSV holds
        # pedestrian-level wind, convert it before this stage rather than
        # here, so both models consume the identical series.
        receptor_xyz = np.column_stack((xy, mrt_xyz[nearest_idx, 2]))
        local_environment = environment.sample(receptor_xyz, h)
        ta_trace = local_environment.air_temperature_c
        rh_trace = local_environment.relative_humidity_pct
        local_wind_trace = local_environment.wind_speed_ms
        wind_trace = local_environment.utci_wind_speed_10m_ms
        wind_u_trace = local_environment.velocity_u_ms
        wind_v_trace = local_environment.velocity_v_ms
        wind_w_trace = local_environment.velocity_w_ms

        # UTCI along the route -- SAME pythermalcomfort call as stage 07,
        # vectorized over all route points at once.
        # Units/range guard: catches an RH fraction-vs-percent slip, kelvin
        # air/Tmrt, or km/h wind before they become plausible-looking UTCI.
        check_utci_inputs(ta_trace, tmrt_trace, wind_trace, rh_trace,
                          f"stage 08 UTCI, route {route_id}")
        utci_trace = utci(tdb=ta_trace, tr=tmrt_trace,
                          v=wind_trace, rh=rh_trace,
                          limit_inputs=False).utci
        utci_trace = np.asarray(utci_trace, dtype=float)

        if contribution_matrices is not None:
            person_emissivity = float(
                contribution_metadata["person_longwave_emissivity"])
            sampled_flux, interpolation_scale = sample_route_contribution_matrices(
                contribution_matrices, time_hours, arrival_hour, nearest_idx,
                tmrt_trace, person_emissivity=person_emissivity, sigma=SIGMA,
                validation=contribution_config["validation"])
            timestamps = (
                pd.Timestamp(times_df["time"].iloc[0]).normalize()
                + pd.to_timedelta(arrival_hour, unit="h"))
            frame_data = {
                "route_id": np.full(n_pts, route_id, dtype=int),
                "point_id": np.arange(n_pts, dtype=int),
                "original_route_index": np.arange(n_pts, dtype=int),
                "time": np.asarray(timestamps.astype(str)),
                "x": xy[:, 0],
                "y": xy[:, 1],
                "z": mrt_xyz[nearest_idx, 2],
                "distance_along_route_m": cumdist,
                "arrival_hour": arrival_hour,
            }
            frame_data.update(sampled_flux)
            frame_data.update({
                "mrt_C": tmrt_trace,
                "mrt_from_recorded_flux_C": (
                    np.power(sampled_flux["total_absorbed_radiant_flux_Wm2"] /
                             (person_emissivity * SIGMA), 0.25) - 273.15),
                "utci_C": utci_trace,
                "air_temperature_C": ta_trace,
                "relative_humidity_pct": rh_trace,
                "wind_speed_ms": wind_trace,
                "local_wind_speed_ms": local_wind_trace,
                "wind_u_ms": wind_u_trace,
                "wind_v_ms": wind_v_trace,
                "wind_w_ms": wind_w_trace,
                "direct_transmission": np.array([
                    np.interp(hour, time_hours,
                              direct_transmission_matrix[:, point_index], period=24.0)
                    for hour, point_index in zip(h, nearest_idx)
                ]),
                "route_interpolation_closure_scale": interpolation_scale,
                "nearest_mrt_point_index": nearest_idx,
            })
            contribution_route_frames[route_id] = pd.DataFrame(frame_data)

        walk_duration_min = (arrival_hour[-1] - arrival_hour[0]) * 60.0
        # exposure "dose" above the strong-heat-stress threshold, in
        # UTCI-degree-minutes (integral of max(0, UTCI-32) dt over the walk)
        dt_min = np.diff(arrival_hour) * 60.0
        excess = np.maximum(0.0, 0.5 * (utci_trace[1:] + utci_trace[:-1])
                            - UTCI_STRONG_STRESS_C)
        strong_stress_dose_degmin = float(np.sum(excess * dt_min))

        results.append({
            "route_id": route_id,
            "route_name": route["name"],
            "timing_source": timing_source,
            "xy": xy,
            "route_input_frame": route["frame"],
            "cumdist_m": cumdist,
            "arrival_hour": arrival_hour,
            "tmrt_trace_c": tmrt_trace,
            "utci_trace_c": utci_trace,
            "ta_trace_c": ta_trace,
            "rh_trace_pct": rh_trace,
            "wind_trace_ms": wind_trace,
            "local_wind_trace_ms": local_wind_trace,
            "wind_u_trace_ms": wind_u_trace,
            "wind_v_trace_ms": wind_v_trace,
            "wind_w_trace_ms": wind_w_trace,
            "length_m": route["length_m"],
            "walk_duration_min": walk_duration_min,
            "mean_utci_c": float(np.mean(utci_trace)),
            "max_utci_c": float(np.max(utci_trace)),
            "mean_tmrt_c": float(np.mean(tmrt_trace)),
            "max_tmrt_c": float(np.max(tmrt_trace)),
            "strong_stress_dose_degmin": strong_stress_dose_degmin,
        })
        print(f"  Route {route_id} ({route['name']}): {route['length_m']:.0f} m, "
              f"{walk_duration_min:.1f} min [{timing_source}], "
              f"mean UTCI = {results[-1]['mean_utci_c']:.1f} C, "
              f"peak UTCI = {results[-1]['max_utci_c']:.1f} C "
              f"[Ta {ta_trace.mean():.1f} C, RH {rh_trace.mean():.0f} %, "
              f"wind {wind_trace.mean():.1f} m/s]")

    # Rank routes by MEAN UTCI (primary), then PEAK UTCI (tie-break);
    # lower = cooler / more comfortable.
    ranking = sorted(results, key=lambda r: (r["mean_utci_c"], r["max_utci_c"]))
    print("\nRoute ranking by UTCI exposure (coolest to hottest):")
    for rank, r in enumerate(ranking):
        print(f"  #{rank+1}: Route {r['route_id']} -- "
              f"mean UTCI {r['mean_utci_c']:.1f} C, peak UTCI {r['max_utci_c']:.1f} C, "
              f"mean Tmrt {r['mean_tmrt_c']:.1f} C")

    # ---- Save results ----
    summary_rows = [{
        "route_id": r["route_id"], "route_name": r["route_name"],
        "timing_source": r["timing_source"], "length_m": r["length_m"],
        "walk_duration_min": r["walk_duration_min"],
        "mean_utci_c": r["mean_utci_c"], "max_utci_c": r["max_utci_c"],
        "mean_tmrt_c": r["mean_tmrt_c"], "max_tmrt_c": r["max_tmrt_c"],
        "strong_stress_dose_degmin": r["strong_stress_dose_degmin"],
        "mean_ta_c": float(np.mean(r["ta_trace_c"])),
        "mean_rh_pct": float(np.mean(r["rh_trace_pct"])),
        "mean_wind_ms": float(np.mean(r["wind_trace_ms"])),
        "forcing_source": ("csv" if weather.provenance()["all_from_csv"]
                           else "PARTLY-PARAMETRIC"),
    } for r in results]
    pd.DataFrame(summary_rows).sort_values(["mean_utci_c", "max_utci_c"]).to_csv(
        out_dir / "route_ranking_summary.csv", index=False)

    if contribution_route_frames:
        flux_out = out_dir / "radiant_flux_contributions"
        flux_out.mkdir(parents=True, exist_ok=True)
        stale = remove_stale_route_outputs(
            flux_out, contribution_route_frames.keys())
        if stale:
            print(f"  Removed {len(stale)} stale per-route radiant-flux output(s) "
                  "from earlier route sets")
        if contribution_config["export"]["receptor_csv"]:
            for route_id, frame in contribution_route_frames.items():
                frame.to_csv(
                    flux_out / f"route_{route_id}_radiant_flux_contributions.csv",
                    index=False)
        if contribution_config["export"]["route_summary_csv"]:
            route_contribution_summary(
                contribution_route_frames, contribution_config).to_csv(
                    flux_out / "route_radiant_flux_contribution_summary.csv",
                    index=False)
        annotations = {
            r["route_id"]: (f"Length {r['length_m']:.0f} m; "
                            f"walking time {r['walk_duration_min']:.1f} min")
            for r in results
        }
        saved_flux_figures = plot_ranked_radiant_flux_contributions(
            contribution_route_frames, flux_out, contribution_config,
            route_annotations=annotations)
        saved_surface_lw_figures, surface_lw_classes = (
            plot_surface_longwave_classifications(
                contribution_route_frames, flux_out, contribution_config,
                route_annotations=annotations))
        grouping = {}
        for route_id, frame in contribution_route_frames.items():
            _, _, grouped = aggregate_plot_categories(
                frame, contribution_config["plot_mode"], contribution_config)
            grouping[str(route_id)] = grouped
        import json
        (flux_out / "radiant_flux_plot_manifest.json").write_text(
            json.dumps({
                "units": "absorbed W m^-2",
                "plot_mode": contribution_config["plot_mode"],
                "sorting": ("each contribution and total independently sorted "
                            "high-to-low; visualization copies only"),
                "plot_style": "line curves; no filled or stacked areas",
                "surface_longwave_classification_plot": {
                    "contents": ("total surface longwave plus nonzero available "
                                 "surface-source classifications only"),
                    "sorting": ("each class and total surface longwave independently "
                                "sorted high-to-low; visualization copies only"),
                    "classes_by_route": surface_lw_classes,
                    "figures": [str(path) for path in saved_surface_lw_figures],
                },
                "source_metadata": contribution_metadata,
                "categories_grouped_into_other_by_route": grouping,
                "figures": [str(path) for path in saved_flux_figures],
            }, indent=2), encoding="utf-8")
        print(f"  Absorbed-flux route tables and figures: {flux_out}")

    # ---- Export route geometries for external validation software ----
    export_routes_for_gis(results, out_dir, origin, args.project_crs)

    # ---- Visualization 1: spatial map of all input routes ----
    building_segments = None
    if args.buildings_stl:
        import trimesh
        mesh = trimesh.load(str(args.buildings_stl), force="mesh")
        edges = mesh.edges_unique
        building_segments = mesh.vertices[:, :2][edges]

    route_colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    fig, ax = plt.subplots(figsize=(11, 9))
    if building_segments is not None:
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(building_segments, colors="lightgray", linewidths=0.4))
    for r, color in zip(results, route_colors):
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], "-", color=color, linewidth=2.5,
                label=f"Route {r['route_id']}: mean UTCI {r['mean_utci_c']:.1f}\u00b0C, "
                      f"peak {r['max_utci_c']:.1f}\u00b0C "
                      f"({r['length_m']:.0f} m, {r['walk_duration_min']:.0f} min)")
    starts = np.array([r["xy"][0] for r in results])
    ends = np.array([r["xy"][-1] for r in results])
    ax.scatter(starts[:, 0], starts[:, 1], marker="o", s=90, color="black",
               zorder=5, label="Route start(s)")
    ax.scatter(ends[:, 0], ends[:, 1], marker="s", s=90, color="black",
               zorder=5, label="Route end(s)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    timing_label = (
        "recorded per-route observation times"
        if any(r["timing_source"] == "recorded_device_timestamps" for r in results)
        else f"departure at {args.departure_hour:.0f}:00"
    )
    ax.set_title(f"{len(results)} input routes, {timing_label} -- UTCI exposure comparison")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    fig.tight_layout()
    fig.savefig(out_dir / "routes_map.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'routes_map.png'}")

    # ---- Visualization 2: UTCI vs distance walked, all routes overlaid ----
    # (UTCI is the requested route-detail quantity; Tmrt shown beneath for
    #  physical context. No core-temperature rise here -- that is JOS-3 only.)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for r, color in zip(results, route_colors):
        axes[0].plot(r["cumdist_m"], r["utci_trace_c"],
                     color=color, linewidth=2,
                     label=f"Route {r['route_id']} (mean {r['mean_utci_c']:.1f}\u00b0C)")
        axes[1].plot(r["cumdist_m"], r["tmrt_trace_c"], color=color, linewidth=1.5, alpha=0.8)
    # UTCI thermal-stress category reference lines
    for thr, lab in [(26, "moderate"), (32, "strong"), (38, "very strong")]:
        axes[0].axhline(thr, color="gray", linewidth=0.6, linestyle="--")
        axes[0].text(0.0, thr, f" {lab} heat stress \u2265{thr}\u00b0C",
                     fontsize=7, color="gray", va="bottom")
    axes[0].set_ylabel("UTCI [\u00b0C]")
    axes[0].legend(fontsize=9)
    axes[0].set_title("UTCI encountered along each route "
                      "(at each point's actual arrival time)")
    axes[1].set_ylabel("Local Tmrt [\u00b0C]")
    axes[1].set_xlabel("Distance walked [m]")
    axes[1].set_title("Tmrt along each route (radiant context)")
    fig.tight_layout()
    fig.savefig(out_dir / "utci_along_route_comparison.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {out_dir / 'utci_along_route_comparison.png'}")

    print(f"\n[route_result] n_routes={len(results)} start={start_node} end={end_node} "
          f"connectivity={connectivity} output_dir={out_dir}")


if __name__ == "__main__":
    main()
