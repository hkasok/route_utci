"""
08_route_thermal_stress.py -- find edge-disjoint corner-to-corner routes
through the pedestrian network and compare their UTCI EXPOSURE as a
person walks each route (encountering different shade/sun at different
times along the way).

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

ROUTE FINDING: "edge-disjoint" is verified via networkx's max-flow-based
algorithm (not just a greedy heuristic), and corner selection
automatically searches nearby candidate node pairs for one whose graph
edge-connectivity can actually support the requested number of disjoint
routes.

Run:
    python3 08_route_thermal_stress.py \
        --graphml osm_paths/pedestrian_network.graphml \
        --mrt-results-dir mrt_network_output/ \
        --output-dir route_stress_output/ \
        --buildings-stl out_full/02_final/building_final.stl \
        --departure-hour 8.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pythermalcomfort.models import utci

from weather_provider import add_weather_args, provider_from_args


# UTCI thermal-stress category boundaries (deg C) for reporting a route's
# exposure in physiologically meaningful terms (Brode et al. 2012).
UTCI_STRONG_STRESS_C = 32.0   # >= this = "strong heat stress" or worse


def report_forcing(weather, args, out_dir):
    """Print, and persist to disk, exactly where each UTCI driver came from.

    Motivation: air temperature, humidity and wind are spatially uniform along
    a route, so a forcing mismatch does not distort the SHAPE of the UTCI
    trace at all -- it shifts the whole curve by a constant. Against a
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
    (out_dir / "forcing_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Wrote {out_dir / 'forcing_provenance.json'}")


def export_routes_for_gis(results, out_dir, origin, project_crs):
    """Write each route's geometry in formats other software can validate
    against: GeoJSON (lat/lon, universal), a per-vertex CSV (lat/lon +
    projected + UTCI/Tmrt/arrival time along the route), and GPX tracks.

    Local frame -> true coordinates: add the origin shift back to recover
    projected CRS coordinates, then reproject to EPSG:4326 (lat/lon).
    """
    import json
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(project_crs, "EPSG:4326", always_xy=True)

    features, csv_rows, gpx_tracks = [], [], []
    for r in results:
        xy_proj = r["xy"] + origin                     # back to projected CRS
        lon, lat = to_wgs84.transform(xy_proj[:, 0], xy_proj[:, 1])
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
# Route finding
# ============================================================
def find_best_corner_pair(G_simple, pos, corner_a_xy, corner_b_xy, k_needed, search_n=20,
                           check_top=8):
    cand_a = sorted(G_simple.nodes(), key=lambda n: (pos[n][0]-corner_a_xy[0])**2 + (pos[n][1]-corner_a_xy[1])**2)[:search_n]
    cand_b = sorted(G_simple.nodes(), key=lambda n: (pos[n][0]-corner_b_xy[0])**2 + (pos[n][1]-corner_b_xy[1])**2)[:search_n]
    best = None
    for a in cand_a[:check_top]:
        for b in cand_b[:check_top]:
            if a == b:
                continue
            try:
                conn = nx.edge_connectivity(G_simple, a, b)
            except nx.NetworkXError:
                continue
            score = (conn >= k_needed, conn)
            if best is None or score > best[0]:
                best = (score, a, b, conn)
    if best is None:
        raise RuntimeError("Could not find any valid corner pair in this graph.")
    return best[1], best[2], best[3]


def reconstruct_route_xy(G_multi, node_path, ds=1.0):
    """Walk the actual edge geometries (not straight node-to-node lines) to
    get a densely-sampled (x,y) polyline for the route."""
    all_xy = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = G_multi.get_edge_data(u, v)
        if edge_data is None:
            edge_data = G_multi.get_edge_data(v, u)
        data = list(edge_data.values())[0]
        if "geometry" in data:
            coords = np.array(data["geometry"].coords)
        else:
            coords = np.array([[G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]],
                                [G_multi.nodes[v]["x"], G_multi.nodes[v]["y"]]])
        # ensure orientation matches u -> v
        if np.linalg.norm(coords[0] - [G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]]) > \
           np.linalg.norm(coords[-1] - [G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]]):
            coords = coords[::-1]
        all_xy.append(coords)
    full = np.vstack(all_xy)
    # densify to ~ds spacing
    seg_lens = np.linalg.norm(np.diff(full, axis=0), axis=1)
    cumlen = np.concatenate(([0], np.cumsum(seg_lens)))
    total = cumlen[-1]
    svals = np.arange(0, total, ds)
    dense = np.empty((len(svals), 2))
    j = 0
    for i, s in enumerate(svals):
        while j < len(seg_lens) - 1 and s > cumlen[j + 1]:
            j += 1
        frac = 0 if seg_lens[j] == 0 else (s - cumlen[j]) / seg_lens[j]
        dense[i] = full[j] + frac * (full[j + 1] - full[j])
    return dense, total


def parse_args():
    p = argparse.ArgumentParser(description="5-route thermal stress comparison")
    p.add_argument("--graphml", required=True, help="pedestrian_network.graphml")
    p.add_argument("--mrt-results-dir", required=True, help="Output dir from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None)

    p.add_argument("--n-routes", type=int, default=5)

    # --- Explicit start / end control (optional) ---
    # By default the two opposite BBOX corners drive endpoint selection.
    # Provide either lat/lon OR local-frame x/y to pin the endpoints.
    p.add_argument("--start-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"),
                    help="Explicit route START as latitude longitude "
                         "(overrides the auto bottom-left corner).")
    p.add_argument("--end-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"),
                    help="Explicit route END as latitude longitude "
                         "(overrides the auto top-right corner).")
    p.add_argument("--start-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="Explicit START in the LOCAL (origin-shifted) frame, "
                         "meters. Alternative to --start-latlon.")
    p.add_argument("--end-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="Explicit END in the LOCAL frame, meters.")

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
    p.add_argument("--route-sample-spacing-m", type=float, default=1.0,
                    help="Spatial resampling interval along each route, meters -- the "
                         "actual heat-balance integration timestep is derived from real "
                         "elapsed walk time between consecutive samples (spacing / walking "
                         "speed), not a fixed value, so this controls resolution not "
                         "physical accuracy directly (default: 1.0)")

    add_weather_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weather = provider_from_args(args)
    report_forcing(weather, args, out_dir)

    print("Loading routable network...")
    G_multi = ox.load_graphml(args.graphml)
    G_simple = nx.Graph(G_multi)  # collapse to simple undirected for connectivity analysis
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G_multi.nodes(data=True)}
    print(f"  {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    # Resolve explicit endpoints (lat/lon or local xy) to LOCAL-frame points.
    origin = np.array([args.local_origin_x, args.local_origin_y])

    def latlon_to_local(lat, lon):
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", args.project_crs, always_xy=True)
        X, Y = tf.transform(lon, lat)          # note: always_xy -> (lon,lat)
        return np.array([X, Y]) - origin

    def nearest_node(pt_local):
        node_ids = list(pos.keys())
        P = np.array([pos[n] for n in node_ids])
        d = np.hypot(P[:, 0] - pt_local[0], P[:, 1] - pt_local[1])
        return node_ids[int(np.argmin(d))]

    start_pt = end_pt = None
    if args.start_latlon is not None:
        start_pt = latlon_to_local(*args.start_latlon)
    elif args.start_xy is not None:
        start_pt = np.array(args.start_xy, dtype=float)
    if args.end_latlon is not None:
        end_pt = latlon_to_local(*args.end_latlon)
    elif args.end_xy is not None:
        end_pt = np.array(args.end_xy, dtype=float)

    if start_pt is not None and end_pt is not None:
        start_node = nearest_node(start_pt)
        end_node = nearest_node(end_pt)
        connectivity = nx.edge_connectivity(G_simple, start_node, end_node)
        print(f"\nUsing EXPLICIT endpoints:")
        print(f"  Start: node {start_node} at {pos[start_node]} "
              f"(requested local {tuple(np.round(start_pt, 1))})")
        print(f"  End:   node {end_node} at {pos[end_node]} "
              f"(requested local {tuple(np.round(end_pt, 1))})")
        print(f"  Edge connectivity between them: {connectivity}")
        if connectivity < args.n_routes:
            print(f"  WARNING: only {connectivity} edge-disjoint routes exist "
                  f"between these endpoints (< requested {args.n_routes}); "
                  f"will return {connectivity}.")
    else:
        if start_pt is not None or end_pt is not None:
            print("  NOTE: provide BOTH start and end to pin endpoints; "
                  "falling back to automatic corner selection.")
        print(f"\nSearching for a corner pair supporting {args.n_routes} "
              f"edge-disjoint routes...")
        start_node, end_node, connectivity = find_best_corner_pair(
            G_simple, pos, (xmin, ymin), (xmax, ymax), args.n_routes
        )
        print(f"  Start: node {start_node} at {pos[start_node]}")
        print(f"  End:   node {end_node} at {pos[end_node]}")
        print(f"  Edge connectivity: {connectivity} "
              f"({'>= requested' if connectivity >= args.n_routes else 'LESS than requested'} {args.n_routes})")

    node_paths = list(nx.edge_disjoint_paths(G_simple, start_node, end_node))[:args.n_routes]
    print(f"  Found {len(node_paths)} routes")

    print("\nReconstructing route geometries...")
    routes = []
    for i, np_path in enumerate(node_paths):
        xy, length = reconstruct_route_xy(G_multi, np_path, ds=args.route_sample_spacing_m)
        routes.append({"xy": xy, "length_m": length})
        print(f"  Route {i+1}: {length:.0f} m, {len(xy)} sample points")

    print("\nLoading MRT results for nearest-point lookup...")
    mrt_dir = Path(args.mrt_results_dir)
    mrt_xyz = np.load(mrt_dir / "path_xyz.npy")
    tmrt_matrix = np.load(mrt_dir / "tmrt_matrix_C.npy")
    times_df = pd.read_csv(mrt_dir / "times.csv", parse_dates=["time"])
    times = times_df["time"].tolist()
    time_hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in times])
    # unwrap in case times cross midnight boundary at the array edges
    mrt_tree = cKDTree(mrt_xyz[:, :2])

    print("\nComputing UTCI along each route (at each point's arrival time)...")
    results = []
    for i, route in enumerate(routes):
        xy = route["xy"]
        n_pts = len(xy)

        # distance -> arrival time (hours) at each point
        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumdist = np.concatenate(([0], np.cumsum(seg_lens)))
        arrival_hour = args.departure_hour + cumdist / args.walking_speed_ms / 3600.0

        # nearest precomputed Tmrt sample point for each route point
        _, nearest_idx = mrt_tree.query(xy)

        # Local Tmrt at each point's actual arrival time
        h = arrival_hour % 24.0
        tmrt_trace = np.empty(n_pts)
        for j in range(n_pts):
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_trace[j] = np.interp(h[j], time_hours, tmrt_series, period=24.0)
        # Air temperature, RH and wind at each arrival time, interpolated from
        # the weather CSV when one was supplied (parametric otherwise). All
        # three are returned together so the values that enter UTCI are the
        # exact values exported alongside it -- no chance of the reported
        # forcing drifting from the applied forcing.
        #
        # NOTE ON WIND: the UTCI polynomial is defined for wind at 10 m
        # reference height, so weather.csv's wind_ms must be a 10 m value.
        # This matches SOLWEIG, whose compute_utci_grid() documents its
        # `wind` argument as "Wind speed at 10m height". If your CSV holds
        # pedestrian-level wind, convert it before this stage rather than
        # here, so both models consume the identical series.
        ta_trace, rh_trace, wind_trace = weather.forcing_at(h)

        # UTCI along the route -- SAME pythermalcomfort call as stage 07,
        # vectorized over all route points at once.
        utci_trace = utci(tdb=ta_trace, tr=tmrt_trace,
                          v=wind_trace, rh=rh_trace,
                          limit_inputs=False).utci
        utci_trace = np.asarray(utci_trace, dtype=float)

        walk_duration_min = cumdist[-1] / args.walking_speed_ms / 60.0
        # exposure "dose" above the strong-heat-stress threshold, in
        # UTCI-degree-minutes (integral of max(0, UTCI-32) dt over the walk)
        dt_min = np.diff(arrival_hour) * 60.0
        excess = np.maximum(0.0, 0.5 * (utci_trace[1:] + utci_trace[:-1])
                            - UTCI_STRONG_STRESS_C)
        strong_stress_dose_degmin = float(np.sum(excess * dt_min))

        results.append({
            "route_id": i + 1,
            "xy": xy,
            "cumdist_m": cumdist,
            "arrival_hour": arrival_hour,
            "tmrt_trace_c": tmrt_trace,
            "utci_trace_c": utci_trace,
            "ta_trace_c": ta_trace,
            "rh_trace_pct": rh_trace,
            "wind_trace_ms": wind_trace,
            "length_m": route["length_m"],
            "walk_duration_min": walk_duration_min,
            "mean_utci_c": float(np.mean(utci_trace)),
            "max_utci_c": float(np.max(utci_trace)),
            "mean_tmrt_c": float(np.mean(tmrt_trace)),
            "max_tmrt_c": float(np.max(tmrt_trace)),
            "strong_stress_dose_degmin": strong_stress_dose_degmin,
        })
        print(f"  Route {i+1}: {route['length_m']:.0f} m, {walk_duration_min:.1f} min walk, "
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
        "route_id": r["route_id"], "length_m": r["length_m"],
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

    # ---- Export route geometries for external validation software ----
    export_routes_for_gis(results, out_dir, origin, args.project_crs)

    # ---- Visualization 1: spatial map of the 5 routes ----
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
    ax.scatter(*pos[start_node], marker="o", s=150, color="black", zorder=5, label="Start")
    ax.scatter(*pos[end_node], marker="s", s=150, color="black", zorder=5, label="End")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"{len(results)} edge-disjoint routes, departure at "
                 f"{args.departure_hour:.0f}:00 -- UTCI exposure comparison")
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
