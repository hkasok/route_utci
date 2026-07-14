"""
08_route_thermal_stress.py -- find 5 edge-disjoint corner-to-corner routes
through the pedestrian network, simulate a person walking each route
(encountering different shade/sun at different times as they walk), and
compute CUMULATIVE thermal stress along each route using a simple
single-compartment human heat-balance model.

WHY NOT pythermalcomfort's two_nodes_gagge/phs models: those are
steady-state predictors -- they estimate a person's physiological state
after equilibrating in ONE FIXED environment, not a person walking
through CHANGING conditions over time. For genuine cumulative exposure
along a route, this script instead integrates a transparent
single-compartment core-body-temperature heat balance forward in time,
step by step, as the walker's position (and therefore local Tmrt) and
time (and therefore solar/met conditions) both change together. This
mirrors the approach used in published route-based heat-exposure studies
(e.g. core temperature rise along sun vs. shaded routes).

ROUTE FINDING: "edge-disjoint" is verified via networkx's max-flow-based
algorithm (not just a greedy heuristic), and corner selection
automatically searches nearby candidate node pairs for one whose graph
edge-connectivity can actually support 5 disjoint routes -- the literal
nearest-to-corner nodes are not guaranteed to support this (a low-degree
or bottlenecked node cannot, as a matter of graph theory, be the source
of more disjoint paths than its own edge connectivity allows).

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


# ============================================================
# Simple single-compartment heat balance model (see module docstring)
# ============================================================
SIGMA = 5.670374419e-8
BODY_MASS_KG = 70.0
BODY_SPECIFIC_HEAT = 3492.0       # J/(kg*K) -- standard value, Fiala/Gagge-type models
BODY_SURFACE_AREA_M2 = 1.8        # DuBois area
METABOLIC_WALKING_WM2 = 135.0     # W/m^2, walking ~4 km/h (matches UTCI's reference activity)
SKIN_TEMP_ASSUMED_C = 35.0        # fixed assumed skin temp (single-compartment simplification)
CORE_TEMP_SETPOINT_C = 37.0
MAX_SWEAT_COOLING_WM2 = 350.0
SWEAT_GAIN_WM2_PER_C = 150.0


def convective_coefficient(wind_ms):
    return 8.3 * max(wind_ms, 0.1) ** 0.6


def radiative_coefficient(tmrt_c, tskin_c=SKIN_TEMP_ASSUMED_C):
    t_mean_k = (tmrt_c + tskin_c) / 2 + 273.15
    return 4 * 0.97 * SIGMA * t_mean_k ** 3


def evaporative_capacity_wm2(rh_pct, wind_ms):
    rh_factor = np.clip(1.0 - (rh_pct - 30) / 100.0, 0.15, 1.0)
    wind_factor = np.clip(0.5 + 0.5 * wind_ms / 3.0, 0.5, 1.3)
    return MAX_SWEAT_COOLING_WM2 * rh_factor * wind_factor


def step_core_temp(tcore_c, ta_c, tmrt_c, rh_pct, wind_ms, dt_s):
    hc = convective_coefficient(wind_ms)
    hr = radiative_coefficient(tmrt_c)
    C = hc * (ta_c - SKIN_TEMP_ASSUMED_C)
    R = hr * (tmrt_c - SKIN_TEMP_ASSUMED_C)
    E_max = evaporative_capacity_wm2(rh_pct, wind_ms)
    E_demand = max(0.0, SWEAT_GAIN_WM2_PER_C * (tcore_c - CORE_TEMP_SETPOINT_C))
    E = min(E_demand, E_max)
    S_wm2 = METABOLIC_WALKING_WM2 + C + R - E
    S_watts = S_wm2 * BODY_SURFACE_AREA_M2
    dtcore = S_watts / (BODY_MASS_KG * BODY_SPECIFIC_HEAT) * dt_s
    return tcore_c + dtcore, S_wm2, E


def air_temperature_c(hour_of_day, mean_c, amp_c, peak_hour):
    return mean_c + amp_c * np.cos(2.0 * np.pi * (hour_of_day - peak_hour) / 24.0)


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
    p.add_argument("--walking-speed-ms", type=float, default=1.1, help="~4 km/h (default: 1.1)")
    p.add_argument("--departure-hour", type=float, default=8.0,
                    help="Hour of day (0-24) the walk begins (default: 8.0)")
    p.add_argument("--route-sample-spacing-m", type=float, default=1.0,
                    help="Spatial resampling interval along each route, meters -- the "
                         "actual heat-balance integration timestep is derived from real "
                         "elapsed walk time between consecutive samples (spacing / walking "
                         "speed), not a fixed value, so this controls resolution not "
                         "physical accuracy directly (default: 1.0)")

    p.add_argument("--air-temp-mean-c", type=float, default=29.0)
    p.add_argument("--air-temp-amp-c", type=float, default=4.0)
    p.add_argument("--air-temp-peak-hour", type=float, default=15.0)
    p.add_argument("--relative-humidity-pct", type=float, default=70.0)
    p.add_argument("--wind-speed-ms", type=float, default=3.1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading routable network...")
    G_multi = ox.load_graphml(args.graphml)
    G_simple = nx.Graph(G_multi)  # collapse to simple undirected for connectivity analysis
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G_multi.nodes(data=True)}
    print(f"  {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    print(f"\nSearching for a corner pair supporting {args.n_routes} edge-disjoint routes...")
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

    print("\nSimulating the walk along each route (heat-balance integration)...")
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

        tcore = CORE_TEMP_SETPOINT_C
        tcore_trace = np.zeros(n_pts)
        tmrt_trace = np.zeros(n_pts)
        heat_storage_trace = np.zeros(n_pts)

        for j in range(n_pts):
            h = arrival_hour[j] % 24.0
            # interpolate Tmrt at this point's time series to the exact arrival hour
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_now = np.interp(h, time_hours, tmrt_series, period=24.0)
            ta_now = air_temperature_c(h, args.air_temp_mean_c, args.air_temp_amp_c,
                                        args.air_temp_peak_hour)
            dt_s = (arrival_hour[j] - arrival_hour[j - 1]) * 3600.0 if j > 0 else 0.0
            tcore, S, E = step_core_temp(tcore, ta_now, tmrt_now, args.relative_humidity_pct,
                                          args.wind_speed_ms, dt_s)
            tcore_trace[j] = tcore
            tmrt_trace[j] = tmrt_now
            heat_storage_trace[j] = S

        walk_duration_min = cumdist[-1] / args.walking_speed_ms / 60.0
        results.append({
            "route_id": i + 1,
            "xy": xy,
            "cumdist_m": cumdist,
            "arrival_hour": arrival_hour,
            "tcore_trace_c": tcore_trace,
            "tmrt_trace_c": tmrt_trace,
            "heat_storage_trace_wm2": heat_storage_trace,
            "length_m": route["length_m"],
            "walk_duration_min": walk_duration_min,
            "final_tcore_rise_c": tcore_trace[-1] - CORE_TEMP_SETPOINT_C,
            "mean_tmrt_c": float(np.mean(tmrt_trace)),
            "max_tmrt_c": float(np.max(tmrt_trace)),
            "peak_heat_storage_wm2": float(np.max(heat_storage_trace)),
        })
        print(f"  Route {i+1}: {route['length_m']:.0f} m, {walk_duration_min:.1f} min walk, "
              f"final core temp rise = {results[-1]['final_tcore_rise_c']:+.3f} C")

    # Rank routes by final core temp rise (lower = better/cooler)
    ranking = sorted(results, key=lambda r: r["final_tcore_rise_c"])
    print("\nRoute ranking (best/coolest to worst/hottest):")
    for rank, r in enumerate(ranking):
        print(f"  #{rank+1}: Route {r['route_id']} -- "
              f"final core temp rise {r['final_tcore_rise_c']:+.3f} C, "
              f"mean Tmrt {r['mean_tmrt_c']:.1f} C")

    # ---- Save results ----
    summary_rows = [{
        "route_id": r["route_id"], "length_m": r["length_m"],
        "walk_duration_min": r["walk_duration_min"],
        "final_tcore_rise_c": r["final_tcore_rise_c"],
        "mean_tmrt_c": r["mean_tmrt_c"], "max_tmrt_c": r["max_tmrt_c"],
        "peak_heat_storage_wm2": r["peak_heat_storage_wm2"],
    } for r in results]
    pd.DataFrame(summary_rows).sort_values("final_tcore_rise_c").to_csv(
        out_dir / "route_ranking_summary.csv", index=False)

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
                label=f"Route {r['route_id']}: {r['final_tcore_rise_c']:+.2f}\u00b0C core rise "
                      f"({r['length_m']:.0f} m, {r['walk_duration_min']:.0f} min)")
    ax.scatter(*pos[start_node], marker="o", s=150, color="black", zorder=5, label="Start")
    ax.scatter(*pos[end_node], marker="s", s=150, color="black", zorder=5, label="End")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"5 edge-disjoint routes, departure at {args.departure_hour:.0f}:00 -- "
                 f"cumulative thermal stress comparison")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    fig.tight_layout()
    fig.savefig(out_dir / "routes_map.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'routes_map.png'}")

    # ---- Visualization 2: cumulative core temp rise vs distance, all routes overlaid ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for r, color in zip(results, route_colors):
        axes[0].plot(r["cumdist_m"], r["tcore_trace_c"] - CORE_TEMP_SETPOINT_C,
                     color=color, linewidth=2, label=f"Route {r['route_id']}")
        axes[1].plot(r["cumdist_m"], r["tmrt_trace_c"], color=color, linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Core temperature rise [\u00b0C]")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Cumulative thermal strain vs. distance walked")
    axes[1].set_ylabel("Local Tmrt [\u00b0C]")
    axes[1].set_xlabel("Distance walked [m]")
    axes[1].set_title("Tmrt encountered along each route (at each point's actual arrival time)")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_stress_comparison.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {out_dir / 'cumulative_stress_comparison.png'}")

    print(f"\n[route_result] n_routes={len(results)} start={start_node} end={end_node} "
          f"connectivity={connectivity} output_dir={out_dir}")


if __name__ == "__main__":
    main()
