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


# UTCI thermal-stress category boundaries (deg C) for reporting a route's
# exposure in physiologically meaningful terms (Brode et al. 2012).
UTCI_STRONG_STRESS_C = 32.0   # >= this = "strong heat stress" or worse


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

        # Local Tmrt and air temperature at each point's actual arrival time
        h = arrival_hour % 24.0
        tmrt_trace = np.empty(n_pts)
        for j in range(n_pts):
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_trace[j] = np.interp(h[j], time_hours, tmrt_series, period=24.0)
        ta_trace = air_temperature_c(h, args.air_temp_mean_c, args.air_temp_amp_c,
                                     args.air_temp_peak_hour)

        # UTCI along the route -- SAME pythermalcomfort call as stage 07,
        # vectorized over all route points at once.
        utci_trace = utci(tdb=ta_trace, tr=tmrt_trace,
                          v=args.wind_speed_ms, rh=args.relative_humidity_pct,
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
              f"peak UTCI = {results[-1]['max_utci_c']:.1f} C")

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
    } for r in results]
    pd.DataFrame(summary_rows).sort_values(["mean_utci_c", "max_utci_c"]).to_csv(
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
