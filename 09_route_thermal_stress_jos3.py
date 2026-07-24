"""
09_route_thermal_stress_jos3.py -- find 5 edge-disjoint corner-to-corner
routes, simulate a person walking each (encountering different shade/sun
at different times), and compute CUMULATIVE thermal stress using JOS-3,
a validated 17-segment/multi-node human thermoregulation model.

WHY JOS-3 over the simpler single-compartment model used previously:
JOS-3 (Takahashi et al., Waseda University) is descended from the same
research lineage as the Fiala model underlying UTCI itself, actively
maintained, and available via pythermalcomfort. Crucially, it was
verified DIRECTLY (not assumed from documentation) to genuinely carry
physiological state forward across sequential simulate() calls with
changing boundary conditions -- confirmed by simulating 10 min of shade
followed by 20 min of sun and observing core temperature continue
evolving from the shade-stage endpoint rather than resetting. It also
gives real per-body-segment output (17 segments: head, chest, arms,
hands, thighs, feet, etc.) -- in testing, extremities (hands, feet)
showed core temperature rises 5-10x larger than the torso under the
same sun exposure, a genuine multi-node effect the single-compartment
model could not represent at all.

WHY NOT pythermalcomfort's phs/two_nodes_gagge: both are steady-state
predictors for a person who has equilibrated in ONE FIXED environment,
not a person walking through changing conditions -- same reasoning as
before, JOS-3's explicit simulate(times, dtime) stepping interface is
what makes it usable for a genuine route simulation.

Performance: benchmarked directly at ~2.2 ms per simulation step, so a
~1900-point route (1m spacing) takes ~4s and all 5 routes together
~20s -- no need to coarsen resolution below what was used before.

Run:
    python3 09_route_thermal_stress_jos3.py \
        --graphml osm_paths/pedestrian_network.graphml \
        --mrt-results-dir mrt_network_output/ \
        --output-dir route_stress_jos3_output/ \
        --buildings-stl out_full/02_final/building_final.stl \
        --departure-hour 13.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pythermalcomfort.models import JOS3

from weather_provider import add_weather_args, provider_from_args
from subject_profiles import PROFILES, get_profile, apply_profile_to_model
from route_selection import select_routes, load_selected_routes


CORE_TEMP_SETPOINT_C = 37.0  # used only for reporting "rise from baseline"


# ============================================================
# Route loading / finding
# ============================================================
def get_routes(args):
    """Return (routes, start_xy, end_xy, start_label, end_label, connectivity).

    Prefers the routes selected up front by 04_select_routes.py (so we score
    exactly the routes MRT was ray-traced along). Falls back to selecting them
    from the graphml here if no --routes-pkl was given (backward compatible).
    """
    if args.routes_pkl:
        print(f"Loading pre-selected routes from {args.routes_pkl} ...")
        sel = load_selected_routes(args.routes_pkl)
        routes = [{"xy": np.asarray(r["xy"], dtype=float),
                   "length_m": float(r["length_m"])} for r in sel["routes"]]
        print(f"  Loaded {len(routes)} routes "
              f"(start node {sel['start_node']}, end node {sel['end_node']}, "
              f"connectivity {sel['connectivity']})")
        return (routes, tuple(sel["start_xy"]), tuple(sel["end_xy"]),
                sel["start_node"], sel["end_node"], sel["connectivity"])

    print("No --routes-pkl given; selecting routes from the graphml...")
    if not args.graphml:
        raise SystemExit("ERROR: provide either --routes-pkl or --graphml.")
    import osmnx as ox
    G_multi = ox.load_graphml(args.graphml)
    sel = select_routes(
        G_multi, n_routes=args.n_routes, ds=args.route_sample_spacing_m,
        project_crs=args.project_crs,
        origin=(args.local_origin_x, args.local_origin_y),
        start_latlon=args.start_latlon, end_latlon=args.end_latlon,
        start_xy=args.start_xy, end_xy=args.end_xy,
    )
    routes = [{"xy": r["xy"], "length_m": r["length_m"]} for r in sel["routes"]]
    pos = sel["pos"]
    return (routes, pos[sel["start_node"]], pos[sel["end_node"]],
            sel["start_node"], sel["end_node"], sel["connectivity"])


def parse_args():
    p = argparse.ArgumentParser(description="Route thermal-stress comparison (JOS-3 core temp)")
    p.add_argument("--routes-pkl", default=None,
                    help="selected_routes.pkl from 04_select_routes.py. When given, the "
                         "exact routes MRT was computed along are scored (preferred). If "
                         "omitted, routes are selected here from --graphml (backward compat).")
    p.add_argument("--graphml", default=None,
                    help="pedestrian_network.graphml (only needed if --routes-pkl is omitted)")
    p.add_argument("--mrt-results-dir", required=True, help="Output dir from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None)

    p.add_argument("--n-routes", type=int, default=3)

    # Endpoint control (only used for the --graphml fallback; when --routes-pkl
    # is given the endpoints are already baked into the selected routes).
    p.add_argument("--start-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"), help="Route START as latitude longitude.")
    p.add_argument("--end-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"), help="Route END as latitude longitude.")
    p.add_argument("--start-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"), help="Route START in the LOCAL frame, meters.")
    p.add_argument("--end-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"), help="Route END in the LOCAL frame, meters.")
    p.add_argument("--local-origin-x", type=float, default=0.0,
                    help="Origin-shift X (must match extract_osm_pedestrian_network.py).")
    p.add_argument("--local-origin-y", type=float, default=0.0,
                    help="Origin-shift Y (see --local-origin-x).")
    p.add_argument("--project-crs", default="EPSG:6346",
                    help="Projected CRS of the network/local frame (default EPSG:6346).")
    p.add_argument("--walking-speed-ms", type=float, default=1.3,
                    help="Average adult walking pace (default: 1.3 m/s ~= 4.7 km/h). "
                         "UTCI's reference activity is ~1.1 m/s; 1.3 better matches "
                         "a healthy adult crossing campus. Faster pace also raises "
                         "metabolic heat in JOS-3, which is realistic.")
    p.add_argument("--departure-hour", type=float, default=13.0,
                    help="Hour of day (0-24) the walk begins (default: 13.0, "
                         "solar-afternoon heat. Use 8.0 for a morning walk).")
    p.add_argument("--route-sample-spacing-m", type=float, default=1.0,
                    help="Spatial resampling interval along each route, meters (default: 1.0)")
    p.add_argument("--equilibration-min", type=float, default=10.0,
                    help="Minutes of simulated equilibration at the route's starting "
                         "conditions before 'official' walk timing begins, to avoid "
                         "JOS-3's default initial state creating a startup transient "
                         "(default: 10.0)")
    p.add_argument("--activity-par", type=float, default=2.5,
                    help="Physical activity ratio (metabolic rate / basal rate) for "
                         "walking pace -- JOS-3 default for sitting quietly is 1.2; "
                         "walking ~4-5 km/h is typically 2.5-3.3 per ISO 8996 (default: 2.5)")
    p.add_argument("--subject-profile", default=None,
                    choices=sorted(PROFILES.keys()),
                    help="Literature-backed virtual subject preset "
                         "(healthy_adult, child, elderly_male, "
                         "elderly_female, obese_adult, acclimatized_adult, "
                         "...). Sets age/sex/height/weight/fat, cardiac "
                         "index, and (for acclimatized) a core-setpoint "
                         "shift, all cited in subject_profiles.py. When "
                         "given, it overrides the --person-* values below "
                         "unless you also pass those explicitly. See the "
                         "profile's printed caveat: for elderly/ill the "
                         "model captures only body geometry + perfusion, "
                         "which under-states real risk.")
    p.add_argument("--person-height-m", type=float, default=None)
    p.add_argument("--person-weight-kg", type=float, default=None)
    p.add_argument("--person-age", type=int, default=None)
    p.add_argument("--person-sex", default=None, choices=["male", "female"])

    add_weather_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weather = provider_from_args(args)
    print(f"Weather source: {weather.describe()}")

    # ---- Resolve the virtual subject -------------------------------------
    # Precedence: an explicit --person-* flag always wins; otherwise the
    # --subject-profile preset supplies the value; otherwise the healthy
    # default. This lets you pick a preset and still tweak one field.
    if args.subject_profile:
        prof = get_profile(args.subject_profile)
        print(f"\nSubject profile: {prof.label}")
        print(f"  rationale: {prof.rationale}")
        if prof.caveat:
            print(f"  CAVEAT: {prof.caveat}")
    else:
        prof = None
    subj_height = (args.person_height_m if args.person_height_m is not None
                   else (prof.height if prof else 1.72))
    subj_weight = (args.person_weight_kg if args.person_weight_kg is not None
                   else (prof.weight if prof else 74.0))
    subj_age = (args.person_age if args.person_age is not None
                else (prof.age if prof else 30))
    subj_sex = (args.person_sex if args.person_sex is not None
                else (prof.sex if prof else "male"))
    subj_fat = prof.fat if prof else 15.0
    subj_ci = prof.ci if prof else 2.59       # JOS-3 default cardiac index
    subj_setpoint_shift = prof.setpoint_shift_c if prof else 0.0
    print(f"  body: age {subj_age}, {subj_sex}, {subj_height} m, "
          f"{subj_weight} kg, fat {subj_fat}%, cardiac index {subj_ci} "
          f"L/min/m^2, setpoint shift {subj_setpoint_shift:+.2f} C")

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

    print("\nSimulating the walk along each route with JOS-3 "
          f"(activity par={args.activity_par}, equilibration={args.equilibration_min} min)...")
    results = []
    for i, route in enumerate(routes):
        xy = route["xy"]
        n_pts = len(xy)

        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumdist = np.concatenate(([0], np.cumsum(seg_lens)))
        arrival_hour = args.departure_hour + cumdist / args.walking_speed_ms / 3600.0

        _, nearest_idx = mrt_tree.query(xy)

        model = JOS3(height=subj_height, weight=subj_weight,
                     age=subj_age, sex=subj_sex, fat=subj_fat, ci=subj_ci)
        if subj_setpoint_shift:
            model.cr_set_point = model.cr_set_point + subj_setpoint_shift
        model.par = args.activity_par
        # surface-area weights for a single scalar "whole-body mean core temp"
        # summary metric from the 17 segment values
        bsa_weights = model.bsa / model.bsa.sum()

        def weighted_core_c(m):
            return float(np.sum(m.t_core * bsa_weights))

        baseline_core_c = weighted_core_c(model)

        # Equilibration: hold at the route's starting conditions before the
        # "official" walk timing begins, to avoid JOS-3's default initial
        # state creating a startup transient in the results.
        h0 = args.departure_hour % 24.0
        tmrt0 = np.interp(h0, time_hours, tmrt_matrix[:, nearest_idx[0]], period=24.0)
        ta0 = weather.air_temp_c(h0)
        model.tdb, model.tr = ta0, tmrt0
        model.rh, model.v = weather.rh_pct(h0), weather.wind_ms(h0)
        if args.equilibration_min > 0:
            model.simulate(times=int(args.equilibration_min), dtime=60, output=False)
        start_core_c = weighted_core_c(model)

        tcore_trace = np.zeros(n_pts)
        tmrt_trace = np.zeros(n_pts)
        hand_foot_trace = np.zeros(n_pts)  # extremity segments, shown separately

        for j in range(n_pts):
            h = arrival_hour[j] % 24.0
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_now = np.interp(h, time_hours, tmrt_series, period=24.0)
            ta_now = weather.air_temp_c(h)
            dt_s = (arrival_hour[j] - arrival_hour[j - 1]) * 3600.0 if j > 0 else 0.0

            model.tdb, model.tr = ta_now, tmrt_now
            model.rh, model.v = weather.rh_pct(h), weather.wind_ms(h)
            if dt_s > 0:
                model.simulate(times=1, dtime=dt_s, output=False)

            tcore_trace[j] = weighted_core_c(model)
            tmrt_trace[j] = tmrt_now
            # JOS-3 body_names order includes hand/foot segments -- average them
            # as a simple "extremity strain" indicator, shown alongside core.
            idx_extreme = [k for k, name in enumerate(model.body_names)
                            if "hand" in name or "foot" in name]
            hand_foot_trace[j] = float(np.mean(model.t_core[idx_extreme])) if idx_extreme else np.nan

        walk_duration_min = cumdist[-1] / args.walking_speed_ms / 60.0
        results.append({
            "route_id": i + 1,
            "xy": xy,
            "cumdist_m": cumdist,
            "arrival_hour": arrival_hour,
            "tcore_trace_c": tcore_trace,
            "tmrt_trace_c": tmrt_trace,
            "hand_foot_trace_c": hand_foot_trace,
            "length_m": route["length_m"],
            "walk_duration_min": walk_duration_min,
            "final_tcore_rise_c": tcore_trace[-1] - start_core_c,
            "final_tcore_c": tcore_trace[-1],
            "mean_tmrt_c": float(np.mean(tmrt_trace)),
            "max_tmrt_c": float(np.max(tmrt_trace)),
            "final_extremity_rise_c": hand_foot_trace[-1] - start_core_c,
        })
        print(f"  Route {i+1}: {route['length_m']:.0f} m, {walk_duration_min:.1f} min walk, "
              f"final core temp rise = {results[-1]['final_tcore_rise_c']:+.3f} C "
              f"(extremities {results[-1]['final_extremity_rise_c']:+.3f} C)")

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
        "final_extremity_rise_c": r["final_extremity_rise_c"],
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
    ax.scatter(*start_xy, marker="o", s=150, color="black", zorder=5, label="Start")
    ax.scatter(*end_xy, marker="s", s=150, color="black", zorder=5, label="End")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"{len(results)} edge-disjoint routes, departure at {args.departure_hour:.0f}:00 -- "
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
