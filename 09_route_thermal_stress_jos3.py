"""
09_route_thermal_stress_jos3.py -- load source-agnostic input routes,
simulate a person walking each (encountering different shade/sun
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
hands, thighs, feet, etc.) -- extremities (hands, feet) respond far
more strongly than the torso to both sun and cold, a genuine
multi-node effect the single-compartment model could not represent
at all.

EXTREMITY REPORTING -- LIKE FOR LIKE
------------------------------------
Extremity tissue sits several degrees below body core even in perfect
thermal balance (about 0.4 C below when hot and vasodilated, about 4.5 C
below when cool and vasoconstricted). Any metric that subtracts a body-core
temperature from an extremity temperature therefore reports that standing
offset as if it were strain, and its size changes with vasomotor state.
Every extremity number here compares one quantity with itself:
``*_change_c`` is end minus that same quantity's own start,
``final_*_temp`` values are absolute, and the core-to-extremity gradient is
taken at a single instant, where it is the meaningful physiological measure
of peripheral vasoconstriction.

WHY NOT pythermalcomfort's phs/two_nodes_gagge: both are steady-state
predictors for a person who has equilibrated in ONE FIXED environment,
not a person walking through changing conditions -- same reasoning as
before, JOS-3's explicit simulate(times, dtime) stepping interface is
what makes it usable for a genuine route simulation.

Performance scales with the number and point count of input routes; route
generation is handled separately by generate_route.py.

Run:
    python3 09_route_thermal_stress_jos3.py \
        --routes-dir "input/MMC/routes" \
        --mrt-results-dir mrt_network_output/ \
        --output-dir route_stress_jos3_output/ \
        --buildings-stl input/MMC/geometry/building_final.stl \
        --departure-hour 13.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pythermalcomfort.models import JOS3

from weather_provider import add_weather_args, provider_from_args
from microclimate_field import EnvironmentField, add_microclimate_argument
from subject_profiles import PROFILES, get_profile, apply_profile_to_model
import clothing_profiles
from generate_route import load_routes_directory, route_arrival_schedule
from physical_checks import check_jos3_inputs


CORE_TEMP_SETPOINT_C = 37.0  # used only for reporting "rise from baseline"


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
    return (routes, tuple(routes[0]["xy"][0]), tuple(routes[0]["xy"][-1]),
            "input_routes", "input_routes", "not_applicable")


def parse_args():
    p = argparse.ArgumentParser(description="Route thermal-stress comparison (JOS-3 core temp)")
    p.add_argument("--routes-dir", default="input/MMC/routes",
                    help="Folder containing route_<id>.csv/json inputs "
                         "(default: 'input/MMC/routes')")
    p.add_argument("--mrt-results-dir", required=True, help="Output dir from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None)

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
    clothing_profiles.add_clothing_arguments(p)
    p.add_argument("--person-height-m", type=float, default=None)
    p.add_argument("--person-weight-kg", type=float, default=None)
    p.add_argument("--person-age", type=int, default=None)
    p.add_argument("--person-sex", default=None, choices=["male", "female"])

    add_weather_args(p)
    add_microclimate_argument(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weather = provider_from_args(args)
    environment = EnvironmentField(
        weather, args.microclimate_dir, args.microclimate_receptor_height_m)
    print(f"Air temperature / velocity forcing: {environment.describe()}")
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

    # Clothing is chosen per route from that walk's own mean conditions, so a
    # night walk and a noon walk over the same street are dressed differently.
    clothing_config = clothing_profiles.load_config(args.clothing_config)
    solar_elevation_deg = (times_df["elevation_deg"].to_numpy(float)
                           if "elevation_deg" in times_df.columns else None)
    if solar_elevation_deg is None:
        print("  NOTE: times.csv has no elevation_deg column; day/night for "
              "clothing selection falls back to 07:00-19:00 local hours.")

    def walk_clothing(route_id, xy_route, nearest, hours):
        """Resolve the clothing worn for one walk and why."""
        sample_xyz = np.column_stack(
            (xy_route[:, 0], xy_route[:, 1], mrt_xyz[nearest, 2]))
        conditions = environment.sample(sample_xyz, hours % 24.0)
        mean_ta = float(np.mean(conditions.air_temperature_c))
        mean_wind = float(np.mean(conditions.wind_speed_ms))
        mean_hour = float(np.mean(hours)) % 24.0
        if solar_elevation_deg is not None:
            elevation = float(np.interp(mean_hour, time_hours,
                                        solar_elevation_deg, period=24.0))
            daytime = elevation > 0.0
        else:
            elevation = float("nan")
            daytime = 7.0 <= mean_hour < 19.0
        segment_clo, provenance = clothing_profiles.resolve(
            args.clothing, air_temp_c=mean_ta, is_daytime=daytime,
            wind_ms=mean_wind, climate=args.clothing_climate,
            config=clothing_config)
        provenance.update({"route_id": route_id,
                           "solar_elevation_deg": elevation,
                           "mean_walk_hour": mean_hour,
                           "requested": args.clothing,
                           "segment_clo": [float(v) for v in segment_clo]})
        return segment_clo, provenance

    print("\nSimulating the walk along each route with JOS-3 "
          f"(activity par={args.activity_par}, equilibration={args.equilibration_min} min, "
          f"clothing={args.clothing} [{args.clothing_climate}])...")
    results = []
    for i, route in enumerate(routes):
        route_id = route["route_id"]
        xy = route["xy"]
        n_pts = len(xy)

        seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumdist = np.concatenate(([0], np.cumsum(seg_lens)))
        arrival_hour, timing_source = route_arrival_schedule(
            route, args.departure_hour, args.walking_speed_ms)

        _, nearest_idx = mrt_tree.query(xy)

        model = JOS3(height=subj_height, weight=subj_weight,
                     age=subj_age, sex=subj_sex, fat=subj_fat, ci=subj_ci)
        if subj_setpoint_shift:
            model.cr_set_point = model.cr_set_point + subj_setpoint_shift
        model.par = args.activity_par
        # Clothing must be on the body BEFORE equilibration, otherwise the
        # subject equilibrates naked and then dresses at the start line.
        segment_clo, clothing_provenance = walk_clothing(
            route_id, xy, nearest_idx, arrival_hour)
        if list(getattr(model, "body_names", clothing_profiles.JOS3_SEGMENTS)) \
                != list(clothing_profiles.JOS3_SEGMENTS):
            raise RuntimeError(
                "JOS-3 segment order differs from clothing_profiles."
                "JOS3_SEGMENTS; refusing to apply clothing to the wrong parts")
        model.clo = segment_clo
        # surface-area weights for a single scalar "whole-body mean core temp"
        # summary metric from the 17 segment values
        bsa_weights = model.bsa / model.bsa.sum()
        clothing_provenance["whole_body_clo"] = clothing_profiles.whole_body_clo(
            segment_clo, model.bsa)
        print(f"  Route {route_id} clothing: "
              f"{clothing_provenance['ensemble_label']} "
              f"({clothing_provenance['whole_body_clo']:.2f} clo whole-body; "
              f"{clothing_provenance['selection']}"
              + (f", dressing temperature "
                 f"{clothing_provenance['dressing_temperature_c']:.1f} C"
                 if "dressing_temperature_c" in clothing_provenance else "")
              + ")")

        def weighted_core_c(m):
            return float(np.sum(m.t_core * bsa_weights))

        baseline_core_c = weighted_core_c(model)

        # Equilibration: hold at the route's starting conditions before the
        # "official" walk timing begins, to avoid JOS-3's default initial
        # state creating a startup transient in the results.
        h0 = arrival_hour[0] % 24.0
        tmrt0 = np.interp(h0, time_hours, tmrt_matrix[:, nearest_idx[0]], period=24.0)
        start_xyz = np.array([xy[0, 0], xy[0, 1],
                              mrt_xyz[nearest_idx[0], 2]])
        start_environment = environment.sample(start_xyz, h0)
        ta0 = float(start_environment.air_temperature_c[0])
        rh0 = float(start_environment.relative_humidity_pct[0])
        v0 = float(start_environment.wind_speed_ms[0])
        # JOS-3's .rh is RELATIVE HUMIDITY IN PERCENT (library default 50), the
        # same convention as UTCI -- guard the units before they enter the
        # thermoregulation model, where a fraction would read as ~0.7% (arid).
        check_jos3_inputs(ta0, tmrt0, v0, rh0,
                          f"stage 09 JOS-3 equilibration, route {route_id}")
        model.tdb, model.tr = ta0, tmrt0
        model.rh, model.v = rh0, v0
        if args.equilibration_min > 0:
            model.simulate(times=int(args.equilibration_min), dtime=60, output=False)
        start_core_c = weighted_core_c(model)

        # Extremity (hand/foot) state is tracked as its OWN quantity, at the
        # same instants as the core trace, so every reported extremity number
        # is a like-for-like comparison. Extremity tissue sits several degrees
        # below body core even in perfect thermal balance -- differencing the
        # two would report that permanent offset as if it were strain.
        idx_extreme = [k for k, name in enumerate(model.body_names)
                       if "hand" in name or "foot" in name]
        if not idx_extreme:
            raise RuntimeError("this JOS-3 build exposes no hand/foot segments")
        extremity_core_c = lambda: float(np.mean(model.t_core[idx_extreme]))
        extremity_skin_c = lambda: float(np.mean(model.t_skin[idx_extreme]))
        start_extremity_core_c = extremity_core_c()
        start_extremity_skin_c = extremity_skin_c()

        tcore_trace = np.zeros(n_pts)
        tmrt_trace = np.zeros(n_pts)
        hand_foot_trace = np.zeros(n_pts)        # extremity tissue temperature
        hand_foot_skin_trace = np.zeros(n_pts)   # extremity skin temperature

        for j in range(n_pts):
            h = arrival_hour[j] % 24.0
            tmrt_series = tmrt_matrix[:, nearest_idx[j]]
            tmrt_now = np.interp(h, time_hours, tmrt_series, period=24.0)
            point_xyz = np.array([xy[j, 0], xy[j, 1],
                                  mrt_xyz[nearest_idx[j], 2]])
            local_environment = environment.sample(point_xyz, h)
            ta_now = float(local_environment.air_temperature_c[0])
            dt_s = (arrival_hour[j] - arrival_hour[j - 1]) * 3600.0 if j > 0 else 0.0

            rh_now = float(local_environment.relative_humidity_pct[0])
            v_now = float(local_environment.wind_speed_ms[0])
            if j == 0:   # guard once per route (uniform drivers, hot loop)
                check_jos3_inputs(ta_now, tmrt_now, v_now, rh_now,
                                  f"stage 09 JOS-3 walk, route {route_id}")
            model.tdb, model.tr = ta_now, tmrt_now
            model.rh, model.v = rh_now, v_now
            if dt_s > 0:
                model.simulate(times=1, dtime=dt_s, output=False)

            tcore_trace[j] = weighted_core_c(model)
            tmrt_trace[j] = tmrt_now
            hand_foot_trace[j] = extremity_core_c()
            hand_foot_skin_trace[j] = extremity_skin_c()

        walk_duration_min = (arrival_hour[-1] - arrival_hour[0]) * 60.0
        results.append({
            "route_id": route_id,
            "route_name": route["name"],
            "timing_source": timing_source,
            "xy": xy,
            "cumdist_m": cumdist,
            "arrival_hour": arrival_hour,
            "tcore_trace_c": tcore_trace,
            "tmrt_trace_c": tmrt_trace,
            "hand_foot_trace_c": hand_foot_trace,
            "hand_foot_skin_trace_c": hand_foot_skin_trace,
            "length_m": route["length_m"],
            "walk_duration_min": walk_duration_min,
            "final_tcore_rise_c": tcore_trace[-1] - start_core_c,
            "final_tcore_c": tcore_trace[-1],
            "mean_tmrt_c": float(np.mean(tmrt_trace)),
            "max_tmrt_c": float(np.max(tmrt_trace)),
            # Extremity strain, all like-for-like:
            #   *_change_c  = same quantity, end minus its own start
            #   *_temp_c    = absolute end-of-walk temperature
            #   gradient    = core minus extremity AT THE SAME INSTANT, the
            #                 physiological core-to-periphery gradient that
            #                 widens with vasoconstriction
            "extremity_core_change_c": hand_foot_trace[-1] - start_extremity_core_c,
            "extremity_skin_change_c": (hand_foot_skin_trace[-1]
                                        - start_extremity_skin_c),
            "final_extremity_core_c": hand_foot_trace[-1],
            "final_extremity_skin_c": hand_foot_skin_trace[-1],
            "final_core_to_extremity_gradient_c": (tcore_trace[-1]
                                                   - hand_foot_trace[-1]),
            "clothing": clothing_provenance,
        })
        print(f"  Route {route_id} ({route['name']}): {route['length_m']:.0f} m, "
              f"{walk_duration_min:.1f} min [{timing_source}], "
              f"final core temp rise = {results[-1]['final_tcore_rise_c']:+.3f} C "
              f"(hand/foot skin {results[-1]['final_extremity_skin_c']:.1f} C, "
              f"{results[-1]['extremity_skin_change_c']:+.2f} C over the walk)")

    # Rank routes by final core temp rise (lower = better/cooler)
    ranking = sorted(results, key=lambda r: r["final_tcore_rise_c"])
    print("\nRoute ranking (best/coolest to worst/hottest):")
    for rank, r in enumerate(ranking):
        print(f"  #{rank+1}: Route {r['route_id']} -- "
              f"final core temp rise {r['final_tcore_rise_c']:+.3f} C, "
              f"mean Tmrt {r['mean_tmrt_c']:.1f} C")

    # ---- Save results ----
    summary_rows = [{
        "route_id": r["route_id"], "route_name": r["route_name"],
        "timing_source": r["timing_source"], "length_m": r["length_m"],
        "walk_duration_min": r["walk_duration_min"],
        "final_tcore_rise_c": r["final_tcore_rise_c"],
        "mean_tmrt_c": r["mean_tmrt_c"], "max_tmrt_c": r["max_tmrt_c"],
        "final_tcore_c": r["final_tcore_c"],
        "final_extremity_skin_c": r["final_extremity_skin_c"],
        "extremity_skin_change_c": r["extremity_skin_change_c"],
        "final_extremity_core_c": r["final_extremity_core_c"],
        "extremity_core_change_c": r["extremity_core_change_c"],
        "final_core_to_extremity_gradient_c":
            r["final_core_to_extremity_gradient_c"],
        "clothing_ensemble": r["clothing"]["ensemble"],
        "clothing_whole_body_clo": r["clothing"]["whole_body_clo"],
        "clothing_selection": r["clothing"]["selection"],
    } for r in results]
    pd.DataFrame(summary_rows).sort_values("final_tcore_rise_c").to_csv(
        out_dir / "route_ranking_summary.csv", index=False)
    # Full clothing provenance: which outfit each walk wore and every term
    # that produced it, so a result is never silently re-clothed.
    (out_dir / "clothing_provenance.json").write_text(json.dumps({
        "requested": args.clothing,
        "climate": args.clothing_climate,
        "config_file": args.clothing_config,
        "segments": list(clothing_profiles.JOS3_SEGMENTS),
        "routes": [r["clothing"] for r in results],
    }, indent=2), encoding="utf-8")

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
                label=f"Route {r['route_id']}: {r['final_tcore_rise_c']:+.2f}\u00b0C core rise "
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
    ax.set_title(f"{len(results)} input routes, {timing_label} -- "
                 "cumulative thermal stress comparison")
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
