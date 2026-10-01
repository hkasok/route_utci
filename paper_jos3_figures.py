#!/usr/bin/env python3
"""paper_jos3_figures.py -- the campus route-strain figures for the paper.

Three figures, all from pipeline outputs:

  jos3_route_map       each route coloured by Tmrt at the walker's arrival
                       time (stage-09 traces), over building footprints and
                       tree canopy projected from the case geometry
  jos3_strain          (a) core-temperature rise against elapsed time, with the
                       radiation-neutral counterfactual (Tmrt = Ta) dashed;
                       (b) the radiation-attributable rise; (c) a sun/shade
                       strip per route
  jos3_departure_sweep radiation-attributable and total rise for every route
                       against departure time. JOS-3 is re-run through stage
                       09's own simulate_walk, so the physics is identical.

Run from the project root after stage 09:
    python3 paper_jos3_figures.py
"""
import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree

import clothing_profiles
from generate_route import load_routes_directory
from microclimate_field import EnvironmentField
from weather_provider import WeatherProvider

stage09 = import_module("09_route_thermal_stress_jos3")

ROUTE_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}
TMRT_CMAP = "inferno"
TMRT_RANGE = (30.0, 65.0)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--paper-dir", type=Path,
                    default=Path("/home/harshin/files/fastUTEC paper"))
    ap.add_argument("--departure-start", type=float, default=7.0)
    ap.add_argument("--departure-end", type=float, default=19.0)
    ap.add_argument("--departure-step", type=float, default=0.5)
    ap.add_argument("--sweep-spacing-m", type=float, default=2.0,
                    help="Route resampling for the sweep; JOS-3 is insensitive "
                         "to step size at this scale (checked against stage 09 "
                         "at the configured departure).")
    ap.add_argument("--walking-speed-ms", type=float, default=1.3)
    ap.add_argument("--activity-par", default="auto")
    ap.add_argument("--equilibration-min", type=float, default=10.0)
    ap.add_argument("--reuse-sweep", action="store_true")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# geometry backdrop
# ----------------------------------------------------------------------------
def plan_polygons(stl_path):
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh")
    return mesh.vertices[:, :2][mesh.faces]


def canopy_mask(stl_path, cell=1.0):
    """Plan-view canopy occupancy on a `cell` grid from the crown meshes."""
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh")
    xy = mesh.vertices[:, :2][mesh.faces].mean(axis=1)
    lo = xy.min(axis=0)
    ij = np.floor((xy - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 1
    grid = np.zeros(shape, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = binary_closing(grid, iterations=2)
    extent = (lo[0], lo[0] + shape[0] * cell, lo[1], lo[1] + shape[1] * cell)
    return grid.T, extent


def add_scale_bar(ax, length_m, x0, y0, label):
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=2,
            solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 12, label, ha="center", va="bottom",
            fontsize=7)


def add_north_arrow(ax, x, y, size):
    ax.annotate("N", xy=(x, y + size), xytext=(x, y), ha="center",
                va="center", fontsize=7,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))


# ----------------------------------------------------------------------------
# figures from stage-09 traces
# ----------------------------------------------------------------------------
def load_traces(jos3_dir):
    summary = pd.read_csv(jos3_dir / "route_ranking_summary.csv")
    traces = {int(r): pd.read_csv(jos3_dir / f"route_{int(r)}_jos3_trace.csv")
              for r in summary["route_id"]}
    return summary.set_index("route_id"), traces


def figure_map(case_dir, summary, traces, out):
    geometry = case_dir / "geometry"
    buildings = plan_polygons(geometry / "building_final.stl")
    canopy, extent = canopy_mask(geometry / "vegetation_final.stl")
    all_xy = np.vstack([t[["x_local_m", "y_local_m"]].values
                        for t in traces.values()])
    pad = 40.0
    lo, hi = all_xy.min(axis=0) - pad, all_xy.max(axis=0) + pad

    order = summary.sort_values("final_tcore_rise_c").index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    norm = plt.Normalize(*TMRT_RANGE)
    for ax, rid in zip(axes.ravel(), order):
        ax.imshow(np.where(canopy, 1.0, np.nan), extent=extent, origin="lower",
                  cmap=matplotlib.colors.ListedColormap(["#9fd39a"]),
                  interpolation="nearest", zorder=1, rasterized=True)
        ax.add_collection(PolyCollection(buildings, facecolors="#b8b8b8",
                                         edgecolors="none", zorder=2,
                                         rasterized=True))
        t = traces[rid]
        for other in order:
            if other != rid:
                o = traces[other]
                ax.plot(o.x_local_m, o.y_local_m, color="#d0d0d0", lw=0.8,
                        zorder=3)
        sc = ax.scatter(t.x_local_m, t.y_local_m, c=t.tmrt_c, cmap=TMRT_CMAP,
                        norm=norm, s=1.2, lw=0, zorder=4, rasterized=True)
        ax.plot(*t[["x_local_m", "y_local_m"]].values[0], "o", ms=5,
                mfc="white", mec="black", zorder=5)
        ax.plot(*t[["x_local_m", "y_local_m"]].values[-1], "s", ms=5,
                mfc="black", mec="black", zorder=5)
        row = summary.loc[rid]
        ax.set_title(f"Route {rid}: {row.length_m:.0f} m, "
                     f"{row.walk_duration_min:.1f} min, "
                     f"$\\Delta T_{{core}}$ {row.final_tcore_rise_c:.3f} °C",
                     color=ROUTE_COLORS.get(rid, "black"))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    add_scale_bar(axes[0, 0], 200.0, lo[0] + 30, lo[1] + 30, "200 m")
    add_north_arrow(axes[0, 0], hi[0] - 40, lo[1] + 40, 70)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("$T_{mrt}$ at arrival time (°C)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"jos3_route_map.{ext}", dpi=300)
    plt.close(fig)


def figure_strain(summary, traces, out):
    order = summary.sort_values("final_tcore_rise_c").index.tolist()
    fig = plt.figure(figsize=(7.0, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 0.8])
    ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)
    for rid in order:
        t = traces[rid]; c = ROUTE_COLORS.get(rid, "black")
        ax_a.plot(t.elapsed_min, t.tcore_rise_c, color=c, lw=1.6,
                  label=f"Route {rid}")
        ax_a.plot(t.elapsed_min, t.tcore_rise_neutral_c, color=c, lw=0.9,
                  ls="--")
        ax_b.plot(t.elapsed_min, t.tcore_rise_c - t.tcore_rise_neutral_c,
                  color=c, lw=1.6)
        ax_b.plot(t.elapsed_min.iloc[-1],
                  (t.tcore_rise_c - t.tcore_rise_neutral_c).iloc[-1], "o",
                  color=c, ms=4)
    ax_a.plot([], [], color="grey", lw=0.9, ls="--",
              label="same walk, $T_{mrt}=T_a$")
    ax_a.set_ylabel("Core temperature\nrise (°C)")
    ax_a.legend(ncol=5, loc="upper left", frameon=False)
    ax_a.text(0.995, 0.04, "(a)", transform=ax_a.transAxes, ha="right")
    ax_b.set_ylabel("Radiation-attributable\nrise (°C)")
    ax_b.axhline(0, color="grey", lw=0.5)
    ax_b.text(0.995, 0.04, "(b)", transform=ax_b.transAxes, ha="right")

    norm = plt.Normalize(*TMRT_RANGE)
    for k, rid in enumerate(order):
        t = traces[rid]
        edges = np.concatenate([t.elapsed_min.values,
                                [t.elapsed_min.values[-1] + 1e-3]])
        mesh = ax_c.pcolormesh(edges, [k - 0.4, k + 0.4],
                               t.tmrt_c.values[None, :], cmap=TMRT_CMAP,
                               norm=norm, shading="flat", rasterized=True)
    ax_c.set_yticks(range(len(order)))
    ax_c.set_yticklabels([f"Route {rid}" for rid in order])
    for tick, rid in zip(ax_c.get_yticklabels(), order):
        tick.set_color(ROUTE_COLORS.get(rid, "black"))
    ax_c.set_ylim(-0.6, len(order) - 0.4); ax_c.invert_yaxis()
    ax_c.set_xlabel("Elapsed time (min)")
    ax_c.set_title("(c)", loc="right", fontsize=8, pad=2)
    cb = fig.colorbar(mesh, ax=ax_c, pad=0.01, aspect=12)
    cb.set_label("$T_{mrt}$ (°C)")
    plt.setp(ax_a.get_xticklabels(), visible=False)
    plt.setp(ax_b.get_xticklabels(), visible=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"jos3_strain.{ext}", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# departure sweep through stage 09's own physics
# ----------------------------------------------------------------------------
def resample(xy, spacing):
    d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0),
                                                        axis=1))))
    s = np.arange(0.0, d[-1], spacing)
    if d[-1] - s[-1] > 1e-6:
        s = np.append(s, d[-1])
    return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])]), s


def run_sweep(args, root, case, stage09_summary):
    case_json = json.loads((root / "input" / case / "case.json").read_text())
    crs = case_json["coordinates"]["project_crs"]
    origin = (case_json["coordinates"]["local_origin_x"],
              case_json["coordinates"]["local_origin_y"])
    routes = load_routes_directory(root / "input" / case / "routes",
                                   expected_project_crs=crs,
                                   expected_origin=origin)
    mrt_dir = root / "run_output" / case / "mrt_facet_out"
    mrt_xyz = np.load(mrt_dir / "path_xyz.npy")
    times_df = pd.read_csv(mrt_dir / "times.csv", parse_dates=["time"])
    time_hours = np.array([t.hour + t.minute / 60 + t.second / 3600
                           for t in times_df["time"]])
    mrt = stage09.MrtField(
        mrt_xyz, np.load(mrt_dir / "tmrt_matrix_C.npy"), time_hours,
        np.load(mrt_dir / "direct_transmission_matrix.npy", mmap_mode="r"))
    tree = cKDTree(mrt_xyz[:, :2])
    weather = WeatherProvider(
        csv_path=root / "input" / case / "weather" / "weather.csv")
    environment = EnvironmentField(weather)
    elevation = times_df["elevation_deg"].to_numpy(float)
    provenance = json.loads((root / "run_output" / case / "viz" / "route_jos3"
                             / "clothing_provenance.json").read_text())
    subject = stage09.Subject(1.72, 74.0, 30, "male", 15.0, 2.59, 0.0)
    config = clothing_profiles.load_config(provenance.get("config_file"))

    rows = []
    departures = np.arange(args.departure_start,
                           args.departure_end + 1e-9, args.departure_step)
    for route in routes:
        xy, dist = resample(np.asarray(route["xy"], float), args.sweep_spacing_m)
        _, nearest = tree.query(xy)
        for dep in departures:
            arrival = dep + dist / args.walking_speed_ms / 3600.0
            conditions = environment.sample(
                np.column_stack([xy, mrt_xyz[nearest, 2]]), arrival % 24.0)
            mean_hour = float(np.mean(arrival)) % 24.0
            daytime = float(np.interp(mean_hour, time_hours, elevation,
                                      period=24.0)) > 0.0
            clo, prov = clothing_profiles.resolve(
                provenance["requested"],
                air_temp_c=float(np.mean(conditions.air_temperature_c)),
                is_daytime=daytime,
                wind_ms=float(np.mean(conditions.wind_speed_ms)),
                climate=provenance["climate"], config=config)
            par = stage09.jos3_protocol.resolve_activity_ratio(
                args.activity_par, subject.make_model(),
                args.walking_speed_ms, subject.weight)
            common = (xy, arrival, nearest, mrt, environment, subject, clo,
                      par, args.equilibration_min)
            walk = stage09.simulate_walk(*common, context="sweep")
            neutral = stage09.simulate_walk(*common, context="sweep neutral",
                                            radiation_neutral=True)
            dt_min = np.diff(arrival, prepend=arrival[0]) * 60.0
            rows.append({
                "route_id": route["route_id"], "departure_hour": dep,
                "tcore_rise_c": walk["final_tcore_rise_c"],
                "neutral_tcore_rise_c": neutral["final_tcore_rise_c"],
                "radiation_attributable_rise_c": (walk["final_tcore_rise_c"]
                                                  - neutral["final_tcore_rise_c"]),
                "mean_tmrt_c": float(walk["tmrt_trace_c"].mean()),
                "time_sunlit_min": float(dt_min[walk["tau_dir_trace"]
                                                > stage09.SUNLIT_TAU].sum()),
                "clothing": prov.get("ensemble", provenance["requested"]),
            })
        print(f"  route {route['route_id']}: {len(departures)} departures done",
              flush=True)
    sweep = pd.DataFrame(rows)

    # Consistency with the authoritative stage-09 run at its own departure.
    check = sweep[np.isclose(sweep.departure_hour, 13.0)]
    for _, r in check.iterrows():
        ref = stage09_summary.loc[r.route_id]
        diff = abs(r.tcore_rise_c - ref.final_tcore_rise_c)
        print(f"  check route {r.route_id} @13:00: sweep {r.tcore_rise_c:.4f} "
              f"vs stage 09 {ref.final_tcore_rise_c:.4f} (|diff| {diff:.4f})")
        if diff > 0.003:
            raise SystemExit("sweep disagrees with stage 09 -- not writing it")
    return sweep


def figure_sweep(sweep, out):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True,
                                     constrained_layout=True)
    for rid, g in sweep.groupby("route_id"):
        c = ROUTE_COLORS.get(int(rid), "black")
        ax_a.plot(g.departure_hour, g.radiation_attributable_rise_c, "-o",
                  color=c, ms=2.5, lw=1.4, label=f"Route {rid}")
        ax_b.plot(g.departure_hour, g.tcore_rise_c, "-o", color=c, ms=2.5,
                  lw=1.4)
    best = sweep.loc[sweep.groupby("departure_hour").tcore_rise_c.idxmin()]
    for _, r in best.iterrows():
        ax_b.plot(r.departure_hour, r.tcore_rise_c, "o", ms=6, mfc="none",
                  mec="black", mew=0.8)
    ax_b.plot([], [], "o", ms=6, mfc="none", mec="black",
              label="lowest at that departure")
    ax_a.set_ylabel("Radiation-attributable\nrise (°C)")
    ax_a.axhline(0, color="grey", lw=0.5)
    ax_a.legend(ncol=4, frameon=False, loc="lower center")
    ax_b.set_ylabel("Core temperature\nrise (°C)")
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.set_xlabel("Departure time (h, local)")
    ax_b.set_xticks(np.arange(7, 20, 1))
    ax_a.text(0.005, 0.92, "(a)", transform=ax_a.transAxes)
    ax_b.text(0.005, 0.92, "(b)", transform=ax_b.transAxes)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"jos3_departure_sweep.{ext}", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    jos3_dir = root / "run_output" / args.case / "viz" / "route_jos3"
    summary, traces = load_traces(jos3_dir)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    print("route map ..."); figure_map(root / "input" / args.case, summary,
                                       traces, args.paper_dir)
    print("strain figure ..."); figure_strain(summary, traces, args.paper_dir)

    sweep_dir = root / "run_output" / args.case / "postprocessing" / "departure_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = sweep_dir / "departure_sweep.csv"
    if args.reuse_sweep and sweep_csv.is_file():
        sweep = pd.read_csv(sweep_csv)
    else:
        print("departure sweep ...")
        sweep = run_sweep(args, root, args.case, summary)
        sweep.to_csv(sweep_csv, index=False)
    figure_sweep(sweep, args.paper_dir)

    best = sweep.loc[sweep.groupby("departure_hour").tcore_rise_c.idxmin()]
    print("\nlowest-strain route by departure:",
          dict(zip(best.departure_hour, best.route_id)))
    print(f"clothing used across the sweep: {sorted(sweep.clothing.unique())}")
    print(f"wrote figures to {args.paper_dir}")


if __name__ == "__main__":
    main()
