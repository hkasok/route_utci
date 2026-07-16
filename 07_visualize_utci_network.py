"""
07_visualize_utci_network.py -- compute and visualize UTCI along the
pedestrian network, combining your already-computed spatially-resolved
Tmrt with representative (spatially-uniform) Miami summer air
temperature, relative humidity, and wind speed.

UTCI itself is computed with `pythermalcomfort` (validated open-source
implementation of the Bröde et al. 2012 operational UTCI polynomial) --
not hand-transcribed, to avoid transcription errors in what is a very
long 6th-order 4-variable regression.

DEFAULT MIAMI SUMMER ASSUMPTIONS (all overridable via CLI args), sourced
from NWS/NOAA climate normals for Miami International Airport:
    Air temperature : diurnal sinusoid, mean 29 C, +/-4 C amplitude
                       (July/August daily lows ~25 C, highs ~33 C)
    Relative humidity: constant 70% (representative summer average;
                       real diurnal range is wider, ~63-86%, but held
                       constant here per your "constant value" request)
    Wind speed       : constant 3.1 m/s (~7 mph, the Miami summer
                       average -- summer is Miami's calmest wind season)

These three are held SPATIALLY UNIFORM across the whole network -- see
the earlier discussion: Tmrt dominates UTCI's spatial variability by a
wide margin (its correlation with UTCI is consistently the highest of
the four inputs, and its real spatial range across sun/shade is far
larger than Ta/RH's real spatial range at this scale), so resolving
only Tmrt spatially while treating Ta/RH/wind as uniform is a
well-justified simplification, not a meaningful accuracy loss.

Color scale: uses the standard, literature-defined 10-category UTCI
thermal stress classification (Brode et al. 2012 / utci.org), NOT a
generic continuous colormap -- so colors correspond to physiologically
meaningful stress categories rather than an arbitrary gradient.

Run:
    python3 07_visualize_utci_network.py \
        --results-dir mrt_network_output/ \
        --output-dir utci_viz/ \
        --buildings-stl out_full/02_final/building_final.stl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pythermalcomfort.models import utci


# ============================================================
# UTCI thermal stress categories (Brode et al. 2012 / utci.org)
# ============================================================
# MIAMI SPECIALIZATION: sub-zero (and indeed all cold-stress) UTCI never
# occurs here -- the observed minimum in a full July day is ~+19 C -- so
# the cold half of the standard 10-category scale is dead range that only
# wastes color resolution. We therefore keep only the categories at or
# above "No thermal stress" and, for CLARITY, split EACH retained category
# into two half-bands (a lighter lower half and a darker upper half). This
# doubles the visual resolution exactly where all Miami values sit.
#
# Set --full-utci-scale to restore the classic full cold-to-hot scale.

# --- Full standard scale (used only with --full-utci-scale) ---
UTCI_FULL_BOUNDS = [-40, -27, -13, 0, 9, 26, 32, 38, 46]
UTCI_FULL_LABELS = [
    "Extreme cold stress", "Very strong cold stress", "Strong cold stress",
    "Moderate cold stress", "Slight cold stress", "No thermal stress",
    "Moderate heat stress", "Strong heat stress", "Very strong heat stress",
    "Extreme heat stress",
]
UTCI_FULL_COLORS = [
    "#4d004b", "#810f7c", "#8856a7", "#8c96c6", "#9ebcda",
    "#66c2a4", "#fed976", "#fd8d3c", "#e31a1c", "#800026",
]

# --- Miami warm-only scale: categories from "No thermal stress" up,
#     each split into two half-bands at its midpoint. The upper open
#     category (Extreme heat, >=46) is split at a nominal 50 C so it, too,
#     gets a light/dark pair. ---
#   No thermal stress   [9, 26)  -> 9,   17.5, 26
#   Moderate heat       [26,32)  -> 26,  29,   32
#   Strong heat         [32,38)  -> 32,  35,   38
#   Very strong heat    [38,46)  -> 38,  42,   46
#   Extreme heat        [46, +)  -> 46,  50,  (+inf)
UTCI_CATEGORY_BOUNDS = [9, 17.5, 26, 29, 32, 35, 38, 42, 46, 50]
UTCI_CATEGORY_LABELS = [
    "No thermal stress (lower)", "No thermal stress (upper)",
    "Moderate heat stress (lower)", "Moderate heat stress (upper)",
    "Strong heat stress (lower)", "Strong heat stress (upper)",
    "Very strong heat stress (lower)", "Very strong heat stress (upper)",
    "Extreme heat stress (lower)", "Extreme heat stress (upper)",
]
# Two shades per category: lighter (lower half) then darker (upper half),
# keeping the conventional green -> yellow -> orange -> red -> maroon hue
# progression so the scale still reads as "hotter = more intense".
UTCI_CATEGORY_COLORS = [
    "#a1d99b", "#66c2a4",   # no thermal stress: light green -> teal-green
    "#ffffb2", "#fed976",   # moderate heat:     pale yellow -> gold
    "#feb24c", "#fd8d3c",   # strong heat:       light orange -> orange
    "#fc4e2a", "#e31a1c",   # very strong heat:  red-orange -> red
    "#bd0026", "#800026",   # extreme heat:      crimson -> maroon
]


def parse_args():
    p = argparse.ArgumentParser(description="Compute and visualize UTCI along the pedestrian network")
    p.add_argument("--results-dir", required=True,
                    help="Output directory from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None)
    p.add_argument("--full-utci-scale", action="store_true",
                    help="Use the classic full cold-to-hot 10-category UTCI "
                         "scale instead of the Miami warm-only split scale")

    p.add_argument("--air-temp-mean-c", type=float, default=29.0,
                    help="Diurnal mean air temp, deg C (default: 29.0, Miami summer)")
    p.add_argument("--air-temp-amp-c", type=float, default=4.0,
                    help="Diurnal amplitude, deg C (default: 4.0 -> ~25-33 C range)")
    p.add_argument("--air-temp-peak-hour", type=float, default=15.0,
                    help="Hour of daily max air temp (default: 15.0)")
    p.add_argument("--relative-humidity-pct", type=float, default=70.0,
                    help="Constant RH, percent (default: 70.0, Miami summer average)")
    p.add_argument("--wind-speed-ms", type=float, default=3.1,
                    help="Constant wind speed, m/s (default: 3.1, ~7mph Miami summer "
                         "average). UTCI's reference/valid range treats <0.5 m/s as "
                         "unreliable -- this default is well above that floor.")

    p.add_argument("--n-static-panels", type=int, default=6)
    p.add_argument("--n-animated-points", type=int, default=20000)
    p.add_argument("--point-size", type=float, default=2.0)
    return p.parse_args()


def air_temperature_c(hour_of_day, mean_c, amp_c, peak_hour):
    return mean_c + amp_c * np.cos(2.0 * np.pi * (hour_of_day - peak_hour) / 24.0)


def select_scale(full_scale):
    """Return (bounds, labels, colors) for the requested scale."""
    if full_scale:
        return UTCI_FULL_BOUNDS, UTCI_FULL_LABELS, UTCI_FULL_COLORS
    return UTCI_CATEGORY_BOUNDS, UTCI_CATEGORY_LABELS, UTCI_CATEGORY_COLORS


def utci_colormap(full_scale=False):
    bounds, _, colors = select_scale(full_scale)
    cmap = mcolors.ListedColormap(colors)
    # For the Miami scale the lowest band starts AT "No thermal stress"
    # (>=9 C); values below that never occur here, so only the TOP is
    # extended (extend='max'). The full scale extends both ends as before.
    extend = "both" if full_scale else "max"
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend)
    return cmap, norm


def load_building_footprint_lines(stl_path):
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh")
    edges = mesh.edges_unique
    verts_xy = mesh.vertices[:, :2]
    return verts_xy[edges]


def make_static_overview(path_xy, utci_matrix, times, out_path, n_panels, point_size,
                          building_segments=None, full_scale=False):
    cmap, norm = utci_colormap(full_scale)
    bounds, labels, colors = select_scale(full_scale)
    nt = len(times)
    panel_indices = np.linspace(0, nt - 1, n_panels).astype(int)

    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    sc = None
    for ax_i, it in enumerate(panel_indices):
        ax = axes[ax_i]
        if building_segments is not None:
            from matplotlib.collections import LineCollection
            lc = LineCollection(building_segments, colors="dimgray", linewidths=0.4, alpha=0.7)
            ax.add_collection(lc)
        sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=utci_matrix[it], cmap=cmap, norm=norm,
                         s=point_size, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_title(times[it].strftime("%H:%M"))
        ax.set_xticks([])
        ax.set_yticks([])

    for ax_i in range(len(panel_indices), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle("UTCI along pedestrian network -- 24 hour progression\n"
                  "(Miami summer representative Ta/RH/wind, spatially-resolved shade)",
                  fontsize=13)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.03, ticks=bounds)
    cbar.set_label("UTCI thermal stress category")
    cbar.ax.set_yticklabels([f"{b:g}" for b in bounds])
    # Secondary text listing category names for reference
    legend_text = "\n".join(f"{c}: {l}" for c, l in zip(colors, labels))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_category_legend(out_path, full_scale=False):
    bounds, labels, colors = select_scale(full_scale)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    # Build the [lo, hi) range string for band i under each scale.
    #  * full scale (extend='both'): band 0 is "< bounds[0]", band n-1 is
    #    ">= bounds[-1]", interior band i is [bounds[i-1], bounds[i]).
    #  * warm-only scale (extend='max'): the `bounds` array has exactly n
    #    entries where bounds[i] is the LOWER edge of band i, so band i is
    #    [bounds[i], bounds[i+1]) and the last band is ">= bounds[-1]".
    def band_range(i):
        if full_scale:
            if i == 0:
                return f"< {bounds[0]:g} °C"
            if i == n - 1:
                return f">= {bounds[-1]:g} °C"
            return f"[{bounds[i - 1]:g}, {bounds[i]:g}) °C"
        # warm-only: bounds[i] is band i's lower edge
        if i == n - 1:
            return f">= {bounds[-1]:g} °C"
        return f"[{bounds[i]:g}, {bounds[i + 1]:g}) °C"

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.barh(n - i, 1, color=color)
        ax.text(1.05, n - i, f"{label}  ({band_range(i)})",
                va="center", fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n + 1)
    ax.axis("off")
    title = ("UTCI thermal stress categories" if full_scale
             else "UTCI categories (Miami warm-only, split shades)")
    ax.set_title(title, fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_interactive_animation(path_xy, utci_matrix, times, out_path, n_sub,
                               building_segments=None, full_scale=False):
    import plotly.graph_objects as go
    bounds_list, _, colors = select_scale(full_scale)

    n_points = len(path_xy)
    n_sub = min(n_sub, n_points)
    sub_idx = np.linspace(0, n_points - 1, n_sub).astype(int)
    xy_sub = path_xy[sub_idx]
    utci_sub = utci_matrix[:, sub_idx]
    nt = len(times)
    time_labels = [t.strftime("%H:%M") for t in times]

    # plotly colorscale built from the same discrete category boundaries/colors
    bounds = np.array(bounds_list, dtype=float)
    lo = bounds[0] - (5 if full_scale else 0)   # warm-only: clamp at first bound
    hi = bounds[-1] + 10
    positions = np.concatenate(([lo], bounds, [hi]))
    norm_positions = (positions - lo) / (hi - lo)
    colorscale = []
    for i in range(len(colors)):
        colorscale.append([norm_positions[i], colors[i]])
        colorscale.append([norm_positions[i + 1], colors[i]])

    base_traces = []
    if building_segments is not None:
        xs, ys = [], []
        for seg in building_segments:
            xs.extend([seg[0, 0], seg[1, 0], None])
            ys.extend([seg[0, 1], seg[1, 1], None])
        base_traces.append(go.Scattergl(x=xs, y=ys, mode="lines",
                                         line=dict(color="lightgray", width=0.5),
                                         hoverinfo="skip", showlegend=False))

    point_trace = go.Scattergl(
        x=xy_sub[:, 0], y=xy_sub[:, 1], mode="markers",
        marker=dict(size=3, color=utci_sub[0], colorscale=colorscale, cmin=lo, cmax=hi,
                    colorbar=dict(title="UTCI [°C]", tickvals=bounds_list)),
        hovertemplate="UTCI: %{marker.color:.1f} °C<extra></extra>",
    )
    frames = []
    for it in range(nt):
        frames.append(go.Frame(
            data=base_traces + [go.Scattergl(
                x=xy_sub[:, 0], y=xy_sub[:, 1], mode="markers",
                marker=dict(size=3, color=utci_sub[it], colorscale=colorscale, cmin=lo, cmax=hi),
            )],
            name=str(it),
        ))

    fig = go.Figure(data=base_traces + [point_trace], frames=frames)
    fig.update_layout(
        width=1000, height=800,
        title="UTCI along pedestrian network (24h, Miami summer conditions)",
        xaxis=dict(scaleanchor="y", title="X [m]"),
        yaxis=dict(title="Y [m]"),
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(i)],
                        dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                        label=time_labels[i]) for i in range(nt)],
            active=0, currentvalue=dict(prefix="Time: "),
        )],
    )
    fig.write_html(str(out_path))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MRT results...")
    path_xyz = np.load(results_dir / "path_xyz.npy")
    tmrt_matrix = np.load(results_dir / "tmrt_matrix_C.npy")
    times_df = pd.read_csv(results_dir / "times.csv", parse_dates=["time"])
    times = times_df["time"].tolist()
    path_xy = path_xyz[:, :2]
    nt, n_points = tmrt_matrix.shape
    print(f"  {n_points:,} points, {nt} time steps")

    print("\nMiami summer assumptions (spatially uniform):")
    print(f"  Air temp : {args.air_temp_mean_c} +/- {args.air_temp_amp_c} C, "
          f"peak at {args.air_temp_peak_hour}:00")
    print(f"  RH       : {args.relative_humidity_pct}% (constant)")
    print(f"  Wind     : {args.wind_speed_ms} m/s (constant)")

    print("\nComputing UTCI for every point and time step...")
    utci_matrix = np.zeros_like(tmrt_matrix)
    for it, t in enumerate(times):
        hour = t.hour + t.minute / 60.0
        ta = air_temperature_c(hour, args.air_temp_mean_c, args.air_temp_amp_c,
                                args.air_temp_peak_hour)
        tr = tmrt_matrix[it]
        v = np.full(n_points, max(args.wind_speed_ms, 0.5))
        rh = np.full(n_points, args.relative_humidity_pct)
        ta_arr = np.full(n_points, ta)
        result = utci(tdb=ta_arr, tr=tr, v=v, rh=rh, limit_inputs=False)
        utci_matrix[it] = result.utci

    np.save(out_dir / "utci_matrix_C.npy", utci_matrix)
    print(f"  UTCI range: {utci_matrix.min():.1f} to {utci_matrix.max():.1f} C")

    building_segments = None
    if args.buildings_stl:
        print("\nLoading building footprints for context...")
        building_segments = load_building_footprint_lines(args.buildings_stl)

    print("\nBuilding static overview with standard UTCI category colors...")
    static_path = out_dir / "utci_static_overview.png"
    make_static_overview(path_xy, utci_matrix, times, static_path, args.n_static_panels,
                          args.point_size, building_segments, full_scale=args.full_utci_scale)
    print(f"  Saved: {static_path}")

    legend_path = out_dir / "utci_category_legend.png"
    make_category_legend(legend_path, full_scale=args.full_utci_scale)
    print(f"  Saved: {legend_path}")

    print("\nBuilding interactive animated HTML...")
    html_path = out_dir / "utci_animated.html"
    make_interactive_animation(path_xy, utci_matrix, times, html_path,
                                args.n_animated_points, building_segments, full_scale=args.full_utci_scale)
    size_mb = html_path.stat().st_size / 1e6
    print(f"  Saved: {html_path} ({size_mb:.1f} MB)")

    # Summary stats
    summary_rows = []
    for it, t in enumerate(times):
        vals = utci_matrix[it]
        summary_rows.append({
            "time": t.isoformat(),
            "utci_mean_C": float(np.mean(vals)),
            "utci_min_C": float(np.min(vals)),
            "utci_max_C": float(np.max(vals)),
            "pct_no_thermal_stress": float(np.mean((vals >= 9) & (vals < 26)) * 100),
            "pct_moderate_heat_stress": float(np.mean((vals >= 26) & (vals < 32)) * 100),
            "pct_strong_heat_stress": float(np.mean((vals >= 32) & (vals < 38)) * 100),
            "pct_very_strong_plus_heat_stress": float(np.mean(vals >= 38) * 100),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "utci_summary_by_time.csv", index=False)

    print(f"\n[utci_result] n_points={n_points} n_times={nt} output_dir={out_dir}")


if __name__ == "__main__":
    main()
