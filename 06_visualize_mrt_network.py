"""
06_visualize_mrt_network.py -- visualize the 24-hour MRT-along-route
results from 05_mrt_network_raytrace.py.

Produces TWO complementary outputs:

  1. STATIC key-times overview (PNG, always fast, uses the FULL point
     count -- no subsampling needed since matplotlib static rendering
     at ~150K+ points takes ~2s/frame, benchmarked directly). A grid of
     snapshots at a handful of representative times across the day,
     sharing one color scale so hotter/cooler is comparable panel to
     panel at a glance.

  2. INTERACTIVE animated HTML (plotly, WebGL-accelerated Scattergl)
     with a play button and a time slider to scrub through all 24
     hours. Uses a SUBSAMPLE of points (default 20,000) to keep the
     file size and browser rendering smooth -- at the full network's
     real point count (hundreds of thousands), embedding every point
     in every frame would make the file impractically large; the
     static overview above already shows the full-resolution picture
     for any single moment.

Run:
    python3 06_visualize_mrt_network.py \
        --results-dir mrt_network_output/ \
        --output-dir mrt_viz/ \
        --buildings-stl out_full/02_final/building_final.stl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args():
    p = argparse.ArgumentParser(description="Visualize MRT-along-route results")
    p.add_argument("--results-dir", required=True,
                    help="Output directory from 05_mrt_network_raytrace.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--buildings-stl", default=None,
                    help="Optional: overlay building footprints for spatial context")
    p.add_argument("--n-static-panels", type=int, default=6,
                    help="Number of key-time snapshots in the static overview (default: 6)")
    p.add_argument("--n-animated-points", type=int, default=20000,
                    help="Point subsample size for the interactive HTML animation "
                         "(default: 20000 -- keeps file size/browser rendering smooth)")
    p.add_argument("--point-size", type=float, default=2.0,
                    help="Marker size for the static overview (default: 2.0)")
    p.add_argument("--colormap", default="inferno",
                    help="Matplotlib/plotly colormap name (default: inferno)")
    p.add_argument("--vmin", type=float, default=None,
                    help="Fix the color scale minimum (deg C). Default: auto from data.")
    p.add_argument("--vmax", type=float, default=None,
                    help="Fix the color scale maximum (deg C). Default: auto from data.")
    return p.parse_args()


def load_building_footprint_lines(stl_path):
    """Project building mesh edges to XY for a lightweight context overlay."""
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh")
    # Use the mesh's edges, projected to XY, as a simple wireframe-style overlay.
    edges = mesh.edges_unique
    verts_xy = mesh.vertices[:, :2]
    segments = verts_xy[edges]  # (n_edges, 2, 2)
    return segments


def make_static_overview(path_xy, tmrt_matrix, times, out_path, n_panels, vmin, vmax,
                          point_size, cmap, building_segments=None):
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
            lc = LineCollection(building_segments, colors="gray", linewidths=0.4, alpha=0.6)
            ax.add_collection(lc)
        sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=tmrt_matrix[it], cmap=cmap,
                         s=point_size, vmin=vmin, vmax=vmax, edgecolors="none")
        ax.set_aspect("equal")
        t = times[it]
        ax.set_title(t.strftime("%H:%M"))
        ax.set_xticks([])
        ax.set_yticks([])

    for ax_i in range(len(panel_indices), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle("Mean Radiant Temperature along pedestrian network -- 24 hour progression",
                  fontsize=14)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Tmrt [°C]")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_interactive_animation(path_xy, tmrt_matrix, times, out_path, n_sub, vmin, vmax,
                                cmap, building_segments=None):
    import plotly.graph_objects as go

    n_points = len(path_xy)
    n_sub = min(n_sub, n_points)
    sub_idx = np.linspace(0, n_points - 1, n_sub).astype(int)
    xy_sub = path_xy[sub_idx]
    tmrt_sub = tmrt_matrix[:, sub_idx]
    nt = len(times)

    time_labels = [t.strftime("%H:%M") for t in times]

    base_traces = []
    if building_segments is not None:
        # Flatten building edge segments into a single line trace with None
        # separators (much faster than one trace per edge).
        xs, ys = [], []
        for seg in building_segments:
            xs.extend([seg[0, 0], seg[1, 0], None])
            ys.extend([seg[0, 1], seg[1, 1], None])
        base_traces.append(go.Scattergl(x=xs, y=ys, mode="lines",
                                         line=dict(color="lightgray", width=0.5),
                                         hoverinfo="skip", showlegend=False))

    point_trace = go.Scattergl(
        x=xy_sub[:, 0], y=xy_sub[:, 1], mode="markers",
        marker=dict(size=3, color=tmrt_sub[0], colorscale=cmap, cmin=vmin, cmax=vmax,
                    colorbar=dict(title="Tmrt [°C]")),
        hovertemplate="Tmrt: %{marker.color:.1f} °C<extra></extra>",
    )

    frames = []
    for it in range(nt):
        frames.append(go.Frame(
            data=base_traces + [go.Scattergl(
                x=xy_sub[:, 0], y=xy_sub[:, 1], mode="markers",
                marker=dict(size=3, color=tmrt_sub[it], colorscale=cmap, cmin=vmin, cmax=vmax),
            )],
            name=str(it),
        ))

    fig = go.Figure(data=base_traces + [point_trace], frames=frames)
    fig.update_layout(
        width=1000, height=800,
        title="Mean Radiant Temperature along pedestrian network (24h)",
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

    print("Loading results...")
    path_xyz = np.load(results_dir / "path_xyz.npy")
    tmrt_matrix = np.load(results_dir / "tmrt_matrix_C.npy")
    times_df = pd.read_csv(results_dir / "times.csv", parse_dates=["time"])
    times = times_df["time"].tolist()

    path_xy = path_xyz[:, :2]
    print(f"  {len(path_xy):,} points, {len(times)} time steps")

    vmin = args.vmin if args.vmin is not None else float(np.percentile(tmrt_matrix, 1))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(tmrt_matrix, 99))
    print(f"  Color scale: {vmin:.1f} to {vmax:.1f} deg C "
          f"(1st-99th percentile; use --vmin/--vmax to override)")

    building_segments = None
    if args.buildings_stl:
        print("Loading building footprints for context overlay...")
        building_segments = load_building_footprint_lines(args.buildings_stl)
        print(f"  {len(building_segments):,} edge segments")

    print("\nBuilding static key-times overview...")
    static_path = out_dir / "mrt_static_overview.png"
    make_static_overview(
        path_xy, tmrt_matrix, times, static_path, args.n_static_panels,
        vmin, vmax, args.point_size, args.colormap, building_segments,
    )
    print(f"  Saved: {static_path}")

    print("\nBuilding interactive animated HTML "
          f"(subsampled to {args.n_animated_points:,} points for smooth scrubbing)...")
    html_path = out_dir / "mrt_animated.html"
    make_interactive_animation(
        path_xy, tmrt_matrix, times, html_path, args.n_animated_points,
        vmin, vmax, args.colormap, building_segments,
    )
    size_mb = html_path.stat().st_size / 1e6
    print(f"  Saved: {html_path} ({size_mb:.1f} MB)")

    print(f"\n[viz_result] static={static_path} animated_html={html_path} "
          f"animated_size_mb={size_mb:.1f}")


if __name__ == "__main__":
    main()
