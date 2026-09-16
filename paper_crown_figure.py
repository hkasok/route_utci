#!/usr/bin/env python3
"""
paper_crown_figure.py -- figure comparing the three crown-geometry models on
one real LiDAR cluster.

STANDALONE. It imports the crown builders read-only and writes a figure into
the paper directory; it changes no pipeline behaviour and writes nothing into
run_output or input.

The point of the figure is the plan-view row. A star-shaped hull about a
vertical axis gives one closed loop per height band, so its horizontal
cross-section is convex however many sectors are used -- which is what makes a
scene built that way read as a field of circles from above. The implicit-field
surface carries no such constraint, so the same measured points can produce a
re-entrant, forked or leaning outline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

import crown_field_reconstruction as cfr
from ground_height_filter import build_ground_height_grid, height_above_ground
from grid_cluster import grid_cluster_2d
from tree_crown_segmentation import segment_tree_crowns, should_segment
from vertical_stratification import split_vertical_strata

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def pick_crown(split_dir: Path, index: int) -> np.ndarray:
    """One real, fully segmented crown from a case's vegetation cloud."""
    pts = np.load(split_dir / "vegetation_points.npy")
    gnd = np.load(split_dir / "ground_and_water_points.npy")
    grid = build_ground_height_grid(gnd, cell_size=2.0)
    pts = pts[height_above_ground(pts, grid) >= 1.5]
    labels, _ = grid_cluster_2d(pts[:, :2], cell_size=0.5, connect_radius_cells=1)
    crowns = []
    for c in np.unique(labels):
        cluster = pts[labels == c]
        subs = [cluster]
        if should_segment(cluster, 0.5, 300, 8.0):
            crown_labels, _ = segment_tree_crowns(
                cluster, chm_res=0.25, smooth_sigma=2.0,
                min_tree_distance_m=3.0, min_fragment_points=50)
            subs = [cluster[crown_labels == k] for k in np.unique(crown_labels)]
        for sub in subs:
            crowns += [s.points for s in split_vertical_strata(sub)[0]]
    # A representative crown: enough points to reconstruct, and an aspect ratio
    # in the range a real broadleaf tree occupies. A very tall sparse cluster
    # would exaggerate the difference between the models rather than illustrate
    # it fairly.
    def aspect(c):
        width = max(c[:, 0].ptp(), c[:, 1].ptp())
        return c[:, 2].ptp() / max(width, 1e-6)
    crowns = [c for c in crowns
              if 1200 <= len(c) <= 6000 and 0.6 <= aspect(c) <= 1.6
              and max(c[:, 0].ptp(), c[:, 1].ptp()) >= 6.0]
    if not crowns:
        raise SystemExit("no representative crown found")
    crowns.sort(key=len)
    return crowns[min(index, len(crowns) - 1)]


def outline(mesh: trimesh.Trimesh, n: int = 220):
    """Plan-view occupancy outline of a mesh, and its convex hull area ratio."""
    pts, _ = trimesh.sample.sample_surface(mesh, 60000)
    xy = pts[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    grid = np.zeros((n, n), bool)
    idx = np.clip(((xy - lo) / np.maximum(hi - lo, 1e-9) * (n - 1)).astype(int), 0, n - 1)
    grid[idx[:, 0], idx[:, 1]] = True
    from scipy import ndimage
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, np.ones((3, 3))))
    cell = np.prod((hi - lo) / (n - 1))
    area = grid.sum() * cell
    hull = ConvexHull(xy).volume
    return grid, lo, hi, area / hull if hull > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=Path("out_crop_test/00_split"))
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--paper-dir", type=Path,
                    default=Path("/home/harshin/files/fastUTEC paper"))
    args = ap.parse_args()

    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("veg", "02_vegetation_to_stl.py")
    veg = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["x"]
    try:
        spec.loader.exec_module(veg)
    except SystemExit:
        pass
    sys.argv = saved

    points = pick_crown(args.split_dir, args.index)
    print(f"crown: {len(points):,} points, "
          f"{points[:,0].ptp():.1f} x {points[:,1].ptp():.1f} x {points[:,2].ptp():.1f} m")

    cx, cy, base_z, radius = veg.fit_hemisphere(points)
    meshes = {
        "Hemisphere": veg.build_hemisphere_mesh(cx, cy, base_z, radius, subdivisions=2),
        "Star-shaped hull": None,
        "Implicit field": cfr.reconstruct_crown(points)[0],
    }
    axis_xy, z_levels, radial, _ = veg.crown_radial_profile(points)
    meshes["Star-shaped hull"] = veg.build_crown_mesh(axis_xy, z_levels, radial)

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.9),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    for col, (name, mesh) in enumerate(meshes.items()):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        axes[0, col].remove()
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
                        triangles=mesh.faces, Z=mesh.vertices[:, 2],
                        color="#4a9c5c", alpha=0.72, edgecolor="none",
                        linewidth=0, antialiased=True)
        sub = points[np.random.default_rng(0).choice(len(points),
                                                     min(700, len(points)), replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.7, c="0.25", alpha=0.5, lw=0)
        ax.set_title(f"{name}\n{len(mesh.faces)} faces", pad=1)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(False)
        ax.set_box_aspect((points[:, 0].ptp(), points[:, 1].ptp(),
                           max(points[:, 2].ptp(), 1.0)))

        grid, lo, hi, solidity = outline(mesh)
        pa = axes[1, col]
        pa.imshow(grid.T, origin="lower", cmap="Greens", vmin=0, vmax=1.6,
                  extent=[lo[0], hi[0], lo[1], hi[1]], interpolation="nearest")
        pa.scatter(points[:, 0], points[:, 1], s=0.4, c="0.25", alpha=0.35, lw=0)
        pa.set_title(f"plan view, solidity {solidity:.2f}", fontsize=8.5, pad=3)
        pa.set_aspect("equal")
        pa.set_xlim(lo[0] - 1, hi[0] + 1); pa.set_ylim(lo[1] - 1, hi[1] + 1)
        pa.set_xticks([]); pa.set_yticks([])
    fig.subplots_adjust(wspace=0.06, hspace=0.16)
    out = args.paper_dir / "crown_model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
