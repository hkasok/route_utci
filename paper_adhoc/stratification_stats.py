#!/usr/bin/env python3
"""Supplement S1 statistics of the vegetation clusters, on the UNTHINNED
Lisbon returns (11 Oct 2026): share of plan-view clusters spanning more than
15 m vertically, clusters with an interior void, void heights, and the
lateral width of the low stratum below each void (the trunk test).

Uses the pipeline's own modules with their default settings, exactly as
02_vegetation_to_stl.py applies them.

usage: python3 paper_adhoc/stratification_stats.py <classified_points_dir> [...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_height_filter import build_ground_height_grid, height_above_ground  # noqa: E402
from grid_cluster import grid_cluster_2d  # noqa: E402
import vertical_stratification as vs  # noqa: E402

settings = vs.StratificationSettings()
rows = []
for d in map(Path, sys.argv[1:]):
    veg = np.load(d / "vegetation_points.npy")
    gnd = np.load(d / "ground_and_water_points.npy")
    grid = build_ground_height_grid(gnd, cell_size=2.0)
    veg = veg[height_above_ground(veg, grid) >= 1.5]
    labels, _ = grid_cluster_2d(veg[:, :2], cell_size=0.5, connect_radius_cells=1)
    n_clusters = n_tall = n_void = 0
    voids, widths = [], []
    for lab in np.unique(labels):
        pts = veg[labels == lab]
        if len(pts) < 10:
            continue
        n_clusters += 1
        if np.ptp(pts[:, 2]) > 15.0:
            n_tall += 1
        gaps = vs.find_vertical_gaps(pts[:, 2], settings)
        if gaps:
            n_void += 1
            lo, hi = gaps[0]
            voids.append(hi - lo)
            widths.append(vs.lateral_width(pts[pts[:, 2] < lo]))
    voids, widths = np.array(voids), np.array(widths)
    rows.append((d.parent.parent.name, n_clusters, n_tall, n_void, voids, widths))
    print(f"{d.parent.parent.name}: {n_clusters} clusters, {100 * n_tall / n_clusters:.0f} % span > 15 m, "
          f"{n_void} with an interior void (median {np.median(voids) if len(voids) else 0:.1f} m, "
          f"range {voids.min() if len(voids) else 0:.1f}-{voids.max() if len(voids) else 0:.1f} m); "
          f"low strata narrower than 1.5 m: {int((widths < 1.5).sum())}, wider than 3.0 m: "
          f"{int((widths > 3.0).sum())}", flush=True)
C = sum(r[1] for r in rows); T = sum(r[2] for r in rows); V = sum(r[3] for r in rows)
allv = np.concatenate([r[4] for r in rows]) if rows else np.array([])
allw = np.concatenate([r[5] for r in rows]) if rows else np.array([])
print(f"ALL: {C} clusters, {100 * T / max(C, 1):.0f} % span > 15 m, {V} with a void "
      f"(median {np.median(allv):.1f} m, IQR {np.percentile(allv, 25):.1f}-{np.percentile(allv, 75):.1f} m); "
      f"low strata < 1.5 m wide: {int((allw < 1.5).sum())}, > 3.0 m: {int((allw > 3.0).sum())} of {len(allw)}")
