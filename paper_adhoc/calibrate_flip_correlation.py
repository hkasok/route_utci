#!/usr/bin/env python3
"""Correlation length for the sun/shade flips of paper_ranking_uncertainty.py:
the length at which the simulated flipped stretches have the MEAN length of the
misclassified stretches measured in the Lisbon validation.

Reads validation_summary.json (shadow.classification_agreement and
shadow.tolerance.misclassified_run_length_m.mean); flips are generated with
paper_ranking_uncertainty.correlated_flips on the 2 m route spacing.

The measured stretches are runs of consecutive misclassified SAMPLES, each run
counted as (last - first) + the walk's median sample spacing (about 7 m); the
simulated 2 m flip series is therefore read at that spacing and its runs are
measured the same way.

usage: python3 paper_adhoc/calibrate_flip_correlation.py [validation_summary.json
                                                          [comparison_points_glob]]
"""
import glob
import json
import sys

import pandas as pd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_ranking_uncertainty import correlated_flips  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/harshin/files/fastUTEC paper/validation_summary.json")
s = json.loads(path.read_text())["shadow"]
rate = 1.0 - s["classification_agreement"]
target = s["tolerance"]["misclassified_run_length_m"]["mean"]
spacing = 2.0
points = sys.argv[2] if len(sys.argv) > 2 else \
    "run_output/lisbon*/validation/mrt_lisbon/radiant_flux_comparison_points.csv"
steps = []
for f in sorted(glob.glob(points)):
    d = pd.read_csv(f)
    d = d[d.period == "day"]
    for _, g in d.groupby("route_id"):
        steps.append(np.median(np.diff(g.sort_values("seq").distance_along_route_m.values)))
sample_m = float(np.median(steps))
stride = max(1, int(round(sample_m / spacing)))


def mean_run_m(length, n=400_000, seed=11):
    f = correlated_flips(n, spacing, rate, length, np.random.default_rng(seed))[::stride]
    edges = np.diff(np.r_[0, f.astype(int), 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(runs.mean() * stride * spacing)


grid = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 50], float)
means = np.array([mean_run_m(L) for L in grid])
L = float(np.interp(target, means, grid))
print(f"flip rate {rate:.3f}; measured mean misclassified stretch {target:.1f} m; "
      f"sample spacing {sample_m:.2f} m (stride {stride} x {spacing} m)")
for g, m in zip(grid, means):
    print(f"  L = {g:4.0f} m -> mean flipped stretch {m:5.1f} m")
print(f"calibrated correlation length {L:.1f} m (rounded: {round(L)} m)")
