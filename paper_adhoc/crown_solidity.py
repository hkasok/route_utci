#!/usr/bin/env python3
"""Median crown solidity (volume / convex-hull volume) per scene, main text Sec. 2.2."""
import sys, numpy as np, trimesh
for p in sys.argv[1:]:
    m = trimesh.load(p, force="mesh", process=False)
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    sol = []
    for c in parts:
        if len(c.faces) < 20:
            continue
        try:
            hv = c.convex_hull.volume
        except Exception:
            continue
        if hv > 1e-6:
            sol.append(abs(c.volume) / hv)
    sol = np.array(sol)
    print(f"{p}: {len(sol)} crowns, median solidity {np.median(sol):.2f}")
