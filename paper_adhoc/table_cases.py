#!/usr/bin/env python3
"""Main-text Table tab:cases and the Sec. 4.1 / abstract numbers: route points,
scene faces (all three meshes), route-visible facets, retained share; Pearson r
of facets against route points and against scene faces."""
import json, re
from pathlib import Path
import numpy as np, trimesh
from scipy.stats import pearsonr
R = Path(__file__).resolve().parents[1]
rows = []
for n in range(1, 7):
    c = f"lisbon{n}"
    pts = len(np.load(R / "run_output" / c / "mrt_facet_out" / "path_xyz.npy"))
    faces = sum(len(trimesh.load(R / "input" / c / "geometry" / f, force="mesh", process=False).faces)
                for f in ("building_final.stl", "vegetation_final.stl", "ground_and_water_final.stl"))
    rep = (R / "run_output" / c / "thermal_out" / "selection_report.txt").read_text()
    facets = int(re.search(r"thermal facets kept\s*:\s*([\d,]+)", rep).group(1).replace(",", ""))
    rows.append((n, pts, faces, facets))
for n, p, f, k in rows:
    print(f"Lisbon {n} & \\num{{{p}}} & \\num{{{f}}} & \\num{{{k}}} & \\SI{{{100*k/f:.2f}}}{{\\percent}} \\\\")
P, F, K = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
print(f"\\textbf{{Total}} & \\num{{{P.sum()}}} & \\num{{{F.sum()}}} & \\num{{{K.sum()}}} & \\SI{{{100*K.sum()/F.sum():.2f}}}{{\\percent}} \\\\")
print(f"retained range {100*(K/F).min():.2f}-{100*(K/F).max():.2f} %; reduction {F.sum()/K.sum():.0f}-fold")
r1, p1 = pearsonr(P, K); r2, p2 = pearsonr(F, K)
print(f"facets vs route points r = {r1:.2f}, p = {p1:.3f}; facets vs faces r = {r2:.2f}, p = {p2:.2f}")
