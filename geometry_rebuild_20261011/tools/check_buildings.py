"""Checks of a building STL against the building returns it was built from.

1. Orientation: a point 5 cm outside each wall/roof face must be in the open
   (nothing of the building mesh directly above it, or only a roof that is
   lower), a point 5 cm inside must have the solid's roof above it.
2. Roof fit: per 0.5 m cell with returns, modelled roof (first hit of a
   downward ray) minus the highest return.
"""
import sys, json
import numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

stl, pts_path = sys.argv[1], sys.argv[2]
ground_path = sys.argv[3] if len(sys.argv) > 3 else None
mesh = trimesh.load(stl, force="mesh", process=False)
inter = RayMeshIntersector(mesh)
n = mesh.face_normals
c = mesh.triangles_center
res = {}
def roof_above(p):
    o = p.copy(); o[:, 2] = 5000.0
    loc, ir, _ = inter.intersects_location(o, np.tile([0, 0, -1.0], (len(p), 1)), multiple_hits=False)
    z = np.full(len(p), -np.inf); z[ir] = loc[:, 2]
    return z
side = np.abs(n[:, 2]) < 0.5
top = n[:, 2] > 0.5
rng = np.random.default_rng(0)
for name, sel in (("walls", side), ("roofs", top)):
    idx = np.flatnonzero(sel)
    if len(idx) > 200000: idx = rng.choice(idx, 200000, replace=False)
    out = c[idx] + 0.05 * n[idx]
    inn = c[idx] - 0.05 * n[idx]
    za_out = roof_above(out); za_in = roof_above(inn)
    ok_out = za_out < out[:, 2] - 1e-6
    ok_in = za_in > inn[:, 2] + 1e-6
    res[name] = {"n": int(len(idx)), "outside_point_in_open": float(ok_out.mean()),
                 "inside_point_under_roof": float(ok_in.mean())}
# roof fit
pts = np.load(pts_path)
r = 0.5
ix = np.floor(pts[:, 0] / r).astype(np.int64); iy = np.floor(pts[:, 1] / r).astype(np.int64)
key = iy * 10_000_000 + ix
order = np.lexsort((-pts[:, 2], key)); first = np.r_[True, key[order][1:] != key[order][:-1]]
top_pts = pts[order[first]]
cent = np.column_stack([(np.floor(top_pts[:, 0] / r) + 0.5) * r, (np.floor(top_pts[:, 1] / r) + 0.5) * r, top_pts[:, 2]])
zm = roof_above(cent)
hit = np.isfinite(zm)
d = zm[hit] - cent[hit, 2]
res["roof_fit_cells_with_returns"] = int(len(cent))
res["fraction_cells_covered_by_model"] = float(hit.mean())
res["roof_minus_return_m"] = {"median": float(np.median(d)), "mae": float(np.mean(np.abs(d))),
                              "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))}
res["faces"] = int(len(mesh.faces))
print(json.dumps(res, indent=1))
