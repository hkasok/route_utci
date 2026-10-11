"""Rays from open-air points 1.1 m above terrain; any first hit on the BACK of a
building face means a gap in the surface (or an inward face)."""
import sys, numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
b = trimesh.load(sys.argv[1], force="mesh", process=False)
g = trimesh.load(sys.argv[2], force="mesh", process=False)
bi, gi = RayMeshIntersector(b), RayMeshIntersector(g)
rng = np.random.default_rng(1)
lo, hi = b.bounds[0], b.bounds[1]
xy = rng.uniform(lo[:2], hi[:2], size=(200000, 2))
o = np.column_stack([xy, np.full(len(xy), 5000.0)])
down = np.tile([0, 0, -1.0], (len(xy), 1))
loc, ir, _ = gi.intersects_location(o, down, multiple_hits=False)
zg = np.full(len(xy), np.nan); zg[ir] = loc[:, 2]
loc, ir, _ = bi.intersects_location(o, down, multiple_hits=False)
zb = np.full(len(xy), -np.inf); zb[ir] = loc[:, 2]
ok = np.isfinite(zg) & (zb < zg + 1.1)
p = np.column_stack([xy[ok], zg[ok] + 1.1])
m = len(p)
d = rng.normal(size=(m, 3)); d[:, 2] = np.abs(d[:, 2]) * 0.5; d /= np.linalg.norm(d, axis=1, keepdims=True)
loc, ir, it = bi.intersects_location(p, d, multiple_hits=False)
gl, gr, _ = gi.intersects_location(p, d, multiple_hits=False)
gdist = np.full(len(p), np.inf); gdist[gr] = np.linalg.norm(gl - p[gr], axis=1)
bdist = np.linalg.norm(loc - p[ir], axis=1)
first = bdist < gdist[ir]          # the building is hit before the terrain
loc, ir, it = loc[first], ir[first], it[first]
back = (b.face_normals[it] * d[ir]).sum(1) > 0
under = b.face_normals[it, 2] < -0.5
print(f"open points {m}, rays hitting buildings {len(ir)}, first hit on a back face: {back.sum()} ({back.mean():.2e}); "
      f"first hit on the underside of a floor: {under.sum()} ({under.mean():.2e})")
for k in np.flatnonzero(back | under)[:5]:
    f = it[k]
    print("  origin", np.round(p[ir[k]], 2), "dir", np.round(d[ir[k]], 3), "hit", np.round(loc[k], 2),
          "face n", np.round(b.face_normals[f], 3), "face z", np.round(b.triangles[f][:, 2], 2),
          "xy", np.round(b.triangles[f][:, :2], 2).tolist())
