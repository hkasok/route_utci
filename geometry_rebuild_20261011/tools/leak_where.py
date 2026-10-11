import sys, numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
b = trimesh.load(sys.argv[1], force="mesh", process=False)
g = trimesh.load(sys.argv[2], force="mesh", process=False)
bi, gi = RayMeshIntersector(b), RayMeshIntersector(g)
rng = np.random.default_rng(1)
lo, hi = b.bounds[0], b.bounds[1]
xy = rng.uniform(lo[:2], hi[:2], size=(int(sys.argv[3]) if len(sys.argv) > 3 else 200000, 2))
o = np.column_stack([xy, np.full(len(xy), 5000.0)])
down = np.tile([0, 0, -1.0], (len(xy), 1))
loc, ir, _ = gi.intersects_location(o, down, multiple_hits=False)
zg = np.full(len(xy), np.nan); zg[ir] = loc[:, 2]
loc, ir, _ = bi.intersects_location(o, down, multiple_hits=False)
zb = np.full(len(xy), -np.inf); zb[ir] = loc[:, 2]
ok = np.isfinite(zg) & (zb < zg + 1.1)
p = np.column_stack([xy[ok], zg[ok] + 1.1])
d = rng.normal(size=(len(p), 3)); d[:, 2] = np.abs(d[:, 2]) * 0.5; d /= np.linalg.norm(d, axis=1, keepdims=True)
loc, ir, it = bi.intersects_location(p, d, multiple_hits=False)
back = (b.face_normals[it] * d[ir]).sum(1) > 0
nz = b.face_normals[it[back], 2]
print("back hits:", back.sum(), " normal z of hit faces:", np.round(nz, 2)[:30])
# for each back hit: hit location, its height vs ground below, and the origin
for k in np.flatnonzero(back)[:12]:
    print("origin", np.round(p[ir[k]], 2), "hit", np.round(loc[k], 2), "face n", np.round(b.face_normals[it[k]], 2),
          "dist", round(float(np.linalg.norm(loc[k] - p[ir[k]])), 2))

def roof_above(q):
    o = q.copy(); o[:, 2] = 5000.0
    l, r, t = bi.intersects_location(o, np.tile([0, 0, -1.0], (len(q), 1)), multiple_hits=False)
    z = np.full(len(q), -np.inf); z[r] = l[:, 2]
    return z
print("--- entry points")
for k in np.flatnonzero(back)[:15]:
    o0 = p[ir[k]]; dd = d[ir[k]]; L = np.linalg.norm(loc[k] - o0)
    s = np.linspace(0, L, int(L / 0.05) + 2)
    q = o0 + s[:, None] * dd
    za = roof_above(q)
    inside = za > q[:, 2]
    j = np.argmax(inside) if inside.any() else -1
    if j >= 0:
        e = q[j]
        print("enter at", np.round(e, 2), "roof above", round(float(za[j]), 2), "ground there?",
              "dist to hit", round(float(L - s[j]), 2))
    else:
        print("never under a roof before hit; hit", np.round(loc[k], 2))
