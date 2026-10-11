"""New (normal-side) vs old (alternating) crown path length.
Outside origins: must agree except where crowns overlap.
Inside origins: new must equal the distance to the exit plus later chords."""
import sys, numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
sys.path.insert(0, ".")
from thermal_common import vegetation_transmission_from_intersections as new_fn

def old_fn(inter, origins, directions, k_lad, min_distance=1e-6, unique_tol=1e-5):
    n_rays = origins.shape[0]
    locations, index_ray, _ = inter.intersects_location(origins, directions, multiple_hits=True)
    if len(index_ray) == 0:
        return np.ones(n_rays), np.zeros(n_rays)
    dist = np.einsum("ij,ij->i", locations - origins[index_ray], directions[index_ray])
    valid = dist > min_distance; index_ray, dist = index_ray[valid], dist[valid]
    order = np.lexsort((dist, index_ray)); index_ray, dist = index_ray[order], dist[order]
    same = np.r_[False, index_ray[1:] == index_ray[:-1]]; close = np.r_[False, np.diff(dist) <= unique_tol]
    keep = ~(same & close); index_ray, dist = index_ray[keep], dist[keep]
    gc = np.r_[True, index_ray[1:] != index_ray[:-1]]; ia = np.arange(len(index_ray))
    gs = np.maximum.accumulate(np.where(gc, ia, 0)); pos = ia - gs
    gid, gcnt = np.unique(index_ray, return_counts=True); sl = np.zeros(n_rays, int); sl[gid] = gcnt
    gsz = sl[index_ray]; odd_last = (pos == gsz - 1) & (gsz % 2 == 1)
    sign = np.where(pos % 2 == 0, -1.0, 1.0); sign[odd_last] = 0
    L = np.maximum(np.bincount(index_ray, weights=sign * dist, minlength=n_rays), 0)
    return np.exp(-k_lad * L), L

veg = trimesh.load(sys.argv[1], force="mesh", process=False)
inter = RayMeshIntersector(veg)
rng = np.random.default_rng(3)
# normals outward? centroid +/- 5 cm, inside test by the new function's own parity is circular, so
# use trimesh contains on a sample (independent ray-parity implementation)
fi = rng.choice(len(veg.faces), 3000, replace=False)
c, n = veg.triangles_center[fi], veg.face_normals[fi]
out_in = veg.contains(c + 0.05 * n); in_in = veg.contains(c - 0.05 * n)
print(f"outward-normal check: outside point inside crown {out_in.mean():.3%}, inside point inside crown {in_in.mean():.3%}")
lo, hi = veg.bounds
# random rays from points above canopy-free ground level and from crown interiors
o = rng.uniform(lo, hi, size=(200000, 3))
d = rng.normal(size=o.shape); d /= np.linalg.norm(d, axis=1, keepdims=True)
inside = veg.contains(o)
_, Ln = new_fn(inter, o, d, 0.45)
_, Lo = old_fn(inter, o, d, 0.45)
out = ~inside
diff = np.abs(Ln - Lo)
print(f"outside origins: {out.sum()} rays, {np.mean(Ln[out] > 0):.2%} cross crowns; "
      f"|new-old| > 1 cm on {np.mean(diff[out] > 0.01):.3%} (overlapping crowns / grazing); "
      f"mean L new {Ln[out][Ln[out]>0].mean():.3f} old {Lo[out][Lo[out]>0].mean():.3f} m")
print(f"inside origins: {inside.sum()} rays; mean L new {Ln[inside].mean():.2f} m, old {Lo[inside].mean():.2f} m; "
      f"old zero on {np.mean(Lo[inside] == 0):.1%}, new zero on {np.mean(Ln[inside] == 0):.1%}")
