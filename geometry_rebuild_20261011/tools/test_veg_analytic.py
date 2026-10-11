import sys, numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
sys.path.insert(0, ".")
from thermal_common import vegetation_transmission_from_intersections as f
def sph(c, r): 
    m = trimesh.creation.icosphere(subdivisions=6, radius=r); m.apply_translation(c); return m
x = np.array([1.0, 0, 0])
cases = []
# 1. two separate spheres r=1 at x=5 and x=10; origin 0 -> chords 2+2=4
m = trimesh.util.concatenate([sph([5,0,0],1), sph([10,0,0],1)]); cases.append(("separate", m, [0,0,0], 4.0))
# 2. overlapping spheres r=1 at x=5 and x=6 -> union [4,7] = 3
m = trimesh.util.concatenate([sph([5,0,0],1), sph([6,0,0],1)]); cases.append(("overlap", m, [0,0,0], 3.0))
# 3. origin inside sphere at x=0 (r=2), second sphere at x=6 r=1 -> 2 + 2 = 4
m = trimesh.util.concatenate([sph([0,0,0],2), sph([6,0,0],1)]); cases.append(("inside", m, [0,0,0], 4.0))
# 4. origin inside the overlap of two spheres at x=0 and x=1 (r=2) -> exits at 2 and 3 -> 3
m = trimesh.util.concatenate([sph([0,0,0],2), sph([1,0,0],2)]); cases.append(("inside overlap", m, [0.5,0,0], 2.5))
# 5. no crown
m = sph([0,10,0],1); cases.append(("miss", m, [0,0,0], 0.0))
ok = True
for name, m, o, expect in cases:
    _, L = f(RayMeshIntersector(m), np.array([o], float), np.array([x]), 0.45)
    good = abs(L[0] - expect) < 0.01
    ok &= good
    print(f"{name:15s} L={L[0]:.4f} expected {expect:.4f} {'OK' if good else 'FAIL'}")
# batch of many rays mixing cases: vectorisation must not leak between rays
m = trimesh.util.concatenate([sph([5,0,0],1), sph([6,0,0],1), sph([0,0,0],0.5)])
o = np.array([[0,0,0],[0,0,0],[0,0,0],[20,0,0]], float)
d = np.array([[1,0,0],[-1,0,0],[0,1,0],[-1,0,0]], float)
_, L = f(RayMeshIntersector(m), o, d, 0.45)
exp = [0.5 + 3.0, 0.5, 0.5, 3.0 + 1.0]
print("batch", np.round(L, 3), "expected", exp)
ok &= np.allclose(L, exp, atol=0.01)
print("ALL OK" if ok else "SOME FAILED")
