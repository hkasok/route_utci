#!/usr/bin/env python3
"""
verify_crown_field.py -- checks for crown_field_reconstruction.py and its
wiring into 02_vegetation_to_stl.py.

Exits nonzero on failure, like every other verify_* suite here.

The property under test is the one the earlier models failed: that the
surface carries NO shape prior. A suite that only checked "watertight" would
pass just as happily on a sphere, so the shape tests here are built from
inputs whose correct answer is impossible for an axisymmetric or star-shaped
model to produce -- an L, a fork, a ring.
"""

import subprocess
import sys

import numpy as np
import trimesh

import crown_field_reconstruction as cfr

FAILURES = []
N = 0


def check(name, cond, detail=""):
    global N
    N += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def _raises(fn):
    """True if fn raises CrownFieldError -- the module's own failure signal."""
    try:
        fn()
    except cfr.CrownFieldError:
        return True
    except Exception:
        return False
    return False


def cloud(centres, spread, n, seed=7):
    g = np.random.default_rng(seed)
    return np.vstack([g.normal(c, spread, (n, 3)) for c in centres])


def solidity(mesh):
    """Volume / convex-hull volume. 1.0 = convex; a primitive model scores ~1.

    Deliberately NOT a shapely cross-section test: trimesh routes polygon
    containment through `rtree`, which is not installed here.
    """
    hull = mesh.convex_hull.volume
    return mesh.volume / hull if hull > 0 else None


def slice_lobes(points, height, settings=None):
    """Connected components of the solid field on one horizontal slice.

    Counted on the FIELD, not on a mesh section, for the same rtree reason.
    """
    from scipy import ndimage
    solid, origin, voxel = cfr.solid_field(points, settings)
    k = int(round((height - origin[2]) / voxel))
    k = max(0, min(solid.shape[2] - 1, k))
    return int(ndimage.label(solid[:, :, k])[1])


# ------------------------------------------------------------ A: the field
print("\n[A] solid_field and containment")
blob = cloud([(0, 0, 0)], 2.0, 3000)
solid, origin, voxel = cfr.solid_field(blob)
check("A1 field is a non-empty boolean volume",
      solid.dtype == bool and solid.sum() > 0)
# The voxel follows the point spacing, but is ALSO capped so the bridge
# radius is resolved -- a sparse cloud would otherwise get a voxel larger
# than the radius and the balls would vanish sub-voxel.
_cap = cfr.FieldSettings.bridge_radius_m / cfr.VOXELS_PER_BRIDGE_RADIUS
check("A2 voxel follows point spacing unless the radius cap binds",
      abs(voxel - min(max(cfr.FieldSettings.target_resolution_m,
                          cfr.characteristic_spacing(blob)), _cap)) < 1e-9,
      f"voxel {voxel:.3f}, spacing {cfr.characteristic_spacing(blob):.3f}, cap {_cap:.3f}")
check("A3 nearly all points land inside the solid",
      cfr.containment(blob, solid, origin, voxel) > 0.95,
      f"got {cfr.containment(blob, solid, origin, voxel):.1%}")
# Compare VOLUME, not voxel count: a larger radius also relaxes the voxel
# cap, so the two fields need not share a cell size.
def _solid_volume(points, settings=None):
    s_, _, v_ = cfr.solid_field(points, settings)
    return s_.sum() * v_ ** 3


check("A4 a larger bridge radius cannot shrink the solid's volume",
      _solid_volume(blob, cfr.FieldSettings(bridge_radius_m=1.2))
      >= _solid_volume(blob) * 0.99)
check("A5 fewer points than minimum_points raises rather than returning junk",
      _raises(lambda: cfr.solid_field(
          np.zeros((cfr.FieldSettings.minimum_points - 1, 3)))))
check("A6 the grid cap coarsens the voxel instead of exhausting memory",
      cfr.solid_field(cloud([(0, 0, 0)], 40.0, 4000),
                      cfr.FieldSettings(target_resolution_m=0.01,
                                        max_voxels=2e5))[2] > 0.01)

# ------------------------------------------------------- B: surface quality
print("\n[B] reconstruct_crown")
mesh, report = cfr.reconstruct_crown(blob)
check("B1 surface is watertight", mesh.is_watertight, f"bodies={mesh.body_count}")
check("B2 winding is consistent", mesh.is_winding_consistent)
check("B3 volume is positive", mesh.volume > 0, f"got {mesh.volume:.1f}")
check("B4 volume is physically sane (below the bounding box)",
      mesh.volume < np.prod(blob.max(0) - blob.min(0)),
      f"{mesh.volume:.0f} vs bbox {np.prod(blob.max(0)-blob.min(0)):.0f}")
check("B5 report containment matches the field", report.containment > 0.95)
check("B6 face budget is respected within tolerance",
      report.n_faces <= 4 * cfr.FieldSettings.face_budget, f"got {report.n_faces}")
check("B7 report records the voxel actually used",
      report.voxel_m == voxel, f"{report.voxel_m} vs {voxel}")

# ------------------------------------------------- C: NO SHAPE PRIOR (core)
print("\n[C] no shape prior -- shapes a primitive model cannot represent")
# An L in plan: any star-shaped-about-an-axis model must bridge the notch.
ell = cloud([(0, 0, 0), (4, 0, 0), (8, 0, 0), (8, 4, 0), (8, 8, 0)], 1.0, 1200)
lmesh, _ = cfr.reconstruct_crown(ell)
lc = solidity(lmesh)
check("C1 an L-shaped cloud stays NON-convex (a star hull would fill the notch)",
      lc is not None and lc < 0.80, f"solidity {lc:.2f}")
check("C2 the L stays watertight", lmesh.is_watertight)

# A fork: one stem carrying two well-separated branches. A hemisphere or a
# star hull about a vertical axis must merge the branches into one blob.
# Built as explicit cylinders rather than Gaussian blobs so the answer is
# deterministic -- overlapping tails would make "how many lobes" ambiguous.
def cylinder(cx, cy, z0, z1, radius, n, seed=3):
    g = np.random.default_rng(seed)
    theta = g.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(g.uniform(0, 1, n))
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta),
                            g.uniform(z0, z1, n)])


fork = np.vstack([cylinder(0, 0, 0.0, 5.0, 1.2, 1500),
                  cylinder(-5, 0, 5.0, 11.0, 1.2, 1500),
                  cylinder(5, 0, 5.0, 11.0, 1.2, 1500)])
lobes = slice_lobes(fork, 8.0)
check("C3 a forked crown resolves as two lobes at the fork height",
      lobes == 2, f"got {lobes}")
check("C3b and as ONE body lower down where the stem is",
      slice_lobes(fork, 2.0) == 1, f"got {slice_lobes(fork, 2.0)}")

# A ring: the classic case an axisymmetric model fills solid.
ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
ring = cloud([(6 * np.cos(a), 6 * np.sin(a), 0) for a in ang], 0.7, 60)
rmesh, _ = cfr.reconstruct_crown(ring)
rc = solidity(rmesh)
check("C4 a ring keeps its hole (far from convex)",
      rc is not None and rc < 0.70, f"solidity {rc:.2f}")
check("C5 the ring has genus > 0 (a real hole, not a disc)",
      rmesh.is_watertight and rmesh.euler_number <= 0,
      f"euler {rmesh.euler_number}")

# Height and width are independent -- the hemisphere failure mode.
tall = cloud([(0, 0, z) for z in range(0, 18, 2)], 1.0, 400)
wide = cloud([(x, 0, 0) for x in range(0, 18, 2)], 1.0, 400)
tm, _ = cfr.reconstruct_crown(tall)
wm, _ = cfr.reconstruct_crown(wide)
tr = (tm.bounds[1, 2] - tm.bounds[0, 2]) / (tm.bounds[1, 0] - tm.bounds[0, 0])
wr = (wm.bounds[1, 2] - wm.bounds[0, 2]) / (wm.bounds[1, 0] - wm.bounds[0, 0])
check("C6 a columnar cloud stays tall and narrow", tr > 2.0, f"depth/width {tr:.2f}")
check("C7 a spreading cloud stays short and wide", wr < 0.5, f"depth/width {wr:.2f}")
check("C8 the two differ -- no fixed axis ratio is imposed", tr / wr > 5.0)

# ------------------------------------- Cx: small and sparse clusters
print("\n[Cx] small and sparse clusters must NOT fall back to a primitive")
g_small = np.random.default_rng(11)
small_ok = 0
small_total = 0
for n in (5, 8, 11, 15, 20, 23, 30):
    for spread in (1.0, 4.0):
        small_total += 1
        try:
            m, r = cfr.reconstruct_crown(g_small.normal((0, 0, 0), spread, (n, 3)))
            if r.watertight:
                small_ok += 1
        except cfr.CrownFieldError:
            pass
check("Cx1 every small/sparse cluster reconstructs watertight",
      small_ok == small_total, f"{small_ok}/{small_total}")
check("Cx2 the field minimum does not exceed --min-hull-points default (10)",
      cfr.FieldSettings.minimum_points <= 10,
      f"minimum_points={cfr.FieldSettings.minimum_points}")
# The voxel must resolve the bridge radius or the balls vanish sub-voxel.
sparse = g_small.normal((0, 0, 0), 6.0, (12, 3))
_, _, sparse_voxel = cfr.solid_field(sparse)
check("Cx3 voxel stays small enough to resolve the bridge radius",
      sparse_voxel <= cfr.FieldSettings.bridge_radius_m / 2.0,
      f"voxel {sparse_voxel:.3f} vs radius {cfr.FieldSettings.bridge_radius_m}")


# ------------------------------------------------------------ D: robustness
print("\n[D] robustness")
noisy = np.vstack([blob, np.random.default_rng(1).uniform(-25, 25, (40, 3))])
nm, nrep = cfr.reconstruct_crown(noisy)
check("D1 scattered outliers do not shatter the crown into specks",
      nm.body_count <= 4, f"bodies {nm.body_count}")
check("D2 the result is still closed", nm.is_watertight or nrep.repaired)
check("D3 translation invariance of containment",
      abs(cfr.reconstruct_crown(blob + 1000.0)[1].containment - report.containment) < 0.02)
def _degenerate_ok():
    """A fully degenerate cloud must not crash: either it is refused, or it
    yields a small closed solid. Both are acceptable; a traceback is not."""
    try:
        m, _ = cfr.reconstruct_crown(np.zeros((30, 3)))
    except cfr.CrownFieldError:
        return True
    except Exception:
        return False
    return m.is_watertight and m.volume >= 0


check("D4 a fully degenerate cloud is handled, not crashed", _degenerate_ok())

# ------------------------------------------------------------ E: integration
print("\n[E] 02_vegetation_to_stl.py wiring")
proc = subprocess.run([sys.executable, "02_vegetation_to_stl.py", "--help"],
                      capture_output=True, text=True)
check("E1 --help runs", proc.returncode == 0, proc.stderr[-300:])
h = proc.stdout
check("E2 'field' is the default crown model", "default: field" in h or "'field' (default)" in h)
for flag in ("--crown-bridge-radius", "--crown-face-budget", "--crown-smooth-iterations"):
    check(f"E3 {flag} exposed", flag in h)
src = open("02_vegetation_to_stl.py").read()
check("E4 the field model is dispatched", 'args.crown_model == "field"' in src)
check("E5 a hemisphere fallback under the field model is REPORTED, not silent",
      'back to a HEMISPHERE' in src)
check("E6 older models remain reachable",
      '"radial"' in src and '"hemisphere"' in src)

print(f"\n{'=' * 62}")
if FAILURES:
    print(f"FAILED {len(FAILURES)}/{N}: " + ", ".join(FAILURES))
    sys.exit(1)
print(f"All {N} crown-field checks passed.")
