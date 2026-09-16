#!/usr/bin/env python3
"""verify_vegetation_crowns.py -- suite for the LiDAR crown geometry.

Covers the height-banded crown fit in 02_vegetation_to_stl.py: shape recovery
from synthetic clouds of known geometry, watertightness (which the
Beer-Lambert path length depends on), robustness to LiDAR noise, and the
backward-compatible hemisphere path.

Run: python3 verify_vegetation_crowns.py   (exits nonzero on failure)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
passed = 0
failed = 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition: bool, description: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {description}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {description}" + (f" ({detail})" if detail else ""))


veg = load_module(HERE / "02_vegetation_to_stl.py", "veg_crowns")
rng = np.random.default_rng(20260829)


def crown_cloud(a, b, c, n=4000, cut=-0.25, noise=0.05):
    """Points on an ellipsoid shell, sampled the way airborne LiDAR would:
    mostly the upper surface, with a little penetration below."""
    points = rng.normal(size=(n * 4, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points = points[points[:, 2] > cut][:n] * np.array([a, b, c])
    return points + rng.normal(scale=noise, size=points.shape)


def fit(points, **kwargs):
    settings = dict(n_lat=8, n_lon=16, max_radius=25.0, min_radius=0.2)
    settings.update(kwargs)
    axis, levels, radius, coverage = veg.crown_radial_profile(points, **settings)
    return veg.build_crown_mesh(axis, levels, radius), coverage


print("=" * 70)
print("LIDAR CROWN GEOMETRY VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------------------
print("\nT1: watertight by construction -- the transmission integral needs it")
# Beer-Lambert path length pairs ray entries with exits. A crown with a hole
# reports a nonsense chord rather than an obvious visual defect, so closure is
# a correctness requirement, not a cosmetic one.
for label, shape in (("spreading", (8, 8, 4)), ("columnar", (3, 3, 9)),
                     ("spherical", (6, 6, 6)), ("asymmetric", (7, 4, 5))):
    mesh, _ = fit(crown_cloud(*shape))
    if not mesh.is_watertight:
        check(False, f"{label} crown is watertight")
        break
else:
    check(True, "every fitted crown shape is watertight")
mesh, _ = fit(crown_cloud(6, 6, 6))
check(mesh.volume > 0, "and encloses a positive volume",
      f"{mesh.volume:.1f} m3")
check(mesh.is_winding_consistent, "with consistent winding")

# A ray through a closed crown must cross an EVEN number of faces -- that is
# what makes the entry/exit pairing in the transmission integral well defined.
# Counted directly with Moller-Trumbore so the check does not depend on an
# optional spatial-index package being installed.
def crossing_count(mesh, origin, direction):
    triangles = mesh.vertices[mesh.faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(determinant) < 1e-12
    inv = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, determinant))
    tvec = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv
    distance = np.einsum("ij,ij->i", edge2, qvec) * inv
    hit = (~parallel) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) \
        & (distance > 1e-9)
    return int(hit.sum())


origins = np.array([[-50.0, 0.0, 0.0], [0.0, -50.0, 2.0], [-50.0, 1.0, -1.0]])
directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
counts = [crossing_count(mesh, o, d) for o, d in zip(origins, directions)]
check(all(count % 2 == 0 and count > 0 for count in counts),
      "rays through the crown cross an EVEN, non-zero number of faces, so "
      "entry/exit pairing is well defined", f"crossings {counts}")

# ---------------------------------------------------------------------------
print("\nT2: height and width are INDEPENDENT (the hemisphere's core failure)")
# A flat-based dome forces height == radius. On the real lisbon1 mesh every one
# of 2441 crowns had a height/radius ratio of exactly 1.00 -- a property of the
# assumption, not of the trees.
ratios = {}
for label, shape in (("spreading", (8, 8, 4)), ("columnar", (3, 3, 9)),
                     ("spherical", (6, 6, 6))):
    cloud = crown_cloud(*shape)
    mesh, _ = fit(cloud)
    extent = mesh.bounds[1] - mesh.bounds[0]
    observed = cloud.max(axis=0) - cloud.min(axis=0)
    ratios[label] = (extent[2] / (0.5 * (extent[0] + extent[1])),
                     observed[2] / (0.5 * (observed[0] + observed[1])))
check(ratios["columnar"][0] > 2.5 * ratios["spreading"][0],
      "a columnar crown comes out far taller-than-wide than a spreading one",
      f"{ratios['columnar'][0]:.2f} vs {ratios['spreading'][0]:.2f}")
for label, (fitted, observed) in ratios.items():
    if abs(fitted - observed) > 0.15 * max(observed, 1e-9):
        check(False, f"{label}: fitted height/width tracks the observed cloud",
              f"{fitted:.2f} vs {observed:.2f}")
        break
else:
    check(True, "fitted height/width tracks the OBSERVED cloud within 15% for "
                "every shape", str({k: round(v[0], 2) for k, v in ratios.items()}))
check(len({round(v[0], 2) for v in ratios.values()}) == len(ratios),
      "and the three shapes give three DIFFERENT ratios, where a hemisphere "
      "would give 1.00 three times")

# Vertical extent must come from the data, not from the radius.
cloud = crown_cloud(3, 3, 9)
mesh, _ = fit(cloud)
observed_height = cloud[:, 2].max() - cloud[:, 2].min()
fitted_height = mesh.bounds[1][2] - mesh.bounds[0][2]
check(abs(fitted_height - observed_height) < 0.15 * observed_height,
      "crown height reproduces the observed vertical extent",
      f"{fitted_height:.1f} m vs {observed_height:.1f} m observed")

# ---------------------------------------------------------------------------
print("\nT3: robust to the noise LiDAR actually has")
base = crown_cloud(6, 6, 6, noise=0.05)
clean_mesh, _ = fit(base)
spiked = np.vstack([base, np.array([[40.0, 0.0, 4.0], [0.0, 35.0, 3.0],
                                    [-38.0, -5.0, 2.0]])])
spiked_mesh, _ = fit(spiked)
clean_extent = clean_mesh.bounds[1] - clean_mesh.bounds[0]
spiked_extent = spiked_mesh.bounds[1] - spiked_mesh.bounds[0]
check(np.max(np.abs(spiked_extent[:2] - clean_extent[:2])) < 2.0,
      "three stray branch-tip returns 30 m out do not inflate the crown -- the "
      "per-cell quantile leaves them outside it",
      f"width change {np.max(np.abs(spiked_extent[:2] - clean_extent[:2])):.2f} m")
check(spiked_mesh.is_watertight, "and the crown stays watertight with them")

sparse = crown_cloud(6, 6, 6, n=60)
sparse_mesh, coverage = fit(sparse)
check(sparse_mesh.is_watertight and sparse_mesh.volume > 0,
      "a sparse 60-point crown still produces a usable closed mesh",
      f"coverage {coverage:.0%}")
check(coverage < 1.0,
      "and reports that not every cell was measured, rather than implying it "
      "was", f"{coverage:.0%} of cells measured")

# An empty band in the middle must not pinch the mesh shut.
gapped = base[(base[:, 2] < 1.0) | (base[:, 2] > 3.0)]
gapped_mesh, _ = fit(gapped)
check(gapped_mesh.is_watertight and gapped_mesh.volume > 0,
      "a crown with an unsampled middle band is interpolated, not pinched",
      f"volume {gapped_mesh.volume:.1f} m3")

# ---------------------------------------------------------------------------
print("\nT4: the fit is bounded and cannot produce absurd crowns")
mesh, _ = fit(crown_cloud(6, 6, 6), max_radius=3.0)
extent = mesh.bounds[1] - mesh.bounds[0]
check(max(extent[0], extent[1]) <= 6.0 + 1e-6,
      "max_radius caps a crown from an under-segmented cluster",
      f"width {max(extent[0], extent[1]):.2f} m vs cap 6.0 m")
mesh, _ = fit(crown_cloud(0.4, 0.4, 0.5), min_radius=1.0)
extent = mesh.bounds[1] - mesh.bounds[0]
check(min(extent[0], extent[1]) >= 2.0 - 1e-6,
      "min_radius prevents degenerate slivers from a tight cluster",
      f"width {min(extent[0], extent[1]):.2f} m")
try:
    veg.crown_radial_profile(np.zeros((2, 3)))
    check(False, "a crown with too few points is rejected")
except ValueError:
    check(True, "a crown with too few points is rejected")
try:
    flat = np.column_stack([rng.normal(size=50), rng.normal(size=50),
                            np.full(50, 5.0)])
    veg.crown_radial_profile(flat)
    check(False, "a crown with no vertical extent is rejected")
except ValueError:
    check(True, "a crown with no vertical extent is rejected")

# ---------------------------------------------------------------------------
print("\nT5: the underside is a MODEL and is treated as one")
# Airborne LiDAR sees a crown from above. Whatever fills the lower cells is a
# modelling choice, and the code must not present it as measurement.
cloud = crown_cloud(6, 6, 6)
axis, levels, radius, coverage = veg.crown_radial_profile(
    cloud, n_lat=8, n_lon=16, max_radius=25.0, lower_taper=0.65)
check(radius[0].mean() < radius[len(radius) // 2].mean(),
      "the bottom band is tapered toward the trunk, not left as the flat disc "
      "the hemisphere used",
      f"bottom {radius[0].mean():.2f} m vs mid {radius[len(radius) // 2].mean():.2f} m")
_, _, wide, _ = veg.crown_radial_profile(cloud, n_lat=8, n_lon=16,
                                         max_radius=25.0, lower_taper=1.0)
check(wide[0].mean() > radius[0].mean(),
      "the taper is a documented, adjustable prior rather than hard-wired")
source = (HERE / "02_vegetation_to_stl.py").read_text(encoding="utf-8")
check("airborne lidar" in source.lower() or "from ABOVE" in source
      or "from above" in source.lower(),
      "the code states that LiDAR cannot see the underside")
check("coverage" in source and "modelled" in source.lower(),
      "and reports measured-versus-modelled coverage to the operator")

# ---------------------------------------------------------------------------
print("\nT5b: the trunk is separated from the crown")
# A vegetation CLUSTER is not a crown -- LiDAR returns include trunk hits and
# understory. Without trimming, canopy-width geometry gets wrapped around the
# trunk: on the real Lisbon cloud that gave a median crown height of 18.6 m.
canopy = rng.normal(size=(12000, 3))
canopy /= np.linalg.norm(canopy, axis=1)[:, None]
canopy = canopy[canopy[:, 2] > -0.3][:3000] * np.array([5, 5, 3]) \
    + np.array([0, 0, 11])
trunk = np.column_stack([rng.normal(scale=0.3, size=120),
                         rng.normal(scale=0.3, size=120),
                         rng.uniform(2.0, 8.0, 120)])
with_trunk = np.vstack([canopy, trunk])
mesh, _ = fit(with_trunk)
check(mesh.bounds[0][2] > 8.0,
      "the sparse trunk column below the canopy is trimmed off",
      f"crown base {mesh.bounds[0][2]:.1f} m, canopy starts at 8 m")
height = mesh.bounds[1][2] - mesh.bounds[0][2]
check(height < 9.0,
      "so crown height reflects the CANOPY, not the whole cluster",
      f"{height:.1f} m from a cluster spanning "
      f"{with_trunk[:, 2].max() - with_trunk[:, 2].min():.1f} m")
check(mesh.is_watertight and mesh.volume > 0,
      "and the trimmed crown is still a valid closed volume")
# The trim must run on MEASURED band widths. Filling sparse bands first lets an
# empty trunk band inherit a canopy radius by interpolation and pass the test --
# that ordering bug left the trim silently inert.
crown_source = (HERE / "02_vegetation_to_stl.py").read_text(encoding="utf-8")
trim_at = crown_source.index("measured_band = np.array")
fill_at = crown_source.index("band_radius = np.array")
check(trim_at < fill_at,
      "the trim is computed BEFORE any filling, or it would test values the "
      "fill invented")
# A crown with no trunk must not be trimmed away.
no_trunk_mesh, _ = fit(canopy)
check(no_trunk_mesh.bounds[1][2] - no_trunk_mesh.bounds[0][2] > 3.0,
      "a crown with no trunk returns keeps its full canopy depth",
      f"{no_trunk_mesh.bounds[1][2] - no_trunk_mesh.bounds[0][2]:.1f} m")

print("\nT5c: crown depth is bounded botanically")
# On the real cloud the fit gave ~18 m depth at ~7 m width -- slender columns.
# Not terrain (ground relief under those footprints is 0.9 m median), so it
# points upstream at clustering. The hemisphere hid it by setting depth =
# radius; this cap is a far weaker bound applied while that is investigated.
column = np.column_stack([rng.normal(scale=2.0, size=4000),
                          rng.normal(scale=2.0, size=4000),
                          rng.uniform(2.0, 22.0, 4000)])
uncapped, _ = fit(column, max_depth_ratio=99.0)
capped, _ = fit(column, max_depth_ratio=2.0)
def ratio_of(mesh):
    extent = mesh.bounds[1] - mesh.bounds[0]
    return extent[2] / (0.5 * (extent[0] + extent[1]))
check(ratio_of(uncapped) > 2.4 and ratio_of(capped) <= 2.05,
      "an over-extended cluster is capped at the configured depth/width ratio",
      f"{ratio_of(uncapped):.2f} -> {ratio_of(capped):.2f}")
check(capped.bounds[1][2] >= uncapped.bounds[1][2] - 1e-6,
      "the cap trims from the BOTTOM, keeping the measured canopy top that "
      "airborne LiDAR actually resolves")
check(capped.is_watertight and capped.volume > 0,
      "and the capped crown is still a valid closed volume")
normal_cloud = crown_cloud(6, 6, 6)
uncapped_normal, _ = fit(normal_cloud, max_depth_ratio=99.0)
capped_normal, _ = fit(normal_cloud, max_depth_ratio=2.0)
check(abs(ratio_of(capped_normal) - ratio_of(uncapped_normal)) < 0.05,
      "a normally proportioned crown is untouched by the cap -- it is a bound, "
      "not a reshaping", f"{ratio_of(capped_normal):.2f}")

print("\nT6: mesh cost stays within the raytracing budget")
mesh, _ = fit(crown_cloud(6, 6, 6), n_lat=8, n_lon=16)
check(len(mesh.faces) <= 256,
      "the default grid costs no more than the hemisphere it replaces",
      f"{len(mesh.faces)} faces (hemisphere used 256)")
coarse, _ = fit(crown_cloud(6, 6, 6), n_lat=5, n_lon=10)
fine, _ = fit(crown_cloud(6, 6, 6), n_lat=12, n_lon=24)
check(len(coarse.faces) < len(mesh.faces) < len(fine.faces),
      "resolution is directly controllable",
      f"{len(coarse.faces)} / {len(mesh.faces)} / {len(fine.faces)} faces")
check(coarse.is_watertight and fine.is_watertight,
      "and every resolution stays watertight")

# ---------------------------------------------------------------------------
print("\nT8: alpha-shape reconstruction assumes NO primitive")
import crown_reconstruction as cr
from scipy.spatial import cKDTree


def lobed_cloud(n=6000, noise=0.05, seed=11):
    """Deliberately non-primitive: three lobes, vertical modulation, a fork.
    No ellipsoid, hemisphere, cylinder or cone can represent this."""
    gen = np.random.default_rng(seed)
    direction = gen.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    lobe = (1.0 + 0.35 * np.sin(3 * np.arctan2(direction[:, 1], direction[:, 0]))
            + 0.25 * np.cos(4 * direction[:, 2]))
    body = direction * lobe[:, None] * np.array([4.5, 3.5, 3.0])
    fork = direction[:n // 4] * np.array([1.4, 1.4, 2.2]) + np.array([3.5, 2.0, 3.5])
    surface = np.vstack([body, fork])
    return surface, surface + gen.normal(scale=noise, size=surface.shape)


truth, cloud = lobed_cloud()
strays = np.array([[25.0, 0.0, 2.0], [0.0, 22.0, 1.0], [-20.0, -18.0, 4.0]])
mesh, report = cr.reconstruct_crown(
    np.vstack([cloud, strays]),
    cr.ReconstructionSettings(voxel_size=0.10, max_faces=4000))
check(mesh.is_watertight, "the reconstruction is watertight",
      f"{report.n_faces} faces")
check(mesh.is_winding_consistent and mesh.volume > 0,
      "manifold with consistent outward winding", f"volume {mesh.volume:.0f} m3")
extent = mesh.bounds[1] - mesh.bounds[0]
check(extent.max() < 20.0,
      "the three stray returns 20-25 m out are rejected, not enclosed",
      f"extent {np.round(extent, 1)} m")
check(report.n_after_filtering < report.n_after_downsample,
      "the outlier and density filters actually removed points",
      f"{report.n_after_downsample} -> {report.n_after_filtering}")
distance, _ = cKDTree(mesh.vertices).query(truth)
check(np.sqrt((distance ** 2).mean()) < 1.0,
      "the surface follows the true irregular shape",
      f"RMS {np.sqrt((distance ** 2).mean()):.2f} m to the true surface")
hull_volume = trimesh.PointCloud(cloud).convex_hull.volume
check(mesh.volume < 0.92 * hull_volume,
      "and is a TIGHT enclosure -- materially smaller than the convex hull, "
      "which is what proves the concavities are captured",
      f"{mesh.volume:.0f} vs hull {hull_volume:.0f} m3")

# Element count must stay controllable.
budget = cr.reconstruct_crown(cloud, cr.ReconstructionSettings(
    voxel_size=0.10, max_faces=800))[0]
check(len(budget.faces) <= 900 and budget.is_watertight,
      "the face budget is enforced and survives decimation watertight",
      f"{len(budget.faces)} faces")
coarse = cr.reconstruct_crown(cloud, cr.ReconstructionSettings(
    voxel_size=0.30, max_faces=None))[1]
fine = cr.reconstruct_crown(cloud, cr.ReconstructionSettings(
    voxel_size=0.10, max_faces=None))[1]
check(fine.n_after_downsample > coarse.n_after_downsample,
      "a finer voxel size retains more structure",
      f"{coarse.n_after_downsample} -> {fine.n_after_downsample} points")

# Adaptivity: alpha must scale with the cloud's own spacing, not be a constant.
sparse_truth, sparse_cloud = lobed_cloud(n=900, seed=12)
sparse_report = cr.reconstruct_crown(sparse_cloud,
                                     cr.ReconstructionSettings(voxel_size=0.10))[1]
check(sparse_report.alpha_m != report.alpha_m,
      "alpha adapts to the cloud's own characteristic spacing rather than "
      "being fixed",
      f"{report.alpha_m:.2f} m dense vs {sparse_report.alpha_m:.2f} m sparse")
check(report.coverage >= 0.95,
      "the enclosure meets the coverage target on the measured points",
      f"{report.coverage:.1%}")

# Exact containment via the Delaunay itself -- no ray casting, no rtree.
solid = cr.alpha_solid(cloud[::4], 2.0)
check(solid is not None, "alpha_solid returns a retained tetrahedron set")
inside = cr.points_inside_solid(solid[0], solid[1],
                                np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
check(bool(inside[0]) and not bool(inside[1]),
      "containment is answered exactly from the tetrahedralisation: the crown "
      "centre is inside, a point 50 m away is not")

try:
    cr.reconstruct_crown(np.zeros((5, 3)))
    check(False, "a cluster with too few points is rejected")
except cr.CrownReconstructionError:
    check(True, "a cluster with too few points is rejected")

print("\nT7: the hemisphere path is preserved for reproducibility")
centre_x, centre_y, base_z, radius_value = veg.fit_hemisphere(
    crown_cloud(6, 6, 6))
dome = veg.build_hemisphere_mesh(centre_x, centre_y, base_z, radius_value)
check(dome.is_watertight, "the legacy hemisphere still builds and is watertight")
extent = dome.bounds[1] - dome.bounds[0]
check(abs(extent[2] / (0.5 * (extent[0] + extent[1])) - 0.5) < 0.02,
      "and still forces height = radius (ratio 0.5 on diameter), which is the "
      "behaviour the radial model exists to remove",
      f"{extent[2] / (0.5 * (extent[0] + extent[1])):.3f}")
check("--crown-model" in source and '"hemisphere"' in source,
      "and is reachable via --crown-model hemisphere")

print("\n" + "=" * 70)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
