#!/usr/bin/env python3
"""verify_geometry_fixes.py -- checks for the 11 Oct 2026 geometry corrections.

  G1  crown path length: canopy union, origins inside crowns
  G2  terrain occludes the sun and the sky
  G3  route samples inside a building are ray traced from open air
  G4  05a facet normals come from the mesh, not from the first ray
  G5  shortwave enclosure: closed-form solution and energy bookkeeping
  G6  ground reflection from the ground facets: equals the old f = 0.5 rule
      on open ground, and is smaller in a street canyon (no double count)
  G7  segmented LoD1 builder: part heights, courtyard, walls below terrain,
      outward winding, no ray passes under or into a building

Run:  python3 verify_geometry_fixes.py
"""
import importlib.util
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from thermal_common import (solve_route_visible_shortwave_enclosure,  # noqa: E402
                            vegetation_transmission_from_intersections)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sphere(c, r, sub=6):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    m.apply_translation(c)
    return m


def plane(x0, x1, y0, y1, n=40, zfun=None):
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X) if zfun is None else zfun(X, Y)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    F = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            F += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return trimesh.Trimesh(V, np.array(F), process=False)


m05 = load_module(HERE / "05_mrt_network_raytrace.py", "m05")

# ----------------------------------------------------------------------------
print("=" * 70)
print("G1 crown path length")
x = np.array([1.0, 0.0, 0.0])
cases = [
    ("two separate crowns: sum of chords", [sphere([5, 0, 0], 1), sphere([10, 0, 0], 1)], [0, 0, 0], 4.0),
    ("overlapping crowns: their union, overlap counted once",
     [sphere([5, 0, 0], 1), sphere([6, 0, 0], 1)], [0, 0, 0], 3.0),
    ("origin inside a crown: origin-to-exit plus later chords",
     [sphere([0, 0, 0], 2), sphere([6, 0, 0], 1)], [0, 0, 0], 4.0),
    ("origin inside two overlapping crowns", [sphere([0, 0, 0], 2), sphere([1, 0, 0], 2)], [0.5, 0, 0], 2.5),
    ("no crown on the ray", [sphere([0, 10, 0], 1)], [0, 0, 0], 0.0),
]
for name, parts, origin, expected in cases:
    mesh = trimesh.util.concatenate(parts)
    tau, L = vegetation_transmission_from_intersections(
        RayMeshIntersector(mesh), np.array([origin], float), np.array([x]), 0.45)
    check(name, abs(L[0] - expected) < 0.01 and abs(tau[0] - np.exp(-0.45 * expected)) < 1e-3,
          f"L={L[0]:.3f} m, expected {expected}")
mesh = trimesh.util.concatenate([sphere([5, 0, 0], 1), sphere([6, 0, 0], 1), sphere([0, 0, 0], 0.5)])
o = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [20, 0, 0]], float)
d = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [-1, 0, 0]], float)
_, L = vegetation_transmission_from_intersections(RayMeshIntersector(mesh), o, d, 0.45)
check("rays of one batch do not leak into each other", np.allclose(L, [3.5, 0.5, 0.5, 4.0], atol=0.01),
      f"{np.round(L, 3).tolist()}")

# ----------------------------------------------------------------------------
print("=" * 70)
print("G2 terrain occlusion")
ridge = plane(0, 200, -50, 50, n=50, zfun=lambda X, Y: np.where(X > 60, 25.0, 0.0))
gi = RayMeshIntersector(ridge)
nothing = RayMeshIntersector(sphere([1000, 1000, -500], 0.1, 1))
receptor = np.array([[40.0, 0.0, 1.1]])
sun_low = m05.sun_vector_enu(90.0, 10.0)        # due east, 10 deg: behind the ridge
sun_high = m05.sun_vector_enu(90.0, 60.0)
t_without = m05.direct_solar_transmission_batched(receptor, sun_low, nothing, nothing, 0.45, 10)
t_with = m05.direct_solar_transmission_batched(receptor, sun_low, nothing, nothing, 0.45, 10,
                                               ground_intersector=gi)
t_high = m05.direct_solar_transmission_batched(receptor, sun_high, nothing, nothing, 0.45, 10,
                                               ground_intersector=gi)
check("a ridge shades the receptor from low sun", t_without[0] == 1.0 and t_with[0] == 0.0)
check("the same ridge does not shade high sun", t_high[0] == 1.0)
dirs, wp, wc, ws = m05.make_sky_directions(48, 12)
svf0 = m05.compute_effective_svf_batched(receptor, dirs, wp, wc, nothing, nothing, 0.3, 10, w_sph=ws)
svf1 = m05.compute_effective_svf_batched(receptor, dirs, wp, wc, nothing, nothing, 0.3, 10, w_sph=ws,
                                         ground_intersector=gi)
expected_loss = 0.5 * (1 - np.cos(np.arctan(23.9 / 20.0)))   # cylinder-free check: planar SVF
check("the ridge removes sky from the planar sky-view factor", svf1[1][0] < svf0[1][0] - 0.05,
      f"planar SVF {svf0[1][0]:.3f} -> {svf1[1][0]:.3f}")
check("building-only SVF is unchanged by terrain (diagnostic stays buildings-only)",
      abs(svf1[0][0] - svf0[0][0]) < 1e-12)

# ----------------------------------------------------------------------------
print("=" * 70)
print("G3 receptors inside building solids")
ground = plane(0, 80, 0, 80, n=40)
box = trimesh.creation.box(extents=[10, 10, 12])
box.apply_translation([40, 40, 6.0])
bi, gi = RayMeshIntersector(box), RayMeshIntersector(ground)
xy = np.array([[40.0, 40.0], [36.0, 40.0], [40.0, 44.5], [20.0, 20.0]])   # centre, near W, near N, open
zg = np.zeros(len(xy))
rxy, rzg, disp, inside, unresolved = m05.place_receptors_in_open_air(xy, zg, bi, gi, clearance_m=0.5)
check("inside points are detected, the open point is not", inside.tolist() == [True, True, True, False])
dist_to_box = np.maximum(np.abs(rxy - 40.0) - 5.0, 0.0).max(axis=1)
check("every relocated receptor is outside the footprint by >= the clearance",
      (dist_to_box[:3] >= 0.5 - 1e-9).all(), f"distances {np.round(dist_to_box[:3], 2).tolist()} m")
check("each receptor moves to the NEAREST wall (within one search step)",
      abs(disp[1] - (1.0 + 0.5)) <= 0.26 and abs(disp[2] - (0.5 + 0.5)) <= 0.26,
      f"displacements {np.round(disp[:3], 2).tolist()} m")
check("an open-air sample is left exactly where it is", disp[3] == 0.0 and (rxy[3] == xy[3]).all())
check("nothing unresolved", not unresolved.any())

# ----------------------------------------------------------------------------
print("=" * 70)
print("G4 facet orientation in 05a (a receptor inside a building)")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    ground.export(work / "ground.stl")
    box.subdivide().export(work / "buildings.stl")
    sphere([450, 450, 5], 3, 1).export(work / "vegetation.stl")
    mrt = work / "mrt"
    mrt.mkdir()
    route = np.array([[30.0, 40.0, 1.1], [40.0, 40.0, 1.1], [50.0, 40.0, 1.1]])
    np.save(mrt / "path_xyz.npy", route)          # middle sample inside the box
    np.save(mrt / "receptor_xyz.npy", route)      # deliberately NOT relocated here
    r = subprocess.run([sys.executable, str(HERE / "05a_thermal_facets_select.py"),
                        "--buildings-stl", str(work / "buildings.stl"),
                        "--vegetation-stl", str(work / "vegetation.stl"),
                        "--ground-stl", str(work / "ground.stl"),
                        "--mrt-dir", str(mrt), "--output-dir", str(work / "th"),
                        "--point-stride", "1"], capture_output=True, text=True)
    check("05a runs", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
    if r.returncode == 0:
        fz = np.load(work / "th" / "facets.npz")
        b = fz["mesh_id"] == 0
        outward = np.einsum("ij,ij->i", fz["normal"][b][:, :2],
                            fz["centroid"][b][:, :2] - 40.0)
        walls = np.abs(fz["normal"][b][:, 2]) < 0.5
        check("every building wall facet points away from the building",
              (outward[walls] > 0).all(), f"{int(walls.sum())} wall facets")
        flat = fz["normal"][b][~walls]
        top = fz["centroid"][b][~walls][:, 2] > 6.0
        check("roof facets point up, floor facets (seen from inside) point down",
              (flat[top][:, 2] > 0.99).all() and (flat[~top][:, 2] < -0.99).all(),
              f"{int(top.sum())} roof, {int((~top).sum())} floor facets")
        back = np.load(work / "th" / "back_face_view_weight.npy")
        check("the receptor inside the solid is flagged by its view of face backs",
              back[1] > 0.9 and back[0] == 0 and back[2] == 0, f"{np.round(back, 3).tolist()}")
        report = (work / "th" / "selection_report.txt").read_text()
        check("the report states what the old first-hit rule would have flipped",
              "first-hit rule would have flipped" in report)

# ----------------------------------------------------------------------------
print("=" * 70)
print("G5 shortwave enclosure")
rng = np.random.default_rng(0)
n = 200
D = rng.uniform(0, 900, n)
rho = rng.uniform(0.05, 0.6, n)
f = rng.uniform(0.0, 0.95, n)
area = rng.uniform(0.5, 20, n)
E, J = solve_route_visible_shortwave_enclosure(D, rho, f, area)
w = area * (1 - f)
check("J is the weighted mean of the reflected radiosity it produces",
      abs(J - np.sum(w * rho * E) / w.sum()) < 1e-9 * max(J, 1))
E0, J0 = solve_route_visible_shortwave_enclosure(np.full(5, 400.0), np.full(5, 0.3),
                                                 np.zeros(5), np.ones(5))
check("closed uniform enclosure: J = rho D / (1 - rho)", abs(J0 - 0.3 * 400 / 0.7) < 1e-9,
      f"J={J0:.3f}")
E1, J1 = solve_route_visible_shortwave_enclosure(D, np.zeros(n), f, area)
check("black facets reflect nothing", J1 == 0 and np.allclose(E1, D))
Et, Jt = solve_route_visible_shortwave_enclosure(np.vstack([D, 2 * D]), rho, f, area)
check("time-batched solve is linear and matches the single-step one",
      np.allclose(Et[0], E) and np.allclose(Jt[1], 2 * J))

# ----------------------------------------------------------------------------
print("=" * 70)
print("G6 ground reflection from the ground facets (full 05 -> 05a -> 05b -> 05 runs)")


def run_scene(work, buildings, label):
    ground_m = plane(-150, 150, -150, 150, n=60)
    ground_m.export(work / "ground.stl")
    buildings.export(work / "buildings.stl")
    sphere([2000, 2000, 5], 3, 1).export(work / "vegetation.stl")
    with open(work / "paths.pkl", "wb") as fh:
        pickle.dump({"polylines": [np.array([[-10.0, 0.0], [10.0, 0.0]])],
                     "highway_tags": ["footway"]}, fh)
    base = [sys.executable, str(HERE / "05_mrt_network_raytrace.py"),
            "--buildings-stl", str(work / "buildings.stl"),
            "--vegetation-stl", str(work / "vegetation.stl"),
            "--ground-stl", str(work / "ground.stl"),
            "--polylines-pkl", str(work / "paths.pkl"),
            "--ds-path", "2.0", "--dt-min", "60", "--date", "2025-06-21",
            "--latitude", "38.7", "--longitude", "-9.1", "--timezone", "Europe/Lisbon",
            "--sky-n-azimuth", "24", "--sky-n-elevation", "8",
            "--radiant-flux-config", str(work / "flux.json")]
    (work / "flux.json").write_text('{"enabled": true}')
    ok = subprocess.run(base + ["--output-dir", str(work / "mrt"), "--prep-only"],
                        capture_output=True, text=True).returncode == 0
    ok &= subprocess.run([sys.executable, str(HERE / "05a_thermal_facets_select.py"),
                          "--buildings-stl", str(work / "buildings.stl"),
                          "--vegetation-stl", str(work / "vegetation.stl"),
                          "--ground-stl", str(work / "ground.stl"),
                          "--mrt-dir", str(work / "mrt"), "--output-dir", str(work / "th"),
                          "--point-stride", "1"], capture_output=True, text=True).returncode == 0
    ok &= subprocess.run([sys.executable, str(HERE / "05b_facet_energy_balance.py"),
                          "--buildings-stl", str(work / "buildings.stl"),
                          "--vegetation-stl", str(work / "vegetation.stl"),
                          "--ground-stl", str(work / "ground.stl"),
                          "--facets-dir", str(work / "th"), "--mrt-dir", str(work / "mrt"),
                          "--output-dir", str(work / "th"), "--spinup-days", "1",
                          "--n-fsky-dirs", "48"], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--output-dir", str(work / "mrt"),
                               "--facet-thermal-dir", str(work / "th")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    ok &= r.returncode == 0
    check(f"{label}: 05 -> 05a -> 05b -> 05 completes", ok)
    return ok


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    far = trimesh.creation.box(extents=[2, 2, 2])
    far.apply_translation([140, 140, 1])
    if run_scene(work, far, "open ground"):
        arc = np.load(work / "mrt" / "radiant_flux_contributions.npz")
        times = __import__("pandas").read_csv(work / "mrt" / "times.csv")
        noon = int(np.argmax(times["elevation_deg"].values))
        ground_part = sum(arc[k][noon] for k in arc.files
                          if k.startswith("sw_reflected_") and "wall" not in k and "roof" not in k
                          and "vegetation" not in k and "total" not in k)
        alb = float(np.load(work / "th" / "facet_albedo.npy")[0])
        tau = np.load(work / "mrt" / "direct_transmission_matrix.npy")[noon]
        svf = np.load(work / "mrt" / "svf_planar.npy")
        el = np.deg2rad(times["elevation_deg"].values[noon])
        K = tau * times["DNI_Wm2"].values[noon] * np.sin(el) + svf * times["DHI_Wm2"].values[noon]
        old = 0.70 * 0.5 * alb * K
        rel = np.abs(ground_part - old) / old
        check("open ground: facet-based ground reflection = the old 0.5 * albedo * K",
              rel.max() < 0.05, f"max relative difference {rel.max():.3%}")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    walls = trimesh.util.concatenate([
        trimesh.creation.box(extents=[200, 4, 15]).apply_translation([0, 7, 7.5]),
        trimesh.creation.box(extents=[200, 4, 15]).apply_translation([0, -7, 7.5])])
    if run_scene(work, walls, "street canyon"):
        arc = np.load(work / "mrt" / "radiant_flux_contributions.npz")
        times = __import__("pandas").read_csv(work / "mrt" / "times.csv")
        noon = int(np.argmax(times["elevation_deg"].values))
        keys = [k for k in arc.files if k.startswith("sw_reflected_") and "wall" not in k
                and "roof" not in k and "vegetation" not in k and "total" not in k]
        ground_part = sum(arc[k][noon] for k in keys)
        alb = float(np.load(work / "th" / "facet_albedo.npy")[
            np.load(work / "th" / "facets.npz")["cls"] == 0][0])
        tau = np.load(work / "mrt" / "direct_transmission_matrix.npy")[noon]
        svf = np.load(work / "mrt" / "svf_planar.npy")
        el = np.deg2rad(times["elevation_deg"].values[noon])
        K = tau * times["DNI_Wm2"].values[noon] * np.sin(el) + svf * times["DHI_Wm2"].values[noon]
        old = 0.70 * 0.5 * alb * K
        th = np.load(work / "th" / "facets.npz")
        W = __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(work / "th" / "lw_view_matrix.npz")
        ground_view = np.asarray(W[:, th["cls"] == 0].sum(axis=1)).ravel()
        check("canyon: the receptor's ground view is below the open-ground 0.5",
              ground_view.max() < 0.45, f"ground view {ground_view.min():.3f}..{ground_view.max():.3f}")
        # Bookkeeping: the ground term is exactly sum over ground facets of
        # view weight x albedo x own irradiance (incl. the enclosure bounce),
        # plus the local estimate on directions that reach no surface.
        alb_all = np.load(work / "th" / "facet_albedo.npy")
        fsky = np.load(work / "th" / "f_sky_facet.npy")
        taud = np.load(work / "th" / "tau_dir_facet.npy")[noon]
        jsw = float(np.load(work / "th" / "sw_radiosity_environment_Wm2.npy")[noon])
        sunv = m05.sun_vector_enu(times["azimuth_deg"].values[noon], times["elevation_deg"].values[noon])
        cos_t = np.clip(th["normal"] @ sunv, 0, None)
        E = (taud * times["DNI_Wm2"].values[noon] * cos_t + fsky * times["DHI_Wm2"].values[noon]
             + (1 - fsky) * jsw)
        g = th["cls"] == 0
        pm = np.load(work / "th" / "point_map.npy")
        wdef = np.load(work / "th" / "lw_point_weights.npz")["w_default"][pm]
        expect = 0.70 * (np.asarray(W[:, g] @ (alb_all[g] * E[g])).ravel()[pm]
                         + wdef * alb * K)
        check("canyon: ground term = sum of ground facets' view x albedo x own irradiance",
              np.allclose(ground_part, expect, rtol=1e-4, atol=1e-3),
              f"max |diff| {np.abs(ground_part - expect).max():.2e} W/m2; new/old "
              f"{(ground_part / old).mean():.3f}")
        lw = np.load(work / "th" / "lw_point_weights.npz")
        total = (np.asarray(W.sum(axis=1)).ravel() + lw["w_sky"] + lw["w_veg"] + lw["w_default"])
        check("canyon: every direction is counted once (view weights partition unity)",
              np.allclose(total, 1.0, atol=1e-9))
        wall = arc["sw_reflected_building_wall_absorbed_Wm2"][noon]
        check("canyon: walls still reflect onto the receptor", (wall > 0).all())

# ----------------------------------------------------------------------------
print("=" * 70)
print("G7 segmented LoD1 builder")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    rng = np.random.default_rng(1)
    slope = 0.1                                   # terrain z = 0.1 x

    def ground_z(x):
        return slope * x
    # ground returns everywhere outside the buildings, 1 m spacing
    gx, gy = np.meshgrid(np.arange(-20, 101, 1.0), np.arange(-20, 61, 1.0))
    gx, gy = gx.ravel(), gy.ravel()
    in_a = (gx >= 0) & (gx <= 30) & (gy >= 0) & (gy <= 30)
    in_b = (gx > 30) & (gx <= 60) & (gy >= 0) & (gy <= 30)
    court = (gx >= 10) & (gx <= 20) & (gy >= 10) & (gy <= 20)
    keep = ~(in_a | in_b) | court
    gpts = np.column_stack([gx[keep], gy[keep], ground_z(gx[keep])])
    # roof returns, 0.3 m spacing: A (x 0-30) roof at 12 m above its lowest
    # terrain, B (x 30-60) at 24 m; a 10 x 10 m courtyard in A
    bx, by = np.meshgrid(np.arange(0.15, 60, 0.3), np.arange(0.15, 30, 0.3))
    bx, by = bx.ravel(), by.ravel()
    hole = (bx > 10) & (bx < 20) & (by > 10) & (by < 20)
    bx, by = bx[~hole], by[~hole]
    bz = np.where(bx < 30, 12.0, 24.0 + 3.0) + rng.normal(0, 0.1, len(bx))
    np.save(work / "b.npy", np.column_stack([bx, by, bz]))
    np.save(work / "g.npy", gpts)
    plane(-20, 100, -20, 60, n=60, zfun=lambda X, Y: slope * X).export(work / "ground.stl")
    r = subprocess.run([sys.executable, str(HERE / "03_buildings_to_stl.py"),
                        "--input", str(work / "b.npy"), "--ground-npy", str(work / "g.npy"),
                        "--ground-stl", str(work / "ground.stl"),
                        "--output", str(work / "bld.stl")], capture_output=True, text=True)
    check("03 builds", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
    if r.returncode == 0:
        bld = trimesh.load(work / "bld.stl", force="mesh", process=False)
        bi = RayMeshIntersector(bld)

        def roof_at(x, y):
            loc, ray, _ = bi.intersects_location(np.array([[x, y, 1e4]]),
                                                 np.array([[0, 0, -1.0]]), multiple_hits=False)
            return loc[0, 2] if len(ray) else None
        ra, rb, rc = roof_at(5, 5), roof_at(45, 15), roof_at(15, 15)
        check("two parts with their own roof heights (12 m and 27 m)",
              ra is not None and rb is not None and abs(ra - 12) < 0.3 and abs(rb - 27) < 0.3,
              f"A {ra}, B {rb}")
        check("the courtyard is kept open", rc is None or rc < ground_z(15) + 0.1)
        n = bld.face_normals
        c = bld.triangles_center
        walls = np.abs(n[:, 2]) < 0.5
        low = bld.triangles[walls][:, :, 2].min(axis=1)
        low_x = bld.triangles[walls][np.arange(walls.sum()),
                                     bld.triangles[walls][:, :, 2].argmin(axis=1), 0]
        step_wall = np.abs(low - ra) < 0.3 if ra is not None else np.zeros_like(low, bool)
        check("every outer wall reaches below the terrain beside it; step walls start "
              "on the lower roof", (step_wall | (low <= ground_z(low_x) + 1e-6)).all(),
              f"{int(step_wall.sum())} step-wall triangles")
        # rays from open points at 1.1 m above the slope must never reach a back face
        # or the underside of a floor before the terrain
        gmesh = trimesh.load(work / "ground.stl", force="mesh", process=False)
        gint = RayMeshIntersector(gmesh)
        p = rng.uniform([-15, -15], [95, 55], size=(20000, 2))
        open_ = ~(((p[:, 0] >= -0.5) & (p[:, 0] <= 60.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= 30.5))
                  & ~((p[:, 0] > 10.5) & (p[:, 0] < 19.5) & (p[:, 1] > 10.5) & (p[:, 1] < 19.5)))
        p = p[open_]
        o = np.column_stack([p, ground_z(p[:, 0]) + 1.1])
        dvec = rng.normal(size=o.shape)
        dvec[:, 2] = np.abs(dvec[:, 2]) * 0.3
        dvec /= np.linalg.norm(dvec, axis=1, keepdims=True)
        loc, ray, tri = bi.intersects_location(o, dvec, multiple_hits=False)
        gl, gr, _ = gint.intersects_location(o, dvec, multiple_hits=False)
        gd = np.full(len(o), np.inf)
        gd[gr] = np.linalg.norm(gl - o[gr], axis=1)
        first = np.linalg.norm(loc - o[ray], axis=1) < gd[ray]
        ray, tri = ray[first], tri[first]
        back = np.einsum("ij,ij->i", n[tri], dvec[ray]) > 0
        under = n[tri, 2] < -0.5
        check("no ray from open air reaches a back face or a floor before the terrain",
              back.sum() == 0 and under.sum() == 0,
              f"{len(ray)} building hits, {back.sum()} back, {under.sum()} floor")

print("\n" + "=" * 70)
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
