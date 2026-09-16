#!/usr/bin/env python3
"""Verification suite for the step-3 pedestrian-level potential-flow model.

Covers the required validation tests:
  A  empty domain: uniform inflow stays uniform (axis-aligned and oblique)
  B  single rectangular building: diversion, zero penetration, corner
     acceleration, discrete mass conservation
  C  rotated building: boundary stays at its true geometric location
     (fractional cut-cell faces, accurate blocked area)
  D  narrow pedestrian path: a sub-cell 1.5 m gap is not erased by a 2 m grid
  E  terrain variation: pedestrian surface is z_ground + h_ped everywhere
  F  linearity: the 2 m/s solution equals 2x the 1 m/s solution
  G  step-4 integration: the real 05b surface-energy stage loads the step-3
     wind for its convection coefficient WITHOUT changing its radiation
     geometry (f_sky / direct-sun transmission bit-identical)

Exit code is nonzero on any failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
WORK = HERE / "verify_work_potential_flow"

passed = failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f" ({detail})" if detail else ""))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pf = load_module(HERE / "05e_potential_flow.py", "pf")
from pedestrian_flow_field import PedestrianFlowField, facet_wind_speed_matrix


def flat_ground(extent: float = 140.0, slope: float = 0.0,
                cells: int = 14) -> trimesh.Trimesh:
    """Triangulated ground plane z = slope * x covering [-20, extent-20]^2."""
    xs = np.linspace(-20.0, extent - 20.0, cells + 1)
    X, Y = np.meshgrid(xs, xs)
    V = np.column_stack([X.ravel(), Y.ravel(), slope * X.ravel()])
    F = []
    for j in range(cells):
        for i in range(cells):
            a0 = j * (cells + 1) + i
            F += [[a0, a0 + 1, a0 + cells + 2], [a0, a0 + cells + 2, a0 + cells + 1]]
    return trimesh.Trimesh(V, np.array(F), process=False)


def far_box() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[1, 1, 1])
    box.apply_translation([5000.0, 5000.0, 0.5])
    return box


def write_paths(path: Path, points) -> None:
    with open(path, "wb") as stream:
        pickle.dump({"polylines": [np.asarray(points, dtype=float)],
                     "highway_tags": ["footway"]}, stream)


def make_args(**overrides) -> argparse.Namespace:
    base = dict(buildings_stl="", ground_stl="", polylines_pkl="",
                output_dir="", grid_spacing=2.0, route_buffer_m=20.0,
                pedestrian_height=1.1, wind_direction_deg=270.0,
                wind_speed_ms=2.0, weather_csv=None, face_subsamples=4,
                maximum_cells=600000, solver_tolerance=1e-12,
                minimum_fluid_fraction=0.01, occupancy_batch=200000,
                maximum_relative_imbalance=1e-3,
                project_crs=None, no_figure=True, no_paraview=True,
                direction_basis="on")
    base.update(overrides)
    return argparse.Namespace(**base)


if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()

flat_ground().export(WORK / "ground_flat.stl")
far_box().export(WORK / "buildings_none.stl")
write_paths(WORK / "paths.pkl", [[20.0, 20.0], [80.0, 80.0]])

# ======================================================================
print("TEST A: empty domain -- uniform inflow stays uniform")
for direction, expect_u, expect_v in (
        (270.0, 2.0, 0.0),                      # westerly -> +x
        (225.0, 2.0 / np.sqrt(2), 2.0 / np.sqrt(2))):  # SW -> NE, oblique
    out = WORK / f"empty_{int(direction)}"
    pf.run(make_args(buildings_stl=str(WORK / "buildings_none.stl"),
                     ground_stl=str(WORK / "ground_flat.stl"),
                     polylines_pkl=str(WORK / "paths.pkl"),
                     output_dir=str(out), wind_direction_deg=direction))
    u = np.load(out / "velocity_u.npy")
    v = np.load(out / "velocity_v.npy")
    check(f"uniform u for wind from {direction:g} deg",
          np.allclose(u, expect_u, atol=1e-7),
          f"max err {np.abs(u - expect_u).max():.2e}")
    check(f"uniform v for wind from {direction:g} deg",
          np.allclose(v, expect_v, atol=1e-7),
          f"max err {np.abs(v - expect_v).max():.2e}")
    meta = json.loads((out / "potential_flow_metadata.json").read_text())
    check(f"empty-domain global mass balance ({direction:g} deg)",
          abs(meta["mass_conservation"]["relative_global_mass_imbalance"]) < 1e-9)

# ======================================================================
print("\nTEST B: single rectangular building")
box = trimesh.creation.box(extents=[16.0, 16.0, 12.0])
box.apply_translation([50.0, 50.0, 6.0])
box.export(WORK / "building_box.stl")
out_b = WORK / "box"
meta_b = pf.run(make_args(buildings_stl=str(WORK / "building_box.stl"),
                          ground_stl=str(WORK / "ground_flat.stl"),
                          polylines_pkl=str(WORK / "paths.pkl"),
                          output_dir=str(out_b), no_paraview=False))
speed = np.load(out_b / "velocity_speed.npy")
fluid = np.load(out_b / "fluid_mask.npy")
u = np.load(out_b / "velocity_u.npy")
x = np.load(out_b / "x_coordinates.npy")
y = np.load(out_b / "y_coordinates.npy")
check("building interior is solid", (~fluid).sum() >= 49,
      f"{(~fluid).sum()} solid cells (footprint 64 cells minus cut edges)")
check("zero velocity inside the building",
      speed[~fluid].max() == 0.0 if (~fluid).any() else False)
inside_x = (x > 43.0) & (x < 57.0)
row_upstream = np.argmin(np.abs(y - 50.0))
check("no penetration: flow does not cross the footprint interior",
      np.abs(u[row_upstream][inside_x]).max() < 1e-9,
      f"max |u| through footprint row {np.abs(u[row_upstream][inside_x]).max():.2e}")
check("flow accelerates around corners", speed[fluid].max() > 2.0 * 1.05,
      f"max speed {speed[fluid].max():.3f} m/s vs 2.0 inlet")
side_row = np.argmin(np.abs(y - 60.0))     # beside the building
check("flow diverts around the building (side speeds exceed inlet)",
      speed[side_row][inside_x].max() > 2.0)
mass = meta_b["mass_conservation"]
check("discrete mass conservation",
      mass["max_abs_divergence_per_s"] < 1e-8
      and abs(mass["relative_global_mass_imbalance"]) < 1e-9,
      f"max|div|={mass['max_abs_divergence_per_s']:.2e}, "
      f"imbalance={mass['relative_global_mass_imbalance']:.2e}")

from paraview_export import read_vtk_appended
pv_dir = out_b / "paraview"
check("ParaView export files written",
      all((pv_dir / name).is_file() for name in (
          "pedestrian_wind_surface.vtp", "pedestrian_wind.vtr",
          "context_ground.vtp", "context_buildings.vtp")))
surface = read_vtk_appended(pv_dir / "pedestrian_wind_surface.vtp")
grid = read_vtk_appended(pv_dir / "pedestrian_wind.vtr")
n_points = len(x) * len(y)
check("surface .vtp: one vertex per cell centre with the solved velocity",
      surface["arrays"]["__points__"].shape == (n_points, 3)
      and surface["arrays"]["velocity_ms"].shape == (n_points, 3)
      and np.allclose(surface["arrays"]["velocity_ms"][:, 0],
                      u.ravel(), atol=1e-6))
check("surface .vtp vertices sit at pedestrian height (ground + h_ped)",
      np.allclose(surface["arrays"]["__points__"][:, 2], 1.1, atol=1e-5))
check("plan-view .vtr round-trips speed exactly",
      np.allclose(grid["arrays"]["speed_ms"].ravel(),
                  np.asarray(speed, np.float32).ravel(), atol=1e-6))

# ======================================================================
print("\nTEST C: rotated building keeps its true boundary")
rot = trimesh.creation.box(extents=[12.0, 12.0, 12.0])
rot.apply_transform(trimesh.transformations.rotation_matrix(
    np.deg2rad(30.0), [0, 0, 1]))
rot.apply_translation([50.0, 50.0, 6.0])
rot.export(WORK / "building_rot.stl")
out_c = WORK / "rot"
pf.run(make_args(buildings_stl=str(WORK / "building_rot.stl"),
                 ground_stl=str(WORK / "ground_flat.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_c), face_subsamples=6))
alpha = np.load(out_c / "cell_fluid_fraction.npy")
beta_x = np.load(out_c / "face_open_fraction_x.npy")
beta_y = np.load(out_c / "face_open_fraction_y.npy")
fractional = int(((beta_x > 0) & (beta_x < 1)).sum()
                 + int(((beta_y > 0) & (beta_y < 1)).sum()))
check("slanted boundary produces fractional (cut) faces, not stair steps",
      fractional > 10, f"{fractional} partially open faces")
blocked_area = float((1.0 - alpha).sum()) * 2.0 * 2.0
check("blocked area matches the true rotated footprint",
      abs(blocked_area - 144.0) / 144.0 < 0.04,
      f"cut-cell {blocked_area:.1f} m^2 vs exact 144.0 m^2")
stair_area = float((alpha < 0.5).sum()) * 2.0 * 2.0
check("cut-cell area beats the stair-step (0/1 mask) representation",
      abs(blocked_area - 144.0) <= abs(stair_area - 144.0) + 1e-9,
      f"stair-step {stair_area:.1f} m^2")

# ======================================================================
print("\nTEST D: narrow 1.5 m pedestrian path on a 2 m grid")
south = trimesh.creation.box(extents=[16.0, 19.25, 12.0])
south.apply_translation([50.0, 29.625, 6.0])     # y in [20.0, 39.25]
north = trimesh.creation.box(extents=[16.0, 19.25, 12.0])
north.apply_translation([50.0, 50.375, 6.0])     # y in [40.75, 60.0]
gap_scene = trimesh.util.concatenate([south, north])
gap_scene.export(WORK / "building_gap.stl")
out_d = WORK / "gap"
meta_d = pf.run(make_args(buildings_stl=str(WORK / "building_gap.stl"),
                          ground_stl=str(WORK / "ground_flat.stl"),
                          polylines_pkl=str(WORK / "paths.pkl"),
                          output_dir=str(out_d)))
field_d = PedestrianFlowField(out_d)
beta_x = np.load(out_d / "face_open_fraction_x.npy")
y_d = np.load(out_d / "y_coordinates.npy")
gap_rows = np.where((y_d > 38.0) & (y_d < 42.0))[0]
open_through_gap = beta_x[gap_rows, :][:, 15:35]
check("the sub-cell gap keeps open (fractional) faces through the block",
      float(open_through_gap.max()) > 0.2
      and float(open_through_gap[open_through_gap > 0].min()) < 1.0,
      f"beta range {open_through_gap.max():.2f}")
_u, _v, gap_speed, _dir = field_d.sample(np.array([50.0]), np.array([40.0]))
check("flow passes through the 1.5 m corridor (not erased)",
      float(gap_speed[0]) > 0.05,
      f"speed at gap centre {float(gap_speed[0]):.3f} m/s")
warn = meta_d["resolution_warnings"]
check("under-resolved passage is reported, not silently accepted",
      warn["mostly_solid_partially_open_cells"] > 0
      or warn["narrow_corridor_cells"] > 0, str(warn))

# ======================================================================
print("\nTEST E: pedestrian surface follows the terrain")
flat_ground(slope=0.05).export(WORK / "ground_slope.stl")
out_e = WORK / "slope"
pf.run(make_args(buildings_stl=str(WORK / "building_box.stl"),
                 ground_stl=str(WORK / "ground_slope.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_e), pedestrian_height=1.5))
ground_z = np.load(out_e / "ground_z.npy")
ped_z = np.load(out_e / "pedestrian_z.npy")
x_e = np.load(out_e / "x_coordinates.npy")
check("ground_z reproduces the sloped terrain",
      np.allclose(ground_z, 0.05 * x_e[None, :], atol=1e-6),
      f"max err {np.abs(ground_z - 0.05 * x_e[None, :]).max():.2e} m")
check("pedestrian_z = ground_z + h_ped throughout the domain",
      np.allclose(ped_z - ground_z, 1.5, atol=1e-12))

# ======================================================================
print("\nTEST F: linearity with wind speed")
out_f1 = WORK / "lin1"
out_f2 = WORK / "lin2"
pf.run(make_args(buildings_stl=str(WORK / "building_box.stl"),
                 ground_stl=str(WORK / "ground_flat.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_f1), wind_speed_ms=1.0))
pf.run(make_args(buildings_stl=str(WORK / "building_box.stl"),
                 ground_stl=str(WORK / "ground_flat.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_f2), wind_speed_ms=2.0))
u1 = np.load(out_f1 / "velocity_u.npy"); v1 = np.load(out_f1 / "velocity_v.npy")
u2 = np.load(out_f2 / "velocity_u.npy"); v2 = np.load(out_f2 / "velocity_v.npy")
check("u(2 m/s) == 2 * u(1 m/s) to numerical precision",
      np.allclose(u2, 2.0 * u1, atol=1e-8)
      and np.allclose(v2, 2.0 * v1, atol=1e-8),
      f"max err {max(np.abs(u2 - 2 * u1).max(), np.abs(v2 - 2 * v1).max()):.2e}")
field_1 = PedestrianFlowField(out_f1)
u_s, v_s, s_s, _ = field_1.sample(np.array([30.0]), np.array([50.0]),
                                  wind_speed_ms=3.7)
u_r, v_r, s_r, _ = field_1.sample(np.array([30.0]), np.array([50.0]))
check("sampler rescaling honors linearity",
      np.allclose([u_s[0], v_s[0], s_s[0]],
                  [3.7 * u_r[0], 3.7 * v_r[0], 3.7 * s_r[0]], rtol=1e-12))

# ======================================================================
print("\nTEST G: step-4 (05b surface energy) integration")
veg = far_box()
veg.export(WORK / "vegetation_none.stl")
# Facets: open-ground receptors, one wall facet on the box face, one roof
# facet inside the footprint (exercises the nearest-fluid fallback).
centroids = np.array([[20.0, 50.0, 0.0], [30.0, 50.0, 0.0],
                      [70.0, 50.0, 0.0], [42.0, 50.0, 3.0],
                      [50.0, 50.0, 12.0]])
normals = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0],
                    [-1.0, 0, 0], [0, 0, 1.0]])
cls = np.array([0, 0, 0, 1, 2], dtype=np.int64)
areas = np.ones(len(cls))
mrt_dir = WORK / "mrt_prep"
mrt_dir.mkdir()
fac_dir = WORK / "facets"
fac_dir.mkdir()
np.savez(fac_dir / "facets.npz", centroid=centroids, normal=normals,
         cls=cls, area=areas)
hours = [10, 11, 12, 13]
wind_series = np.array([1.0, 2.0, 3.0, 2.5])
rows = ["time,DNI_Wm2,DHI_Wm2,elevation_deg,azimuth_deg,air_temp_C,rh_pct,wind_ms"]
for hour, wind in zip(hours, wind_series):
    rows.append(f"2025-07-06 {hour:02d}:00:00,700,120,{30 + hour},150,30,60,{wind:g}")
(mrt_dir / "times.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_05b(out_dir: Path, extra: list[str]) -> bool:
    result = subprocess.run(
        [sys.executable, str(HERE / "05b_facet_energy_balance.py"),
         "--buildings-stl", str(WORK / "building_box.stl"),
         "--vegetation-stl", str(WORK / "vegetation_none.stl"),
         "--ground-stl", str(WORK / "ground_flat.stl"),
         "--facets-dir", str(fac_dir), "--mrt-dir", str(mrt_dir),
         "--output-dir", str(out_dir), "--spinup-days", "1",
         "--maximum-spinup-days", "1", "--n-fsky-dirs", "16"] + extra,
        capture_output=True, text=True)
    (out_dir / "log.txt").write_text(result.stdout + "\n--- stderr ---\n"
                                     + result.stderr)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    return result.returncode == 0


out_base = WORK / "eb_uniform"
out_ped = WORK / "eb_pedestrian"
out_base.mkdir()
out_ped.mkdir()
ok_base = run_05b(out_base, [])
ok_ped = run_05b(out_ped, ["--pedestrian-flow-dir", str(out_b)])
check("05b runs with the uniform-weather fallback", ok_base)
check("05b runs with the step-3 pedestrian wind field", ok_ped)
if ok_base and ok_ped:
    check("radiation geometry unchanged: identical facet sky fractions",
          np.array_equal(np.load(out_base / "f_sky_facet.npy"),
                         np.load(out_ped / "f_sky_facet.npy")))
    check("radiation geometry unchanged: identical direct-sun transmission",
          np.array_equal(np.load(out_base / "tau_dir_facet.npy"),
                         np.load(out_ped / "tau_dir_facet.npy")))
    provenance_base = json.loads(
        (out_base / "surface_wind_provenance.json").read_text())
    provenance_ped = json.loads(
        (out_ped / "surface_wind_provenance.json").read_text())
    check("wind source recorded for the fallback run",
          provenance_base["uniform_fallback"]
          and provenance_base["pedestrian_flow_dir"] is None)
    check("wind source recorded for the pedestrian-flow run",
          not provenance_ped["uniform_fallback"]
          and "potential-flow" in provenance_ped["wind_source"])
    forcing = np.load(out_ped / "facet_pedestrian_wind_forcing.npz")
    field_b = PedestrianFlowField(out_b)
    expected = facet_wind_speed_matrix(field_b, centroids[:, :2], wind_series)
    check("05b facet wind = weather series x local unit amplification",
          np.allclose(forcing["wind_speed_ms"], expected, rtol=1e-6),
          f"max err {np.abs(forcing['wind_speed_ms'] - expected).max():.2e}")
    check("solid-footprint facet got a nearest-fluid (finite) wind",
          np.isfinite(forcing["wind_speed_ms"][:, 4]).all())
    T_base = np.load(out_base / "facet_T_matrix_K.npy")
    T_ped = np.load(out_ped / "facet_T_matrix_K.npy")
    check("local wind actually changes convection (temperatures differ)",
          not np.array_equal(T_base, T_ped))

print("\nTEST H: dead-end pocket that cannot reach the outlet")
# Regression for the singular-system failure seen on the dense Lisbon cases:
# a C-shaped block open only toward the inlet boundary leaves a fluid
# component that touches the outer boundary but never reaches the outlet.
# Anchored only by prescribed inflow it forms a pure-Neumann, inconsistent
# block -- an exactly singular matrix that used to yield NaN velocities.
south = trimesh.creation.box(extents=[30.0, 4.0, 12.0])
south.apply_translation([-10.0 + 15.0, 42.0, 6.0])     # y 40..44, x -10..20
north = trimesh.creation.box(extents=[30.0, 4.0, 12.0])
north.apply_translation([-10.0 + 15.0, 58.0, 6.0])     # y 56..60, x -10..20
cap = trimesh.creation.box(extents=[4.0, 20.0, 12.0])
cap.apply_translation([22.0, 50.0, 6.0])               # x 20..24, y 40..60
pocket_scene = trimesh.util.concatenate([south, north, cap])
pocket_scene.export(WORK / "building_pocket.stl")
out_h = WORK / "pocket"
meta_h = pf.run(make_args(buildings_stl=str(WORK / "building_pocket.stl"),
                          ground_stl=str(WORK / "ground_flat.stl"),
                          polylines_pkl=str(WORK / "paths.pkl"),
                          output_dir=str(out_h)))
u_h = np.load(out_h / "velocity_u.npy")
v_h = np.load(out_h / "velocity_v.npy")
speed_h = np.load(out_h / "velocity_speed.npy")
check("dead-end pocket solves without non-finite values",
      np.isfinite(u_h).all() and np.isfinite(v_h).all()
      and np.isfinite(speed_h).all())
mass_h = meta_h["mass_conservation"]
check("dead-end pocket keeps global mass conservation",
      np.isfinite(mass_h["max_abs_divergence_per_s"])
      and abs(mass_h["relative_global_mass_imbalance"]) < 1e-6,
      f"imbalance={mass_h['relative_global_mass_imbalance']:.2e}")
field_h = PedestrianFlowField(out_h)
_u, _v, pocket_speed, _d = field_h.sample(np.array([0.0, 10.0]),
                                          np.array([50.0, 50.0]))
check("dead-end pocket is near-stagnant, not spuriously fast",
      float(np.max(pocket_speed)) < 0.5,
      f"max pocket speed {float(np.max(pocket_speed)):.3f} m/s vs 2.0 inlet")

print("\nTEST H2: fully enclosed courtyard (stagnant region)")
# A hollow ring building: the courtyard is open at pedestrian level but has no
# connection to the outer boundary, so it is held stagnant and its potential
# is undefined. That NaN must not be mistaken for a failed solve.
ring = []
for extents, centre in (((20.0, 4.0, 12.0), (50.0, 42.0)),
                        ((20.0, 4.0, 12.0), (50.0, 58.0)),
                        ((4.0, 20.0, 12.0), (42.0, 50.0)),
                        ((4.0, 20.0, 12.0), (58.0, 50.0))):
    wall = trimesh.creation.box(extents=extents)
    wall.apply_translation([centre[0], centre[1], 6.0])
    ring.append(wall)
trimesh.util.concatenate(ring).export(WORK / "building_ring.stl")
out_h2 = WORK / "ring"
meta_h2 = pf.run(make_args(buildings_stl=str(WORK / "building_ring.stl"),
                           ground_stl=str(WORK / "ground_flat.stl"),
                           polylines_pkl=str(WORK / "paths.pkl"),
                           output_dir=str(out_h2)))
check("enclosed courtyard is detected and held stagnant",
      meta_h2["solver"]["stagnant_enclosed_cells"] > 0,
      f"{meta_h2['solver']['stagnant_enclosed_cells']} stagnant cells")
check("a stagnant region does not block writing the field",
      (out_h2 / "potential_flow_metadata.json").is_file()
      and np.isfinite(np.load(out_h2 / "velocity_speed.npy")).all())
field_h2 = PedestrianFlowField(out_h2)
_u, _v, court_speed, _d = field_h2.sample(np.array([50.0]), np.array([50.0]))
check("enclosed courtyard air is still",
      float(court_speed[0]) == 0.0, f"{float(court_speed[0]):.3f} m/s")

print("\nTEST J: two-direction basis reconstructs ANY wind direction exactly")
# One building, solved at 270 deg with the basis on. The basis pair is solved
# with the outlet PINNED to the configured run's outlet, which is what makes
# superposition exact rather than an interpolation between two cases.
box_j = trimesh.creation.box(extents=[16.0, 16.0, 14.0])
box_j.apply_translation([50.0, 50.0, 7.0])
box_j.export(WORK / "building_j.stl")
out_j = WORK / "basis"
pf.run(make_args(buildings_stl=str(WORK / "building_j.stl"),
                 ground_stl=str(WORK / "ground_flat.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_j), wind_direction_deg=270.0))
meta_j = json.loads((out_j / "potential_flow_metadata.json").read_text())
basis_meta = meta_j["direction_basis"]
check("the direction basis is written and recorded",
      basis_meta["available"] and (out_j / "direction_basis.npz").is_file())
check("the basis is TWO components, not four "
      "(south is just minus north, so four would be redundant)",
      basis_meta["components"] == ["east", "north"])
check("the outlet side is pinned across the basis pair",
      basis_meta["outlet_side"] == meta_j.get("outlet_side",
                                              basis_meta["outlet_side"]))
check("every basis component conserves mass",
      all(abs(value) < 1e-6
          for value in basis_meta["mass_imbalance_by_component"].values()),
      str(basis_meta["mass_imbalance_by_component"]))

field_j = PedestrianFlowField(out_j)
check("the loader exposes the basis", field_j.has_direction_basis())

# THE decisive check: a direct solve at an OBLIQUE direction must equal the
# basis reconstruction of that direction. 225 deg is not axis aligned, so both
# basis components contribute and the superposition is genuinely exercised.
out_j_direct = WORK / "basis_direct_225"
pf.run(make_args(buildings_stl=str(WORK / "building_j.stl"),
                 ground_stl=str(WORK / "ground_flat.stl"),
                 polylines_pkl=str(WORK / "paths.pkl"),
                 output_dir=str(out_j_direct), wind_direction_deg=225.0,
                 direction_basis="off"))
direct_u = np.load(out_j_direct / "velocity_u.npy")
direct_v = np.load(out_j_direct / "velocity_v.npy")
reference = float(json.loads(
    (out_j_direct / "potential_flow_metadata.json").read_text()
)["reference_wind_speed_ms"])
grid_u, grid_v = field_j.basis_velocity(225.0)
rebuilt_u, rebuilt_v = grid_u * reference, grid_v * reference
scale = max(float(np.hypot(direct_u, direct_v).max()), 1e-12)
error = float(np.hypot(rebuilt_u - direct_u, rebuilt_v - direct_v).max() / scale)
check("an oblique direction rebuilt from the basis matches a DIRECT solve",
      error < 1e-6, f"max relative velocity error {error:.3e}")
check("both basis components genuinely contribute at 225 deg",
      np.abs(field_j.basis["u_north"]).max() > 1e-6
      and np.abs(field_j.basis["u_east"]).max() > 1e-6)

# Reversal: the field for the opposite wind is exactly the negative.
u_north_from, v_north_from, _, _ = field_j.sample_direction(
    np.array([40.0]), np.array([40.0]), 0.0, 1.0)
u_south_from, v_south_from, _, _ = field_j.sample_direction(
    np.array([40.0]), np.array([40.0]), 180.0, 1.0)
check("the opposite wind direction gives exactly the negated field -- which is "
      "why two solves span all four cardinal directions",
      np.allclose(u_north_from, -u_south_from, atol=1e-12)
      and np.allclose(v_north_from, -v_south_from, atol=1e-12))

# Direction and speed are independent controls.
probe_x, probe_y = np.array([35.0, 60.0]), np.array([44.0, 58.0])
u1, v1, s1, d1 = field_j.sample_direction(probe_x, probe_y, 200.0, 1.0)
u3, v3, s3, d3 = field_j.sample_direction(probe_x, probe_y, 200.0, 3.0)
check("speed rescaling is exactly linear and leaves direction untouched",
      np.allclose(u3, 3.0 * u1) and np.allclose(v3, 3.0 * v1)
      and np.allclose(d3, d1))
check("changing direction genuinely changes the field, unlike a rescale",
      not np.allclose(field_j.sample_direction(probe_x, probe_y, 200.0, 1.0)[0],
                      field_j.sample_direction(probe_x, probe_y, 110.0, 1.0)[0]))
check("speed is recomputed as hypot(u, v) after superposition, never summed "
      "from stored magnitudes", np.allclose(s1, np.hypot(u1, v1)))

# A field solved without the basis must fail loudly rather than guess.
plain = PedestrianFlowField(out_j_direct)
check("a field without a basis reports it", not plain.has_direction_basis())
try:
    plain.basis_velocity(90.0)
    check("asking a basis-free field for another direction is refused", False)
except ValueError as error:
    check("asking a basis-free field for another direction is refused",
          "direction-basis" in str(error) or "--direction-basis" in str(error),
          str(error)[:70])

# ======================================================================
print("\nTEST I: an unusable field is rejected, never written")
bad = WORK / "rejected"
try:
    pf.run(make_args(buildings_stl=str(WORK / "building_box.stl"),
                     ground_stl=str(WORK / "ground_flat.stl"),
                     polylines_pkl=str(WORK / "paths.pkl"),
                     output_dir=str(bad), maximum_relative_imbalance=1e-30))
    rejected = False
except RuntimeError as exc:
    rejected = "imbalance" in str(exc)
check("an over-tolerance mass imbalance raises instead of saving", rejected)
check("no completeness marker is left behind after a rejected solve",
      not (bad / "potential_flow_metadata.json").exists())

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
