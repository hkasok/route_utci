"""
verify_thermal_pipeline.py -- verification suite for the facet surface-
temperature pipeline (thermal_common / 05a / 05b / patched 05).

Run:  python3 verify_thermal_pipeline.py

Unit tests (analytic, no geometry):
  T1  LW direction weights partition unity; an isothermal blackbody
      enclosure reproduces sigma*T^4 exactly (both body models)
  T2  (run inside the integration test, with the REAL view matrix)
      uniform facet temperatures == legacy surface temperature must give
      L_surround identical to the legacy scalar -> Tmrt unchanged
  T3  batched Thomas tridiagonal solver vs dense np.linalg.solve
  T4  conduction: steady state under constant forcing -> surface flux
      equals bottom-boundary flux (energy conservation through the slab)
  T5  radiative-convective equilibrium with adiabatic interior matches
      the EXACT algebraic fixed point (Brent root of the surface balance)
  T6  cosine-weighted hemisphere sampler: unit vectors, correct side of
      the facet, orthonormal frames

Integration test (synthetic scene: flat ground + one box building +
distant vegetation; runs the real scripts via subprocess):
  I1  05a weight partition + facet culling report
  I2  T2 exact backward compatibility through the real FacetLongwave
  I3  physics: sunlit ground >> building-shaded ground at midday;
      surfaces radiatively cool below air temperature at night;
      all temperatures within physical bounds; spin-up converged
  I4  patched 05 runs with --facet-thermal-dir; Tmrt fields are finite,
      differ where they should, and stay within a plausible envelope
"""

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from thermal_common import (SIGMA, make_hemisphere_directions_about_normal,
                            make_sphere_directions, solve_tridiagonal_batched)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
print("=" * 70)
print("T1: sphere direction weights / isothermal enclosure")
for body in ("cylinder", "sphere"):
    dirs, w = make_sphere_directions(24, 18, body=body)
    check(f"weights sum to 1 ({body})", abs(w.sum() - 1.0) < 1e-12,
          f"err={abs(w.sum() - 1.0):.1e}")
    check(f"directions are unit vectors ({body})",
          np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12))
    T = 310.0
    L = float(np.sum(w * SIGMA * T ** 4))
    check(f"isothermal enclosure -> sigma*T^4 ({body})",
          abs(L - SIGMA * T ** 4) < 1e-9, f"err={abs(L - SIGMA * T**4):.1e}")
# upward/downward split should be exactly 50/50 by symmetry
dirs, w = make_sphere_directions(24, 18, body="cylinder")
up = w[dirs[:, 2] > 0].sum()
check("hemisphere symmetry (up weight = 0.5)", abs(up - 0.5) < 1e-12)

# ======================================================================
print("\nT3: batched tridiagonal solver vs dense solve")
rng = np.random.default_rng(7)
m, n = 40, 9
a = rng.random((m, n)); c = rng.random((m, n))
b = 2.5 + a + c + rng.random((m, n))            # diagonally dominant
d = rng.standard_normal((m, n))
x = solve_tridiagonal_batched(a, b, c, d)
err = 0.0
for i in range(m):
    A = np.diag(b[i]) + np.diag(a[i, 1:], -1) + np.diag(c[i, :-1], 1)
    err = max(err, np.abs(np.linalg.solve(A, d[i]) - x[i]).max())
check("max |x_thomas - x_dense|", err < 1e-10, f"err={err:.1e}")

# ======================================================================
print("\nT4/T5: 1D energy balance solver (ClassSolver)")
eb = load_module(HERE / "05b_facet_energy_balance.py", "eb")

# T4 -- steady conduction: constant forcing, fixed-T bottom. At steady
# state the flux entering the surface must equal the flux leaving through
# the bottom boundary (exact energy conservation through the slab).
mat = dict(eb.DEFAULT_MATERIALS["ground"])
sol = eb.ClassSolver(mat, n_facets=3, dt=600.0, T_init=300.0,
                     T_bottom_ref=298.0, h_bottom=0.0)
Q_ext = np.array([420.0, 500.0, 380.0])   # W/m2, constant in time
h_conv, T_air = 11.4, 303.15
for _ in range(3000):                     # ~20 days -> deep steady state
    Ts = sol.step(Q_ext, h_conv, T_air)
F_surf = Q_ext - mat["emissivity"] * SIGMA * Ts ** 4 - h_conv * (Ts - T_air)
F_bot = sol.g_bot * (sol.T[:, -1] - 298.0)
check("steady state: surface flux == bottom flux",
      np.abs(F_surf - F_bot).max() < 1e-3,
      f"max err={np.abs(F_surf - F_bot).max():.2e} W/m2")
# internal profile must carry that same flux between every node pair
F_internal = sol.g[None, :] * (sol.T[:, :-1] - sol.T[:, 1:])
check("steady state: uniform flux through all layers",
      np.abs(F_internal - F_surf[:, None]).max() < 1e-3)

# T5 -- radiative-convective equilibrium, adiabatic interior: steady Ts
# must satisfy the EXACT nonlinear balance Q - eps*sig*T^4 - h(T-Ta) = 0.
from scipy.optimize import brentq
mat_w = dict(eb.DEFAULT_MATERIALS["wall"])
sol_w = eb.ClassSolver(mat_w, n_facets=1, dt=600.0, T_init=300.0,
                       T_bottom_ref=297.0, h_bottom=0.0)  # adiabatic
Q = 600.0
for _ in range(3000):
    Ts_w = sol_w.step(np.array([Q]), h_conv, T_air)
T_exact = brentq(lambda T: Q - mat_w["emissivity"] * SIGMA * T ** 4
                 - h_conv * (T - T_air), 200.0, 450.0)
check("adiabatic equilibrium matches exact Newton fixed point",
      abs(Ts_w[0] - T_exact) < 1e-3,
      f"solver={Ts_w[0]:.4f} K exact={T_exact:.4f} K")

# ======================================================================
print("\nT6: cosine-weighted hemisphere sampler")
normals = np.array([[0, 0, 1.0], [1, 0, 0.0], [0.6, -0.4, 0.3]])
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
hemi = make_hemisphere_directions_about_normal(normals, n_dirs=256)
check("unit vectors", np.allclose(np.linalg.norm(hemi, axis=2), 1.0,
                                  atol=1e-10))
dots = np.einsum("fd,fnd->fn", normals, hemi)
check("all directions on the outward side", (dots > 0).all())
# cosine-weighted samples: E[cos] = 2/3 -- Monte-Carlo check
check("cosine weighting (mean n.d ~ 2/3)",
      abs(dots.mean() - 2.0 / 3.0) < 0.02, f"mean={dots.mean():.4f}")

# ======================================================================
print("\n" + "=" * 70)
print("INTEGRATION: synthetic scene, full pipeline via the real scripts")
import trimesh

work = HERE / "verify_work"
work.mkdir(exist_ok=True)

# ---- scene: 80x80 m flat ground (2 m cells), 10x10x12 m box building,
# ---- vegetation sphere 500+ m away (outside the LW distance cull)
nx = 40
xs = np.linspace(0, 80, nx + 1)
X, Y = np.meshgrid(xs, xs)
V = np.column_stack([X.ravel(), Y.ravel(), np.zeros((nx + 1) ** 2)])
F = []
for j in range(nx):
    for i in range(nx):
        a0 = j * (nx + 1) + i
        F += [[a0, a0 + 1, a0 + nx + 2], [a0, a0 + nx + 2, a0 + nx + 1]]
ground = trimesh.Trimesh(V, np.array(F), process=False)

box = trimesh.creation.box(extents=[10, 10, 12])
box.apply_translation([40, 40, 6.0])
box = box.subdivide().subdivide()                     # 192 faces

veg = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
veg.apply_translation([450, 450, 5.0])

ground.export(work / "ground.stl")
box.export(work / "buildings.stl")
veg.export(work / "vegetation.stl")

# ---- east-west route passing the south side of the building
poly = np.array([[15.0, 30.0], [65.0, 30.0]])
with open(work / "paths.pkl", "wb") as f:
    pickle.dump({"polylines": [poly], "highway_tags": ["footway"]}, f)

common_05 = [
    sys.executable, str(HERE / "05_mrt_network_raytrace.py"),
    "--buildings-stl", str(work / "buildings.stl"),
    "--vegetation-stl", str(work / "vegetation.stl"),
    "--ground-stl", str(work / "ground.stl"),
    "--polylines-pkl", str(work / "paths.pkl"),
    "--ds-path", "2.0", "--dt-min", "30",
    "--sky-n-azimuth", "24", "--sky-n-elevation", "8",
]


def run(cmd, log):
    r = subprocess.run(cmd, capture_output=True, text=True)
    (work / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
    return r.returncode == 0


print("\n[1/4] legacy 05 run (produces path_xyz + times.csv)...")
mrt_dir = work / "mrt_legacy"
check("05 legacy run completes",
      run(common_05 + ["--output-dir", str(mrt_dir)], "log_05_legacy.txt"))

print("[2/4] 05a facet selection...")
fac_dir = work / "thermal"
ok = run([sys.executable, str(HERE / "05a_thermal_facets_select.py"),
          "--buildings-stl", str(work / "buildings.stl"),
          "--vegetation-stl", str(work / "vegetation.stl"),
          "--ground-stl", str(work / "ground.stl"),
          "--mrt-dir", str(mrt_dir), "--output-dir", str(fac_dir),
          "--point-stride", "1", "--n-lw-azimuth", "24",
          "--n-lw-elevation", "12", "--max-distance", "300"],
         "log_05a.txt")
check("05a run completes (includes internal weight-partition assert)", ok)
print((fac_dir / "selection_report.txt").read_text())

fz = np.load(fac_dir / "facets.npz")
n_facets = len(fz["cls"])
check("I1 culling: facet set is a strict subset of the scene",
      0 < n_facets < len(ground.faces) + len(box.faces),
      f"{n_facets} facets vs {len(ground.faces) + len(box.faces)} total faces")
check("I1 building walls were captured", (fz["cls"] == 1).sum() > 0)
check("I1 outward normals: ground facets point up",
      (fz["normal"][fz["cls"] == 0][:, 2] > 0.99).all())

# ---- 05b real energy balance ----
print("\n[3/4] 05b energy balance + 05 with facet temperatures...")
ok = run([sys.executable, str(HERE / "05b_facet_energy_balance.py"),
          "--buildings-stl", str(work / "buildings.stl"),
          "--vegetation-stl", str(work / "vegetation.stl"),
          "--ground-stl", str(work / "ground.stl"),
          "--facets-dir", str(fac_dir), "--mrt-dir", str(mrt_dir),
          "--output-dir", str(fac_dir), "--spinup-days", "2",
          "--n-fsky-dirs", "48"], "log_05b.txt")
check("05b run completes", ok)

# ---- I2: exact backward compatibility through the real FacetLongwave ----
print("\n[4/4] I2 backward-compatibility identity + 05 with facet temps...")
m05 = load_module(HERE / "05_mrt_network_raytrace.py", "m05")
args = SimpleNamespace(surface_temp_offset_day_c=8.0,
                       surrounding_emissivity=0.95,
                       vegetation_emissivity=0.98)
path_xyz = np.load(mrt_dir / "path_xyz.npy")
import pandas as pd
times_df = pd.read_csv(mrt_dir / "times.csv")
nt = len(times_df)
flw = m05.FacetLongwave(fac_dir, len(path_xyz), nt, args)
check("I2 precondition: no vegetation in LW view (scene built that way)",
      flw.w_veg.max() < 1e-12)
air_C, el = 31.0, 55.0
legacy_K = (air_C + 273.15) + 8.0 * np.sin(np.deg2rad(el))
flw.facet_T = np.full((nt, len(fz["cls"])), legacy_K, dtype=np.float32)
flw.facet_eps = np.full(len(fz["cls"]), 0.95)
L = flw.surround_at(0, air_C, el)
L_legacy = 0.95 * SIGMA * legacy_K ** 4
check("I2 uniform facet T == legacy scalar (exact identity)",
      np.abs(L - L_legacy).max() < 1e-5 * L_legacy,
      f"max rel err={np.abs(L - L_legacy).max() / L_legacy:.1e}")

facet_T = np.load(fac_dir / "facet_T_matrix_K.npy")
tau_dir = np.load(fac_dir / "tau_dir_facet.npy")
air = times_df["air_temp_C"].values + 273.15
elv = times_df["elevation_deg"].values
cls = fz["cls"]

# I3a: sunlit vs building-shaded ground contrast. NOTE: at this latitude
# and date the sun culminates near 87 deg elevation, so a 12 m building
# casts a <1 m shadow at solar noon -- ZERO shaded route-visible ground
# at noon is correct physics, not a bug. Test therefore uses the daytime
# step (elev > 25 deg) where the shaded population is largest (morning/
# afternoon, ~15 m shadows).
g = cls == 0
day_steps = np.where(elv > 25.0)[0]
shaded_counts = [(it, int((g & (tau_dir[it] < 0.05)).sum()))
                 for it in day_steps]
it_sh, n_sh = max(shaded_counts, key=lambda x: x[1])
sunlit = g & (tau_dir[it_sh] > 0.9)
shaded = g & (tau_dir[it_sh] < 0.05)
check("I3 sunlit and shaded ground facets both exist in daytime",
      sunlit.sum() > 5 and shaded.sum() > 5,
      f"step {it_sh} (elev {elv[it_sh]:.0f} deg): "
      f"sunlit={sunlit.sum()} shaded={shaded.sum()}")
dT = facet_T[it_sh, sunlit].mean() - facet_T[it_sh, shaded].mean()
check("I3 sunlit ground much hotter than building-shaded ground",
      dT > 5.0, f"dT={dT:.1f} K "
      f"(sunlit {facet_T[it_sh, sunlit].mean() - 273.15:.1f} C, "
      f"shaded {facet_T[it_sh, shaded].mean() - 273.15:.1f} C)")

# I3b: night radiative cooling below air temperature (clear sky)
it_night = int(np.argmin(elv))
open_ground = g & (np.load(fac_dir / "f_sky_facet.npy") > 0.9)
night_dT = facet_T[it_night, open_ground].mean() - air[it_night]
check("I3 night: open ground cools below air temperature",
      night_dT < 0.0, f"T_surf - T_air = {night_dT:.1f} K")

# I3c: bounds and spin-up
check("I3 all temperatures physically bounded",
      (facet_T > air.min() - 25).all() and (facet_T < air.max() + 45).all(),
      f"range {facet_T.min() - 273.15:.1f}..{facet_T.max() - 273.15:.1f} C "
      f"(air {air.min() - 273.15:.1f}..{air.max() - 273.15:.1f} C)")
spin = (fac_dir / "spinup_report.txt").read_text()
spin_val = float(spin.split("=")[1].split("K")[0])
check("I3 spin-up converged (< 1 K cycle-to-cycle)", spin_val < 1.0,
      f"delta={spin_val:.3f} K")

# I4: patched 05 with facet thermal longwave
mrt_dir2 = work / "mrt_facets"
check("05 run with --facet-thermal-dir completes",
      run(common_05 + ["--output-dir", str(mrt_dir2),
                       "--facet-thermal-dir", str(fac_dir)],
          "log_05_facets.txt"))
t_leg = np.load(mrt_dir / "tmrt_matrix_C.npy")
t_fac = np.load(mrt_dir2 / "tmrt_matrix_C.npy")
check("I4 shapes match & finite", t_leg.shape == t_fac.shape
      and np.isfinite(t_fac).all())
diff = t_fac - t_leg
check("I4 facet LW changes Tmrt (it should!)", np.abs(diff).max() > 0.3,
      f"max |dTmrt|={np.abs(diff).max():.2f} C")
check("I4 change is bounded/plausible (< 15 C)", np.abs(diff).max() < 15.0,
      f"max |dTmrt|={np.abs(diff).max():.2f} C, "
      f"mean {diff.mean():+.2f} C")

print("\n" + "=" * 70)
print(f"RESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
