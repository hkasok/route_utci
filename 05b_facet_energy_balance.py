"""
05b_facet_energy_balance.py -- surface temperatures for the route-visible
facets selected by 05a, via a per-facet 1D multilayer energy balance.

This REPLACES volumetric CHT for the purpose of route MRT: each facet gets

    absorbed shortwave (full-geometry sun shading + vegetation attenuation)
  + absorbed longwave  (sky + surrounding surfaces)
  - emitted longwave   (eps * sigma * Ts^4)
  - convection         (h_c * (Ts - Tair),  h_c = 5.7 + 3.8*U  [Jurges])
  = conduction into a 1D multilayer substrate (implicit Euler, exact
    tridiagonal solve, vectorized across all facets of a class)

with the day repeated --spinup-days times so the substrate reaches a
quasi-periodic diurnal state before the saved cycle.

Design decisions that matter for correctness
--------------------------------------------
* SHADOWS USE FULL GEOMETRY: the per-timestep sun rays from every facet
  are tested against the COMPLETE building/ground/vegetation meshes --
  the facet CULLING (05a) only limits which facets get a temperature,
  never which geometry can cast shade.
* Forcing comes from times.csv written by 05 (same DNI/DHI/GHI/air temp/
  solar position), so 05 and 05b can never silently disagree on weather.
* The surface radiation term is linearized about the previous step's
  surface temperature (standard Newton linearization); with 10-minute
  steps the linearization error is negligible (checked in tests: the
  no-conduction steady state matches an exact Newton fixed point).
* Facet-to-facet longwave coupling uses the previous step's area-weighted
  mean surface temperature as the environment temperature (lagged, cheap,
  unconditionally stable; a 1-2 K effect on facet T, far less on Tmrt).

Outputs (in --output-dir)
-------------------------
  facet_T_matrix_K.npy   (n_times x n_facets) surface temperature, K
  f_sky_facet.npy        per-facet sky fraction (cosine-weighted)
  facet_eps.npy          per-facet emissivity used
  tau_dir_facet.npy      (n_times x n_facets) direct-sun transmission
  facet_summary_by_time.csv, spinup_report.txt

Run:
    python3 05b_facet_energy_balance.py \
        --buildings-stl ... --vegetation-stl ... --ground-stl ... \
        --facets-dir thermal_facets_output/ \
        --mrt-dir mrt_network_output/ \
        --output-dir thermal_facets_output/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from thermal_common import (CLASS_GROUND, CLASS_NAMES, CLASS_ROOF, CLASS_WALL,
                            SIGMA, get_intersector,
                            make_hemisphere_directions_about_normal,
                            sky_longwave_down, solve_tridiagonal_batched,
                            sun_vector_enu,
                            vegetation_transmission_from_intersections)

# Per-class material / model defaults. Override any entry via --material-json.
DEFAULT_MATERIALS = {
    "ground": {"k": 1.00, "C": 2.0e6, "depth": 0.50, "n_layers": 8,
               "albedo": 0.12, "emissivity": 0.95,
               "bottom_bc": "fixed"},     # fixed T_deep at depth
    "wall":   {"k": 1.40, "C": 1.8e6, "depth": 0.25, "n_layers": 6,
               "albedo": 0.30, "emissivity": 0.90,
               "bottom_bc": "interior"},  # convective to indoor air
    "roof":   {"k": 1.00, "C": 1.6e6, "depth": 0.25, "n_layers": 6,
               "albedo": 0.15, "emissivity": 0.92,
               "bottom_bc": "interior"},
}
CLASS_TO_NAME = {CLASS_GROUND: "ground", CLASS_WALL: "wall", CLASS_ROOF: "roof"}


def parse_args():
    p = argparse.ArgumentParser(description="1D facet energy balance for "
                                            "route-visible surfaces")
    p.add_argument("--buildings-stl", required=True)
    p.add_argument("--vegetation-stl", required=True)
    p.add_argument("--ground-stl", required=True)
    p.add_argument("--facets-dir", required=True, help="Output dir of 05a")
    p.add_argument("--mrt-dir", required=True,
                   help="Output dir of 05 (needs times.csv)")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--wind-speed", type=float, default=1.5,
                   help="Near-surface wind speed for convection, m/s")
    p.add_argument("--interior-temp-c", type=float, default=24.0,
                   help="Building interior air temperature, deg C")
    p.add_argument("--h-interior", type=float, default=8.0,
                   help="Interior convective coefficient W/m2K "
                        "(0 = adiabatic interior; used by tests)")
    p.add_argument("--deep-soil-temp-c", type=float, default=None,
                   help="Deep soil temperature; default = daily mean air T")
    p.add_argument("--env-emissivity", type=float, default=0.95,
                   help="Emissivity of surrounding surfaces as seen BY a "
                        "facet (for its incoming LW)")
    p.add_argument("--environment-albedo", type=float, default=0.20,
                   help="Albedo of surroundings reflecting SW onto facets")
    p.add_argument("--cloud-cover-fraction", type=float, default=0.0,
                   help="MUST match the value used in 05 (times.csv stores "
                        "cloud-adjusted DNI/DHI but L_sky needs the fraction)")
    p.add_argument("--k-lad-direct", type=float, default=0.45,
                   help="Vegetation extinction for the direct beam "
                        "(match 05)")
    p.add_argument("--k-lad-diffuse", type=float, default=0.30,
                   help="Vegetation extinction for diffuse sky (match 05)")
    p.add_argument("--spinup-days", type=int, default=2,
                   help="Diurnal cycles run before the saved cycle")
    p.add_argument("--n-fsky-dirs", type=int, default=64,
                   help="Hemisphere rays per facet for its sky fraction")
    p.add_argument("--facet-batch-size", type=int, default=5000)
    p.add_argument("--material-json", default=None,
                   help="JSON file overriding entries of the built-in "
                        "material table")
    p.add_argument("--surface-offset", type=float, default=2e-3,
                   help="Ray-origin offset along the facet normal, m")
    return p.parse_args()


def build_layers(depth, n_layers, grading=1.6):
    """Layer thicknesses graded thin->thick from the surface (sum=depth).
    A thin first layer makes 'node-0 temperature' a good surface proxy."""
    r = grading ** np.arange(n_layers)
    return depth * r / r.sum()


def facet_sky_fraction(centroids, normals, solid_inters, veg_inter,
                       k_lad_diffuse, n_dirs, offset, batch):
    """Cosine-weighted sky fraction of each facet: fraction of the
    Lambertian-weighted hemisphere that reaches sky, with vegetation
    attenuating (Beer-Lambert) and buildings/ground fully blocking."""
    nf = len(centroids)
    f_sky = np.zeros(nf)
    hemi = make_hemisphere_directions_about_normal(normals, n_dirs=n_dirs)
    for s in range(0, nf, batch):
        e = min(s + batch, nf)
        m = e - s
        origins = np.repeat(centroids[s:e] + offset * normals[s:e], n_dirs,
                            axis=0)
        dirs = hemi[s:e].reshape(m * n_dirs, 3)
        blocked = np.zeros(m * n_dirs, dtype=bool)
        for inter in solid_inters:
            blocked |= inter.intersects_any(origins, dirs)
        tau_veg, _ = vegetation_transmission_from_intersections(
            veg_inter, origins, dirs, k_lad=k_lad_diffuse)
        vis = np.where(blocked, 0.0, tau_veg)
        f_sky[s:e] = vis.reshape(m, n_dirs).mean(axis=1)
    return f_sky


def facet_sun_transmission(centroids, normals, sun_vec, solid_inters,
                           veg_inter, k_lad_direct, offset, batch):
    """Direct-beam transmission per facet (0 where the facet faces away
    from the sun or is shaded by ANY geometry; Beer-Lambert through
    vegetation otherwise)."""
    nf = len(centroids)
    tau = np.zeros(nf)
    facing = (normals @ sun_vec) > 1e-6
    idx = np.where(facing)[0]
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        origins = centroids[ii] + offset * normals[ii]
        dirs = np.tile(sun_vec, (len(ii), 1))
        blocked = np.zeros(len(ii), dtype=bool)
        for inter in solid_inters:
            blocked |= inter.intersects_any(origins, dirs)
        t_veg, _ = vegetation_transmission_from_intersections(
            veg_inter, origins, dirs, k_lad=k_lad_direct)
        tau[ii] = np.where(blocked, 0.0, t_veg)
    return tau


class ClassSolver:
    """Implicit-Euler 1D conduction for all facets of one class.

    State T has shape (n_facets_in_class, n_layers); node 0 is the surface
    layer (made thin by the graded grid). Each step solves an exact
    tridiagonal system per facet (batched Thomas algorithm)."""

    def __init__(self, mat, n_facets, dt, T_init, T_bottom_ref, h_bottom):
        self.k, self.C = mat["k"], mat["C"]
        self.dz = build_layers(mat["depth"], mat["n_layers"])
        self.nl = mat["n_layers"]
        self.dt = dt
        self.eps = mat["emissivity"]
        self.albedo = mat["albedo"]
        nodes_z = np.cumsum(self.dz) - 0.5 * self.dz
        # conductance between adjacent nodes, and to the bottom boundary
        self.g = self.k / (nodes_z[1:] - nodes_z[:-1])          # (nl-1,)
        self.bottom_bc = mat["bottom_bc"]
        if self.bottom_bc == "fixed":
            self.g_bot = self.k / (0.5 * self.dz[-1])
        else:  # interior convective (h may be 0 = adiabatic)
            self.g_bot = h_bottom
        self.T_bottom_ref = T_bottom_ref
        self.T = np.full((n_facets, self.nl), T_init, dtype=float)
        self.cap = self.C * self.dz / dt                        # (nl,)

    def step(self, Q_ext, h_conv, T_air_K):
        """One implicit step. Q_ext = SW_abs + eps*L_in (W/m2, per facet).
        The nonlinear emission eps*sigma*T^4 and convection are linearized
        about the current surface temperature (exact for the emitted term
        to first order; step size 600 s keeps the error << 0.01 K)."""
        nf, nl = self.T.shape
        Ts = self.T[:, 0]
        G_rad = 4.0 * self.eps * SIGMA * Ts ** 3
        Q_surf = (Q_ext + 3.0 * self.eps * SIGMA * Ts ** 4
                  + h_conv * T_air_K)
        G_surf = G_rad + h_conv

        a = np.zeros((nf, nl)); b = np.zeros((nf, nl))
        c = np.zeros((nf, nl)); d = np.zeros((nf, nl))
        b += self.cap[None, :]
        d += self.cap[None, :] * self.T
        # internal conduction
        b[:, :-1] += self.g[None, :]
        b[:, 1:] += self.g[None, :]
        c[:, :-1] = -self.g[None, :]
        a[:, 1:] = -self.g[None, :]
        # surface boundary
        b[:, 0] += G_surf
        d[:, 0] += Q_surf
        # bottom boundary
        b[:, -1] += self.g_bot
        d[:, -1] += self.g_bot * self.T_bottom_ref
        self.T = solve_tridiagonal_batched(a, b, c, d)
        return self.T[:, 0]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    facets_dir = Path(args.facets_dir)

    materials = {k: dict(v) for k, v in DEFAULT_MATERIALS.items()}
    if args.material_json:
        with open(args.material_json) as f:
            for name, over in json.load(f).items():
                materials[name].update(over)

    print("=" * 70)
    print("Loading facets and forcing...")
    fz = np.load(facets_dir / "facets.npz")
    centroids, normals = fz["centroid"], fz["normal"]
    classes = fz["cls"]; nf = len(centroids)
    times_df = pd.read_csv(Path(args.mrt_dir) / "times.csv")
    nt = len(times_df)
    dni = times_df["DNI_Wm2"].values
    dhi = times_df["DHI_Wm2"].values
    elev = times_df["elevation_deg"].values
    azim = times_df["azimuth_deg"].values
    air_C = times_df["air_temp_C"].values
    air_K = air_C + 273.15
    # Time step from times.csv. IMPORTANT: computed via total_seconds(),
    # which is correct for ANY datetime64 resolution. (An earlier version
    # used .astype("int64")/1e9, which silently assumes nanosecond epoch
    # integers; this pandas parses tz-aware strings at microsecond
    # resolution, making dt 1000x too small -- surfaces then barely
    # responded to forcing. Caught by the verification suite's physics
    # checks: sunlit ground stayed near air temperature at noon.)
    tt = pd.to_datetime(times_df["time"])
    dts = tt.diff().dt.total_seconds().values[1:]
    assert np.allclose(dts, dts[0]), "times.csv must be uniformly spaced"
    dt = float(dts[0])
    assert 30.0 <= dt <= 7200.0, (
        f"parsed dt = {dt} s is outside any plausible timestep range -- "
        f"times.csv parsing is broken, refusing to run")
    print(f"  {nf:,} facets, {nt} time steps, dt = {dt:.0f} s")

    print("Loading geometry (FULL meshes -- shading is never culled)...")
    b_mesh = trimesh.load(args.buildings_stl, force="mesh")
    g_mesh = trimesh.load(args.ground_stl, force="mesh")
    v_mesh = trimesh.load(args.vegetation_stl, force="mesh")
    solid_inters = [get_intersector(b_mesh), get_intersector(g_mesh)]
    veg_inter = get_intersector(v_mesh, quiet=True)

    print("\nFacet sky fractions (one-time hemisphere raytrace)...")
    t0 = time.time()
    f_sky = facet_sky_fraction(centroids, normals, solid_inters, veg_inter,
                               args.k_lad_diffuse, args.n_fsky_dirs,
                               args.surface_offset, args.facet_batch_size)
    print(f"  done in {time.time() - t0:.0f}s; "
          f"f_sky mean {f_sky.mean():.3f} range "
          f"[{f_sky.min():.3f}, {f_sky.max():.3f}]")

    print("\nPer-timestep direct-sun transmission for every facet...")
    tau_dir = np.zeros((nt, nf), dtype=np.float32)
    t0 = time.time()
    for it in range(nt):
        if elev[it] > 0.0:
            sv = sun_vector_enu(azim[it], elev[it])
            tau_dir[it] = facet_sun_transmission(
                centroids, normals, sv, solid_inters, veg_inter,
                args.k_lad_direct, args.surface_offset,
                args.facet_batch_size)
        if (it + 1) % 24 == 0 or it == nt - 1:
            print(f"  step {it + 1}/{nt} -- {time.time() - t0:.0f}s elapsed")

    # cos(theta) between each facet normal and the sun, per timestep
    sun_vecs = np.array([sun_vector_enu(a, e) for a, e in zip(azim, elev)])
    cos_theta = np.clip(normals @ sun_vecs.T, 0.0, None).T   # (nt, nf)
    sin_el = np.sin(np.deg2rad(np.clip(elev, 0.0, None)))
    L_sky_t = sky_longwave_down(air_C, args.cloud_cover_fraction)

    T_deep = (args.deep_soil_temp_c + 273.15 if args.deep_soil_temp_c
              is not None else float(air_K.mean()))
    T_int = args.interior_temp_c + 273.15
    h_conv = 5.7 + 3.8 * args.wind_speed   # Jurges correlation

    print(f"\nEnergy balance: h_conv = {h_conv:.1f} W/m2K, "
          f"T_deep = {T_deep - 273.15:.1f} C, T_int = {T_int - 273.15:.1f} C")

    solvers, members = {}, {}
    for cc, name in CLASS_TO_NAME.items():
        members[cc] = np.where(classes == cc)[0]
        if len(members[cc]):
            bot_ref = T_deep if materials[name]["bottom_bc"] == "fixed" else T_int
            solvers[cc] = ClassSolver(materials[name], len(members[cc]), dt,
                                      T_init=float(air_K.mean()),
                                      T_bottom_ref=bot_ref,
                                      h_bottom=args.h_interior)
            print(f"  {name:6s}: {len(members[cc]):,} facets, "
                  f"{materials[name]['n_layers']} layers, "
                  f"depth {materials[name]['depth']} m")

    eps_facet = np.zeros(nf)
    alb_facet = np.zeros(nf)
    for cc, name in CLASS_TO_NAME.items():
        eps_facet[members[cc]] = materials[name]["emissivity"]
        alb_facet[members[cc]] = materials[name]["albedo"]

    area = fz["area"]
    T_surf = np.full(nf, float(air_K.mean()))
    T_env = float(air_K.mean())        # lagged mean environment temperature
    facet_T = np.zeros((nt, nf), dtype=np.float32)
    n_cycles = args.spinup_days + 1
    cycle_end_snapshots = []

    print(f"\nTime integration: {n_cycles} diurnal cycles "
          f"({args.spinup_days} spin-up + 1 saved)...")
    for cyc in range(n_cycles):
        for it in range(nt):
            L_in = (f_sky * L_sky_t[it]
                    + (1.0 - f_sky) * args.env_emissivity * SIGMA * T_env ** 4)
            K_local = tau_dir[it] * dni[it] * sin_el[it] + f_sky * dhi[it]
            SW_in = (tau_dir[it] * dni[it] * cos_theta[it]
                     + f_sky * dhi[it]
                     + (1.0 - f_sky) * args.environment_albedo * K_local)
            for cc, sol in solvers.items():
                mem = members[cc]
                Q_ext = (1.0 - sol.albedo) * SW_in[mem] + sol.eps * L_in[mem]
                T_surf[mem] = sol.step(Q_ext, h_conv, air_K[it])
            T_env = float(np.average(T_surf, weights=area))
            if cyc == n_cycles - 1:
                facet_T[it] = T_surf
        cycle_end_snapshots.append(T_surf.copy())
        if cyc > 0:
            dmax = np.abs(cycle_end_snapshots[-1]
                          - cycle_end_snapshots[-2]).max()
            print(f"  cycle {cyc + 1}/{n_cycles}: max |dT| vs previous "
                  f"cycle end = {dmax:.3f} K")

    spinup_delta = (np.abs(cycle_end_snapshots[-1] - cycle_end_snapshots[-2])
                    .max() if n_cycles > 1 else np.nan)

    np.save(out_dir / "facet_T_matrix_K.npy", facet_T)
    np.save(out_dir / "f_sky_facet.npy", f_sky)
    np.save(out_dir / "facet_eps.npy", eps_facet)
    np.save(out_dir / "tau_dir_facet.npy", tau_dir)

    rows = []
    for it in range(nt):
        row = {"time": times_df["time"].iloc[it], "air_temp_C": air_C[it]}
        for cc, name in CLASS_TO_NAME.items():
            if len(members[cc]):
                Tc = facet_T[it, members[cc]] - 273.15
                row[f"{name}_mean_C"] = float(Tc.mean())
                row[f"{name}_max_C"] = float(Tc.max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "facet_summary_by_time.csv",
                              index=False)
    report = (f"Spin-up convergence: max |T(end of last cycle) - "
              f"T(end of previous)| = {spinup_delta:.4f} K over "
              f"{n_cycles} cycles\n"
              f"(values < ~0.5 K indicate a converged quasi-periodic "
              f"state; increase --spinup-days otherwise)\n")
    (out_dir / "spinup_report.txt").write_text(report)
    print("\n" + report)
    print(f"[facet_energy_balance] n_facets={nf} n_times={nt} "
          f"spinup_delta_K={spinup_delta:.4f} output_dir={out_dir}")


if __name__ == "__main__":
    main()
