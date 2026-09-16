"""
05b_facet_energy_balance.py -- surface temperatures for the route-visible
facets selected by 05a, via a per-facet 1D multilayer energy balance.

This REPLACES volumetric CHT for the purpose of route MRT: each facet gets

    absorbed shortwave (full-geometry sun shading + vegetation attenuation)
  + absorbed longwave  (sky + surrounding surfaces)
  - emitted longwave   (eps * sigma * Ts^4)
  - convection         (h_c * (Ts - Tair), resolved from the same time-varying
                         wind series used by the route calculation)
  - optional latent cooling for moisture-capable surfaces using a documented
    equilibrium-evaporation approximation
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
* Longwave exchange uses an energy-conserving grey-surface mean-field
  enclosure over only the route-visible facets selected by 05a.  Each
  facet's sky fraction closes the enclosure to the sky; the remaining view
  exchanges repeated reflected longwave with the area/view-weighted visible
  surface field.  This deliberately avoids solving irrelevant full-domain
  facet-to-facet view factors.

Outputs (in --output-dir)
-------------------------
  facet_T_matrix_K.npy   (n_times x n_facets) surface temperature, K
  f_sky_facet.npy        per-facet sky fraction (cosine-weighted)
  facet_eps.npy          per-facet emissivity used
  facet_radiosity_matrix_Wm2.npy
                         energy-conserving grey-surface radiosity for every
                         route-visible facet and time step
  radiosity_environment_Wm2.npy
                         enclosure-mean radiosity used for reflected LW
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
                            DEFAULT_MATERIALS, MATERIALS_MANIFEST, SIGMA,
                            RADIOSITY_ENVIRONMENT, RADIOSITY_MATRIX,
                            get_intersector, load_materials,
                            make_hemisphere_directions_about_normal,
                            solve_route_visible_grey_radiosity,
                            sky_longwave_down, solve_tridiagonal_batched,
                            sun_vector_enu,
                            vegetation_transmission_from_intersections)
from osm_ground_materials import GROUND_MATERIAL_CATALOG
from microclimate_field import MicroclimateField, add_microclimate_argument
from pedestrian_flow_field import (PedestrianFlowField,
                                   add_pedestrian_flow_argument,
                                   facet_wind_speed_matrix)

# Per-class material / model defaults now live in thermal_common.py so that
# 05 (pedestrian-side reflected shortwave) and 05b (facet absorption) cannot
# drift apart. Override any entry via --material-json.
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
    add_microclimate_argument(p)
    add_pedestrian_flow_argument(p)
    p.add_argument("--ground-material-dir", default=None,
                   help="Optional OSM ground-material output directory. Its "
                        "catalog supplies the same per-material properties "
                        "used to classify ground faces in 05a.")

    p.add_argument("--wind-speed", type=float, default=1.5,
                   help="Fallback constant near-surface wind speed, m/s")
    p.add_argument("--wind-source", choices=["times_csv", "constant"],
                   default="times_csv",
                   help="Use time-resolved wind_ms from stage-05 times.csv (default), "
                        "or the fallback --wind-speed constant.")
    p.add_argument(
        "--convection-model",
        choices=["mcadams", "watmuff", "legacy_subtracted", "combined", "convective"],
        default="mcadams",
        help=("Exterior surface film coefficient. mcadams=5.7+3.8U (default) "
              "is a COMBINED coefficient, so the radiative film it already "
              "contains is removed automatically (4*eps*sigma*T_air^3) before "
              "use -- this stage computes eps*sigma*T^4 explicitly and would "
              "otherwise count radiation twice. watmuff=2.8+3.0U was proposed "
              "as the convective-only alternative for exactly that reason and "
              "is used as-is. legacy_subtracted uses the older fixed "
              "--radiative-film-wm2k constant instead. combined and convective "
              "remain aliases for backward-compatible command lines. Set "
              "--no-radiative-film-removal to reproduce pre-fix runs."),
    )
    p.add_argument("--convection-reference-wind",
                   choices=["free_stream", "facet_local"],
                   default="free_stream",
                   help="Which velocity drives the a+b*U film coefficient. "
                        "'free_stream' (default) uses the experiment-derived "
                        "wind_freestream_ms from times.csv -- the cart wind "
                        "lifted through the urban canopy profile to above the "
                        "roughness sublayer, which is the UNDISTURBED approach "
                        "velocity these correlations were calibrated against. "
                        "'facet_local' uses the sheltered potential-flow wind "
                        "at each facet, which preserves spatial variation but "
                        "feeds the correlation a velocity roughly 3x smaller "
                        "than its calibration basis. Falls back to facet_local "
                        "when the CSV has no free-stream column.")
    p.add_argument("--h-conv-a", type=float, default=5.7,
                   help="Intercept of the McAdams film coefficient a + b*U (default 5.7)")
    p.add_argument("--h-conv-b", type=float, default=3.8,
                   help="Wind slope of the McAdams film coefficient a + b*U (default 3.8)")
    p.add_argument("--no-radiative-film-removal", action="store_true",
                   help="Keep the COMBINED film coefficient as if it were "
                        "purely convective, reproducing runs made before the "
                        "radiative double-count was fixed. Physically "
                        "inconsistent: it counts longwave exchange twice, once "
                        "inside the correlation's intercept and once in the "
                        "explicit eps*sigma*T^4 term.")
    p.add_argument("--radiative-film-wm2k", type=float, default=6.0,
                   help="Radiative part of the combined film coefficient to remove under "
                        "--convection-model=convective, W/m2K (~4*eps*sigma*T^3 at ~305 K; "
                        "default 6.0)")
    p.add_argument("--h-conv-floor", type=float, default=2.0,
                   help="Lower bound on the convective coefficient after removing the "
                        "radiative part, W/m2K (natural-convection floor; default 2.0)")
    p.add_argument("--interior-temp-c", type=float, default=24.0,
                   help="Building interior air temperature, deg C")
    p.add_argument("--h-interior", type=float, default=8.0,
                   help="Interior convective coefficient W/m2K "
                        "(0 = adiabatic interior; used by tests)")
    p.add_argument("--wall-insulation-r", type=float, default=None,
                   help="Override wall interior insulation resistance, m2K/W (in series "
                        "with --h-interior at the back face). Higher = exterior wall more "
                        "decoupled from the conditioned interior, so shaded walls stay near "
                        "ambient instead of being pulled below it. Default from "
                        "thermal_common (1.5). Set 0 for the old bare-wall-to-AC behaviour.")
    p.add_argument("--roof-insulation-r", type=float, default=None,
                   help="Override roof interior insulation resistance, m2K/W (default 2.5).")
    p.add_argument("--deep-soil-temp-c", type=float, default=None,
                   help="Deep soil temperature; default = daily mean air T")
    p.add_argument("--env-emissivity", type=float, default=0.95,
                   help="Emissivity of surrounding surfaces as seen BY a "
                        "facet under --longwave-radiosity-model=legacy. The "
                        "grey-radiosity model uses each selected facet's "
                        "resolved material emissivity instead.")
    p.add_argument("--longwave-radiosity-model",
                   choices=["grey", "legacy"], default="grey",
                   help="Longwave exchange among the route-visible facets. "
                        "'grey' (default) solves emitted plus reflected "
                        "incident LW with repeated-reflection enclosure "
                        "closure. 'legacy' reproduces the earlier emitted-only "
                        "pedestrian radiosity and mean-temperature surface "
                        "irradiation for comparison only.")
    p.add_argument("--environment-albedo", type=float, default=None,
                   help="Albedo of surroundings reflecting SW onto facets. "
                        "Default: the resolved ground albedo, since at "
                        "pedestrian level the reflecting surround is "
                        "predominantly ground.")
    p.add_argument("--cloud-cover-fraction", type=float, default=0.0,
                   help="MUST match the value used in 05 (times.csv stores "
                        "cloud-adjusted DNI/DHI but L_sky needs the fraction)")
    p.add_argument("--clear-sky-emissivity", choices=["prata", "constant"], default="prata",
                   help="Clear-sky longwave emissivity model; MUST match 05 so the "
                        "surfaces heated here and the pedestrian in 05 see the same sky. "
                        "'prata' (default) is humidity-dependent; 'constant' = old 0.78.")
    p.add_argument("--fallback-rh-pct", type=float, default=70.0,
                   help="RH used for sky longwave only if times.csv lacks an rh_pct "
                        "column (older 05 output). Default 70%%.")
    p.add_argument("--k-lad-direct", type=float, default=0.45,
                   help="Vegetation extinction for the direct beam "
                        "(match 05)")
    p.add_argument("--k-lad-diffuse", type=float, default=0.30,
                   help="Vegetation extinction for diffuse sky (match 05)")
    p.add_argument("--spinup-days", type=int, default=2,
                   help="Minimum diurnal spin-up cycles before convergence may stop")
    p.add_argument("--maximum-spinup-days", type=int, default=10,
                   help="Maximum diurnal spin-up cycles before the saved cycle")
    p.add_argument("--spinup-convergence-tolerance-k", type=float, default=0.10,
                   help="Maximum cycle-end temperature change required for convergence")
    p.add_argument("--latent-heat-model", choices=["equilibrium", "none"],
                   default="equilibrium",
                   help="Material-dependent equilibrium evaporative cooling (default) "
                        "or none for the former dry-surface behavior.")
    p.add_argument("--latent-priestley-taylor-alpha", type=float, default=1.26)
    p.add_argument("--latent-net-radiation-cap", type=float, default=0.95,
                   help="Maximum fraction of positive instantaneous net radiation "
                        "removed as latent heat.")
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
        else:  # interior: air film h_bottom in SERIES with any insulation R
            R_ins = float(mat.get("insulation_R_m2K_W", 0.0))
            if h_bottom > 0:
                self.g_bot = 1.0 / (1.0 / h_bottom + R_ins)
            else:
                self.g_bot = 0.0            # adiabatic interior (tests)
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


def convection_coefficient(wind_ms, model="mcadams", *, a=5.7, b=3.8,
                           radiative_film=6.0, floor=2.0):
    """Return exterior convective heat-transfer coefficients in W m-2 K-1.

    ``legacy_subtracted`` reproduces the previous implementation exactly.
    The aliases ``combined`` and ``convective`` preserve old command lines.
    """
    wind = np.asarray(wind_ms, dtype=float)
    if not np.isfinite(wind).all() or np.any(wind < 0):
        raise ValueError("surface-energy wind speed must be finite and non-negative")
    if model in ("mcadams", "combined"):
        coefficient = a + b * wind
    elif model == "watmuff":
        coefficient = 2.8 + 3.0 * wind
    elif model in ("legacy_subtracted", "convective"):
        coefficient = np.maximum(a + b * wind - radiative_film, floor)
    else:
        raise ValueError(f"unsupported convection model: {model}")
    return coefficient


def equilibrium_latent_heat_flux(net_radiation_wm2, air_temp_c,
                                 evaporative_efficiency,
                                 alpha=1.26, cap_fraction=0.95):
    """Simplified Priestley-Taylor equilibrium latent cooling, W m-2.

    The material ``evaporative_efficiency`` (0=dry, 1=unlimited wet surface)
    represents water availability.  This is a transparent first-order surface
    energy term, not a soil-moisture or vegetation-physiology model.
    """
    if not 0.0 <= cap_fraction <= 1.0:
        raise ValueError("latent net-radiation cap must lie in [0, 1]")
    efficiency = np.asarray(evaporative_efficiency, dtype=float)
    if (not np.isfinite(efficiency).all()
            or np.any((efficiency < 0) | (efficiency > 1))):
        raise ValueError("evaporative_efficiency must be finite and in [0, 1]")
    temperature = np.asarray(air_temp_c, dtype=float)
    available = np.maximum(np.asarray(net_radiation_wm2, dtype=float), 0.0)
    saturation_kpa = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    slope_kpa_k = 4098.0 * saturation_kpa / (temperature + 237.3) ** 2
    psychrometric_kpa_k = 0.066
    equilibrium_fraction = alpha * slope_kpa_k / (
        slope_kpa_k + psychrometric_kpa_k)
    latent = efficiency * equilibrium_fraction * available
    return np.minimum(latent, cap_fraction * available)


def main():
    args = parse_args()
    if args.spinup_days < 1 or args.maximum_spinup_days < args.spinup_days:
        raise ValueError("spin-up days must satisfy 1 <= minimum <= maximum")
    if args.spinup_convergence_tolerance_k <= 0:
        raise ValueError("spin-up convergence tolerance must be positive")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    facets_dir = Path(args.facets_dir)

    materials = load_materials(args.material_json)
    if args.ground_material_dir:
        catalog_path = Path(args.ground_material_dir) / GROUND_MATERIAL_CATALOG
        if not catalog_path.is_file():
            raise FileNotFoundError(
                f"ground material catalog not found: {catalog_path}")
        with open(catalog_path, encoding="utf-8") as stream:
            ground_catalog = json.load(stream)
        for name, values in ground_catalog["materials"].items():
            if name not in materials:
                raise ValueError(f"unknown ground material in catalog: {name}")
            materials[name].update(values)
        # An explicit --material-json is a deliberate CLI override and must win.
        # Applying the catalog on top of it silently discarded every ground-class
        # override the user asked for, which is the normal pipeline path (the
        # catalog is always present once stage 2 has run) -- so the flag looked
        # like it worked while changing nothing.
        if args.material_json:
            with open(args.material_json, encoding="utf-8") as stream:
                explicit = json.load(stream)
            for name, values in explicit.items():
                materials[name].update(values)
    if args.wall_insulation_r is not None:
        materials["wall"]["insulation_R_m2K_W"] = args.wall_insulation_r
    if args.roof_insulation_r is not None:
        materials["roof"]["insulation_R_m2K_W"] = args.roof_insulation_r

    # The surround reflecting shortwave back onto a facet at street level is
    # predominantly ground, so default it to the ground albedo rather than an
    # unrelated hard-coded constant.
    if args.environment_albedo is None:
        default_ground_name = ("generic_ground" if args.ground_material_dir
                               else "ground")
        args.environment_albedo = materials[default_ground_name]["albedo"]
        env_alb_src = f"inherited from {default_ground_name} albedo"
    else:
        env_alb_src = "explicit --environment-albedo"

    print("=" * 70)
    print("Surface radiative properties (single source: thermal_common.py)")
    display_names = ["ground", "wall", "roof"]
    if args.ground_material_dir:
        display_names = list(ground_catalog["material_names"]) + ["wall", "roof"]
    for name in display_names:
        m = materials[name]
        ins = (f"   insulation_R {m['insulation_R_m2K_W']:.2f} m2K/W"
               if m.get("bottom_bc") == "interior" else "")
        print(f"  {name:<7s} albedo {m['albedo']:.3f}   "
              f"emissivity {m['emissivity']:.3f}{ins}")
    print(f"  environment albedo {args.environment_albedo:.3f} "
          f"({env_alb_src})")

    # Write the manifest that 05 reads back, so the pedestrian-side reflected
    # shortwave uses the SAME ground albedo that actually heated the ground.
    manifest = {
        name: {
            key: (float(value) if isinstance(value, (int, float, np.number))
                  else value)
            for key, value in m.items()
        }
        for name, m in materials.items()
    }
    manifest["_environment_albedo"] = float(args.environment_albedo)
    selection_config_path = facets_dir / "config.json"
    selection_config = {}
    if selection_config_path.is_file():
        with open(selection_config_path) as f:
            raw_selection_config = json.load(f)
        selection_config = {
            key: raw_selection_config.get(key)
            for key in (
                "max_distance", "point_stride", "n_lw_azimuth",
                "n_lw_elevation", "body_model",
            )
        }
    manifest["_longwave_radiosity"] = {
        "model": args.longwave_radiosity_model,
        "scope": "stage-05a first-hit route-visible facets within max_distance",
        "selection": selection_config,
    }
    with open(out_dir / MATERIALS_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {out_dir / MATERIALS_MANIFEST}")

    print("=" * 70)
    print("Loading facets and forcing...")
    fz = np.load(facets_dir / "facets.npz")
    centroids, normals = fz["centroid"], fz["normal"]
    classes = fz["cls"]; nf = len(centroids)
    if "material_name" in fz.files:
        facet_material_name = fz["material_name"].astype(str)
    else:
        facet_material_name = np.array(
            [CLASS_TO_NAME[int(value)] for value in classes], dtype="U32")
    unknown_materials = sorted(set(facet_material_name) - set(materials))
    if unknown_materials:
        raise ValueError(f"facets reference unknown materials: {unknown_materials}")
    times_df = pd.read_csv(Path(args.mrt_dir) / "times.csv")
    nt = len(times_df)
    dni = times_df["DNI_Wm2"].values
    dhi = times_df["DHI_Wm2"].values
    elev = times_df["elevation_deg"].values
    azim = times_df["azimuth_deg"].values
    air_C = times_df["air_temp_C"].values
    air_K = air_C + 273.15
    # RH per timestep drives the humidity-dependent clear-sky longwave. 05
    # writes it into times.csv; fall back with a warning for older files.
    if "rh_pct" in times_df.columns:
        rh_pct = times_df["rh_pct"].values
    else:
        rh_pct = np.full(nt, args.fallback_rh_pct)
        print(f"  WARNING: times.csv has no rh_pct column; using "
              f"--fallback-rh-pct {args.fallback_rh_pct}% for sky longwave. "
              f"Re-run 05 to write RH for an exact match.")
    if args.wind_source == "times_csv":
        if "wind_ms" not in times_df.columns:
            raise ValueError(
                "--wind-source=times_csv requires wind_ms in stage-05 times.csv")
        wind_ms = times_df["wind_ms"].to_numpy(float)
        wind_source = "times.csv wind_ms"
    else:
        wind_ms = np.full(nt, args.wind_speed, dtype=float)
        wind_source = f"constant --wind-speed={args.wind_speed:g} m/s"
    # The optional solved field makes the exterior boundary condition local
    # to each route-visible facet.  Without it, these broadcast views preserve
    # the historical spatially uniform calculation exactly.
    microclimate = (MicroclimateField(args.microclimate_dir)
                    if args.microclimate_dir else None)
    pedestrian_flow = None
    if args.pedestrian_flow_dir and microclimate is not None:
        print("  NOTE: both --microclimate-dir and --pedestrian-flow-dir "
              "given; the solved 3-D microclimate field takes precedence "
              "for facet Ta and wind.")
    elif args.pedestrian_flow_dir:
        pedestrian_flow = PedestrianFlowField(args.pedestrian_flow_dir)
    if microclimate is not None:
        field_times = pd.to_datetime(times_df["time"])
        field_hours = np.array([
            value.hour + value.minute / 60.0 + value.second / 3600.0
            for value in field_times], dtype=float)
        air_C_facet = np.empty((nt, nf), dtype=np.float32)
        wind_facet = np.empty((nt, nf), dtype=np.float32)
        for it, hour in enumerate(field_hours):
            local_ta, local_u, local_v, local_w = microclimate.sample(
                centroids, hour)
            air_C_facet[it] = local_ta
            wind_facet[it] = np.sqrt(
                local_u * local_u + local_v * local_v + local_w * local_w)
        if not (np.isfinite(air_C_facet).all()
                and np.isfinite(wind_facet).all()
                and np.all(wind_facet >= 0)):
            raise ValueError("sampled facet microclimate forcing is invalid")
        wind_source = f"solved microclimate field {args.microclimate_dir}"
        print(f"  Microclimate coupling: local facet Ta "
              f"{air_C_facet.min():.1f}..{air_C_facet.max():.1f} C, "
              f"speed {wind_facet.min():.2f}..{wind_facet.max():.2f} m/s")
    elif pedestrian_flow is not None:
        # Step-3 pedestrian-level potential flow supplies only a local wind
        # MAGNITUDE for the existing parameterized convection coefficient.
        # Air temperature stays spatially uniform, the convection
        # correlation is unchanged, and the radiation calculation is
        # untouched.  Linearity of the Laplace solve lets the one solved
        # unit-normalized field scale with the shared time-varying wind.
        air_C_facet = np.broadcast_to(air_C[:, None], (nt, nf))
        # Scale the field by the INLET (free-stream) series when stage 05
        # carried one: the potential-flow solution is linear in the boundary
        # speed, so a time-varying inlet gives a time-varying domain field from
        # the single solve. wind_ms is the wind a PERSON feels and is not the
        # boundary condition; using it to scale the field double-counts the
        # local sheltering the field already represents.
        if "wind_inlet_ms" in times_df.columns:
            inlet_ms = times_df["wind_inlet_ms"].to_numpy(float)
            if not np.isfinite(inlet_ms).all() or np.any(inlet_ms < 0):
                raise ValueError("times.csv wind_inlet_ms must be finite and "
                                 "non-negative")
            scaling_source = (f"times.csv wind_inlet_ms "
                              f"({inlet_ms.min():.2f}..{inlet_ms.max():.2f} m/s)")
        else:
            inlet_ms = wind_ms
            scaling_source = f"{wind_source} (no separate inlet series)"
        wind_facet = facet_wind_speed_matrix(
            pedestrian_flow, centroids[:, :2], inlet_ms).astype(np.float32)
        if not (np.isfinite(wind_facet).all() and np.all(wind_facet >= 0)):
            raise ValueError("sampled pedestrian-flow facet wind is invalid")
        wind_source = (f"pedestrian potential-flow field "
                       f"{args.pedestrian_flow_dir} scaled by {scaling_source}")
        print(f"  Pedestrian-flow coupling: {pedestrian_flow.describe()}")
        print(f"  Local facet wind speed "
              f"{wind_facet.min():.2f}..{wind_facet.max():.2f} m/s "
              f"(uniform Ta retained)")
    else:
        air_C_facet = np.broadcast_to(air_C[:, None], (nt, nf))
        wind_facet = np.broadcast_to(wind_ms[:, None], (nt, nf))
    air_K_facet = air_C_facet + 273.15
    # ------------------------------------------------------------------
    # WHICH VELOCITY DRIVES CONVECTION
    #
    # McAdams-type a + b*U correlations were measured on flat plates in a
    # uniform wind-tunnel stream, so U is the UNDISTURBED APPROACH velocity.
    # Feeding them the sheltered in-canopy wind at a facet is a category error:
    # on lisbon1 the free stream is ~3x the cart-measured wind, which is most of
    # the gap between the h ~ 10 this stage produced and the 15-30 that urban
    # schemes report. wind_freestream_ms is the measured wind lifted through the
    # canopy profile (wind_profile.py) -- an experiment-derived boundary
    # condition, never taken from the flow solution it goes on to drive.
    #
    # The trade-off is explicit: the free stream is spatially uniform, so it
    # buys correct magnitude at the cost of the local variation the
    # potential-flow field provides. Genuine spatial variation of the
    # convective coefficient needs a CHTC model, not a rescaled correlation.
    # ------------------------------------------------------------------
    convection_wind = wind_facet
    convection_wind_source = "facet-local potential-flow wind"
    if args.convection_reference_wind == "free_stream":
        if "wind_freestream_ms" in times_df.columns:
            free_stream = times_df["wind_freestream_ms"].to_numpy(float)
            convection_wind = np.broadcast_to(free_stream[:, None], (nt, nf))
            convection_wind_source = (
                "experiment-derived free stream (wind_freestream_ms, cart wind "
                "lifted through the urban canopy profile)")
        else:
            convection_wind_source = (
                "facet-local potential-flow wind (no wind_freestream_ms in "
                "times.csv; run weather_from_sensors.py to generate it)")
    print(f"  Convection reference wind: {convection_wind_source}; "
          f"{convection_wind.mean():.2f} m/s mean "
          f"(facet-local mean {wind_facet.mean():.2f} m/s)")
    h_conv_facet = convection_coefficient(
        convection_wind, args.convection_model, a=args.h_conv_a, b=args.h_conv_b,
        radiative_film=args.radiative_film_wm2k, floor=args.h_conv_floor)
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
    cloud_fraction_time = (times_df["cloud_fraction"].to_numpy(float)
                           if "cloud_fraction" in times_df.columns
                           else np.full(nt, args.cloud_cover_fraction))
    if (not np.isfinite(cloud_fraction_time).all()
            or np.any((cloud_fraction_time < 0) | (cloud_fraction_time > 1))):
        raise ValueError("times.csv cloud_fraction must be finite and in [0, 1]")
    parameterized_lsky = sky_longwave_down(
        air_C, rh_pct, cloud_fraction_time,
        clear_sky_model=args.clear_sky_emissivity)
    if "LWin_Wm2" in times_df.columns:
        supplied_lsky = times_df["LWin_Wm2"].to_numpy(float)
        invalid = np.isfinite(supplied_lsky) & (supplied_lsky < 0)
        if invalid.any():
            raise ValueError("times.csv LWin_Wm2 contains negative values")
        L_sky_t = np.where(np.isfinite(supplied_lsky), supplied_lsky,
                           parameterized_lsky)
        lsky_source = ("times.csv where supplied; Prata/selected clear-sky model "
                       "otherwise")
    else:
        L_sky_t = parameterized_lsky
        lsky_source = "parameterized from times.csv Ta/RH/cloud"

    T_deep = (args.deep_soil_temp_c + 273.15 if args.deep_soil_temp_c
              is not None else float(air_K.mean()))
    T_int = args.interior_temp_c + 273.15
    print(f"\nEnergy balance: h_conv = {h_conv_facet.min():.1f}.."
          f"{h_conv_facet.max():.1f} W/m2K "
          f"({args.convection_model}; {wind_source}), "
          f"T_deep = {T_deep - 273.15:.1f} C, T_int = {T_int - 273.15:.1f} C")
    print(f"  Sky longwave: {lsky_source}")
    print(f"  Latent heat: {args.latent_heat_model} "
          f"(material evaporative efficiency; alpha={args.latent_priestley_taylor_alpha:g})")
    print(f"  Longwave exchange: {args.longwave_radiosity_model} "
          f"({'emission + reflected incident LW' if args.longwave_radiosity_model == 'grey' else 'legacy emitted-only comparison'})")

    solvers, members = {}, {}
    active_materials = list(dict.fromkeys(facet_material_name.tolist()))
    for name in active_materials:
        members[name] = np.where(facet_material_name == name)[0]
        if len(members[name]):
            bot_ref = T_deep if materials[name]["bottom_bc"] == "fixed" else T_int
            solvers[name] = ClassSolver(materials[name], len(members[name]), dt,
                                        T_init=float(air_K.mean()),
                                        T_bottom_ref=bot_ref,
                                        h_bottom=args.h_interior)
            print(f"  {name:25s}: {len(members[name]):,} facets, "
                  f"{materials[name]['n_layers']} layers, "
                  f"depth {materials[name]['depth']} m")

    eps_facet = np.zeros(nf)
    alb_facet = np.zeros(nf)
    for name in active_materials:
        eps_facet[members[name]] = materials[name]["emissivity"]
        alb_facet[members[name]] = materials[name]["albedo"]
    # ------------------------------------------------------------------
    # RADIATIVE DOUBLE-COUNT REMOVAL
    #
    # McAdams' 5.7 + 3.8U is a COMBINED surface film coefficient: its intercept
    # already contains the radiative exchange of a surface sitting near ambient.
    # This stage ALSO computes longwave emission explicitly as eps*sigma*T^4,
    # linearised in the solver as 4*eps*sigma*Ts^3. Using the combined
    # coefficient as if it were purely convective therefore counts radiation
    # twice, roughly 6 W/m2K out of a total surface conductance near 30 -- about
    # a fifth of everything carrying heat away from the surface.
    #
    # The removal subtracts the model's OWN radiative film rather than a
    # hard-coded constant, so the two halves cannot drift apart:
    #
    #     h_c = max(a + b*U - 4*eps*sigma*T_air^3, floor)
    #
    # Evaluated at AIR temperature, not surface temperature, because that is
    # the condition under which the combined correlation was measured: the
    # radiative part embedded in the intercept is the one a near-ambient
    # surface has. Subtracting 4*eps*sigma*Ts^3 at a 50 C surface would remove
    # more than was ever included.
    #
    # NOTE ON DIRECTION: this removes dissipation, so surfaces get HOTTER. It
    # is nonetheless the physically consistent choice -- the previous behaviour
    # was masking part of a genuine convective deficit rather than correcting
    # it. See MATERIAL/README notes and verify_thermal_pipeline.
    # ------------------------------------------------------------------
    # Watmuff et al. (1977) proposed 2.8 + 3.0U precisely BECAUSE they argued
    # McAdams' 5.7 + 3.8U already contains radiation. So 'watmuff' is already a
    # convective-only correlation and must NOT have a radiative film removed
    # from it a second time -- only the combined forms are corrected here.
    if (args.convection_model in ("mcadams", "combined")
            and not args.no_radiative_film_removal):
        embedded_radiative_film = (4.0 * eps_facet[None, :] * SIGMA
                                   * air_K_facet ** 3)
        h_conv_raw_mean = float(h_conv_facet.mean())
        h_conv_facet = np.maximum(h_conv_facet - embedded_radiative_film,
                                  float(args.h_conv_floor))
        print(f"  Radiative double-count removed from the combined film "
              f"coefficient: h_conv {h_conv_raw_mean:.1f} -> "
              f"{h_conv_facet.mean():.1f} W/m2K "
              f"(embedded radiative film "
              f"{float(embedded_radiative_film.mean()):.1f}, floor "
              f"{args.h_conv_floor:.1f}); explicit eps*sigma*T^4 now carries "
              "the radiative exchange alone")

    evaporation_facet = np.zeros(nf)
    for name in active_materials:
        evaporation_facet[members[name]] = float(
            materials[name].get("evaporative_efficiency", 0.0))
    if (not np.isfinite(evaporation_facet).all()
            or np.any((evaporation_facet < 0) | (evaporation_facet > 1))):
        raise ValueError("material evaporative_efficiency must lie in [0, 1]")

    area = fz["area"]
    T_surf = np.full(nf, float(air_K.mean()))
    T_env = float(air_K.mean())        # legacy-only lagged mean temperature
    facet_T = np.zeros((nt, nf), dtype=np.float32)
    cycle_end_snapshots = []
    facet_latent = np.zeros((nt, nf), dtype=np.float32)

    maximum_cycles = args.maximum_spinup_days + 1
    print(f"\nTime integration: minimum {args.spinup_days + 1}, maximum "
          f"{maximum_cycles} diurnal cycles; stop at cycle-end change <= "
          f"{args.spinup_convergence_tolerance_k:g} K...")
    converged = False
    for cyc in range(maximum_cycles):
        for it in range(nt):
            if args.longwave_radiosity_model == "grey":
                # Route-local grey enclosure: only the facets selected by 05a
                # participate. The analytic solve includes repeated reflection
                # and returns the incident LW absorbed by each grey surface.
                _, L_in, _ = solve_route_visible_grey_radiosity(
                    T_surf, eps_facet, f_sky, L_sky_t[it], area)
            else:
                L_in = (f_sky * L_sky_t[it]
                        + (1.0 - f_sky) * args.env_emissivity
                        * SIGMA * T_env ** 4)
            K_local = tau_dir[it] * dni[it] * sin_el[it] + f_sky * dhi[it]
            SW_in = (tau_dir[it] * dni[it] * cos_theta[it]
                     + f_sky * dhi[it]
                     + (1.0 - f_sky) * args.environment_albedo * K_local)
            latent_all = np.zeros(nf)
            for name, sol in solvers.items():
                mem = members[name]
                Q_ext = (1.0 - sol.albedo) * SW_in[mem] + sol.eps * L_in[mem]
                if args.latent_heat_model == "equilibrium":
                    net_radiation = ((1.0 - sol.albedo) * SW_in[mem]
                                     + sol.eps * (L_in[mem]
                                                  - SIGMA * T_surf[mem] ** 4))
                    latent = equilibrium_latent_heat_flux(
                        net_radiation, air_C_facet[it, mem],
                        evaporation_facet[mem],
                        alpha=args.latent_priestley_taylor_alpha,
                        cap_fraction=args.latent_net_radiation_cap)
                    Q_ext = Q_ext - latent
                    latent_all[mem] = latent
                T_surf[mem] = sol.step(
                    Q_ext, h_conv_facet[it, mem], air_K_facet[it, mem])
            if args.longwave_radiosity_model == "legacy":
                T_env = float(np.average(T_surf, weights=area))
            facet_T[it] = T_surf
            facet_latent[it] = latent_all
        cycle_end_snapshots.append(T_surf.copy())
        if cyc > 0:
            dmax = np.abs(cycle_end_snapshots[-1]
                          - cycle_end_snapshots[-2]).max()
            print(f"  cycle {cyc + 1}/{maximum_cycles}: max |dT| vs previous "
                  f"cycle end = {dmax:.3f} K")
            if (cyc >= args.spinup_days
                    and dmax <= args.spinup_convergence_tolerance_k):
                converged = True
                break
        else:
            # Also expose completion of the first (often longest) cycle so
            # the TREC-Route UI progress bar does not remain static until the
            # second full diurnal integration has finished.
            print(f"  cycle 1/{maximum_cycles}: initial diurnal integration complete")

    n_cycles = len(cycle_end_snapshots)

    spinup_delta = (np.abs(cycle_end_snapshots[-1] - cycle_end_snapshots[-2])
                    .max() if n_cycles > 1 else np.nan)

    np.save(out_dir / "facet_T_matrix_K.npy", facet_T)
    np.save(out_dir / "f_sky_facet.npy", f_sky)
    np.save(out_dir / "facet_eps.npy", eps_facet)
    np.save(out_dir / "facet_albedo.npy", alb_facet)
    np.save(out_dir / "facet_material_name.npy", facet_material_name)
    np.save(out_dir / "tau_dir_facet.npy", tau_dir)
    np.save(out_dir / "facet_latent_heat_matrix_Wm2.npy", facet_latent)
    np.save(out_dir / "facet_evaporative_efficiency.npy", evaporation_facet)
    if microclimate is not None:
        np.savez_compressed(
            out_dir / "facet_microclimate_forcing.npz",
            air_temperature_C=air_C_facet,
            wind_speed_ms=wind_facet,
            h_conv_Wm2K=np.asarray(h_conv_facet, dtype=np.float32))
    if pedestrian_flow is not None:
        np.savez_compressed(
            out_dir / "facet_pedestrian_wind_forcing.npz",
            wind_speed_ms=wind_facet,
            h_conv_Wm2K=np.asarray(h_conv_facet, dtype=np.float32))
    # Record which convection-wind source actually drove this run so
    # downstream comparisons can prove no silent forcing switch occurred.
    (out_dir / "surface_wind_provenance.json").write_text(json.dumps({
        "wind_source": wind_source,
        "convection_reference_wind": args.convection_reference_wind,
        "convection_wind_source": convection_wind_source,
        "convection_wind_mean_ms": float(convection_wind.mean()),
        "facet_local_wind_mean_ms": float(wind_facet.mean()),
        "h_conv_mean_Wm2K": float(h_conv_facet.mean()),
        "radiative_film_removed": bool(
            args.convection_model in ("mcadams", "combined")
            and not args.no_radiative_film_removal),
        "inlet_series_used": bool(
            pedestrian_flow is not None
            and "wind_inlet_ms" in times_df.columns),
        "convection_model": args.convection_model,
        "microclimate_dir": args.microclimate_dir,
        "pedestrian_flow_dir": (args.pedestrian_flow_dir
                                if pedestrian_flow is not None else None),
        "uniform_fallback": (microclimate is None
                            and pedestrian_flow is None),
    }, indent=2), encoding="utf-8")

    radiosity_report = [
        f"Longwave radiosity model: {args.longwave_radiosity_model}",
        "Scope: stage-05a first-hit route-visible facets within its configured max distance",
    ]
    if args.longwave_radiosity_model == "grey":
        facet_J, facet_G, environment_J = solve_route_visible_grey_radiosity(
            facet_T.astype(float), eps_facet, f_sky, L_sky_t, area)
        closure = (
            eps_facet[None, :] * SIGMA * facet_T.astype(float) ** 4
            + (1.0 - eps_facet)[None, :] * facet_G
        )
        closure_error = float(np.max(np.abs(facet_J - closure)))
        if closure_error > 1e-8:
            raise RuntimeError(
                f"Grey-radiosity energy closure failed: {closure_error:.3e} W/m2")
        np.save(out_dir / RADIOSITY_MATRIX, facet_J.astype(np.float32))
        np.save(out_dir / RADIOSITY_ENVIRONMENT,
                np.asarray(environment_J, dtype=np.float32))
        radiosity_report.extend([
            "Equation: J = eps*sigma*T^4 + (1-eps)*G",
            "Closure: G = f_sky*L_sky + (1-f_sky)*J_environment",
            f"Maximum radiosity identity residual: {closure_error:.3e} W/m2",
            f"Environment radiosity range: {environment_J.min():.2f}..{environment_J.max():.2f} W/m2",
        ])
    else:
        radiosity_report.append(
            "Legacy comparison mode: no grey-radiosity matrix is authoritative; final MRT uses emitted-only fallback.")
    (out_dir / "radiosity_report.txt").write_text(
        "\n".join(radiosity_report) + "\n", encoding="utf-8")

    rows = []
    for it in range(nt):
        row = {"time": times_df["time"].iloc[it],
               "air_temp_C": float(np.mean(air_C_facet[it])),
               "air_temp_min_C": float(np.min(air_C_facet[it])),
               "air_temp_max_C": float(np.max(air_C_facet[it])),
               "wind_ms": float(np.mean(wind_facet[it])),
               "wind_min_ms": float(np.min(wind_facet[it])),
               "wind_max_ms": float(np.max(wind_facet[it])),
               "h_conv_Wm2K": float(np.mean(h_conv_facet[it])),
               "latent_heat_mean_Wm2": float(facet_latent[it].mean()),
               "latent_heat_max_Wm2": float(facet_latent[it].max())}
        for name in active_materials:
            if len(members[name]):
                Tc = facet_T[it, members[name]] - 273.15
                row[f"{name}_mean_C"] = float(Tc.mean())
                row[f"{name}_max_C"] = float(Tc.max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "facet_summary_by_time.csv",
                              index=False)
    status = "converged" if converged else "maximum cycles reached"
    report = (f"Spin-up convergence ({status}): max |T(end of last cycle) - "
              f"T(end of previous)| = {spinup_delta:.4f} K over "
              f"{n_cycles} cycles\n"
              f"Target tolerance = {args.spinup_convergence_tolerance_k:.4f} K; "
              f"minimum spin-up days = {args.spinup_days}, maximum = "
              f"{args.maximum_spinup_days}\n")
    (out_dir / "spinup_report.txt").write_text(report)
    print("\n" + report)
    print(f"[facet_energy_balance] n_facets={nf} n_times={nt} "
          f"spinup_delta_K={spinup_delta:.4f} output_dir={out_dir}")


if __name__ == "__main__":
    main()
