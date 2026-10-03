"""campus_walks.py -- run the four campus routes through JOS-3 under a
perturbed radiant field or a perturbed air temperature.

Shared by paper_ranking_uncertainty.py (validation error propagated to the
routing decision) and paper_ta_heterogeneity.py (spatial air-temperature
anomalies). Everything that is NOT perturbed -- routes, pace, subject,
clothing, start state, JOS-3 protocol -- is taken from stage 09 exactly as the
paper's departure sweep does (paper_jos3_figures.run_sweep), so a run with no
perturbation reproduces the stage-09 numbers.

Two kinds of perturbation are supported:

  * a replacement Tmrt matrix for a route (one row per stage-05 time step, one
    column per resampled route point), built from the body's absorbed-flux
    record so that sun/shade flips and load offsets are applied to the
    radiant flux, not to a temperature;
  * an air-temperature anomaly per route point, applied with the vapour
    pressure held fixed (a shaded pocket is cooler, not drier).

Clothing is resolved from the UNPERTURBED walk so every perturbation of a
route wears the same ensemble as its reference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import clothing_profiles
from generate_route import load_routes_directory
from microclimate_field import EnvironmentField
from weather_provider import WeatherProvider

stage09 = import_module("09_route_thermal_stress_jos3")
SIGMA = 5.670374419e-8


def projected_area_factor_standing(elevation_deg):
    """Fanger (1972) f_p for a standing person, as stage 05 uses it."""
    b = np.maximum(np.asarray(elevation_deg, dtype=float), 0.0)
    return 0.308 * np.cos(np.deg2rad(b * (0.998 - b * b / 50000.0)))


def saturation_vapour_pressure_hpa(ta_c):
    ta_c = np.asarray(ta_c, dtype=float)
    return 6.112 * np.exp(17.62 * ta_c / (243.12 + ta_c))


class AnomalyEnvironment:
    """Wrap an EnvironmentField so air temperature carries a per-route-point
    anomaly, with vapour pressure (not relative humidity) held fixed."""

    def __init__(self, base, route_xy, anomaly_c):
        self.base = base
        self.tree = cKDTree(np.asarray(route_xy, dtype=float))
        self.anomaly = np.asarray(anomaly_c, dtype=float)

    def sample(self, xyz, hours):
        s = self.base.sample(xyz, hours)
        pts = np.atleast_2d(np.asarray(xyz, dtype=float))[:, :2]
        _, idx = self.tree.query(pts)
        da = self.anomaly[idx]
        ta = np.asarray(s.air_temperature_c, dtype=float)
        rh = np.asarray(s.relative_humidity_pct, dtype=float)
        e = rh / 100.0 * saturation_vapour_pressure_hpa(ta)
        ta_new = ta + da
        rh_new = np.clip(100.0 * e / saturation_vapour_pressure_hpa(ta_new), 1.0, 100.0)
        return replace(s, air_temperature_c=ta_new, relative_humidity_pct=rh_new)

    def describe(self):
        return self.base.describe() + " + per-point air-temperature anomaly"


@dataclass
class RouteGeometry:
    route_id: int
    xy: np.ndarray          # resampled (n, 2)
    dist_m: np.ndarray      # cumulative along-route distance (n,)
    nearest: np.ndarray     # index into the stage-05 path points (n,)
    length_m: float


class CampusCase:
    """Stage-05 products, routes, weather and the JOS-3 subject of one case."""

    def __init__(self, root: Path, case: str = "MMC", *, spacing_m: float = 2.0,
                 walking_speed_ms: float = 1.3, activity_par="auto",
                 equilibration_min: float = 10.0):
        self.root = Path(root)
        self.case = case
        self.spacing_m = float(spacing_m)
        self.walking_speed_ms = float(walking_speed_ms)
        self.equilibration_min = float(equilibration_min)
        case_json = json.loads((self.root / "input" / case / "case.json").read_text())
        crs = case_json["coordinates"]["project_crs"]
        origin = (case_json["coordinates"]["local_origin_x"],
                  case_json["coordinates"]["local_origin_y"])
        routes = load_routes_directory(self.root / "input" / case / "routes",
                                       expected_project_crs=crs,
                                       expected_origin=origin)
        mrt_dir = self.root / "run_output" / case / "mrt_facet_out"
        self.xyz = np.load(mrt_dir / "path_xyz.npy")
        self.tmrt = np.load(mrt_dir / "tmrt_matrix_C.npy")
        self.tau = np.load(mrt_dir / "direct_transmission_matrix.npy")
        times_df = pd.read_csv(mrt_dir / "times.csv", parse_dates=["time"])
        self.time_hours = np.array([t.hour + t.minute / 60 + t.second / 3600
                                    for t in times_df["time"]])
        self.elevation_deg = times_df["elevation_deg"].to_numpy(float)
        self.dni = times_df["DNI_Wm2"].to_numpy(float)
        flux = np.load(mrt_dir / "radiant_flux_contributions.npz")
        meta = json.loads((mrt_dir / "radiant_flux_contributions_metadata.json").read_text())
        self.person_emissivity = float(meta["person_longwave_emissivity"])
        self.person_absorptivity = float(meta["person_shortwave_absorptivity"])
        self.total_flux = flux["total_absorbed_radiant_flux_Wm2"]
        self.sw_direct = flux["sw_direct_absorbed_Wm2"]
        # vegetation share of the sky obstruction above each path point
        svf_b = np.load(mrt_dir / "svf_building_only.npy")
        svf_e = np.load(mrt_dir / "svf_effective.npy")
        self.vegetation_sky_block = np.clip((svf_b - svf_e) / np.maximum(svf_b, 1e-6), 0, 1)
        self.tree = cKDTree(self.xyz[:, :2])
        self.weather = WeatherProvider(csv_path=self.root / "input" / case / "weather" / "weather.csv")
        self.environment = EnvironmentField(self.weather)
        provenance = json.loads((self.root / "run_output" / case / "viz" / "route_jos3"
                                 / "clothing_provenance.json").read_text())
        self.clothing_provenance = provenance
        self.clothing_config = clothing_profiles.load_config(provenance.get("config_file"))
        self.subject = stage09.Subject(1.72, 74.0, 30, "male", 15.0, 2.59, 0.0)
        self.activity_par = stage09.jos3_protocol.resolve_activity_ratio(
            activity_par, self.subject.make_model(), self.walking_speed_ms,
            self.subject.weight)
        self.routes = {}
        for route in routes:
            xy, dist = self._resample(np.asarray(route["xy"], float))
            _, nearest = self.tree.query(xy)
            self.routes[int(route["route_id"])] = RouteGeometry(
                int(route["route_id"]), xy, dist, nearest, float(dist[-1]))
        self.stage09_summary = pd.read_csv(
            self.root / "run_output" / case / "viz" / "route_jos3"
            / "route_ranking_summary.csv").set_index("route_id")

    def _resample(self, xy):
        d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
        s = np.arange(0.0, d[-1], self.spacing_m)
        if d[-1] - s[-1] > 1e-6:
            s = np.append(s, d[-1])
        return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])]), s

    # ------------------------------------------------------------------
    def arrival_hours(self, route: RouteGeometry, departure_hour: float):
        return departure_hour + route.dist_m / self.walking_speed_ms / 3600.0

    def clothing(self, route: RouteGeometry, departure_hour: float):
        arrival = self.arrival_hours(route, departure_hour)
        conditions = self.environment.sample(
            np.column_stack([route.xy, self.xyz[route.nearest, 2]]), arrival % 24.0)
        mean_hour = float(np.mean(arrival)) % 24.0
        daytime = float(np.interp(mean_hour, self.time_hours, self.elevation_deg,
                                  period=24.0)) > 0.0
        clo, prov = clothing_profiles.resolve(
            self.clothing_provenance["requested"],
            air_temp_c=float(np.mean(conditions.air_temperature_c)),
            is_daytime=daytime, wind_ms=float(np.mean(conditions.wind_speed_ms)),
            climate=self.clothing_provenance["climate"], config=self.clothing_config)
        return clo, prov.get("ensemble", self.clothing_provenance["requested"])

    def route_tmrt_matrix(self, route: RouteGeometry):
        """Baseline Tmrt (nt, n_pts) for the route's resampled points."""
        return self.tmrt[:, route.nearest]

    def route_flux(self, route: RouteGeometry):
        """Baseline total absorbed flux and its beam part, (nt, n_pts)."""
        return (self.total_flux[:, route.nearest].astype(float),
                self.sw_direct[:, route.nearest].astype(float))

    def beam_if_sunlit(self):
        """Absorbed beam a fully sunlit standing body would receive, per time step."""
        return (self.person_absorptivity * projected_area_factor_standing(self.elevation_deg)
                * self.dni)

    def tmrt_from_flux(self, flux):
        return (np.maximum(flux, 1.0) / (self.person_emissivity * SIGMA)) ** 0.25 - 273.15

    def walk(self, route: RouteGeometry, departure_hour: float, *, clo=None,
             tmrt_matrix=None, ta_anomaly_c=None, radiation_neutral=False,
             context="campus"):
        """One JOS-3 walk; returns stage-09's trace dict."""
        arrival = self.arrival_hours(route, departure_hour)
        if clo is None:
            clo, _ = self.clothing(route, departure_hour)
        if tmrt_matrix is None:
            mrt = stage09.MrtField(self.xyz, self.tmrt, self.time_hours, self.tau)
            nearest = route.nearest
        else:
            tmrt_matrix = np.asarray(tmrt_matrix, dtype=float)
            if tmrt_matrix.shape != (len(self.time_hours), len(route.xy)):
                raise ValueError("replacement Tmrt matrix must be (n_times, n_route_points)")
            mrt = stage09.MrtField(
                np.column_stack([route.xy, self.xyz[route.nearest, 2]]),
                tmrt_matrix, self.time_hours, self.tau[:, route.nearest])
            nearest = np.arange(len(route.xy))
        environment = self.environment
        if ta_anomaly_c is not None:
            environment = AnomalyEnvironment(self.environment, route.xy, ta_anomaly_c)
        return stage09.simulate_walk(route.xy, arrival, nearest, mrt, environment,
                                     self.subject, clo, self.activity_par,
                                     self.equilibration_min, context,
                                     radiation_neutral=radiation_neutral)

    def check_against_stage09(self, departure_hour: float = 13.0, tolerance=0.003):
        """The unperturbed walk must reproduce stage 09 at its own departure."""
        worst = 0.0
        for rid, route in self.routes.items():
            rise = self.walk(route, departure_hour)["final_tcore_rise_c"]
            ref = float(self.stage09_summary.loc[rid, "final_tcore_rise_c"])
            worst = max(worst, abs(rise - ref))
            if abs(rise - ref) > tolerance:
                raise SystemExit(f"route {rid}: {rise:.4f} vs stage 09 {ref:.4f}")
        return worst
