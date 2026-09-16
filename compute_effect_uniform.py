#!/usr/bin/env python3
"""compute_effect_uniform.py -- error introduced by uniform Ta and vapor
pressure in JOS-3 core-temperature rise on the six Lisbon daytime walks.

Scientific question
-------------------
How much error in JOS-3-predicted core-temperature rise is introduced by
replacing the REAL, time-varying air temperature and vapor pressure measured
during a pedestrian walk with uniform values, while retaining the same
TREC-Route simulated, time-varying MRT exposure?

Benchmark (per route)
    REAL measured Ta(t) and vapor pressure e(t) from the mobile campaign,
    REAL measured wind speed WS(t), the recorded route timestamps and pace,
    and the TREC-Route simulated MRT(t) sampled at the recorded arrival
    times -> JOS-3 -> DeltaTcore_real.

Uniform sensitivity cases (per route, per grid point)
    Identical MRT(t), wind(t), timing, subject, activity, clothing, and
    initialization; ONLY Ta -> Ta_uniform and e -> e_uniform for the whole
    walk -> JOS-3 -> DeltaTcore_uniform.

    signed_error_C = DeltaTcore_uniform - DeltaTcore_real
        positive: uniform meteorology predicts MORE core-temperature rise.

Uniform-value grid (route-specific by default)
    A practitioner who replaces the true along-route variation with one
    uniform value would base it on LOCAL information for that walk -- a
    route-mean or a spot measurement -- not on one city-wide number.  The
    default grid is therefore unique to each route: mean + k*sigma of that
    route's own measured series, k in {-2, -1, 0, +1, +2} for both Ta and
    vapor pressure (sigma = population standard deviation along the walk).
    The k-offsets are shared by all routes, so the six heatmaps remain
    directly comparable in offset space and use one common color scale.
    Passing explicit --ta-values/--vp-values (or --ta-min/max/step ...)
    instead evaluates one common ABSOLUTE grid for all routes.

Provenance of the variables (kept explicit throughout):
    MEASURED   Ta, vapor pressure (from measured RH and Ta), wind, timing.
    SIMULATED  MRT from the existing TREC-Route radiation output
               (never recomputed here, never a function of Ta_uniform).
    MODELED    physiological response from the project's JOS-3 workflow.

Moisture handling: the sensitivity variable is VAPOR PRESSURE in hPa.  The
Lisbon instruments record relative humidity (HRel, %) and air temperature;
JOS-3 (via pythermalcomfort) takes RELATIVE HUMIDITY IN PERCENT (see
physical_checks.py).  Both directions of the conversion use the project's
existing Magnus formulation from sensitivity.py:
    es(Ta) = 6.112 * exp(17.67*Ta/(Ta+243.5))   [hPa]
    e_real(t)  = HRel(t)/100 * es(Ta(t))
    RH_uniform = 100 * e_uniform / es(Ta_uniform)   (rejected if > 100 %)

This is an analysis driver around the EXISTING framework: route loading and
recorded timing come from generate_route.py, JOS-3 construction/subject
resolution and the vapor-pressure conversion from sensitivity.py, the input
guards from physical_checks.py, and the MRT sampling convention (nearest
route point + 24 h periodic time interpolation) from stage 09.  Core
temperature is the stage-09 definition: BSA-weighted mean of the 17 JOS-3
segment core temperatures; the rise is (end - start after equilibration).

Only new persistent outputs are created, all under run_output/impact_uniform/.

Run:
    python3 compute_effect_uniform.py                     # per-route mu+k*sigma
    python3 compute_effect_uniform.py --offsets -1 0 1    # narrower offsets
    python3 compute_effect_uniform.py --ta-values 28 30 32 --vp-values 14 18 22
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr

from case_config import load_case
from generate_route import load_routes_directory, route_arrival_schedule
from physical_checks import check_jos3_inputs
from sensitivity import (initialize_jos3_model, resolve_subject,
                         vapor_pressure_to_rh_pct)
from subject_profiles import PROFILES

VP_UNIT = "hPa"
EXPECTED_CASES = 6
MRT_SNAP_TOLERANCE_M = 5.0
PNG_DPI = 300


def saturation_vapor_pressure_hpa(ta_c: np.ndarray) -> np.ndarray:
    """Magnus saturation vapor pressure, identical to sensitivity.py."""
    ta_c = np.asarray(ta_c, dtype=float)
    return 6.112 * np.exp(17.67 * ta_c / (ta_c + 243.5))


def offset_label(k: float) -> str:
    """Human label for a mean + k*sigma grid level, e.g. 'μ−2σ'."""
    if k == 0:
        return "μ"
    magnitude = abs(k)
    magnitude_text = "" if magnitude == 1 else f"{magnitude:g}"
    return f"μ{'+' if k > 0 else '−'}{magnitude_text}σ"


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------
@dataclass
class MeasuredRoute:
    """One Lisbon daytime walk: measured meteorology + simulated MRT."""
    case_name: str
    route_id: int
    label: str                      # "Route 1" .. "Route 6" for figures
    directory_label: str            # "route_01" .. "route_06"
    xy: np.ndarray
    arrival_hour: np.ndarray        # recorded, unwrapped local hours
    step_duration_s: np.ndarray
    ta_measured_c: np.ndarray
    rh_measured_pct: np.ndarray
    vp_measured_hpa: np.ndarray
    wind_measured_ms: np.ndarray
    mrt_simulated_c: np.ndarray
    start_datetime: str
    end_datetime: str
    duration_min: float
    distance_m: float
    mean_walking_speed_ms: float
    timing_source: str
    measurement_csv: str
    mrt_dir: str
    mrt_snap_max_m: float
    # JOS-3 segment clothing, as stage 09 resolved it for this walk. It cancels
    # in the benchmark-minus-uniform difference but sets the absolute level of
    # DeltaTcore, which is reported, so it is carried explicitly rather than
    # left at the library default of nude.
    segment_clo: list[float] = field(default_factory=list)
    benchmark: dict = field(default_factory=dict)

    @property
    def ta_mean_c(self) -> float:
        return float(self.ta_measured_c.mean())

    @property
    def ta_std_c(self) -> float:
        return float(self.ta_measured_c.std())

    @property
    def vp_mean_hpa(self) -> float:
        return float(self.vp_measured_hpa.mean())

    @property
    def vp_std_hpa(self) -> float:
        return float(self.vp_measured_hpa.std())


def discover_lisbon_day_cases(input_root: Path, run_output_root: Path,
                              mrt_dirname: str) -> list[dict]:
    """Resolve exactly the six Lisbon daytime validation walks.

    Nighttime routes, MMC, and any other case are excluded by construction:
    only input/lisbon<N> cases are considered and only each case's declared
    day route is used.
    """
    cases = []
    for case_dir in sorted(input_root.glob("lisbon*")):
        match = re.fullmatch(r"lisbon(\d+)", case_dir.name)
        if not match or not case_dir.is_dir():
            continue
        case = load_case(case_dir)
        day_route_id = int(case["defaults"].get("day_route_id", 1))
        measurement_csv = (case_dir / "measurements"
                           / f"route_{day_route_id}_day_experimental_measurements.csv")
        mrt_dir = run_output_root / case_dir.name / mrt_dirname
        for path, what in ((measurement_csv, "day measurement CSV"),
                           (mrt_dir / "tmrt_matrix_C.npy", "simulated MRT matrix"),
                           (mrt_dir / "path_xyz.npy", "MRT route points"),
                           (mrt_dir / "times.csv", "MRT time axis")):
            if not path.is_file():
                raise FileNotFoundError(
                    f"case {case_dir.name}: missing {what}: {path}")
        cases.append({"number": int(match.group(1)), "case": case,
                      "day_route_id": day_route_id,
                      "measurement_csv": measurement_csv, "mrt_dir": mrt_dir})
    cases.sort(key=lambda item: item["number"])
    names = [f"lisbon{item['number']}" for item in cases]
    if len(cases) != EXPECTED_CASES:
        raise FileNotFoundError(
            f"expected the {EXPECTED_CASES} Lisbon daytime validation cases "
            f"under {input_root}, resolved {len(cases)}: {names}")
    return cases


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_measured_route(entry: dict, position: int) -> MeasuredRoute:
    """Load one daytime walk, aligning route timing, measurements, and MRT."""
    case = entry["case"]
    case_name = case["case_dir"].name
    routes = load_routes_directory(
        case["files"]["routes_dir"],
        expected_project_crs=case["project_crs"],
        expected_origin=(case["local_origin_x"], case["local_origin_y"]))
    selected = [route for route in routes
                if route["route_id"] == entry["day_route_id"]]
    if len(selected) != 1:
        raise ValueError(f"{case_name}: day route {entry['day_route_id']} "
                         f"not found among {[r['route_id'] for r in routes]}")
    route = selected[0]

    # Recorded campaign timing is mandatory for this study -- a synthetic
    # distance/speed schedule would defeat the "real walk" benchmark.
    arrival_hour, timing_source = route_arrival_schedule(route, 0.0, 1.0)
    if timing_source != "recorded_device_timestamps":
        raise ValueError(
            f"{case_name} route {route['route_id']}: expected recorded "
            f"device timestamps, got timing source {timing_source!r}")
    step_duration_s = np.diff(arrival_hour) * 3600.0

    measured = pd.read_csv(entry["measurement_csv"])
    frame = route["frame"]
    if len(measured) != len(frame):
        raise ValueError(f"{case_name}: measurement rows ({len(measured)}) "
                         f"!= route points ({len(frame)})")
    if not np.array_equal(measured["seq"].to_numpy(), frame["seq"].to_numpy()):
        raise ValueError(f"{case_name}: measurement/route seq mismatch")
    xy = np.asarray(route["xy"], dtype=float)[:, :2]
    if not np.allclose(measured[["x_local_m", "y_local_m"]].to_numpy(float),
                       xy, atol=1e-6):
        raise ValueError(f"{case_name}: measurement/route coordinate mismatch")
    route_utc = pd.to_datetime(frame["timestamp_utc"], format="ISO8601")
    measured_utc = pd.to_datetime(measured["timestamp_utc_refined"],
                                  format="ISO8601")
    offset_s = (measured_utc - route_utc).dt.total_seconds().abs()
    if float(offset_s.max()) > 1.0:
        raise ValueError(f"{case_name}: measurement/route timestamps differ "
                         f"by up to {offset_s.max():.1f} s")

    ta = measured["AirTemp"].to_numpy(float)
    rh = measured["HRel"].to_numpy(float)
    wind = measured["WS"].to_numpy(float)
    for name, values in (("AirTemp", ta), ("HRel", rh), ("WS", wind)):
        if not np.isfinite(values).all():
            raise ValueError(f"{case_name}: non-finite measured {name}")
    vp = rh / 100.0 * saturation_vapor_pressure_hpa(ta)

    # Simulated MRT along the walk, stage-09 convention: nearest MRT route
    # point in XY, then 24 h periodic interpolation to the recorded arrival
    # time.  Computed ONCE; benchmark and every uniform case reuse this array.
    mrt_dir = Path(entry["mrt_dir"])
    mrt_xyz = np.load(mrt_dir / "path_xyz.npy")
    tmrt_matrix = np.load(mrt_dir / "tmrt_matrix_C.npy")
    times = pd.read_csv(mrt_dir / "times.csv")
    stamps = pd.to_datetime(times["time"])
    time_hours = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0
                           for t in stamps], dtype=float)
    if tmrt_matrix.shape != (len(time_hours), len(mrt_xyz)):
        raise ValueError(f"{case_name}: inconsistent MRT array dimensions")
    snap_distance, nearest = cKDTree(mrt_xyz[:, :2]).query(xy)
    if float(np.max(snap_distance)) > MRT_SNAP_TOLERANCE_M:
        raise ValueError(
            f"{case_name}: route point {int(np.argmax(snap_distance))} is "
            f"{np.max(snap_distance):.2f} m from the nearest simulated MRT "
            f"point (tolerance {MRT_SNAP_TOLERANCE_M} m)")
    mrt = np.array([
        float(np.interp(hour % 24.0, time_hours, tmrt_matrix[:, index],
                        period=24.0))
        for hour, index in zip(arrival_hour, nearest)])
    if not np.isfinite(mrt).all():
        raise ValueError(f"{case_name}: non-finite simulated MRT along route")

    distance_m = float(frame["cumdist_m"].iloc[-1])
    duration_min = float((arrival_hour[-1] - arrival_hour[0]) * 60.0)
    return MeasuredRoute(
        case_name=case_name, route_id=int(route["route_id"]),
        label=f"Route {position}", directory_label=f"route_{position:02d}",
        xy=xy, arrival_hour=arrival_hour, step_duration_s=step_duration_s,
        ta_measured_c=ta, rh_measured_pct=rh, vp_measured_hpa=vp,
        wind_measured_ms=wind, mrt_simulated_c=mrt,
        start_datetime=str(frame["timestamp_local"].iloc[0]),
        end_datetime=str(frame["timestamp_local"].iloc[-1]),
        duration_min=duration_min, distance_m=distance_m,
        mean_walking_speed_ms=distance_m / (duration_min * 60.0),
        timing_source=timing_source,
        measurement_csv=str(entry["measurement_csv"]), mrt_dir=str(mrt_dir),
        mrt_snap_max_m=float(np.max(snap_distance)),
        segment_clo=stage09_segment_clo(Path(mrt_dir).parent.parent, case_name,
                                        int(route["route_id"])))


# ---------------------------------------------------------------------------
# JOS-3 engine (stage-09 conventions, shared by benchmark and uniform cases)
# ---------------------------------------------------------------------------
def stage09_segment_clo(run_output_root: Path, case_name: str,
                        route_id: int) -> list[float]:
    """Segment clothing stage 09 resolved for this walk.

    Clothing cancels in the benchmark-minus-uniform difference this study
    reports, but it sets the absolute level of DeltaTcore, which is also
    reported and compared against a 0.10 degC criterion, so the walk is
    dressed as stage 09 dressed it rather than left nude.
    """
    path = (Path(run_output_root) / case_name / "viz" / "route_jos3"
            / "clothing_provenance.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"{case_name}: stage-09 clothing provenance not found at {path}; "
            "run stage 09 for this case first")
    record = json.loads(path.read_text())
    for entry in record.get("routes", []):
        if int(entry.get("route_id", -1)) == int(route_id):
            return [float(v) for v in entry["segment_clo"]]
    return []


def run_jos3_walk(route: MeasuredRoute, ta_series: np.ndarray,
                  rh_series: np.ndarray, subject: dict,
                  args: argparse.Namespace, context: str) -> dict:
    """Simulate one walk and return core temperatures plus the applied inputs.

    Everything except the supplied Ta/RH series is taken from the route
    object, so the simulated MRT, measured wind, and recorded timing are
    identical by construction for the benchmark and every uniform case; the
    returned copies of the applied inputs let the caller verify that
    machine-precision identity explicitly.
    """
    ta_series = np.asarray(ta_series, dtype=float)
    rh_series = np.asarray(rh_series, dtype=float)
    mrt = route.mrt_simulated_c
    wind = route.wind_measured_ms
    n_points = len(route.arrival_hour)
    if not (len(ta_series) == len(rh_series) == n_points):
        raise ValueError(f"{context}: Ta/RH series length mismatch")
    check_jos3_inputs(ta_series, mrt, wind, rh_series, context)

    model, weights = initialize_jos3_model(subject, args.activity_par,
                                           route.segment_clo or None)

    def core_c() -> float:
        value = float(np.sum(np.asarray(model.t_core) * weights))
        if not np.isfinite(value):
            raise ValueError(f"{context}: non-finite core temperature")
        return value

    # Stage-09 initialization: equilibrate at the walk's starting conditions
    # so JOS-3's default state cannot create a startup transient.  Every run
    # begins from a FRESH model followed by this same procedure; no case
    # inherits another case's physiological state.
    model.tdb, model.tr = float(ta_series[0]), float(mrt[0])
    model.rh, model.v = float(rh_series[0]), float(wind[0])
    if args.equilibration_min > 0:
        model.simulate(times=int(args.equilibration_min), dtime=60,
                       output=False)
    start_core = core_c()

    final_core = start_core
    for j in range(n_points):
        model.tdb, model.tr = float(ta_series[j]), float(mrt[j])
        model.rh, model.v = float(rh_series[j]), float(wind[j])
        if j:
            dt_s = float(route.step_duration_s[j - 1])
            if not np.isfinite(dt_s) or dt_s <= 0:
                raise ValueError(f"{context}: invalid step duration {dt_s}")
            model.simulate(times=1, dtime=dt_s, output=False)
        final_core = core_c()

    return {"tcore_start_c": start_core, "tcore_end_c": final_core,
            "delta_tcore_c": final_core - start_core,
            "applied_mrt_c": mrt.copy(), "applied_wind_ms": wind.copy(),
            "applied_time_hour": route.arrival_hour.copy(),
            "applied_ta_c": ta_series.copy(), "applied_rh_pct": rh_series.copy()}


def verify_only_ta_and_moisture_differ(benchmark: dict, uniform: dict,
                                       context: str) -> None:
    """Section-14 critical check: MRT, wind, and timing identical to machine
    precision between the benchmark and a uniform case; only Ta and moisture
    are allowed to differ."""
    for name in ("applied_mrt_c", "applied_wind_ms", "applied_time_hour"):
        if not np.array_equal(benchmark[name], uniform[name]):
            raise RuntimeError(
                f"{context}: {name} differs between benchmark and uniform "
                "case -- only Ta and vapor pressure may change")
    if np.array_equal(benchmark["applied_ta_c"], uniform["applied_ta_c"]) \
            and np.array_equal(benchmark["applied_rh_pct"],
                               uniform["applied_rh_pct"]):
        raise RuntimeError(f"{context}: uniform case applied exactly the "
                           "measured Ta and humidity; nothing was perturbed")
    if np.ptp(uniform["applied_ta_c"]) != 0.0 \
            or np.ptp(uniform["applied_rh_pct"]) != 0.0:
        raise RuntimeError(f"{context}: uniform case Ta/RH are not uniform")


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformCase:
    """One grid level pair for one route (actual values + grid keys)."""
    ta_c: float
    vp_hpa: float
    ta_key: float                   # sigma offset, or Ta value in absolute mode
    vp_key: float


def build_absolute_axis(explicit: list[float] | None, minimum: float | None,
                        maximum: float | None, step: float | None,
                        levels: int, name: str) -> np.ndarray:
    if explicit:
        values = np.asarray(sorted(set(float(v) for v in explicit)))
    else:
        lo, hi = float(minimum), float(maximum)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"invalid {name} range: {lo} .. {hi}")
        if step is not None:
            if step <= 0:
                raise ValueError(f"{name} step must be positive")
            values = np.round(np.arange(lo, hi + step / 2.0, step), 6)
        else:
            values = np.round(np.linspace(lo, hi, levels), 2)
    if len(values) < 2:
        raise ValueError(f"{name} grid needs at least two values")
    return values


def build_route_cases(route: MeasuredRoute, args: argparse.Namespace,
                      grid_mode: str, offsets: np.ndarray) -> list[UniformCase]:
    """The uniform grid for ONE route.

    route_sigma mode: each uniform level is informed only by that route's
    own measured statistics -- mean + k*sigma -- reflecting a practitioner
    who has a local average or spot value but no along-route variation.
    absolute mode: one common explicit grid shared by all routes.
    """
    if grid_mode == "route_sigma":
        if route.ta_std_c <= 0 or route.vp_std_hpa <= 0:
            raise ValueError(
                f"{route.case_name}: zero measured variability; use the "
                "explicit absolute grid options instead")
        return [UniformCase(route.ta_mean_c + k_ta * route.ta_std_c,
                            route.vp_mean_hpa + k_vp * route.vp_std_hpa,
                            float(k_ta), float(k_vp))
                for k_ta in offsets for k_vp in offsets]
    ta_values = build_absolute_axis(args.ta_values, args.ta_min, args.ta_max,
                                    args.ta_step, args.ta_levels, "Ta")
    vp_values = build_absolute_axis(args.vp_values, args.vp_min, args.vp_max,
                                    args.vp_step, args.vp_levels,
                                    "vapor pressure")
    if np.any(vp_values < 0):
        raise ValueError("vapor pressure cannot be negative")
    return [UniformCase(float(ta), float(vp), float(ta), float(vp))
            for ta in ta_values for vp in vp_values]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def draw_heatmap(ax, x_labels, y_labels, z, norm, cmap, tick_fontsize=9):
    image = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(x_labels)), x_labels, fontsize=tick_fontsize)
    ax.set_yticks(range(len(y_labels)), y_labels, fontsize=tick_fontsize)
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            if np.isfinite(z[i, j]):
                fraction = float(norm(z[i, j]))
                color = "white" if fraction < 0.2 or fraction > 0.8 else "black"
                ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color=color)
            else:
                ax.text(j, i, "invalid", ha="center", va="center",
                        fontsize=7, color="dimgray")
    return image


def route_grid(rows: pd.DataFrame, column: str, ta_keys: np.ndarray,
               vp_keys: np.ndarray) -> np.ndarray:
    pivot = rows.pivot(index="vp_grid_key", columns="Ta_grid_key",
                       values=column)
    return pivot.reindex(index=vp_keys, columns=ta_keys).to_numpy(float)


def route_tick_labels(route: MeasuredRoute, cases: list[UniformCase],
                      grid_mode: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    ta_keys = np.array(sorted({case.ta_key for case in cases}))
    vp_keys = np.array(sorted({case.vp_key for case in cases}))
    if grid_mode == "route_sigma":
        x_labels = [f"{offset_label(k)}\n"
                    f"{route.ta_mean_c + k * route.ta_std_c:.1f}"
                    for k in ta_keys]
        y_labels = [f"{offset_label(k)}\n"
                    f"{route.vp_mean_hpa + k * route.vp_std_hpa:.1f}"
                    for k in vp_keys]
    else:
        x_labels = [f"{k:g}" for k in ta_keys]
        y_labels = [f"{k:g}" for k in vp_keys]
    return x_labels, y_labels, ta_keys, vp_keys


def save_route_heatmaps(route: MeasuredRoute, rows: pd.DataFrame,
                        cases: list[UniformCase], grid_mode: str,
                        signed_norm, abs_max, out_dir: Path) -> None:
    x_labels, y_labels, ta_keys, vp_keys = route_tick_labels(
        route, cases, grid_mode)
    signed = route_grid(rows, "signed_error_C", ta_keys, vp_keys)
    absolute = route_grid(rows, "absolute_error_C", ta_keys, vp_keys)
    statistics = (f"measured μ(Ta)={route.ta_mean_c:.2f} °C, "
                  f"σ(Ta)={route.ta_std_c:.2f} °C; "
                  f"μ(e)={route.vp_mean_hpa:.2f}, "
                  f"σ(e)={route.vp_std_hpa:.2f} {VP_UNIT}"
                  if grid_mode == "route_sigma" else "common absolute grid")
    for z, norm, cmap, label, filename in (
            (signed, signed_norm, "RdBu_r",
             "Signed core-rise error, uniform - real [°C]",
             "signed_error_heatmap.png"),
            (absolute, plt.Normalize(vmin=0.0, vmax=abs_max), "viridis",
             "Absolute core-rise error [°C]", "absolute_error_heatmap.png")):
        fig, ax = plt.subplots(figsize=(8.6, 6.6))
        image = draw_heatmap(ax, x_labels, y_labels, z, norm, cmap)
        ax.set_xlabel("Uniform air temperature, Ta [°C]")
        ax.set_ylabel(f"Uniform vapor pressure, e [{VP_UNIT}]")
        ax.set_title(f"{route.label} ({route.case_name}): "
                     f"uniform-meteorology error vs real measured walk\n"
                     f"ΔTcore_real = {route.benchmark['delta_tcore_c']:+.3f} °C, "
                     f"{route.duration_min:.0f} min, "
                     f"{route.start_datetime[:16]}\n{statistics}",
                     fontsize=11)
        fig.colorbar(image, ax=ax).set_label(label)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)


def save_combined_figure(routes: list[MeasuredRoute], results: pd.DataFrame,
                         cases_by_route: dict[str, list[UniformCase]],
                         grid_mode: str, signed_norm, abs_max,
                         figures_dir: Path) -> dict[str, Path]:
    subtitle = ("route-specific uniform values μ+kσ from each walk's own "
                "measured series" if grid_mode == "route_sigma"
                else "one common absolute uniform grid")
    paths = {}
    for column, norm, cmap, label, filename in (
            ("signed_error_C", signed_norm, "RdBu_r",
             "Signed core-rise error, ΔTcore_uniform − ΔTcore_real [°C]",
             "signed_error_all_routes.png"),
            ("absolute_error_C", plt.Normalize(vmin=0.0, vmax=abs_max),
             "viridis", "Absolute core-rise error [°C]",
             "absolute_error_all_routes.png")):
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.4))
        image = None
        for ax, route in zip(axes.ravel(), routes):
            rows = results[results["route_label"] == route.label]
            x_labels, y_labels, ta_keys, vp_keys = route_tick_labels(
                route, cases_by_route[route.label], grid_mode)
            z = route_grid(rows, column, ta_keys, vp_keys)
            image = draw_heatmap(ax, x_labels, y_labels, z, norm, cmap,
                                 tick_fontsize=8)
            ax.set_title(f"{route.label} ({route.case_name}) · "
                         f"ΔTcore_real {route.benchmark['delta_tcore_c']:+.3f} °C",
                         fontsize=11)
        for ax in axes[-1]:
            ax.set_xlabel("Uniform air temperature, Ta [°C]")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Uniform vapor pressure, e [{VP_UNIT}]")
        fig.suptitle(
            "Error from uniform Ta and vapor pressure vs the real measured "
            f"Lisbon walks — {subtitle}\n(same TREC-Route simulated MRT(t), "
            "measured wind and recorded pace in every run)", fontsize=13)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.015)
        colorbar.set_label(label)
        path = figures_dir / filename
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        plt.close(fig)
        paths[column] = path
    return paths


# ---------------------------------------------------------------------------
# Ranking analysis across the six routes
# ---------------------------------------------------------------------------
def ranking_analysis(routes: list[MeasuredRoute], results: pd.DataFrame,
                     grid_mode: str) -> pd.DataFrame:
    """Cross-route ranking fidelity per grid level.

    Grid levels align across routes by sigma offset (route_sigma mode) or by
    absolute value (absolute mode).  In route_sigma mode this compares the
    deployment scenario in which every route uses its own locally informed
    uniform value at the same statistical offset.
    """
    real = {route.label: route.benchmark["delta_tcore_c"] for route in routes}
    labels = [route.label for route in routes]
    real_values = np.array([real[label] for label in labels])
    real_best = labels[int(np.argmin(real_values))]
    key_names = (["Ta_offset_sigma", "vp_offset_sigma"]
                 if grid_mode == "route_sigma"
                 else ["Ta_uniform_C", "vp_uniform"])
    rows = []
    for keys, group in results[results["status"] == "ok"].groupby(
            ["Ta_grid_key", "vp_grid_key"]):
        group = group.set_index("route_label").reindex(labels)
        if group["DeltaTcore_uniform_C"].isna().any():
            continue        # a route lost this level to an invalid RH
        uniform_values = group["DeltaTcore_uniform_C"].to_numpy(float)
        rho = spearmanr(real_values, uniform_values)
        tau = kendalltau(real_values, uniform_values)
        rows.append({
            key_names[0]: keys[0], key_names[1]: keys[1],
            "vp_unit": VP_UNIT,
            "real_best_route": real_best,
            "uniform_best_route": labels[int(np.argmin(uniform_values))],
            "best_route_preserved":
                real_best == labels[int(np.argmin(uniform_values))],
            "complete_ranking_preserved": bool(np.array_equal(
                np.argsort(real_values), np.argsort(uniform_values))),
            "spearman_rho": float(rho.statistic),
            "kendall_tau": float(tau.statistic),
            "max_route_absolute_error_C":
                float(group["absolute_error_C"].max()),
            "mean_route_absolute_error_C":
                float(group["absolute_error_C"].mean())})
    return pd.DataFrame(rows).sort_values(key_names)


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uniform Ta/vapor-pressure error study on the six "
                    "Lisbon daytime walks (JOS-3 core-temperature rise)")
    p.add_argument("--input-root", default="input")
    p.add_argument("--run-output-root", default="run_output")
    p.add_argument("--mrt-dirname", default="mrt_facet_out",
                   help="Per-case TREC-Route MRT result directory name "
                        "(default: the authoritative facet-thermal MRT)")
    p.add_argument("--output-dir", default=None,
                   help="Default: <run-output-root>/impact_uniform")
    p.add_argument("--offsets", type=float, nargs="+",
                   default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                   help="Route-specific grid: sigma offsets k applied to "
                        "each walk's own measured mean, uniform = mean + "
                        "k*sigma, for both Ta and vapor pressure "
                        "(default: -2 -1 0 1 2)")
    p.add_argument("--ta-values", type=float, nargs="+", default=None,
                   help="Switch to one common ABSOLUTE grid: explicit "
                        "uniform Ta values [°C]")
    p.add_argument("--vp-values", type=float, nargs="+", default=None,
                   help=f"Absolute grid: uniform vapor pressures [{VP_UNIT}]")
    p.add_argument("--ta-min", type=float, default=None)
    p.add_argument("--ta-max", type=float, default=None)
    p.add_argument("--ta-step", type=float, default=None)
    p.add_argument("--ta-levels", type=int, default=5)
    p.add_argument("--vp-min", type=float, default=None)
    p.add_argument("--vp-max", type=float, default=None)
    p.add_argument("--vp-step", type=float, default=None)
    p.add_argument("--vp-levels", type=int, default=5)
    # Production JOS-3 settings (stage-09 defaults)
    p.add_argument("--equilibration-min", type=float, default=10.0)
    p.add_argument("--activity-par", type=float, default=2.5)
    p.add_argument("--subject-profile", choices=sorted(PROFILES), default=None)
    p.add_argument("--person-height-m", type=float, default=None)
    p.add_argument("--person-weight-kg", type=float, default=None)
    p.add_argument("--person-age", type=int, default=None)
    p.add_argument("--person-sex", choices=("male", "female"), default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    run_output_root = Path(args.run_output_root)
    out_dir = (Path(args.output_dir) if args.output_dir
               else run_output_root / "impact_uniform")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    subject = resolve_subject(args)
    absolute_requested = any(
        value is not None for value in (args.ta_values, args.vp_values,
                                        args.ta_min, args.ta_max, args.ta_step,
                                        args.vp_min, args.vp_max, args.vp_step))
    grid_mode = "absolute_common" if absolute_requested else "route_sigma"
    offsets = np.array(sorted(set(float(k) for k in args.offsets)))
    if grid_mode == "route_sigma" and len(offsets) < 2:
        raise ValueError("--offsets needs at least two distinct values")

    print("Uniform-meteorology impact study (six Lisbon daytime walks)")
    entries = discover_lisbon_day_cases(input_root, run_output_root,
                                        args.mrt_dirname)
    routes = [load_measured_route(entry, position)
              for position, entry in enumerate(entries, 1)]
    for route in routes:
        print(f"  {route.label} = {route.case_name} route {route.route_id}: "
              f"{route.start_datetime} .. {route.end_datetime}, "
              f"{route.duration_min:.1f} min, {route.distance_m:.0f} m, "
              f"{route.mean_walking_speed_ms:.2f} m/s "
              f"[{route.timing_source}], "
              f"Ta {route.ta_measured_c.min():.1f}..{route.ta_measured_c.max():.1f} °C, "
              f"e {route.vp_measured_hpa.min():.1f}..{route.vp_measured_hpa.max():.1f} {VP_UNIT}")

    # ---- Route-specific (or common absolute) uniform grids ----------------
    cases_by_route = {route.label: build_route_cases(route, args, grid_mode,
                                                     offsets)
                      for route in routes}
    cases_per_route = len(next(iter(cases_by_route.values())))
    if grid_mode == "route_sigma":
        print("\nRoute-specific uniform grids "
              f"(mean + k*sigma of each walk's own measured series, "
              f"k = {[f'{k:g}' for k in offsets]}; "
              f"{len(offsets)} x {len(offsets)} = {cases_per_route} cases "
              "per route):")
        for route in routes:
            ta_axis = [route.ta_mean_c + k * route.ta_std_c for k in offsets]
            vp_axis = [route.vp_mean_hpa + k * route.vp_std_hpa
                       for k in offsets]
            print(f"  {route.label}: Ta μ={route.ta_mean_c:.2f} "
                  f"σ={route.ta_std_c:.2f} -> "
                  f"{[f'{v:.1f}' for v in ta_axis]} °C | "
                  f"e μ={route.vp_mean_hpa:.2f} σ={route.vp_std_hpa:.2f} -> "
                  f"{[f'{v:.1f}' for v in vp_axis]} {VP_UNIT}")
    else:
        example = cases_by_route[routes[0].label]
        ta_values = sorted({case.ta_c for case in example})
        vp_values = sorted({case.vp_hpa for case in example})
        print(f"\nCommon absolute uniform grid ({len(ta_values)} Ta x "
              f"{len(vp_values)} e = {cases_per_route} cases per route):")
        print(f"  Ta_uniform  [°C]:   {[f'{v:g}' for v in ta_values]}")
        print(f"  e_uniform   [{VP_UNIT}]:  {[f'{v:g}' for v in vp_values]}")
    invalid_preview = []
    for route in routes:
        for case in cases_by_route[route.label]:
            rh = (100.0 * case.vp_hpa
                  / float(saturation_vapor_pressure_hpa(case.ta_c)))
            if case.vp_hpa < 0 or rh > 100.0:
                invalid_preview.append((route.label, case.ta_c, case.vp_hpa, rh))
    if invalid_preview:
        print(f"  {len(invalid_preview)} physically invalid combination(s) "
              "(RH > 100 % or e < 0) will be flagged and excluded:")
        for label, ta, vp, rh in invalid_preview:
            print(f"    {label}: Ta={ta:.2f} °C, e={vp:.2f} {VP_UNIT} "
                  f"-> RH={rh:.1f} %")

    # ---- Benchmarks: real measured meteorology + simulated MRT ------------
    print("\nReal-meteorology benchmarks (measured Ta/e/wind, recorded pace, "
          "simulated MRT):")
    for route in routes:
        route.benchmark = run_jos3_walk(
            route, route.ta_measured_c, route.rh_measured_pct, subject, args,
            f"benchmark {route.case_name}")
        print(f"  {route.label}: Tcore {route.benchmark['tcore_start_c']:.3f} "
              f"-> {route.benchmark['tcore_end_c']:.3f} °C, "
              f"ΔTcore_real = {route.benchmark['delta_tcore_c']:+.4f} °C")

    # ---- Uniform sensitivity cases ----------------------------------------
    rows = []
    total = sum(len(cases) for cases in cases_by_route.values())
    done = 0
    for route in routes:
        for case in cases_by_route[route.label]:
            done += 1
            base = {
                "route_id": f"{route.case_name}:route_{route.route_id}",
                "route_label": route.label,
                "route_start_datetime": route.start_datetime,
                "route_end_datetime": route.end_datetime,
                "route_duration_min": route.duration_min,
                "route_distance_m": route.distance_m,
                "mean_walking_speed_ms": route.mean_walking_speed_ms,
                "Ta_real_mean_C": route.ta_mean_c,
                "Ta_real_std_C": route.ta_std_c,
                "Ta_real_min_C": float(route.ta_measured_c.min()),
                "Ta_real_max_C": float(route.ta_measured_c.max()),
                "vp_real_mean": route.vp_mean_hpa,
                "vp_real_std": route.vp_std_hpa,
                "vp_real_min": float(route.vp_measured_hpa.min()),
                "vp_real_max": float(route.vp_measured_hpa.max()),
                "vp_unit": VP_UNIT,
                "grid_mode": grid_mode,
                "Ta_offset_sigma": (case.ta_key if grid_mode == "route_sigma"
                                    else np.nan),
                "vp_offset_sigma": (case.vp_key if grid_mode == "route_sigma"
                                    else np.nan),
                "Ta_grid_key": case.ta_key, "vp_grid_key": case.vp_key,
                "Ta_uniform_C": case.ta_c, "vp_uniform": case.vp_hpa,
                "Tcore_start_real_C": route.benchmark["tcore_start_c"],
                "Tcore_end_real_C": route.benchmark["tcore_end_c"],
                "DeltaTcore_real_C": route.benchmark["delta_tcore_c"],
                "mrt_dataset": route.mrt_dir,
                "measured_dataset": route.measurement_csv,
            }
            try:
                if case.vp_hpa < 0:
                    raise ValueError("negative vapor pressure")
                rh_uniform = vapor_pressure_to_rh_pct(case.ta_c, case.vp_hpa)
            except ValueError:
                rows.append({**base, "RH_uniform_equivalent_pct": np.nan,
                             "Tcore_start_uniform_C": np.nan,
                             "Tcore_end_uniform_C": np.nan,
                             "DeltaTcore_uniform_C": np.nan,
                             "signed_error_C": np.nan,
                             "absolute_error_C": np.nan,
                             "status": "invalid_rh_or_vapor_pressure"})
                continue
            n = len(route.arrival_hour)
            context = (f"uniform {route.case_name} Ta={case.ta_c:.2f} "
                       f"e={case.vp_hpa:.2f}")
            uniform = run_jos3_walk(
                route, np.full(n, case.ta_c), np.full(n, rh_uniform),
                subject, args, context)
            verify_only_ta_and_moisture_differ(route.benchmark, uniform,
                                               context)
            signed = (uniform["delta_tcore_c"]
                      - route.benchmark["delta_tcore_c"])
            rows.append({**base, "RH_uniform_equivalent_pct": rh_uniform,
                         "Tcore_start_uniform_C": uniform["tcore_start_c"],
                         "Tcore_end_uniform_C": uniform["tcore_end_c"],
                         "DeltaTcore_uniform_C": uniform["delta_tcore_c"],
                         "signed_error_C": signed,
                         "absolute_error_C": abs(signed), "status": "ok"})
            if done % 10 == 0 or done == total:
                print(f"  uniform case {done}/{total} "
                      f"({route.label}, Ta={case.ta_c:.2f} °C, "
                      f"e={case.vp_hpa:.2f} {VP_UNIT}) "
                      f"error {signed:+.4f} °C")
    results = pd.DataFrame(rows)

    # ---- Tables ------------------------------------------------------------
    results.to_csv(out_dir / "uniform_sensitivity_all_routes.csv", index=False)
    measured_summary = pd.DataFrame([{
        "route_id": f"{r.case_name}:route_{r.route_id}",
        "route_label": r.label, "case_name": r.case_name,
        "start_datetime": r.start_datetime, "end_datetime": r.end_datetime,
        "duration_min": r.duration_min, "distance_m": r.distance_m,
        "mean_walking_speed_ms": r.mean_walking_speed_ms,
        "n_points": len(r.arrival_hour), "timing_source": r.timing_source,
        "Ta_real_mean_C": r.ta_mean_c, "Ta_real_std_C": r.ta_std_c,
        "Ta_real_min_C": float(r.ta_measured_c.min()),
        "Ta_real_max_C": float(r.ta_measured_c.max()),
        "RH_real_mean_pct": float(r.rh_measured_pct.mean()),
        "vp_real_mean": r.vp_mean_hpa, "vp_real_std": r.vp_std_hpa,
        "vp_real_min": float(r.vp_measured_hpa.min()),
        "vp_real_max": float(r.vp_measured_hpa.max()), "vp_unit": VP_UNIT,
        "wind_measured_mean_ms": float(r.wind_measured_ms.mean()),
        "mrt_simulated_mean_C": float(r.mrt_simulated_c.mean()),
        "mrt_simulated_min_C": float(r.mrt_simulated_c.min()),
        "mrt_simulated_max_C": float(r.mrt_simulated_c.max()),
        "mrt_snap_max_m": r.mrt_snap_max_m,
        "Tcore_start_real_C": r.benchmark["tcore_start_c"],
        "Tcore_end_real_C": r.benchmark["tcore_end_c"],
        "DeltaTcore_real_C": r.benchmark["delta_tcore_c"],
        "measured_dataset": r.measurement_csv, "mrt_dataset": r.mrt_dir,
    } for r in routes])
    measured_summary.to_csv(out_dir / "measured_route_summary.csv", index=False)

    valid = results[results["status"] == "ok"]
    route_summary = pd.DataFrame([{
        "route_id": f"{r.case_name}:route_{r.route_id}",
        "route_label": r.label,
        "DeltaTcore_real_C": r.benchmark["delta_tcore_c"],
        "mean_signed_error_C": float(
            valid.loc[valid.route_label == r.label, "signed_error_C"].mean()),
        "mean_absolute_error_C": float(
            valid.loc[valid.route_label == r.label, "absolute_error_C"].mean()),
        "max_absolute_error_C": float(
            valid.loc[valid.route_label == r.label, "absolute_error_C"].max()),
        "error_at_route_mean_C": float(
            valid.loc[(valid.route_label == r.label)
                      & (valid.Ta_grid_key == 0.0)
                      & (valid.vp_grid_key == 0.0), "signed_error_C"].iloc[0])
            if grid_mode == "route_sigma" and not valid.loc[
                (valid.route_label == r.label) & (valid.Ta_grid_key == 0.0)
                & (valid.vp_grid_key == 0.0)].empty else np.nan,
        "rmse_C": float(np.sqrt(np.mean(
            valid.loc[valid.route_label == r.label, "signed_error_C"] ** 2))),
        "valid_cases": int((valid.route_label == r.label).sum()),
        "invalid_cases": int(((results.route_label == r.label)
                              & (results.status != "ok")).sum()),
    } for r in routes])
    route_summary.to_csv(out_dir / "route_summary.csv", index=False)
    ranking = ranking_analysis(routes, results, grid_mode)
    ranking.to_csv(out_dir / "ranking_analysis.csv", index=False)

    # ---- Figures -----------------------------------------------------------
    error_limit = max(float(valid["absolute_error_C"].max()),
                      np.finfo(float).eps)
    signed_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0,
                               vmax=error_limit)
    for route in routes:
        route_dir = out_dir / route.directory_label
        route_dir.mkdir(exist_ok=True)
        route_rows = results[results["route_label"] == route.label]
        route_rows.to_csv(route_dir / "sensitivity_results.csv", index=False)
        save_route_heatmaps(route, route_rows, cases_by_route[route.label],
                            grid_mode, signed_norm, error_limit, route_dir)
    combined = save_combined_figure(routes, results, cases_by_route,
                                    grid_mode, signed_norm, error_limit,
                                    figures_dir)

    # ---- Metadata ----------------------------------------------------------
    metadata = {
        "study": "uniform Ta/vapor-pressure error in JOS-3 core-temperature "
                 "rise on the six Lisbon daytime walks",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "measured Ta(t), measured vapor pressure e(t) (from "
                     "measured HRel and AirTemp), measured wind WS(t), "
                     "recorded campaign timestamps/pace, TREC-Route "
                     "simulated MRT(t)",
        "perturbed_variables_only": ["air temperature", "vapor pressure"],
        "grid_mode": grid_mode,
        "grid_rationale": ("route-specific: uniform value informed only by "
                           "each walk's own measured statistics, "
                           "uniform = mean + k*sigma (population sigma along "
                           "the walk)" if grid_mode == "route_sigma"
                           else "one common absolute grid for all routes"),
        "sigma_offsets": [float(k) for k in offsets]
                         if grid_mode == "route_sigma" else None,
        "route_grids": [{
            "route_label": r.label,
            "Ta_mean_C": r.ta_mean_c, "Ta_std_C": r.ta_std_c,
            "vp_mean": r.vp_mean_hpa, "vp_std": r.vp_std_hpa,
            "Ta_uniform_values_C": sorted({c.ta_c for c
                                           in cases_by_route[r.label]}),
            "vp_uniform_values": sorted({c.vp_hpa for c
                                         in cases_by_route[r.label]}),
        } for r in routes],
        "mrt_policy": "MRT is a fixed simulated exposure input; identical "
                      "array in benchmark and every uniform case (verified "
                      "with np.array_equal each case); never recomputed "
                      "from Ta_uniform",
        "vapor_pressure_unit": VP_UNIT,
        "saturation_vapor_pressure": "Magnus es=6.112*exp(17.67*Ta/(Ta+243.5))"
                                     " hPa (sensitivity.py formulation)",
        "core_temperature_definition": "stage-09 BSA-weighted mean of the 17 "
                                       "JOS-3 segment core temperatures",
        "core_rise_definition": "Tcore(end of walk) - Tcore(after "
                                "equilibration at the walk's start "
                                "conditions)",
        "initialization": f"fresh JOS-3 model per run, "
                          f"{args.equilibration_min:g} min equilibration at "
                          "start conditions (stage-09 procedure); no run "
                          "inherits another run's state",
        "jos3_settings": {"activity_par": args.activity_par,
                          "equilibration_min": args.equilibration_min,
                          "subject": subject},
        "cases_per_route": cases_per_route,
        "invalid_combinations": [
            {"route_label": label, "Ta_uniform_C": ta, "vp_uniform": vp,
             "RH_equivalent_pct": rh}
            for label, ta, vp, rh in invalid_preview],
        "routes": [{
            "route_label": r.label, "case_name": r.case_name,
            "route_id": r.route_id, "timing_source": r.timing_source,
            "measured_dataset": r.measurement_csv,
            "measured_dataset_sha256": file_sha256(r.measurement_csv),
            "mrt_dataset": r.mrt_dir,
            "mrt_matrix_sha256": file_sha256(
                Path(r.mrt_dir) / "tmrt_matrix_C.npy"),
        } for r in routes],
    }
    (out_dir / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    # ---- Completion report -------------------------------------------------
    print("\nUniform meteorology impact study complete\n")
    print(f"Routes processed: {len(routes)}")
    if grid_mode == "route_sigma":
        print(f"Grid: route-specific uniform = mean + k*sigma, "
              f"k = {[f'{k:g}' for k in offsets]} (Ta and e independently)")
        for route in routes:
            print(f"  {route.label} Ta values [°C]: "
                  + ", ".join(f"{route.ta_mean_c + k * route.ta_std_c:.2f}"
                              for k in offsets))
            print(f"  {route.label} e values [{VP_UNIT}]: "
                  + ", ".join(f"{route.vp_mean_hpa + k * route.vp_std_hpa:.2f}"
                              for k in offsets))
    print(f"Cases per route: {cases_per_route} "
          f"({len(invalid_preview)} invalid, flagged)")
    print(f"Total uniform simulations: {len(valid)}\n")
    for route in routes:
        route_valid = valid[valid.route_label == route.label]
        print(f"{route.label} ({route.case_name}): "
              f"measured Ta {route.ta_measured_c.min():.1f}.."
              f"{route.ta_measured_c.max():.1f} °C "
              f"(μ {route.ta_mean_c:.2f}, σ {route.ta_std_c:.2f}), "
              f"e {route.vp_measured_hpa.min():.1f}.."
              f"{route.vp_measured_hpa.max():.1f} {VP_UNIT} "
              f"(μ {route.vp_mean_hpa:.2f}, σ {route.vp_std_hpa:.2f}), "
              f"{route.duration_min:.1f} min | "
              f"ΔTcore_real {route.benchmark['delta_tcore_c']:+.4f} °C, "
              f"max |uniform error| "
              f"{route_valid['absolute_error_C'].max():.4f} °C")
    print(f"\nCombined heatmaps: {combined['signed_error_C']}")
    print(f"                   {combined['absolute_error_C']}")
    print(f"Master CSV:        {out_dir / 'uniform_sensitivity_all_routes.csv'}")
    print(f"Route summary:     {out_dir / 'route_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
