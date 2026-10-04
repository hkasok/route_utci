#!/usr/bin/env python3
"""
paper_globe_attribution.py -- where does the daytime black-globe bias come from?

STANDALONE. Reads the per-sample comparison points that compare_mrt_lisbon_data
wrote for the six Lisbon campaigns, plus stage-05 products already on disk,
and answers four questions about the +2.6 K daytime offset of the emulated
globe. Nothing modelled is re-run; the only physics evaluated here is the
globe emulator itself (black_globe.py), integrated again under perturbed
instrument parameters, and the ISO 7726 globe-to-Tmrt conversion applied to
the MEASUREMENT.

  (a) REGRESSION of the daytime residual (emulated minus measured globe) on
      the quantities that would carry a convective error (measured Tg - Ta,
      wind combined in quadrature with the cart's own speed), a radiative one
      (modelled sunlit state, modelled beam load on the sphere) and an
      operator-shading one (walking heading relative to the sun). Standardised
      coefficients, partial R^2, and moving-block-bootstrap confidence
      intervals, because consecutive samples are not independent.
  (b) SUSTAINED SHADE versus SUSTAINED SUN: the residual on stretches that
      stayed in one state for at least 60 s, where shadow-edge registration
      cannot contribute.
  (c) MONTE CARLO budget of the emulator: forced-convection coefficient,
      paint absorptivity, sensor height, beam projected-area factor and clock
      offset drawn together; reports the band of pooled daytime bias the
      emulator's own uncertainty spans and whether the observed +2.63 K lies
      inside it. Needs the per-term globe flux that stage 05 now records
      (globe_sw_direct_Wm2, ...); height needs the 0.8 m and 1.2 m re-runs.
  (d) DIAMETER SCALING of the outdoor calibration of Thorsson et al. (2007),
      made on a 38 mm globe, to the 152 mm instrument: D^-0.4 (ISO) and D^-0.5.
  (e) CONVERSION-ERROR demonstration: the measured globe converted to Tmrt
      with ISO 7726 forced convection against the modelled standing-cylinder
      Tmrt -- the conventional comparison -- next to the like-for-like one.
  (f) Block-bootstrap confidence intervals for the headline globe statistics.

Usage:
    python3 paper_globe_attribution.py [--root DIR] [--paper-dir DIR]
        [--height-run-root 0.8=DIR 1.2=DIR] [--draws N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

import black_globe as bg
from paper_validation_figures import (CASES, apply_measurement_qc, load_points,
                                      stats)

SIGMA = 5.670374419e-8
GLOBE_D, GLOBE_EPS, GLOBE_ALPHA = 0.152, 0.957, 0.95
SUN_FRACTION = 0.5          # sunlit when K-down exceeds this fraction of GHI
SUSTAINED_S = 60.0          # a run of one sun/shade state at least this long
BLOCK = 60                  # moving-block length (samples) for the bootstrap
COMPONENTS = ["globe_sw_direct_Wm2", "globe_sw_diffuse_Wm2",
              "globe_sw_reflected_Wm2", "globe_lw_sky_Wm2", "globe_lw_surface_Wm2"]

# Monte Carlo ranges. ISO 7726 gives no uncertainty for its sphere
# correlation; +-50 % brackets the outdoor calibrations in the literature
# (Thorsson et al. 2007 found +27 % on a 38 mm globe). Absorptivity spans matte
# black paints. Height brackets the stated 1.0 m sensor height on a hand cart.
# A sphere's projected-area factor is exactly 0.25; the range below allows up
# to 12 % of the beam to be blocked by the globe's stem and mast. The clock
# offset is the +-60 s quantisation of the source timestamps.
MC_RANGES = {
    "h_scale": (0.5, 1.5),
    "sw_absorptivity": (0.90, 0.97),
    "height_m": (0.8, 1.2),
    "beam_factor": (0.22, 0.25),
    "clock_offset_s": (-60.0, 60.0),
}


# ----------------------------------------------------------------------------
# Per-sample quantities
# ----------------------------------------------------------------------------
def load_case_geometry(root: Path, case: str) -> dict:
    case_json = json.loads((root / "input" / case / "case.json").read_text())
    mrt = root / "run_output" / case / "mrt_facet_out"
    times = pd.read_csv(mrt / "times.csv")
    tt = pd.to_datetime(times["time"])
    hours = (tt.dt.hour + tt.dt.minute / 60.0 + tt.dt.second / 3600.0).to_numpy(float)
    return {
        "origin": (case_json["coordinates"]["local_origin_x"],
                   case_json["coordinates"]["local_origin_y"]),
        "hours": hours,
        "dni": times["DNI_Wm2"].to_numpy(float),
        "ghi": times["GHI_Wm2"].to_numpy(float),
        "azimuth": times["azimuth_deg"].to_numpy(float),
        "elevation": times["elevation_deg"].to_numpy(float),
    }


def walk_frames(points: pd.DataFrame) -> list[pd.DataFrame]:
    """One time-ordered frame per daytime walk (spin-up samples kept)."""
    out = []
    for (case, route), g in points[points["period"] == "day"].groupby(
            ["case_id", "route_id"], sort=True):
        g = g.sort_values("seq").reset_index(drop=True)
        t = pd.to_datetime(g["timestamp_utc"], utc=True, format="ISO8601")
        g["elapsed_s"] = (t - t.iloc[0]).dt.total_seconds().to_numpy(float)
        out.append(g)
    return out


def add_sample_quantities(walks: list[pd.DataFrame], root: Path) -> pd.DataFrame:
    frames = []
    for g in walks:
        case = g["case_id"].iloc[0]
        geo = load_case_geometry(root, case)
        t_s = g["elapsed_s"].to_numpy(float)
        x = g["x_projected_m"].to_numpy(float) - geo["origin"][0]
        y = g["y_projected_m"].to_numpy(float) - geo["origin"][1]
        u, v = bg.receptor_velocity(x, y, t_s)
        g["cart_speed_ms"] = np.hypot(u, v)
        g["heading_deg"] = np.degrees(np.arctan2(u, v)) % 360.0   # from north, cw
        hour = g["arrival_hour_local"].to_numpy(float)
        solar_az = np.interp(hour, geo["hours"], geo["azimuth"], period=24.0)
        rel = (g["heading_deg"].to_numpy(float) - solar_az + 180.0) % 360.0 - 180.0
        g["sun_relative_heading_deg"] = rel
        g["cos_sun_heading"] = np.cos(np.radians(rel))
        g["dni_Wm2"] = np.interp(hour, geo["hours"], geo["dni"], period=24.0)
        g["solar_elevation_deg"] = np.interp(hour, geo["hours"],
                                             geo["elevation"], period=24.0)
        ghi = g["trec_atmospheric_ghi_wm2"].to_numpy(float)
        g["measured_sunlit"] = g["measured_swin_wm2"].to_numpy(float) > SUN_FRACTION * ghi
        g["model_sunlit"] = g["sensor_shortwave_down_Wm2"].to_numpy(float) > SUN_FRACTION * ghi
        if "globe_sw_direct_Wm2" not in g:
            # Older comparison files: the beam the sphere absorbs, from the
            # same transmission the sensor channel implies.
            g["globe_sw_direct_Wm2"] = np.nan
        g["residual_K"] = (g["globe_transient_temperature_C"]
                           - g["measured_black_globe_temperature_c"])
        g["measured_tg_minus_ta_K"] = (g["measured_black_globe_temperature_c"]
                                       - g["measured_air_temperature_c"])
        g["wind_quadrature_ms"] = np.hypot(g["measured_wind_ms"].to_numpy(float),
                                           g["cart_speed_ms"].to_numpy(float))
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def usable_day(points: pd.DataFrame) -> pd.DataFrame:
    """Daytime samples outside the emulator spin-up with every term present."""
    d = points[(points["period"] == "day")
               & ~points["globe_spinup_affected"].astype(bool)]
    return d.dropna(subset=["residual_K", "measured_tg_minus_ta_K",
                            "wind_quadrature_ms", "cos_sun_heading",
                            "globe_absorbed_flux_Wm2"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Moving-block bootstrap
# ----------------------------------------------------------------------------
def block_bootstrap_indices(groups: list[np.ndarray], rng, block: int = BLOCK):
    """Resample each walk by contiguous blocks; keep its length."""
    idx = []
    for g in groups:
        n = len(g)
        if n <= block:
            idx.append(g)
            continue
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        take = np.concatenate([g[s:s + block] for s in starts])[:n]
        idx.append(take)
    return np.concatenate(idx)


def bootstrap(points: pd.DataFrame, func, reps: int, rng, block: int = BLOCK):
    groups = [np.asarray(ix) for _, ix in
              points.groupby(["case_id", "route_id"]).indices.items()]
    values = []
    for _ in range(reps):
        values.append(func(points.iloc[block_bootstrap_indices(groups, rng, block)]))
    return np.asarray(values)


def ci(values, lo=2.5, hi=97.5):
    return [float(np.nanpercentile(values, lo)), float(np.nanpercentile(values, hi))]


# ----------------------------------------------------------------------------
# (a) regression
# ----------------------------------------------------------------------------
REGRESSORS = [
    ("measured_tg_minus_ta_K", r"$(T_g - T_a)_{\mathrm{meas}}$", "K"),
    ("wind_quadrature_ms", r"$\sqrt{V_{\mathrm{meas}}^2 + V_{\mathrm{cart}}^2}$", r"m\,s$^{-1}$"),
    ("model_sunlit", "Modelled sunlit (0/1)", "--"),
    ("globe_sw_direct_Wm2", "Modelled beam on sphere", r"W\,m$^{-2}$"),
    ("cos_sun_heading", r"$\cos(\text{heading} - \text{solar azimuth})$", "--"),
]


def fit_regression(d: pd.DataFrame) -> dict:
    y = d["residual_K"].to_numpy(float)
    X = np.column_stack([d[k].to_numpy(float) for k, _, _ in REGRESSORS])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    fitted = Z @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / ss_tot
    partial = []
    for j in range(len(REGRESSORS)):
        keep = [0] + [k + 1 for k in range(len(REGRESSORS)) if k != j]
        b, *_ = np.linalg.lstsq(Z[:, keep], y, rcond=None)
        r2_without = 1.0 - float(((y - Z[:, keep] @ b) ** 2).sum()) / ss_tot
        partial.append(r2 - r2_without)
    return {"intercept_K": float(beta[0]),
            "coef_per_sd_K": beta[1:].tolist(),
            "coef_raw": (beta[1:] / sd).tolist(),
            "sd": sd.tolist(), "r2": r2, "partial_r2": partial,
            "n": int(len(y))}


# ----------------------------------------------------------------------------
# (b) sustained segments
# ----------------------------------------------------------------------------
def implied_excess(d: pd.DataFrame) -> np.ndarray:
    """Modelled sphere load minus the load the measurement implies (W/m2)."""
    tg = d["measured_black_globe_temperature_c"].to_numpy(float)
    ta = d["measured_air_temperature_c"].to_numpy(float)
    vent = d["globe_ventilation_ms"].to_numpy(float)
    h = bg.convection_coefficient(tg, ta, vent, GLOBE_D)
    implied = GLOBE_EPS * SIGMA * (tg + 273.15) ** 4 + h * (tg - ta)
    return d["globe_absorbed_flux_Wm2"].to_numpy(float) - implied


def sustained_segments(walks: list[pd.DataFrame], flag_column, label) -> dict:
    """Residual on runs of one state lasting >= SUSTAINED_S, spin-up excluded."""
    rows = {"shade": [], "sun": []}
    n_segments = {"shade": 0, "sun": 0}
    for g in walks:
        state = flag_column(g).to_numpy()
        valid = ~np.isnan(state.astype(float))
        t = g["elapsed_s"].to_numpy(float)
        dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or state[i] != state[start] or not valid[i]:
                if valid[start]:
                    duration = t[i - 1] - t[start] + dt_med
                    if duration >= SUSTAINED_S:
                        key = "sun" if state[start] else "shade"
                        seg = g.iloc[start:i]
                        seg = seg[~seg["globe_spinup_affected"].astype(bool)]
                        if len(seg):
                            rows[key].append(seg)
                            n_segments[key] += 1
                start = i
    out = {"definition": label}
    for key in ("shade", "sun"):
        if not rows[key]:
            out[key] = {"n": 0}
            continue
        seg = pd.concat(rows[key])
        out[key] = {
            "n": int(len(seg)), "segments": n_segments[key],
            "residual_mean_K": float(seg["residual_K"].mean()),
            "residual_rmse_K": float(np.sqrt((seg["residual_K"] ** 2).mean())),
            "tg_minus_ta_meas_K": float(seg["measured_tg_minus_ta_K"].mean()),
            "excess_Wm2": float(implied_excess(seg).mean()),
            "model_beam_Wm2": float(seg["globe_sw_direct_Wm2"].mean())
            if seg["globe_sw_direct_Wm2"].notna().all() else float("nan"),
        }
    return out


# ----------------------------------------------------------------------------
# (b') residual by walking direction relative to the sun
# ----------------------------------------------------------------------------
HEADING_BINS = [("toward the sun", 0.0, 60.0), ("across", 60.0, 120.0),
                ("away from the sun", 120.0, 180.0)]


def heading_bins(d: pd.DataFrame) -> dict:
    """Residual, and measured/modelled K-down on samples both sides class as
    sunlit, by |heading - solar azimuth|. The cart is pushed from behind, so
    with the sun behind the walker the operator's shadow falls on the cart."""
    rel = np.abs(d["sun_relative_heading_deg"].to_numpy(float))
    both_sun = (d["measured_sunlit"] & d["model_sunlit"]).to_numpy()
    out = {"mean_cos_sun_heading": float(d["cos_sun_heading"].mean())}
    for name, lo, hi in HEADING_BINS:
        m = (rel >= lo) & (rel < hi) if hi < 180 else (rel >= lo)
        s = d[m]
        bs = s[both_sun[m]]
        out[name] = {
            "n": int(m.sum()),
            "residual_mean_K": float(s["residual_K"].mean()),
            "n_both_sunlit": int(len(bs)),
            "kdown_measured_over_model_both_sunlit": float(
                (bs["measured_swin_wm2"] / bs["sensor_shortwave_down_Wm2"]).mean())
            if len(bs) else float("nan"),
            "residual_both_sunlit_K": float(bs["residual_K"].mean()) if len(bs) else float("nan"),
            "residual_both_shaded_K": float(
                s[(~s["measured_sunlit"] & ~s["model_sunlit"]).to_numpy()]["residual_K"].mean()),
        }
    return out


# ----------------------------------------------------------------------------
# (c) Monte Carlo over the emulator's parameters
# ----------------------------------------------------------------------------
def integrate_many(t_s, flux, ta, vent, h_scale, capacity=None):
    """Backward-Euler globe integration, vectorised over parameter draws.

    ``flux``, ``h_scale`` have shape (n_draws, n_samples) / (n_draws,);
    ``ta``, ``vent`` are per-sample. Reproduces black_globe.integrate_globe_
    temperature_C draw by draw (same scheme, same tolerance)."""
    n_draws, n = flux.shape
    eps_sigma = GLOBE_EPS * SIGMA
    C = bg.CAMPBELL_BLACKGLOBE_L.areal_heat_capacity_J_m2K if capacity is None else capacity
    forced0 = bg.ISO_FORCED_COEFFICIENT * np.maximum(vent, 0.0) ** 0.6 / GLOBE_D ** 0.4

    def h_of(T, k):
        natural = bg.NATURAL_CONVECTION_COEFFICIENT * (np.abs(T - ta[k]) / GLOBE_D) ** 0.25
        return np.maximum(h_scale * forced0[k], natural)

    # steady start, as the emulator does
    T = (flux[:, 0] / eps_sigma) ** 0.25 - 273.15
    for _ in range(60):
        h = h_of(T, 0)
        TK = T + 273.15
        res = flux[:, 0] - eps_sigma * TK ** 4 - h * (T - ta[0])
        step = res / (-4.0 * eps_sigma * TK ** 3 - h)
        T = T - step
        if np.max(np.abs(step)) < 1e-9:
            break
    out = np.empty((n_draws, n))
    out[:, 0] = T
    for k in range(1, n):
        dt = t_s[k] - t_s[k - 1]
        if dt <= 0:
            out[:, k] = out[:, k - 1]
            continue
        prev = out[:, k - 1]
        cur = prev.copy()
        for _ in range(40):
            h = h_of(cur, k)
            TK = cur + 273.15
            res = (C * (cur - prev) / dt - flux[:, k] + eps_sigma * TK ** 4
                   + h * (cur - ta[k]))
            step = res / (C / dt + 4.0 * eps_sigma * TK ** 3 + h)
            cur = cur - step
            if np.max(np.abs(step)) < 1e-9:
                break
        out[:, k] = cur
    return out


def component_arrays(g: pd.DataFrame, height_runs: dict, case, route) -> dict:
    """Per-term sphere flux at 1.0 m and its height derivative (per metre)."""
    comps = {c: g[c].to_numpy(float) for c in COMPONENTS}
    slope = {c: np.zeros(len(g)) for c in COMPONENTS}
    if height_runs:
        frames = {}
        for z, frame in height_runs.items():
            f = frame[(frame["case_id"] == case) & (frame["route_id"] == route)]
            f = f.set_index("seq").reindex(g["seq"].to_numpy())
            frames[z] = f
        zs = sorted(frames)
        if len(zs) == 2:
            dz = zs[1] - zs[0]
            for c in COMPONENTS:
                hi = frames[zs[1]][c].to_numpy(float)
                lo = frames[zs[0]][c].to_numpy(float)
                s = (hi - lo) / dz
                slope[c] = np.where(np.isfinite(s), s, 0.0)
    return comps, slope


def monte_carlo(walks, draws, rng, height_runs, qc_keep) -> dict:
    """Pooled daytime bias of the emulated globe under joint parameter draws."""
    params = {k: rng.uniform(lo, hi, size=draws) for k, (lo, hi) in MC_RANGES.items()}
    if not height_runs:
        params["height_m"] = np.full(draws, 1.0)
    # one-at-a-time sets, each parameter at both ends with the others nominal
    nominal = {"h_scale": 1.0, "sw_absorptivity": GLOBE_ALPHA, "height_m": 1.0,
               "beam_factor": 0.25, "clock_offset_s": 0.0}
    oat = []
    for k, (lo, hi) in MC_RANGES.items():
        if k == "height_m" and not height_runs:
            continue
        for v in (lo, hi):
            p = dict(nominal); p[k] = v; oat.append((k, v, p))
    n_extra = len(oat) + 1
    full = {k: np.concatenate([params[k], [p[k] for _, _, p in oat], [nominal[k]]])
            for k in nominal}
    n_total = draws + n_extra

    sum_res = np.zeros(n_total); sum_sq = np.zeros(n_total); count = np.zeros(n_total)
    for g in walks:
        case, route = g["case_id"].iloc[0], g["route_id"].iloc[0]
        comps, slope = component_arrays(g, height_runs, case, route)
        t = g["elapsed_s"].to_numpy(float)
        ta = g["trec_route_air_temperature_c"].to_numpy(float)
        vent = g["globe_ventilation_ms"].to_numpy(float)
        tg = g["measured_black_globe_temperature_c"].to_numpy(float)
        keep = (~g["globe_spinup_affected"].astype(bool).to_numpy()) & qc_keep(g)
        dz = (full["height_m"] - 1.0)[:, None]
        sw = ((comps["globe_sw_direct_Wm2"] + dz * slope["globe_sw_direct_Wm2"])
              * (full["beam_factor"][:, None] / 0.25)
              + comps["globe_sw_diffuse_Wm2"] + dz * slope["globe_sw_diffuse_Wm2"]
              + comps["globe_sw_reflected_Wm2"] + dz * slope["globe_sw_reflected_Wm2"])
        lw = (comps["globe_lw_sky_Wm2"] + dz * slope["globe_lw_sky_Wm2"]
              + comps["globe_lw_surface_Wm2"] + dz * slope["globe_lw_surface_Wm2"])
        flux = sw * (full["sw_absorptivity"][:, None] / GLOBE_ALPHA) + lw
        model = integrate_many(t, flux, ta, vent, full["h_scale"])
        # clock offset: the measurement the model should be compared with
        for i in range(n_total):
            shifted_t = t + full["clock_offset_s"][i]
            inside = (shifted_t >= t[0]) & (shifted_t <= t[-1]) & keep
            meas = np.interp(shifted_t[inside], t, tg)
            r = model[i, inside] - meas
            sum_res[i] += r.sum(); sum_sq[i] += (r ** 2).sum(); count[i] += len(r)
    bias = sum_res / count
    rmse = np.sqrt(sum_sq / count)
    out = {
        "draws": int(draws), "ranges": MC_RANGES,
        "height_runs_available": bool(height_runs),
        "bias_percentiles_K": {str(p): float(np.percentile(bias[:draws], p))
                               for p in (2.5, 5, 25, 50, 75, 95, 97.5)},
        "bias_min_max_K": [float(bias[:draws].min()), float(bias[:draws].max())],
        "rmse_percentiles_K": {str(p): float(np.percentile(rmse[:draws], p))
                               for p in (2.5, 50, 97.5)},
        "fraction_of_draws_with_bias_le_0": float(np.mean(bias[:draws] <= 0.0)),
        "nominal_bias_K": float(bias[-1]), "nominal_rmse_K": float(rmse[-1]),
        "one_at_a_time": {},
    }
    for j, (k, v, _) in enumerate(oat):
        out["one_at_a_time"].setdefault(k, {})[str(v)] = {
            "bias_K": float(bias[draws + j]), "rmse_K": float(rmse[draws + j])}
    # the convection scale at which the pooled bias vanishes (others nominal)
    scales = np.linspace(1.0, 2.0, 21)
    return out, scales


def zero_bias_convection_scale(walks, qc_keep) -> dict:
    """Forced-convection factor (others nominal) at which the daytime bias is 0."""
    scales = np.linspace(0.8, 2.0, 61)
    sum_res = np.zeros(len(scales)); count = 0
    for g in walks:
        t = g["elapsed_s"].to_numpy(float)
        ta = g["trec_route_air_temperature_c"].to_numpy(float)
        vent = g["globe_ventilation_ms"].to_numpy(float)
        tg = g["measured_black_globe_temperature_c"].to_numpy(float)
        flux = np.tile(g["globe_absorbed_flux_Wm2"].to_numpy(float), (len(scales), 1))
        keep = (~g["globe_spinup_affected"].astype(bool).to_numpy()) & qc_keep(g)
        model = integrate_many(t, flux, ta, vent, scales)
        sum_res += (model[:, keep] - tg[keep]).sum(axis=1); count += int(keep.sum())
    bias = sum_res / count
    root = float(np.interp(0.0, -bias, scales)) if (bias.min() < 0 < bias.max()) else float("nan")
    return {"scales": scales.tolist(), "bias_K": bias.tolist(),
            "zero_bias_scale": root,
            "bias_at_1p3": float(np.interp(1.3, scales, bias)),
            "bias_at_1p5": float(np.interp(1.5, scales, bias))}


# ----------------------------------------------------------------------------
# (d) diameter scaling of the 38 mm outdoor calibration
# ----------------------------------------------------------------------------
def thorsson_scaling(mean_ventilation_ms: float) -> dict:
    """Thorsson et al. (2007): Tmrt^4 = Tg^4 + 1.335e8 V^0.71 / (eps D^0.4) (Tg-Ta),
    fitted outdoors on a 38 mm globe. Relative to ISO 7726 (1.1e8 V^0.6 D^-0.4)
    the forced coefficient is larger by (1.335/1.1) V^0.11 at the calibration
    diameter. Carrying that to 152 mm assumes the enhancement's own diameter
    dependence: unchanged if it follows ISO's D^-0.4, reduced by
    (0.038/0.152)^0.1 if the true law is D^-0.5."""
    V = float(mean_ventilation_ms)
    ratio_38 = (1.335e8 / 1.1e8) * V ** (0.71 - 0.6)
    return {
        "mean_ventilation_ms": V,
        "enhancement_at_38mm": ratio_38,
        "enhancement_at_152mm_D-0.4": ratio_38,
        "enhancement_at_152mm_D-0.5": ratio_38 * (0.038 / 0.152) ** 0.1,
    }


# ----------------------------------------------------------------------------
# (e) conventional conversion against the standing-cylinder Tmrt
# ----------------------------------------------------------------------------
def conversion_error(points: pd.DataFrame) -> dict:
    """Measured globe -> Tmrt by ISO 7726 forced convection with the cart wind,
    compared with the modelled standing-cylinder Tmrt (the conventional
    approach), beside the like-for-like globe comparison."""
    d = points[~points["globe_spinup_affected"].astype(bool)].dropna(
        subset=["measured_black_globe_temperature_c", "measured_air_temperature_c",
                "measured_wind_ms", "trec_route_mrt_c", "globe_transient_temperature_C"])
    tg = d["measured_black_globe_temperature_c"].to_numpy(float)
    ta = d["measured_air_temperature_c"].to_numpy(float)
    V = np.maximum(d["measured_wind_ms"].to_numpy(float), 0.0)
    # ISO 7726 forced convection, 150 mm, eps 0.95 (the convention)
    mrt_iso = ((tg + 273.15) ** 4 + 1.1e8 * V ** 0.6 / (0.95 * 0.15 ** 0.4)
               * (tg - ta)) ** 0.25 - 273.15
    out = {}
    for period in ("day", "night"):
        m = (d["period"] == period).to_numpy()
        out[period] = {
            "n": int(m.sum()),
            "iso_globe_tmrt_vs_cylinder_tmrt": stats(d["trec_route_mrt_c"].to_numpy(float)[m],
                                                     mrt_iso[m]),
            "campaign_tmrt_vs_cylinder_tmrt": stats(
                d["trec_route_mrt_c"].to_numpy(float)[m],
                d["measured_mrt_c"].to_numpy(float)[m]),
            "emulated_globe_vs_measured_globe": stats(
                d["globe_transient_temperature_C"].to_numpy(float)[m], tg[m]),
            "measured_globe_vs_iso_tmrt_mean_gap_K": float((mrt_iso[m] - tg[m]).mean()),
        }
    return out


# ----------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------
def fmt_ci(lo, hi, nd=2):
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def table_attribution(reg, reg_ci, seg_meas, seg_both, mc, zero, thor, globe_ci) -> str:
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"\multicolumn{5}{l}{\textbf{(a) Regression of the daytime residual} "
         + rf"($n = {reg['n']}$, $R^2 = {reg['r2']:.2f}$)" + r"} \\",
         r"\multicolumn{2}{l}{Term} & Coef.\ per SD (K) & 95\,\% CI & Partial $R^2$ \\",
         r"\midrule"]
    for j, (key, label, _) in enumerate(REGRESSORS):
        lo, hi = reg_ci[j]
        L.append(f"\\multicolumn{{2}}{{l}}{{{label}}} & {reg['coef_per_sd_K'][j]:+.2f} & "
                 f"{fmt_ci(lo, hi)} & {reg['partial_r2'][j]:.3f} \\\\")
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\textbf{(b) Residual on sustained "
          r"($\geq$\SI{60}{\second}) sun/shade stretches}} \\",
          r"State & $n$ & Residual & $(T_g - T_a)_{\mathrm{meas}}$ & Excess \\",
          r"(measured) & (segments) & (K) & (K) & (\si{\watt\per\metre\squared}) \\",
          r"\midrule"]
    for key, name in (("shade", "Shade"), ("sun", "Sun")):
        s = seg_meas[key]
        L.append(f"{name} & {s['n']} ({s['segments']}) & {s['residual_mean_K']:+.2f} & "
                 f"{s['tg_minus_ta_meas_K']:.1f} & {s['excess_Wm2']:+.1f} \\\\")
    for key, name in (("shade", "Shade, model agrees"), ("sun", "Sun, model agrees")):
        s = seg_both[key]
        L.append(f"{name} & {s['n']} ({s['segments']}) & {s['residual_mean_K']:+.2f} & "
                 f"{s['tg_minus_ta_meas_K']:.1f} & {s['excess_Wm2']:+.1f} \\\\")
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\textbf{(c) Emulator uncertainty "
          + rf"(Monte Carlo, {mc['draws']} draws): pooled daytime bias" + r"}} \\",
          r"\multicolumn{2}{l}{Quantity} & \multicolumn{3}{l}{Value} \\", r"\midrule",
          rf"\multicolumn{{2}}{{l}}{{Nominal (ISO 7726)}} & \multicolumn{{3}}{{l}}{{{mc['nominal_bias_K']:+.2f} K "
          rf"(95\,\% block-bootstrap CI {fmt_ci(*globe_ci['day']['mbe'])})}} \\",
          rf"\multicolumn{{2}}{{l}}{{2.5--97.5\,\% band of draws}} & \multicolumn{{3}}{{l}}{{"
          rf"{mc['bias_percentiles_K']['2.5']:+.2f} to {mc['bias_percentiles_K']['97.5']:+.2f} K; "
          rf"median {mc['bias_percentiles_K']['50']:+.2f} K}} \\",
          rf"\multicolumn{{2}}{{l}}{{Draws with bias $\leq 0$}} & \multicolumn{{3}}{{l}}{{"
          rf"{100 * mc['fraction_of_draws_with_bias_le_0']:.1f}\,\%}} \\",
          r"\multicolumn{5}{l}{\emph{One at a time, others nominal:}} \\"]
    names = {"h_scale": "Forced convection $\\times$0.5 / $\\times$1.5",
             "sw_absorptivity": "Absorptivity 0.90 / 0.97",
             "height_m": "Height 0.8 / 1.2 m",
             "beam_factor": "Beam factor 0.22 / 0.25",
             "clock_offset_s": "Clock $-60$ / $+60$ s"}
    for k, label in names.items():
        if k not in mc["one_at_a_time"]:
            continue
        vals = list(mc["one_at_a_time"][k].values())
        L.append(rf"\multicolumn{{2}}{{l}}{{\quad {label}}} & \multicolumn{{3}}{{l}}{{"
                 rf"{vals[0]['bias_K']:+.2f} / {vals[1]['bias_K']:+.2f} K}} \\")
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\textbf{(d) Convective enhancement}} \\",
          r"\midrule",
          rf"\multicolumn{{2}}{{l}}{{Required for zero daytime bias}} & \multicolumn{{3}}{{l}}{{"
          rf"$\times${zero['zero_bias_scale']:.2f}}} \\",
          rf"\multicolumn{{2}}{{l}}{{Thorsson et al.\ (2007), \SI{{38}}{{\milli\metre}}}} & \multicolumn{{3}}{{l}}{{"
          rf"$\times${thor['enhancement_at_38mm']:.2f} at $V = {thor['mean_ventilation_ms']:.2f}$ m\,s$^{{-1}}$}} \\",
          rf"\multicolumn{{2}}{{l}}{{Scaled to \SI{{152}}{{\milli\metre}} ($D^{{-0.4}}$ / $D^{{-0.5}}$)}} & "
          rf"\multicolumn{{3}}{{l}}{{$\times${thor['enhancement_at_152mm_D-0.4']:.2f} / "
          rf"$\times${thor['enhancement_at_152mm_D-0.5']:.2f}}} \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_conversion(conv) -> str:
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Comparison & Period & $n$ & MBE (K) & RMSE (K) & $r$ \\", r"\midrule"]
    rows = [("iso_globe_tmrt_vs_cylinder_tmrt",
             r"Cylinder $T_{\mathrm{mrt}}$ vs ISO-converted globe"),
            ("campaign_tmrt_vs_cylinder_tmrt",
             r"Cylinder $T_{\mathrm{mrt}}$ vs campaign $T_{\mathrm{mrt}}$"),
            ("emulated_globe_vs_measured_globe", "Emulated vs measured globe")]
    for key, label in rows:
        for period in ("day", "night"):
            s = conv[period][key]
            L.append(f"{label if period == 'day' else ''} & {period} & {s['n']} & "
                     f"{s['mbe']:+.2f} & {s['rmse']:.2f} & {s['r']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--run-output", type=Path, default=None,
                    help="run_output tree to read (default <root>/run_output)")
    ap.add_argument("--paper-dir", type=Path,
                    default=Path("/home/harshin/files/fastUTEC paper"))
    ap.add_argument("--height-run-root", nargs="*", default=[],
                    help="HEIGHT=ROOT pairs for the 0.8 m and 1.2 m re-runs")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    root = args.root
    if args.run_output is not None:
        # paper_validation_figures.load_points expects <root>/run_output
        class _Root:  # minimal shim
            pass
        tmp = Path(args.run_output).resolve().parent
        points_root = tmp if (tmp / "run_output").is_dir() else root
    else:
        points_root = root
    points = load_points(points_root) if args.run_output is None else pd.concat(
        [pd.read_csv(p) for p in sorted(Path(args.run_output).glob(
            "lisbon*/validation/mrt_lisbon/radiant_flux_comparison_points.csv"))],
        ignore_index=True)
    points, qc = apply_measurement_qc(points)
    rejected_keys = set()
    walks = walk_frames(points)
    all_day = add_sample_quantities(walks, root)
    walks = [g for _, g in all_day.groupby(["case_id", "route_id"], sort=True)]
    walks = [g.sort_values("seq").reset_index(drop=True) for g in walks]
    d = usable_day(all_day)
    print(f"daytime samples outside spin-up: {len(d)} in {len(walks)} walks")

    have_components = all(c in d and d[c].notna().all() for c in COMPONENTS)
    if not have_components:
        print("NOTE: globe component columns absent -- using beam from tau*DNI; "
              "Monte Carlo skipped")
        for g in walks:
            g["globe_sw_direct_Wm2"] = np.nan
        d["globe_sw_direct_Wm2"] = (GLOBE_ALPHA * 0.25 * d["dni_Wm2"]
                                    * np.clip(d["sensor_shortwave_down_Wm2"]
                                              / np.maximum(d["trec_atmospheric_ghi_wm2"], 1), 0, 1)
                                    * d["model_sunlit"])
        for g in walks:
            g["globe_sw_direct_Wm2"] = (GLOBE_ALPHA * 0.25 * g["dni_Wm2"]
                                        * g["model_sunlit"])

    # (a)
    reg = fit_regression(d)
    boots = bootstrap(d, lambda s: np.asarray(fit_regression(s)["coef_per_sd_K"]),
                      args.bootstrap_reps, rng)
    reg_ci = [ci(boots[:, j]) for j in range(len(REGRESSORS))]
    print("\n(a) REGRESSION of residual (K):  R2 = %.3f" % reg["r2"])
    for j, (k, _, unit) in enumerate(REGRESSORS):
        print(f"   {k:28s} {reg['coef_per_sd_K'][j]:+.3f} K/SD  CI {reg_ci[j]}  "
              f"raw {reg['coef_raw'][j]:+.4f} K/{unit}  partial R2 {reg['partial_r2'][j]:.3f}")

    # (b)
    seg_meas = sustained_segments(walks, lambda g: g["measured_sunlit"], "measured state")
    seg_both = sustained_segments(
        walks, lambda g: g["measured_sunlit"].where(g["measured_sunlit"] == g["model_sunlit"]),
        "measured state, model agrees")
    print("\n(b) SUSTAINED SEGMENTS")
    for s in (seg_meas, seg_both):
        for key in ("shade", "sun"):
            print(f"   {s['definition']:28s} {key:5s} n={s[key].get('n', 0):4d} "
                  f"seg={s[key].get('segments', 0):3d} residual {s[key].get('residual_mean_K', float('nan')):+.2f} K "
                  f"Tg-Ta {s[key].get('tg_minus_ta_meas_K', float('nan')):.1f} K "
                  f"excess {s[key].get('excess_Wm2', float('nan')):+.1f} W/m2")

    heading = heading_bins(d)
    print("\n(b') RESIDUAL BY HEADING RELATIVE TO THE SUN  (mean cos %.2f)" % heading["mean_cos_sun_heading"])
    for name, _, _ in HEADING_BINS:
        h = heading[name]
        print(f"   {name:18s} n={h['n']:4d} residual {h['residual_mean_K']:+.2f} K | both sunlit n={h['n_both_sunlit']:4d} "
              f"residual {h['residual_both_sunlit_K']:+.2f} K, measured/model K-down {h['kdown_measured_over_model_both_sunlit']:.3f} | "
              f"both shaded residual {h['residual_both_shaded_K']:+.2f} K")

    # (f) block-bootstrap CIs of the headline globe statistics
    spin = points[~points["globe_spinup_affected"].astype(bool)]
    globe_ci = {}
    for period in ("day", "night"):
        s = spin[spin["period"] == period].dropna(
            subset=["globe_transient_temperature_C", "measured_black_globe_temperature_c"])
        s = s.reset_index(drop=True)
        b = bootstrap(s, lambda f: np.array([
            stats(f["globe_transient_temperature_C"].to_numpy(float),
                  f["measured_black_globe_temperature_c"].to_numpy(float))[k]
            for k in ("mbe", "rmse", "crmse")]), args.bootstrap_reps, rng)
        globe_ci[period] = {"mbe": ci(b[:, 0]), "rmse": ci(b[:, 1]), "crmse": ci(b[:, 2]),
                            "n": int(len(s))}
    print("\n(f) BLOCK-BOOTSTRAP 95% CIs:", json.dumps(globe_ci))

    # (c)
    qc_keep = lambda g: np.ones(len(g), dtype=bool)
    zero = zero_bias_convection_scale(walks, qc_keep)
    print(f"\n(c) zero-bias forced-convection scale: x{zero['zero_bias_scale']:.2f} "
          f"(bias at x1.3 {zero['bias_at_1p3']:+.2f}, x1.5 {zero['bias_at_1p5']:+.2f} K)")
    mc = None
    if have_components:
        height_runs = {}
        for item in args.height_run_root:
            z, path = item.split("=", 1)
            frames = [pd.read_csv(p) for p in sorted(Path(path).glob(
                "run_output/lisbon*/validation/mrt_lisbon/radiant_flux_comparison_points.csv"))]
            height_runs[float(z)] = pd.concat(frames, ignore_index=True)
        mc, _ = monte_carlo(walks, args.draws, rng, height_runs, qc_keep)
        print(f"    MC pooled daytime bias: nominal {mc['nominal_bias_K']:+.2f} K; "
              f"2.5-97.5% {mc['bias_percentiles_K']['2.5']:+.2f}..{mc['bias_percentiles_K']['97.5']:+.2f} K; "
              f"P(bias<=0) = {mc['fraction_of_draws_with_bias_le_0']:.3f}")
        for k, v in mc["one_at_a_time"].items():
            print(f"      {k:16s} " + "  ".join(f"{vv}: {r['bias_K']:+.2f} K" for vv, r in v.items()))

    # (d)
    thor = thorsson_scaling(float(d["globe_ventilation_ms"].mean()))
    print("\n(d) THORSSON SCALING:", json.dumps(thor))

    # (e)
    conv = conversion_error(points)
    print("\n(e) CONVERSION ERROR")
    for period in ("day", "night"):
        for k in ("iso_globe_tmrt_vs_cylinder_tmrt", "campaign_tmrt_vs_cylinder_tmrt",
                  "emulated_globe_vs_measured_globe"):
            s = conv[period][k]
            print(f"   {period:5s} {k:36s} n={s['n']:4d} MBE {s['mbe']:+.2f} RMSE {s['rmse']:.2f} r={s['r']:.2f}")

    out = args.paper_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_day": int(len(d)), "regression": reg, "regression_ci": reg_ci,
               "sustained_measured": seg_meas, "sustained_both": seg_both,
               "heading_bins": heading,
               "monte_carlo": mc, "zero_bias_scale": zero, "thorsson": thor,
               "conversion_error": conv, "globe_block_bootstrap_ci": globe_ci,
               "measurement_qc": qc}
    (out / "globe_attribution_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "table_conversion_error.tex").write_text(table_conversion(conv) + "\n")
    if mc is not None:
        (out / "table_globe_attribution.tex").write_text(
            table_attribution(reg, reg_ci, seg_meas, seg_both, mc, zero, thor, globe_ci) + "\n")
    print(f"\nwrote {out / 'globe_attribution_summary.json'}")


if __name__ == "__main__":
    main()
