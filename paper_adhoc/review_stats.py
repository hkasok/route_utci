#!/usr/bin/env python3
"""
review_stats.py -- reviewer-facing statistics for the Lisbon validation.

STANDALONE and read-only with respect to the pipeline: it imports the data
loaders, QC, statistics and bootstrap of paper_validation_figures.py and
paper_globe_attribution.py and re-runs only the black-globe emulator ODE
(black_globe.integrate_globe_temperature_C) under an alternative ventilation.

  1. Null baselines for the sun/shade agreement and its tolerance curves:
     (a) the modelled sun/shade field shifted circularly along each route by
         >= 200 m (every 10 m offset), (b) a random two-state renewal sequence
         with the model's sun/shade run-length distributions (and hence its
         sunlit fraction). Per-campaign sunlit fractions and directional
         disagreement rates, with a block-bootstrap test of symmetry.
  2. Campaign-level (two-stage) bootstrap of the daytime globe bias/RMSE/cRMSE.
  3. Ventilation sensitivity: wind only (no walking speed) vs the published
     quadrature, plus data tests of whether the cart anemometer reports
     apparent (cart-relative) or true wind.
  4. cRMSE vs sigma_obs per campaign (daytime globe).
  5. Heading-class residuals: effective sample size, block-bootstrap CIs and a
     campaign-adjusted toward-vs-away test.

Usage:  OMP_NUM_THREADS=1 nice -n 19 python3 paper_adhoc/review_stats.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import black_globe as bg                                               # noqa: E402
from paper_validation_figures import (load_points, apply_measurement_qc,  # noqa: E402
                                      shadow_registration,
                                      shadow_registration_tolerance, stats,
                                      SPATIAL_TOLERANCES_M, TIME_TOLERANCES_S)
from paper_globe_attribution import (walk_frames, add_sample_quantities,  # noqa: E402
                                     usable_day, block_bootstrap_indices,
                                     bootstrap, ci, heading_bins, HEADING_BINS,
                                     integrate_many, BLOCK)

DEFAULT_OUT = Path("/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/"
                   "98394eca-18c6-484b-8728-17aede7e7821/scratchpad/reruns/stats")
SEED = 20261020
REPS = 2000
MIN_OFFSET_M = 200.0
OFFSET_STEP_M = 10.0
N_RENEWAL = 500
REPORT_S = (0.0, 1.0, 3.0, 5.0)
REPORT_T = (5.0, 60.0)


def pct(values, ps=(5, 50, 95)):
    v = np.asarray(values, float)
    return {f"p{p}": float(np.nanpercentile(v, p)) for p in ps}


# =============================================================================
# 1. sun/shade agreement: observed and null
# =============================================================================
def registration_subset(points):
    """The exact daytime subset of shadow_registration_tolerance."""
    return points[(points["period"] == "day")
                  & (points["trec_atmospheric_ghi_wm2"] > 50)].dropna(
        subset=["measured_swin_wm2", "sensor_shortwave_down_Wm2",
                "trec_atmospheric_ghi_wm2", "timestamp_utc"])


def dense_case(g, case, root):
    """Per-walk dense modelled sun/shade field along the route, set up exactly
    as paper_validation_figures.shadow_registration_tolerance does."""
    from scipy.spatial import cKDTree
    g = g.sort_values("seq")
    cj = json.loads((root / "input" / case / "case.json").read_text())
    ox, oy = cj["coordinates"]["local_origin_x"], cj["coordinates"]["local_origin_y"]
    mrt = root / "run_output" / case / "mrt_facet_out"
    xyz = np.load(mrt / "path_xyz.npy")
    seg = np.load(mrt / "path_segment_id.npy")
    kdown = np.load(mrt / "radiant_flux_contributions.npz")["sensor_shortwave_down_Wm2"]
    times = pd.read_csv(mrt / "times.csv")
    tt = pd.to_datetime(times["time"])
    hours = (tt.dt.hour + tt.dt.minute / 60 + tt.dt.second / 3600).to_numpy(float)
    ghi_t = times["GHI_Wm2"].to_numpy(float)
    _, nearest = cKDTree(xyz[:, :2]).query(
        np.column_stack([g["x_projected_m"] - ox, g["y_projected_m"] - oy]))
    route_seg = int(np.bincount(seg[nearest]).argmax())
    on_route = np.flatnonzero(seg == route_seg)
    s_route = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(xyz[on_route, :2], axis=0), axis=1))))
    pos = np.clip(np.searchsorted(on_route, nearest), 0, len(on_route) - 1)
    s_sample = s_route[pos]
    t_s = (pd.to_datetime(g["timestamp_utc"], utc=True, format="ISO8601")
           - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(float)
    hour = g["arrival_hour_local"].to_numpy(float)
    ghi = g["trec_atmospheric_ghi_wm2"].to_numpy(float)
    meas = g["measured_swin_wm2"].to_numpy(float) > 0.5 * ghi
    model_here = g["sensor_shortwave_down_Wm2"].to_numpy(float) > 0.5 * ghi
    i1 = np.clip(np.searchsorted(hours, hour), 1, len(hours) - 1)
    w = (hour - hours[i1 - 1]) / (hours[i1] - hours[i1 - 1])
    rows = ((1 - w)[:, None] * kdown[i1 - 1][:, on_route]
            + w[:, None] * kdown[i1][:, on_route])
    ghi_k = (1 - w) * ghi_t[i1 - 1] + w * ghi_t[i1]
    dense = rows > 0.5 * ghi_k[:, None]
    own = dense[np.arange(len(g)), pos]
    return dict(case=case, g=g, s_route=s_route, s_sample=s_sample, pos=pos,
                t_s=t_s, meas=meas, model_here=model_here, dense=dense, own=own)


def doubled_cumsum(dense):
    """Cumulative sun counts over the route traversed twice (for wrap-around)."""
    dd = np.concatenate([dense, dense], axis=-1).astype(np.int32)
    pad = np.zeros(dd.shape[:-1] + (1,), np.int32)
    return np.concatenate([pad, np.cumsum(dd, axis=-1)], axis=-1)


def window_match(cs2, s2, a, b, meas, rows=None, min_one=False):
    """Does any dense point in [a, b] (doubled coordinates) match meas?"""
    lo = np.searchsorted(s2, a, side="left")
    hi = np.searchsorted(s2, b, side="right")
    if min_one:
        hi = np.maximum(hi, lo + 1)
    hi = np.minimum(hi, len(s2))
    if cs2.ndim == 2:
        sun = cs2[rows, hi] - cs2[rows, lo]
    else:
        sun = cs2[hi] - cs2[lo]
    n = hi - lo
    return np.where(meas, sun > 0, (n - sun) > 0)


def tolerance_curves(c, cs2, centre_idx, wrap_ok=True):
    """Agreement at every spatial / time tolerance with the modelled field read
    at dense index ``centre_idx`` (the sample's own position for the observed
    case, a shifted one for the null). Returns per-sample boolean arrays."""
    s_route, s_sample, t_s, meas = c["s_route"], c["s_sample"], c["t_s"], c["meas"]
    m = len(s_route)
    L = s_route[-1]
    delta = float(np.median(np.diff(s_route)))
    period = L + delta
    s2 = np.concatenate([s_route, s_route + period])
    rows = np.arange(len(meas))
    centre = s_route[centre_idx]
    shift = centre - s_sample            # rigid along-track displacement per sample
    out_s, out_t = [], []
    for tol in SPATIAL_TOLERANCES_M:
        a = centre - tol; b = centre + tol
        wrap = (a < 0) & wrap_ok
        a = np.where(wrap, a + period, a); b = np.where(wrap, b + period, b)
        if not wrap_ok:
            b = np.minimum(b, L)
        out_s.append(window_match(cs2, s2, a, b, meas, rows))
    for tol in TIME_TOLERANCES_S:
        s_lo = np.interp(t_s - tol, t_s, s_sample) + shift
        s_hi = np.interp(t_s + tol, t_s, s_sample) + shift
        a = np.minimum(s_lo, s_hi); b = np.maximum(s_lo, s_hi)
        wrap = (a < 0) & wrap_ok
        a = np.where(wrap, a + period, a); b = np.where(wrap, b + period, b)
        over = (a >= period) & wrap_ok
        a = np.where(over, a - period, a); b = np.where(over, b - period, b)
        if not wrap_ok:
            a = np.minimum(a, L); b = np.minimum(b, L)
        out_t.append(window_match(cs2, s2, a, b, meas, rows, min_one=True))
    return np.array(out_s), np.array(out_t)


def kappa(meas, model):
    po = float(np.mean(meas == model))
    pe = float(meas.mean() * model.mean() + (1 - meas.mean()) * (1 - model.mean()))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def run_lengths(state, s):
    """Along-route lengths (m) of sun and shade runs in a boolean row."""
    edges = np.flatnonzero(np.diff(state.astype(int)) != 0)
    starts = np.concatenate(([0], edges + 1)); ends = np.concatenate((edges, [len(state) - 1]))
    delta = float(np.median(np.diff(s)))
    lengths = s[ends] - s[starts] + delta
    vals = state[starts]
    return lengths[vals], lengths[~vals]


def renewal_pattern(s_route, sun_runs, shade_runs, p_sun, rng):
    """Random alternating sun/shade sequence along the route with the given
    empirical run lengths; a random phase so the route start is not a run start."""
    L = s_route[-1]
    total = 0.0
    state = bool(rng.random() < p_sun)
    bounds, states = [0.0], []
    # burn in one run so the first boundary is not at s = 0
    first = rng.choice(sun_runs if state else shade_runs) * rng.random()
    total = first; bounds.append(total); states.append(state)
    while total < L:
        state = not state
        runs = sun_runs if state else shade_runs
        total += float(rng.choice(runs)); bounds.append(total); states.append(state)
    idx = np.searchsorted(np.asarray(bounds[1:]), s_route, side="right")
    idx = np.clip(idx, 0, len(states) - 1)
    return np.asarray(states)[idx]


def analysis_null(points, root, out, rng):
    published = shadow_registration(points, out)
    published_tol = shadow_registration_tolerance(points, root, out)
    d_all = registration_subset(points)
    cases = [dense_case(g, case, root) for case, g in d_all.groupby("case_id")]
    n_total = sum(len(c["meas"]) for c in cases)

    # observed, through the same window code as the null (validates it)
    obs_s = np.zeros(len(SPATIAL_TOLERANCES_M)); obs_t = np.zeros(len(TIME_TOLERANCES_S))
    for c in cases:
        c["cs2"] = doubled_cumsum(c["dense"])
        s_ok, t_ok = tolerance_curves(c, c["cs2"], c["pos"], wrap_ok=False)
        obs_s += s_ok.sum(1); obs_t += t_ok.sum(1)
    obs_s /= n_total; obs_t /= n_total
    reproduce = {
        "max_abs_diff_spatial": float(np.max(np.abs(
            obs_s - np.asarray(published_tol["agreement_vs_spatial_tolerance"])))),
        "max_abs_diff_time": float(np.max(np.abs(
            obs_t - np.asarray(published_tol["agreement_vs_time_tolerance"])))),
    }

    # (a) circular shift of the modelled field along the route, >= 200 m
    # Offsets are expressed as a fraction of each route's length so that every
    # walk contributes at every null draw; each fraction maps to >= 200 m.
    shift_s, shift_t, shift_po, shift_k = [], [], [], []
    per_case_offsets = {}
    for c in cases:
        L = c["s_route"][-1]
        offs = np.arange(MIN_OFFSET_M, L - MIN_OFFSET_M + 1e-9, OFFSET_STEP_M)
        per_case_offsets[c["case"]] = {"route_length_m": float(L),
                                       "n_offsets": int(len(offs))}
        c["offsets"] = offs
    n_draw = min(len(c["offsets"]) for c in cases)
    fracs = np.linspace(0, 1, n_draw, endpoint=False) + 0.5 / n_draw
    for f in fracs:
        agg_s = np.zeros(len(SPATIAL_TOLERANCES_M)); agg_t = np.zeros(len(TIME_TOLERANCES_S))
        meas_all, mod_all = [], []
        for c in cases:
            offs = c["offsets"]
            off = offs[min(int(f * len(offs)), len(offs) - 1)]
            L = c["s_route"][-1]
            target = (c["s_sample"] + off) % L
            idx = np.clip(np.searchsorted(c["s_route"], target), 0, len(c["s_route"]) - 1)
            s_ok, t_ok = tolerance_curves(c, c["cs2"], idx)
            agg_s += s_ok.sum(1); agg_t += t_ok.sum(1)
            meas_all.append(c["meas"]); mod_all.append(c["dense"][np.arange(len(idx)), idx])
        shift_s.append(agg_s / n_total); shift_t.append(agg_t / n_total)
        po, k = kappa(np.concatenate(meas_all), np.concatenate(mod_all))
        shift_po.append(po); shift_k.append(k)
    shift_s = np.array(shift_s); shift_t = np.array(shift_t)

    # (b) random renewal sequences with the model's run-length distributions
    ren_s, ren_t, ren_po, ren_k, ren_frac = [], [], [], [], []
    for c in cases:
        mid = len(c["meas"]) // 2
        sun_runs, shade_runs = run_lengths(c["dense"][mid], c["s_route"])
        c["runs"] = (sun_runs, shade_runs)
        c["p_sun_route"] = float(sun_runs.sum() / (sun_runs.sum() + shade_runs.sum()))
    for _ in range(N_RENEWAL):
        agg_s = np.zeros(len(SPATIAL_TOLERANCES_M)); agg_t = np.zeros(len(TIME_TOLERANCES_S))
        meas_all, mod_all = [], []
        for c in cases:
            pat = renewal_pattern(c["s_route"], *c["runs"], c["p_sun_route"], rng)
            cs2 = doubled_cumsum(pat)
            s_ok, t_ok = tolerance_curves(c, cs2, c["pos"])
            agg_s += s_ok.sum(1); agg_t += t_ok.sum(1)
            meas_all.append(c["meas"]); mod_all.append(pat[c["pos"]])
        ren_s.append(agg_s / n_total); ren_t.append(agg_t / n_total)
        mod_cat = np.concatenate(mod_all)
        po, k = kappa(np.concatenate(meas_all), mod_cat)
        ren_po.append(po); ren_k.append(k); ren_frac.append(float(mod_cat.mean()))
    ren_s = np.array(ren_s); ren_t = np.array(ren_t)

    def summarise(arr_s, arr_t):
        res = {}
        for tol in REPORT_S:
            j = int(np.argmin(np.abs(SPATIAL_TOLERANCES_M - tol)))
            res[f"spatial_{tol:g}m"] = {"mean": float(arr_s[:, j].mean()), **pct(arr_s[:, j])}
        for tol in REPORT_T:
            j = int(np.argmin(np.abs(TIME_TOLERANCES_S - tol)))
            res[f"time_{tol:g}s"] = {"mean": float(arr_t[:, j].mean()), **pct(arr_t[:, j])}
        return res

    observed = {}
    for tol in REPORT_S:
        j = int(np.argmin(np.abs(SPATIAL_TOLERANCES_M - tol)))
        observed[f"spatial_{tol:g}m"] = float(obs_s[j])
    for tol in REPORT_T:
        j = int(np.argmin(np.abs(TIME_TOLERANCES_S - tol)))
        observed[f"time_{tol:g}s"] = float(obs_t[j])

    null_shift = summarise(shift_s, shift_t)
    null_shift["kappa"] = {"mean": float(np.mean(shift_k)), **pct(shift_k)}
    null_ren = summarise(ren_s, ren_t)
    null_ren["kappa"] = {"mean": float(np.mean(ren_k)), **pct(ren_k)}
    null_ren["model_sunlit_fraction_realised"] = {"mean": float(np.mean(ren_frac)), **pct(ren_frac)}

    # excess over chance, kappa-style: (obs - null) / (1 - null)
    excess = {}
    for key, ob in observed.items():
        for name, nl in (("shift", null_shift), ("renewal", null_ren)):
            m = nl[key]["mean"]
            excess.setdefault(key, {})[name] = {
                "obs_minus_null_pp": 100 * (ob - m),
                "chance_corrected": (ob - m) / (1 - m) if m < 1 else float("nan")}

    # per-campaign sunlit fractions and directional disagreement
    per_case = {}
    for c in cases:
        meas, mod = c["meas"], c["model_here"]
        n = len(meas)
        b = int((mod & ~meas).sum()); cc = int((~mod & meas).sum())
        from scipy.stats import binomtest
        p_mc = binomtest(b, b + cc, 0.5).pvalue if b + cc else float("nan")
        # block bootstrap of (b - c)/n within the walk
        boots = []
        grp = [np.arange(n)]
        for _ in range(REPS):
            ix = block_bootstrap_indices(grp, rng)
            boots.append(float(((mod[ix] & ~meas[ix]).sum() - (~mod[ix] & meas[ix]).sum()) / n))
        po, k = kappa(meas, mod)
        per_case[c["case"]] = {
            "n": n,
            "measured_sunlit_fraction": float(meas.mean()),
            "model_sunlit_fraction": float(mod.mean()),
            "agreement": po, "kappa": k,
            "model_sun_measured_shade": b / n,
            "model_shade_measured_sun": cc / n,
            "asymmetry_pp": 100 * (b - cc) / n,
            "asymmetry_block_bootstrap_95ci_pp": [100 * x for x in ci(boots)],
            "mcnemar_exact_p_nominal_iid": float(p_mc),
            "p_meas_shade_given_model_sun": float((mod & ~meas).sum() / max(mod.sum(), 1)),
            "p_meas_sun_given_model_shade": float((~mod & meas).sum() / max((~mod).sum(), 1)),
        }
    return {
        "published_check": {
            "agreement": published["classification_agreement"],
            "kappa": published["cohen_kappa"],
            "model_sun_measured_shade": published["model_sun_measured_shade"],
            "model_shade_measured_sun": published["model_shade_measured_sun"],
            "spatial": dict(zip(map(str, published_tol["spatial_tolerance_m"]),
                                published_tol["agreement_vs_spatial_tolerance"])),
            "time": dict(zip(map(str, published_tol["time_tolerance_s"]),
                             published_tol["agreement_vs_time_tolerance"])),
            "window_code_reproduces_published": reproduce},
        "n": int(n_total),
        "observed": observed,
        "null_circular_shift": {"min_offset_m": MIN_OFFSET_M, "offset_step_m": OFFSET_STEP_M,
                                "n_draws": int(n_draw), "routes": per_case_offsets,
                                **null_shift},
        "null_renewal_runlength_matched": {"n_realisations": N_RENEWAL,
                                           "route_sunlit_fraction_by_case": {
                                               c["case"]: c["p_sun_route"] for c in cases},
                                           **null_ren},
        "excess_over_null": excess,
        "per_campaign": per_case,
    }


# =============================================================================
# 2. campaign-level bootstrap of the daytime globe statistics
# =============================================================================
def globe_frame(points, period):
    spin = points[~points["globe_spinup_affected"].astype(bool)]
    s = spin[spin["period"] == period].dropna(
        subset=["globe_transient_temperature_C", "measured_black_globe_temperature_c"])
    return s.reset_index(drop=True)


def globe_stats_vec(f):
    st = stats(f["globe_transient_temperature_C"].to_numpy(float),
               f["measured_black_globe_temperature_c"].to_numpy(float))
    return np.array([st["mbe"], st["rmse"], st["crmse"]])


def analysis_campaign_bootstrap(points, rng):
    out = {}
    for period in ("day", "night"):
        s = globe_frame(points, period)
        cases = sorted(s["case_id"].unique())
        walks = {c: [np.asarray(ix) for _, ix in
                     s[s["case_id"] == c].groupby("route_id").indices.items()]
                 for c in cases}
        # remap: groupby on a subset returns positional indices of that subset
        walks = {c: [s.index[s["case_id"] == c].to_numpy()[ix] for ix in w]
                 for c, w in walks.items()}
        two_stage, camp_only = [], []
        for _ in range(REPS):
            pick = rng.choice(cases, size=len(cases), replace=True)
            ix = np.concatenate([block_bootstrap_indices(walks[c], rng) for c in pick])
            two_stage.append(globe_stats_vec(s.iloc[ix]))
            ix2 = np.concatenate([np.concatenate(walks[c]) for c in pick])
            camp_only.append(globe_stats_vec(s.iloc[ix2]))
        two_stage = np.array(two_stage); camp_only = np.array(camp_only)
        within = bootstrap(s, globe_stats_vec, REPS, rng)
        per = {c: stats(s.loc[s.case_id == c, "globe_transient_temperature_C"].to_numpy(float),
                        s.loc[s.case_id == c, "measured_black_globe_temperature_c"].to_numpy(float))
               for c in cases}
        biases = np.array([per[c]["mbe"] for c in cases])
        from scipy.stats import t as tdist
        half = tdist.ppf(0.975, len(cases) - 1) * biases.std(ddof=1) / np.sqrt(len(cases))
        point = globe_stats_vec(s)
        out[period] = {
            "n": int(len(s)), "point": dict(zip(("mbe", "rmse", "crmse"), point.tolist())),
            "two_stage_campaign_and_block_95ci": {k: ci(two_stage[:, j]) for j, k in
                                                  enumerate(("mbe", "rmse", "crmse"))},
            "campaign_only_95ci": {k: ci(camp_only[:, j]) for j, k in
                                   enumerate(("mbe", "rmse", "crmse"))},
            "within_walk_block_95ci_recomputed": {k: ci(within[:, j]) for j, k in
                                                  enumerate(("mbe", "rmse", "crmse"))},
            "per_campaign_mbe": {c: per[c]["mbe"] for c in cases},
            "campaign_mean_of_biases": float(biases.mean()),
            "campaign_sd_of_biases": float(biases.std(ddof=1)),
            "t_interval_over_6_campaign_biases_95": [float(biases.mean() - half),
                                                     float(biases.mean() + half)],
            "published_within_walk_ci_paper": {"day_mbe": [2.06, 2.87],
                                               "day_rmse": [2.52, 3.22],
                                               "day_crmse": [1.33, 1.63]},
        }
    return out


# =============================================================================
# 3. ventilation sensitivity and apparent-vs-true wind evidence
# =============================================================================
def walk_inputs(g, root):
    case = g["case_id"].iloc[0]
    cj = json.loads((root / "input" / case / "case.json").read_text())
    ox, oy = cj["coordinates"]["local_origin_x"], cj["coordinates"]["local_origin_y"]
    el = (g["arrival_hour_local"].to_numpy(float) - g["arrival_hour_local"].iloc[0]) * 3600.0
    x = g["x_projected_m"].to_numpy(float) - ox
    y = g["y_projected_m"].to_numpy(float) - oy
    u, v = bg.receptor_velocity(x, y, el)                # as stage 08 does
    w = pd.read_csv(root / "input" / case / "weather" / "weather_from_sensors.csv")
    wind_binned = np.interp(g["arrival_hour_local"].to_numpy(float) % 24,
                            w["hour"], w["wind_ms"], period=24)
    return el, u, v, wind_binned


def analysis_ventilation(raw, qc_points, root, rng):
    spec = bg.DEFAULT_GLOBE
    variants = ("published_control", "binned_wind_only", "sample_wind_only",
                "sample_wind_quadrature")
    frames = []
    evidence_rows = []
    zero_scale_inputs = []
    for (case, route), g in raw.groupby(["case_id", "route_id"], sort=True):
        g = g.sort_values("seq").reset_index(drop=True)
        el, u, v, wb = walk_inputs(g, root)
        cart = np.hypot(u, v)
        ws = g["measured_wind_ms"].to_numpy(float)
        ws_f = pd.Series(ws).interpolate(limit_direction="both").to_numpy()
        flux = g["globe_absorbed_flux_Wm2"].to_numpy(float)
        ta = g["trec_route_air_temperature_c"].to_numpy(float)
        vents = {"published_control": g["globe_ventilation_ms"].to_numpy(float),
                 "binned_wind_only": wb,
                 "sample_wind_only": ws_f,
                 "sample_wind_quadrature": np.hypot(ws_f, cart)}
        out = g[["case_id", "route_id", "period", "seq",
                 "measured_black_globe_temperature_c", "globe_transient_temperature_C",
                 "globe_spinup_affected"]].copy()
        out["cart_speed_ms"] = cart
        out["binned_wind_ms"] = wb
        out["quadrature_reconstructed_ms"] = np.hypot(wb, cart)
        for k, vent in vents.items():
            T, spin = bg.integrate_globe_temperature_C(el, flux, ta, vent, spec)
            out[f"Tg_{k}"] = T
            out[f"spin_{k}"] = spin
            out[f"vent_{k}"] = vent
        frames.append(out)
        # evidence: raw WDir from the case's measurement file
        m = pd.read_csv(root / "input" / case / "measurements"
                        / "experimental_measurements_all_routes.csv")
        m = m[m["route_id"] == route][["seq", "WS", "WDir"]]
        e = g[["case_id", "route_id", "period", "seq", "arrival_hour_local"]].copy()
        e["cart_u"] = u; e["cart_v"] = v; e["cart_speed"] = cart
        e = e.merge(m, on="seq", how="left")
        evidence_rows.append(e)
        if g["period"].iloc[0] == "day":
            zero_scale_inputs.append((case, g, el, flux, ta, wb))
    V = pd.concat(frames, ignore_index=True)
    keep = qc_points[["case_id", "route_id", "seq"]].assign(_qc=True)
    V = V.merge(keep, on=["case_id", "route_id", "seq"], how="left")
    V["_qc"] = V["_qc"].fillna(False).astype(bool)

    res = {"control_reintegration_max_abs_diff_K": float(np.nanmax(np.abs(
        V["Tg_published_control"] - V["globe_transient_temperature_C"]))),
        "quadrature_reconstruction_max_abs_diff_ms": float(np.nanmax(np.abs(
            V["quadrature_reconstructed_ms"] - V["vent_published_control"]))),
        "note": ("The published ventilation is sqrt(binned-median measured WS^2 + "
                 "cart speed^2), the WS binned to 10 min in weather_from_sensors.csv.")}
    for k in variants:
        res[k] = {}
        for period in ("day", "night"):
            for mask_name, mask in (
                    ("published_spinup_mask", ~V["globe_spinup_affected"].astype(bool)),
                    ("own_spinup_mask", ~V[f"spin_{k}"].astype(bool))):
                sel = (V["period"] == period) & V["_qc"] & mask
                sub = V[sel].dropna(subset=["measured_black_globe_temperature_c"])
                st = stats(sub[f"Tg_{k}"].to_numpy(float),
                           sub["measured_black_globe_temperature_c"].to_numpy(float))
                st = {kk: st[kk] for kk in ("n", "mbe", "rmse", "crmse", "sd_obs", "r")}
                st["median_ventilation_ms"] = float(sub[f"vent_{k}"].median())
                res[k].setdefault(period, {})[mask_name] = st
            sel = ((V["period"] == period) & V["_qc"]
                   & ~V["globe_spinup_affected"].astype(bool))
            sub = V[sel].dropna(subset=["measured_black_globe_temperature_c"])
            res[k][period]["per_campaign_mbe"] = {
                c: float((h[f"Tg_{k}"] - h["measured_black_globe_temperature_c"]).mean())
                for c, h in sub.groupby("case_id")}

    # forced-convection factor for zero daytime bias under wind-only ventilation
    scales = np.linspace(0.8, 2.5, 69)
    sums = {"binned_wind_only": np.zeros(len(scales)), "published_control": np.zeros(len(scales))}
    count = 0
    for case, g, el, flux, ta, wb in zero_scale_inputs:
        kmask = (~g["globe_spinup_affected"].astype(bool).to_numpy())
        qk = g[["case_id", "route_id", "seq"]].merge(keep, how="left",
                                                    on=["case_id", "route_id", "seq"])
        kmask &= qk["_qc"].fillna(False).astype(bool).to_numpy()
        tg = g["measured_black_globe_temperature_c"].to_numpy(float)
        kmask &= np.isfinite(tg)
        F = np.tile(flux, (len(scales), 1))
        for name, vent in (("binned_wind_only", wb),
                           ("published_control", g["globe_ventilation_ms"].to_numpy(float))):
            T = integrate_many(el, F, ta, vent, scales)
            sums[name] += (T[:, kmask] - tg[kmask]).sum(axis=1)
        count += int(kmask.sum())
    res["zero_bias_forced_convection_scale"] = {}
    for name, s in sums.items():
        bias = s / count
        root_ = float(np.interp(0.0, -bias, scales)) if bias.min() < 0 < bias.max() else float("nan")
        res["zero_bias_forced_convection_scale"][name] = {
            "scale": root_, "bias_at_1": float(np.interp(1.0, scales, bias))}

    # ---- evidence on what the anemometer reports
    E = pd.concat(evidence_rows, ignore_index=True)
    E = E.dropna(subset=["WS", "WDir"])
    ev = {"by_walk": {}}
    beta_rows = []
    for (case, period), h in E.groupby(["case_id", "period"]):
        ws, sp = h["WS"].to_numpy(float), h["cart_speed"].to_numpy(float)
        stop = sp < 0.3; move = sp > 0.8
        # air-motion vector (toward), from the meteorological FROM direction
        au = -ws * np.sin(np.radians(h["WDir"].to_numpy(float)))
        av = -ws * np.cos(np.radians(h["WDir"].to_numpy(float)))
        heading = np.degrees(np.arctan2(h["cart_u"], h["cart_v"])) % 360
        rel = np.radians(h["WDir"].to_numpy(float) - heading)
        ev["by_walk"][f"{case}_{period}"] = {
            "n": int(len(h)),
            "corr_WS_cart_speed": float(np.corrcoef(ws, sp)[0, 1]),
            "frac_WS_below_cart_speed": float(np.mean(ws < sp)),
            "frac_WS_below_cart_speed_when_moving_gt0p8": float(np.mean(ws[move] < sp[move]))
            if move.any() else None,
            "mean_WS_stopped_lt0p3": float(ws[stop].mean()) if stop.sum() >= 5 else None,
            "n_stopped": int(stop.sum()),
            "mean_WS_moving_gt0p8": float(ws[move].mean()) if move.any() else None,
            "mean_cos_WDir_minus_heading": float(np.mean(np.cos(rel))),
        }
        # 10-min windows: A = W_window - beta * cart
        win = np.floor(h["arrival_hour_local"].to_numpy(float) * 6).astype(int)
        beta_rows.append(pd.DataFrame({"walk": f"{case}_{period}", "case": case,
                                       "period": period, "win": win,
                                       "au": au, "av": av,
                                       "cu": h["cart_u"].to_numpy(float),
                                       "cv": h["cart_v"].to_numpy(float),
                                       "ws": ws, "sp": sp}))
    B = pd.concat(beta_rows, ignore_index=True)

    def beta_fit(f):
        # within-window demeaning of both components; common slope
        key = f["walk"] + "_" + f["win"].astype(str)
        y = np.concatenate([f["au"] - f.groupby(key)["au"].transform("mean"),
                            f["av"] - f.groupby(key)["av"].transform("mean")])
        x = np.concatenate([f["cu"] - f.groupby(key)["cu"].transform("mean"),
                            f["cv"] - f.groupby(key)["cv"].transform("mean")])
        return float(-(x @ y) / (x @ x))

    pooled_beta = beta_fit(B)
    B = B.rename(columns={"case": "case_id"}).assign(route_id=B["period"])
    boots = bootstrap(B.reset_index(drop=True), beta_fit, 500, rng)
    within_corr = []
    for _, h in B.groupby("walk"):
        within_corr.append(h[["ws", "sp"]].assign(ws=h.ws - h.ws.mean(), sp=h.sp - h.sp.mean()))
    wc = pd.concat(within_corr)
    ev["pooled"] = {
        "beta_air_vector_on_minus_cart_velocity": pooled_beta,
        "beta_block_bootstrap_95ci": ci(boots),
        "beta_interpretation": "1 = apparent (cart-relative) wind; 0 = true wind",
        "within_walk_corr_WS_cart_speed": float(np.corrcoef(wc.ws, wc.sp)[0, 1]),
        "frac_WS_below_cart_speed": float(np.mean(B.ws < B.sp)),
        "frac_WS_below_cart_speed_moving_gt0p8": float(np.mean(B.ws[B.sp > 0.8] < B.sp[B.sp > 0.8])),
        "median_cart_speed_moving": float(B.sp[B.sp > 0.8].median()),
        "mean_cos_WDir_minus_heading": float(np.mean(np.cos(np.radians(
            np.degrees(np.arctan2(-B.au, -B.av)) % 360
            - np.degrees(np.arctan2(B.cu, B.cv)) % 360)))),
    }
    key = B["walk"] + "_" + B["win"].astype(str)
    wfrom = np.arctan2(-B.groupby(key)["au"].transform("mean"),
                       -B.groupby(key)["av"].transform("mean"))
    ev["pooled"]["mean_cos_10min_mean_wind_from_minus_heading"] = float(
        np.mean(np.cos(wfrom - np.arctan2(B["cu"], B["cv"]))))
    ev["pooled"]["within_window_sd_cart_velocity_component_ms"] = float(np.sqrt(
        ((B["cu"] - B.groupby(key)["cu"].transform("mean")) ** 2).mean()))
    ev["pooled"]["beta_if_WS_were_apparent_by_construction"] = pooled_beta + 1.0
    # Synthetic check: what the same statistics look like if WS were apparent
    # wind = |W_true - cart| with W_true the 10-min binned vector mean of the data.
    ev["zenodo_wind_columns"] = ["WS", "WDir", "WDirCard"]
    ev["zenodo_has_corrected_wind_field"] = False
    res["anemometer_evidence"] = ev
    return res


# =============================================================================
# 4. cRMSE vs sigma_obs per campaign (daytime globe)
# =============================================================================
def analysis_crmse(points):
    out = {}
    for period in ("day", "night"):
        s = globe_frame(points, period)
        per = {}
        for c, h in s.groupby("case_id"):
            st = stats(h["globe_transient_temperature_C"].to_numpy(float),
                       h["measured_black_globe_temperature_c"].to_numpy(float))
            per[c] = {"n": st["n"], "crmse": st["crmse"], "sd_obs": st["sd_obs"],
                      "crmse_lt_sd_obs": bool(st["crmse"] < st["sd_obs"]),
                      "margin_K": st["sd_obs"] - st["crmse"]}
        pooled = stats(s["globe_transient_temperature_C"].to_numpy(float),
                       s["measured_black_globe_temperature_c"].to_numpy(float))
        r = s["globe_transient_temperature_C"] - s["measured_black_globe_temperature_c"]
        r_w = r - r.groupby(s["case_id"]).transform("mean")
        o_w = (s["measured_black_globe_temperature_c"]
               - s.groupby("case_id")["measured_black_globe_temperature_c"].transform("mean"))
        out[period] = {
            "per_campaign": per,
            "n_campaigns_crmse_lt_sd_obs": int(sum(v["crmse_lt_sd_obs"] for v in per.values())),
            "pooled_crmse": pooled["crmse"], "pooled_sd_obs": pooled["sd_obs"],
            "pooled_within_campaign_crmse": float(np.sqrt((r_w ** 2).mean())),
            "pooled_within_campaign_sd_obs": float(np.sqrt((o_w ** 2).mean())),
        }
    return out


# =============================================================================
# 5. heading classes
# =============================================================================
def heading_class(d):
    rel = np.abs(d["sun_relative_heading_deg"].to_numpy(float))
    cls = np.full(len(d), -1)
    for j, (_, lo, hi) in enumerate(HEADING_BINS):
        m = (rel >= lo) & (rel < hi) if hi < 180 else (rel >= lo)
        cls[m] = j
    return cls


def heading_stats(f):
    cls = heading_class(f)
    r = f["residual_K"].to_numpy(float)
    means = [r[cls == j].mean() if np.any(cls == j) else np.nan for j in range(3)]
    # campaign fixed effects: r ~ across + away + campaign dummies
    camps = pd.get_dummies(f["case_id"]).to_numpy(float)
    X = np.column_stack([(cls == 1), (cls == 2), camps]).astype(float)
    ok = cls >= 0
    coef, *_ = np.linalg.lstsq(X[ok], r[ok], rcond=None)
    return np.array(means + [means[0] - means[2], -coef[1], -coef[0]])


def analysis_heading(qc_points, root, rng):
    walks = walk_frames(qc_points)
    all_day = add_sample_quantities(walks, root)
    d = usable_day(all_day)
    pub = heading_bins(d)
    cls = heading_class(d)
    names = [n for n, _, _ in HEADING_BINS]
    # effective sample size: decorrelation lag of each walk's residual series
    neff, runs, by_case = {n: 0.0 for n in names}, {n: 0 for n in names}, {}
    lags = {}
    for (case, route), g in d.groupby(["case_id", "route_id"]):
        g = g.sort_values("seq")
        x = g["residual_K"].to_numpy(float); xc = x - x.mean()
        ac = np.correlate(xc, xc, "full")[len(xc) - 1:]; ac /= ac[0]
        below = np.flatnonzero(ac < 1 / np.e)
        lag = int(below[0]) if below.size else len(ac)
        lags[case] = lag
        c = heading_class(g)
        by_case[case] = {}
        for j, n in enumerate(names):
            k = int((c == j).sum())
            neff[n] += k / (2.0 * lag)
            # contiguous runs of this class
            m = (c == j).astype(int)
            nr = int(np.sum(np.diff(np.concatenate(([0], m))) == 1))
            runs[n] += nr
            by_case[case][n] = {"n": k, "runs": nr,
                                "residual_mean_K": float(g["residual_K"][c == j].mean()) if k else None}
    boots = bootstrap(d, heading_stats, REPS, rng)
    # two-stage (campaign + block) bootstrap of the campaign-adjusted difference
    cases = sorted(d["case_id"].unique())
    groups = {c: [d.index[d["case_id"] == c].to_numpy()] for c in cases}
    two = []
    for _ in range(REPS):
        pick = rng.choice(cases, size=len(cases), replace=True)
        parts = []
        for i, c in enumerate(pick):
            ix = block_bootstrap_indices(groups[c], rng)
            parts.append(d.loc[ix].assign(case_id=f"{c}#{i}"))
        two.append(heading_stats(pd.concat(parts, ignore_index=True)))
    two = np.array(two)
    point = heading_stats(d)
    diffs = []
    for c in cases:
        t = by_case[c][names[0]]; a = by_case[c][names[2]]
        if t["n"] and a["n"]:
            diffs.append(t["residual_mean_K"] - a["residual_mean_K"])
    from scipy.stats import ttest_1samp, wilcoxon
    out = {
        "published_check": {n: pub[n]["residual_mean_K"] for n in names},
        "n_nominal": {n: int((cls == j).sum()) for j, n in enumerate(names)},
        "n_effective": neff, "n_contiguous_runs": runs,
        "residual_decorrelation_lag_samples": lags,
        "by_campaign": by_case,
        "class_mean_K": {n: float(point[j]) for j, n in enumerate(names)},
        "class_mean_block_95ci": {n: ci(boots[:, j]) for j, n in enumerate(names)},
        "toward_minus_away_K": float(point[3]),
        "toward_minus_away_block_95ci": ci(boots[:, 3]),
        "toward_minus_away_campaign_fixed_effects_K": float(point[4]),
        "toward_minus_away_FE_block_95ci": ci(boots[:, 4]),
        "toward_minus_away_FE_two_stage_95ci": ci(two[:, 4]),
        "toward_minus_across_FE_K": float(point[5]),
        "toward_minus_across_FE_block_95ci": ci(boots[:, 5]),
        "toward_minus_across_FE_two_stage_95ci": ci(two[:, 5]),
        "per_campaign_toward_minus_away_K": diffs,
        "n_campaigns_with_both_classes": len(diffs),
    }
    if len(diffs) >= 2:
        out["t_test_over_campaigns_p"] = float(ttest_1samp(diffs, 0.0).pvalue)
        if len(diffs) >= 3:
            out["wilcoxon_over_campaigns_p"] = float(wilcoxon(diffs).pvalue)
    return out


# =============================================================================
def main():
    global REPS, N_RENEWAL
    import time
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--parts", default="1,2,3,4,5")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--renewal", type=int, default=N_RENEWAL)
    args = ap.parse_args()
    REPS, N_RENEWAL = args.reps, args.renewal
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_points(ROOT)
    points, qc = apply_measurement_qc(raw)
    jobs = {
        "1": ("1_sun_shade_null", lambda r: analysis_null(points, ROOT, args.out, r)),
        "2": ("2_campaign_bootstrap", lambda r: analysis_campaign_bootstrap(points, r)),
        "3": ("3_ventilation", lambda r: analysis_ventilation(raw, points, ROOT, r)),
        "4": ("4_crmse_vs_sd_obs", lambda r: analysis_crmse(points)),
        "5": ("5_heading", lambda r: analysis_heading(points, ROOT, r)),
    }
    for part in args.parts.split(","):
        key, fn = jobs[part]
        t0 = time.time()
        print(f"{key} ...", flush=True)
        res = fn(np.random.default_rng(SEED + int(part)))
        res["_settings"] = {"reps": REPS, "renewal": N_RENEWAL, "seed": SEED + int(part),
                            "seconds": round(time.time() - t0, 1)}
        (args.out / f"review_stats_part{part}.json").write_text(
            json.dumps(res, indent=1, default=float) + "\n")
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    # merge whatever parts exist
    result = {"qc": {k: qc[k] for k in ("n_before", "n_rejected")}}
    for part, (key, _) in jobs.items():
        f = args.out / f"review_stats_part{part}.json"
        if f.is_file():
            result[key] = json.loads(f.read_text())
    path = args.out / "review_stats.json"
    path.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
