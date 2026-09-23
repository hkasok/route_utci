#!/usr/bin/env python3
"""
paper_validation_figures.py -- pooled validation figures and LaTeX tables for
the TREC-Route manuscript.

STANDALONE. This reads existing run_output/ artefacts and writes figures and
table fragments into the paper directory. It does not import, modify or
re-run any pipeline stage. With two stated exceptions it computes no new
physics: every number it reports is read from, or aggregated over, files that
stage 05/08/10 already wrote. The exceptions are the two inversions that locate
the daytime bias -- the globe and surface energy balances run backwards on the
MEASUREMENT -- and the solar-geometry check in ``solar_forcing_audit``, which
compares measured irradiance against the extraterrestrial horizontal limit.
Both act on the measured side only and change nothing on the modelled side.

Why pooling matters here: each case's own scatter contains one daytime and one
nighttime survey, so a single-case figure shows two clusters and conveys almost
nothing about agreement. Pooling the six campaigns spans 20-46 C and makes the
day/night structure, and the daytime warm bias, legible.

Honesty constraints carried through from the pipeline's own comparison rules:

  * Only LIKE-FOR-LIKE quantities are plotted. The four radiometer channels are
    horizontal, cosine-weighted irradiance on both sides. The globe comparison
    is the emulated globe against the measured globe -- never a standing-cylinder
    MRT against a globe-derived one.
  * Night rows whose simulation date differs from the observation date are
    flagged. The surface-temperature history preceding a night walk is not an
    independently simulated night-date history, so those rows are reported
    separately rather than silently pooled.
  * Globe samples still inside the emulator's thermal spin-up are excluded, as
    they carry the initial condition rather than the modelled environment.

Usage:
    python3 paper_validation_figures.py [--paper-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

CASES = [f"lisbon{i}" for i in range(1, 7)]
CHANNELS = [
    ("sensor_shortwave_down_Wm2", "measured_swin_wm2", r"Shortwave down $K\downarrow$"),
    ("sensor_shortwave_up_Wm2", "measured_swout_wm2", r"Shortwave up $K\uparrow$"),
    ("sensor_longwave_down_Wm2", "measured_lwin_wm2", r"Longwave down $L\downarrow$"),
    ("sensor_longwave_up_Wm2", "measured_lwout_wm2", r"Longwave up $L\uparrow$"),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

DAY, NIGHT = "#c0392b", "#2471a3"


def load_points(root: Path) -> pd.DataFrame:
    frames = []
    for case in CASES:
        path = root / "run_output" / case / "validation" / "mrt_lisbon" / \
            "radiant_flux_comparison_points.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no comparison points found -- run the pipeline first")
    return pd.concat(frames, ignore_index=True)


# Quality control applied to the MEASUREMENTS before any comparison. The rule
# uses measured quantities only -- it contains no modelled term -- so it cannot
# select for or against agreement with the model.
SHORTWAVE_DROPOUT_KDOWN_WM2 = 10.0


def apply_measurement_qc(points: pd.DataFrame) -> tuple:
    """Reject daytime samples in which BOTH shortwave channels read at or below
    their zero simultaneously.

    A four-component radiometer cannot report near-zero downwelling and negative
    upwelling shortwave at the same time while the sun is above the horizon.
    Complete building shade still admits diffuse sky irradiance: across these six
    campaigns the 1st percentile of genuine deep-shade samples (those the model
    also calls shaded) is 19 W m-2, and the 5th percentile 31 W m-2. A pair of
    simultaneous zeros is the instrument reporting its own offset, not the sky.

    Small negative upwelling values on their own are the ordinary pyranometer
    thermal offset at a dark point and are RETAINED; it is the coincidence of
    both channels that identifies a dropout.
    """
    day = points["period"] == "day"
    dropout = (day
               & (points["measured_swin_wm2"] < SHORTWAVE_DROPOUT_KDOWN_WM2)
               & (points["measured_swout_wm2"] < 0.0))
    audit = {
        "rule": ("daytime AND measured K-down < "
                 f"{SHORTWAVE_DROPOUT_KDOWN_WM2:g} W/m2 AND measured K-up < 0"),
        "n_before": int(len(points)),
        "n_rejected": int(dropout.sum()),
        "rejected_fraction": float(dropout.mean()),
        "by_case": {k: int(v) for k, v in
                    points.loc[dropout, "case_id"].value_counts().items()},
    }
    if dropout.any():
        audit["extent_m"] = {
            case: [float(g["distance_along_route_m"].min()),
                   float(g["distance_along_route_m"].max())]
            for case, g in points.loc[dropout].groupby("case_id")}
    return points.loc[~dropout].reset_index(drop=True), audit


def stats(model: np.ndarray, measured: np.ndarray) -> dict:
    ok = np.isfinite(model) & np.isfinite(measured)
    model, measured = model[ok], measured[ok]
    if len(model) < 3:
        return {"n": len(model)}
    residual = model - measured
    return {
        "n": int(len(model)),
        "mbe": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        # Centred (bias-removed) RMSE. Separates the systematic offset from the
        # ability to track variation, which a pooled RMSE conflates.
        "crmse": float(np.sqrt(((residual - residual.mean()) ** 2).mean())),
        # Observed standard deviation: the signal the model is asked to resolve.
        # crmse < sd_obs means the model beats predicting that walk's own mean.
        "sd_obs": float(measured.std()),
        "r": float(np.corrcoef(model, measured)[0, 1]),
        "measured_mean": float(measured.mean()),
        "model_mean": float(model.mean()),
    }


def within_case_r(frame: pd.DataFrame, model_col: str, meas_col: str) -> float:
    """Correlation after removing each case's own mean from both series.

    A correlation pooled over campaigns is inflated by between-campaign spread:
    six walks at different air temperatures correlate well even if nothing is
    resolved along any individual walk. Centring per case removes that and
    reports only within-walk skill.
    """
    d = frame.dropna(subset=[model_col, meas_col])
    if len(d) < 3 or d["case_id"].nunique() < 2:
        return float("nan")
    m = d[model_col] - d.groupby("case_id")[model_col].transform("mean")
    o = d[meas_col] - d.groupby("case_id")[meas_col].transform("mean")
    return float(np.corrcoef(m, o)[0, 1])


def globe_flux_budget(points: pd.DataFrame) -> dict:
    """Invert the measured globe's energy balance to locate the daytime bias.

    The emulator solves C dTg/dt = R_abs - eps*sigma*Tg^4 - h_c*(Tg - Ta). Run
    backwards on the MEASURED globe temperature, air temperature and wind, it
    returns the absorbed flux the measurement implies. Differencing that against
    the modelled absorbed flux separates a radiative error from a registration
    one: registration is near-symmetric and cancels in the mean, so any surviving
    mean offset in absorbed flux is radiative.
    """
    from black_globe import convection_coefficient

    sigma, eps, diameter = 5.670374419e-8, 0.95, 0.152
    d = points.copy()
    if "globe_spinup_affected" in d:
        d = d[~d["globe_spinup_affected"].astype(bool)]
    need = ["measured_black_globe_temperature_c", "measured_air_temperature_c",
            "measured_wind_ms", "globe_absorbed_flux_Wm2",
            "globe_transient_temperature_C"]
    d = d.dropna(subset=need)

    out = {}
    for period in ("day", "night"):
        s = d[d["period"] == period]
        if len(s) < 3:
            continue
        tg = s["measured_black_globe_temperature_c"].values
        ta = s["measured_air_temperature_c"].values
        h = convection_coefficient(tg, ta, s["measured_wind_ms"].values, diameter)
        implied = eps * sigma * (tg + 273.15) ** 4 + h * (tg - ta)
        excess = s["globe_absorbed_flux_Wm2"].values - implied
        h_rad = 4.0 * eps * sigma * (tg.mean() + 273.15) ** 3
        out[period] = {
            "n": int(len(s)),
            "model_absorbed_Wm2": float(s["globe_absorbed_flux_Wm2"].mean()),
            "implied_absorbed_Wm2": float(implied.mean()),
            "excess_Wm2": float(excess.mean()),
            "h_total_Wm2K": float(h.mean() + h_rad),
            "implied_bias_K": float(excess.mean() / (h.mean() + h_rad)),
            "actual_bias_K": float((s["globe_transient_temperature_C"].values
                                    - tg).mean()),
            "lw_from_surfaces_Wm2": float(s["lw_surface_total_absorbed_Wm2"].mean())
            if "lw_surface_total_absorbed_Wm2" in s else float("nan"),
            "lw_from_sky_Wm2": float(s["lw_sky_absorbed_Wm2"].mean())
            if "lw_sky_absorbed_Wm2" in s else float("nan"),
        }
    return out


def surface_temperature_budget(points: pd.DataFrame) -> dict:
    """Ground surface temperature inverted from the two longwave channels.

    L_up = eps*sigma*Ts^4 + (1 - eps)*L_down, so the upwelling channel gives the
    surface temperature each side implies once the reflected part is removed.
    Reporting day and night separately separates a mean offset from a
    diurnal-amplitude error: a surface whose thermal admittance is too low runs
    hot by day AND cold by night, which a single daytime bias cannot reveal.
    """
    sigma, eps = 5.670374419e-8, 0.95

    def ts(l_up, l_down):
        return ((l_up - (1.0 - eps) * l_down) / (eps * sigma)) ** 0.25 - 273.15

    out = {}
    for period in ("day", "night"):
        s = points[points["period"] == period].dropna(
            subset=["measured_lwout_wm2", "sensor_longwave_up_Wm2",
                    "measured_lwin_wm2", "sensor_longwave_down_Wm2"])
        if len(s) < 3:
            continue
        out[period] = {
            "n": int(len(s)),
            "measured_surface_C": float(ts(s["measured_lwout_wm2"].mean(),
                                           s["measured_lwin_wm2"].mean())),
            "model_surface_C": float(ts(s["sensor_longwave_up_Wm2"].mean(),
                                        s["sensor_longwave_down_Wm2"].mean())),
            "measured_sky_C": float((s["measured_lwin_wm2"].mean() / sigma) ** 0.25
                                    - 273.15),
            "model_sky_C": float((s["sensor_longwave_down_Wm2"].mean() / sigma) ** 0.25
                                 - 273.15),
        }
        out[period]["surface_bias_K"] = (out[period]["model_surface_C"]
                                         - out[period]["measured_surface_C"])
    if "day" in out and "night" in out:
        ma = out["day"]["measured_surface_C"] - out["night"]["measured_surface_C"]
        da = out["day"]["model_surface_C"] - out["night"]["model_surface_C"]
        out["amplitude"] = {
            "measured_K": float(ma), "model_K": float(da),
            "excess_fraction": float(da / ma - 1.0),
            # Surface amplitude under periodic forcing scales as 1/mu with
            # mu = sqrt(k*rho*c); this is the admittance factor that would close it.
            "admittance_factor_needed": float(da / ma),
            "volumetric_factor_needed": float((da / ma) ** 2),
            "mean_offset_K": float((out["day"]["surface_bias_K"]
                                    + out["night"]["surface_bias_K"]) / 2.0),
            "amplitude_bias_K": float((out["day"]["surface_bias_K"]
                                       - out["night"]["surface_bias_K"]) / 2.0),
        }
    return out


def _square(ax, lo, hi, xlabel, ylabel, title):
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def figure_globe(points: pd.DataFrame, out: Path) -> dict:
    """Pooled measured vs emulated black-globe temperature."""
    d = points.copy()
    if "globe_spinup_affected" in d:
        d = d[~d["globe_spinup_affected"].astype(bool)]
    d = d.dropna(subset=["measured_black_globe_temperature_c",
                         "globe_transient_temperature_C"])
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    summary = {}
    for period, colour in (("day", DAY), ("night", NIGHT)):
        s = d[d["period"] == period]
        if s.empty:
            continue
        st = stats(s["globe_transient_temperature_C"].values,
                   s["measured_black_globe_temperature_c"].values)
        summary[period] = st
        ax.scatter(s["measured_black_globe_temperature_c"],
                   s["globe_transient_temperature_C"], s=5, alpha=0.35,
                   c=colour, lw=0, zorder=2,
                   label=f"{period} (n={st['n']}, MBE {st['mbe']:+.2f} K)")
    lo = float(np.nanmin([d["measured_black_globe_temperature_c"].min(),
                          d["globe_transient_temperature_C"].min()])) - 1.5
    hi = float(np.nanmax([d["measured_black_globe_temperature_c"].max(),
                          d["globe_transient_temperature_C"].max()])) + 1.5
    _square(ax, lo, hi, "Measured globe temperature (°C)",
            "Emulated globe temperature (°C)",
            "Black globe, like-for-like (6 campaigns)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.savefig(out / "validation_black_globe_pooled.png")
    plt.close(fig)
    return summary


def figure_radiometer(points: pd.DataFrame, out: Path) -> dict:
    """Pooled four-component radiometer channels, drawn as 2-D density.

    Two presentation choices matter for what this figure is asked to show.

    A scatter of 4600 points at any usable marker size saturates wherever the
    data is dense, so the four corner modes of the shortwave panels -- which are
    the sun/shade registration quadrants and the substance of the result --
    become indistinguishable from one another however different their
    occupancies are. Hexagonal binning with logarithmic counts keeps them
    distinguishable.

    Axis limits are percentile-clipped rather than set by the extremes. A
    handful of tilt-inflated measured samples reach 1492 W m-2 against a 99th
    percentile of 1081, and letting them set the range leaves roughly half of
    the panel empty and compresses the structure into a corner. Clipped points
    are counted and annotated rather than silently dropped: they are present in
    every statistic reported, and only their position on the page is affected.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 7.0), constrained_layout=True)
    summary = {}
    for ax, (mcol, ocol, label) in zip(axes.ravel(), CHANNELS):
        d = points.dropna(subset=[mcol, ocol])
        st_all = stats(d[mcol].values, d[ocol].values)
        summary[label] = st_all

        both = np.concatenate([d[mcol].values, d[ocol].values])
        lo, hi = np.percentile(both, [0.2, 99.8])
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        clipped = int((~((d[mcol].between(lo, hi)) & (d[ocol].between(lo, hi)))).sum())

        for period, cmap in (("day", "Reds"), ("night", "Blues")):
            sub = d[d["period"] == period]
            if sub.empty:
                continue
            ax.hexbin(sub[ocol], sub[mcol], gridsize=42, bins="log", cmap=cmap,
                      mincnt=1, linewidths=0.0, extent=(lo, hi, lo, hi),
                      alpha=0.85, zorder=2)
        _square(ax, lo, hi, "Measured (W m$^{-2}$)", "TREC-Route (W m$^{-2}$)",
                label)

        note = (f"n={st_all['n']}\nMBE {st_all['mbe']:+.1f}\n"
                f"RMSE {st_all['rmse']:.1f}\nr={st_all['r']:.2f}")
        if clipped:
            note += f"\n{clipped} off-scale"
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.2,
                bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.92, pad=2.2))

    handles = [plt.Line2D([], [], marker="h", ls="", ms=7, mec="none", mfc=c)
               for c in (DAY, NIGHT)]
    axes[0, 0].legend(handles, ["day", "night"], loc="lower right", frameon=True,
                      framealpha=0.9, fontsize=7.5, handletextpad=0.4)
    fig.savefig(out / "validation_radiometer_pooled.png")
    plt.close(fig)
    return summary


def shadow_registration(points: pd.DataFrame, out: Path) -> dict:
    """Where the daytime shortwave error actually comes from.

    A pedestrian route crosses shadow edges every few metres. Whether a given
    sample is sunlit is therefore decided by geometry at a scale comparable to
    GPS accuracy and to the crown reconstruction itself, so the instantaneous
    horizontal irradiance is the most demanding quantity in the whole
    comparison. Splitting the daytime residual by whether model and measurement
    agree on sunlit-versus-shaded separates a registration error from a
    radiative one, which a pooled RMSE cannot.
    """
    d = points[(points["period"] == "day") &
               (points["trec_atmospheric_ghi_wm2"] > 50)].dropna(
        subset=["measured_swin_wm2", "sensor_shortwave_down_Wm2",
                "trec_atmospheric_ghi_wm2"])
    ghi = d["trec_atmospheric_ghi_wm2"].values
    meas_sun = d["measured_swin_wm2"].values > 0.5 * ghi
    model_sun = d["sensor_shortwave_down_Wm2"].values > 0.5 * ghi
    agree = meas_sun == model_sun
    residual = d["sensor_shortwave_down_Wm2"].values - d["measured_swin_wm2"].values

    result = {
        "n": int(len(d)),
        "measured_sunlit_fraction": float(meas_sun.mean()),
        "model_sunlit_fraction": float(model_sun.mean()),
        "classification_agreement": float(agree.mean()),
        "model_sun_measured_shade": float((model_sun & ~meas_sun).mean()),
        "model_shade_measured_sun": float((meas_sun & ~model_sun).mean()),
        "rmse_agree": float(np.sqrt((residual[agree] ** 2).mean())),
        "rmse_disagree": float(np.sqrt((residual[~agree] ** 2).mean())),
        "rmse_all": float(np.sqrt((residual ** 2).mean())),
        "mbe_all": float(residual.mean()),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    ax1.scatter(d["measured_swin_wm2"][agree], d["sensor_shortwave_down_Wm2"][agree],
                s=4, alpha=0.3, c="#27ae60", lw=0,
                label=f"same sun/shade class ({agree.mean():.0%})")
    ax1.scatter(d["measured_swin_wm2"][~agree], d["sensor_shortwave_down_Wm2"][~agree],
                s=4, alpha=0.3, c="#c0392b", lw=0,
                label=f"opposite class ({1 - agree.mean():.0%})")
    hi = float(max(d["measured_swin_wm2"].max(), d["sensor_shortwave_down_Wm2"].max()))
    _square(ax1, -30, hi * 1.03, r"Measured $K\downarrow$ (W m$^{-2}$)",
            r"TREC-Route $K\downarrow$ (W m$^{-2}$)", "(a) Daytime shortwave down")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, markerscale=2.2,
               fontsize=7)

    labels = ["classes\nagree", "classes\ndisagree", "all\ndaytime"]
    values = [result["rmse_agree"], result["rmse_disagree"], result["rmse_all"]]
    bars = ax2.bar(labels, values, color=["#27ae60", "#c0392b", "0.55"], width=0.62)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.0f}",
                 ha="center", fontsize=8)
    ax2.set_ylabel(r"RMSE in $K\downarrow$ (W m$^{-2}$)")
    ax2.set_title("(b) Error is shadow registration")
    ax2.set_ylim(0, max(values) * 1.18)
    ax2.grid(axis="x", visible=False)
    fig.savefig(out / "validation_shadow_registration.png")
    plt.close(fig)
    return result


def autocorrelation_audit(points: pd.DataFrame) -> dict:
    """How much independent information the along-route comparison carries.

    Every walk is a time series sampled every 5-10 s, so consecutive samples are
    not independent draws: the globe's own time constant alone guarantees that
    neighbouring points repeat most of their information. RMSE and bias remain
    valid descriptive statistics of the comparison, but the nominal sample count
    must not be read as a count of independent tests.

    The decorrelation lag is taken as the first lag at which the measured
    series' autocorrelation falls below 1/e, and the effective sample size as
    n / (2 * that lag). Computed on the MEASUREMENT only, so it describes the
    observation's own structure rather than the model's.
    """
    by_walk, n_tot, neff_tot = {}, 0, 0
    for (case, period), g in points.groupby(["case_id", "period"]):
        x = g.sort_values("seq")["measured_black_globe_temperature_c"].values
        x = x[np.isfinite(x)].astype(float)
        if len(x) < 60:
            continue
        xc = x - x.mean()
        ac = np.correlate(xc, xc, "full")[len(xc) - 1:]
        ac = ac / ac[0]
        below = np.flatnonzero(ac < 1.0 / np.e)
        lag = int(below[0]) if below.size else len(ac)
        neff = max(1, len(x) // max(1, 2 * lag))
        by_walk[f"{case}_{period}"] = {"n": int(len(x)), "decorrelation_lag": lag,
                                       "n_effective": int(neff)}
        n_tot += len(x)
        neff_tot += neff
    return {"by_walk": by_walk, "n_total": int(n_tot),
            "n_effective_total": int(neff_tot),
            "effective_fraction": (neff_tot / n_tot) if n_tot else None}


def albedo_skill(points: pd.DataFrame) -> dict:
    """Separate the LEVEL of the modelled ground albedo from its PLACEMENT.

    The upwelling shortwave channel is the product of two things the framework
    gets right to very different degrees: how much sun reaches the ground
    (shading, which it resolves well) and what the ground is made of (material
    identity, which for 85% of surface area is a generic fallback). Comparing
    K-up directly conflates them, because a shaded sample is dark whatever its
    albedo.

    Restricting to samples that BOTH sides class as sunlit removes the shading
    term, and the ratio K-up / K-down is then the effective albedo of whatever
    the sensor is over. Its mean, its spread and its correlation answer three
    different questions: is the typical surface right, is the scene's variety
    right, and is that variety in the right PLACE. A model can score well on the
    first two and near zero on the third, which is a materially different defect
    from "the model varies too little" and points at a different fix.
    """
    d = points[(points["period"] == "day") &
               (points["trec_atmospheric_ghi_wm2"] > 50)].dropna(
        subset=["measured_swin_wm2", "measured_swout_wm2",
                "sensor_shortwave_down_Wm2", "sensor_shortwave_up_Wm2",
                "trec_atmospheric_ghi_wm2"])
    ghi = d["trec_atmospheric_ghi_wm2"].values
    both_sunlit = ((d["measured_swin_wm2"].values > 0.5 * ghi) &
                   (d["sensor_shortwave_down_Wm2"].values > 0.5 * ghi))

    def _ratios(frame: pd.DataFrame, mask: np.ndarray) -> tuple:
        s = frame[mask]
        return (s["measured_swout_wm2"].values / s["measured_swin_wm2"].values,
                s["sensor_shortwave_up_Wm2"].values / s["sensor_shortwave_down_Wm2"].values)

    def _block(frame: pd.DataFrame, mask: np.ndarray) -> dict:
        meas_a, model_a = _ratios(frame, mask)
        ok = np.isfinite(meas_a) & np.isfinite(model_a)
        meas_a, model_a = meas_a[ok], model_a[ok]
        sub = frame[mask]
        out = {
            "n_both_sunlit": int(len(meas_a)),
            "measured_albedo_mean": float(meas_a.mean()) if len(meas_a) else float("nan"),
            "model_albedo_mean": float(model_a.mean()) if len(meas_a) else float("nan"),
            "measured_albedo_sd": float(meas_a.std()) if len(meas_a) else float("nan"),
            "model_albedo_sd": float(model_a.std()) if len(meas_a) else float("nan"),
            "albedo_r": float(np.corrcoef(model_a, meas_a)[0, 1]) if len(meas_a) > 2
            else float("nan"),
            # The same spread question asked of the raw channel, which is what a
            # reader sees in the along-route figure.
            "kup_sd_measured_Wm2": float(sub["measured_swout_wm2"].std()),
            "kup_sd_model_Wm2": float(sub["sensor_shortwave_up_Wm2"].std()),
        }
        return out

    result = {"pooled": _block(d, both_sunlit)}
    # Whole-daytime channel spread, shading included -- this is the number the
    # along-route figure's K-up panel displays.
    result["pooled"]["kup_sd_measured_all_day_Wm2"] = float(d["measured_swout_wm2"].std())
    result["pooled"]["kup_sd_model_all_day_Wm2"] = float(d["sensor_shortwave_up_Wm2"].std())
    result["by_case"] = {}
    for case, g in d.groupby("case_id"):
        gm = ((g["measured_swin_wm2"].values > 0.5 * g["trec_atmospheric_ghi_wm2"].values) &
              (g["sensor_shortwave_down_Wm2"].values > 0.5 * g["trec_atmospheric_ghi_wm2"].values))
        if gm.sum() >= 10:
            result["by_case"][case] = _block(g, gm)
    return result


def solar_forcing_audit(root: Path, points: pd.DataFrame) -> dict:
    """Audit the one place the validation is not fully independent.

    The atmospheric shortwave forcing is fitted to the unshaded upper envelope
    of the same mobile pyranometer whose samples the K-down channel is later
    compared against, so that channel's LEVEL is not an independent test; only
    its spatial structure is. This reads back what the fit actually did, from
    each case's own provenance record, and measures the residual level offset
    on samples both sides class as sunlit -- the quantity a reader needs in
    order to see exactly how much the fit did and did not constrain.

    The envelope diagnostics also record that the measured unshaded samples sit
    ABOVE the brightest clear sky the radiative model can represent, which the
    forcing fit declines to chase. That surplus is carried here rather than
    silently dropped.
    """
    cases: dict = {}
    for case in CASES:
        prov = root / "input" / case / "weather" / "weather_from_sensors_provenance.json"
        entry: dict = {}
        if prov.is_file():
            solar = json.loads(prov.read_text()).get("solar", {})
            env = solar.get("envelope_diagnostics", {})
            entry.update({
                "measured_swin_max_wm2": env.get("measured_swin_max_wm2"),
                "clear_sky_ghi_max_wm2": env.get("clear_ghi_max_wm2"),
                "residual_instrument_bias_fraction":
                    env.get("residual_instrument_bias_fraction"),
                "samples_above_physical_ceiling_fraction":
                    env.get("samples_above_physical_ceiling_fraction"),
                "ceiling_linke_turbidity": env.get("ceiling_linke_turbidity"),
                "baseline_linke_turbidity": solar.get("baseline_linke_turbidity"),
            })
        g = points[(points["case_id"] == case) & (points["period"] == "day") &
                   (points["trec_atmospheric_ghi_wm2"] > 50)].dropna(
            subset=["measured_swin_wm2", "sensor_shortwave_down_Wm2",
                    "trec_atmospheric_ghi_wm2"])
        if len(g):
            ghi = g["trec_atmospheric_ghi_wm2"].values
            both = ((g["measured_swin_wm2"].values > 0.5 * ghi) &
                    (g["sensor_shortwave_down_Wm2"].values > 0.5 * ghi))
            if both.sum() >= 10:
                s = g[both]
                entry["n_both_sunlit"] = int(both.sum())
                entry["sunlit_kdown_measured_Wm2"] = float(s["measured_swin_wm2"].mean())
                entry["sunlit_kdown_model_Wm2"] = float(s["sensor_shortwave_down_Wm2"].mean())
                entry["sunlit_kdown_bias_Wm2"] = float(
                    (s["sensor_shortwave_down_Wm2"] - s["measured_swin_wm2"]).mean())
            # Hard physical ceiling on the measurement itself. Horizontal GHI
            # cannot exceed the extraterrestrial horizontal irradiance except
            # for instants of cloud-edge enhancement, so a sustained excess is
            # an instrument statement, not a sky one. Computed on the measured
            # series only; nothing modelled depends on it.
            t = pd.to_datetime(g["timestamp_utc"], utc=True, format="ISO8601")
            zenith = pvlib.solarposition.get_solarposition(
                t, g["latitude"].mean(), g["longitude"].mean())["zenith"].values
            mu = np.clip(np.cos(np.radians(zenith)), 1e-3, None)
            toa = np.asarray(pvlib.irradiance.get_extra_radiation(
                t.dt.dayofyear.to_numpy())) * mu
            ratio = g["measured_swin_wm2"].values / toa
            entry["n_daytime"] = int(len(g))
            entry["fraction_above_toa_horizontal"] = float(np.mean(ratio > 1.0))
            entry["max_measured_over_toa_horizontal"] = float(np.nanmax(ratio))
            entry["max_solar_elevation_deg"] = float(90.0 - zenith.min())
        if entry:
            cases[case] = entry

    biases = [c["sunlit_kdown_bias_Wm2"] for c in cases.values()
              if "sunlit_kdown_bias_Wm2" in c]
    surplus = [c["residual_instrument_bias_fraction"] for c in cases.values()
               if c.get("residual_instrument_bias_fraction") is not None]
    ceiling = [c["samples_above_physical_ceiling_fraction"] for c in cases.values()
               if c.get("samples_above_physical_ceiling_fraction") is not None]
    n_tot = sum(c.get("n_daytime", 0) for c in cases.values())
    n_over = sum(c.get("n_daytime", 0) * c.get("fraction_above_toa_horizontal", 0.0)
                 for c in cases.values())
    return {
        "by_case": cases,
        "pooled_fraction_above_toa_horizontal": (n_over / n_tot) if n_tot else None,
        "sunlit_kdown_bias_range_Wm2": [min(biases), max(biases)] if biases else None,
        "residual_instrument_bias_range": [min(surplus), max(surplus)] if surplus else None,
        "above_physical_ceiling_range": [min(ceiling), max(ceiling)] if ceiling else None,
    }


def table_radiometer(points: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Channel & Period & $n$ & Meas. & Model & MBE & RMSE \\",
        r"\multicolumn{3}{l}{} & \multicolumn{4}{c}{(W\,m$^{-2}$)} \\",
        r"\midrule",
    ]
    for mcol, ocol, label in CHANNELS:
        clean = label.replace("$", "").replace(r"\downarrow", " down").replace(
            r"\uparrow", " up").replace("K", "").replace("L", "").strip()
        name = label.split("$")[0].strip()
        for period in ("day", "night"):
            d = points[(points["period"] == period)].dropna(subset=[mcol, ocol])
            if d.empty:
                continue
            st = stats(d[mcol].values, d[ocol].values)
            shown = name if period == "day" else ""
            lines.append(
                f"{shown} & {period} & {st['n']} & {st['measured_mean']:.1f} & "
                f"{st['model_mean']:.1f} & {st['mbe']:+.1f} & {st['rmse']:.1f} \\\\")
        lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_globe(points: pd.DataFrame) -> str:
    d = points.copy()
    if "globe_spinup_affected" in d:
        d = d[~d["globe_spinup_affected"].astype(bool)]
    d = d.dropna(subset=["measured_black_globe_temperature_c",
                         "globe_transient_temperature_C"])
    lines = [
        # Nine columns overflow the elsarticle text block at default column
        # separation; 4pt keeps it inside without shrinking the font further.
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Case & Period & $n$ & Measured & Emulated & MBE & RMSE & cRMSE & "
        r"$\sigma_{\mathrm{obs}}$ \\",
        r" & & & (\si{\celsius}) & (\si{\celsius}) & (K) & (K) & (K) & (K) \\",
        r"\midrule",
    ]
    for case in CASES:
        for period in ("day", "night"):
            s = d[(d["case_id"] == case) & (d["period"] == period)]
            if s.empty:
                continue
            st = stats(s["globe_transient_temperature_C"].values,
                       s["measured_black_globe_temperature_c"].values)
            shown = case.replace("lisbon", "Lisbon ") if period == "day" else ""
            lines.append(
                f"{shown} & {period} & {st['n']} & {st['measured_mean']:.1f} & "
                f"{st['model_mean']:.1f} & {st['mbe']:+.2f} & {st['rmse']:.2f} & "
                f"{st['crmse']:.2f} & {st['sd_obs']:.2f} \\\\")
    lines.append(r"\midrule")
    for period in ("day", "night"):
        s = d[d["period"] == period]
        st = stats(s["globe_transient_temperature_C"].values,
                   s["measured_black_globe_temperature_c"].values)
        lines.append(
            rf"\textbf{{Pooled}} & {period} & {st['n']} & {st['measured_mean']:.1f} & "
            f"{st['model_mean']:.1f} & {st['mbe']:+.2f} & {st['rmse']:.2f} & "
            f"{st['crmse']:.2f} & {st['sd_obs']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--paper-dir", type=Path,
                    default=Path("/home/harshin/files/fastUTEC paper"))
    args = ap.parse_args()
    out = args.paper_dir
    out.mkdir(parents=True, exist_ok=True)

    points = load_points(args.root)
    points, qc_audit = apply_measurement_qc(points)
    print(f"measurement QC: rejected {qc_audit['n_rejected']} of "
          f"{qc_audit['n_before']:,} samples "
          f"({qc_audit['rejected_fraction']:.2%}) -- {qc_audit['rule']}")
    if qc_audit["by_case"]:
        print(f"  by case: {qc_audit['by_case']}  extent {qc_audit.get('extent_m')}")
    print(f"pooled comparison points: {len(points):,} from {points.case_id.nunique()} cases")
    aligned = points["simulation_date_matches_observation"].astype(bool)
    print(f"  date-aligned {int(aligned.sum()):,} / date-mismatched {int((~aligned).sum()):,}")

    globe = figure_globe(points, out)
    radio = figure_radiometer(points, out)
    shadow = shadow_registration(points, out)
    budget = globe_flux_budget(points)
    surface = surface_temperature_budget(points)
    albedo = albedo_skill(points)
    forcing = solar_forcing_audit(args.root, points)
    spin = points[~points["globe_spinup_affected"].astype(bool)] \
        if "globe_spinup_affected" in points else points
    within_r = {p: within_case_r(spin[spin["period"] == p],
                                 "globe_transient_temperature_C",
                                 "measured_black_globe_temperature_c")
                for p in ("day", "night")}
    autocorr = autocorrelation_audit(spin)
    print("\nINDEPENDENT INFORMATION (measured globe autocorrelation)")
    print(f"  n={autocorr['n_total']}  ->  n_effective ~ {autocorr['n_effective_total']}  "
          f"({autocorr['effective_fraction']:.1%} of nominal)")
    (out / "table_radiometer.tex").write_text(table_radiometer(points) + "\n")
    (out / "table_globe.tex").write_text(table_globe(points) + "\n")

    print("\nBLACK GLOBE (pooled, spin-up excluded)")
    for period, st in globe.items():
        print(f"  {period:5s} n={st['n']:5d}  measured {st['measured_mean']:5.2f}  "
              f"model {st['model_mean']:5.2f}  MBE {st['mbe']:+.2f}  "
              f"RMSE {st['rmse']:.2f}  cRMSE {st['crmse']:.2f}  "
              f"sd_obs {st['sd_obs']:.2f}  r={st['r']:.3f} "
              f"(within-case r={within_r[period]:.3f})")
    print("\nRADIOMETER (pooled)")
    for label, st in radio.items():
        print(f"  {label[:26]:26s} n={st['n']:5d}  MBE {st['mbe']:+7.1f}  "
              f"RMSE {st['rmse']:6.1f}  r={st['r']:.3f}")

    print("\nSHADOW REGISTRATION (daytime)")
    print(f"  sunlit fraction  measured {shadow['measured_sunlit_fraction']:.1%}  "
          f"model {shadow['model_sunlit_fraction']:.1%}")
    print(f"  class agreement  {shadow['classification_agreement']:.1%}  "
          f"(model-sun/meas-shade {shadow['model_sun_measured_shade']:.1%}, "
          f"model-shade/meas-sun {shadow['model_shade_measured_sun']:.1%})")
    print(f"  K-down RMSE  agree {shadow['rmse_agree']:.0f}  "
          f"disagree {shadow['rmse_disagree']:.0f}  all {shadow['rmse_all']:.0f} W/m2")

    print("\nGLOBE FLUX BUDGET (absorbed flux inverted from the measurement)")
    for period, b in budget.items():
        print(f"  {period:5s} model {b['model_absorbed_Wm2']:6.1f}  "
              f"implied {b['implied_absorbed_Wm2']:6.1f}  "
              f"excess {b['excess_Wm2']:+6.1f} W/m2  -> "
              f"dTg {b['implied_bias_K']:+.2f} K (actual {b['actual_bias_K']:+.2f} K)")

    print("\nGROUND ALBEDO SKILL (samples both sides class as sunlit)")
    a = albedo["pooled"]
    print(f"  n={a['n_both_sunlit']:5d}  effective albedo: "
          f"measured {a['measured_albedo_mean']:.3f} (sd {a['measured_albedo_sd']:.3f})  "
          f"model {a['model_albedo_mean']:.3f} (sd {a['model_albedo_sd']:.3f})  "
          f"r={a['albedo_r']:+.3f}")
    print(f"  K-up spread over all daytime samples: measured "
          f"{a['kup_sd_measured_all_day_Wm2']:.1f}  model "
          f"{a['kup_sd_model_all_day_Wm2']:.1f} W/m2")
    for case, b in albedo["by_case"].items():
        print(f"    {case:8s} n={b['n_both_sunlit']:4d}  "
              f"meas {b['measured_albedo_mean']:.3f}  model {b['model_albedo_mean']:.3f}  "
              f"r={b['albedo_r']:+.3f}")

    print("\nSOLAR FORCING AUDIT (K-down level is fitted, not independent)")
    for case, b in forcing["by_case"].items():
        print(f"  {case:8s} sunlit K-down measured "
              f"{b.get('sunlit_kdown_measured_Wm2', float('nan')):6.0f}  model "
              f"{b.get('sunlit_kdown_model_Wm2', float('nan')):6.0f}  bias "
              f"{b.get('sunlit_kdown_bias_Wm2', float('nan')):+6.0f} W/m2   "
              f"unshaded surplus over brightest clear sky "
              f"{100 * (b.get('residual_instrument_bias_fraction') or float('nan')):.1f}% "
              f"on {100 * (b.get('samples_above_physical_ceiling_fraction') or float('nan')):.0f}% "
              f"of samples; {100 * b.get('fraction_above_toa_horizontal', float('nan')):.1f}% "
              f"above TOA horizontal (max {b.get('max_measured_over_toa_horizontal', float('nan')):.2f}x)")
    print(f"  POOLED   {100 * (forcing['pooled_fraction_above_toa_horizontal'] or 0):.1f}% "
          f"of daytime samples exceed the extraterrestrial horizontal irradiance")

    if "amplitude" in surface:
        a = surface["amplitude"]
        print("\nSURFACE TEMPERATURE (inverted from the longwave channels)")
        for period in ("day", "night"):
            b = surface[period]
            print(f"  {period:5s} measured {b['measured_surface_C']:6.2f}  "
                  f"model {b['model_surface_C']:6.2f}  bias {b['surface_bias_K']:+.2f} K")
        print(f"  amplitude measured {a['measured_K']:.2f} K  model {a['model_K']:.2f} K "
              f"({a['excess_fraction']:+.0%})")
        print(f"  mean offset {a['mean_offset_K']:+.2f} K, amplitude error "
              f"{a['amplitude_bias_K']:+.2f} K -> admittance factor needed "
              f"{a['admittance_factor_needed']:.2f}")

    summary = {"n_points": int(len(points)), "measurement_qc": qc_audit,
               "globe": globe, "shadow": shadow,
               "globe_flux_budget": budget, "globe_within_case_r": within_r,
               "surface_temperature": surface,
               "albedo_skill": albedo,
               "autocorrelation": autocorr,
               "solar_forcing_audit": forcing,
               "radiometer": {k: v for k, v in radio.items()},
               "date_aligned": int(aligned.sum()),
               "date_mismatched": int((~aligned).sum())}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote figures and tables to {out}")


if __name__ == "__main__":
    main()
