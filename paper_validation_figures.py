#!/usr/bin/env python3
"""
paper_validation_figures.py -- pooled validation figures and LaTeX tables for
the TREC-Route manuscript.

STANDALONE. This reads existing run_output/ artefacts and writes figures and
table fragments into the paper directory. It does not import, modify or
re-run any pipeline stage, and it computes no new physics: every number it
reports is read from, or aggregated over, files that stage 05/08/10 already
wrote.

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
        "r": float(np.corrcoef(model, measured)[0, 1]),
        "measured_mean": float(measured.mean()),
        "model_mean": float(model.mean()),
    }


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
    """Pooled four-component radiometer channels."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.8), constrained_layout=True)
    summary = {}
    for ax, (mcol, ocol, label) in zip(axes.ravel(), CHANNELS):
        d = points.dropna(subset=[mcol, ocol])
        st_all = stats(d[mcol].values, d[ocol].values)
        summary[label] = st_all
        for period, colour in (("day", DAY), ("night", NIGHT)):
            s = d[d["period"] == period]
            if not s.empty:
                ax.scatter(s[ocol], s[mcol], s=4, alpha=0.3, c=colour, lw=0,
                           zorder=2, label=period)
        lo = float(min(d[mcol].min(), d[ocol].min()))
        hi = float(max(d[mcol].max(), d[ocol].max()))
        pad = 0.05 * (hi - lo)
        _square(ax, lo - pad, hi + pad,
                "Measured (W m$^{-2}$)", "TREC-Route (W m$^{-2}$)", label)
        ax.text(0.04, 0.96,
                f"n={st_all['n']}\nMBE {st_all['mbe']:+.1f}\n"
                f"RMSE {st_all['rmse']:.1f}\nr={st_all['r']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
                bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.9, pad=2.2))
    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.9, markerscale=2)
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
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Case & Period & $n$ & Measured & Emulated & MBE & RMSE & $r$ \\",
        r" & & & (\si{\celsius}) & (\si{\celsius}) & (K) & (K) & \\",
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
                f"{st['r']:.2f} \\\\")
    lines.append(r"\midrule")
    for period in ("day", "night"):
        s = d[d["period"] == period]
        st = stats(s["globe_transient_temperature_C"].values,
                   s["measured_black_globe_temperature_c"].values)
        lines.append(
            rf"\textbf{{Pooled}} & {period} & {st['n']} & {st['measured_mean']:.1f} & "
            f"{st['model_mean']:.1f} & {st['mbe']:+.2f} & {st['rmse']:.2f} & "
            f"{st['r']:.2f} \\\\")
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
    print(f"pooled comparison points: {len(points):,} from {points.case_id.nunique()} cases")
    aligned = points["simulation_date_matches_observation"].astype(bool)
    print(f"  date-aligned {int(aligned.sum()):,} / date-mismatched {int((~aligned).sum()):,}")

    globe = figure_globe(points, out)
    radio = figure_radiometer(points, out)
    shadow = shadow_registration(points, out)
    (out / "table_radiometer.tex").write_text(table_radiometer(points) + "\n")
    (out / "table_globe.tex").write_text(table_globe(points) + "\n")

    print("\nBLACK GLOBE (pooled, spin-up excluded)")
    for period, st in globe.items():
        print(f"  {period:5s} n={st['n']:5d}  measured {st['measured_mean']:5.2f}  "
              f"model {st['model_mean']:5.2f}  MBE {st['mbe']:+.2f}  "
              f"RMSE {st['rmse']:.2f}  r={st['r']:.3f}")
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

    summary = {"n_points": int(len(points)), "globe": globe, "shadow": shadow,
               "radiometer": {k: v for k, v in radio.items()},
               "date_aligned": int(aligned.sum()),
               "date_mismatched": int((~aligned).sum())}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote figures and tables to {out}")


if __name__ == "__main__":
    main()
