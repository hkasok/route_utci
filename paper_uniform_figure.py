#!/usr/bin/env python3
"""paper_uniform_figure.py -- redraw the signed-error heatmaps of the Lisbon
uniform-meteorology test (Supplementary Fig. S3) with publication labels.

STANDALONE, plotting only. Reads the per-cell results written by
compute_effect_uniform.py (run_output/impact_uniform/uniform_sensitivity_all_routes.csv);
no JOS-3 run is repeated, so the numbers are exactly those of that study.

Usage: python3 paper_uniform_figure.py [--results CSV] [--out PNG]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent


def offset_label(k: float) -> str:
    if k == 0:
        return "μ"
    sign = "+" if k > 0 else "−"
    mag = abs(k)
    return f"μ{sign}{'' if mag == 1 else f'{mag:g}'}σ"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path,
                    default=ROOT / "run_output" / "impact_uniform" / "uniform_sensitivity_all_routes.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "run_output" / "impact_uniform" / "figures"
                    / "signed_error_all_routes_paper.png")
    args = ap.parse_args()
    R = pd.read_csv(args.results)
    R = R[R.status == "ok"]
    lim = float(R.absolute_error_C.max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    cases = sorted(R.route_id.unique(), key=lambda s: int(s.split(":")[0].replace("lisbon", "")))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))
    image = None
    for ax, cid in zip(axes.ravel(), cases):
        g = R[R.route_id == cid]
        ks_t = np.sort(g.Ta_offset_sigma.unique())
        ks_e = np.sort(g.vp_offset_sigma.unique())[::-1]          # high e at the top
        z = np.full((len(ks_e), len(ks_t)), np.nan)
        for i, ke in enumerate(ks_e):
            for j, kt in enumerate(ks_t):
                z[i, j] = g[(g.Ta_offset_sigma == kt) & (g.vp_offset_sigma == ke)].signed_error_C.iloc[0]
        image = ax.imshow(z, cmap="RdBu_r", norm=norm, aspect="auto")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j, i, f"{z[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(z[i, j]) > 0.6 * lim else "black")
        ta = {k: g[g.Ta_offset_sigma == k].Ta_uniform_C.iloc[0] for k in ks_t}
        vp = {k: g[g.vp_offset_sigma == k].vp_uniform.iloc[0] for k in ks_e}
        ax.set_xticks(range(len(ks_t)), [f"{offset_label(k)}\n{ta[k]:.1f}" for k in ks_t], fontsize=8)
        ax.set_yticks(range(len(ks_e)), [f"{offset_label(k)}\n{vp[k]:.1f}" for k in ks_e], fontsize=8)
        n = int(cid.split(":")[0].replace("lisbon", ""))
        ax.set_title(f"Lisbon {n} (measured forcing: "
                     f"{g.DeltaTcore_real_C.iloc[0]:.3f} °C)", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Uniform air temperature $T_a$ (°C)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Uniform vapour pressure $e$ (hPa)")
    cb = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("Signed error in $\\Delta T_{\\mathrm{core}}$, uniform − measured forcing (°C)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250, bbox_inches="tight")
    print(f"wrote {args.out}  (max |error| {lim:.4f} C)")


if __name__ == "__main__":
    main()
