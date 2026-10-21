#!/usr/bin/env python3
"""paper_along_route_figure.py -- Supplementary Fig. S2 (along_route_radiometer.png).

Redraws, for the paper only, the figure that compare_mrt_lisbon_data.py
writes as route_<id>_<name>_radiometer_along_route_comparison.png
(plot_day_radiometer_along_route): measured against emulated radiometer
channels along one daytime walk. It reads the per-sample CSV that script
already saved (radiant_flux_comparison_points.csv) and writes only to --out,
so nothing under run_output is touched.

Same panels, channels, statistics and line data as the pipeline figure; only
the presentation changes for print: drawn near its printed size with
9-10 pt text, the legend above the panels rather than over the data, and the
model line in Okabe-Ito blue so it separates from the black measured line in
greyscale and under protanopia.

Usage:
    python3 paper_adhoc/paper_along_route_figure.py --out FILE.png
        [--csv run_output/lisbon1/validation/mrt_lisbon/radiant_flux_comparison_points.csv]
        [--route-id 1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = (ROOT / "run_output" / "lisbon1" / "validation" / "mrt_lisbon"
               / "radiant_flux_comparison_points.csv")

# (measured column, model column, title) -- as in
# compare_mrt_lisbon_data.plot_day_radiometer_along_route
SPECIFICATIONS = [
    ("measured_swin_wm2", "sensor_shortwave_down_Wm2", "Downwelling shortwave"),
    ("measured_swout_wm2", "sensor_shortwave_up_Wm2", "Upwelling shortwave"),
    ("measured_lwin_wm2", "sensor_longwave_down_Wm2", "Downwelling longwave"),
    ("measured_lwout_wm2", "sensor_longwave_up_Wm2", "Upwelling longwave"),
]
MEASURED_COLOUR = "black"
MODEL_COLOUR = "#0072B2"   # Okabe-Ito blue

# Printed at 0.92 text width (about 5.0 in): 9-10 pt here prints at 7-7.5 pt.
plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--route-id", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    route = frame[frame["period"].eq("day") & frame["route_id"].eq(args.route_id)]
    if route.empty:
        raise SystemExit(f"no daytime samples for route {args.route_id} in {args.csv}")
    route = route.sort_values("seq")
    distance = route["distance_along_route_m"].to_numpy(float)
    available = [s for s in SPECIFICATIONS
                 if s[0] in route.columns and s[1] in route.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(6.5, 6.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (measured_column, model_column, title) in zip(axes, available):
        measured = route[measured_column].to_numpy(float)
        modelled = route[model_column].to_numpy(float)
        ax.plot(distance, measured, color=MEASURED_COLOUR, lw=1.0,
                label="Measured (four-component radiometer)")
        ax.plot(distance, modelled, color=MODEL_COLOUR, lw=0.9,
                label="TREC-Route (emulated radiometer)")
        residual = modelled - measured
        valid = np.isfinite(measured) & np.isfinite(modelled)
        correlation = (float(stats.pearsonr(measured[valid],
                                            modelled[valid]).statistic)
                       if valid.sum() > 2 and np.std(measured[valid]) > 0
                       and np.std(modelled[valid]) > 0 else float("nan"))
        ax.set_ylabel("Irradiance\n(W m$^{-2}$)")
        ax.set_title(f"{title}; mean bias {residual[valid].mean():+.1f} "
                     f"W m$^{{-2}}$, r = {correlation:.2f}", pad=3)
        ax.grid(alpha=0.25)
        print(f"{title:22s} mean bias {residual[valid].mean():+.1f}  r {correlation:.2f}")
    axes[-1].set_xlabel("Distance along measured route (m)")

    case = str(route["case_id"].iloc[0])
    case_label = f"Lisbon {case[len('lisbon'):]}" if case.startswith("lisbon") else case
    # panels below a top band that holds the title and, under it, the legend:
    # the legend never sits over the data
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=0.6)
    fig.suptitle(f"{case_label}, daytime walk: radiometer channels along "
                 "the route (like-for-like)", fontsize=10, y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=2, frameon=False, handlelength=2.0, columnspacing=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
