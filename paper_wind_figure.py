#!/usr/bin/env python3
"""Publication figure: pedestrian-level 2-D potential-flow wind (stage 05e).

Panels
  (a) east-basis speed ratio |u_E|/U   (unit free stream travelling east)
  (b) north-basis speed ratio |u_N|/U  (unit free stream travelling north)
  (c) superposed field for one wind direction, u = d_x u_E + d_y u_N,
      with streamlines and the measured pedestrian route.

Everything is read from the stored stage-05e outputs (direction_basis.npz,
cell_fluid_fraction.npy, x/y coordinates, metadata); nothing in run_output is
written. Buildings are drawn from the cut-cell solid fraction (1 - alpha), i.e.
the pedestrian-level cross-sections exactly as the solver sees them.

    python3 paper_wind_figure.py [--case lisbon3] [--direction 330]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from pedestrian_flow_field import PedestrianFlowField

PROJECT = Path(__file__).resolve().parent
DEFAULT_OUT = Path("/home/harshin/files/fastUTEC paper/pedestrian_wind_field")

# Crop windows (local metres, x0, x1, y0, y1) chosen around the route.
WINDOWS = {"lisbon3": (500.0, 1060.0, 1250.0, 1810.0)}

VMAX = 3.0            # colour-scale ceiling for |u|/U_ref (extend='max')
ROUTE_COLOUR = "#D55E00"   # Okabe-Ito vermilion
BUILDING_GREY = 0.62


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="lisbon3")
    p.add_argument("--direction", type=float, default=330.0,
                   help="meteorological FROM bearing for panel (c), deg")
    p.add_argument("--route", type=int, default=1)
    p.add_argument("--window", type=float, nargs=4, default=None,
                   metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="output path without extension (.png and .pdf written)")
    return p.parse_args()


def to_vector(from_deg: float) -> np.ndarray:
    b = np.deg2rad(from_deg)
    return np.array([-np.sin(b), -np.cos(b)])


def direction_inset(ax, vec, label):
    """Small boxed arrow showing the free-stream direction (flow toward)."""
    cx, cy, r = 0.10, 0.135, 0.045
    ax.add_patch(Rectangle((cx - 0.08, cy - 0.11), 0.16, 0.20,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor="0.3", linewidth=0.5, alpha=0.92,
                           zorder=9))
    ax.add_patch(FancyArrowPatch((cx - r * vec[0], cy - r * vec[1] + 0.03),
                                 (cx + r * vec[0], cy + r * vec[1] + 0.03),
                                 transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=7, linewidth=1.1, color="k",
                                 zorder=10))
    ax.text(cx, cy - 0.10, label, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, zorder=10)


def scale_bar(ax, x0, y0, length):
    ax.add_patch(Rectangle((x0, y0), length, 5, facecolor="k",
                           edgecolor="white", linewidth=0.4, zorder=9))
    ax.text(x0 + length / 2, y0 + 10, f"{int(length)} m", ha="center",
            va="bottom", fontsize=7, zorder=9,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def main():
    args = parse_args()
    wind_dir = PROJECT / "run_output" / args.case / "pedestrian_wind"
    meta = json.loads((wind_dir / "potential_flow_metadata.json").read_text())
    field = PedestrianFlowField(str(wind_dir))
    basis = np.load(wind_dir / "direction_basis.npz")
    x = np.load(wind_dir / "x_coordinates.npy")
    y = np.load(wind_dir / "y_coordinates.npy")
    alpha = np.load(wind_dir / "cell_fluid_fraction.npy")
    dx = float(meta["grid"]["spacing_m"])

    x0, x1, y0, y1 = args.window or WINDOWS.get(
        args.case, (x[0], x[-1], y[0], y[-1]))
    ix = np.where((x >= x0) & (x <= x1))[0]
    iy = np.where((y >= y0) & (y <= y1))[0]
    sx, sy = slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1)
    xc, yc = x[sx], y[sy]
    xe = np.concatenate([xc - dx / 2, [xc[-1] + dx / 2]])
    ye = np.concatenate([yc - dx / 2, [yc[-1] + dx / 2]])
    a = alpha[sy, sx]
    fluid = a > 0

    uE, vE = basis["u_east"][sy, sx], basis["v_east"][sy, sx]
    uN, vN = basis["u_north"][sy, sx], basis["v_north"][sy, sx]
    uC, vC = field.basis_velocity(args.direction)
    uC, vC = uC[sy, sx], vC[sy, sx]
    fields = [(uE, vE), (uN, vN), (uC, vC)]
    speeds = [np.where(fluid, np.hypot(u, v), np.nan) for u, v in fields]

    route = pd.read_csv(PROJECT / "input" / args.case / "routes"
                        / f"route_{args.route}.csv")
    rx, ry = route["x_local_m"].to_numpy(), route["y_local_m"].to_numpy()

    # Building layer: opacity = cut-cell solid fraction (1 - alpha).
    bld = np.zeros(a.shape + (4,))
    bld[..., :3] = BUILDING_GREY
    bld[..., 3] = np.clip(1.0 - a, 0.0, 1.0)

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.72),
                             gridspec_kw={"wspace": 0.06})
    cmap = plt.get_cmap("viridis").copy()
    labels = ["(a) East basis  $|\\mathbf{u}_E|/U$",
              "(b) North basis  $|\\mathbf{u}_N|/U$",
              f"(c) Superposed, wind from {args.direction:.0f}°"]
    insets = [(np.array([1.0, 0.0]), "E"),
              (np.array([0.0, 1.0]), "N"),
              (to_vector(args.direction), f"{args.direction:.0f}°")]

    for k, (ax, spd, (u, v)) in enumerate(zip(axes, speeds, fields)):
        im = ax.pcolormesh(xe, ye, spd, cmap=cmap, vmin=0, vmax=VMAX,
                           shading="flat", rasterized=True)
        ax.imshow(bld, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]),
                  interpolation="nearest", zorder=2)
        ax.contour(xc, yc, a, levels=[0.5], colors="0.25", linewidths=0.3,
                   zorder=3)
        if k == 2:
            us = np.where(fluid, u, 0.0)
            vs = np.where(fluid, v, 0.0)
            ax.streamplot(xc, yc, us, vs, density=1.4, color="black",
                          linewidth=0.4, arrowsize=0.5, arrowstyle="-|>",
                          broken_streamlines=True, zorder=4)
            ax.plot(rx, ry, color=ROUTE_COLOUR, linewidth=1.6, zorder=6,
                    solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=2.8, foreground="k"),
                                  pe.Normal()])
        else:
            ax.plot(rx, ry, color=ROUTE_COLOUR, linewidth=1.0, zorder=6,
                    alpha=0.9, linestyle=(0, (3, 1.5)),
                    path_effects=[pe.Stroke(linewidth=1.9, foreground="k"),
                                  pe.Normal()])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(labels[k], fontsize=7.5, loc="left", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        direction_inset(ax, *insets[k])
        if k == 0:
            scale_bar(ax, x1 - 130, y0 + 18, 100)

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.03, aspect=45, extend="max",
                        ticks=np.arange(0, VMAX + 0.01, 0.5))
    cbar.set_label("Pedestrian-level speed ratio $|\\mathbf{u}|/U_{ref}$"
                   "  (grey: building cross-section at z = ground + 1.1 m;"
                   "  orange: route)", fontsize=7)
    cbar.ax.tick_params(labelsize=7, length=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(args.out.with_suffix(".pdf"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)

    for name, s in zip(("east", "north", f"{args.direction:.0f}"), speeds):
        vals = s[np.isfinite(s) & (a >= 0.5)]
        print(f"{name:>5}: |u|/U window  min {vals.min():.3f}  p5 "
              f"{np.percentile(vals, 5):.3f}  median {np.median(vals):.3f}  "
              f"p95 {np.percentile(vals, 95):.3f}  p99 "
              f"{np.percentile(vals, 99):.3f}  max {vals.max():.3f}  "
              f"frac>{VMAX}: {(vals > VMAX).mean():.4f}")
    print("route samples in window:",
          int(((rx >= x0) & (rx <= x1) & (ry >= y0) & (ry <= y1)).sum()))
    print("wrote", args.out.with_suffix(".png"), args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
