"""
compare_mrt_solweig.py -- STANDALONE comparison of per-route MRT (and UTCI)
between this pipeline's route output and a SOLWEIG route output.

This is a separate utility: it does NOT import or modify the route_utci
pipeline. Point it at the two sets of per-point CSVs and it prints agreement
statistics and shows/saves overlay plots.

Along a route the air temperature, humidity, and wind are spatially uniform,
so MRT is the ONLY spatially varying driver -- any per-point UTCI difference
between the two models comes from MRT. The script therefore reports MRT
agreement first, UTCI second.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
By DEFAULT it reads the three SOLWEIG files straight from
  /media/harshin/data_drive/solweig/route_results_solweig/route_*_point_results.csv
and our pipeline's
  run_output/viz/route_utci/routes_points.csv
so on that machine you can just run:

  python3 compare_mrt_solweig.py --output-dir compare_out

Override either side to point elsewhere. Each input may be a single CSV, a
glob, OR several files (one per route). Column names are auto-detected; the
SOLWEIG files use route_id / distance_m / tmrt_c / utci_c (confirmed), and
our pipeline uses route_id / cumdist_m / tmrt_c / utci_c -- both handled.

  # explicit paths
  python3 compare_mrt_solweig.py \
      --solweig "/media/harshin/data_drive/solweig/route_results_solweig/route_*_point_results.csv" \
      --ours    "run_output/viz/route_utci/routes_points.csv" \
      --output-dir compare_out

If --output-dir is omitted, outputs are saved automatically in compare_out.
--------------------------------------------------------------------------
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Column-name aliases (lower-cased match). Extend if your headers differ.
ALIASES = {
    "route": ["route_id", "route", "routeid", "route_no", "route_number"],
    "dist":  ["cumdist_m", "distance_m", "dist_m", "distance", "cumulative_distance_m",
              "s_m", "chainage_m"],
    "tmrt":  ["tmrt_c", "tmrt", "mrt_c", "mrt", "t_mrt_c", "tmrt_degc", "meanradianttemp"],
    "utci":  ["utci_c", "utci", "utci_degc", "utci_c_time"],
}


def find_col(columns, key):
    low = {str(c).strip().lower(): c for c in columns}
    for cand in ALIASES[key]:
        if cand in low:
            return low[cand]
    return None


def route_id_from_name(path):
    m = re.search(r"route[_-]?(\d+)", Path(path).stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def expand_inputs(items):
    """Accept a single glob string, or a list of paths/globs; return files."""
    files = []
    for it in items:
        g = glob.glob(it)
        files.extend(g if g else [it])
    files = [f for f in files if Path(f).exists()]
    return sorted(set(files))


def load_side(inputs, label):
    """Load one model's per-point data into a tidy frame:
    route_id, dist_m, tmrt_c, utci_c (utci may be NaN)."""
    files = expand_inputs(inputs)
    if not files:
        raise SystemExit(f"[{label}] no existing files match: {inputs}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        rcol = find_col(df.columns, "route")
        dcol = find_col(df.columns, "dist")
        tcol = find_col(df.columns, "tmrt")
        ucol = find_col(df.columns, "utci")
        if dcol is None or tcol is None:
            raise SystemExit(f"[{label}] {f}: could not find distance and MRT "
                             f"columns. Has: {list(df.columns)}")
        rid = df[rcol] if rcol else None
        if rid is None:
            fid = route_id_from_name(f)
            if fid is None:
                raise SystemExit(f"[{label}] {f}: no route_id column and no "
                                 f"route number in filename.")
            rid = pd.Series(fid, index=df.index)
        out = pd.DataFrame({
            "route_id": pd.to_numeric(rid, errors="coerce").astype("Int64"),
            "dist_m": pd.to_numeric(df[dcol], errors="coerce"),
            "tmrt_c": pd.to_numeric(df[tcol], errors="coerce"),
            "utci_c": (pd.to_numeric(df[ucol], errors="coerce")
                       if ucol else np.nan),
        })
        frames.append(out.dropna(subset=["route_id", "dist_m", "tmrt_c"]))
    tidy = pd.concat(frames, ignore_index=True)
    tidy["route_id"] = tidy["route_id"].astype(int)
    print(f"[{label}] loaded {len(files)} file(s), routes "
          f"{sorted(tidy.route_id.unique())}, {len(tidy)} points; "
          f"MRT {tidy.tmrt_c.min():.1f}-{tidy.tmrt_c.max():.1f} C")
    return tidy


def resample(sub, grid):
    sub = sub.sort_values("dist_m").drop_duplicates("dist_m")
    tmrt = np.interp(grid, sub["dist_m"], sub["tmrt_c"])
    utci = (np.interp(grid, sub["dist_m"], sub["utci_c"])
            if sub["utci_c"].notna().any() else np.full_like(grid, np.nan))
    return tmrt, utci


def stats(a, b):
    """a = ours, b = solweig; positive bias => SOLWEIG higher."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return dict(bias=np.nan, mae=np.nan, rmse=np.nan, maxabs=np.nan, r=np.nan)
    a, b = a[m], b[m]
    d = b - a
    r = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else np.nan
    return dict(bias=float(d.mean()), mae=float(np.abs(d).mean()),
                rmse=float(np.sqrt((d**2).mean())), maxabs=float(np.abs(d).max()),
                r=float(r))


def main():
    ap = argparse.ArgumentParser(description="Compare per-route MRT/UTCI: "
                                             "our pipeline vs SOLWEIG")
    ap.add_argument("--ours", nargs="+",
                    default=["run_output/viz/route_utci/routes_points.csv"],
                    help="Our pipeline per-point CSV(s) or glob "
                         "(default: run_output/viz/route_utci/routes_points.csv)")
    ap.add_argument("--solweig", nargs="+",
                    default=["/media/harshin/data_drive/solweig/"
                             "route_results_solweig/route_*_point_results.csv"],
                    help="SOLWEIG per-point CSV(s) or glob (default: the 3 "
                         "route_*_point_results.csv in the SOLWEIG "
                         "route_results_solweig folder)")
    ap.add_argument("--output-dir", default="compare_out",
                    help="Directory for plots and CSV outputs "
                         "(default: compare_out)")
    ap.add_argument("--grid-spacing-m", type=float, default=2.0)
    args = ap.parse_args()

    ours = load_side(args.ours, "ours")
    solw = load_side(args.solweig, "solweig")

    routes = sorted(set(ours.route_id) & set(solw.route_id))
    if not routes:
        raise SystemExit(f"No common route_id. ours={sorted(set(ours.route_id))} "
                         f"solweig={sorted(set(solw.route_id))}")
    miss = (set(ours.route_id) ^ set(solw.route_id))
    if miss:
        print(f"NOTE: comparing common routes {routes}; not in both: {sorted(miss)}")

    if args.output_dir:
        matplotlib.use("Agg")
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    stat_rows, point_rows = [], []
    fig, axes = plt.subplots(2, len(routes), figsize=(6*len(routes), 9),
                             squeeze=False)

    for col, rid in enumerate(routes):
        o = ours[ours.route_id == rid]
        s = solw[solw.route_id == rid]
        dmax = min(o.dist_m.max(), s.dist_m.max())
        if dmax <= 0:
            continue
        grid = np.arange(0.0, dmax + 1e-9, args.grid_spacing_m)
        ot, ou = resample(o, grid)
        st_, su = resample(s, grid)

        for var, oa, sa in (("tmrt", ot, st_), ("utci", ou, su)):
            row = stats(oa, sa)
            row.update(route_id=rid, variable=var,
                       ours_mean=float(np.nanmean(oa)), ours_max=float(np.nanmax(oa)),
                       solweig_mean=float(np.nanmean(sa)), solweig_max=float(np.nanmax(sa)),
                       n=int(len(grid)))
            stat_rows.append(row)
        for k in range(len(grid)):
            point_rows.append(dict(route_id=rid, dist_m=float(grid[k]),
                tmrt_ours=float(ot[k]), tmrt_solweig=float(st_[k]),
                tmrt_diff=float(st_[k]-ot[k]),
                utci_ours=float(ou[k]), utci_solweig=float(su[k]),
                utci_diff=float(su[k]-ou[k])))

        ax = axes[0][col]
        ax.plot(grid, ot, color="#1f77b4", lw=2, label="ours (route_utci)")
        ax.plot(grid, st_, color="#d62728", lw=2, ls="--", label="SOLWEIG")
        ax.set_title(f"Route {rid} - MRT"); ax.set_ylabel("Tmrt [C]")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
        m = next(x for x in stat_rows if x["route_id"] == rid and x["variable"] == "tmrt")
        ax.text(0.02, 0.02, f"bias {m['bias']:+.1f}  RMSE {m['rmse']:.1f}  r {m['r']:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        ax = axes[1][col]
        ax.plot(grid, ou, color="#1f77b4", lw=2)
        ax.plot(grid, su, color="#d62728", lw=2, ls="--")
        ax.set_title(f"Route {rid} - UTCI"); ax.set_xlabel("Distance [m]")
        ax.set_ylabel("UTCI [C]"); ax.grid(alpha=0.3)
        u = next(x for x in stat_rows if x["route_id"] == rid and x["variable"] == "utci")
        if np.isfinite(u["bias"]):
            ax.text(0.02, 0.02, f"bias {u['bias']:+.1f}  RMSE {u['rmse']:.1f}  r {u['r']:.2f}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig.suptitle("Per-route MRT & UTCI: route_utci vs SOLWEIG", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    sdf = pd.DataFrame(stat_rows)[["route_id", "variable", "bias", "mae",
        "rmse", "maxabs", "r", "ours_mean", "ours_max", "solweig_mean",
        "solweig_max", "n"]]

    print("\n" + "=" * 70)
    print("MRT agreement (drives the spatial UTCI differences):")
    for _, r in sdf[sdf.variable == "tmrt"].iterrows():
        print(f"  Route {int(r.route_id)}: bias {r.bias:+.2f} C, MAE {r.mae:.2f}, "
              f"RMSE {r.rmse:.2f}, max|d| {r.maxabs:.1f}, r {r.r:.3f}   "
              f"(ours {r.ours_mean:.1f} / SOLWEIG {r.solweig_mean:.1f})")
    print("UTCI agreement:")
    for _, r in sdf[sdf.variable == "utci"].iterrows():
        if np.isfinite(r.bias):
            print(f"  Route {int(r.route_id)}: bias {r.bias:+.2f} C, "
                  f"RMSE {r.rmse:.2f}, r {r.r:.3f}")

    if out_dir:
        fig.savefig(out_dir / "mrt_utci_comparison.png", dpi=150)
        sdf.to_csv(out_dir / "comparison_stats.csv", index=False)
        pd.DataFrame(point_rows).to_csv(out_dir / "comparison_points.csv", index=False)
        print(f"\nSaved to {out_dir}/: mrt_utci_comparison.png, "
              f"comparison_stats.csv, comparison_points.csv")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
