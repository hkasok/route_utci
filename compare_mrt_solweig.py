"""
compare_mrt_solweig.py -- STANDALONE comparison of per-route MRT (and UTCI)
between this pipeline's route output and a SOLWEIG route output..

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


def dist_row(vals, **extra):
    """Distribution summary (min / p10 / p50 / p90 / max / mean) of an array."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    row = dict(extra)
    if len(v):
        row.update(n=int(len(v)), min=float(v.min()),
                   p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)),
                   p90=float(np.percentile(v, 90)), max=float(v.max()),
                   mean=float(v.mean()))
    else:
        row.update(n=0, min=np.nan, p10=np.nan, p50=np.nan, p90=np.nan,
                   max=np.nan, mean=np.nan)
    return row


def gap_by_level(ours, solweig, edges=(0, 30, 35, 40, 45, 50, 55, 100)):
    """Mean SOLWEIG-minus-ours gap binned by OUR Tmrt, to reveal whether the
    disagreement is in cool/shaded points or hot/sunlit points."""
    o = np.asarray(ours, float); s = np.asarray(solweig, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]; d = s - o
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (o >= lo) & (o < hi)
        if sel.sum():
            rows.append(dict(our_tmrt_bin=f"{lo}-{hi}", n=int(sel.sum()),
                             ours_mean=float(o[sel].mean()),
                             solweig_mean=float(s[sel].mean()),
                             mean_gap=float(d[sel].mean())))
    return pd.DataFrame(rows)


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
    ap.add_argument("--air-temp-c", type=float, default=None,
                    help="If given, flag points whose Tmrt is below air temperature "
                         "(physically implausible outdoors in daytime) in the distribution "
                         "report -- useful for spotting the cool-wall / deep-shade artifact.")
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

    # ------------------------------------------------------------------
    # RICHER DIAGNOSTICS (added). These are NEW outputs; the existing
    # comparison_stats.csv / comparison_points.csv above are unchanged.
    # ------------------------------------------------------------------
    pts = pd.DataFrame(point_rows)

    # 1) Tmrt DISTRIBUTION per route + overall, for ours and SOLWEIG. This is
    #    what exposes a floor/spread difference that a single mean/bias hides
    #    (e.g. SOLWEIG holding a high shade floor while ours drops much lower).
    dist_rows = []
    for rid in routes:
        p = pts[pts.route_id == rid]
        dist_rows.append(dist_row(p.tmrt_ours, route_id=rid, model="ours"))
        dist_rows.append(dist_row(p.tmrt_solweig, route_id=rid, model="solweig"))
    dist_rows.append(dist_row(pts.tmrt_ours, route_id="ALL", model="ours"))
    dist_rows.append(dist_row(pts.tmrt_solweig, route_id="ALL", model="solweig"))
    ddf = pd.DataFrame(dist_rows)[["route_id", "model", "n", "min", "p10",
                                   "p50", "p90", "max", "mean"]]

    # 2) Where does the gap live? Mean gap binned by OUR Tmrt level.
    lvl = gap_by_level(pts.tmrt_ours.values, pts.tmrt_solweig.values)

    print("\n" + "=" * 70)
    print("Tmrt DISTRIBUTION (min / p10 / p50 / p90 / max) -- reveals the floor:")
    for _, r in ddf.iterrows():
        print(f"  route {str(r.route_id):>3} {r.model:<7}: "
              f"min {r['min']:5.1f}  p10 {r.p10:5.1f}  p50 {r.p50:5.1f}  "
              f"p90 {r.p90:5.1f}  max {r['max']:5.1f}")
    print("\nGap (SOLWEIG-ours) by OUR Tmrt level -- is the gap in cool or hot points?")
    for _, r in lvl.iterrows():
        print(f"  our Tmrt {r.our_tmrt_bin:>6}: n={int(r.n):>4}  "
              f"ours {r.ours_mean:5.1f} -> SOLWEIG {r.solweig_mean:5.1f}  "
              f"gap {r.mean_gap:+5.1f}")
    if args.air_temp_c is not None:
        below = pts.tmrt_ours < args.air_temp_c
        print(f"\nPhysical-plausibility check (air temp {args.air_temp_c:.1f} C):")
        print(f"  ours: {int(below.sum())}/{len(pts)} points "
              f"({100*below.mean():.1f}%) have Tmrt BELOW air temp -- "
              f"implausible outdoors in daytime (cool-wall / deep-shade artifact).")
        belows = pts.tmrt_solweig < args.air_temp_c
        print(f"  SOLWEIG: {int(belows.sum())}/{len(pts)} points "
              f"({100*belows.mean():.1f}%).")

    if out_dir:
        fig.savefig(out_dir / "mrt_utci_comparison.png", dpi=150)
        sdf.to_csv(out_dir / "comparison_stats.csv", index=False)
        pts.to_csv(out_dir / "comparison_points.csv", index=False)
        # NEW files (do not overwrite the two above):
        ddf.to_csv(out_dir / "comparison_distribution.csv", index=False)
        lvl.to_csv(out_dir / "comparison_gap_by_level.csv", index=False)

        # Scatter (ours vs SOLWEIG) with 1:1 line + diff histogram: shows the
        # systematic offset, the spread, and how much is a shift vs scatter.
        o = pts.tmrt_ours.values; s = pts.tmrt_solweig.values
        fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5.5))
        for rid in routes:
            p = pts[pts.route_id == rid]
            ax2[0].scatter(p.tmrt_ours, p.tmrt_solweig, s=6, alpha=0.4,
                           label=f"Route {rid}")
        lo = float(np.nanmin([o.min(), s.min()])); hi = float(np.nanmax([o.max(), s.max()]))
        ax2[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax2[0].set_xlabel("ours Tmrt [C]"); ax2[0].set_ylabel("SOLWEIG Tmrt [C]")
        ax2[0].set_title("Point Tmrt: ours vs SOLWEIG"); ax2[0].legend(fontsize=8)
        ax2[0].set_aspect("equal"); ax2[0].grid(alpha=0.3)
        d = s - o; d = d[np.isfinite(d)]
        ax2[1].hist(d, bins=40, color="#8888cc", edgecolor="white")
        ax2[1].axvline(0, color="k", lw=1); ax2[1].axvline(float(d.mean()), color="r", lw=1.5,
                     label=f"mean {d.mean():+.1f}")
        ax2[1].set_xlabel("SOLWEIG - ours Tmrt [C]"); ax2[1].set_ylabel("count")
        ax2[1].set_title("Tmrt difference distribution"); ax2[1].legend(fontsize=8)
        ax2[1].grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(out_dir / "comparison_scatter_diff.png", dpi=150)
        plt.close(fig2)

        print(f"\nSaved to {out_dir}/: mrt_utci_comparison.png, "
              f"comparison_stats.csv, comparison_points.csv, "
              f"comparison_distribution.csv, comparison_gap_by_level.csv, "
              f"comparison_scatter_diff.png")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
