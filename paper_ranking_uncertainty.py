#!/usr/bin/env python3
"""
paper_ranking_uncertainty.py -- does the validated error change the route choice?

Monte Carlo on the four campus routes. Each draw perturbs the radiant field
every route is walked through, in the two ways the Lisbon validation showed
the model to be wrong, and re-runs JOS-3:

  1. SUN/SHADE FLIPS. A fraction p of route points (the Lisbon per-sample
     misclassification rate) has its beam state inverted: a sunlit point loses
     the direct beam, a shaded point receives the beam a fully sunlit standing
     body would. Flips are spatially correlated along the route with the
     correlation length of the misclassified stretches (exponential
     correlation; a Gaussian field thresholded at the p-quantile).
  2. LOAD OFFSET. A common-mode offset in absorbed radiant flux, drawn from the
     band the globe analysis established, added to every point of every route
     in the draw (the same value for all routes: it is a model bias, not noise).

Everything else (routes, pace, clothing, start state, protocol) is the
stage-09 configuration via campus_walks.CampusCase. Reported per departure:
P(best route unchanged), P(complete ranking unchanged), the distribution of
every pairwise gap and P(gap keeps its sign).

Usage:
    python3 paper_ranking_uncertainty.py --draws 500 --departures 13 16 17
        --flip-rate 0.27 --correlation-length-m 6 --load-offset-range -10 70
"""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from campus_walks import CampusCase

CASE = None     # set in each worker (fork) before the draws run
CLO = {}


def correlated_flips(n, spacing_m, rate, length_m, rng):
    """Boolean flips with marginal rate ``rate`` and exponential correlation."""
    if rate <= 0:
        return np.zeros(n, dtype=bool)
    rho = np.exp(-spacing_m / max(length_m, 1e-6))
    z = np.empty(n)
    z[0] = rng.standard_normal()
    eps = rng.standard_normal(n) * np.sqrt(1.0 - rho ** 2)
    for k in range(1, n):
        z[k] = rho * z[k - 1] + eps[k]
    from scipy.stats import norm
    return z < norm.ppf(rate)


def perturbed_tmrt(case: CampusCase, route, flips, load_offset_Wm2):
    """Tmrt matrix (nt, n_pts) with the beam state inverted at flipped points
    and a uniform flux offset everywhere."""
    flux, beam = case.route_flux(route)
    tau = case.tau[:, route.nearest].astype(float)
    full_beam = case.beam_if_sunlit()[:, None]
    sunlit = tau > 0.5
    # inverted state at flipped points: sunlit -> no beam, shaded -> full beam
    new_tau = np.where(flips[None, :], np.where(sunlit, 0.0, 1.0), tau)
    flux_new = flux + full_beam * (new_tau - tau) + load_offset_Wm2
    return case.tmrt_from_flux(flux_new)


def run_draw(task):
    draw, seed, departures, rate, length_m, offset_lo, offset_hi = task
    rng = np.random.default_rng(seed)
    offset = rng.uniform(offset_lo, offset_hi)
    rows = []
    for rid, route in CASE.routes.items():
        flips = correlated_flips(len(route.xy), CASE.spacing_m, rate, length_m, rng)
        tmrt = perturbed_tmrt(CASE, route, flips, offset)
        for dep in departures:
            walk = CASE.walk(route, dep, clo=CLO[(rid, dep)], tmrt_matrix=tmrt,
                             context=f"draw {draw}")
            rows.append({"draw": draw, "departure_hour": dep, "route_id": rid,
                         "load_offset_Wm2": offset, "flipped_fraction": float(flips.mean()),
                         "tcore_rise_c": walk["final_tcore_rise_c"],
                         "mean_tmrt_c": float(walk["tmrt_trace_c"].mean())})
    return rows


def summarise(draws: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    out = {}
    for dep, g in draws.groupby("departure_hour"):
        base = baseline[baseline["departure_hour"] == dep].set_index("route_id")["tcore_rise_c"]
        order0 = tuple(base.sort_values().index)
        best0 = order0[0]
        wide = g.pivot(index="draw", columns="route_id", values="tcore_rise_c")
        orders = wide.apply(lambda r: tuple(r.sort_values().index), axis=1)
        entry = {
            "n_draws": int(len(wide)),
            "baseline_order": [int(r) for r in order0],
            "baseline_rise_c": {int(k): float(v) for k, v in base.items()},
            "p_best_unchanged": float(np.mean([o[0] == best0 for o in orders])),
            "p_ranking_unchanged": float(np.mean([o == order0 for o in orders])),
            "rise_percentiles_c": {int(r): [float(np.percentile(wide[r], q)) for q in (2.5, 50, 97.5)]
                                   for r in wide.columns},
            "pairs": {},
        }
        for a, b in itertools.combinations(order0, 2):
            gap = wide[b] - wide[a]           # positive when the baseline order holds
            gap0 = float(base[b] - base[a])
            entry["pairs"][f"{a}-{b}"] = {
                "baseline_gap_c": gap0,
                "gap_percentiles_c": [float(np.percentile(gap, q)) for q in (2.5, 25, 50, 75, 97.5)],
                "p_order_kept": float(np.mean(gap > 0)),
            }
        out[str(dep)] = entry
    return out


def latex_table(summary: dict, flip_rate, length_m, offsets) -> str:
    deps = sorted(summary, key=float)
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Departure & Quantity & Baseline & 95\,\% band of draws & Kept \\", r"\midrule"]
    for d in deps:
        e = summary[d]
        hh = f"{float(d):02.0f}:00"
        L.append(f"{hh} & Best route & R{e['baseline_order'][0]} & --- & "
                 f"{100 * e['p_best_unchanged']:.0f}\\,\\% \\\\")
        L.append(f" & Ranking, lowest first & R{', R'.join(str(r) for r in e['baseline_order'])} & --- & "
                 f"{100 * e['p_ranking_unchanged']:.0f}\\,\\% \\\\")
        order = e["baseline_order"]
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            pp = e["pairs"][f"{a}-{b}"]
            q = pp["gap_percentiles_c"]
            L.append(f" & Gap R{b}$-$R{a} (\\si{{\\celsius}}) & {pp['baseline_gap_c']:.3f} & "
                     f"[{q[0]:+.3f}, {q[4]:+.3f}] & {100 * pp['p_order_kept']:.0f}\\,\\% \\\\")
        if d != deps[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--paper-dir", type=Path, default=Path("/home/harshin/files/fastUTEC paper"))
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--departures", type=float, nargs="+", default=[13.0, 16.0, 17.0])
    ap.add_argument("--flip-rate", type=float, required=True,
                    help="Lisbon per-sample sun/shade misclassification rate")
    ap.add_argument("--correlation-length-m", type=float, required=True,
                    help="along-route correlation length of the flips")
    ap.add_argument("--load-offset-range", type=float, nargs=2, required=True,
                    metavar=("LO", "HI"), help="uniform band of the flux offset, W/m2")
    ap.add_argument("--spacing-m", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite the LaTeX table from the saved summary.json")
    args = ap.parse_args()
    if args.table_only:
        out_dir = args.root / "run_output" / args.case / "postprocessing" / "ranking_uncertainty"
        meta = json.loads((out_dir / "summary.json").read_text())
        (args.paper_dir / "table_ranking_uncertainty.tex").write_text(
            latex_table(meta["summary"], meta["flip_rate"], meta["correlation_length_m"],
                        meta["load_offset_range_Wm2"]) + "\n")
        print(latex_table(meta["summary"], 0, 0, 0)); return

    global CASE, CLO
    CASE = CampusCase(args.root, args.case, spacing_m=args.spacing_m)
    worst = CASE.check_against_stage09()
    print(f"unperturbed walks reproduce stage 09 to {worst:.4f} C")
    baseline_rows = []
    for rid, route in CASE.routes.items():
        for dep in args.departures:
            clo, label = CASE.clothing(route, dep)
            CLO[(rid, dep)] = clo
            walk = CASE.walk(route, dep, clo=clo, context="baseline")
            baseline_rows.append({"departure_hour": dep, "route_id": rid,
                                  "tcore_rise_c": walk["final_tcore_rise_c"],
                                  "clothing": label})
    baseline = pd.DataFrame(baseline_rows)
    print(baseline.pivot(index="route_id", columns="departure_hour", values="tcore_rise_c"))

    lo, hi = args.load_offset_range
    tasks = [(d, args.seed * 100003 + d, tuple(args.departures), args.flip_rate,
              args.correlation_length_m, lo, hi) for d in range(args.draws)]
    rows = []
    with Pool(args.workers) as pool:
        for k, result in enumerate(pool.imap_unordered(run_draw, tasks, chunksize=2)):
            rows.extend(result)
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{args.draws} draws", flush=True)
    draws = pd.DataFrame(rows)
    summary = summarise(draws, baseline)
    out_dir = args.root / "run_output" / args.case / "postprocessing" / "ranking_uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    draws.to_csv(out_dir / "draws.csv", index=False)
    baseline.to_csv(out_dir / "baseline.csv", index=False)
    meta = {"draws": args.draws, "flip_rate": args.flip_rate,
            "correlation_length_m": args.correlation_length_m,
            "load_offset_range_Wm2": [lo, hi], "spacing_m": args.spacing_m,
            "departures": args.departures, "seed": args.seed,
            "mean_flipped_fraction": float(draws["flipped_fraction"].mean()),
            "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "table_ranking_uncertainty.tex").write_text(
        latex_table(summary, args.flip_rate, args.correlation_length_m, (lo, hi)) + "\n")
    (args.paper_dir / "ranking_uncertainty_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    for dep, e in summary.items():
        print(f"\n{float(dep):05.2f}  best kept {100 * e['p_best_unchanged']:.1f}%  "
              f"ranking kept {100 * e['p_ranking_unchanged']:.1f}%  order {e['baseline_order']}")
        for pair, p in e["pairs"].items():
            q = p["gap_percentiles_c"]
            print(f"   {pair}: gap {p['baseline_gap_c']:+.3f}  2.5-97.5% [{q[0]:+.3f}, {q[4]:+.3f}]  "
                  f"order kept {100 * p['p_order_kept']:.1f}%")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
