#!/usr/bin/env python3
"""
paper_ta_heterogeneity.py -- would a spatially varying air temperature change
the campus route ranking?

Two synthetic tests, both re-running JOS-3 with everything but air temperature
(and the relative humidity that keeps vapour pressure fixed) held at the
stage-09 configuration (campus_walks.CampusCase):

  1. MEASURED ANOMALIES. Each Lisbon daytime walk carries, per sample, the
     difference between the cart's air temperature and the 10-minute series
     the pipeline was forced with -- the along-walk heterogeneity the forcing
     misses. Each such series is imposed on each campus route by elapsed
     time. Every assignment of the six series to the four routes (6^4) is then
     scored from the per-route results, because routes are independent walks:
     P(best unchanged), P(ranking unchanged), largest contrast error.
  2. CANOPY ANOMALIES. A uniform cooling of -0.5 ... -6 K on the points of each
     route whose sky is obstructed mostly by vegetation (vegetation share of
     the sky obstruction > 0.5). The routes differ in canopy cover, so this is
     a route-specific perturbation. Reported: the anomaly at which an adjacent
     pair first swaps and at which the best route first changes.

Usage:
    python3 paper_ta_heterogeneity.py [--departures 13 16 17]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from campus_walks import CampusCase
from paper_validation_figures import apply_measurement_qc, load_points

CANOPY_ANOMALIES_K = [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
CANOPY_THRESHOLD = 0.5


def lisbon_anomaly_series(root: Path) -> dict:
    """Per daytime walk: elapsed time (s) and Ta anomaly (measured minus the
    10-min forcing series), both from the comparison points."""
    points, _ = apply_measurement_qc(load_points(root))
    out = {}
    for (case, _), g in points[points["period"] == "day"].groupby(["case_id", "route_id"]):
        g = g.sort_values("seq")
        t = pd.to_datetime(g["timestamp_utc"], utc=True, format="ISO8601")
        elapsed = (t - t.iloc[0]).dt.total_seconds().to_numpy(float)
        anomaly = (g["measured_air_temperature_c"] - g["trec_route_air_temperature_c"]).to_numpy(float)
        ok = np.isfinite(anomaly)
        out[case] = {"elapsed_s": elapsed[ok], "anomaly_c": anomaly[ok],
                     "sd_c": float(np.std(anomaly[ok])),
                     "range_c": [float(np.min(anomaly[ok])), float(np.max(anomaly[ok]))],
                     "duration_s": float(elapsed[ok][-1])}
    return out


def impose_by_elapsed_time(series: dict, arrival_hours: np.ndarray) -> np.ndarray:
    """Map a Lisbon anomaly series onto a route by elapsed time, repeating the
    series if the campus walk is longer than the Lisbon one."""
    elapsed = (arrival_hours - arrival_hours[0]) * 3600.0
    period = series["duration_s"] + np.median(np.diff(series["elapsed_s"]))
    return np.interp(np.mod(elapsed, period), series["elapsed_s"], series["anomaly_c"])


def rankings_from_table(values: dict, order0: tuple) -> dict:
    """values[rid] -> list of candidate rises; score every combination."""
    rids = list(order0)
    combos = list(itertools.product(*[range(len(values[r])) for r in rids]))
    best_kept = rank_kept = 0
    contrast = 0.0
    base = {r: values[r][0] for r in rids}   # index 0 = unperturbed
    for combo in combos:
        rise = {r: values[r][k] for r, k in zip(rids, combo)}
        order = tuple(sorted(rise, key=rise.get))
        best_kept += order[0] == order0[0]
        rank_kept += order == order0
        for a, b in itertools.combinations(rids, 2):
            contrast = max(contrast, abs((rise[b] - rise[a]) - (base[b] - base[a])))
    return {"n_combinations": len(combos), "p_best_unchanged": best_kept / len(combos),
            "p_ranking_unchanged": rank_kept / len(combos),
            "max_contrast_error_c": contrast}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--paper-dir", type=Path, default=Path("/home/harshin/files/fastUTEC paper"))
    ap.add_argument("--departures", type=float, nargs="+", default=[13.0, 16.0, 17.0])
    args = ap.parse_args()

    case = CampusCase(args.root, args.case)
    print(f"stage-09 reproduction: worst |diff| {case.check_against_stage09():.4f} C")
    series = lisbon_anomaly_series(args.root)
    for k, s in series.items():
        print(f"  {k}: Ta anomaly sd {s['sd_c']:.2f} K, range {s['range_c'][0]:+.2f}..{s['range_c'][1]:+.2f} K, "
              f"{s['duration_s'] / 60:.0f} min")
    canopy = {rid: case.vegetation_sky_block[r.nearest] > CANOPY_THRESHOLD
              for rid, r in case.routes.items()}
    print("  canopy fraction per route:",
          {rid: round(float(m.mean()), 2) for rid, m in canopy.items()})

    result = {"canopy_threshold": CANOPY_THRESHOLD,
              "canopy_fraction": {int(r): float(m.mean()) for r, m in canopy.items()},
              "lisbon_anomaly_sd_c": {k: s["sd_c"] for k, s in series.items()},
              "departures": {}}
    rows = []
    for dep in args.departures:
        clo = {rid: case.clothing(r, dep)[0] for rid, r in case.routes.items()}
        base = {}
        for rid, r in case.routes.items():
            base[rid] = case.walk(r, dep, clo=clo[rid])["final_tcore_rise_c"]
            rows.append({"departure_hour": dep, "route_id": rid, "test": "baseline",
                         "value": 0.0, "tcore_rise_c": base[rid]})
        order0 = tuple(sorted(base, key=base.get))
        entry = {"baseline_rise_c": {int(k): v for k, v in base.items()},
                 "baseline_order": [int(r) for r in order0]}

        # 1. measured Lisbon anomalies imposed by elapsed time
        per_route = {rid: [base[rid]] for rid in case.routes}
        for name, s in series.items():
            for rid, r in case.routes.items():
                anomaly = impose_by_elapsed_time(s, case.arrival_hours(r, dep))
                rise = case.walk(r, dep, clo=clo[rid], ta_anomaly_c=anomaly)["final_tcore_rise_c"]
                per_route[rid].append(rise)
                rows.append({"departure_hour": dep, "route_id": rid, "test": f"lisbon_{name}",
                             "value": s["sd_c"], "tcore_rise_c": rise})
        # all assignments of the six series (index 1..6) to the four routes
        assigned = {rid: v[1:] for rid, v in per_route.items()}
        scored = rankings_from_table({rid: [base[rid]] + assigned[rid] for rid in case.routes}, order0)
        # restrict to combinations that use a Lisbon series on EVERY route
        combos_all = list(itertools.product(range(1, len(series) + 1), repeat=len(case.routes)))
        rids = list(order0)
        best_kept = rank_kept = 0; contrast = 0.0
        for combo in combos_all:
            rise = {r: per_route[r][k] for r, k in zip(rids, combo)}
            order = tuple(sorted(rise, key=rise.get))
            best_kept += order[0] == order0[0]; rank_kept += order == order0
            for a, b in itertools.combinations(rids, 2):
                contrast = max(contrast, abs((rise[b] - rise[a]) - (base[b] - base[a])))
        same_series = []
        for k in range(1, len(series) + 1):
            rise = {r: per_route[r][k] for r in rids}
            same_series.append(tuple(sorted(rise, key=rise.get)) == order0)
        entry["lisbon_anomalies"] = {
            "n_combinations": len(combos_all),
            "p_best_unchanged": best_kept / len(combos_all),
            "p_ranking_unchanged": rank_kept / len(combos_all),
            "max_contrast_error_c": contrast,
            "max_absolute_change_c": float(max(abs(per_route[r][k] - base[r])
                                               for r in rids for k in range(1, len(series) + 1))),
            "p_ranking_unchanged_same_series_all_routes": float(np.mean(same_series)),
        }

        # 2. canopy cooling
        canopy_rows = {}
        first_swap = first_best = None
        for dT in CANOPY_ANOMALIES_K:
            rise = {}
            for rid, r in case.routes.items():
                anomaly = np.where(canopy[rid], dT, 0.0)
                rise[rid] = case.walk(r, dep, clo=clo[rid], ta_anomaly_c=anomaly)["final_tcore_rise_c"]
                rows.append({"departure_hour": dep, "route_id": rid, "test": "canopy",
                             "value": dT, "tcore_rise_c": rise[rid]})
            order = tuple(sorted(rise, key=rise.get))
            canopy_rows[str(dT)] = {"rise_c": {int(k): v for k, v in rise.items()},
                                    "order": [int(r) for r in order]}
            if first_swap is None and order != order0:
                first_swap = dT
            if first_best is None and order[0] != order0[0]:
                first_best = dT
        entry["canopy"] = {"by_anomaly": canopy_rows,
                           "first_ranking_change_K": first_swap,
                           "first_best_change_K": first_best}
        result["departures"][str(dep)] = entry
        print(f"\n{dep:05.2f}: order {order0}; Lisbon anomalies: best kept "
              f"{100 * entry['lisbon_anomalies']['p_best_unchanged']:.1f}%, ranking kept "
              f"{100 * entry['lisbon_anomalies']['p_ranking_unchanged']:.1f}%, max contrast "
              f"{entry['lisbon_anomalies']['max_contrast_error_c']:.4f} C, max abs change "
              f"{entry['lisbon_anomalies']['max_absolute_change_c']:.4f} C")
        print(f"       canopy: first ranking change at {first_swap} K, first best change at {first_best} K")
        for dT, c in canopy_rows.items():
            print(f"         {float(dT):+.1f} K -> order {c['order']}  "
                  + "  ".join(f"R{r}: {v:.3f}" for r, v in c["rise_c"].items()))

    out_dir = args.root / "run_output" / args.case / "postprocessing" / "ta_heterogeneity"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.paper_dir / "ta_heterogeneity_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
