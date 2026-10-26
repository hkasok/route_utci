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
every pairwise gap and P(gap keeps its sign), and per pair the mean shift of
the draws from the baseline gap (systematic) next to the 2.5-97.5 band.

The defaults (--flip-model uniform --offset-model uniform) are the original
run. The paper reports the state-dependent run below (run ws_main: --flip-model
state --offset-model beam --correlation-length-m 5; 20-25 Oct 2026 review):

  --flip-model state   a point flips with P(measured shaded | modelled sunlit)
                       when modelled sunlit at that time step, and with
                       P(measured sunlit | modelled shaded) when shaded
                       (Lisbon daytime samples: 264/1517 and 283/750). The
                       state is time-dependent (tau per stage-05 step); one
                       correlated latent Gaussian field per route and draw is
                       thresholded at the quantile of the state's rate.
  --offset-model beam  the drawn offset O is the Lisbon sample-weighted mean:
                       sunlit points (after flips) get r times the shaded
                       value, f r O_sh + (1 - f) O_sh = O, with r = 3.1/1.6
                       (globe bias in sun / in shade, paper Sec. 4.4) and f the
                       Lisbon daytime measured sunlit share.

Usage:
    python3 paper_ranking_uncertainty.py --draws 500 --departures 13 16 17
        --flip-rate 0.241 --correlation-length-m 2 --load-offset-range -64 0
    python3 paper_ranking_uncertainty.py --flip-model state --offset-model beam
        --correlation-length-m 5 --load-offset-range -64 0 --out-name ws_main   # paper
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
from pythermalcomfort.models import utci as utci_fn
import route_measures
import jos3_protocol

# every route measure of Table 9 is propagated, not only the core rise
MEASURES = ("tcore_rise_c", "tcore_rad_c", "peak_tcore_rise_c", "heat_stored_kj",
            "heat_stored_rad_kj", "wettedness_discomfort_min", "walk_utci_c",
            "worst_3min_utci_c", "very_strong_stress_min", "strong_stress_dose_degmin")
START_UTCI_C = float(utci_fn(tdb=jos3_protocol.INDOOR_AIR_TEMPERATURE_C,
                             tr=jos3_protocol.INDOOR_AIR_TEMPERATURE_C, v=0.5,
                             rh=jos3_protocol.INDOOR_RH_PCT, round_output=False).utci)
NEUTRAL = {}    # (route, departure) -> radiation-neutral walk, independent of the draw


def walk_measures(route, dep, walk, neutral):
    """Every Table-9 measure of one walk (UTCI ones from the walk's own Tmrt)."""
    arrival = CASE.arrival_hours(route, dep)
    xyz = np.column_stack([route.xy, CASE.xyz[route.nearest, 2]])
    env = CASE.environment.sample(xyz, arrival % 24.0)
    u = np.asarray(utci_fn(tdb=np.asarray(env.air_temperature_c, float),
                           tr=walk["tmrt_trace_c"],
                           v=np.asarray(env.utci_wind_speed_10m_ms, float),
                           rh=np.asarray(env.relative_humidity_pct, float),
                           limit_inputs=False, round_output=False).utci, float)
    t = (arrival - arrival[0]) * 60.0
    ut, _ = route_measures.utci_route_measures(u, t, route.dist_m, START_UTCI_C)
    dt = np.diff(t)
    dose = float(np.sum(np.maximum(0.0, 0.5 * (u[1:] + u[:-1]) - 32.0) * dt))
    return {"tcore_rise_c": walk["final_tcore_rise_c"],
            "tcore_rad_c": walk["final_tcore_rise_c"] - neutral["final_tcore_rise_c"],
            "peak_tcore_rise_c": walk["peak_tcore_rise_c"],
            "heat_stored_kj": walk["heat_stored_kj"],
            "heat_stored_rad_kj": walk["heat_stored_kj"] - neutral["heat_stored_kj"],
            "wettedness_discomfort_min": walk["wettedness_discomfort_min"],
            "walk_utci_c": ut["walk_utci_c"], "worst_3min_utci_c": ut["worst_3min_utci_c"],
            "very_strong_stress_min": ut["very_strong_stress_min"],
            "strong_stress_dose_degmin": dose}

CASE = None     # set in each worker (fork) before the draws run
CLO = {}


# Lisbon daytime paired samples after measurement QC (paper_validation_figures.
# shadow_registration: sunlit = K-down > 0.5 GHI, n = 2267):
P_MEAS_SHADE_GIVEN_MODEL_SUN = 264 / 1517     # 0.1740
P_MEAS_SUN_GIVEN_MODEL_SHADE = 283 / 750      # 0.3773
LISBON_MEASURED_SUNLIT_SHARE = 0.6775474194971328
GLOBE_BIAS_SUN_OVER_SHADE = 3.1 / 1.6         # paper Sec. 4.4, agreeing stretches


def correlated_latent(n, spacing_m, length_m, rng):
    """Standard-normal AR(1) series with exponential correlation along the route."""
    rho = np.exp(-spacing_m / max(length_m, 1e-6))
    z = np.empty(n)
    z[0] = rng.standard_normal()
    eps = rng.standard_normal(n) * np.sqrt(1.0 - rho ** 2)
    for k in range(1, n):
        z[k] = rho * z[k - 1] + eps[k]
    return z


def correlated_flips(n, spacing_m, rate, length_m, rng):
    """Boolean flips with marginal rate ``rate`` and exponential correlation."""
    if rate <= 0:
        return np.zeros(n, dtype=bool)
    from scipy.stats import norm
    return correlated_latent(n, spacing_m, length_m, rng) < norm.ppf(rate)


def state_flips(z, sunlit, p_sun, p_shade):
    """(nt, n) flips: the latent field ``z`` (n,) thresholded at the quantile of
    the rate belonging to each point's modelled state at each time step."""
    from scipy.stats import norm
    with np.errstate(divide="ignore"):
        q_sun, q_shade = norm.ppf(p_sun), norm.ppf(p_shade)
    return np.where(sunlit, z[None, :] < q_sun, z[None, :] < q_shade)


def beam_offsets(load_offset_Wm2, sunlit_after, ratio, sunlit_share):
    """Per-point offset: sunlit points get ``ratio`` x the shaded value, and the
    mean weighted by the Lisbon sunlit share equals the drawn offset."""
    shade = load_offset_Wm2 / (sunlit_share * ratio + (1.0 - sunlit_share))
    return np.where(sunlit_after, ratio * shade, shade)


def perturbed_tmrt(case: CampusCase, route, flips, load_offset_Wm2, *,
                   offset_model="uniform", offset_ratio=GLOBE_BIAS_SUN_OVER_SHADE,
                   sunlit_share=LISBON_MEASURED_SUNLIT_SHARE, return_state=False):
    """Tmrt matrix (nt, n_pts) with the beam state inverted at flipped points
    (``flips`` (n_pts,) for every time step, or (nt, n_pts)) and a flux offset:
    the same everywhere ("uniform") or scaled by the post-flip beam state ("beam")."""
    flux, beam = case.route_flux(route)
    tau = case.tau[:, route.nearest].astype(float)
    full_beam = case.beam_if_sunlit()[:, None]
    sunlit = tau > 0.5
    flips = np.asarray(flips, dtype=bool)
    if flips.ndim == 1:
        flips = flips[None, :]
    # inverted state at flipped points: sunlit -> no beam, shaded -> full beam
    new_tau = np.where(flips, np.where(sunlit, 0.0, 1.0), tau)
    if offset_model == "uniform":
        offset = load_offset_Wm2
    elif offset_model == "beam":
        offset = beam_offsets(load_offset_Wm2, new_tau > 0.5, offset_ratio, sunlit_share)
    else:
        raise ValueError(f"unknown offset model {offset_model!r}")
    flux_new = flux + full_beam * (new_tau - tau) + offset
    tmrt = case.tmrt_from_flux(flux_new)
    if return_state:
        return tmrt, sunlit, new_tau > 0.5
    return tmrt


def draw_flips(route, P, rng):
    """Flips of one route in one draw under the configured flip model."""
    n = len(route.xy)
    if P["flip_model"] == "uniform":
        return correlated_flips(n, CASE.spacing_m, P["flip_rate"], P["length_m"], rng)
    if max(P["p_sun"], P["p_shade"]) <= 0:
        return np.zeros(n, dtype=bool)
    z = correlated_latent(n, CASE.spacing_m, P["length_m"], rng)
    sunlit = CASE.tau[:, route.nearest] > 0.5
    return state_flips(z, sunlit, P["p_sun"], P["p_shade"])


def run_draw(task):
    draw, seed, departures, P = task
    rng = np.random.default_rng(seed)
    offset = rng.uniform(P["offset_lo"], P["offset_hi"])
    day = CASE.elevation_deg > 0
    rows = []
    for rid, route in CASE.routes.items():
        flips = draw_flips(route, P, rng)
        tmrt, sunlit, sunlit_after = perturbed_tmrt(
            CASE, route, flips, offset, offset_model=P["offset_model"],
            offset_ratio=P["offset_ratio"], sunlit_share=P["sunlit_share"],
            return_state=True)
        # flipped share: of points (uniform model), of daytime point-times (state)
        flipped = float(flips.mean()) if flips.ndim == 1 else float(flips[day].mean())
        for dep in departures:
            walk = CASE.walk(route, dep, clo=CLO[(rid, dep)], tmrt_matrix=tmrt,
                             context=f"draw {draw}")
            rows.append({"draw": draw, "departure_hour": dep, "route_id": rid,
                         "load_offset_Wm2": offset, "flipped_fraction": flipped,
                         "sunlit_fraction_base": float(sunlit[day].mean()),
                         "sunlit_fraction_draw": float(sunlit_after[day].mean()),
                         "mean_tmrt_c": float(walk["tmrt_trace_c"].mean()),
                         **walk_measures(route, dep, walk, NEUTRAL[(rid, dep)])})
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
            band = [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]
            entry["pairs"][f"{a}-{b}"] = {
                "baseline_gap_c": gap0,
                "gap_percentiles_c": [float(np.percentile(gap, q)) for q in (2.5, 25, 50, 75, 97.5)],
                "p_order_kept": float(np.mean(gap > 0)),
                # systematic shift and spread, separately
                "mean_gap_c": float(gap.mean()),
                "mean_shift_c": float(gap.mean() - gap0),
                "band_95_c": band,
                "band_half_width_c": 0.5 * (band[1] - band[0]),
                "baseline_inside_band": bool(band[0] <= gap0 <= band[1]),
            }
        out[str(dep)] = entry
    return out


def measure_resolution(draws: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Per measure: the largest shift of any pairwise route difference from
    its baseline at the 2.5/97.5 percentiles of the draws, over all departures
    -- the resolution below which two routes are tied under that measure --
    and, per departure and pair, whether the 95 % band excludes zero."""
    out = {}
    for m in MEASURES:
        shift, sys_shift, half, pairs = 0.0, 0.0, 0.0, {}
        for dep, g in draws.groupby("departure_hour"):
            base = baseline[baseline["departure_hour"] == dep].set_index("route_id")[m]
            wide = g.pivot(index="draw", columns="route_id", values=m)
            for a, b in itertools.combinations(sorted(wide.columns), 2):
                gap = wide[b] - wide[a]
                gap0 = float(base[b] - base[a])
                lo, hi = float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))
                shift = max(shift, abs(gap0 - lo), abs(hi - gap0))
                sys_shift = max(sys_shift, abs(float(gap.mean()) - gap0))
                half = max(half, 0.5 * (hi - lo))
                pairs[f"{dep}:{a}-{b}"] = {"baseline_gap": gap0, "band": [lo, hi],
                                           "resolved": bool(lo > 0 or hi < 0),
                                           "mean_gap": float(gap.mean()),
                                           "mean_shift": float(gap.mean()) - gap0}
        out[m] = {"resolution": shift, "max_abs_mean_shift": sys_shift,
                  "max_band_half_width": half, "pairs": pairs}
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--out-name", default="",
                    help="write to postprocessing/ranking_uncertainty_<NAME>/ instead of the "
                         "published ranking_uncertainty/")
    ap.add_argument("--paper-dir", type=Path, default=None,
                    help="where the LaTeX table and summary copy go (default: the paper "
                         "folder for the published run, the output directory with --out-name)")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--departures", type=float, nargs="+", default=[13.0, 16.0, 17.0])
    ap.add_argument("--flip-model", choices=("uniform", "state"), default="uniform")
    ap.add_argument("--flip-rate", type=float, default=None,
                    help="uniform model: Lisbon per-sample sun/shade misclassification rate")
    ap.add_argument("--flip-rate-sun", type=float, default=P_MEAS_SHADE_GIVEN_MODEL_SUN,
                    help="state model: P(measured shaded | modelled sunlit)")
    ap.add_argument("--flip-rate-shade", type=float, default=P_MEAS_SUN_GIVEN_MODEL_SHADE,
                    help="state model: P(measured sunlit | modelled shaded)")
    ap.add_argument("--correlation-length-m", type=float, required=True,
                    help="along-route correlation length of the flips")
    ap.add_argument("--load-offset-range", type=float, nargs=2, required=True,
                    metavar=("LO", "HI"), help="uniform band of the flux offset, W/m2")
    ap.add_argument("--offset-model", choices=("uniform", "beam"), default="uniform")
    ap.add_argument("--offset-sun-shade-ratio", type=float, default=GLOBE_BIAS_SUN_OVER_SHADE)
    ap.add_argument("--offset-sunlit-share", type=float, default=LISBON_MEASURED_SUNLIT_SHARE,
                    help="beam model: sunlit share the drawn offset is the weighted mean over")
    ap.add_argument("--routes", type=int, nargs="+", default=None,
                    help="subset of routes (smoke tests only)")
    ap.add_argument("--skip-stage09-check", action="store_true",
                    help="skip the unperturbed-walk check against stage 09 (smoke tests only)")
    ap.add_argument("--spacing-m", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite the LaTeX table from the saved summary.json")
    args = ap.parse_args()
    if args.flip_model == "uniform" and args.flip_rate is None:
        ap.error("--flip-rate is required with --flip-model uniform")
    name = "ranking_uncertainty" + (f"_{args.out_name}" if args.out_name else "")
    out_dir = args.root / "run_output" / args.case / "postprocessing" / name
    if args.paper_dir is None:
        args.paper_dir = (out_dir if args.out_name
                          else Path("/home/harshin/files/fastUTEC paper"))
    if args.table_only:
        meta = json.loads((out_dir / "summary.json").read_text())
        (args.paper_dir / "table_ranking_uncertainty.tex").write_text(
            latex_table(meta["summary"], meta["flip_rate"], meta["correlation_length_m"],
                        meta["load_offset_range_Wm2"]) + "\n")
        print(latex_table(meta["summary"], 0, 0, 0)); return

    global CASE, CLO
    CASE = CampusCase(args.root, args.case, spacing_m=args.spacing_m)
    if args.routes:
        CASE.routes = {r: CASE.routes[r] for r in args.routes}
    if not args.skip_stage09_check:
        worst = CASE.check_against_stage09()
        print(f"unperturbed walks reproduce stage 09 to {worst:.4f} C")
    baseline_rows = []
    for rid, route in CASE.routes.items():
        for dep in args.departures:
            clo, label = CASE.clothing(route, dep)
            CLO[(rid, dep)] = clo
            walk = CASE.walk(route, dep, clo=clo, context="baseline")
            NEUTRAL[(rid, dep)] = CASE.walk(route, dep, clo=clo, radiation_neutral=True,
                                            context="baseline neutral")
            baseline_rows.append({"departure_hour": dep, "route_id": rid,
                                  "clothing": label,
                                  **walk_measures(route, dep, walk, NEUTRAL[(rid, dep)])})
    baseline = pd.DataFrame(baseline_rows)
    print(baseline.pivot(index="route_id", columns="departure_hour", values="tcore_rise_c"))

    lo, hi = args.load_offset_range
    P = {"flip_model": args.flip_model, "flip_rate": args.flip_rate,
         "p_sun": args.flip_rate_sun, "p_shade": args.flip_rate_shade,
         "length_m": args.correlation_length_m, "offset_lo": lo, "offset_hi": hi,
         "offset_model": args.offset_model, "offset_ratio": args.offset_sun_shade_ratio,
         "sunlit_share": args.offset_sunlit_share}
    tasks = [(d, args.seed * 100003 + d, tuple(args.departures), P) for d in range(args.draws)]
    rows = []
    with Pool(args.workers) as pool:
        for k, result in enumerate(pool.imap_unordered(run_draw, tasks, chunksize=2)):
            rows.extend(result)
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{args.draws} draws", flush=True)
    draws = pd.DataFrame(rows).sort_values(["draw", "departure_hour", "route_id"])
    summary = summarise(draws, baseline)
    measures = measure_resolution(draws, baseline)
    out_dir.mkdir(parents=True, exist_ok=True)
    draws.to_csv(out_dir / "draws.csv", index=False)
    baseline.to_csv(out_dir / "baseline.csv", index=False)
    meta = {"draws": args.draws, "flip_model": args.flip_model,
            "flip_rate": args.flip_rate if args.flip_model == "uniform" else None,
            "flip_rate_sun": args.flip_rate_sun if args.flip_model == "state" else None,
            "flip_rate_shade": args.flip_rate_shade if args.flip_model == "state" else None,
            "correlation_length_m": args.correlation_length_m,
            "load_offset_range_Wm2": [lo, hi], "offset_model": args.offset_model,
            "offset_sun_shade_ratio": (args.offset_sun_shade_ratio
                                       if args.offset_model == "beam" else None),
            "offset_sunlit_share": (args.offset_sunlit_share
                                    if args.offset_model == "beam" else None),
            "routes": sorted(int(r) for r in CASE.routes), "spacing_m": args.spacing_m,
            "departures": args.departures, "seed": args.seed,
            "mean_flipped_fraction": float(draws["flipped_fraction"].mean()),
            "mean_sunlit_fraction_base": float(draws["sunlit_fraction_base"].mean()),
            "mean_sunlit_fraction_draw": float(draws["sunlit_fraction_draw"].mean()),
            "summary": summary, "measures": measures}
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
            print(f"   {pair}: gap {p['baseline_gap_c']:+.3f}  mean shift {p['mean_shift_c']:+.4f}  "
                  f"2.5-97.5% [{q[0]:+.3f}, {q[4]:+.3f}]  order kept {100 * p['p_order_kept']:.1f}%")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
