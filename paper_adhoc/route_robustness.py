#!/usr/bin/env python3
"""Deterministic robustness of the campus route contrasts (no Monte Carlo).

Run on the four campus routes with campus_walks.CampusCase (the stage-09 subject,
pace, clothing, start state and protocol), at the 13:00, 16:00 and 17:00
departures, reporting the core-temperature rise of every route and every pairwise
gap (Rb - Ra, b > a):

  a. RADIANT SENSITIVITY. Tmrt' = Ta + s (Tmrt - Ta) at every route point and
     stage-05 time step, s in --scales (default 0.6..1.4); the s at which
     R3 - R1 changes sign is found by bisection when it lies in --bisect-range
     (default 0.4..1.6).
     CLOTHING. The stage-09 ensemble of every route rescaled to its whole-body
     (BSA-weighted) clo +- 0.1, all routes together; the per-route rises are
     kept so cross combinations can be formed.
  b. MISSING OUT-OF-TILE SHADE. Route 3 points within --edge-m (25 m) of the
     LiDAR tile (bounding box of input/<case>/geometry/ground_and_water_final.stl,
     as the paper's "a third of Route 3" was computed) lose the direct beam at
     every sunlit time step for a seeded random fraction q of those points
     (nested sets: the points shaded at q are also shaded at every larger q).
     Gaps for --q-values; the q at which R3 - R1 reaches zero by bisection on
     the number of shaded points.

Outputs (--out-dir, default run_output/<case>/postprocessing/route_robustness):
rises.csv (one row per evaluated walk), summary.json.

usage: python3 paper_adhoc/route_robustness.py --workers 6
       (smoke) python3 paper_adhoc/route_robustness.py --routes 1 3 --departures 16
               --scales 1.0 --clo-deltas --q-values 1 --no-bisect --workers 1
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campus_walks import CampusCase  # noqa: E402
import clothing_profiles  # noqa: E402

CASE: CampusCase | None = None
CLO = {}            # (route, departure) -> stage-09 segment clo
TA = {}             # route -> Ta matrix (nt, n_pts) at the stage-05 time steps
EDGE = {}           # route -> (indices of edge points in shading order)


def whole_body_clo(seg):
    return float(np.sum(np.asarray(seg) * clothing_profiles.SEGMENT_BSA_FRACTION))


def shifted_clo(seg, delta):
    tot = whole_body_clo(seg)
    return np.asarray(seg, float) * (tot + delta) / tot


def tile_edge_distance(case, xy):
    """Distance to the bounding box of the ground mesh (the LiDAR tile)."""
    raw = (case.root / "input" / case.case / "geometry" / "ground_and_water_final.stl").read_bytes()[84:]
    v = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                           ("a", "<u2")]))["v"].reshape(-1, 3).astype(float)
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    return np.minimum(xy - lo, hi - xy).min(1), lo, hi


def ta_matrix(route):
    xyz = np.column_stack([route.xy, CASE.xyz[route.nearest, 2]])
    nt, n = len(CASE.time_hours), len(xyz)
    env = CASE.environment.sample(np.tile(xyz, (nt, 1)), np.repeat(CASE.time_hours, n))
    return np.asarray(env.air_temperature_c, float).reshape(nt, n)


def scaled_tmrt(route, s):
    tmrt = CASE.route_tmrt_matrix(route).astype(float)
    ta = TA[route.route_id]
    return ta + s * (tmrt - ta)


def edge_shaded_tmrt(route, k):
    """Beam removed at the sunlit time steps of the first k edge points."""
    flux, _ = CASE.route_flux(route)
    tau = CASE.tau[:, route.nearest].astype(float)
    mask = np.zeros(len(route.xy), dtype=bool)
    mask[EDGE[route.route_id][:k]] = True
    shade = mask[None, :] & (tau > 0.5)
    flux_new = flux - CASE.beam_if_sunlit()[:, None] * np.where(shade, tau, 0.0)
    return CASE.tmrt_from_flux(flux_new), float(shade[CASE.elevation_deg > 0].sum())


def evaluate(task):
    """task = (analysis, route_id, departure, value) -> row with the rise."""
    kind, rid, dep, value = task
    route = CASE.routes[rid]
    clo = CLO[(rid, dep)]
    extra = {}
    if kind == "baseline":
        tmrt = None
    elif kind == "scale":
        tmrt = None if value == 1.0 else scaled_tmrt(route, value)
    elif kind == "clo":
        tmrt, clo = None, shifted_clo(clo, value)
        extra["clo_total"] = whole_body_clo(clo)
    elif kind == "edge":
        k = int(value)
        tmrt, n_shaded = (None, 0.0) if k == 0 else edge_shaded_tmrt(route, k)
        extra.update(n_points_shaded=k, q=k / max(len(EDGE[rid]), 1),
                     shaded_point_times_daytime=n_shaded)
    else:
        raise ValueError(kind)
    walk = CASE.walk(route, dep, clo=clo, tmrt_matrix=tmrt, context=f"{kind} {value}")
    return {"analysis": kind, "route_id": rid, "departure_hour": dep, "value": float(value),
            "tcore_rise_c": float(walk["final_tcore_rise_c"]),
            "peak_tcore_rise_c": float(walk["peak_tcore_rise_c"]),
            "mean_tmrt_c": float(walk["tmrt_trace_c"].mean()), **extra}


def bisect_scale(job):
    """s in [lo, hi] at which rise(b) - rise(a) changes sign, or None."""
    dep, a, b, lo, hi, tol = job
    f = lambda s: (evaluate(("scale", b, dep, s))["tcore_rise_c"]
                   - evaluate(("scale", a, dep, s))["tcore_rise_c"])
    flo, fhi = f(lo), f(hi)
    out = {"departure_hour": dep, "pair": f"{a}-{b}", "range": [lo, hi],
           "gap_at_lo": flo, "gap_at_hi": fhi, "s_zero": None, "evaluations": []}
    if np.sign(flo) == np.sign(fhi):
        return out
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        out["evaluations"].append([mid, fm])
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    out["s_zero"] = lo - flo * (hi - lo) / (fhi - flo)       # linear within the last bracket
    return out


def bisect_edge(job):
    """Smallest number of shaded edge points at which rise(q_route) - rise(ref)
    reaches zero, or None if it does not at q = 1."""
    dep, ref, qr = job
    r_ref = evaluate(("baseline", ref, dep, 0))["tcore_rise_c"]   # Route ref unshaded
    n = len(EDGE[qr])
    cache = {}

    def g(k):
        if k not in cache:
            cache[k] = evaluate(("edge", qr, dep, k))["tcore_rise_c"] - r_ref
        return cache[k]
    g0, gn = g(0), g(n)
    out = {"departure_hour": dep, "pair": f"{ref}-{qr}", "n_edge_points": n,
           "gap_q0": g0, "gap_q1": gn, "q_zero": None}
    if np.sign(g0) == np.sign(gn):
        return out
    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np.sign(g(mid)) == np.sign(g0):
            lo = mid
        else:
            hi = mid
    out.update(q_zero=hi / n, k_zero=hi, gap_at_k_zero=g(hi))
    return out


def pair_gaps(df, value_col="value"):
    """{value: {dep: {"a-b": rise_b - rise_a}}}"""
    out = {}
    for (val, dep), g in df.groupby([value_col, "departure_hour"]):
        r = g.set_index("route_id")["tcore_rise_c"]
        out.setdefault(f"{val:g}", {})[f"{dep:g}"] = {
            f"{a}-{b}": float(r[b] - r[a]) for a, b in itertools.combinations(sorted(r.index), 2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--departures", type=float, nargs="+", default=[13.0, 16.0, 17.0])
    ap.add_argument("--routes", type=int, nargs="+", default=None)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4])
    ap.add_argument("--bisect-range", type=float, nargs=2, default=[0.4, 1.6])
    ap.add_argument("--bisect-tol", type=float, default=0.01)
    ap.add_argument("--clo-deltas", type=float, nargs="*", default=[-0.1, 0.1])
    ap.add_argument("--edge-m", type=float, default=25.0)
    ap.add_argument("--q-route", type=int, default=3)
    ap.add_argument("--ref-route", type=int, default=1)
    ap.add_argument("--q-values", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--no-bisect", action="store_true")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or ROOT / "run_output" / args.case / "postprocessing" / "route_robustness"

    global CASE
    CASE = CampusCase(ROOT, args.case)
    if args.routes:
        CASE.routes = {r: CASE.routes[r] for r in args.routes}
    for rid, route in CASE.routes.items():
        TA[rid] = ta_matrix(route)
        for dep in args.departures:
            CLO[(rid, dep)] = CASE.clothing(route, dep)[0]
    edge_info = {}
    for rid, route in CASE.routes.items():
        dist, lo, hi = tile_edge_distance(CASE, route.xy)
        idx = np.flatnonzero(dist < args.edge_m)
        EDGE[rid] = np.random.default_rng(args.seed + rid).permutation(idx)
        edge_info[rid] = {"n_points": int(len(route.xy)), "n_edge": int(len(idx)),
                          "fraction_within_edge_m": float(len(idx) / len(route.xy))}
    print("tile bounds", lo.round(1).tolist(), hi.round(1).tolist(), "edge points:", edge_info)
    q_route = args.q_route if args.q_route in CASE.routes else None

    tasks = []
    for dep in args.departures:
        for rid in CASE.routes:
            tasks += [("scale", rid, dep, s) for s in args.scales]
            tasks += [("clo", rid, dep, d) for d in args.clo_deltas]
            if 1.0 not in args.scales:
                tasks.append(("baseline", rid, dep, 0))
        if q_route is not None:
            n = len(EDGE[q_route])
            tasks += [("edge", q_route, dep, int(round(q * n))) for q in args.q_values if q > 0]
    with Pool(args.workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(evaluate, tasks)):
            rows.append(r)
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(tasks)} walks", flush=True)
        jobs_s = [(dep, args.ref_route, args.q_route, *args.bisect_range, args.bisect_tol)
                  for dep in args.departures] if not args.no_bisect and q_route else []
        jobs_q = [(dep, args.ref_route, q_route) for dep in args.departures] \
            if not args.no_bisect and q_route else []
        sign_s = pool.map(bisect_scale, jobs_s) if jobs_s else []
        sign_q = pool.map(bisect_edge, jobs_q) if jobs_q else []
    df = pd.DataFrame(rows)
    base = df[(df.analysis == "scale") & (df.value == 1.0)]
    if base.empty:
        base = df[df.analysis == "baseline"]
    base = base.assign(value=1.0)

    # gaps, with the baseline route set filling in the routes an analysis leaves alone
    def with_base(sub, value):
        b = base.drop(columns=["value"]).assign(value=value)
        keep = b[~b.set_index(["route_id", "departure_hour"]).index.isin(
            sub.set_index(["route_id", "departure_hour"]).index)]
        return pd.concat([sub, keep])

    scale_gaps = pair_gaps(df[df.analysis == "scale"])
    clo_gaps = pair_gaps(df[df.analysis == "clo"])
    clo_cross = {}
    for dep in args.departures:
        c = df[(df.analysis == "clo") & (df.departure_hour == dep)]
        r = {(row.route_id, row.value): row.tcore_rise_c for row in c.itertuples()}
        a, b = args.ref_route, args.q_route
        if (a, -0.1) in r and (b, 0.1) in r:
            clo_cross[f"{dep:g}"] = {f"R{b}(-0.1)-R{a}(+0.1)": r[(b, -0.1)] - r[(a, 0.1)],
                                     f"R{b}(+0.1)-R{a}(-0.1)": r[(b, 0.1)] - r[(a, -0.1)]}
    edge_gaps = {}
    if q_route is not None:
        e = df[df.analysis == "edge"]
        for q in args.q_values:
            k = int(round(q * len(EDGE[q_route])))
            sub = e[e.value == k] if k > 0 else base[base.route_id == q_route]
            edge_gaps[f"{q:g}"] = pair_gaps(with_base(sub.assign(value=q), q))[f"{q:g}"]
    summary = {"case": args.case, "departures": args.departures,
               "routes": sorted(CASE.routes), "edge_m": args.edge_m,
               "edge_points": {str(k): v for k, v in edge_info.items()},
               "edge_shading_seed": args.seed,
               "clo_total_stage09": {f"{rid}@{dep:g}": whole_body_clo(c)
                                     for (rid, dep), c in CLO.items()},
               "baseline_gaps": pair_gaps(base)["1"],
               "radiant_scale_gaps": scale_gaps, "radiant_scale_sign_change": sign_s,
               "clothing_gaps": clo_gaps, "clothing_cross": clo_cross,
               "edge_shade_gaps": edge_gaps, "edge_shade_zero": sign_q}
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rises.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("baseline_gaps", "radiant_scale_sign_change",
                                              "edge_shade_zero")}, indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
