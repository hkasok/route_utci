#!/usr/bin/env python3
"""Correlation length for the sun/shade flips of paper_ranking_uncertainty.py:
the length at which the simulated flipped stretches have the MEAN length of the
misclassified stretches measured in the Lisbon validation.

Reads validation_summary.json (shadow.classification_agreement and
shadow.tolerance.misclassified_run_length_m.mean); flips are generated with
paper_ranking_uncertainty.correlated_flips on the 2 m route spacing.

The measured stretches are runs of consecutive misclassified SAMPLES, each run
counted as (last - first) + the walk's median sample spacing (about 7 m); the
simulated 2 m flip series is therefore read at that spacing and its runs are
measured the same way.

--flip-model state (20 Oct 2026): the flip rate depends on the modelled state
(P(measured shaded | modelled sunlit), P(measured sunlit | modelled shaded)),
so the simulated stretches depend on the sequence of modelled states along the
walk. The latent field is then simulated at the Lisbon samples' own along-track
positions and modelled states (reconstructed exactly as
paper_validation_figures.shadow_registration_tolerance does; the measured runs
are recomputed from them and must reproduce validation_summary.json), and its
runs are measured with the same rule. The uniform model is calibrated by the
same sample-position method alongside, so the two can be compared like for like.

usage: python3 paper_adhoc/calibrate_flip_correlation.py [validation_summary.json
                                                          [comparison_points_glob]]
       python3 paper_adhoc/calibrate_flip_correlation.py --flip-model state
"""
import argparse
import glob
import json
import sys

import pandas as pd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_ranking_uncertainty import (correlated_flips, P_MEAS_SHADE_GIVEN_MODEL_SUN,  # noqa: E402
                                       P_MEAS_SUN_GIVEN_MODEL_SHADE)

GRID = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 50], float)


def uniform_stride(summary, points):
    """The published calibration (uniform rate, 2 m series read at the median
    sample spacing)."""
    s = summary
    rate = 1.0 - s["classification_agreement"]
    target = s["tolerance"]["misclassified_run_length_m"]["mean"]
    spacing = 2.0
    steps = []
    for f in sorted(glob.glob(points)):
        d = pd.read_csv(f)
        d = d[d.period == "day"]
        for _, g in d.groupby("route_id"):
            steps.append(np.median(np.diff(g.sort_values("seq").distance_along_route_m.values)))
    sample_m = float(np.median(steps))
    stride = max(1, int(round(sample_m / spacing)))

    def mean_run_m(length, n=400_000, seed=11):
        f = correlated_flips(n, spacing, rate, length, np.random.default_rng(seed))[::stride]
        edges = np.diff(np.r_[0, f.astype(int), 0])
        runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        return float(runs.mean() * stride * spacing)

    means = np.array([mean_run_m(L) for L in GRID])
    L = float(np.interp(target, means, GRID))
    print(f"flip rate {rate:.3f}; measured mean misclassified stretch {target:.1f} m; "
          f"sample spacing {sample_m:.2f} m (stride {stride} x {spacing} m)")
    for g, m in zip(GRID, means):
        print(f"  L = {g:4.0f} m -> mean flipped stretch {m:5.1f} m")
    print(f"calibrated correlation length {L:.1f} m (rounded: {round(L)} m)")


def lisbon_sequences(root):
    """Per Lisbon walk: along-track position, modelled and measured sunlit state
    of every daytime sample, as shadow_registration_tolerance builds them."""
    from scipy.spatial import cKDTree
    from paper_validation_figures import load_points, apply_measurement_qc
    points, _ = apply_measurement_qc(load_points(root))
    d_all = points[(points["period"] == "day")
                   & (points["trec_atmospheric_ghi_wm2"] > 50)].dropna(
        subset=["measured_swin_wm2", "sensor_shortwave_down_Wm2",
                "trec_atmospheric_ghi_wm2", "timestamp_utc"])
    out = []
    for case, g in d_all.groupby("case_id"):
        g = g.sort_values("seq")
        cj = json.loads((root / "input" / case / "case.json").read_text())
        ox, oy = cj["coordinates"]["local_origin_x"], cj["coordinates"]["local_origin_y"]
        mrt = root / "run_output" / case / "mrt_facet_out"
        xyz = np.load(mrt / "path_xyz.npy")
        seg = np.load(mrt / "path_segment_id.npy")
        kdown = np.load(mrt / "radiant_flux_contributions.npz")["sensor_shortwave_down_Wm2"]
        times = pd.read_csv(mrt / "times.csv")
        tt = pd.to_datetime(times["time"])
        hours = (tt.dt.hour + tt.dt.minute / 60 + tt.dt.second / 3600).to_numpy(float)
        ghi_t = times["GHI_Wm2"].to_numpy(float)
        _, nearest = cKDTree(xyz[:, :2]).query(
            np.column_stack([g["x_projected_m"] - ox, g["y_projected_m"] - oy]))
        on = np.flatnonzero(seg == int(np.bincount(seg[nearest]).argmax()))
        s_route = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(xyz[on, :2], axis=0), axis=1))))
        pos = np.clip(np.searchsorted(on, nearest), 0, len(on) - 1)
        hour = g["arrival_hour_local"].to_numpy(float)
        i1 = np.clip(np.searchsorted(hours, hour), 1, len(hours) - 1)
        w = (hour - hours[i1 - 1]) / (hours[i1] - hours[i1 - 1])
        pt = on[pos]
        model = ((1 - w) * kdown[i1 - 1, pt] + w * kdown[i1, pt]
                 > 0.5 * ((1 - w) * ghi_t[i1 - 1] + w * ghi_t[i1]))
        meas = (g["measured_swin_wm2"].to_numpy(float)
                > 0.5 * g["trec_atmospheric_ghi_wm2"].to_numpy(float))
        out.append({"case": case, "s": s_route[pos], "model_sun": model, "meas_sun": meas})
    return out


def run_lengths(s, wrong):
    """Validation rule: (last - first) + median spacing, per run of True.
    ``wrong`` may be (n,) or (R, n)."""
    w = np.atleast_2d(wrong).astype(np.int8)
    edges = np.diff(np.pad(w, ((0, 0), (1, 1))), axis=1)
    r0, i0 = np.nonzero(edges == 1)
    r1, i1 = np.nonzero(edges == -1)
    return s[i1 - 1] - s[i0] + np.median(np.diff(s))


def simulated_mean(seqs, length, p_sun, p_shade, reps, seed):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    total, count = 0.0, 0
    for q in seqs:
        s = q["s"]
        rho = np.exp(-np.abs(np.diff(s)) / max(length, 1e-6))
        z = np.empty((reps, len(s)))
        z[:, 0] = rng.standard_normal(reps)
        for k in range(1, len(s)):
            z[:, k] = rho[k - 1] * z[:, k - 1] + np.sqrt(1 - rho[k - 1] ** 2) * rng.standard_normal(reps)
        thr = np.where(q["model_sun"], norm.ppf(p_sun), norm.ppf(p_shade))
        r = run_lengths(s, z < thr[None, :])
        total += r.sum(); count += r.size
    return total / count


def by_samples(summary, p_sun, p_shade, reps, out_json):
    seqs = lisbon_sequences(ROOT)
    measured = np.concatenate([run_lengths(q["s"], q["model_sun"] != q["meas_sun"]) for q in seqs])
    target = summary["tolerance"]["misclassified_run_length_m"]["mean"]
    print(f"measured runs recomputed: {measured.size}, mean {measured.mean():.3f} m "
          f"(validation_summary.json: {summary['tolerance']['misclassified_run_length_m']['n_runs']}, "
          f"{target:.3f} m)")
    if abs(measured.mean() - target) > 1e-6:
        raise SystemExit("Lisbon sequences do not reproduce the validated run lengths")
    bad = measured[measured <= 0]
    print(f"  of which non-physical (<= 0 m, projection jumps backward): {bad.size} "
          f"({', '.join(f'{b:.0f}' for b in bad)} m); mean without them "
          f"{measured[measured > 0].mean():.2f} m")
    rate = 1.0 - summary["classification_agreement"]
    grid = np.r_[1e-3, GRID]
    res = {"target_mean_run_m": target, "grid_m": grid.tolist()}
    for name, ps, ph in (("uniform", rate, rate), ("state", p_sun, p_shade)):
        means = np.array([simulated_mean(seqs, L, ps, ph, reps, 11) for L in grid])
        L = float(np.interp(target, means, grid)) if target >= means[0] else float(grid[1])
        floor = target < means[0]
        res[name] = {"p_sun": ps, "p_shade": ph, "mean_run_m": means.tolist(),
                     "calibrated_length_m": L, "below_uncorrelated": bool(floor)}
        print(f"\n{name} model (P flip | sunlit {ps:.3f}, | shaded {ph:.3f}), "
              f"latent field at the {sum(len(q['s']) for q in seqs)} sample positions, {reps} replicates")
        for g, m in zip(grid, means):
            print(f"  L = {g:6.3f} m -> mean flipped stretch {m:5.1f} m")
        print(f"  calibrated correlation length {'<= 2 m (target below the uncorrelated mean; use the 2 m floor)' if floor else f'{L:.1f} m'}")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", nargs="?",
                    default="/home/harshin/files/fastUTEC paper/validation_summary.json")
    ap.add_argument("points", nargs="?",
                    default="run_output/lisbon*/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
    ap.add_argument("--flip-model", choices=("uniform", "state"), default="uniform",
                    help="uniform: the published stride calibration; state: sample-position "
                         "calibration of both models")
    ap.add_argument("--flip-rate-sun", type=float, default=P_MEAS_SHADE_GIVEN_MODEL_SUN)
    ap.add_argument("--flip-rate-shade", type=float, default=P_MEAS_SUN_GIVEN_MODEL_SHADE)
    ap.add_argument("--replicates", type=int, default=400)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text())["shadow"]
    if a.flip_model == "uniform":
        uniform_stride(summary, a.points)
    else:
        by_samples(summary, a.flip_rate_sun, a.flip_rate_shade, a.replicates, a.out_json)


if __name__ == "__main__":
    main()
