#!/usr/bin/env python3
"""Tabulate the state-dependent ranking Monte Carlo runs and the deterministic
route-robustness run (20 Oct 2026 review) into a scratch folder -- never the
paper folder.

Writes to OUT (default: the scratch folder given with --out):
  table_ranking_uncertainty_<run>.tex   main-text table format, one per run
  table_ranking_shift_<run>.tex         every pair: baseline, mean shift, band, inside?
  table_mc_sensitivity.tex              published rows + rows for the new runs
                                        (+ column: largest systematic shift)
  route_measures_<run>/                 paper_route_measures_table.py re-derived
                                        from that run's draws (tie rule, resolution)
  table_route_robustness.tex            radiant scaling, clothing, edge shade
  tabulation_summary.json               the numbers behind all of the above

Missing runs are skipped with a note, so this can run on partial results.

usage: python3 paper_adhoc/tabulate_ranking_mc_state.py --out DIR [--runs mc_state ...]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_ranking_uncertainty import latex_table  # noqa: E402

B = ROOT / "run_output" / "MMC" / "postprocessing"
PUBLISHED = (("500, 2 m (reported)", "ranking_uncertainty"),
             ("2000, 2 m", "ranking_uncertainty_n2000"),
             ("500, 10 m", "ranking_uncertainty_corr10m"),
             ("500, 40 m", "ranking_uncertainty_corr40m"))
NEW_LABELS = {"ws_main": "state/beam, 500, 5 m (walking speed)",
              "ws_n2000": "state/beam, 2000, 5 m (walking speed)",
              "ws_corr3m": "state/beam, 500, 3 m (walking speed)",
              "ws_corr10m": "state/beam, 500, 10 m (walking speed)",
              "ws_corr40m": "state/beam, 500, 40 m (walking speed)",
              "ws_flip_only": "state/uniform offset, 500, 5 m (walking speed)",
              "ws_uniform_beam": "uniform/beam offset, 500, 5 m (walking speed)",
              "mc_state_corr5m": "state/beam, 500, 5 m", "mc_state": "state/beam, 500", "mc_state_n2000": "state/beam, 2000",
              "mc_state_corr10m": "state/beam, 500, 10 m",
              "mc_state_corr40m": "state/beam, 500, 40 m",
              "mc_state_flip_only": "state/uniform offset, 500",
              "mc_uniform_beam": "uniform/beam offset, 500"}


def pairs_of(S):
    """(dep, pair, entry) with mean shift and band computed if the run predates them."""
    for dep, e in S["summary"].items():
        for key, g in e["pairs"].items():
            q = g["gap_percentiles_c"]
            band = g.get("band_95_c", [q[0], q[-1]])
            yield dep, key, {"baseline": g["baseline_gap_c"], "band": band,
                             "mean_shift": g.get("mean_shift_c"),
                             "inside": band[0] <= g["baseline_gap_c"] <= band[1],
                             "kept": g["p_order_kept"]}


def mean_shift_from_draws(d):
    """Mean shift for runs whose summary predates the field (published runs)."""
    import pandas as pd
    S = json.loads((d / "summary.json").read_text())
    draws = pd.read_csv(d / "draws.csv")
    base = pd.read_csv(d / "baseline.csv")
    out = {}
    for dep, g in draws.groupby("departure_hour"):
        w = g.pivot(index="draw", columns="route_id", values="tcore_rise_c")
        b = base[base.departure_hour == dep].set_index("route_id")["tcore_rise_c"]
        for key in S["summary"][str(dep)]["pairs"]:
            a, c = (int(x) for x in key.split("-"))
            out[(str(dep), key)] = float((w[c] - w[a]).mean() - (b[c] - b[a]))
    return out


def run_rows(label, d):
    S = json.loads((d / "summary.json").read_text())
    s = S["summary"]
    ms = None
    mx = sys_mx = 0.0
    outside = []
    for dep, key, p in pairs_of(S):
        if p["mean_shift"] is None:
            ms = ms or mean_shift_from_draws(d)
            p["mean_shift"] = ms[(dep, key)]
        g0, (lo, hi) = p["baseline"], p["band"]
        mx = max(mx, abs(g0 - lo), abs(hi - g0))
        sys_mx = max(sys_mx, abs(p["mean_shift"]))
        if not p["inside"]:
            outside.append(f"{dep}:{key}")
    pb = min(v["p_best_unchanged"] for v in s.values())
    pr = min(v["p_ranking_unchanged"] for v in s.values())
    b = {dep: s[dep]["pairs"]["1-3"]["gap_percentiles_c"] for dep in ("16.0", "17.0")
         if dep in s and "1-3" in s[dep]["pairs"]}
    cell = lambda dep: "[{:.3f}, {:.3f}]".format(b[dep][0], b[dep][-1]) if dep in b else "---"
    tex = (f"{label} & {100 * pb:.0f}\\,\\% / {100 * pr:.0f}\\,\\% & {cell('16.0')} & "
           f"{cell('17.0')} & {mx:.3f} & {sys_mx:.3f} \\\\")
    return tex, {"label": label, "dir": str(d), "p_best_min": pb, "p_ranking_min": pr,
                 "largest_shift_c": mx, "largest_mean_shift_c": sys_mx,
                 "pairs_outside_band": outside, "n_pairs": sum(len(v["pairs"]) for v in s.values()),
                 "flip_model": S.get("flip_model", "uniform"),
                 "offset_model": S.get("offset_model", "uniform"),
                 "correlation_length_m": S["correlation_length_m"], "draws": S["draws"],
                 "mean_flipped_fraction": S.get("mean_flipped_fraction"),
                 "mean_sunlit_fraction_base": S.get("mean_sunlit_fraction_base"),
                 "mean_sunlit_fraction_draw": S.get("mean_sunlit_fraction_draw")}


def shift_table(d):
    S = json.loads((d / "summary.json").read_text())
    ms = None
    L = [r"\begin{tabular}{llrrrc}", r"\toprule",
         r"Departure & Gap & Baseline & Mean shift & 95\,\% band & Baseline in band \\",
         r"\midrule"]
    for dep, key, p in pairs_of(S):
        if p["mean_shift"] is None:
            ms = ms or mean_shift_from_draws(d)
            p["mean_shift"] = ms[(dep, key)]
        a, b = key.split("-")
        L.append(f"{float(dep):02.0f}:00 & R{b}$-$R{a} & {p['baseline']:.3f} & "
                 f"{p['mean_shift']:+.4f} & [{p['band'][0]:+.3f}, {p['band'][1]:+.3f}] & "
                 f"{'yes' if p['inside'] else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def robustness_table(R):
    L = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Analysis & Setting & \multicolumn{3}{c}{R3$-$R1 (\si{\celsius})} \\",
         r"\cmidrule(lr){3-5}",
         r" & & 13:00 & 16:00 & 17:00 \\", r"\midrule"]
    deps = ("13", "16", "17")
    cell = lambda m, dep: f"{m[dep]['1-3']:+.3f}" if dep in m and "1-3" in m[dep] else "---"
    L.append("Baseline & --- & " + " & ".join(cell(R["baseline_gaps"], d) for d in deps) + r" \\")
    for s, m in sorted(R["radiant_scale_gaps"].items(), key=lambda kv: float(kv[0])):
        L.append(f"Radiant scaling & $s={float(s):.1f}$ & " + " & ".join(cell(m, d) for d in deps) + r" \\")
    zs = {f"{x['departure_hour']:g}": x["s_zero"] for x in R["radiant_scale_sign_change"]}
    if zs:
        L.append(r" & $s$ at zero & " + " & ".join(
            "none in [0.4, 1.6]" if zs.get(d) is None else f"{zs[d]:.2f}" for d in deps) + r" \\")
    for c, m in sorted(R["clothing_gaps"].items(), key=lambda kv: float(kv[0])):
        L.append(f"Clothing & {float(c):+.1f} clo & " + " & ".join(cell(m, d) for d in deps) + r" \\")
    for q, m in sorted(R["edge_shade_gaps"].items(), key=lambda kv: float(kv[0])):
        L.append(f"R3 edge shade & $q={float(q):.2f}$ & " + " & ".join(cell(m, d) for d in deps) + r" \\")
    qz = {f"{x['departure_hour']:g}": x["q_zero"] for x in R["edge_shade_zero"]}
    if qz:
        L.append(r" & $q$ at zero & " + " & ".join(
            "not reached" if qz.get(d) is None else f"{qz[d]:.2f}" for d in deps) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=list(NEW_LABELS))
    ap.add_argument("--robustness", type=Path, default=B / "route_robustness")
    ap.add_argument("--no-route-measures", action="store_true")
    a = ap.parse_args()
    if "fastUTEC paper" in str(a.out.resolve()):
        raise SystemExit("refusing to write into the paper folder")
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [], "missing": []}
    rows = []
    for label, name in PUBLISHED:
        if (B / name / "summary.json").is_file():
            t, info = run_rows(label, B / name); rows.append(t); summary["runs"].append(info)
    rows.append(r"\midrule")
    for name in a.runs:
        d = B / f"ranking_uncertainty_{name}"
        if not (d / "summary.json").is_file():
            summary["missing"].append(name); print(f"missing: {d}"); continue
        t, info = run_rows(NEW_LABELS.get(name, name), d)
        rows.append(t); summary["runs"].append(info)
        S = json.loads((d / "summary.json").read_text())
        (a.out / f"table_ranking_uncertainty_{name}.tex").write_text(
            latex_table(S["summary"], None, None, None) + "\n")
        (a.out / f"table_ranking_shift_{name}.tex").write_text(shift_table(d))
        if not a.no_route_measures:
            rm = a.out / f"route_measures_{name}"
            r = subprocess.run([sys.executable, str(ROOT / "paper_route_measures_table.py"),
                                "--mc", str(d), "--paper-dir", str(rm)], cwd=ROOT,
                               capture_output=True, text=True)
            info["route_measures"] = str(rm) if r.returncode == 0 else f"FAILED: {r.stderr[-400:]}"
            print(f"route measures {name}: {info['route_measures']}")
    (a.out / "table_ranking_shift_published.tex").write_text(shift_table(B / "ranking_uncertainty"))
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Draws, & Best route / & \multicolumn{2}{c}{R3$-$R1, 95\,\% band (\si{\celsius})} & Largest & Largest mean \\",
         r"\cmidrule(lr){3-4}",
         r"correlation & ranking kept & 16:00 & 17:00 & shift (\si{\celsius}) & shift (\si{\celsius}) \\",
         r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (a.out / "table_mc_sensitivity.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    rj = a.robustness / "summary.json"
    if rj.is_file():
        R = json.loads(rj.read_text())
        (a.out / "table_route_robustness.tex").write_text(robustness_table(R))
        summary["robustness"] = {k: R[k] for k in ("baseline_gaps", "radiant_scale_sign_change",
                                                    "clothing_cross", "edge_shade_zero",
                                                    "edge_points")}
        print(robustness_table(R))
    else:
        summary["missing"].append(str(rj)); print(f"missing: {rj}")
    for info in summary["runs"]:
        print(f"{info['label']:32s} outside band {len(info['pairs_outside_band']):2d}/{info['n_pairs']}  "
              f"largest shift {info['largest_shift_c']:.3f}  largest mean shift {info['largest_mean_shift_c']:.4f}")
    (a.out / "tabulation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
