#!/usr/bin/env python3
"""
paper_convergence.py -- wall-clock breakdown per scene, convergence of the
route-first discretisation, and the ablation table.

STANDALONE. Reads stage_timings.log written by start.sh for each case, the
variant trees produced by paper_variant_runs.sh (variants/<name>/...), and the
comparison points of each variant; computes no new physics.

  * TIMING: seconds per stage per Lisbon scene (one sequential run, same
    machine), as a supplementary table.
  * CONVERGENCE (lisbon1): route Tmrt of each discretisation variant against
    the reference run -- maximum and RMS difference over daytime steps and all
    route points -- and the daytime globe / radiometer statistics of each.
    Variants: facet-selection rays 48x36 (ref 24x18), point stride 4 (ref 8),
    distance cap 150 and 600 m (ref 300), and the full-scene energy balance.
  * ABLATION (six scenes): daytime globe bias/RMSE, L-up RMSE and K-down RMSE
    of each model variant from its pooled validation summary.

Usage:
    python3 paper_convergence.py [--paper-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_validation_figures import CASES, apply_measurement_qc, stats

STAGES = [
    ("Facet-thermal MRT -- preparation", "Prep."),
    ("Selecting route-visible thermal facets", "Selection"),
    ("Facet 1D surface-energy balance", "Energy bal."),
    ("Facet-thermal MRT ray tracing", "Assembly"),
    ("Route-point ground-material diagnostics", "Diagn."),
    ("UTCI exposure", "UTCI/globe"),
    ("JOS-3 core", "JOS-3"),
]
CONVERGENCE = [
    ("conv_rays48x36", r"Selection rays $48 \times 36$ (ref.\ $24 \times 18$)"),
    ("conv_stride4", "Traced-point stride 4 (ref.\\ 8)"),
    ("conv_range150", "Distance cap \\SI{150}{\\metre} (ref.\\ 300)"),
    ("conv_range600", "Distance cap \\SI{600}{\\metre} (ref.\\ 300)"),
    ("conv_fullscene", "Full-scene energy balance"),
]
ABLATION = [
    ("abl_surfaces_at_ta", "Surfaces at air temperature"),
    ("abl_steady_state", "Steady-state surfaces (no storage)"),
    ("abl_hemisphere_crowns", "Hemisphere crowns"),
    ("abl_global_reflected", "Global ground-reflected shortwave"),
    ("abl_generic_paving", "Generic paving instead of limestone"),
]


def timings(root: Path) -> pd.DataFrame:
    rows = []
    for case in CASES:
        path = root / "run_output" / case / "stage_timings.log"
        if not path.is_file():
            continue
        entries = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines()]
        t = np.array([float(e[0]) for e in entries])
        names = [e[1] for e in entries]
        row = {"case": case}
        for key, label in STAGES:
            for i, name in enumerate(names[:-1]):
                if key in name:
                    row[label] = t[i + 1] - t[i]
        row["total"] = t[-1] - t[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def table_timings(frame: pd.DataFrame, facets: dict) -> str:
    labels = [label for _, label in STAGES if label in frame.columns]
    L = [r"\begin{tabular}{l" + "r" * (len(labels) + 2) + "}", r"\toprule",
         "Scene & facets & " + " & ".join(labels) + r" & Total \\",
         " & & " + " & ".join(["(s)"] * len(labels)) + r" & (s) \\", r"\midrule"]
    for case, r in frame.iterrows():
        L.append(f"{case.replace('lisbon', 'Lisbon ')} & {facets.get(case, ''):,} & "
                 + " & ".join(f"{r[l]:.0f}" for l in labels) + f" & {r['total']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def facet_counts(root: Path) -> dict:
    out = {}
    for case in CASES:
        rep = root / "run_output" / case / "thermal_out" / "selection_report.txt"
        if rep.is_file():
            for line in rep.read_text().splitlines():
                if "thermal facets kept" in line:
                    out[case] = int(line.split(":")[1].replace(",", ""))
    return out


def daytime_mask(times_csv: Path) -> np.ndarray:
    t = pd.read_csv(times_csv)
    return t["elevation_deg"].to_numpy(float) > 0.0


def variant_points(path: Path, case: str) -> pd.DataFrame:
    """Comparison points of one case of a variant.

    A variant tree starts as a copy of the reference outputs, so a comparison
    file is only trusted when the variant's own comparison step has run
    (compare_run.log present) and the file is newer than the variant's
    radiant assembly; otherwise the stale copy would masquerade as a result."""
    p = path / "run_output" / case / "validation" / "mrt_lisbon" / "radiant_flux_comparison_points.csv"
    log = path / "run_output" / "compare_run.log"
    tmrt = path / "run_output" / case / "mrt_facet_out" / "tmrt_matrix_C.npy"
    if path.name != "" and (path / "variants").exists() is False and path.parent.name == "variants":
        if not log.is_file() or p.stat().st_mtime < tmrt.stat().st_mtime:
            raise FileNotFoundError(f"{path.name}/{case}: comparison not yet run for this variant")
    pts, _ = apply_measurement_qc(pd.read_csv(p))
    return pts


def day_stats(pts: pd.DataFrame) -> dict:
    d = pts[(pts["period"] == "day") & ~pts["globe_spinup_affected"].astype(bool)]
    g = stats(d["globe_transient_temperature_C"].to_numpy(float),
              d["measured_black_globe_temperature_c"].to_numpy(float))
    dd = pts[pts["period"] == "day"]
    lup = stats(dd["sensor_longwave_up_Wm2"].to_numpy(float), dd["measured_lwout_wm2"].to_numpy(float))
    ldn = stats(dd["sensor_longwave_down_Wm2"].to_numpy(float), dd["measured_lwin_wm2"].to_numpy(float))
    kdn = stats(dd["sensor_shortwave_down_Wm2"].to_numpy(float), dd["measured_swin_wm2"].to_numpy(float))
    nn = pts[(pts["period"] == "night") & ~pts["globe_spinup_affected"].astype(bool)]
    gn = stats(nn["globe_transient_temperature_C"].to_numpy(float),
               nn["measured_black_globe_temperature_c"].to_numpy(float))
    return {"globe_day_mbe": g["mbe"], "globe_day_rmse": g["rmse"],
            "globe_night_mbe": gn["mbe"], "globe_night_rmse": gn["rmse"],
            "lup_rmse": lup["rmse"], "lup_mbe": lup["mbe"], "ldn_mbe": ldn["mbe"],
            "kdn_rmse": kdn["rmse"], "kdn_mbe": kdn["mbe"]}


def convergence(root: Path, case: str = "lisbon1") -> list[dict]:
    ref_dir = root / "run_output" / case / "mrt_facet_out"
    ref = np.load(ref_dir / "tmrt_matrix_C.npy").astype(float)
    day = daytime_mask(ref_dir / "times.csv")
    ref_stats = day_stats(variant_points(root, case))
    rows = [{"variant": "reference", "label": "Reference (as reported)",
             "max_dT": 0.0, "rms_dT": 0.0, "facets": facet_counts(root).get(case),
             **ref_stats}]
    for name, label in CONVERGENCE:
        vdir = root / "variants" / name
        mat = vdir / "run_output" / case / "mrt_facet_out" / "tmrt_matrix_C.npy"
        if not mat.is_file():
            print(f"  {name}: missing"); continue
        T = np.load(mat).astype(float)
        diff = (T - ref)[day]
        rep = vdir / "run_output" / case / "thermal_out" / "selection_report.txt"
        n_facets = None
        if rep.is_file():
            for line in rep.read_text().splitlines():
                if "thermal facets kept" in line:
                    n_facets = int(line.split(":")[1].replace(",", ""))
        tl = vdir / "run_output" / case / "stage_timings.log"
        seconds = None
        if tl.is_file():
            e = [l.split("\t") for l in tl.read_text().splitlines()]
            seconds = float(e[-1][0]) - float(e[0][0])
        try:
            vstats = day_stats(variant_points(vdir, case))
        except FileNotFoundError as exc:
            print(f"  {exc}"); continue
        rows.append({"variant": name, "label": label, "facets": n_facets,
                     "max_dT": float(np.abs(diff).max()),
                     "rms_dT": float(np.sqrt((diff ** 2).mean())),
                     "mean_dT": float(diff.mean()), "seconds": seconds, **vstats})
    return rows


def table_convergence(rows) -> str:
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Variant & Facets & $\max|\Delta T_{\mathrm{mrt}}|$ & RMS $\Delta T_{\mathrm{mrt}}$ & "
         r"Globe MBE/RMSE & $L\!\uparrow$ RMSE & $K\!\downarrow$ RMSE \\",
         r" & & (K) & (K) & (K) & (\si{\watt\per\metre\squared}) & (\si{\watt\per\metre\squared}) \\",
         r"\midrule"]
    for r in rows:
        L.append(f"{r['label']} & {r['facets'] or '':,} & {r['max_dT']:.2f} & {r['rms_dT']:.3f} & "
                 f"{r['globe_day_mbe']:+.2f} / {r['globe_day_rmse']:.2f} & {r['lup_rmse']:.1f} & "
                 f"{r['kdn_rmse']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def ablation(root: Path) -> list[dict]:
    def pooled(points_root: Path) -> dict:
        frames = []
        for case in CASES:
            try:
                frames.append(variant_points(points_root, case))
            except (FileNotFoundError, OSError) as exc:
                print(f"  {exc}"); return None
        return day_stats(pd.concat(frames, ignore_index=True))
    rows = [{"variant": "reference", "label": "As reported", **pooled(root)}]
    for name, label in ABLATION:
        s = pooled(root / "variants" / name)
        if s is None:
            print(f"  {name}: missing"); continue
        rows.append({"variant": name, "label": label, **s})
    return rows


def table_ablation(rows) -> str:
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Variant & \multicolumn{2}{c}{Globe, day} & Globe, night & $L\!\uparrow$ RMSE & "
         r"$L\!\downarrow$ MBE & $K\!\downarrow$ RMSE \\",
         r" & MBE (K) & RMSE (K) & MBE (K) & (\si{\watt\per\metre\squared}) & "
         r"(\si{\watt\per\metre\squared}) & (\si{\watt\per\metre\squared}) \\", r"\midrule"]
    for r in rows:
        L.append(f"{r['label']} & {r['globe_day_mbe']:+.2f} & {r['globe_day_rmse']:.2f} & "
                 f"{r['globe_night_mbe']:+.2f} & {r['lup_rmse']:.1f} & {r['ldn_mbe']:+.1f} & "
                 f"{r['kdn_rmse']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--paper-dir", type=Path, default=Path("/home/harshin/files/fastUTEC paper"))
    args = ap.parse_args()
    out = args.paper_dir
    tf = timings(args.root)
    print(tf.round(0).to_string())
    (out / "table_timings.tex").write_text(table_timings(tf, facet_counts(args.root)) + "\n")
    conv = convergence(args.root)
    for r in conv:
        print(f"  {r['label']:45s} facets {r['facets']}  max|dT| {r['max_dT']:.3f}  rms {r['rms_dT']:.4f}  "
              f"globe {r['globe_day_mbe']:+.2f}/{r['globe_day_rmse']:.2f}  Lup {r['lup_rmse']:.1f}  Kdn {r['kdn_rmse']:.0f}"
              + (f"  {r['seconds']:.0f}s" if r.get('seconds') else ""))
    (out / "table_convergence.tex").write_text(table_convergence(conv) + "\n")
    abl = ablation(args.root)
    for r in abl:
        print(f"  {r['label']:40s} globe day {r['globe_day_mbe']:+.2f}/{r['globe_day_rmse']:.2f}  "
              f"night {r['globe_night_mbe']:+.2f}  Lup RMSE {r['lup_rmse']:.1f}  Ldn MBE {r['ldn_mbe']:+.1f}  Kdn RMSE {r['kdn_rmse']:.0f}")
    (out / "table_ablation.tex").write_text(table_ablation(abl) + "\n")
    (out / "convergence_summary.json").write_text(json.dumps(
        {"timings_s": tf.to_dict(orient="index"), "convergence": conv, "ablation": abl},
        indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
