#!/usr/bin/env python3
"""Validation statistics for the held-out Lisbon daytime walks (2nd and 3rd).

Reads run_output/<case>/validation/mrt_lisbon/radiant_flux_comparison_points.csv
(written by compare_mrt_lisbon_data.py) and reports, per walk and pooled, the
quantities of the paper's Tables 4-5 and Sec. 4.4:

  * radiometer channels (K-down, K-up, L-down, L-up): n, means, MBE, RMSE
  * black globe, spin-up excluded: n, means, MBE, RMSE, cRMSE, sigma_obs, r
    (+ within-walk r for pooled sets)
  * daytime sun/shade registration: agreement, Cohen's kappa, measured and
    modelled sunlit fraction, K-down RMSE where the classes agree/disagree

Every statistic and the measurement QC are IMPORTED from
paper_validation_figures.py (apply_measurement_qc, stats, shadow_registration,
CHANNELS, within_case_r), so the definitions are the paper's. Nothing is
written to the paper folder. By default only the day route (route 1) of a
held-out case is used: its night route is the published night walk again,
simulated on another date. The undated walk (lisbon2_w3, date assumed) is
reported per walk but left out of the pooled held-out set unless
--pool-assumed-date is given.

  python3 paper_adhoc/heldout_summary.py                 # held-out walks
  python3 paper_adhoc/heldout_summary.py --verify-published
      # recompute the published 1st-walk numbers (lisbon1..6, day and night)
      # and compare them with the paper's validation_summary.json and Table 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_validation_figures import (CHANNELS, apply_measurement_qc,  # noqa: E402
                                      shadow_registration, stats, within_case_r)
from paper_tex import latex_minus  # noqa: E402

HELDOUT = [f"lisbon{a}_w{w}" for a in range(1, 7) for w in (2, 3)]
PUBLISHED = [f"lisbon{a}" for a in range(1, 7)]
PAPER_DIR = Path("/home/harshin/files/fastUTEC paper")
DEFAULT_OUT = Path("/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/"
                   "98394eca-18c6-484b-8728-17aede7e7821/scratchpad/reruns/heldout")
GLOBE_MODEL, GLOBE_MEAS = "globe_transient_temperature_C", "measured_black_globe_temperature_c"
CHANNEL_KEY = {"sensor_shortwave_down_Wm2": "K_down", "sensor_shortwave_up_Wm2": "K_up",
               "sensor_longwave_down_Wm2": "L_down", "sensor_longwave_up_Wm2": "L_up"}


def load(cases: list[str], root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for case in cases:
        path = root / "run_output" / case / "validation" / "mrt_lisbon" / \
            "radiant_flux_comparison_points.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
        else:
            missing.append(case)
    if not frames:
        raise SystemExit(f"no comparison points for {cases}; run "
                         "paper_adhoc/run_heldout_walks.sh first")
    return pd.concat(frames, ignore_index=True), missing


def walk_info(case: str, root: Path) -> dict:
    case_dir = root / "input" / case
    manifest = json.loads((case_dir / "case.json").read_text())
    route = json.loads((case_dir / "routes" / "route_1.json").read_text())
    info = {"date": manifest["simulation_defaults"]["date"],
            "day_layer": route["measurement_provenance"]["source_layer"],
            "date_is_assumed": bool(route["measurement_provenance"].get("date_is_assumed")),
            "route_length_m": route["length_m"], "n_route_samples": route["n_points"]}
    prov = case_dir / "weather" / "weather_from_sensors_provenance.json"
    if prov.is_file():
        solar = json.loads(prov.read_text())["solar"]
        info["baseline_linke_turbidity"] = solar.get("baseline_linke_turbidity")
        info["unshaded_clearness_median"] = solar["envelope_diagnostics"].get(
            "unshaded_ratio_median")
    return info


def globe_points(points: pd.DataFrame) -> pd.DataFrame:
    """As figure_globe/table_globe: spin-up samples excluded, pairs complete."""
    d = points
    if "globe_spinup_affected" in d:
        d = d[~d["globe_spinup_affected"].astype(bool)]
    return d.dropna(subset=[GLOBE_MEAS, GLOBE_MODEL])


def block(points: pd.DataFrame, periods: list[str]) -> dict:
    """All statistics for one set of (QC'd) comparison points."""
    out = {"cases": sorted(points["case_id"].unique()), "n_points": int(len(points))}
    radio = {}
    for mcol, ocol, _label in CHANNELS:
        for period in periods:
            d = points[points["period"] == period].dropna(subset=[mcol, ocol])
            if len(d):
                radio[f"{CHANNEL_KEY[mcol]}_{period}"] = stats(d[mcol].values, d[ocol].values)
    out["radiometer"] = radio
    g = globe_points(points)
    out["globe"] = {}
    for period in periods:
        s = g[g["period"] == period]
        if len(s):
            st = stats(s[GLOBE_MODEL].values, s[GLOBE_MEAS].values)
            if s["case_id"].nunique() > 1:
                st["within_walk_r"] = within_case_r(s, GLOBE_MODEL, GLOBE_MEAS)
            out["globe"][period] = st
    if "day" in periods and (points["period"] == "day").any():
        out["shadow"] = shadow_registration(points, None)
    return out


def summarise(cases: list[str], root: Path, periods: list[str],
              pool_exclude: list[str]) -> dict:
    raw, missing = load(cases, root)
    raw = raw[raw["period"].isin(periods)]
    points, qc = apply_measurement_qc(raw)
    result = {"periods": periods, "measurement_qc": qc, "missing_cases": missing,
              "per_walk": {}}
    for case, sub in points.groupby("case_id", sort=False):
        entry = walk_info(case, root)
        entry.update(block(sub, periods))
        result["per_walk"][case] = entry
    pooled = points[~points["case_id"].isin(pool_exclude)]
    result["pooled"] = block(pooled, periods)
    result["pooled"]["excluded_from_pool"] = sorted(set(pool_exclude) & set(points["case_id"]))
    return result


def fmt(value, spec):
    return "--" if value is None or not np.isfinite(value) else format(value, spec)


def latex_tables(held: dict, ref: dict | None) -> str:
    rows = [r"% Held-out daytime walks (2nd, 3rd); statistics as Tables 4-5 and Sec. 4.4.",
            r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
            r"Walk & Date & $n_g$ & MBE$_g$ & RMSE$_g$ & cRMSE$_g$ & $\sigma_{\mathrm{obs}}$ & "
            r"MBE$_{K\downarrow}$ & Agree & $\kappa$ & Sunlit \\",
            r" & & & (K) & (K) & (K) & (K) & (W\,m$^{-2}$) & (\%) & & (\%) \\", r"\midrule"]

    def row(name, date, b):
        g = b["globe"].get("day", {})
        k = b["radiometer"].get("K_down_day", {})
        s = b.get("shadow", {})
        return (f"{name} & {date} & {g.get('n', 0)} & {fmt(g.get('mbe'), '+.2f')} & "
                f"{fmt(g.get('rmse'), '.2f')} & {fmt(g.get('crmse'), '.2f')} & "
                f"{fmt(g.get('sd_obs'), '.2f')} & {fmt(k.get('mbe'), '+.0f')} & "
                f"{fmt(100 * s.get('classification_agreement', np.nan), '.1f')} & "
                f"{fmt(s.get('cohen_kappa'), '.2f')} & "
                f"{fmt(100 * s.get('model_sunlit_fraction', np.nan), '.0f')}/"
                f"{fmt(100 * s.get('measured_sunlit_fraction', np.nan), '.0f')} \\\\")

    for case, b in held["per_walk"].items():
        name = case.replace("lisbon", "Lisbon ").replace("_w", ", walk ")
        rows.append(row(name, b["date"] + ("$^*$" if b["date_is_assumed"] else ""), b))
    rows.append(r"\midrule")
    rows.append(row(r"\textbf{Held-out pooled}", "", held["pooled"]))
    if ref is not None:
        rows.append(row(r"\textbf{1st walks (published)}", "", ref["pooled"]))
    rows += [r"\bottomrule", r"\end{tabular}", "",
             r"% Radiometer, daytime, pooled (as Table 4).",
             r"\begin{tabular}{lrrrrr" + ("rrr" if ref else "") + "}", r"\toprule",
             r"Channel & $n$ & Meas. & Model & MBE & RMSE"
             + (r" & $n_{1st}$ & MBE$_{1st}$ & RMSE$_{1st}$" if ref else "") + r" \\",
             r"\midrule"]
    for mcol, _ocol, label in CHANNELS:
        key = f"{CHANNEL_KEY[mcol]}_day"
        st = held["pooled"]["radiometer"].get(key, {})
        line = (f"{label} & {st.get('n', 0)} & {fmt(st.get('measured_mean'), '.1f')} & "
                f"{fmt(st.get('model_mean'), '.1f')} & {fmt(st.get('mbe'), '+.1f')} & "
                f"{fmt(st.get('rmse'), '.1f')}")
        if ref:
            r = ref["pooled"]["radiometer"].get(key, {})
            line += (f" & {r.get('n', 0)} & {fmt(r.get('mbe'), '+.1f')} & "
                     f"{fmt(r.get('rmse'), '.1f')}")
        rows.append(line + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"% $^*$ date absent from the source layer; 2023-08-13 assumed "
             r"(bracketed 11-16 Aug by logger record order). Agree/kappa: "
             r"daytime sun/shade classification; Sunlit: modelled/measured sunlit share."]
    return latex_minus("\n".join(rows)) + "\n"


def verify_published(root: Path, out: Path) -> int:
    """Recompute the published numbers and compare with the paper's files."""
    mine = summarise(PUBLISHED, root, ["day", "night"], [])
    paper = json.loads((PAPER_DIR / "validation_summary.json").read_text())
    checks = []

    def check(label, a, b, tol=1e-9):
        ok = (a is not None and b is not None and abs(float(a) - float(b)) <= tol)
        checks.append({"quantity": label, "recomputed": a, "published": b, "match": ok})

    check("measurement QC rejected", mine["measurement_qc"]["n_rejected"],
          paper["measurement_qc"]["n_rejected"])
    for period in ("day", "night"):
        for key in ("n", "mbe", "rmse", "crmse", "sd_obs", "r"):
            check(f"globe pooled {period} {key}", mine["pooled"]["globe"][period][key],
                  paper["globe"][period][key])
    for key in ("n", "classification_agreement", "cohen_kappa",
                "measured_sunlit_fraction", "model_sunlit_fraction",
                "rmse_agree", "rmse_disagree"):
        check(f"shadow {key}", mine["pooled"]["shadow"][key], paper["shadow"][key])
    for mcol, ocol, label in CHANNELS:
        check(f"radiometer day+night {CHANNEL_KEY[mcol]} RMSE",
              _all_period_rmse(root, mcol, ocol), paper["radiometer"][label]["rmse"])
    # Table 4 rows (per channel and period, rounded as printed)
    names = iter(CHANNEL_KEY.values())
    channel = None
    for line in (PAPER_DIR / "table_radiometer.tex").read_text().splitlines():
        m = re.match(r"\s*([A-Za-z ]*?)\s*&\s*(day|night)\s*&\s*(\d+)\s*&\s*(\S+)\s*&\s*"
                     r"(\S+)\s*&\s*(\S+)\s*&\s*([\d.]+)", line)
        if not m:
            continue
        if m.group(2) == "day":
            channel = next(names)
        st = mine["pooled"]["radiometer"][f"{channel}_{m.group(2)}"]
        num = lambda v: float(v.replace("$-$", "-"))
        check(f"Table 4 {channel} {m.group(2)} n", st["n"], int(m.group(3)))
        check(f"Table 4 {channel} {m.group(2)} MBE", round(st["mbe"], 1), num(m.group(6)))
        check(f"Table 4 {channel} {m.group(2)} RMSE", round(st["rmse"], 1), num(m.group(7)))
    # Table 5 rows (per case, rounded as printed)
    table = (PAPER_DIR / "table_globe.tex").read_text()
    current = None
    for line in table.splitlines():
        m = re.match(r"\s*(Lisbon (\d)|)\s*&\s*(day|night)\s*&\s*(\d+)\s*&\s*([\d.]+)\s*&\s*"
                     r"([\d.]+)\s*&\s*(\S+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)", line)
        if not m:
            continue
        if m.group(2):
            current = f"lisbon{m.group(2)}"
        period = m.group(3)
        st = mine["per_walk"][current]["globe"][period]
        mbe = float(m.group(7).replace("$-$", "-"))
        check(f"Table 5 {current} {period} n", st["n"], int(m.group(4)))
        check(f"Table 5 {current} {period} MBE", round(st["mbe"], 2), mbe, 1e-9)
        check(f"Table 5 {current} {period} RMSE", round(st["rmse"], 2), float(m.group(8)), 1e-9)
        check(f"Table 5 {current} {period} cRMSE", round(st["crmse"], 2), float(m.group(9)), 1e-9)
        check(f"Table 5 {current} {period} sigma_obs", round(st["sd_obs"], 2),
              float(m.group(10)), 1e-9)
    n_ok = sum(c["match"] for c in checks)
    print(f"verify-published: {n_ok}/{len(checks)} quantities reproduced exactly")
    for c in checks:
        if not c["match"]:
            print(f"  MISMATCH {c['quantity']}: {c['recomputed']} vs {c['published']}")
    l1 = mine["per_walk"]["lisbon1"]
    print(f"  lisbon1 day globe: n={l1['globe']['day']['n']} MBE {l1['globe']['day']['mbe']:+.2f} "
          f"RMSE {l1['globe']['day']['rmse']:.2f} cRMSE {l1['globe']['day']['crmse']:.2f} "
          f"(Table 5: 259, +3.62, 3.71, 0.84); agreement "
          f"{100 * l1['shadow']['classification_agreement']:.1f}%, kappa "
          f"{l1['shadow']['cohen_kappa']:.2f}")
    (out / "verify_published.json").write_text(
        json.dumps({"checks": checks, "recomputed": mine}, indent=2, default=float) + "\n")
    return 0 if n_ok == len(checks) else 1


def _all_period_rmse(root: Path, mcol: str, ocol: str) -> float:
    """validation_summary.json 'radiometer' pools day and night (figure_radiometer)."""
    raw, _ = load(PUBLISHED, root)
    points, _qc = apply_measurement_qc(raw)
    d = points.dropna(subset=[mcol, ocol])
    return stats(d[mcol].values, d[ocol].values)["rmse"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cases", nargs="+", default=HELDOUT)
    ap.add_argument("--include-night", action="store_true",
                    help="also tabulate the night route of each case")
    ap.add_argument("--pool-assumed-date", action="store_true",
                    help="include walks whose date was assumed in the pooled set")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify-published", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_dir.resolve().is_relative_to(PAPER_DIR):
        raise SystemExit("refusing to write into the paper folder")
    if args.verify_published:
        return verify_published(args.root, args.out_dir)

    periods = ["day", "night"] if args.include_night else ["day"]
    assumed = [] if args.pool_assumed_date else [
        c for c in args.cases
        if (args.root / "input" / c / "case.json").is_file()
        and walk_info(c, args.root)["date_is_assumed"]]
    held = summarise(args.cases, args.root, periods, assumed)
    ref = summarise(PUBLISHED, args.root, ["day"], [])
    summary = {"heldout": held, "published_first_walks_day": ref}
    (args.out_dir / "heldout_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")
    (args.out_dir / "heldout_table.tex").write_text(latex_tables(held, ref))

    def line(name, b):
        g, s = b["globe"].get("day", {}), b.get("shadow", {})
        k = b["radiometer"].get("K_down_day", {})
        return (f"  {name:24s} globe n={g.get('n', 0):4d} MBE {g.get('mbe', np.nan):+.2f} "
                f"RMSE {g.get('rmse', np.nan):.2f} cRMSE {g.get('crmse', np.nan):.2f} "
                f"sd {g.get('sd_obs', np.nan):.2f} | Kdown MBE {k.get('mbe', np.nan):+6.1f} "
                f"RMSE {k.get('rmse', np.nan):5.1f} | agree "
                f"{100 * s.get('classification_agreement', np.nan):5.1f}% kappa "
                f"{s.get('cohen_kappa', np.nan):.2f} sunlit model/meas "
                f"{100 * s.get('model_sunlit_fraction', np.nan):.0f}/"
                f"{100 * s.get('measured_sunlit_fraction', np.nan):.0f}%")
    print(f"measurement QC: {held['measurement_qc']['n_rejected']} of "
          f"{held['measurement_qc']['n_before']} held-out samples rejected")
    if held["missing_cases"]:
        print(f"no results yet for: {', '.join(held['missing_cases'])}")
    for case, b in held["per_walk"].items():
        print(line(f"{case} {b['date']}{'*' if b['date_is_assumed'] else ''}", b))
    print(line("HELD-OUT POOLED", held["pooled"]))
    print(line("1st walks (published)", ref["pooled"]))
    print(f"wrote {args.out_dir / 'heldout_summary.json'} and heldout_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
