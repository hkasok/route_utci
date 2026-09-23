#!/usr/bin/env python3
"""paper_lisbon_route_table.py -- modelled exposure along the twelve Lisbon
walks (stage 08 UTCI + stage 09 JOS-3), written as a LaTeX tabular for the
paper. Reads pipeline outputs only; performs no physics."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-output", type=Path, default=Path("run_output"))
    ap.add_argument("--out", type=Path, default=Path(
        "/home/harshin/files/fastUTEC paper/table_lisbon_routes.tex"))
    args = ap.parse_args()

    rows = []
    for case in range(1, 7):
        base = args.run_output / f"lisbon{case}" / "viz"
        utci = pd.read_csv(base / "route_utci" / "route_ranking_summary.csv")
        jos = pd.read_csv(base / "route_jos3" / "route_ranking_summary.csv")
        merged = utci.merge(jos[["route_id", "final_tcore_rise_c"]], on="route_id")
        for period in ("day", "night"):
            r = merged[merged["route_name"].str.startswith(period)]
            if len(r) != 1:
                raise SystemExit(f"lisbon{case}: expected one {period} walk")
            r = r.iloc[0]
            rows.append((case, period, r))

    lines = [r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Case & Walk & Length & Dur. & $\overline{T_a}$ & "
             r"$\overline{T_{\mathrm{mrt}}}$ & $\overline{UTCI}$ & "
             r"$UTCI_{\max}$ & $D$ & $\Delta T_{\mathrm{core}}$ \\",
             r" & & (\si{\metre}) & (\si{\minute}) & (\si{\celsius}) & "
             r"(\si{\celsius}) & (\si{\celsius}) & (\si{\celsius}) & "
             r"(\si{\celsius\minute}) & (\si{\celsius}) \\", r"\midrule"]
    for case, period, r in rows:
        label = f"Lisbon {case}" if period == "day" else ""
        lines.append(
            f"{label} & {period} & {r.length_m:.0f} & {r.walk_duration_min:.0f} & "
            f"{r.mean_ta_c:.1f} & {r.mean_tmrt_c:.1f} & {r.mean_utci_c:.1f} & "
            f"{r.max_utci_c:.1f} & {r.strong_stress_dose_degmin:.0f} & "
            f"{r.final_tcore_rise_c:.3f} \\\\")
        if period == "night" and case < 6:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
