#!/usr/bin/env python3
"""Paper tables of the route comfort and heat-risk measures (campus).

table_route_measures.tex        13:00 departure, every measure per route and
                                the route order it gives (main text)
table_route_measures_sweep.tex  least-loaded route under each measure at every
                                departure of the fixed-clothing sweep
                                (supplement; paper_adhoc/run_route_measures_sweep.sh)
route_measures_summary.json     numbers quoted in the text
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = (1, 2, 3, 4)
# (label, unit, source, column, decimals)
ROWS = [
    ("route", None),
    ("Length", r"\si{\metre}", "jos3", "length_m", 0),
    ("Duration", r"\si{\minute}", "jos3", "walk_duration_min", 1),
    ("Time sunlit, $t_{\\mathrm{sun}}$", r"\si{\minute}", "jos3", "time_sunlit_min", 1),
    ("Mean $T_{\\mathrm{mrt}}$", r"\si{\celsius}", "jos3", "mean_tmrt_c", 1),
    ("comfort", None),
    ("Walk UTCI, $\\overline{UTCI}$", r"\si{\celsius}", "utci", "walk_utci_c", 2),
    ("Very strong stress, $t_{38}$", r"\si{\minute}", "utci", "very_strong_stress_lagged_min", 1),
    ("Worst stretch, $UTCI_{3}$", r"\si{\celsius}", "utci", "worst_3min_utci_c", 2),
    ("Wettedness $>0.36$, $t_{w}$", r"\si{\minute}", "jos3", "wettedness_discomfort_min", 1),
    ("Strong-stress dose, $D$", r"\si{\celsius\minute}", "utci", "strong_stress_dose_degmin", 0),
    ("strain", None),
    ("Core rise, $\\Delta T_{\\mathrm{core}}$", r"\si{\celsius}", "jos3", "final_tcore_rise_c", 3),
    ("\\quad radiant part, $\\Delta T_{\\mathrm{core}}^{\\mathrm{rad}}$", r"\si{\celsius}", "jos3",
     "radiation_attributable_rise_c", 3),
    ("Peak core rise, $\\Delta T_{\\mathrm{core}}^{\\mathrm{peak}}$", r"\si{\celsius}", "jos3", "peak_tcore_rise_c", 3),
    ("Heat stored, $\\Delta H$", r"\si{\kilo\joule}", "jos3", "heat_stored_kj", 0),
    ("\\quad radiant part, $\\Delta H^{\\mathrm{rad}}$", r"\si{\kilo\joule}", "jos3", "heat_stored_radiant_kj", 0),
]
BLOCKS = {"route": "Route", "comfort": "Comfort while walking",
          "strain": "Heat strain"}
RANKED = {"walk_utci_c", "very_strong_stress_lagged_min", "worst_3min_utci_c",
          "wettedness_discomfort_min", "strong_stress_dose_degmin",
          "final_tcore_rise_c", "radiation_attributable_rise_c", "peak_tcore_rise_c",
          "heat_stored_kj", "heat_stored_radiant_kj"}


def order_string(values, decimals):
    """Lowest-to-highest route order; routes equal at the shown precision tie."""
    r = {k: round(float(v), decimals) for k, v in values.items()}
    keys = sorted(r, key=lambda k: (r[k], k))
    out = str(keys[0])
    for a, b in zip(keys, keys[1:]):
        out += ("$=$" if r[b] == r[a] else "$<$") + str(b)
    return out


def load(viz):
    u = pd.read_csv(viz / "route_utci" / "route_ranking_summary.csv").set_index("route_id")
    j = pd.read_csv(viz / "route_jos3" / "route_ranking_summary.csv").set_index("route_id")
    return {"utci": u, "jos3": j}


def main_table(viz, out):
    src = load(viz)
    lines = [r"\begin{tabular}{lrrrrl}", r"\toprule",
             r"Measure & Route~1 & Route~2 & Route~3 & Route~4 & Order \\",
             r"\midrule"]
    summary = {}
    for row in ROWS:
        if row[1] is None:
            if row[0] != "route":
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{6}{l}{\emph{" + BLOCKS[row[0]] + r"}} \\")
            continue
        label, unit, s, col, dec = row
        vals = {r: float(src[s].loc[r, col]) for r in ROUTES}
        summary[col] = vals
        cells = [f"{vals[r]:.{dec}f}" for r in ROUTES]
        order = order_string(vals, dec) if col in RANKED else ""
        lines.append(f"{label} ({unit}) & " + " & ".join(cells) + f" & {order} \\\\")
    peak_t = {r: float(src["jos3"].loc[r, "t_peak_after_arrival_min"]) for r in ROUTES}
    summary["t_peak_after_arrival_min"] = peak_t
    summary["worst_3min_end_distance_m"] = {
        r: float(src["utci"].loc[r, "worst_3min_end_distance_m"]) for r in ROUTES}
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_route_measures.tex").write_text("\n".join(lines) + "\n")
    return summary


SWEEP_COLS = [("walk_utci_c", "utci", 2, r"$\overline{UTCI}$"),
              ("very_strong_stress_lagged_min", "utci", 1, r"$t_{38}$"),
              ("worst_3min_utci_c", "utci", 2, r"$UTCI_{3}$"),
              ("wettedness_discomfort_min", "jos3", 1, r"$t_{w}$"),
              ("final_tcore_rise_c", "jos3", 3, r"$\Delta T_{\mathrm{core}}$"),
              ("peak_tcore_rise_c", "jos3", 3, r"$\Delta T_{\mathrm{core}}^{\mathrm{peak}}$"),
              ("heat_stored_kj", "jos3", 0, r"$\Delta H$")]


def sweep_table(sweep_dir, out):
    rows, best_counts = [], {c[0]: {} for c in SWEEP_COLS}
    lines = [r"\begin{tabular}{l" + "c" * len(SWEEP_COLS) + "}", r"\toprule",
             "Departure & " + " & ".join(c[3] for c in SWEEP_COLS) + r" \\", r"\midrule"]
    for d in sorted(sweep_dir.glob("dep_*")):
        src = load(d)
        tag = d.name[4:]
        cells = []
        for col, s, dec, _ in SWEEP_COLS:
            vals = {r: round(float(src[s].loc[r, col]), dec) for r in ROUTES}
            lo = min(vals.values())
            best = [r for r in ROUTES if vals[r] == lo]
            if len(best) == len(ROUTES):
                cells.append("all " + f"{lo:.{dec}f}")
                key = "all"
            else:
                cells.append("{=}".join(map(str, best)))
                key = "=".join(map(str, best))
            best_counts[col][key] = best_counts[col].get(key, 0) + 1
        rows.append(tag)
        lines.append(f"{tag[:2]}:{tag[2:]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_route_measures_sweep.tex").write_text("\n".join(lines) + "\n")
    return {"n_departures": len(rows), "best_route_counts": best_counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viz", default="run_output/MMC/viz")
    p.add_argument("--sweep", default="run_output/MMC/postprocessing/route_measures_sweep")
    p.add_argument("--paper-dir", default="/home/harshin/files/fastUTEC paper")
    a = p.parse_args()
    out = Path(a.paper_dir)
    summary = {"departure_13h": main_table(Path(a.viz), out),
               "sweep": sweep_table(Path(a.sweep), out)}
    (out / "route_measures_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["sweep"], indent=1))


if __name__ == "__main__":
    main()
