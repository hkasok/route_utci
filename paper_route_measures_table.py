#!/usr/bin/env python3
"""Paper tables of the route comfort and heat-strain measures (campus).

table_route_measures.tex        13:00 departure, every measure per route, its
                                resolution and the route order it gives (main text)
table_route_measures_sweep.tex  least-loaded route(s) under each measure at every
                                departure of the fixed-clothing sweep
                                (supplement; paper_adhoc/run_route_measures_sweep.sh)
route_measures_summary.json     numbers quoted in the text

Ties use the ranking Monte Carlo (paper_ranking_uncertainty.py), which
propagates the validated sun/shade and radiant-load error at 13:00, 16:00 and
17:00. At those departures two routes are tied under a measure when the
2.5-97.5 percentile band of their difference over the draws includes zero; at
the other sweep departures, when their difference is within the largest shift
of that pair's difference found at the three. The measure's resolution quoted
in the text is the largest such shift over all pairs.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROUTES = (1, 2, 3, 4)
# (label, unit, source, column, decimals, Monte Carlo key or None)
ROWS = [
    ("route", None),
    ("Length", r"\si{\metre}", "jos3", "length_m", 0, None),
    ("Duration", r"\si{\minute}", "jos3", "walk_duration_min", 1, None),
    ("Time sunlit, $t_{\\mathrm{sun}}$", r"\si{\minute}", "jos3", "time_sunlit_min", 1, None),
    ("Mean $T_{\\mathrm{mrt}}$", r"\si{\celsius}", "jos3", "mean_tmrt_c", 1, None),
    ("comfort", None),
    ("Walk UTCI, $\\overline{\\mathrm{UTCI}}$", r"\si{\celsius}", "utci", "walk_utci_c", 2, "walk_utci_c"),
    ("Worst stretch, $\\mathrm{UTCI}_{3}$", r"\si{\celsius}", "utci", "worst_3min_utci_c", 2,
     "worst_3min_utci_c"),
    ("stress", None),
    ("Very strong stress, $t_{38}$", r"\si{\minute}", "utci", "very_strong_stress_min", 1,
     "very_strong_stress_min"),
    ("Strong-stress dose, $D$", r"\si{\celsius\minute}", "utci", "strong_stress_dose_degmin", 0,
     "strong_stress_dose_degmin"),
    ("body", None),
    ("Wettedness $\\geq 0.36$, $t_{w}$", r"\si{\minute}", "jos3", "wettedness_discomfort_min", 1,
     "wettedness_discomfort_min"),
    ("Core rise, $\\Delta T_{\\mathrm{core}}$", r"\si{\celsius}", "jos3", "final_tcore_rise_c", 3,
     "tcore_rise_c"),
    ("\\quad radiant part, $\\Delta T_{\\mathrm{core}}^{\\mathrm{rad}}$", r"\si{\celsius}", "jos3",
     "radiation_attributable_rise_c", 3, "tcore_rad_c"),
    ("Peak core rise, $\\Delta T_{\\mathrm{core}}^{\\mathrm{peak}}$", r"\si{\celsius}", "jos3",
     "peak_tcore_rise_c", 3, "peak_tcore_rise_c"),
    ("Heat stored, $\\Delta H$", r"\si{\kilo\joule}", "jos3", "heat_stored_kj", 0, "heat_stored_kj"),
    ("\\quad radiant part, $\\Delta H^{\\mathrm{rad}}$", r"\si{\kilo\joule}", "jos3",
     "heat_stored_radiant_kj", 0, "heat_stored_rad_kj"),
]
BLOCKS = {"route": "Route", "comfort": "Thermal comfort while walking (UTCI)",
          "stress": "Heat-stress exposure (UTCI)", "body": "Physiological response (JOS-3)"}


def pair_shifts(mc_measure):
    """Largest shift of each route pair's difference over the MC departures."""
    out = {}
    for key, q in mc_measure["pairs"].items():
        pair = tuple(int(x) for x in key.split(":")[1].split("-"))
        lo, hi = q["band"]
        g = q["baseline_gap"]
        out[pair] = max(out.get(pair, 0.0), abs(g - lo), abs(hi - g))
    return out


def tie_rule(mc_measure, departure):
    """Function (values, a, b) -> True if routes a and b are tied."""
    bands = {}
    for key, q in mc_measure["pairs"].items():
        dep, pair = key.split(":")
        if abs(float(dep) - departure) < 1e-6:
            bands[tuple(int(x) for x in pair.split("-"))] = q["resolved"]
    shifts = pair_shifts(mc_measure)

    def tied(values, a, b):
        pair = tuple(sorted((a, b)))
        if pair in bands:                       # Monte Carlo at this departure
            return not bands[pair]
        return abs(values[b] - values[a]) <= shifts[pair]
    return tied


def order_string(values, tied):
    """Lowest-to-highest route order with ties marked."""
    keys = sorted(values, key=lambda k: (values[k], k))
    out = str(keys[0])
    for a, b in zip(keys, keys[1:]):
        out += (r"$\approx$" if tied(values, a, b) else "$<$") + str(b)
    return out


def load(viz):
    u = pd.read_csv(viz / "route_utci" / "route_ranking_summary.csv").set_index("route_id")
    j = pd.read_csv(viz / "route_jos3" / "route_ranking_summary.csv").set_index("route_id")
    return {"utci": u, "jos3": j}


def main_table(viz, out, res):
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
        label, unit, s, col, dec, key = row
        vals = {r: float(src[s].loc[r, col]) for r in ROUTES}
        summary[col] = vals
        cells = [f"{vals[r]:.{dec}f}" for r in ROUTES]
        if key is None:
            order = ""
        else:
            order = order_string(vals, tie_rule(res[key], 13.0))
            summary[col + "_order"] = order
        lines.append(f"{label} ({unit}) & " + " & ".join(cells) + f" & {order} \\\\")
    summary["t_peak_after_arrival_min"] = {
        r: float(src["jos3"].loc[r, "t_peak_after_arrival_min"]) for r in ROUTES}
    summary["worst_3min_end_distance_m"] = {
        r: float(src["utci"].loc[r, "worst_3min_end_distance_m"]) for r in ROUTES}
    summary["max_wettedness"] = {r: float(src["jos3"].loc[r, "max_wettedness"]) for r in ROUTES}
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_route_measures.tex").write_text("\n".join(lines) + "\n")
    return summary


SWEEP_COLS = [("walk_utci_c", "utci", r"$\overline{\mathrm{UTCI}}$", "walk_utci_c"),
              ("worst_3min_utci_c", "utci", r"$\mathrm{UTCI}_{3}$", "worst_3min_utci_c"),
              ("very_strong_stress_min", "utci", r"$t_{38}$", "very_strong_stress_min"),
              ("wettedness_discomfort_min", "jos3", r"$t_{w}$", "wettedness_discomfort_min"),
              ("final_tcore_rise_c", "jos3", r"$\Delta T_{\mathrm{core}}$", "tcore_rise_c"),
              ("peak_tcore_rise_c", "jos3", r"$\Delta T_{\mathrm{core}}^{\mathrm{peak}}$",
               "peak_tcore_rise_c"),
              ("heat_stored_kj", "jos3", r"$\Delta H$", "heat_stored_kj")]


def sweep_table(sweep_dir, out, res):
    counts = {c[0]: {} for c in SWEEP_COLS}
    lines = [r"\begin{tabular}{l" + "c" * len(SWEEP_COLS) + "}", r"\toprule",
             "Departure & " + " & ".join(c[2] for c in SWEEP_COLS) + r" \\", r"\midrule"]
    n = 0
    for d in sorted(sweep_dir.glob("dep_*")):
        src = load(d)
        tag = d.name[4:]
        cells = []
        for col, s, _, key in SWEEP_COLS:
            vals = {r: float(src[s].loc[r, col]) for r in ROUTES}
            tied = tie_rule(res[key], int(tag[:2]) + int(tag[2:]) / 60.0)
            r0 = min(ROUTES, key=lambda r: (vals[r], r))
            best = [r for r in ROUTES if r == r0 or tied(vals, r0, r)]
            if len(best) == len(ROUTES):
                cell = "none" if max(vals.values()) == 0 else "all"
                k = cell
            else:
                cell = k = r"$\approx$".join(map(str, best))
            cells.append(cell)
            counts[col][k] = counts[col].get(k, 0) + 1
        n += 1
        lines.append(f"{tag[:2]}:{tag[2:]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_route_measures_sweep.tex").write_text("\n".join(lines) + "\n")
    return {"n_departures": n, "best_route_counts": counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viz", default="run_output/MMC/viz")
    p.add_argument("--sweep", default="run_output/MMC/postprocessing/route_measures_sweep")
    p.add_argument("--mc", default="run_output/MMC/postprocessing/ranking_uncertainty/summary.json")
    p.add_argument("--paper-dir", default="/home/harshin/files/fastUTEC paper")
    a = p.parse_args()
    res = json.loads(Path(a.mc).read_text())["measures"]
    out = Path(a.paper_dir)
    summary = {"resolution": {k: v["resolution"] for k, v in res.items()},
               "departure_13h": main_table(Path(a.viz), out, res),
               "sweep": sweep_table(Path(a.sweep), out, res)}
    (out / "route_measures_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"resolution": summary["resolution"], "sweep": summary["sweep"]}, indent=1))


if __name__ == "__main__":
    main()
