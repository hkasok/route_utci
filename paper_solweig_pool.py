#!/usr/bin/env python3
"""paper_solweig_pool.py -- pool the six Lisbon SOLWEIG/TREC-Route comparisons
written by paper_solweig_compare.py (same samples, same QC, spin-up excluded
for the globe) and print the pooled and per-campaign statistics."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def stats(model, meas):
    m = np.isfinite(model) & np.isfinite(meas)
    d = model[m] - meas[m]
    return dict(n=int(m.sum()), mbe=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                crmse=float(np.sqrt(((d - d.mean()) ** 2).mean())),
                r=float(np.corrcoef(model[m], meas[m])[0, 1]) if m.sum() > 2 and np.std(model[m]) > 0 else float("nan"))


def write_tables(out, paper_dir: Path):
    def cell(sx, unit_k=False):
        f = "{:+.2f}" if unit_k else "{:+.1f}"
        g = "{:.2f}" if unit_k else "{:.1f}"
        return f.format(sx["mbe"]), g.format(sx["rmse"])
    rows = [("$K\\!\\downarrow$ (\\si{\\watt\\per\\metre\\squared})", "K_down", ("day",), False),
            ("$K\\!\\uparrow$ (\\si{\\watt\\per\\metre\\squared})", "K_up", ("day",), False),
            ("$L\\!\\downarrow$ (\\si{\\watt\\per\\metre\\squared})", "L_down", ("day", "night"), False),
            ("$L\\!\\uparrow$ (\\si{\\watt\\per\\metre\\squared})", "L_up", ("day", "night"), False),
            ("Emulated globe (\\si{\\kelvin})", "globe (emulated)", ("day", "night"), True),
            ("Conventional $T_{\\mathrm{mrt}}$ (\\si{\\kelvin})",
             "conventional: cylinder Tmrt vs ISO-converted globe", ("day", "night"), True)]
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r" & & & \multicolumn{2}{c}{TREC-Route} & \multicolumn{2}{c}{SOLWEIG} \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"Quantity & Period & $n$ & MBE & RMSE & MBE & RMSE \\", r"\midrule"]
    for label, key, periods, k in rows:
        for j, per in enumerate(periods):
            a, b = out[per][key]["TREC-Route"], out[per][key]["SOLWEIG"]
            L.append(f"{label if j == 0 else ''} & {per} & {a['n']} & {' & '.join(cell(a, k))} & {' & '.join(cell(b, k))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (paper_dir / "table_solweig.tex").write_text("\n".join(L) + "\n")
    C = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{TREC-Route} & \multicolumn{2}{c}{SOLWEIG} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"Case & Period & MBE (K) & RMSE (K) & MBE (K) & RMSE (K) \\", r"\midrule"]
    for case in sorted(out["day"]["per_case_globe"]):
        for per in ("day", "night"):
            a, b = out[per]["per_case_globe"][case]["TREC-Route"], out[per]["per_case_globe"][case]["SOLWEIG"]
            name = f"Lisbon~{case[-1]}" if per == "day" else ""
            C.append(f"{name} & {per} & {a['mbe']:+.2f} & {a['rmse']:.2f} & {b['mbe']:+.2f} & {b['rmse']:.2f} \\\\")
    C += [r"\bottomrule", r"\end{tabular}"]
    (paper_dir / "table_solweig_cases.tex").write_text("\n".join(C) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", type=Path, default=Path("/home/harshin/files/fastUTEC paper"))
    args = ap.parse_args()
    P = pd.concat([pd.read_csv(ROOT / "run_output" / f"lisbon{i}" / "solweig" / "solweig_comparison_points.csv")
                   for i in range(1, 7)], ignore_index=True)
    tg = P.measured_black_globe_temperature_c.to_numpy(float)
    ta = P.measured_air_temperature_c.to_numpy(float); V = np.maximum(P.measured_wind_ms.to_numpy(float), 0)
    P["iso_tmrt"] = ((tg + 273.15) ** 4 + 1.1e8 * V ** 0.6 / (0.95 * 0.15 ** 0.4) * (tg - ta)) ** 0.25 - 273.15
    chans = (("K_down", "measured_swin_wm2", "sensor_shortwave_down_Wm2", "solweig_kdown"),
             ("K_up", "measured_swout_wm2", "sensor_shortwave_up_Wm2", "solweig_kup"),
             ("L_down", "measured_lwin_wm2", "sensor_longwave_down_Wm2", "solweig_ldown"),
             ("L_up", "measured_lwout_wm2", "sensor_longwave_up_Wm2", "solweig_lup"))
    out = {}
    for per in ("day", "night"):
        D = P[P.period == per]
        G = D[~D.globe_spinup_affected.astype(bool)]
        o = {}
        for name, meas, trec, sol in chans:
            o[name] = {"TREC-Route": stats(D[trec].to_numpy(float), D[meas].to_numpy(float)),
                       "SOLWEIG": stats(D[sol].to_numpy(float), D[meas].to_numpy(float))}
        mg = G.measured_black_globe_temperature_c.to_numpy(float)
        o["globe (emulated)"] = {"TREC-Route": stats(G.globe_transient_temperature_C.to_numpy(float), mg),
                                 "SOLWEIG": stats(G.solweig_globe_C.to_numpy(float), mg),
                                 "SOLWEIG, no beam correction": stats(G.solweig_globe_uncorr_C.to_numpy(float), mg),
                                 "TREC-Route control (resampled)": stats(G.trec_globe_resampled_C.to_numpy(float), mg)}
        o["conventional: cylinder Tmrt vs ISO-converted globe"] = {
            "TREC-Route": stats(G.trec_route_mrt_c.to_numpy(float), G.iso_tmrt.to_numpy(float)),
            "SOLWEIG": stats(G.solweig_tmrt.to_numpy(float), G.iso_tmrt.to_numpy(float))}
        o["per_case_globe"] = {c: {"TREC-Route": stats(g.globe_transient_temperature_C.to_numpy(float), g.measured_black_globe_temperature_c.to_numpy(float)),
                                   "SOLWEIG": stats(g.solweig_globe_C.to_numpy(float), g.measured_black_globe_temperature_c.to_numpy(float))}
                               for c, g in G.groupby("case_id")}
        out[per] = o
    (ROOT / "run_output" / "solweig_lisbon_pooled.json").write_text(json.dumps(out, indent=1))
    (args.paper_dir / "solweig_lisbon_pooled.json").write_text(json.dumps(out, indent=1))
    write_tables(out, args.paper_dir)
    for per in ("day", "night"):
        print(f"\n===== pooled {per}")
        for k, v in out[per].items():
            if k == "per_case_globe":
                continue
            for model, s in v.items():
                print(f"  {k:52s} {model:32s} n={s['n']:5d} MBE {s['mbe']:+8.2f} RMSE {s['rmse']:7.2f} cRMSE {s['crmse']:6.2f} r {s['r']:+.2f}")
        print("  per-case globe (MBE / RMSE / r):")
        for c, v in out[per]["per_case_globe"].items():
            a, b = v["TREC-Route"], v["SOLWEIG"]
            print(f"    {c}: TREC {a['mbe']:+.2f}/{a['rmse']:.2f}/{a['r']:+.2f}   SOLWEIG {b['mbe']:+.2f}/{b['rmse']:.2f}/{b['r']:+.2f}")


if __name__ == "__main__":
    main()
