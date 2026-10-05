#!/usr/bin/env python3
"""paper_solweig_pool.py -- pool the six Lisbon SOLWEIG/TREC-Route comparisons
written by paper_solweig_compare.py (same samples, same QC, spin-up excluded
for the globe) and print the pooled and per-campaign statistics."""
from __future__ import annotations
import json
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


def main():
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
