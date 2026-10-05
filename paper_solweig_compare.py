#!/usr/bin/env python3
"""paper_solweig_compare.py -- score SOLWEIG and TREC-Route against the same
Lisbon measurements, instrument for instrument.

Run with the SOLWEIG environment (needs rasterio):
    /media/harshin/data_drive/solweig/.venv/bin/python paper_solweig_compare.py --case lisbon1

For every paired cart sample in TREC-Route's comparison file
(run_output/<case>/validation/mrt_lisbon/radiant_flux_comparison_points.csv):

* RADIOMETER: SOLWEIG's kdown, kup, ldown, lup at the sample's pixel,
  interpolated linearly between the two bracketing 10-min steps.
* GLOBE: SOLWEIG's globe-configured run (six equal weights, absorptivity 0.95,
  emittance 0.957) gives the absorbed load S = 0.957 sigma Tmrt^4; its beam
  term is corrected from the cube projection (0.20 cos h + sin h / 6) to the
  sphere's 0.25 using SOLWEIG's own shadow output. The load is then passed
  through the SAME transient globe emulator, ventilation, air temperature and
  timestamps TREC-Route's globe used (black_globe.integrate_globe_temperature_C,
  integrated along the cart samples). As a control, TREC-Route's own absorbed
  load is re-integrated along the same samples.
* CONVENTIONAL: SOLWEIG's standing Tmrt against the measured globe converted by
  ISO 7726 with the cart wind, beside the like-for-like globe comparison.

The measurement QC rule of the paper is applied (a daytime sample is dropped
when measured K-down < 10 W m-2 and K-up < 0). Globe statistics exclude each
walk's spin-up (3 tau), as in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import black_globe as bg

ROOT = Path(__file__).resolve().parent
SIGMA = 5.670374419e-8


def stats(model, meas):
    m = np.isfinite(model) & np.isfinite(meas)
    d = model[m] - meas[m]
    if m.sum() < 3:
        return dict(n=int(m.sum()))
    return dict(n=int(m.sum()), mbe=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                crmse=float(np.sqrt(((d - d.mean()) ** 2).mean())),
                r=float(np.corrcoef(model[m], meas[m])[0, 1]))


class RasterSeries:
    def __init__(self, folder: Path, var: str, date: str):
        self.folder, self.var, self.date, self.cache = folder / var, var, date, {}

    def at(self, hh: int, mm: int):
        key = (hh, mm)
        if key not in self.cache:
            p = self.folder / f"{self.var}_{self.date}_{hh:02d}{mm:02d}.tif"
            with rasterio.open(p) as src:
                self.cache[key] = src.read(1).astype(np.float64)
        return self.cache[key]

    def sample(self, hour, row, col, step_h):
        """Linear in time between the bracketing steps, nearest pixel in space."""
        out = np.full(len(hour), np.nan)
        t0 = np.floor(hour / step_h + 1e-9) * step_h
        for i, (h, r, c) in enumerate(zip(hour, row, col)):
            a = t0[i]; b = a + step_h
            w = (h - a) / step_h
            def grid(t):
                t = t % 24.0
                hh = int(t + 1e-9); mm = int(round((t - hh) * 60)) % 60
                return self.at(hh, mm)
            out[i] = (1 - w) * grid(a)[r, c] + w * grid(b)[r, c]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    args = ap.parse_args()
    run = ROOT / "run_output" / args.case
    meta = json.loads((run / "solweig" / "inputs" / "raster_meta.json").read_text())
    case = json.loads((ROOT / "input" / args.case / "case.json").read_text())
    ox, oy = case["coordinates"]["local_origin_x"], case["coordinates"]["local_origin_y"]
    times = pd.read_csv(run / "mrt_facet_out" / "times.csv")
    tt = pd.to_datetime(times.time)
    th = (tt.dt.hour + tt.dt.minute / 60).to_numpy(float)
    step_h = float(th[1] - th[0])
    date = tt.iloc[0].strftime("%Y%m%d")

    P = pd.read_csv(run / "validation" / "mrt_lisbon" / "radiant_flux_comparison_points.csv")
    qc = (P.period == "day") & (P.measured_swin_wm2 < 10) & (P.measured_swout_wm2 < 0)
    P = P[~qc].reset_index(drop=True)
    xl = P.x_projected_m.to_numpy(float) - ox
    yl = P.y_projected_m.to_numpy(float) - oy
    px = meta["pixel_size"]
    rows, cols = meta["shape"]
    col = np.clip(((xl - meta["x_min"]) / px).astype(int), 0, cols - 1)
    row = np.clip(((meta["y_max"] - yl) / px).astype(int), 0, rows - 1)
    hour = P.arrival_hour_local.to_numpy(float)

    sd = run / "solweig" / "run_standing"; sg = run / "solweig" / "run_globe"
    for v in ("kdown", "kup", "ldown", "lup", "tmrt"):
        P[f"solweig_{v}"] = RasterSeries(sd, v, date).sample(hour, row, col, step_h)
    tm_g = RasterSeries(sg, "tmrt", date).sample(hour, row, col, step_h)
    shadow = RasterSeries(sg, "shadow", date).sample(hour, row, col, step_h)
    elev = np.interp(hour, th, times.elevation_deg.to_numpy(float))
    dni = np.interp(hour, th, times.DNI_Wm2.to_numpy(float))
    h = np.radians(np.clip(elev, 0, 90))
    cube = 0.20 * np.cos(h) + np.sin(h) / 6.0
    beam_corr = np.where(elev > 0, 0.95 * dni * shadow * (0.25 - cube), 0.0)
    P["solweig_globe_load_Wm2"] = 0.957 * SIGMA * (tm_g + 273.15) ** 4 + beam_corr
    P["solweig_globe_load_uncorrected_Wm2"] = 0.957 * SIGMA * (tm_g + 273.15) ** 4
    P["solweig_shadow"] = shadow

    spec = bg.DEFAULT_GLOBE
    P["solweig_globe_C"] = np.nan; P["solweig_globe_uncorr_C"] = np.nan; P["trec_globe_resampled_C"] = np.nan
    for _, g in P.groupby(["route_id", "period"]):
        g = g.sort_values("seq"); idx = g.index
        el = (g.arrival_hour_local.to_numpy(float) - g.arrival_hour_local.iloc[0]) * 3600.0
        ta = g.trec_route_air_temperature_c.to_numpy(float)
        vent = g.globe_ventilation_ms.to_numpy(float)
        for src, dst in (("solweig_globe_load_Wm2", "solweig_globe_C"),
                         ("solweig_globe_load_uncorrected_Wm2", "solweig_globe_uncorr_C"),
                         ("globe_absorbed_flux_Wm2", "trec_globe_resampled_C")):
            T, _ = bg.integrate_globe_temperature_C(el, g[src].to_numpy(float), ta, vent, spec)
            P.loc[idx, dst] = T

    out = {}
    chans = (("K_down", "measured_swin_wm2", "sensor_shortwave_down_Wm2", "solweig_kdown"),
             ("K_up", "measured_swout_wm2", "sensor_shortwave_up_Wm2", "solweig_kup"),
             ("L_down", "measured_lwin_wm2", "sensor_longwave_down_Wm2", "solweig_ldown"),
             ("L_up", "measured_lwout_wm2", "sensor_longwave_up_Wm2", "solweig_lup"))
    for per in ("day", "night"):
        m = (P.period == per).to_numpy()
        out[per] = {}
        for name, meas, trec, sol in chans:
            out[per][name] = dict(trec=stats(P[trec].to_numpy(float)[m], P[meas].to_numpy(float)[m]),
                                  solweig=stats(P[sol].to_numpy(float)[m], P[meas].to_numpy(float)[m]))
        mg = m & ~P.globe_spinup_affected.astype(bool).to_numpy()
        tg = P.measured_black_globe_temperature_c.to_numpy(float)
        out[per]["globe"] = dict(
            trec=stats(P.globe_transient_temperature_C.to_numpy(float)[mg], tg[mg]),
            trec_resampled_control=stats(P.trec_globe_resampled_C.to_numpy(float)[mg], tg[mg]),
            solweig=stats(P.solweig_globe_C.to_numpy(float)[mg], tg[mg]),
            solweig_no_beam_correction=stats(P.solweig_globe_uncorr_C.to_numpy(float)[mg], tg[mg]))
        ta = P.measured_air_temperature_c.to_numpy(float); V = np.maximum(P.measured_wind_ms.to_numpy(float), 0)
        iso = ((tg + 273.15) ** 4 + 1.1e8 * V ** 0.6 / (0.95 * 0.15 ** 0.4) * (tg - ta)) ** 0.25 - 273.15
        out[per]["conventional_iso_globe_vs_cylinder_tmrt"] = dict(
            trec=stats(P.trec_route_mrt_c.to_numpy(float)[mg], iso[mg]),
            solweig=stats(P.solweig_tmrt.to_numpy(float)[mg], iso[mg]))
    od = run / "solweig"
    P.to_csv(od / "solweig_comparison_points.csv", index=False)
    (od / "solweig_comparison_summary.json").write_text(json.dumps(out, indent=1))
    for per in ("day", "night"):
        print(f"\n=== {args.case} {per}")
        for k, v in out[per].items():
            for model, s in v.items():
                if "mbe" in s:
                    print(f"  {k:42s} {model:28s} n={s['n']:5d}  MBE {s['mbe']:+8.2f}  RMSE {s['rmse']:7.2f}  r {s['r']:+.2f}")
    print(f"\nwrote {od}")


if __name__ == "__main__":
    main()
