#!/usr/bin/env python3
"""paper_solweig_run.py -- run SOLWEIG on a TREC-Route scene with TREC-Route's forcing.

Run with the SOLWEIG environment:
    /media/harshin/data_drive/solweig/.venv/bin/python paper_solweig_run.py --case lisbon1

Inputs are the rasters from paper_solweig_inputs.py and the 10-min forcing
TREC-Route used (run_output/<case>/mrt_facet_out/times.csv): air temperature,
humidity, wind, global, direct-normal and diffuse irradiance, and the sun
position at each instant -- so both models see the same sun and the same
direct/diffuse split. SOLWEIG's own schemes (ground and wall temperatures,
land-cover albedo and emissivity, anisotropic sky) are used at their defaults.

Two runs:
  standing   standard SOLWEIG person (cylinder; absorptivities 0.70 / 0.97);
             writes tmrt, kdown, kup, ldown, lup, shadow -- the radiometer
             channels and the conventional Tmrt.
  globe      SOLWEIG's sitting posture, whose six directions are equally
             weighted (1/6), with the globe's absorptivity 0.95 and emittance
             0.957; writes tmrt and shadow. Its Tmrt is the radiant
             temperature of the absorbed load on a six-directional sphere; the
             beam projection (0.20 cos h + 1/6 sin h vs 0.25 for a sphere) is
             corrected afterwards from the shadow output (paper_solweig_compare.py).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import solweig

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--runs", nargs="+", default=["standing", "globe"])
    args = ap.parse_args()
    base = ROOT / "run_output" / args.case / "solweig"
    inp = base / "inputs"
    meta = json.loads((inp / "raster_meta.json").read_text())
    case = json.loads((ROOT / "input" / args.case / "case.json").read_text())

    t = pd.read_csv(ROOT / "run_output" / args.case / "mrt_facet_out" / "times.csv")
    stamps = pd.to_datetime(t.time)
    utc_offset = stamps.iloc[0].utcoffset().total_seconds() / 3600
    dt_min = float((stamps.iloc[1] - stamps.iloc[0]).total_seconds() / 60)
    altmax = float(t.elevation_deg.max())
    weather = []
    for i, r in t.iterrows():
        up = r.elevation_deg > 0
        weather.append(solweig.Weather(
            datetime=stamps.iloc[i].to_pydatetime().replace(tzinfo=None),
            ta=float(r.air_temp_C), rh=float(r.rh_pct), ws=float(r.wind_ms),
            global_rad=float(max(r.GHI_Wm2, 0.0)), timestep_minutes=dt_min,
            measured_direct_rad=float(max(r.DNI_Wm2, 0.0)) if up else 0.0,
            measured_diffuse_rad=float(max(r.DHI_Wm2, 0.0)) if up else 0.0,
            precomputed_sun_altitude=float(r.elevation_deg),
            precomputed_sun_azimuth=float(r.azimuth_deg),
            precomputed_altmax=altmax))
    loc = solweig.Location(latitude=case["location"]["latitude"],
                           longitude=case["location"]["longitude"], utc_offset=utc_offset)

    surface = solweig.SurfaceData.prepare(
        dsm=np.load(inp / "dsm.npy"), dem=np.load(inp / "dem.npy"),
        cdsm=np.load(inp / "cdsm.npy"), tdsm=np.load(inp / "tdsm.npy"),
        land_cover=np.load(inp / "land_cover.npy").astype(np.int32),
        pixel_size=meta["pixel_size"], working_dir=str(base / "cache"),
        dsm_relative=False, cdsm_relative=True, tdsm_relative=True)

    configs = {
        "standing": (solweig.HumanParams(posture="standing", abs_k=0.70, abs_l=0.97),
                     ["tmrt", "kdown", "kup", "ldown", "lup", "shadow"]),
        "globe": (solweig.HumanParams(posture="sitting", abs_k=0.95, abs_l=0.957),
                  ["tmrt", "shadow"]),
    }
    for name in args.runs:
        human, outputs = configs[name]
        od = base / f"run_{name}"
        od.mkdir(parents=True, exist_ok=True)
        print(f"== {args.case} / {name}: {len(weather)} steps of {dt_min:g} min ({datetime.now():%H:%M:%S})", flush=True)
        summary = solweig.calculate(surface=surface, weather=weather, location=loc,
                                    output_dir=str(od), human=human, outputs=outputs)
        (od / "summary.txt").write_text(str(summary.report()))
    print("done")


if __name__ == "__main__":
    main()
