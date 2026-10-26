#!/usr/bin/env python3
"""Sensitivity test: shortwave forcing at the radiometer's unshaded level.

The paper's forcing follows the cart pyranometer's unshaded envelope but is
capped at a clean sky (Linke turbidity 2.0), so it sits below what the sensor
read in the open. This writes a copy of one case's solar components with DNI,
DHI and GHI scaled by one factor: the median, over the samples the forcing fit
itself calls unshaded, of measured SWin over the written forcing GHI at that
time. Cloud fraction (and so the sky longwave) is unchanged.

  python3 paper_adhoc/scale_solar_forcing.py --case-dir VARIANT_INPUT/lisbonN

Only the components file inside --case-dir is rewritten; point it at a copy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weather_from_sensors import (clear_sky_series, load_case_measurements,  # noqa: E402
                                  robust_unshaded_ratio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", type=Path, required=True)
    args = ap.parse_args()
    case_dir = args.case_dir.resolve()
    if case_dir.parent == (ROOT / "input").resolve():
        raise SystemExit("refusing to rewrite a published input case; use a copy")
    manifest = json.loads((case_dir / "case.json").read_text())
    loc = manifest["location"]
    date = manifest["simulation_defaults"]["date"]
    diag = json.loads((case_dir / "weather" / "weather_from_sensors_provenance.json")
                      .read_text())["solar"]["envelope_diagnostics"]
    path = case_dir / "config" / "weather_from_sensors_solar_components.csv"
    comp = pd.read_csv(path)

    meas = load_case_measurements(case_dir).dropna(subset=["SWin"])
    _dni, _dhi, ghi_clim, _el = clear_sky_series(
        meas["hour"].to_numpy(float), latitude=loc["latitude"],
        longitude=loc["longitude"], timezone_name=loc["timezone"], date=date)
    keep = (ghi_clim > 80.0) & (meas["SWin"].to_numpy(float) >= 0.0)  # fit defaults
    hours = meas["hour"].to_numpy(float)[keep]
    swin = meas["SWin"].to_numpy(float)[keep]
    robust = robust_unshaded_ratio(hours, swin / ghi_clim[keep],
                                   window_samples=int(diag.get("robust_window_samples", 31)),
                                   shade_tolerance=float(diag.get("shade_tolerance", 0.12)))
    open_sky = ~np.asarray(robust["shaded"], bool)
    forcing = np.interp(hours, comp["hour"], comp["GHI_Wm2"])
    ok = open_sky & (forcing > 50.0)
    factor = float(np.median(swin[ok] / forcing[ok]))
    for col in ("DNI_Wm2", "DHI_Wm2", "GHI_Wm2"):
        comp[col] = comp[col] * factor
    comp.to_csv(path, index=False)
    record = {"factor": factor, "n_unshaded_samples": int(ok.sum()),
              "forcing_ghi_median_before_wm2": float(np.median(forcing[ok])),
              "measured_unshaded_swin_median_wm2": float(np.median(swin[ok]))}
    (case_dir / "config" / "solar_forcing_scale.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print(f"{case_dir.name}: forcing x {factor:.3f} ({ok.sum()} unshaded samples; "
          f"median forcing {record['forcing_ghi_median_before_wm2']:.0f} -> measured "
          f"{record['measured_unshaded_swin_median_wm2']:.0f} W/m2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
