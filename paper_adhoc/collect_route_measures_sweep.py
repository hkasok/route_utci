#!/usr/bin/env python3
"""Collect the per-departure route measures of run_route_measures_sweep.sh into
run_output/MMC/postprocessing/route_measures_sweep/route_measures_sweep.csv
(one row per departure and route; read by the number audit)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_route_measures_table import ROUTES, load  # noqa: E402

COLS = {"walk_utci": ("utci", "walk_utci_c"), "vs_min": ("utci", "very_strong_stress_min"),
        "worst3": ("utci", "worst_3min_utci_c"), "wet_min": ("jos3", "wettedness_discomfort_min"),
        "rise": ("jos3", "final_tcore_rise_c"), "peak": ("jos3", "peak_tcore_rise_c"),
        "tpk": ("jos3", "t_peak_after_arrival_min"), "heat": ("jos3", "heat_stored_kj"),
        "heat_rad": ("jos3", "heat_stored_radiant_kj"), "D": ("utci", "strong_stress_dose_degmin")}

sweep = Path(sys.argv[1] if len(sys.argv) > 1 else "run_output/MMC/postprocessing/route_measures_sweep")
rows = []
for d in sorted(sweep.glob("dep_*")):
    src = load(d)
    for r in ROUTES:
        rows.append({"dep": f"{d.name[4:6]}:{d.name[6:]}", "route": r,
                     **{k: float(src[s].loc[r, c]) for k, (s, c) in COLS.items()}})
pd.DataFrame(rows).to_csv(sweep / "route_measures_sweep.csv", index=False)
print(f"wrote {sweep / 'route_measures_sweep.csv'} ({len(rows)} rows)")
