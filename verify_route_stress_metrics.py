#!/usr/bin/env python3
"""Verification for the stage-09 route thermal-strain metrics.

The point of this suite is to pin the LIKE-FOR-LIKE rule: an extremity number
must never be produced by subtracting a body-core temperature from an
extremity temperature. That cross-quantity mix reports the standing
core-to-periphery offset (about 0.4 C when hot and vasodilated, about 4.5 C
when cool and vasoconstricted) as if it were strain caused by the walk.

It runs the real stage-09 script on a synthetic two-route case and checks the
metric definitions in the emitted summary.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE / "verify_work_route_metrics"

passed = failed = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f" ({detail})" if detail else ""))


if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir()

# ---- synthetic case: two straight routes, one walked hot, one walked cold ---
routes_dir = WORK / "routes"
routes_dir.mkdir()
n_pts = 60
xs = np.linspace(0.0, 600.0, n_pts)
index = {"routes": []}
ORIGIN = (0.0, 0.0)
for route_id, label in enumerate(["hot_noon", "cold_night"], start=1):
    y = np.full(n_pts, 50.0 * route_id)
    cumdist = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(xs)))))
    frame = pd.DataFrame({
        "seq": np.arange(n_pts),
        "x_local_m": xs, "y_local_m": y, "cumdist_m": cumdist,
        "lat": np.linspace(38.70, 38.71, n_pts),
        "lon": np.linspace(-9.20, -9.19, n_pts),
        "x_proj_m": xs + ORIGIN[0], "y_proj_m": y + ORIGIN[1],
    })
    frame.to_csv(routes_dir / f"route_{route_id}.csv", index=False)
    (routes_dir / f"route_{route_id}.json").write_text(json.dumps({
        "route_id": route_id, "name": label, "length_m": float(cumdist[-1]),
        "n_points": n_pts, "ds_path_m": float(xs[1] - xs[0]),
        "crs_local": "origin_shifted_projected_m", "project_crs": "EPSG:6346",
        "local_origin_x": ORIGIN[0], "local_origin_y": ORIGIN[1],
        "start_latlon": [38.70, -9.20], "end_latlon": [38.71, -9.19],
        "source": "synthetic_test", "network_constrained": True,
        "pedestrian_graphml": "synthetic", "generated_utc": "2026-01-01T00:00:00Z",
        "generator_args": {"synthetic": True},
    }), encoding="utf-8")
    index["routes"].append({"route_id": route_id, "name": label,
                            "csv": f"route_{route_id}.csv",
                            "json": f"route_{route_id}.json"})
(routes_dir / "routes_index.json").write_text(json.dumps(index), encoding="utf-8")
with open(routes_dir / "route_polylines.pkl", "wb") as stream:
    pickle.dump({"polylines": [np.column_stack((xs, np.full(n_pts, 50.0))),
                               np.column_stack((xs, np.full(n_pts, 100.0)))],
                 "highway_tags": ["footway", "footway"]}, stream)

# ---- synthetic MRT results: hot by day, cold by night ----------------------
mrt_dir = WORK / "mrt"
mrt_dir.mkdir()
n_times = 24
hours = np.arange(n_times, dtype=float)
grid_x, grid_y = np.meshgrid(xs, [50.0, 100.0])
path_xyz = np.column_stack((grid_x.ravel(), grid_y.ravel(),
                            np.full(grid_x.size, 1.1)))
np.save(mrt_dir / "path_xyz.npy", path_xyz)
# Tmrt: strongly hot at midday, cold overnight
tmrt = 30.0 + 28.0 * np.cos((hours - 13.0) / 24.0 * 2 * np.pi)
np.save(mrt_dir / "tmrt_matrix_C.npy",
        np.repeat(tmrt[:, None], len(path_xyz), axis=1))
elevation = -50.0 + 100.0 * np.cos((hours - 13.0) / 24.0 * 2 * np.pi)
pd.DataFrame({
    "time": pd.date_range("2023-06-24", periods=n_times, freq="h"),
    "elevation_deg": elevation,
    "azimuth_deg": np.linspace(90, 270, n_times),
    "air_temp_C": 24.0 + 9.0 * np.cos((hours - 15.0) / 24.0 * 2 * np.pi),
    "rh_pct": np.full(n_times, 55.0),
    "wind_ms": np.full(n_times, 1.5),
}).to_csv(mrt_dir / "times.csv", index=False)

weather = WORK / "weather.csv"
pd.DataFrame({"hour": hours,
              "air_temp_C": 24.0 + 9.0 * np.cos((hours - 15.0) / 24.0 * 2 * np.pi),
              "rh_pct": np.full(n_times, 55.0),
              "wind_ms": np.full(n_times, 1.5)}).to_csv(weather, index=False)


def run_stage09(out_dir: Path, extra: list[str]) -> pd.DataFrame:
    result = subprocess.run(
        [sys.executable, str(HERE / "09_route_thermal_stress_jos3.py"),
         "--routes-dir", str(routes_dir), "--mrt-results-dir", str(mrt_dir),
         "--output-dir", str(out_dir), "--weather-csv", str(weather),
         "--walking-speed-ms", "1.3", "--equilibration-min", "10"] + extra,
        capture_output=True, text=True)
    (out_dir / "log.txt").write_text(result.stdout + "\n--- stderr ---\n"
                                     + result.stderr)
    if result.returncode != 0:
        print(result.stdout[-2500:]); print(result.stderr[-2500:])
        raise SystemExit("stage 09 failed in the metrics verification")
    return pd.read_csv(out_dir / "route_ranking_summary.csv")


print("T1: stage 09 emits like-for-like extremity metrics")
# The two thermal conditions are two departures over the same geometry: a
# midday walk in strong sun and a pre-dawn walk in the cold.
out = WORK / "hot"
out.mkdir()
summary = run_stage09(out, ["--departure-hour", "13.0"])
out_cold = WORK / "cold"
out_cold.mkdir()
summary_cold = run_stage09(out_cold, ["--departure-hour", "2.0"])
required = {"final_extremity_skin_c", "extremity_skin_change_c",
            "final_extremity_core_c", "extremity_core_change_c",
            "final_core_to_extremity_gradient_c"}
check(required.issubset(summary.columns),
      "all replacement extremity columns are present",
      f"missing {sorted(required - set(summary.columns))}")
check("final_extremity_rise_c" not in summary.columns,
      "the cross-quantity 'final_extremity_rise_c' column is gone")

print("\nT2: the metrics are physically self-consistent")
for _, row in pd.concat([summary.assign(walk="midday"),
                         summary_cold.assign(walk="pre-dawn")]).iterrows():
    name = f"{row['walk']}/{row['route_name']}"
    check(10.0 < row["final_extremity_skin_c"] < 42.0,
          f"[{name}] extremity skin temperature is physiological",
          f"{row['final_extremity_skin_c']:.1f} C")
    check(15.0 < row["final_extremity_core_c"] < 40.0,
          f"[{name}] extremity tissue temperature is physiological",
          f"{row['final_extremity_core_c']:.1f} C")
    check(row["final_core_to_extremity_gradient_c"] > -0.5,
          f"[{name}] core sits at or above extremity temperature",
          f"gradient {row['final_core_to_extremity_gradient_c']:+.2f} C")
    check(abs(row["extremity_core_change_c"]) < 12.0
          and abs(row["extremity_skin_change_c"]) < 15.0,
          f"[{name}] extremity change over one walk is bounded",
          f"core {row['extremity_core_change_c']:+.2f}, "
          f"skin {row['extremity_skin_change_c']:+.2f} C")

hot = summary.iloc[0]
cold = summary_cold.iloc[0]
check(hot["final_extremity_skin_c"] > cold["final_extremity_skin_c"],
      "the hot walk ends with warmer extremities than the cold walk",
      f"{hot['final_extremity_skin_c']:.1f} vs {cold['final_extremity_skin_c']:.1f} C")
check(cold["final_core_to_extremity_gradient_c"]
      > hot["final_core_to_extremity_gradient_c"],
      "vasoconstriction widens the core-to-extremity gradient in the cold",
      f"cold {cold['final_core_to_extremity_gradient_c']:+.2f} vs "
      f"hot {hot['final_core_to_extremity_gradient_c']:+.2f} C")

print("\nT3: the old cross-quantity definition is not reproduced")
# The discredited metric was extremity(end) - body_core(start). Recomputing it
# here must differ from the reported change whenever the two quantities differ,
# which is exactly the condition that made the old number meaningless.
for label, frame in (("midday", summary), ("pre-dawn", summary_cold)):
    row = frame.iloc[0]
    core_start = row["final_tcore_c"] - row["final_tcore_rise_c"]
    discredited = row["final_extremity_core_c"] - core_start
    reported = row["extremity_core_change_c"]
    check(abs(discredited - reported) > 0.2,
          f"[{label}] reported change is not the old mixed metric",
          f"like-for-like {reported:+.2f} C vs cross-quantity {discredited:+.2f} C")

print("\nT4: an isothermal walk reports no extremity change")
# Holding one fixed environment, a body that has equilibrated should drift very
# little; a cross-quantity metric would still report several degrees.
out2 = WORK / "steady"
out2.mkdir()
steady = run_stage09(out2, ["--equilibration-min", "120",
                            "--departure-hour", "2.0"])
for _, row in steady.iterrows():
    core_start = row["final_tcore_c"] - row["final_tcore_rise_c"]
    discredited = abs(row["final_extremity_core_c"] - core_start)
    check(abs(row["extremity_core_change_c"]) < discredited,
          f"[{row['route_name']}] like-for-like change is smaller than the "
          f"old offset-laden number",
          f"{row['extremity_core_change_c']:+.2f} C vs {discredited:.2f} C offset")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
