#!/usr/bin/env python3
"""Verification for the JOS-3 clothing ensembles and automatic selection.

Covers the contract stage 09 relies on: segment order, whole-body anchoring,
monotonic response to climate / time of day / weather, config overrides, input
validation, and that dressed walkers actually behave differently from the
previous unclothed default.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import clothing_profiles as cp

passed = failed = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f" ({detail})" if detail else ""))


print("T1: segment contract")
from pythermalcomfort.models import JOS3
model = JOS3(height=1.72, weight=74.0, age=30, sex="male", fat=15.0)
check(list(model.body_names) == list(cp.JOS3_SEGMENTS),
      "module segment order matches this JOS-3 build")
check(len(cp.SEGMENT_BSA_FRACTION) == 17
      and abs(cp.SEGMENT_BSA_FRACTION.sum() - 1.0) < 1e-4,
      "reference BSA fractions are complete and normalised",
      f"sum={cp.SEGMENT_BSA_FRACTION.sum():.5f}")
reference = model.bsa / model.bsa.sum()
check(np.allclose(reference, cp.SEGMENT_BSA_FRACTION, atol=1e-4),
      "hardcoded BSA fractions match the live model")
child = JOS3(height=1.28, weight=26.0, age=8, sex="male", fat=18.0)
check(np.allclose(child.bsa / child.bsa.sum(), cp.SEGMENT_BSA_FRACTION, atol=1e-4),
      "BSA fractions are subject-independent, so anchoring is exact for all")


print("\nT2: ensembles are anchored to their whole-body value")
for name, ensemble in cp.ENSEMBLES.items():
    segments = ensemble.segment_clo()
    total = cp.whole_body_clo(segments, model.bsa)
    ok = abs(total - ensemble.approximate_total_clo) < 0.01
    check(ok, f"'{name}' reaches its declared whole-body clo",
          f"{total:.3f} vs {ensemble.approximate_total_clo:.2f}")
    check(np.all(segments >= 0) and np.isfinite(segments).all(),
          f"'{name}' has finite non-negative segment values")
check(cp.whole_body_clo(cp.ENSEMBLES["nude"].segment_clo(), model.bsa) == 0.0,
      "'nude' remains exactly zero (reproduces pre-clothing runs)")
check(cp.ENSEMBLES["summer_light"].segment_clo()[
          cp.JOS3_SEGMENTS.index("left_hand")] == 0.0
      and cp.ENSEMBLES["winter_heavy"].segment_clo()[
          cp.JOS3_SEGMENTS.index("left_hand")] > 0.5,
      "hands are bare in summer and gloved in winter")


print("\nT3: automatic selection responds to weather, time of day, climate")
warm, _ = cp.select_ensemble(31.0, is_daytime=True, wind_ms=1.0)
mild, _ = cp.select_ensemble(19.0, is_daytime=True, wind_ms=1.0)
cold, _ = cp.select_ensemble(0.0, is_daytime=True, wind_ms=1.0)
check(warm.key == "hot_minimal" and mild.key == "mild_longsleeve"
      and cold.key == "winter_heavy",
      "temperature drives the ensemble",
      f"{warm.key} / {mild.key} / {cold.key}")

totals = []
for temperature in range(-5, 40):
    ensemble, _ = cp.select_ensemble(float(temperature), is_daytime=True, wind_ms=1.0)
    totals.append(ensemble.approximate_total_clo)
check(all(b <= a for a, b in zip(totals, totals[1:])),
      "clothing decreases monotonically as it gets warmer")

day, day_terms = cp.select_ensemble(22.4, is_daytime=True, wind_ms=1.0)
night, night_terms = cp.select_ensemble(22.4, is_daytime=False, wind_ms=1.0)
check(night.approximate_total_clo > day.approximate_total_clo,
      "the same temperature is dressed warmer after dark",
      f"day {day.key} -> night {night.key}")
calm, _ = cp.select_ensemble(17.4, is_daytime=True, wind_ms=0.5)
windy, _ = cp.select_ensemble(17.4, is_daytime=True, wind_ms=12.0)
check(windy.approximate_total_clo > calm.approximate_total_clo,
      "strong wind adds a layer", f"{calm.key} -> {windy.key}")
tropic, _ = cp.select_ensemble(26.2, is_daytime=True, wind_ms=1.0, climate="tropical")
polar, _ = cp.select_ensemble(26.2, is_daytime=True, wind_ms=1.0, climate="cold")
check(tropic.approximate_total_clo < polar.approximate_total_clo,
      "hot-climate residents dress lighter at the same temperature",
      f"tropical {tropic.key} vs cold-climate {polar.key}")
_e, terms = cp.select_ensemble(20.0, is_daytime=False, wind_ms=9.0, climate="tropical")
check(abs(terms["dressing_temperature_c"]
          - (20.0 - 1.5 - min(6.0 * 0.4, 4.0) + 2.0)) < 1e-9,
      "every adjustment term is reported and adds up",
      f"{terms['dressing_temperature_c']:.2f} C")


print("\nT4: explicit requests and validation")
segments, prov = cp.resolve("summer_light", air_temp_c=5.0, is_daytime=True, wind_ms=1.0)
check(prov["selection"] == "named_ensemble" and prov["ensemble"] == "summer_light",
      "a named ensemble overrides the conditions")
segments, prov = cp.resolve("0.75", air_temp_c=20.0, is_daytime=True, wind_ms=1.0)
check(prov["selection"] == "uniform_clo" and np.allclose(segments, 0.75),
      "a bare number is read as uniform clo on every segment")
for bad in ("tuxedo", "-1", "nan"):
    try:
        cp.resolve(bad, air_temp_c=20.0, is_daytime=True, wind_ms=1.0)
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, f"invalid clothing request '{bad}' is rejected")
for bad_kwargs in ({"air_temp_c": float("nan")}, {"wind_ms": -2.0},
                   {"climate": "martian"}):
    kwargs = {"air_temp_c": 20.0, "is_daytime": True, "wind_ms": 1.0}
    kwargs.update(bad_kwargs)
    try:
        cp.select_ensemble(**kwargs)
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, f"invalid selection input {list(bad_kwargs)[0]} is rejected")


print("\nT5: case-supplied configuration overrides")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "clothing.json"
    path.write_text(json.dumps({
        "night_offset_c": 0.0,
        "ensembles": {"uniform_hi_vis": {
            "label": "Hi-vis work vest over shirt",
            "parts": {"torso": 1.0, "arm": 0.4, "thigh": 0.6, "leg": 0.5,
                      "foot": 0.4},
            "approximate_total_clo": 0.70}},
        "thresholds_c": [[20.0, "uniform_hi_vis"], [-100.0, "winter_heavy"]],
    }), encoding="utf-8")
    config = cp.load_config(path)
    ensemble, terms = cp.select_ensemble(25.0, is_daytime=False, wind_ms=0.0,
                                         config=config)
    check(ensemble.key == "uniform_hi_vis", "a case can add its own ensemble")
    check(abs(cp.whole_body_clo(ensemble.segment_clo(), model.bsa) - 0.70) < 0.01,
          "a case-supplied ensemble is anchored the same way")
    check(terms["night_offset_c"] == 0.0, "a case can disable the night offset")

bad_config = {"thresholds_c": [[10.0, "summer_light"], [20.0, "cool_layer"]]}
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "bad.json"
    path.write_text(json.dumps(bad_config), encoding="utf-8")
    try:
        cp.load_config(path)
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, "out-of-order thresholds are rejected")


print("\nT6: clothing actually changes JOS-3 physiology")
def walk(clo_array, ta, tmrt, wind, rh, minutes=30):
    m = JOS3(height=1.72, weight=74.0, age=30, sex="male", fat=15.0)
    m.par = 2.5
    m.clo = clo_array
    m.tdb, m.tr, m.rh, m.v = ta, tmrt, rh, wind
    m.simulate(times=10, dtime=60, output=False)
    weights = m.bsa / m.bsa.sum()
    idx = [k for k, n in enumerate(m.body_names) if "hand" in n or "foot" in n]
    core0 = float(np.sum(m.t_core * weights))
    hand0 = float(np.mean(m.t_core[idx]))
    m.simulate(times=minutes, dtime=60, output=False)
    return (float(np.sum(m.t_core * weights)) - core0,
            float(np.mean(m.t_core[idx])) - hand0,
            float(np.mean(m.t_core[idx])))

nude_core, nude_ext, nude_abs = walk(cp.ENSEMBLES["nude"].segment_clo(),
                                     21.5, 19.0, 2.2, 63.0)
dressed_core, dressed_ext, dressed_abs = walk(
    cp.ENSEMBLES["mild_longsleeve"].segment_clo(), 21.5, 19.0, 2.2, 63.0)
check(dressed_core > nude_core,
      "on a cool night the dressed walker retains more core heat",
      f"nude {nude_core:+.3f} C vs dressed {dressed_core:+.3f} C")
check(dressed_ext > nude_ext,
      "clothing reduces extremity cooling",
      f"nude {nude_ext:+.2f} C vs dressed {dressed_ext:+.2f} C")
check(15.0 < dressed_abs < 37.0,
      "dressed extremity temperature stays physiologically plausible",
      f"{dressed_abs:.1f} C")
hot_core, _e, _a = walk(cp.ENSEMBLES["summer_light"].segment_clo(),
                        32.0, 55.0, 1.0, 55.0)
check(hot_core > 0.0, "a hot sunny walk still raises core temperature",
      f"{hot_core:+.3f} C")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
