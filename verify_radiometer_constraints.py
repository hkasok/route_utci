#!/usr/bin/env python3
"""verify_radiometer_constraints.py -- suite for the measurement-constrained layer.

Tests A-H from the specification are labelled inline, plus the scientific
guards from sections 13, 14, 21 and 30.

Run: python3 verify_radiometer_constraints.py   (exits nonzero on failure)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import material_classification as mcl
import radiometer_constraints as rc

HERE = Path(__file__).resolve().parent
passed = 0
failed = 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition: bool, description: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {description}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {description}" + (f" ({detail})" if detail else ""))


def synthetic(n=10, sw_down=800.0, sw_up=160.0, lw_down=380.0, lw_up=500.0,
              spacing_m=5.0, spacing_s=5.0):
    return pd.DataFrame({
        "case_id": ["test"] * n,
        "route_id": [1] * n,
        "seq": np.arange(n),
        "timestamp_utc": pd.to_datetime(
            ["2023-06-24T14:00:00Z"] * n, format="ISO8601", utc=True),
        "x": np.arange(n) * spacing_m,
        "y": np.zeros(n),
        "SW_down_measured_Wm2": np.full(n, sw_down, dtype=float),
        "SW_up_measured_Wm2": np.full(n, sw_up, dtype=float),
        "LW_down_measured_Wm2": np.full(n, lw_down, dtype=float),
        "LW_up_measured_Wm2": np.full(n, lw_up, dtype=float),
        "distance_along_route_m": np.arange(n) * spacing_m,
        "elapsed_s": np.arange(n) * spacing_s,
        "source_file": ["synthetic"] * n,
    })


print("=" * 72)
print("MEASUREMENT-CONSTRAINED RADIATIVE LAYER VERIFICATION")
print("=" * 72)

# ---------------------------------------------------------------------------
print("\nTEST A: a known synthetic pair gives the expected reflectance")
reflectance, flag = rc.effective_lower_sw_reflectance([800.0], [160.0])
check(abs(float(reflectance[0]) - 0.20) < 1e-12,
      "SW_down=800, SW_up=160 -> effective reflectance 0.20",
      f"{float(reflectance[0]):.4f}")
check(flag[0] == rc.QC_VALID, "and it is flagged valid")
reflectance, _ = rc.effective_lower_sw_reflectance([1000.0, 500.0],
                                                   [450.0, 100.0])
check(np.allclose(reflectance, [0.45, 0.20]),
      "the ratio is elementwise, not aggregated", f"{reflectance}")

print("\nTEST B: near-zero SW_down yields an INVALID reflectance, not a huge one")
reflectance, flag = rc.effective_lower_sw_reflectance([0.5, -2.0, 5.0],
                                                      [0.4, 1.0, 2.0])
check(np.all(np.isnan(reflectance)),
      "reflectance is NaN below the SW_down threshold, never a large number",
      f"{reflectance}")
check(np.all(flag == rc.QC_INVALID), "and every such sample is flagged invalid")
config = rc.QualityControlConfig(minimum_sw_down_for_reflectance_Wm2=200.0)
reflectance, flag = rc.effective_lower_sw_reflectance([150.0], [30.0], config)
check(np.isnan(reflectance[0]),
      "the threshold is configurable (--min-sw-down-for-albedo)")
# Out of range is reported as computed, then flagged -- not clipped.
reflectance, flag = rc.effective_lower_sw_reflectance([400.0], [500.0])
check(abs(float(reflectance[0]) - 1.25) < 1e-12 and flag[0] == rc.QC_INVALID,
      "a reflectance above 1 is returned AS COMPUTED and flagged, not clipped",
      f"{float(reflectance[0]):.2f}")

print("\nTEST C: a measurement outranks the OSM material-derived flux")
check(rc.CONSTRAINT_CONFIDENCE
      > max(mcl.SOURCE_CONFIDENCE[mcl.SOURCE_OSM_DIRECT],
            mcl.SOURCE_CONFIDENCE[mcl.SOURCE_IMAGERY],
            mcl.SOURCE_CONFIDENCE[mcl.SOURCE_OSM_INFERRED],
            mcl.SOURCE_CONFIDENCE[mcl.SOURCE_DEFAULT]),
      "the radiometer constraint ranks above every classification source",
      f"{rc.CONSTRAINT_CONFIDENCE} vs osm_direct "
      f"{mcl.SOURCE_CONFIDENCE[mcl.SOURCE_OSM_DIRECT]}")
frame = rc.build_constraints(synthetic(), sensor_height_m=1.0)
mapped = rc.interpolate_to_route(frame, frame["distance_along_route_m"],
                                 frame["elapsed_s"])
modelled = {channel: np.full(len(mapped), 999.0) for channel in
            rc.CONSTRAINED_CHANNELS}
resolved = rc.resolve_flux_sources(mapped, modelled,
                                   material_source=["osm_direct"] * len(mapped))
check(np.allclose(resolved["SW_up_resolved_Wm2"], 160.0),
      "the measured SW_up is used, NOT the OSM-derived modelled value",
      f"{resolved['SW_up_resolved_Wm2'].iloc[0]}")
check(np.allclose(resolved["LW_up_resolved_Wm2"], 500.0),
      "the measured LW_up is used, NOT a generic-emissivity prediction")
check((resolved["SW_up_source"] == rc.SOURCE_RADIOMETER).all(),
      "and the provenance says so")

print("\nTEST D: with no measurement the existing hierarchy still works")
far = rc.interpolate_to_route(frame, np.array([10000.0, 20000.0]),
                              np.array([0.0, 0.0]))
check(not far["constraint_available"].any(),
      "receptors far from every sample get no constraint")
modelled = {channel: np.array([111.0, 222.0]) for channel in rc.CONSTRAINED_CHANNELS}
resolved = rc.resolve_flux_sources(far, modelled,
                                   material_source=["osm_direct", "default"])
check(np.allclose(resolved["SW_up_resolved_Wm2"], [111.0, 222.0]),
      "the modelled value stands unchanged")
check((resolved["SW_up_source"] == rc.SOURCE_MODELLED_SURFACE).all(),
      "downward channels fall back labelled modeled_surface",
      resolved["SW_up_source"].iloc[0])
check((resolved["LW_down_source"] == rc.SOURCE_MODELLED_SKY).all(),
      "upward channels fall back labelled modeled_sky",
      resolved["LW_down_source"].iloc[0])
check((resolved["radiative_constraint_source"] == "material_hierarchy").all(),
      "and the row records that the material hierarchy supplied the flux")

print("\nTEST E: the sensor-equivalent model uses NO human-body weighting")
stage05 = load_module(HERE / "05_mrt_network_raytrace.py", "stage05_rc")
import inspect

source = inspect.getsource(stage05.sensor_radiometer_quantities)
for forbidden, why in (
        ("projected_area_factor_standing", "standing-person projected area"),
        ("person_sw_absorptivity", "human shortwave absorptivity"),
        ("person_emissivity", "human emissivity"),
        ("f_projected_direct", "a projected-area factor")):
    check(forbidden not in source,
          f"the sensor-equivalent model never uses {why}")
check("svf_planar" in source,
      "it uses the PLANAR sky-view factor -- the correct cosine weighting for "
      "a horizontal sensor")
# And the emitted channels must not be body-absorbed quantities.
prep = load_module(HERE / "prepare_radiometer_constraints.py", "prep_rc")
rfc = load_module(HERE / "radiant_flux_contributions.py", "rfc_rc")
model_columns = set(prep.MODEL_CHANNEL_COLUMNS.values())
body_columns = set(rfc.PRIMARY_COLUMNS + rfc.TOTAL_COLUMNS
                   + rfc.SW_SOURCE_COLUMNS + rfc.LW_SOURCE_COLUMNS)
check(not (model_columns & body_columns),
      "no body-absorbed column is used as a model channel for this comparison")
check(model_columns <= set(rfc.SENSOR_COLUMNS),
      "every model channel is an instrument-equivalent sensor column",
      f"{sorted(model_columns)}")

print("\nTEST F: interpolation refuses to bridge oversized gaps")
sparse = rc.build_constraints(synthetic(n=4, spacing_m=5.0, spacing_s=5.0),
                              sensor_height_m=1.0)
config = rc.InterpolationConfig(max_gap_m=10.0, max_gap_s=1e9)
mapped = rc.interpolate_to_route(sparse, np.array([0.0, 5.0, 200.0]),
                                 np.array([0.0, 5.0, 0.0]), config)
check(bool(mapped["constraint_available"].iloc[0])
      and bool(mapped["constraint_available"].iloc[1]),
      "receptors within the distance gap ARE constrained")
check(not bool(mapped["constraint_available"].iloc[2]),
      "a receptor 185 m from the nearest sample is NOT constrained",
      f"gap {mapped['constraint_gap_m'].iloc[2]:.0f} m")
check(np.isnan(mapped["SW_down_constrained_Wm2"].iloc[2]),
      "and no value is fabricated for it")
time_config = rc.InterpolationConfig(max_gap_m=1e9, max_gap_s=2.0)
mapped = rc.interpolate_to_route(sparse, np.array([0.0, 5.0]),
                                 np.array([0.0, 600.0]), time_config)
check(not bool(mapped["constraint_available"].iloc[1]),
      "the TIME gap threshold is enforced independently of distance",
      f"gap {mapped['constraint_gap_s'].iloc[1]:.0f} s")
for method in rc.VALID_METHODS:
    result = rc.interpolate_to_route(
        sparse, np.array([2.5]), np.array([2.5]),
        rc.InterpolationConfig(method=method))
    check(np.isfinite(result["SW_down_constrained_Wm2"].iloc[0]),
          f"interpolation method '{method}' produces a value")
try:
    rc.interpolate_to_route(sparse, np.array([0.0]), None,
                            rc.InterpolationConfig(method="magic"))
    check(False, "an unknown interpolation method is rejected")
except rc.RadiometerConstraintError:
    check(True, "an unknown interpolation method is rejected")

print("\nTEST G: route variation survives -- no collapse to one value")
varying = synthetic(n=6)
varying["SW_down_measured_Wm2"] = [900.0, 880.0, 120.0, 110.0, 870.0, 900.0]
varying["SW_up_measured_Wm2"] = [180.0, 176.0, 30.0, 28.0, 200.0, 190.0]
varying["LW_up_measured_Wm2"] = [560.0, 555.0, 470.0, 468.0, 545.0, 550.0]
constraints = rc.build_constraints(varying, sensor_height_m=1.0)
mapped = rc.interpolate_to_route(constraints,
                                 constraints["distance_along_route_m"],
                                 constraints["elapsed_s"])
sw = mapped["SW_down_constrained_Wm2"].to_numpy(float)
check(np.nanmax(sw) - np.nanmin(sw) > 700.0,
      "a sun/shade excursion of 780 W/m2 is preserved end to end",
      f"range {np.nanmin(sw):.0f}-{np.nanmax(sw):.0f}")
check(len(np.unique(np.round(sw, 3))) >= 4,
      "distinct measured values stay distinct after association",
      f"{len(np.unique(np.round(sw, 3)))} distinct values")
lw = mapped["LW_up_constrained_Wm2"].to_numpy(float)
check(np.nanmax(lw) - np.nanmin(lw) > 80.0,
      "the longwave excursion survives too",
      f"range {np.nanmin(lw):.0f}-{np.nanmax(lw):.0f}")
# The default method must not smooth the sun/shade step away.
step = rc.interpolate_to_route(constraints, np.array([10.0]), np.array([10.0]))
check(abs(float(step["SW_down_constrained_Wm2"].iloc[0]) - 120.0) < 1e-9,
      "the DEFAULT method assigns the measured shade value, not a blend",
      f"{float(step['SW_down_constrained_Wm2'].iloc[0]):.1f}")
reflectance = constraints["effective_lower_SW_reflectance"].to_numpy(float)
finite = np.isfinite(reflectance)
check(finite.sum() >= 4 and reflectance[finite].std() > 0.0,
      "effective reflectance varies along the route rather than being one number",
      f"{np.round(reflectance[finite], 3)}")

print("\nTEST H: every overridden flux records the radiometer as its source")
frame = rc.build_constraints(synthetic(), sensor_height_m=1.0)
mapped = rc.interpolate_to_route(frame, frame["distance_along_route_m"],
                                 frame["elapsed_s"])
modelled = {channel: np.full(len(mapped), 1.0) for channel in rc.CONSTRAINED_CHANNELS}
resolved = rc.resolve_flux_sources(mapped, modelled)
for channel in rc.CONSTRAINED_CHANNELS:
    if not (resolved[f"{channel}_source"] == rc.SOURCE_RADIOMETER).all():
        check(False, f"{channel} records source=four_component_radiometer")
        break
else:
    check(True, "all four channels record source=four_component_radiometer")
check(set(resolved["radiative_constraint_source"]) == {rc.CONSTRAINT_SOURCE},
      "the row-level constraint source is recorded too")
check("SW_down_measured_Wm2" in resolved.columns
      and "SW_down_modelled_Wm2" in resolved.columns,
      "both the measured and the modelled value are retained, not just the winner")

print("\nS16: material identity survives the measurement override")
resolved = rc.resolve_flux_sources(
    mapped, modelled, material_source=["osm_direct"] * len(mapped))
check((resolved["material_source"] == "osm_direct").all(),
      "the surface is still classified as it was; the measurement constrains "
      "the radiation, it does not un-classify the asphalt")
check("material_source" in resolved.columns
      and "radiative_constraint_source" in resolved.columns,
      "material identity and radiative constraint are SEPARATE fields")

print("\nS7: longwave is kept as radiosity/irradiance, never as emissivity")
constraints = rc.build_constraints(synthetic(), sensor_height_m=1.0)
check(np.allclose(constraints["effective_lower_LW_radiosity_Wm2"],
                  constraints["LW_up_measured_Wm2"]),
      "LW_up is carried through as effective lower-hemisphere radiosity")
check(np.allclose(constraints["effective_upper_LW_irradiance_Wm2"],
                  constraints["LW_down_measured_Wm2"]),
      "LW_down is carried through as effective upper-hemisphere irradiance")
check(not any("emissivity" in column for column in constraints.columns),
      "no emissivity column is produced by default")
diagnostic = rc.optional_emissivity_diagnostic([500.0], [380.0], [45.0])
check(np.isfinite(diagnostic[0]) and 0.0 <= diagnostic[0] <= 1.0,
      "the OPTIONAL diagnostic works when a surface temperature is supplied",
      f"eps ~ {float(diagnostic[0]):.3f}")
check(np.isnan(rc.optional_emissivity_diagnostic([500.0], [380.0], [np.nan])[0]),
      "and returns nothing without one")

print("\nS18: quality control")
config = rc.QualityControlConfig()
night = synthetic(n=5, sw_down=-2.5, sw_up=1.0)
flagged = rc.apply_quality_control(night, config)
check((flagged["qc_flag"] == rc.QC_VALID).all(),
      "a small negative nighttime shortwave is VALID -- it is the documented "
      "pyranometer thermal offset, not a fault")
tilted = synthetic(n=3, sw_down=1450.0)
flagged = rc.apply_quality_control(tilted, config)
check((flagged["qc_flag"] == rc.QC_QUESTIONABLE).all(),
      "shortwave above the physical ceiling is flagged questionable (sensor tilt)")
check(np.allclose(flagged["SW_down_measured_Wm2"], 1450.0),
      "and the value is KEPT, not clipped -- the excursion is the evidence")
impossible = synthetic(n=2, lw_up=5.0)
flagged = rc.apply_quality_control(impossible, config)
check((flagged["qc_flag"] == rc.QC_INVALID).all(),
      "a physically impossible longwave value is invalid")
check(set(flagged.columns) >= {"qc_SW_down", "qc_SW_up", "qc_LW_down",
                               "qc_LW_up", "qc_flag", "confidence"},
      "QC is reported per channel AND overall")
mapped = rc.interpolate_to_route(
    rc.build_constraints(impossible, 1.0), np.array([0.0]), np.array([0.0]))
check(not mapped["constraint_available"].any()
      or np.isnan(mapped["LW_up_constrained_Wm2"].iloc[0]),
      "an invalid sample never becomes a constraint")

print("\nS3/S4: raw observations preserved and georeferenced")
files = sorted(Path("input").glob("lisbon*/measurements/route_1_day_*.csv"))
if files:
    original = pd.read_csv(files[0])
    loaded = rc.load_route_measurements(files[0])
    check(np.allclose(loaded["SW_down_measured_Wm2"], original["SWin"]),
          "the raw SWin values are preserved verbatim")
    check(np.allclose(loaded["LW_up_measured_Wm2"], original["LWout"]),
          "the raw LWout values are preserved verbatim")
    check(np.allclose(loaded["x"], original["x_local_m"]),
          "x comes from the local metric frame the STL and stage 05 already use")
    check(loaded["distance_along_route_m"].is_monotonic_increasing,
          "distance along route is monotonic")
    built = rc.build_constraints(loaded, sensor_height_m=1.0,
                                 ground_z=np.zeros(len(loaded)))
    check(np.allclose(built["sensor_z"], 1.0),
          "sensor_z = ground_z + configured sensor height", "1.0 m")
    check((built["sensor_height_m"] == 1.0).all(),
          "the assumed sensor height travels with every row")
else:
    check(False, "Lisbon measurement files were found")

print("\nS21: constrained is never reported as independent validation")
prep_source = (HERE / "prepare_radiometer_constraints.py").read_text(encoding="utf-8")
check("--constraint-routes" in prep_source and "--validation-routes" in prep_source,
      "the constraint/validation split is exposed for cross-validation")
check("cannot constrain a model and independently" in prep_source,
      "naming a case as both constraint and validation is refused")
check("MEASUREMENT-CONSTRAINED (not independent)" in prep_source,
      "constrained results are labelled as such in the report")

print("\nS30: final scientific check -- like-for-like channel pairing")
pairs = {
    "SW_down": ("measured_swin_wm2", "sensor_shortwave_down_Wm2"),
    "SW_up": ("measured_swout_wm2", "sensor_shortwave_up_Wm2"),
    "LW_down": ("measured_lwin_wm2", "sensor_longwave_down_Wm2"),
    "LW_up": ("measured_lwout_wm2", "sensor_longwave_up_Wm2"),
}
for channel, (measured_column, model_column) in pairs.items():
    check(prep.MODEL_CHANNEL_COLUMNS[channel] == model_column,
          f"measured {channel} is compared against modelled {channel}",
          f"{measured_column} vs {model_column}")
check(all(column not in prep.MODEL_CHANNEL_COLUMNS.values()
          for column in prep.FORBIDDEN_BODY_COLUMNS),
      "no absorbed human-body quantity appears anywhere in the comparison")

print("\n" + "=" * 72)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
