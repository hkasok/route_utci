#!/usr/bin/env python3
"""Verification for sensor-derived weather and time-dependent solar forcing.

Pins the properties this forcing has to have:

* the shortwave envelope recovers the ATMOSPHERE from a shaded mobile series,
  and is not contaminated by the shaded samples;
* clearness is inverted through the pipeline's own cloud adjustment, so a
  round trip reproduces the requested irradiance;
* the reconstructed hours restore a real night instead of the straight chord
  the pipeline would otherwise interpolate, and are labelled as reconstructed;
* an amplitude the measurements cannot determine is rejected, not shipped;
* the products are readable by the stages that consume them.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import weather_from_sensors as wfs
from radiation_forcing import apply_cloud_adjustment

HERE = Path(__file__).resolve().parent
passed = failed = 0

LATITUDE, LONGITUDE, TZ, DATE = 38.72, -9.15, "Europe/Lisbon", "2023-06-24"


def check(condition: bool, label: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f" ({detail})" if detail else ""))


print("T1: humidity conversions round trip")
temperature = np.array([5.0, 18.0, 30.0, 40.0])
rh = np.array([30.0, 55.0, 80.0, 95.0])
dew = wfs.dewpoint_from_relative_humidity(temperature, rh)
check(np.allclose(wfs.relative_humidity_from_dewpoint(temperature, dew), rh,
                  atol=1e-6),
      "RH -> dew point -> RH is lossless")
check(np.all(dew <= temperature + 1e-9),
      "dew point never exceeds air temperature")


print("\nT2: the shortwave envelope recovers the atmosphere through shade")
# Synthetic walk: a known clear-sky day, sampled every 5 s, where the sensor
# spends most of its time shaded. The envelope must recover the ATMOSPHERE,
# i.e. the applied clearness, not the shaded mean.
hours = np.arange(9.0, 17.0, 5.0 / 3600.0)
_dni, _dhi, ghi_clear, elevation = wfs.clear_sky_series(
    hours, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ, date=DATE)
true_clearness = 0.75
rng = np.random.default_rng(20260823)
# 70% of samples heavily shaded, 30% in the open
shade = np.where(rng.random(len(hours)) < 0.7, rng.uniform(0.05, 0.5, len(hours)), 1.0)
measured = pd.DataFrame({
    "hour": hours, "SWin": ghi_clear * true_clearness * shade,
    "AirTemp": 25.0, "HRel": 50.0, "WS": 1.0, "DewP": 14.0,
    "timestamp_local_refined": pd.Timestamp(DATE)})
envelope = wfs.solar_clearness_envelope(
    measured, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ,
    date=DATE, window_min=10.0, quantile=0.95, smooth_windows=3,
    minimum_clear_ghi=80.0, maximum_clearness=1.5)
recovered = float(np.median(envelope["clearness"]))
check(abs(recovered - true_clearness) < 0.05,
      "envelope recovers the applied clearness from a mostly shaded walk",
      f"{recovered:.3f} vs {true_clearness:.2f}")
shaded_mean = float(np.mean(measured["SWin"] / ghi_clear))
check(abs(recovered - true_clearness) < abs(shaded_mean - true_clearness),
      "envelope is much closer than the shaded mean would be",
      f"shaded mean would give {shaded_mean:.3f}")

varying = np.clip(0.9 - 0.4 * np.sin((hours - 9.0) / 8.0 * np.pi), 0.2, 1.0)
measured_varying = measured.assign(SWin=ghi_clear * varying * shade)
envelope_varying = wfs.solar_clearness_envelope(
    measured_varying, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ,
    date=DATE, window_min=10.0, quantile=0.95, smooth_windows=3,
    minimum_clear_ghi=80.0, maximum_clearness=1.5)
truth = np.interp(envelope_varying["hours"], hours, varying)
error = float(np.max(np.abs(envelope_varying["clearness"] - truth)))
check(error < 0.1, "a TIME-VARYING atmosphere is tracked, not averaged away",
      f"max error {error:.3f}")


print("\nT3: clearness inverts through the pipeline's own cloud adjustment")
grid = np.arange(0.0, 24.0, 1.0 / 6.0)
dni_clear, dhi_clear, ghi_clear_grid, elevation_grid = wfs.clear_sky_series(
    grid, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ, date=DATE)
for requested in (0.95, 1.0):
    clearness = np.full(len(grid), requested)
    cloud = wfs.clearness_to_cloud_fraction(
        clearness, dni_clear, dhi_clear, elevation_grid)
    _d, _f, ghi = apply_cloud_adjustment(dni_clear, dhi_clear, elevation_grid, cloud)
    day = ghi_clear_grid > 100.0
    achieved = float(np.median(ghi[day] / ghi_clear_grid[day]))
    check(abs(achieved - requested) < 0.03,
          f"round trip reproduces reachable clearness {requested:.2f}",
          f"achieved {achieved:.3f}")

# The established scalar cloud adjustment has a limited dynamic range. A
# clearness below it must be REPORTED, never silently shipped as if achieved.
for unreachable in (0.35, 0.6):
    clearness = np.full(len(grid), unreachable)
    cloud = wfs.clearness_to_cloud_fraction(
        clearness, dni_clear, dhi_clear, elevation_grid)
    report = wfs.clearness_inversion_report(
        clearness, cloud, dni_clear, dhi_clear, elevation_grid)
    check(not report["representable"] and report["unreachable_samples"] > 0,
          f"clearness {unreachable:.2f} is flagged as unreachable",
          f"max error {report['maximum_absolute_error']:.3f}, floor "
          f"{report['minimum_representable_clearness']:.2f}")
check(np.all((wfs.clearness_to_cloud_fraction(
          np.full(len(grid), 0.4), dni_clear, dhi_clear, elevation_grid) >= 0)
      & (wfs.clearness_to_cloud_fraction(
          np.full(len(grid), 0.4), dni_clear, dhi_clear, elevation_grid) <= 1)),
      "recovered cloud fraction stays inside [0, 1]")


print("\nT4: the reconstructed day restores a night that cools")
walk = pd.DataFrame({
    "hour": np.r_[np.arange(15.0, 15.7, 1 / 360), np.arange(21.7, 22.2, 1 / 360)],
    "AirTemp": np.r_[np.full(252, 29.0), np.full(180, 22.0)],
    "HRel": 55.0, "WS": 1.2, "DewP": 17.0, "SWin": 500.0,
    "timestamp_local_refined": pd.Timestamp(DATE)})
weather = wfs.build_meteorology(
    walk, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ, date=DATE,
    bin_min=10.0, reconstruct=True)
check(len(weather) == 144, "a full 10-minute day is produced",
      f"{len(weather)} rows")
check(set(weather["source"]) == {"measured", "reconstructed"},
      "every row is labelled measured or reconstructed")
dawn = float(np.interp(6.0, weather["hour"], weather["air_temp_C"]))
chord = float(np.interp(6.0, walk["hour"], walk["AirTemp"], period=24.0))
check(dawn < 24.0 and dawn < chord - 3.0,
      "pre-dawn temperature is a real minimum, not the straight chord",
      f"reconstructed {dawn:.1f} C vs chord {chord:.1f} C")
check(weather["air_temp_C"].idxmin() is not None
      and 3.0 <= float(weather.loc[weather["air_temp_C"].idxmin(), "hour"]) <= 8.0,
      "the daily minimum lands near sunrise",
      f"{float(weather.loc[weather['air_temp_C'].idxmin(), 'hour']):.1f} h")
check(12.0 <= float(weather.loc[weather["air_temp_C"].idxmax(), "hour"]) <= 18.0,
      "the daily maximum lands in the afternoon",
      f"{float(weather.loc[weather['air_temp_C'].idxmax(), 'hour']):.1f} h")
measured_rows = weather[weather["source"] == "measured"]
check(abs(float(measured_rows["air_temp_C"].max()) - 29.0) < 0.01
      and abs(float(measured_rows["air_temp_C"].min()) - 22.0) < 0.01,
      "measured values are preserved exactly, never smoothed by the fit")
check(weather["rh_pct"].between(1.0, 100.0).all()
      and weather["wind_ms"].ge(0.0).all(),
      "reconstructed humidity and wind stay physical",
      f"RH {weather['rh_pct'].min():.0f}..{weather['rh_pct'].max():.0f}%")


print("\nT5: an undeterminable amplitude is rejected, not shipped")
flat = walk.copy()
flat["AirTemp"] = np.r_[np.full(252, 29.0), np.full(180, 28.7)]
flat_weather = wfs.build_meteorology(
    flat, latitude=LATITUDE, longitude=LONGITUDE, timezone_name=TZ, date=DATE,
    bin_min=10.0, reconstruct=True, default_range_c=10.0)
fit = flat_weather.attrs["diurnal_fit"]
check(fit.get("rejected_least_squares_fit") is not None,
      "a near-zero fitted range is rejected")
check(abs(fit["diurnal_range_c"] - 10.0) < 1e-6,
      "the documented default range is used instead",
      f"{fit['diurnal_range_c']:.1f} K")
check("fell outside the plausible band" in fit["amplitude_source"],
      "the substitution is explained in the fit record")
good_fit = weather.attrs["diurnal_fit"]
check(good_fit.get("rejected_least_squares_fit") is None
      and good_fit["amplitude_source"].startswith("least squares"),
      "a well-determined amplitude is kept")


print("\nT6: products are consumable by the stages that read them")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp) / "synthetic"
    (case / "weather").mkdir(parents=True)
    (case / "config").mkdir(parents=True)
    (case / "measurements").mkdir(parents=True)
    stamps = pd.Timestamp(f"{DATE} 00:00:00") + pd.to_timedelta(walk["hour"], unit="h")
    walk.assign(timestamp_local_refined=stamps.dt.strftime("%Y-%m-%dT%H:%M:%S"),
                LWin=380.0).to_csv(
        case / wfs.MEASUREMENT_RELATIVE, index=False)
    (case / "case.json").write_text(json.dumps({
        "case_id": "synthetic", "files": {},
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE,
                     "timezone": TZ},
        "simulation_defaults": {"date": DATE}}), encoding="utf-8")
    provenance = wfs.build_case_inputs(
        case, name="sensor_forcing", bin_min=10.0, window_min=10.0,
        quantile=0.95, smooth_windows=3, minimum_clear_ghi=80.0,
        maximum_clearness=1.0, reconstruct=True, activate=True,
        longwave_from_sensor=False)

    from weather_provider import WeatherProvider
    provider = WeatherProvider(csv_path=case / "weather" / "sensor_forcing.csv")
    check(set(provider.columns_from_csv) == {"air_temp_C", "rh_pct", "wind_ms"},
          "WeatherProvider reads every required column from the generated file")
    check(abs(float(provider.air_temp_c(15.2)) - 29.0) < 0.5,
          "the provider returns the measured afternoon temperature",
          f"{float(provider.air_temp_c(15.2)):.1f} C")

    from radiation_forcing import resolve_radiation_forcing
    model_times = pd.date_range(f"{DATE} 00:00", periods=144, freq="10min",
                                tz=TZ)
    forcing = resolve_radiation_forcing(
        case / "config" / "sensor_forcing.json", model_times,
        np.zeros(144), np.zeros(144), np.zeros(144), np.zeros(144), 0.0)
    check(len(forcing.ghi_wm2) == 144 and np.isfinite(forcing.ghi_wm2).all(),
          "the components file resolves through the components_csv mode")
    check(np.all(forcing.ghi_wm2 >= 0) and forcing.ghi_wm2.max() > 100,
          "resolved irradiance is non-negative with a real daytime peak",
          f"max {forcing.ghi_wm2.max():.0f} W/m2")
    manifest = json.loads((case / "case.json").read_text())
    check(manifest["files"]["weather_csv"] == "weather/sensor_forcing.csv"
          and manifest["files"]["radiation_forcing_config"]
          == "config/sensor_forcing.json",
          "activation repoints case.json at both generated inputs")
    check((case / "weather" / "sensor_forcing_provenance.json").is_file()
          and "clearness" in provenance["solar"]["method"],
          "provenance records how the forcing was built")
    diagnostics = provenance["solar"].get("envelope_diagnostics", {})
    check(diagnostics.get("estimator") in {"robust_median", "window_quantile"},
          "provenance names which solar estimator ran",
          str(diagnostics.get("estimator")))
    if diagnostics.get("estimator") == "robust_median":
        check("shaded_sample_fraction" in diagnostics,
              "and how much of the walk it judged shaded",
              f"{diagnostics.get('shaded_sample_fraction', float('nan')):.1%}")
        check("residual_instrument_bias_fraction" in diagnostics,
              "and how far the unshaded sensor sat above a clean sky",
              f"{diagnostics.get('residual_instrument_bias_fraction', float('nan')):+.1%}")
    check("baseline_linke_turbidity" in provenance["solar"],
          "provenance records whether the clear-sky baseline was refitted",
          str(provenance["solar"].get("baseline_linke_turbidity")))

print("\nT8: inlet wind is inverted from the measured local wind")


def synthetic_wind_field(directory: Path, amplification):
    """Minimal step-3 field whose amplification is a known function of x."""
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(0.0, 200.0, 2.0)
    y = np.arange(0.0, 40.0, 2.0)
    reference = 2.0
    amp = np.asarray([amplification(value) for value in x], dtype=float)
    speed = np.tile(amp, (len(y), 1)) * reference
    np.save(directory / "x_coordinates.npy", x)
    np.save(directory / "y_coordinates.npy", y)
    np.save(directory / "velocity_u.npy", speed)
    np.save(directory / "velocity_v.npy", np.zeros_like(speed))
    np.save(directory / "velocity_speed.npy", speed)
    np.save(directory / "ground_z.npy", np.zeros_like(speed))
    np.save(directory / "pedestrian_z.npy", np.full_like(speed, 1.1))
    np.save(directory / "fluid_mask.npy", np.ones_like(speed, dtype=bool))
    (directory / "potential_flow_metadata.json").write_text(json.dumps({
        "reference_wind_speed_ms": reference, "pedestrian_height_m": 1.1,
        "wind_direction_from_deg": 270.0}), encoding="utf-8")
    return x


with tempfile.TemporaryDirectory() as tmp:
    field_dir = Path(tmp) / "pedestrian_wind"
    # Sheltered at one end, accelerated at the other.
    xs = synthetic_wind_field(field_dir, lambda value: 0.2 + 1.1 * value / 200.0)

    hours = np.arange(15.0, 16.0, 1 / 720)
    true_inlet = 2.5 + 1.5 * np.sin((hours - 15.0) * 2 * np.pi)
    walk_x = np.interp(np.linspace(0, 1, len(hours)), [0, 1], [5.0, 195.0])
    amp_true = 0.2 + 1.1 * walk_x / 200.0
    samples = pd.DataFrame({
        "hour": hours, "WS": true_inlet * amp_true,
        "x_local_m": walk_x, "y_local_m": 20.0})
    estimate = wfs.estimate_inlet_wind(
        samples, field_dir, bin_min=10.0, quantile=0.5,
        minimum_amplification=0.8, minimum_speed_ms=0.0,
        maximum_inlet_ms=25.0)
    recovered = np.interp(estimate["hours"], hours, true_inlet)
    error = float(np.max(np.abs(estimate["inlet_ms"] - recovered)))
    check(error < 0.25, "inverted inlet recovers the true boundary speed",
          f"max error {error:.3f} m/s")
    # The inversion has to CORRECT for local amplification, not echo the
    # measurement. Whether the inlet ends up above or below the measured mean
    # depends on where the walk spent its time, so the meaningful test is that
    # the inverted series tracks the true boundary speed better than the raw
    # measurement does.
    binned_measured = np.interp(estimate["hours"], hours, samples["WS"].to_numpy(float))
    inverted_error = float(np.mean(np.abs(estimate["inlet_ms"] - recovered)))
    raw_error = float(np.mean(np.abs(binned_measured - recovered)))
    check(inverted_error < 0.5 * raw_error,
          "the inversion corrects local amplification rather than echoing the "
          "measurement",
          f"inverted MAE {inverted_error:.3f} vs raw measured MAE {raw_error:.3f} m/s")
    check(estimate["amplification_threshold_used"] == 0.8
          and estimate["n_open_ground_samples"] < estimate["n_samples_total"],
          "only open-ground samples are inverted",
          f"{estimate['n_open_ground_samples']}/{estimate['n_samples_total']}")

    # A walk entirely in shelter must relax the threshold rather than fail.
    sheltered_field = Path(tmp) / "sheltered"
    synthetic_wind_field(sheltered_field, lambda value: 0.35)
    sheltered = samples.assign(WS=true_inlet * 0.35)
    relaxed = wfs.estimate_inlet_wind(
        sheltered, sheltered_field, bin_min=10.0, quantile=0.5,
        minimum_amplification=0.8, minimum_speed_ms=0.0, maximum_inlet_ms=25.0)
    check(relaxed["amplification_threshold_used"] < 0.8,
          "a fully sheltered walk relaxes the threshold instead of failing",
          f"used {relaxed['amplification_threshold_used']:.2f}")


print("\nT9: the inlet series reaches the surface-energy stage")
from weather_provider import WeatherProvider
with tempfile.TemporaryDirectory() as tmp:
    both = Path(tmp) / "both.csv"
    pd.DataFrame({"hour": [0.0, 12.0, 23.0], "air_temp_C": [20.0, 30.0, 21.0],
                  "rh_pct": [60.0, 40.0, 62.0], "wind_ms": [1.0, 1.5, 1.1],
                  "wind_inlet_ms": [2.0, 3.6, 2.2]}).to_csv(both, index=False)
    provider = WeatherProvider(csv_path=both)
    check(provider.has_inlet_wind(), "an inlet column is detected")
    check(abs(float(provider.inlet_wind_ms(12.0)) - 3.6) < 1e-9
          and abs(float(provider.wind_ms(12.0)) - 1.5) < 1e-9,
          "inlet and pedestrian wind stay distinct quantities")

    only = Path(tmp) / "only.csv"
    pd.DataFrame({"hour": [0.0, 12.0, 23.0], "air_temp_C": [20.0, 30.0, 21.0],
                  "rh_pct": [60.0, 40.0, 62.0],
                  "wind_ms": [1.0, 1.5, 1.1]}).to_csv(only, index=False)
    legacy = WeatherProvider(csv_path=only)
    check(not legacy.has_inlet_wind()
          and abs(float(legacy.inlet_wind_ms(12.0))
                  - float(legacy.wind_ms(12.0))) < 1e-12,
          "without an inlet column the historical single series is reproduced")

    bad = Path(tmp) / "bad.csv"
    pd.DataFrame({"hour": [0.0, 12.0], "air_temp_C": [20.0, 30.0],
                  "rh_pct": [60.0, 40.0], "wind_ms": [1.0, 1.5],
                  "wind_inlet_ms": [-1.0, 3.0]}).to_csv(bad, index=False)
    try:
        WeatherProvider(csv_path=bad)
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, "a negative inlet wind is rejected at the source")


print("\nT7: a case without a measurement campaign fails cleanly")
# This runs from a UI button where ANY case can be selected, so a case with no
# campaign must produce an actionable message and a non-zero exit, not a
# traceback.
import subprocess
import sys

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "input"
    (root / "no_campaign").mkdir(parents=True)
    (root / "no_campaign" / "case.json").write_text(json.dumps({
        "case_id": "no_campaign", "files": {},
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE, "timezone": TZ},
        "simulation_defaults": {"date": DATE}}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(HERE / "weather_from_sensors.py"),
         "--input-root", str(root), "--case", "no_campaign"],
        capture_output=True, text=True)
    check(result.returncode == 1, "exits non-zero", f"code {result.returncode}")
    check("Traceback" not in result.stderr and "Traceback" not in result.stdout,
          "no traceback is shown to the user")
    check("no mobile measurements" in result.stdout
          and "Nothing to do" in result.stdout,
          "the message says what is missing and why")
    check("no_campaign" in result.stdout,
          "the message names the case that was skipped")
    check((root / "no_campaign" / "case.json").read_text().find('"files": {}') >= 0,
          "a case that could not be processed is left untouched")

print("\nT-SOLAR: robust unshaded-envelope estimator")
# A synthetic clear day with deep shade dips punched into it. The estimator
# must recover the CLEAR level, not an average dragged down by the shade.
hours = np.linspace(10.0, 16.0, 400)
clear_ratio = np.full_like(hours, 0.95)
shaded = np.zeros_like(hours, dtype=bool)
for start in (40, 120, 200, 300):
    shaded[start:start + 25] = True          # short dips, quick recovery
ratio = np.where(shaded, 0.18, clear_ratio)
robust = wfs.robust_unshaded_ratio(hours, ratio, window_samples=31,
                                   shade_tolerance=0.12)
check(abs(float(np.median(robust["envelope"])) - 0.95) < 0.02,
      "short shade dips do not drag the envelope down",
      f"envelope median {float(np.median(robust['envelope'])):.3f} vs true 0.95")
overlap = (robust["shaded"] == shaded).mean()
check(overlap > 0.95, "the shade mask recovers the injected shade",
      f"{overlap:.1%} agreement")
check(float(np.nanmedian(robust["unshaded_ratio"])) > 0.9,
      "the unshaded subset keeps the clear-sky level")
# A single upward spike must not set the forcing either.
spiky = clear_ratio.copy()
spiky[200] = 3.0
robust = wfs.robust_unshaded_ratio(hours, spiky, window_samples=31,
                                   shade_tolerance=0.12)
check(float(np.max(robust["envelope"])) < 1.05,
      "a single upward spike cannot set the envelope -- the median is robust "
      "in BOTH directions", f"max {float(np.max(robust['envelope'])):.3f}")
# A LONG shaded stretch is genuinely ambiguous; the estimator must not invent
# a clear level for it, but must recover afterwards.
long_shade = clear_ratio.copy()
long_shade[100:250] = 0.2
robust = wfs.robust_unshaded_ratio(hours, long_shade, window_samples=31,
                                   shade_tolerance=0.12)
check(abs(float(np.median(robust["envelope"][300:])) - 0.95) < 0.02,
      "the envelope recovers the clear level after a long shaded stretch")
check(wfs.robust_unshaded_ratio(np.empty(0), np.empty(0), window_samples=31,
                                shade_tolerance=0.12)["n_unshaded"] == 0,
      "an empty series is handled without error")
try:
    wfs.robust_unshaded_ratio([1.0, 2.0], [1.0], window_samples=5,
                              shade_tolerance=0.1)
    check(False, "mismatched input lengths are rejected")
except ValueError:
    check(True, "mismatched input lengths are rejected")

print("\nT-PROFILE: urban canopy wind profile (free-stream boundary condition)")
import wind_profile as wpf

# A low-rise district: the sensor sits below the displacement height, which is
# exactly the case a plain log law cannot handle.
morph = wpf.canopy_morphology(building_height_m=19.0, plan_area_fraction=0.14,
                              frontal_area_index=0.08)
check(morph.displacement_height_m > 1.0,
      "the displacement height is well above a 1 m sensor -- a plain log law "
      "would take the log of a negative number here",
      f"d = {morph.displacement_height_m:.1f} m")
check(0.0 < morph.roughness_length_m < morph.building_height_m,
      "Macdonald roughness is positive and below canopy height",
      f"z0 = {morph.roughness_length_m:.2f} m")

# The profile must increase monotonically with height and be continuous across
# the roof-level match between the exponential and logarithmic branches.
heights = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 18.9, 19.0, 19.1, 30.0, 38.0])
speeds = np.array([float(wpf.speed_at_height(1.0, 1.0, z, morph))
                   for z in heights])
check(np.all(np.diff(speeds) > 0),
      "wind increases monotonically with height through both branches")
# The two branches are matched in VALUE at roof level but not in slope -- a
# kink there is expected and standard for this profile, so continuity must be
# tested as a shrinking one-sided gap, not by straddling the join.
gaps = []
for eps in (1e-2, 1e-3, 1e-4):
    below = float(wpf.speed_at_height(1.0, 1.0, 19.0 - eps, morph))
    above = float(wpf.speed_at_height(1.0, 1.0, 19.0 + eps, morph))
    gaps.append(abs(above - below))
# A gap that shrinks in exact proportion to the step is the signature of a
# continuous value with a discontinuous derivative. A genuine value jump would
# leave the gap constant as the step shrinks.
ratio = gaps[0] / max(gaps[-1], 1e-30)
check(90.0 < ratio < 110.0,
      "the branches match in VALUE at roof level: the one-sided gap shrinks in "
      "exact proportion to the step, which is a slope kink and not a jump",
      f"gap {gaps[0]:.2e} -> {gaps[-1]:.2e} for a 100x smaller step "
      f"(ratio {ratio:.0f})")
check(abs(float(wpf._log_factor(19.0, morph)) - 1.0) < 1e-12,
      "the logarithmic branch is exactly 1.0 at roof level, so the match is "
      "exact rather than approximate")
check(abs(float(wpf.speed_at_height(1.0, 1.0, 1.0, morph)) - 1.0) < 1e-12,
      "moving a speed to its own height is the identity")
# Round trip: up then back down must return the original.
lifted = float(wpf.speed_at_height(1.37, 1.0, 38.0, morph))
returned = float(wpf.speed_at_height(lifted, 38.0, 1.0, morph))
check(abs(returned - 1.37) < 1e-9,
      "the profile is invertible: lifting then lowering returns the original",
      f"{returned:.6f}")

# The magnitude is the whole point: the free stream must be substantially
# larger than the sheltered in-canopy measurement.
free = float(wpf.free_stream_speed(1.37, 1.0, morph))
check(free > 2.5 * 1.37,
      "the free stream is far above the sheltered cart wind, which is why "
      "feeding the sheltered value to a+b*U under-predicts convection",
      f"{free:.2f} m/s from 1.37 m/s (x{free / 1.37:.2f})")
check(12.0 < 5.7 + 3.8 * free < 30.0,
      "the resulting McAdams coefficient lands in the range urban schemes "
      "report, without any tuning", f"h = {5.7 + 3.8 * free:.1f} W/m2K")

# The pedestrian-height approach wind is a DIFFERENT quantity and must stay
# close to the measurement rather than being lifted.
approach = float(wpf.approach_speed_at_pedestrian_height(1.37, 1.0, morph))
check(abs(approach - 1.37) < 0.1,
      "the pedestrian-height approach wind (the flow-solve boundary condition) "
      "stays near the measurement -- it must NOT be lifted to the free stream",
      f"{approach:.2f} m/s")
check(approach < free,
      "and it is distinct from the free stream; conflating the two is the "
      "error this module exists to prevent")

# Denser canopy -> more sheltering -> a larger lift for the same measurement.
dense = wpf.canopy_morphology(19.0, 0.45, 0.35)
check(float(wpf.free_stream_speed(1.37, 1.0, dense))
      > float(wpf.free_stream_speed(1.37, 1.0, morph)),
      "a denser canopy implies a larger free stream behind the same sheltered "
      "reading")
check(dense.displacement_height_m > morph.displacement_height_m,
      "and a higher displacement height")

# INVERTIBILITY FLOOR. The canopy-average exponential is the ill-conditioned
# direction when going UP from a street reading: in a dense canopy it puts the
# sensor at a few percent of roof level, so inverting multiplies the
# measurement by an order of magnitude.
dense_canopy = wpf.canopy_morphology(17.0, 0.35, 0.36)
raw = np.exp(dense_canopy.attenuation * (1.0 / 17.0 - 1.0))
check(raw < wpf.MINIMUM_STREET_TO_ROOF_RATIO,
      "a dense canopy's RAW average profile puts a 1 m sensor implausibly low",
      f"u(1m)/u(H) = {raw:.3f} before flooring")
check(wpf.street_ratio_is_floored(1.0, dense_canopy),
      "and the floor is reported as binding there")
check(abs(wpf.in_canopy_ratio(1.0, dense_canopy)
          - wpf.MINIMUM_STREET_TO_ROOF_RATIO) < 1e-12,
      "the floored ratio is used instead of the raw one")
dense_free = float(wpf.free_stream_speed(1.04, 1.0, dense_canopy))
check(dense_free < 12.0,
      "so the dense-canopy free stream stays physically plausible instead of "
      "reaching tens of m/s", f"{dense_free:.2f} m/s from 1.04 m/s")
check(5.0 < 5.7 + 3.8 * dense_free < 45.0,
      "and the implied convection coefficient stays in the urban range",
      f"h = {5.7 + 3.8 * dense_free:.1f} W/m2K")
# The floor must NOT bind for an open canopy -- it is a bound, not a default.
check(not wpf.street_ratio_is_floored(1.0, morph),
      "the floor does not bind for an open low-rise canopy, so the physical "
      "profile is used unchanged there")
check(abs(wpf.in_canopy_ratio(1.0, morph)
          - np.exp(morph.attenuation * (1.0 / 19.0 - 1.0))) < 1e-12,
      "and that case keeps the raw Cionco exponential exactly")
# Building HEIGHT must be measured above its own base, not as absolute
# elevation -- Lisbon is hilly and the two differ by a factor of four.
profile_source = (HERE / "wind_profile.py").read_text(encoding="utf-8")
check("z.max() - z.min()" in profile_source,
      "canopy height is a building's height above its OWN base, not terrain "
      "elevation plus height")
check("plausible" in profile_source and "2.0 <= height <= 120.0" in profile_source,
      "an implausible derived canopy height is rejected rather than used")

try:
    wpf.friction_velocity(3.0, 2.0, morph)
    check(False, "a reference height inside the canopy is refused")
except wpf.WindProfileError:
    check(True, "a reference height inside the canopy is refused")
try:
    wpf.speed_at_height(-1.0, 1.0, 10.0, morph)
    check(False, "a negative wind speed is rejected")
except wpf.WindProfileError:
    check(True, "a negative wind speed is rejected")

print("\nT-CEILING: the clean-sky ceiling is physical, not arbitrary")
clean = wfs.clean_sky_ceiling(np.array([13.0]), latitude=38.7, longitude=-9.2,
                              timezone_name="Europe/Lisbon", date="2023-06-24",
                              linke_turbidity=2.0)
climat = wfs.clear_sky_series(np.array([13.0]), latitude=38.7, longitude=-9.2,
                              timezone_name="Europe/Lisbon", date="2023-06-24")[2]
check(clean[0] > climat[0],
      "a cleaner atmosphere gives a HIGHER clear-sky irradiance than the "
      "monthly climatology", f"{clean[0]:.0f} vs {climat[0]:.0f} W/m2")
dirty = wfs.clean_sky_ceiling(np.array([13.0]), latitude=38.7, longitude=-9.2,
                              timezone_name="Europe/Lisbon", date="2023-06-24",
                              linke_turbidity=5.0)
check(dirty[0] < climat[0], "and a hazier one gives less",
      f"{dirty[0]:.0f} W/m2")
components = wfs.clear_sky_at_turbidity(
    np.array([13.0]), latitude=38.7, longitude=-9.2,
    timezone_name="Europe/Lisbon", date="2023-06-24", linke_turbidity=2.0)
check(len(components) == 4 and components[2][0] > 0,
      "the pinned-turbidity helper returns full DNI/DHI/GHI/elevation")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
