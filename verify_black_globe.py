#!/usr/bin/env python3
"""verify_black_globe.py -- verification suite for the black-globe emulator.

Covers black_globe.py (convection laws, steady solve, transient integration,
ISO 7726 inversion, campaign wind identification), its wiring into the
contribution record, and the invariant that the globe never contaminates the
body-absorbed product.

Run: python3 verify_black_globe.py   (exits nonzero on failure)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import black_globe as bg

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


print("=" * 68)
print("BLACK-GLOBE EMULATION VERIFICATION")
print("=" * 68)

# ---------------------------------------------------------------------------
print("\nT1: instrument specification matches the Campbell BLACKGLOBE-L")
spec = bg.CAMPBELL_BLACKGLOBE_L
check(abs(spec.diameter_m - 0.152) < 1e-9,
      "diameter is the 15.2 cm (6 in) copper sphere", f"{spec.diameter_m} m")
check(abs(spec.emissivity - 0.957) < 1e-9,
      "emittance is the published near-normal 0.957", f"{spec.emissivity}")
check(0.9 <= spec.sw_absorptivity <= 1.0,
      "black paint shortwave absorptivity is near unity")
check(250.0 < spec.time_constant_s() < 350.0,
      "time constant is ISO-7726-consistent (20-30 min to equilibrium)",
      f"tau={spec.time_constant_s():.0f} s")
check(spec.surface_area_m2() > 0 and spec.heat_capacity_J_K() > 0,
      "derived area and heat capacity are positive")

# ---------------------------------------------------------------------------
print("\nT2: sphere geometry -- the reason globe MRT is not body MRT")
check(bg.sphere_projected_area_factor() == 0.25,
      "sphere projected-area factor is exactly 0.25 at every solar altitude")
stage05 = load_module(HERE / "05_mrt_network_raytrace.py", "stage05_globe")
for altitude in (0.0, 15.0, 45.0, 90.0):
    standing = float(stage05.projected_area_factor_standing(altitude))
    check(abs(standing - 0.25) > 1e-3,
          f"standing body differs from a sphere at {altitude:.0f} deg altitude",
          f"f_p standing={standing:.3f} vs sphere=0.250")
check(float(stage05.projected_area_factor_standing(90.0)) < 0.25
      < float(stage05.projected_area_factor_standing(0.0)),
      "a sphere absorbs MORE beam than a standing body at high sun and LESS "
      "at low sun -- the difference reverses sign, so no fixed factor converts "
      "one to the other")

# ---------------------------------------------------------------------------
print("\nT3: convection coefficients follow ISO 7726")
# h must reproduce the standard's Tmrt form exactly: setting the balance's
# convective term equal to the standard's bracket is the definition.
wind, diameter, emissivity = 2.0, 0.15, 0.95
h = float(bg.forced_convection_coefficient(wind, diameter))
iso_bracket = 1.1e8 * wind ** 0.6 / (emissivity * diameter ** 0.4)
check(abs(h / (emissivity * bg.SIGMA) - iso_bracket) / iso_bracket < 1e-12,
      "forced coefficient reproduces the ISO 7726 bracket exactly",
      f"h={h:.3f} W/m2/K")
check(abs(float(bg.forced_convection_coefficient(0.0, diameter))) < 1e-12,
      "forced convection vanishes in still air")
doubled = float(bg.forced_convection_coefficient(4.0, diameter))
check(abs(doubled / h - 2.0 ** 0.6) < 1e-12,
      "forced coefficient scales as V^0.6")
smaller = float(bg.forced_convection_coefficient(wind, 0.04))
check(smaller > h, "a smaller globe is more strongly coupled to the air",
      f"40 mm h={smaller:.1f} vs 150 mm h={h:.1f}")
mixed = float(bg.convection_coefficient(40.0, 20.0, 0.0, diameter))
check(mixed > 0.0, "still air falls back to natural convection, not zero",
      f"h_nat={mixed:.2f} W/m2/K")
check(float(bg.convection_coefficient(40.0, 20.0, 5.0, diameter))
      > float(bg.convection_coefficient(40.0, 20.0, 0.5, diameter)),
      "mixed convection increases with wind")

# ---------------------------------------------------------------------------
print("\nT4: steady globe solve inverts its own energy balance")
rng = np.random.default_rng(20260824)
flux = rng.uniform(300.0, 900.0, 400)
air = rng.uniform(-5.0, 40.0, 400)
winds = rng.uniform(0.0, 6.0, 400)
globe = bg.steady_globe_temperature_C(flux, air, winds, spec)
h_solved = bg.convection_coefficient(globe, air, winds, spec.diameter_m)
residual = (flux - spec.emissivity * bg.SIGMA * (globe + 273.15) ** 4
            - h_solved * (globe - air))
check(np.max(np.abs(residual)) < 1e-6,
      "solved temperature satisfies R = eps*sigma*Tg^4 + h*(Tg-Ta)",
      f"max residual {np.max(np.abs(residual)):.2e} W/m2")
check(np.all(np.isfinite(globe)), "steady solve is finite everywhere")

# A globe in still air must sit at its radiative equilibrium; wind must pull it
# toward air temperature and never past it.
equilibrium = bg.radiative_equilibrium_temperature_C(flux, spec.emissivity)
calm = bg.steady_globe_temperature_C(flux, air, np.zeros_like(air), spec)
windy = bg.steady_globe_temperature_C(flux, air, np.full_like(air, 8.0), spec)
hot = equilibrium > air
check(np.all(np.abs(calm[hot] - air[hot]) >= np.abs(windy[hot] - air[hot]) - 1e-9),
      "wind moves the globe toward air temperature")
check(np.all(windy[hot] >= air[hot] - 1e-6) and np.all(windy[hot] <= equilibrium[hot] + 1e-6),
      "the globe always sits between air temperature and radiative equilibrium")

# ---------------------------------------------------------------------------
print("\nT5: transient integration -- the part that makes a WALK comparable")
seconds = np.arange(0.0, 3600.0, 1.0)
constant_flux = np.full_like(seconds, 700.0)
constant_air = np.full_like(seconds, 25.0)
constant_wind = np.full_like(seconds, 1.5)
trace, spinup = bg.integrate_globe_temperature_C(
    seconds, constant_flux, constant_air, constant_wind, spec)
target = float(bg.steady_globe_temperature_C(
    np.array([700.0]), np.array([25.0]), np.array([1.5]), spec)[0])
check(abs(trace[-1] - target) < 1e-6,
      "under constant forcing the transient converges to the steady solution",
      f"{trace[-1]:.4f} vs {target:.4f} C")
check(np.all(np.isfinite(trace)), "transient trace is finite")

# Started away from equilibrium, the approach must be exponential with the
# advertised time constant.
trace2, _ = bg.integrate_globe_temperature_C(
    seconds, constant_flux, constant_air, constant_wind, spec,
    initial_temp_C=target - 20.0)
tau = spec.time_constant_s(1.5, target + 273.15)
after_one_tau = np.interp(tau, seconds, trace2)
recovered = (after_one_tau - (target - 20.0)) / 20.0
check(abs(recovered - (1.0 - np.exp(-1.0))) < 0.06,
      "one time constant recovers ~63% of the initial offset",
      f"{recovered:.3f} vs {1 - np.exp(-1.0):.3f}")

# The headline behaviour: a step in radiation must be strongly damped over the
# seconds a pedestrian spends crossing a shadow.
step_seconds = np.arange(0.0, 120.0, 1.0)
step_flux = np.where(step_seconds < 60.0, 400.0, 900.0)
step_trace, _ = bg.integrate_globe_temperature_C(
    step_seconds, step_flux, np.full_like(step_seconds, 25.0),
    np.full_like(step_seconds, 1.5), spec)
steady_step = bg.steady_globe_temperature_C(
    step_flux, np.full_like(step_seconds, 25.0),
    np.full_like(step_seconds, 1.5), spec)
transient_swing = step_trace.max() - step_trace.min()
steady_swing = steady_step.max() - steady_step.min()
check(transient_swing < 0.35 * steady_swing,
      "a 60 s sun/shade step is heavily damped versus the steady response",
      f"transient {transient_swing:.2f} K vs steady {steady_swing:.2f} K")
fast = bg.GLOBE_PRESETS["small_globe_40mm"]
fast_trace, _ = bg.integrate_globe_temperature_C(
    step_seconds, step_flux, np.full_like(step_seconds, 25.0),
    np.full_like(step_seconds, 1.5), fast)
check((fast_trace.max() - fast_trace.min()) > transient_swing,
      "a 40 mm globe follows the same step far more closely than a 150 mm one")

# Robustness the field data actually needs.
irregular = np.array([0.0, 0.5, 0.5, 3.0, 400.0, 401.0])
rough, _ = bg.integrate_globe_temperature_C(
    irregular, np.full(6, 700.0), np.full(6, 25.0), np.full(6, 1.5), spec)
check(np.all(np.isfinite(rough)),
      "non-uniform sampling with a duplicate timestamp integrates cleanly")
big_step, _ = bg.integrate_globe_temperature_C(
    np.array([0.0, 5000.0]), np.array([700.0, 700.0]),
    np.array([25.0, 25.0]), np.array([1.5, 1.5]), spec)
check(abs(big_step[-1] - target) < 1e-6,
      "a step far longer than the time constant stays stable (backward Euler)")
check(bg.integrate_globe_temperature_C(
    np.empty(0), np.empty(0), np.empty(0), np.empty(0), spec)[0].size == 0,
      "empty input is handled without error")
try:
    bg.integrate_globe_temperature_C(np.array([1.0, 0.0]), np.array([700.0, 700.0]),
                                     np.array([25.0, 25.0]), np.array([1.5, 1.5]),
                                     spec)
    check(False, "backwards time is rejected")
except ValueError:
    check(True, "backwards time is rejected")
_, flags = bg.integrate_globe_temperature_C(
    seconds, constant_flux, constant_air, constant_wind, spec)
check(flags[0] and not flags[-1],
      "spin-up flag marks the start of the walk and clears later",
      f"{int(flags.sum())} of {flags.size} samples flagged")

# ---------------------------------------------------------------------------
print("\nT5b: a MOVING globe is ventilated by the RELATIVE air speed")
# Head-on, the walker's motion adds; downwind at the same speed it subtracts.
head_on = float(bg.relative_air_speed(2.0, 0.0, -1.0, 0.0))
tail = float(bg.relative_air_speed(2.0, 0.0, 1.0, 0.0))
across = float(bg.relative_air_speed(2.0, 0.0, 0.0, 1.0))
check(abs(head_on - 3.0) < 1e-12, "walking into the wind adds to ventilation",
      f"{head_on:.2f} m/s")
check(abs(tail - 1.0) < 1e-12, "walking downwind subtracts from it",
      f"{tail:.2f} m/s")
check(abs(across - np.hypot(2.0, 1.0)) < 1e-12,
      "crossing the wind combines in quadrature", f"{across:.2f} m/s")
check(head_on > across > tail,
      "direction matters -- a scalar sum would lose this entirely")
check(float(bg.relative_air_speed(0.0, 0.0, 1.0, 0.0)) == 1.0,
      "in dead calm a walker still ventilates the globe at walking pace")

# Velocity recovered from route geometry and a clock.
straight_t = np.arange(0.0, 100.0, 1.0)
u_walk, v_walk = bg.receptor_velocity(1.25 * straight_t,
                                      np.zeros_like(straight_t), straight_t)
check(np.allclose(u_walk, 1.25, atol=1e-6) and np.allclose(v_walk, 0.0, atol=1e-6),
      "constant-pace straight walk recovers its own speed", f"{u_walk.mean():.3f} m/s")
jumpy_x = 1.25 * straight_t.copy()
jumpy_x[50] += 60.0          # a single GPS spike
u_jump, v_jump = bg.receptor_velocity(jumpy_x, np.zeros_like(straight_t), straight_t)
check(np.max(np.hypot(u_jump, v_jump)) <= 4.0 + 1e-9,
      "a coordinate spike cannot inject an impossible ventilation speed",
      f"max {np.max(np.hypot(u_jump, v_jump)):.2f} m/s")
check(abs(np.median(u_jump) - 1.25) < 0.05,
      "median filtering keeps the real pace despite the spike")
turning_t = np.arange(0.0, 60.0, 1.0)
u_turn, v_turn = bg.receptor_velocity(
    20.0 * np.cos(turning_t / 20.0), 20.0 * np.sin(turning_t / 20.0), turning_t)
check(np.std(np.arctan2(v_turn, u_turn)) > 0.1,
      "a turning route produces a genuinely varying heading")
check(bg.receptor_velocity([0.0], [0.0], [0.0])[0].size == 1,
      "a single-sample route is handled without error")

# Choosing the basis: vector when the direction is real, quadrature when it is
# a single global assumption. Getting this wrong is not academic -- against an
# assumed direction the vector form can REDUCE ventilation below the ambient
# wind, purely because the guess happened to point along the route.
uniform_u = np.full(50, 1.3)
uniform_v = np.zeros(50)
check(not bg.air_direction_is_resolved(uniform_u, uniform_v),
      "one globally assumed wind direction is NOT treated as resolved")
turning_air_u = 1.3 * np.cos(np.linspace(0.0, 2.0, 50))
turning_air_v = 1.3 * np.sin(np.linspace(0.0, 2.0, 50))
check(bg.air_direction_is_resolved(turning_air_u, turning_air_v),
      "a direction that turns along the route IS treated as resolved")
check(not bg.air_direction_is_resolved(np.zeros(10), np.zeros(10)),
      "a dead-calm field is not treated as resolved")
check(not bg.air_direction_is_resolved(np.array([np.nan, 1.0]), np.zeros(2)),
      "a non-finite field is not treated as resolved")

walker_speed = np.full(50, 1.0)
assumed = bg.ventilation_speed(np.full(50, 1.3), walker_speed,
                               air_u_ms=uniform_u, air_v_ms=uniform_v,
                               receptor_u_ms=walker_speed,
                               receptor_v_ms=np.zeros(50))
check(np.allclose(assumed, np.hypot(1.3, 1.0)),
      "with an assumed direction it falls back to quadrature, NOT to the "
      "vector difference that would have given 0.3 m/s here",
      f"{assumed[0]:.2f} m/s")
check(np.all(assumed >= 1.3),
      "the fallback can never ventilate the globe LESS than the ambient wind")
resolved = bg.ventilation_speed(np.full(50, 1.3), walker_speed,
                                air_u_ms=turning_air_u, air_v_ms=turning_air_v,
                                receptor_u_ms=walker_speed,
                                receptor_v_ms=np.zeros(50))
check(not np.allclose(resolved, np.hypot(1.3, 1.0)),
      "with a resolved direction the exact vector difference is used instead")
check(np.allclose(bg.ventilation_speed(np.full(50, 1.3), walker_speed),
                  np.hypot(1.3, 1.0)),
      "omitting the vectors entirely also gives the quadrature form")

# The headline: including self-motion cools the globe toward air temperature.
walk_t = np.arange(0.0, 2400.0, 1.0)
walk_flux = np.full_like(walk_t, 780.0)
walk_air = np.full_like(walk_t, 30.0)
ambient = np.full_like(walk_t, 1.4)
still_globe, _ = bg.integrate_globe_temperature_C(
    walk_t, walk_flux, walk_air, ambient, spec)
moving = bg.relative_air_speed(np.full_like(walk_t, 1.4), np.zeros_like(walk_t),
                               np.full_like(walk_t, -1.0), np.zeros_like(walk_t))
moving_globe, _ = bg.integrate_globe_temperature_C(
    walk_t, walk_flux, walk_air, moving, spec)
check(moving_globe[-1] < still_globe[-1] - 1.0,
      "a globe carried through the air settles COOLER than a static one in the "
      "same radiation field",
      f"{moving_globe[-1]:.2f} vs {still_globe[-1]:.2f} C")
check(moving_globe[-1] > walk_air[-1],
      "but it still stays above air temperature under a positive radiation load")

print("\nT6: ISO 7726 inversion round-trips the steady solve")
mrt = bg.globe_mrt_iso7726_C(globe, air, winds, spec)
check(np.max(np.abs(mrt - equilibrium)) < 1e-6,
      "inverting a steady globe recovers the radiative equilibrium it came from",
      f"max error {np.max(np.abs(mrt - equilibrium)):.2e} K")
check(np.all(bg.globe_mrt_iso7726_C(np.array([45.0]), np.array([25.0]),
                                    np.array([3.0]), spec)
             > bg.globe_mrt_iso7726_C(np.array([45.0]), np.array([25.0]),
                                      np.array([0.5]), spec)),
      "the same globe reading in stronger wind implies a HOTTER radiant field")

# ---------------------------------------------------------------------------
print("\nT7: campaign wind convention is identifiable from a synthetic series")
true_wind = 1.35
synthetic_air = rng.uniform(24.0, 32.0, 500)
synthetic_globe = synthetic_air + rng.uniform(2.0, 14.0, 500)
synthetic_mrt = bg.globe_mrt_iso7726_C(
    synthetic_globe, synthetic_air, np.full(500, true_wind), spec)
report = bg.fit_campaign_wind_convention(synthetic_globe, synthetic_air,
                                         synthetic_mrt, spec)
check(abs(report["implied_wind_ms"] - true_wind) < 0.12,
      "the constant wind used in an ISO inversion is recovered",
      f"{report['implied_wind_ms']:.2f} vs {true_wind:.2f} m/s")
check(report["implied_coefficient_iqr_fraction"] < 0.02,
      "a genuinely constant convention shows a near-zero implied-coefficient spread",
      f"IQR/median={report['implied_coefficient_iqr_fraction']:.4f}")
varying = rng.uniform(0.4, 4.0, 500)
varying_mrt = bg.globe_mrt_iso7726_C(synthetic_globe, synthetic_air, varying, spec)
varying_report = bg.fit_campaign_wind_convention(
    synthetic_globe, synthetic_air, varying_mrt, spec)
check(varying_report["implied_coefficient_iqr_fraction"]
      > 10.0 * report["implied_coefficient_iqr_fraction"],
      "an instantaneous-wind convention is distinguishable from a constant one",
      f"IQR/median={varying_report['implied_coefficient_iqr_fraction']:.3f}")

# ---------------------------------------------------------------------------
print("\nT8: preset resolution and validation")
check(bg.resolve_globe_spec(None) is bg.DEFAULT_GLOBE,
      "no preset falls back to the Campbell default")
override = bg.resolve_globe_spec("campbell_blackglobe_l", emissivity=0.90)
check(abs(override.emissivity - 0.90) < 1e-12
      and abs(override.diameter_m - spec.diameter_m) < 1e-12,
      "a single-field override leaves the other fields alone")
rescaled = bg.resolve_globe_spec("campbell_blackglobe_l", diameter_m=0.04)
check(abs(rescaled.areal_heat_capacity_J_m2K
          - spec.areal_heat_capacity_J_m2K) < 1e-9,
      "resizing keeps the PER-AREA heat capacity, which is what 'same "
      "construction, different size' means for a shell (rho*c*t has no D in it)")
check(rescaled.time_constant_s() < spec.time_constant_s(),
      "the time constant then shortens on its own, because a smaller sphere "
      "is more strongly coupled to the air",
      f"tau {rescaled.time_constant_s():.0f} s vs {spec.time_constant_s():.0f} s")
for bad in ({"diameter_m": -1.0}, {"emissivity": 0.0}, {"emissivity": 1.4},
            {"areal_heat_capacity_J_m2K": 0.0}):
    try:
        bg.resolve_globe_spec("campbell_blackglobe_l", **bad)
        check(False, f"invalid globe field rejected: {bad}")
    except ValueError:
        check(True, f"invalid globe field rejected: {bad}")
try:
    bg.resolve_globe_spec("no_such_globe")
    check(False, "unknown preset is rejected")
except ValueError:
    check(True, "unknown preset is rejected")

# ---------------------------------------------------------------------------
print("\nT9: the globe never contaminates the body-absorbed product")
rfc = load_module(HERE / "radiant_flux_contributions.py", "rfc_globe")
body = set(rfc.PRIMARY_COLUMNS + rfc.TOTAL_COLUMNS
           + rfc.SW_SOURCE_COLUMNS + rfc.LW_SOURCE_COLUMNS)
check(not (set(rfc.GLOBE_COLUMNS) & body),
      "no globe column is part of the body-absorbed record")
check(set(rfc.GLOBE_COLUMNS) <= set(rfc.INSTRUMENT_COLUMNS),
      "globe columns are registered as instrument-equivalent")

n_times, n_points = 6, 5
matrices = {key: np.full((n_times, n_points), 100.0)
            for key in rfc.PRIMARY_COLUMNS}
matrices["sw_total_absorbed_Wm2"] = np.full((n_times, n_points), 300.0)
matrices["lw_total_absorbed_Wm2"] = np.full((n_times, n_points), 200.0)
matrices["total_absorbed_radiant_flux_Wm2"] = np.full((n_times, n_points), 500.0)
matrices["globe_absorbed_flux_Wm2"] = np.full((n_times, n_points), 640.0)
matrices["globe_steady_temperature_C"] = np.full((n_times, n_points), 42.0)
matrices["globe_radiative_equilibrium_C"] = np.full((n_times, n_points), 55.0)
time_hours = np.linspace(0.0, 20.0, n_times)
arrival = np.array([1.0, 5.0, 9.0])
indices = np.array([0, 2, 4])
# Deliberately ask for an MRT that forces a large closure rescale.
authoritative = np.full(3, 70.0)
sampled, scale = rfc.sample_route_contribution_matrices(
    matrices, time_hours, arrival, indices, authoritative,
    person_emissivity=0.97, sigma=rfc.np.float64(5.670374419e-8),
    validation={"absolute_tolerance_Wm2": 1e-4, "relative_tolerance": 1e-6,
                "mrt_absolute_tolerance_C": 1e-3})
check(np.max(np.abs(scale - 1.0)) > 0.1,
      "the test exercises a genuinely non-unit closure rescale",
      f"scale={scale[0]:.3f}")
check(np.allclose(sampled["globe_steady_temperature_C"], 42.0),
      "the body MRT rescale never touches the globe TEMPERATURE")
check(np.allclose(sampled["globe_absorbed_flux_Wm2"], 640.0),
      "the body MRT rescale never touches the globe absorbed flux")
check(np.allclose(sampled["sw_total_absorbed_Wm2"], 300.0 * scale),
      "body-absorbed columns are still rescaled as before")

# A negative globe temperature (cold clear night) must not trip the
# non-negativity guard that protects the flux record.
cold = dict(matrices)
cold["globe_steady_temperature_C"] = np.full((n_times, n_points), -3.5)
try:
    rfc.validate_contribution_arrays(
        cold, expected_mrt_c=None, person_emissivity=0.97)
    check(True, "a below-zero globe temperature is accepted by the record")
except ValueError as error:
    check(False, "a below-zero globe temperature is accepted by the record",
          str(error))
negative_flux = dict(matrices)
negative_flux["sw_total_absorbed_Wm2"] = np.full((n_times, n_points), -1.0)
try:
    rfc.validate_contribution_arrays(
        negative_flux, expected_mrt_c=None, person_emissivity=0.97)
    check(False, "a negative absorbed FLUX is still rejected")
except ValueError:
    check(True, "a negative absorbed FLUX is still rejected")

# ---------------------------------------------------------------------------
print("\nT10: stage 05 gives the globe the same scene as the pedestrian")
import argparse

base = argparse.Namespace(
    projected_area_model="standing", f_projected_direct=0.25,
    person_sw_absorptivity=0.70, person_emissivity=0.97,
    f_sky_diffuse=0.5, f_ground_reflected=0.5, ground_albedo=0.18,
    reflected_model="local", surrounding_emissivity=0.95,
    surface_temp_offset_day_c=8.0, clear_sky_emissivity="prata",
)
shim = stage05.globe_radiation_args(base, spec)
check(shim.projected_area_model == "sphere" and shim.f_projected_direct == 0.25,
      "the globe view uses the sphere projected-area factor")
check(abs(shim.person_sw_absorptivity - spec.sw_absorptivity) < 1e-12
      and abs(shim.person_emissivity - spec.emissivity) < 1e-12,
      "the globe view uses the globe's own optical properties")
check(base.projected_area_model == "standing"
      and base.person_sw_absorptivity == 0.70,
      "building the globe view does not mutate the pedestrian's arguments")
for field in ("f_sky_diffuse", "f_ground_reflected", "ground_albedo",
              "reflected_model", "surrounding_emissivity",
              "surface_temp_offset_day_c"):
    check(getattr(shim, field) == getattr(base, field),
          f"the globe sees the same {field} as the pedestrian")

# The two receptors must genuinely differ under a high sun, and by the sign the
# geometry predicts (sphere catches more beam than a standing body overhead).
common = dict(dni=800.0, dhi=120.0, ghi=850.0, tau_direct=np.ones(3),
              svf_person=np.full(3, 0.9), svf_ground=np.full(3, 0.9),
              air_temp_C=np.full(3, 28.0), rh_pct=45.0, cloud_fraction=0.0)
_, body_flux, body_sw, _ = stage05.estimate_mrt_from_radiation(
    common["dni"], common["dhi"], common["ghi"], 80.0, common["tau_direct"],
    common["svf_person"], common["svf_ground"], common["air_temp_C"],
    common["rh_pct"], common["cloud_fraction"], base)
_, sphere_flux, sphere_sw, _ = stage05.estimate_mrt_from_radiation(
    common["dni"], common["dhi"], common["ghi"], 80.0, common["tau_direct"],
    common["svf_person"], common["svf_ground"], common["air_temp_C"],
    common["rh_pct"], common["cloud_fraction"], shim)
def beam_factor_ratio(altitude: float) -> float:
    """Sphere beam absorption / standing-body beam absorption at one altitude."""
    return ((0.25 * spec.sw_absorptivity)
            / (float(stage05.projected_area_factor_standing(altitude))
               * base.person_sw_absorptivity))


high_ratio = beam_factor_ratio(80.0)
check(high_ratio > 1.0 and np.all(sphere_sw > body_sw),
      "at 80 deg sun the sphere absorbs more shortwave than the standing body",
      f"beam factor ratio {high_ratio:.2f}")
_, _, low_body_sw, _ = stage05.estimate_mrt_from_radiation(
    common["dni"], common["dhi"], common["ghi"], 10.0, common["tau_direct"],
    common["svf_person"], common["svf_ground"], common["air_temp_C"],
    common["rh_pct"], common["cloud_fraction"], base)
_, _, low_sphere_sw, _ = stage05.estimate_mrt_from_radiation(
    common["dni"], common["dhi"], common["ghi"], 10.0, common["tau_direct"],
    common["svf_person"], common["svf_ground"], common["air_temp_C"],
    common["rh_pct"], common["cloud_fraction"], shim)
low_ratio = beam_factor_ratio(10.0)
# The GEOMETRY reverses (checked in T2). Absorbed flux need not, because the
# globe's black paint absorbs 0.95 against a clothed body's 0.70 -- the optics
# can outweigh the geometry at low sun. What matters for the comparison rule is
# that the ratio is strongly altitude-dependent, so no fixed conversion factor
# can turn one receptor's reading into the other's.
check(high_ratio / low_ratio > 2.0,
      "the sphere-to-body beam ratio swings with solar altitude, so no fixed "
      "factor converts a globe reading into a body one",
      f"ratio {low_ratio:.2f} at 10 deg vs {high_ratio:.2f} at 80 deg")
check(float(low_sphere_sw[0]) / float(low_body_sw[0])
      < float(sphere_sw[0]) / float(body_sw[0]),
      "the sphere's shortwave advantage shrinks toward low sun",
      f"{float(low_sphere_sw[0]) / float(low_body_sw[0]):.2f} at 10 deg vs "
      f"{float(sphere_sw[0]) / float(body_sw[0]):.2f} at 80 deg")
check(np.all(sphere_flux > 0) and np.all(np.isfinite(sphere_flux)),
      "the sphere-weighted absorbed flux is positive and finite")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
