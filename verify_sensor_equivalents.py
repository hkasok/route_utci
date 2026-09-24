#!/usr/bin/env python3
"""Verification for the instrument-equivalent outputs and wall-reflected shortwave.

Pins the two rules these features exist to enforce:

1. Instrument-equivalent channels are a DIFFERENT QUANTITY from body-absorbed
   flux. They must never enter an absorbed-flux closure identity, and must
   never be touched by the stage-08 MRT closure rescaling.
2. Wall/roof reflected shortwave is additive, non-negative, closes exactly into
   the reflected-shortwave total, and is attributed to the wall/roof source
   columns rather than invented into the ground terms.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import radiant_flux_contributions as rfc

HERE = Path(__file__).resolve().parent
passed = failed = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f" ({detail})" if detail else ""))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m05 = load_module(HERE / "05_mrt_network_raytrace.py", "m05")


print("T1: sensor channels are separated from the absorbed-flux record")
check(set(rfc.SENSOR_COLUMNS).isdisjoint(
          rfc.PRIMARY_COLUMNS + rfc.TOTAL_COLUMNS
          + rfc.SW_SOURCE_COLUMNS + rfc.LW_SOURCE_COLUMNS),
      "sensor channels share no name with any absorbed-flux column")

n = 6
base = {key: np.full(n, 10.0) for key in rfc.PRIMARY_COLUMNS}
base["sw_total_absorbed_Wm2"] = np.full(n, 30.0)
base["lw_total_absorbed_Wm2"] = np.full(n, 20.0)
base["total_absorbed_radiant_flux_Wm2"] = np.full(n, 50.0)
with_sensor = dict(base)
for key in rfc.SENSOR_COLUMNS:
    with_sensor[key] = np.full(n, 400.0)      # far larger than any body term
report_plain = rfc.validate_contribution_arrays(base)
report_sensor = rfc.validate_contribution_arrays(with_sensor)
check(report_plain == report_sensor,
      "adding sensor channels does not change any closure residual")

print("\nT2: the route MRT closure never rescales an instrument reading")
sigma, emissivity = 5.670374419e-8, 0.97
time_hours = np.array([0.0, 12.0])
matrices = {key: np.vstack([value, value]) for key, value in with_sensor.items()}
# Ask for an MRT well above what the raw flux implies, forcing a large scale.
raw_mrt = (np.full(n, 50.0) / (emissivity * sigma)) ** 0.25 - 273.15
target_mrt = raw_mrt + 25.0
sampled, scale = rfc.sample_route_contribution_matrices(
    matrices, time_hours, np.zeros(n), np.arange(n), target_mrt,
    person_emissivity=emissivity, sigma=sigma,
    validation={"absolute_tolerance_Wm2": 1e-4, "relative_tolerance": 1e-6,
                "mrt_absolute_tolerance_C": 1e-3})
check(np.all(scale > 1.5), "the test forces a large closure scale",
      f"scale {scale[0]:.2f}")
check(all(np.allclose(sampled[key], 400.0) for key in rfc.SENSOR_COLUMNS),
      "sensor channels pass through the closure completely unscaled")
check(np.allclose(sampled["total_absorbed_radiant_flux_Wm2"], 50.0 * scale),
      "body-absorbed flux is still rescaled onto the authoritative MRT")


print("\nT3: emulated radiometer physics")
args = SimpleNamespace(ground_albedo=0.2, person_sw_absorptivity=0.7,
                       person_emissivity=0.97)
svf = np.array([1.0, 0.5, 0.0])
sensor = m05.sensor_radiometer_quantities(
    dni=800.0, dhi=120.0, elevation_deg=50.0,
    tau_direct=np.array([1.0, 1.0, 0.0]), svf_planar=svf,
    L_sky=np.full(3, 350.0), L_surround=np.full(3, 480.0),
    ground_albedo=np.full(3, 0.2),
    ground_emitted_Wm2=np.full(3, 0.95 * sigma * 320.0 ** 4),
    ground_emissivity=np.full(3, 0.95), surround_sw_radiance=np.full(3, 100.0),
    args=args)
down_sw = sensor["sensor_shortwave_down_Wm2"]
expected_open = 800.0 * np.sin(np.deg2rad(50.0)) + 120.0
check(abs(down_sw[0] - expected_open) < 1e-9,
      "open sunlit point reproduces beam-on-horizontal plus full sky diffuse",
      f"{down_sw[0]:.1f} W/m2")
check(down_sw[2] < down_sw[0],
      "a fully obstructed shaded point receives far less shortwave",
      f"{down_sw[2]:.1f} vs {down_sw[0]:.1f} W/m2")
check(np.allclose(sensor["sensor_shortwave_up_Wm2"], 0.2 * down_sw),
      "upwelling shortwave is the footprint ground albedo times downwelling")
down_lw = sensor["sensor_longwave_down_Wm2"]
check(abs(down_lw[0] - 350.0) < 1e-9 and abs(down_lw[2] - 480.0) < 1e-9,
      "downwelling longwave runs from pure sky to pure surround with sky view",
      f"{down_lw[0]:.0f} -> {down_lw[2]:.0f} W/m2")
up_lw = sensor["sensor_longwave_up_Wm2"]
expected_up = 0.95 * sigma * 320.0 ** 4 + 0.05 * down_lw
check(np.allclose(up_lw, expected_up),
      "upwelling longwave is ground emission plus reflected downwelling")
check(np.all(up_lw > down_lw),
      "a ground warmer than the sky emits more than it receives")

print("\nT3b: the downward footprint is the instrument's view, not the body's")
# Build a synthetic flat ground of small tiles and check the kernel against the
# analytic result for a downward-facing sensor over a uniform plane.
import scipy.sparse as sp


class _FootprintStub:
    """Minimal stand-in exposing exactly what build_sensor_ground_footprint
    touches, so the kernel can be exercised without a solved case."""

    def __init__(self, centroids, areas, albedo, eps, temps):
        n = len(areas)
        self._ground_mask = np.ones(n, dtype=bool)
        self._facet_centroid = centroids
        self._facet_area = areas
        normals = np.zeros((n, 3)); normals[:, 2] = 1.0
        self._facet_normal_ground = normals
        self.facet_albedo = albedo
        self.facet_eps = eps
        self.facet_T = temps[None, :]
        self.local_ground_albedo = None
        self.args = SimpleNamespace(ground_albedo=0.18)
        self._footprint = None
        self.sensor_ground_albedo = None
        self.sensor_ground_emissivity = None
        self.sensor_footprint_coverage = 0.0
        self.sensor_footprint_radius_m = None

    build_sensor_ground_footprint = m05.FacetLongwave.build_sensor_ground_footprint
    sensor_ground_emitted = m05.FacetLongwave.sensor_ground_emitted


step = 0.25
grid = np.arange(-40.0, 40.0 + step, step)
gx, gy = np.meshgrid(grid, grid)
centroids = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
areas = np.full(gx.size, step * step)
temps = np.full(gx.size, 300.0)
eps = np.full(gx.size, 0.95)
albedo = np.full(gx.size, 0.20)

height = 1.0
stub = _FootprintStub(centroids, areas, albedo, eps, temps)
built = stub.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
check(built, "footprint builds over a resolved ground mesh")
uniform = float(stub.sensor_ground_emitted(0)[0])
check(abs(uniform - 0.95 * sigma * 300.0 ** 4) < 1e-6,
      "over a uniform surface the footprint returns exactly that surface",
      f"{uniform:.3f} W/m2")

# The defining property: the weight is concentrated near the sensor. Analytic
# fraction of the signal from within radius R over flat ground is R^2/(R^2+h^2).
weights = np.asarray(stub._footprint.todense()).ravel()
radius = np.hypot(centroids[:, 0], centroids[:, 1])
total_weight = weights.sum()
for R, expected in ((1.0, 0.50), (3.0, 0.90), (10.0, 0.99)):
    share = weights[radius <= R].sum() / total_weight
    check(abs(share - expected) < 0.02,
          f"{expected:.0%} of the weight lies within {R:.0f} m of the sensor",
          f"{share:.1%}")
check(weights[radius <= 3.0].sum() / total_weight > 0.85,
      "the emulated pyrgeometer is a LOCAL instrument, not a 300 m average")

# A hot patch far away must barely register; the same patch underfoot must
# dominate. This is precisely what the old cylinder-view version got wrong.
hot = temps.copy()
far = (radius > 20.0)
hot[far] = 340.0
stub_far = _FootprintStub(centroids, areas, albedo, eps, hot)
stub_far.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
far_effect = float(stub_far.sensor_ground_emitted(0)[0]) - uniform
near = temps.copy()
near[radius <= 2.0] = 340.0
stub_near = _FootprintStub(centroids, areas, albedo, eps, near)
stub_near.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
near_effect = float(stub_near.sensor_ground_emitted(0)[0]) - uniform
check(far_effect < 3.0 and near_effect > 60.0,
      "ground beyond 20 m barely moves the reading while ground within 2 m "
      "dominates it",
      f"far {far_effect:+.1f} vs near {near_effect:+.1f} W/m2")

# Radiosity averaging, not temperature averaging: sigma*T^4 is convex, so the
# two differ whenever the surface is not isothermal.
mixed = np.where(radius <= 2.0, 330.0, 290.0)
stub_mixed = _FootprintStub(centroids, areas, albedo, eps, mixed)
stub_mixed.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
radiosity_mean = float(stub_mixed.sensor_ground_emitted(0)[0])
w = np.asarray(stub_mixed._footprint.todense()).ravel()
temperature_mean = 0.95 * sigma * (np.sum(w * mixed) / w.sum()) ** 4
check(radiosity_mean > temperature_mean + 1.0,
      "averaging radiosity exceeds averaging temperature then taking T^4, as "
      "Jensen requires -- the old code took the lower, wrong one",
      f"{radiosity_mean:.1f} vs {temperature_mean:.1f} W/m2")

# Sloped ground must still integrate to the surface's own emission.
tilt = np.deg2rad(20.0)
sloped = centroids.copy()
sloped[:, 2] = sloped[:, 0] * np.tan(tilt)
stub_slope = _FootprintStub(sloped, areas / np.cos(tilt), albedo, eps, temps)
normals = np.zeros((len(areas), 3))
normals[:, 0] = -np.sin(tilt); normals[:, 2] = np.cos(tilt)
stub_slope._facet_normal_ground = normals
stub_slope.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
check(abs(float(stub_slope.sensor_ground_emitted(0)[0])
          - 0.95 * sigma * 300.0 ** 4) < 1.0,
      "a uniformly warm SLOPED surface still returns its own emission",
      f"{float(stub_slope.sensor_ground_emitted(0)[0]):.2f} W/m2")

flipped = _FootprintStub(centroids, areas, albedo, eps, temps)
flipped._facet_normal_ground = -flipped._facet_normal_ground
flipped.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
check(abs(float(flipped.sensor_ground_emitted(0)[0]) - uniform) < 1e-6,
      "downward-wound ground normals are re-oriented rather than silently "
      "dropping those facets")

empty = _FootprintStub(centroids, areas, albedo, eps, temps)
empty._ground_mask = np.zeros(len(areas), dtype=bool)
check(empty.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height) is False,
      "a case with no ground facets reports no footprint instead of failing")

print("\nT4: sensor channels answer the measured convention, not the body one")
# The same scene, expressed as a standing-cylinder absorbed load, is a very
# different number -- which is exactly why the two must not be compared.
body = m05.estimate_mrt_from_radiation(
    800.0, 120.0, 800.0 * np.sin(np.deg2rad(50.0)) + 120.0, 50.0,
    np.array([1.0]), np.array([0.5]), np.array([0.5]),
    np.array([25.0]), np.array([50.0]), 0.0,
    SimpleNamespace(person_sw_absorptivity=0.7, person_emissivity=0.97,
                    f_projected_direct=0.25, f_sky_diffuse=0.5,
                    f_ground_reflected=0.5, ground_albedo=0.2,
                    projected_area_model="sphere", reflected_model="local",
                    surrounding_emissivity=0.95, clear_sky_emissivity="prata",
                    surface_temp_offset_day_c=8.0),
    L_surround_override=np.array([480.0]), lw_sky_frac=np.array([0.5]))
body_lw_absorbed = body[3][0]
sensor_lw_pair = 0.5 * (down_lw[1] + up_lw[1])
check(abs(body_lw_absorbed - sensor_lw_pair) > 20.0,
      "body-absorbed longwave differs materially from the sensor pair mean",
      f"body {body_lw_absorbed:.0f} vs two-hemisphere sensor mean "
      f"{sensor_lw_pair:.0f} W/m2")


print("\nT5: wall-reflected shortwave closes and is attributed correctly")
common = dict(person_sw_absorptivity=0.7, person_emissivity=0.97,
              f_projected_direct=0.25, f_sky_diffuse=0.5,
              f_ground_reflected=0.5, ground_albedo=0.2,
              projected_area_model="sphere", reflected_model="local",
              surrounding_emissivity=0.95, clear_sky_emissivity="prata",
              surface_temp_offset_day_c=8.0)
call = dict(dni=700.0, dhi=140.0, ghi=600.0, elevation_deg=45.0,
            tau_direct=np.array([1.0]), svf_person=np.array([0.4]),
            svf_ground=np.array([0.4]), air_temp_C=np.array([28.0]),
            rh_pct=np.array([50.0]), cloud_fraction=0.0,
            L_surround_override=np.array([470.0]),
            lw_sky_frac=np.array([0.4]))
without = m05.estimate_mrt_from_radiation(
    args=SimpleNamespace(**common), return_contributions=True, **call)
wall_parts = {"wall": np.array([90.0]), "roof": np.array([10.0])}
with_wall = m05.estimate_mrt_from_radiation(
    args=SimpleNamespace(**common), return_contributions=True,
    wall_reflected_incident=np.array([100.0]),
    wall_reflected_parts=wall_parts, **call)
c0, c1 = without[4], with_wall[4]
check(c1["sw_reflected_total_absorbed_Wm2"][0]
      > c0["sw_reflected_total_absorbed_Wm2"][0],
      "wall reflection increases reflected shortwave",
      f"{c0['sw_reflected_total_absorbed_Wm2'][0]:.1f} -> "
      f"{c1['sw_reflected_total_absorbed_Wm2'][0]:.1f} W/m2")
check(abs(c1["sw_reflected_total_absorbed_Wm2"][0]
          - c0["sw_reflected_total_absorbed_Wm2"][0] - 0.7 * 100.0) < 1e-9,
      "the added flux is exactly absorptivity times the incident reflection")
check(abs(c1["sw_reflected_building_wall_absorbed_Wm2"][0] - 0.7 * 90.0) < 1e-9
      and abs(c1["sw_reflected_roof_absorbed_Wm2"][0] - 0.7 * 10.0) < 1e-9,
      "wall and roof shares are attributed to their own source columns")
check(np.allclose(c1["lw_total_absorbed_Wm2"], c0["lw_total_absorbed_Wm2"]),
      "longwave is untouched by the shortwave addition")
check(with_wall[0][0] > without[0][0],
      "MRT rises when sunlit facades are counted",
      f"{without[0][0]:.2f} -> {with_wall[0][0]:.2f} C")
for label, contributions in (("without", c0), ("with wall", c1)):
    report = rfc.validate_contribution_arrays(
        contributions, expected_mrt_c=(without if label == "without"
                                       else with_wall)[0])
    check(report["maximum_total_closure_error_Wm2"] < 1e-6,
          f"absorbed-flux closure holds {label} wall reflection",
          f"{report['maximum_total_closure_error_Wm2']:.2e} W/m2")
check(m05.estimate_mrt_from_radiation(
          args=SimpleNamespace(**common), return_contributions=False,
          wall_reflected_incident=None, wall_reflected_parts=None,
          **call)[0][0] == without[0][0],
      "omitting wall reflection reproduces the previous result exactly")

print("\nT6: the field comparison offers only like-for-like quantities")
# The two-hemisphere absorbed-flux comparison was removed outright, not gated.
# Nothing may reintroduce a body-absorbed-versus-horizontal-sensor pairing.
comparison = load_module(HERE / "compare_mrt_lisbon_data.py", "cmp")
body_absorbed = set(rfc.PRIMARY_COLUMNS + rfc.TOTAL_COLUMNS
                    + rfc.SW_SOURCE_COLUMNS + rfc.LW_SOURCE_COLUMNS)
paired_model_columns = {model for _measured, model
                        in comparison.FLUX_COMPONENTS.values()}
check(paired_model_columns <= set(rfc.SENSOR_COLUMNS),
      "every compared model column is an instrument-equivalent channel",
      f"{sorted(paired_model_columns)}")
check(not (paired_model_columns & body_absorbed),
      "no body-absorbed column is ever compared against the sensor")
check(all(measured.startswith("measured_") and "diagnostic" not in measured
          and "proxy" not in measured
          for measured, _model in comparison.FLUX_COMPONENTS.values()),
      "every compared measured column is a raw radiometer channel")
for retired in ("LEGACY_PROXY_FLUX_COMPONENTS", "add_measured_flux_diagnostics",
                "plot_absorbed_flux_scatter", "plot_day_absorbed_flux_along_route",
                "TWO_HEMISPHERE_SW_EFFECTIVE_FACTOR",
                "TWO_HEMISPHERE_LW_VIEW_FACTOR"):
    check(not hasattr(comparison, retired),
          f"retired proxy machinery '{retired}' is gone")
check(hasattr(comparison, "plot_day_radiometer_along_route"),
      "the along-route figure is the radiometer-channel version")


print("\nT_facet: facet-resolved upwelling shortwave and up-facing downwelling")
import scipy.sparse as _sp

# (a) Each ground facet reflects its OWN irradiance. Over a shadow edge
# through the sensor the reading is the footprint average of the two sides,
# whatever the sensor itself sees.
stub_sw = _FootprintStub(centroids, areas, albedo, eps, temps)
stub_sw.build_sensor_ground_footprint(
    np.array([[0.0, 0.0, height]]), height, height, maximum_radius_m=40.0)
stub_sw.sensor_ground_reflected = m05.FacetLongwave.sensor_ground_reflected.__get__(stub_sw)
uniform_in = np.full(len(areas), 700.0)
check(abs(float(stub_sw.sensor_ground_reflected(uniform_in)[0]) - 0.20 * 700.0) < 1e-6,
      "uniform irradiance reflects exactly albedo x irradiance")
edge_in = np.where(centroids[:, 0] < 0.0, 800.0, 100.0)
edge_in[centroids[:, 0] == 0.0] = 450.0   # cells ON the edge: half each
edge = float(stub_sw.sensor_ground_reflected(edge_in)[0])
check(abs(edge - 0.20 * 450.0) < 0.5,
      "a shadow edge under the sensor gives the average of both sides, not "
      "the sensor's own shade state", f"{edge:.1f} W/m2")


class _UpStub:
    """Minimal FacetLongwave for the up-facing sensor methods."""
    def __init__(self, W_up, w_sky_up, w_veg_up, J, J_env, albedo):
        self.W_up, self.w_sky_up, self.w_veg_up = W_up, w_sky_up, w_veg_up
        self.facet_J = J[None, :]
        self.environment_J = np.array([J_env])
        self.facet_albedo = albedo
        self.point_map = np.arange(W_up.shape[0])
        self.args = SimpleNamespace(vegetation_emissivity=1.0)
    _radiosities = m05.FacetLongwave._radiosities
    sensor_downwelling_at = m05.FacetLongwave.sensor_downwelling_at


T0 = 300.0
bb = sigma * T0 ** 4
# column 0: a wall above the horizon; column 1: ground below it (zero up-weight)
W_up = _sp.csr_matrix(np.array([[0.4, 0.0]]))
up = _UpStub(W_up, np.array([0.5]), np.array([0.1]),
             np.array([bb, bb]), bb, np.array([0.3, 0.2]))
lw, _ = up.sensor_downwelling_at(0, T0 - 273.15, 30.0, bb)
check(abs(float(lw[0]) - bb) < 1e-6,
      "an isothermal black enclosure reads sigma*T^4 on the up-facing sensor")
hot_ground = _UpStub(W_up, np.array([0.5]), np.array([0.1]),
                     np.array([bb, 3.0 * bb]), bb, np.array([0.3, 0.2]))
lw_hot, _ = hot_ground.sensor_downwelling_at(0, T0 - 273.15, 30.0, bb)
check(abs(float(lw_hot[0]) - bb) < 1e-6,
      "ground below the horizon cannot reach the up-facing sensor -- the "
      "cylinder surround mean it replaces could")
_, sw = up.sensor_downwelling_at(0, T0 - 273.15, 30.0, bb,
                                 facet_incident_sw=np.array([500.0, 900.0]))
check(abs(float(sw[0]) - 0.4 * 0.3 * 500.0) < 1e-9,
      "reflected shortwave from above comes only from surfaces above the horizon")

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
