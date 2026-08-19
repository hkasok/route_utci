#!/usr/bin/env python3
"""Fast synthetic verification for the optional TREC-Route air-field stage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from microclimate_field import (
    AXES_FILE, EnvironmentField, FLUID_MASK_FILE, METADATA_FILE,
    MicroclimateField, TEMPERATURE_FILE, VELOCITY_FILES, GROUND_HEIGHT_FILE)
from importlib import import_module


solver = import_module("05c_microclimate_solver")
passed = failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS {label}")
    else:
        failed += 1
        print(f"  FAIL {label}")


class DummyWeather:
    def air_temp_c(self, hours):
        return 25.0 + 0.1 * np.asarray(hours)

    def rh_pct(self, hours):
        return np.zeros_like(np.asarray(hours), dtype=float) + 60.0

    def wind_ms(self, hours):
        return np.zeros_like(np.asarray(hours), dtype=float) + 2.5


print("T1: solved-field contract and interpolation")
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    x = y = z = np.array([0.0, 1.0])
    hours = np.array([0.0, 12.0])
    np.savez(directory / AXES_FILE, x_m=x, y_m=y, z_m=z,
             time_hours=hours)
    shape = (2, 2, 2, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    temperature = np.stack((20 + xx + 2 * yy + 3 * zz,
                            22 + xx + 2 * yy + 3 * zz)).astype(np.float32)
    np.save(directory / TEMPERATURE_FILE, temperature)
    np.save(directory / VELOCITY_FILES[0], np.ones(shape, np.float32))
    np.save(directory / VELOCITY_FILES[1], np.ones(shape, np.float32) * 2)
    np.save(directory / VELOCITY_FILES[2], np.ones(shape, np.float32) * 2)
    np.save(directory / FLUID_MASK_FILE, np.ones(shape[1:], dtype=bool))
    np.save(directory / GROUND_HEIGHT_FILE, np.zeros((2, 2), dtype=np.float32))
    (directory / METADATA_FILE).write_text(json.dumps({
        "format_version": 1,
        "config": {"flow": {"roughness_length_m": 0.5,
                            "reference_wind_height_m": 10.0}}}))
    field = MicroclimateField(directory)
    ta, u, v, w = field.sample(np.array([[0.5, 0.5, 0.5]]), 6.0)
    check(np.allclose(ta, 25.5),
          "trilinear space + time with first-fluid-level sampling")
    check(np.allclose([u[0], v[0], w[0]], [1, 2, 2]),
          "all three velocity components survive interpolation")
    environment = EnvironmentField(DummyWeather(), directory)
    sample = environment.sample(np.array([[0.5, 0.5, 0.5]]), 6.0)
    check(np.allclose(sample.wind_speed_ms, 3.0),
          "vector magnitude, not a component, drives thermal models")
    check(sample.utci_wind_speed_10m_ms[0] > sample.wind_speed_ms[0],
          "pedestrian velocity converts to UTCI 10 m reference speed")
    check(sample.source == "solved_microclimate_field",
          "solved source provenance is explicit")


print("T2: backward-compatible uniform fallback")
fallback = EnvironmentField(DummyWeather())
points = np.array([[0, 0, 1], [100, 50, 2]], dtype=float)
sample = fallback.sample(points, np.array([2.0, 2.0]))
check(np.allclose(sample.air_temperature_c, 25.2),
      "fallback broadcasts original weather temperature")
check(np.allclose(sample.wind_speed_ms, 2.5),
      "fallback preserves original weather wind magnitude")
check(np.allclose(sample.utci_wind_speed_10m_ms, 2.5),
      "fallback weather wind retains its established UTCI reference")
check(np.allclose(sample.relative_humidity_pct, 60.0),
      "humidity remains with shared weather provider")


print("T3: elliptic projection and thermal transport")
shape = (10, 10, 10)
solid = np.zeros(shape, dtype=bool)
solid[3:7, 4:6, 4:6] = True
rng = np.random.default_rng(42)
u = rng.normal(1.0, 0.3, shape)
v = rng.normal(0.0, 0.3, shape)
w = rng.normal(0.0, 0.2, shape)
pu, pv, pw, before, after = solver.project_velocity_fft(
    u, v, w, solid, 2.0, 2.0, 2.0, passes=4)
check(after < before, "Poisson projection reduces RMS divergence")
check(np.allclose(pu[solid], 0) and np.allclose(pv[solid], 0)
      and np.allclose(pw[solid], 0), "solid cells retain zero velocity")

temperature = np.full(shape, 25.0)
source = np.zeros(shape); source[5, 5, 7] = 1.0 / 3600.0
advanced, substeps = solver.advance_temperature(
    temperature, np.zeros(shape), np.zeros(shape), np.zeros(shape),
    source, ~solid, 25.0, 600.0, 2.0, 2.0, 2.0,
    diffusivity=0.5, maximum_cfl=0.65, maximum_anomaly_k=10.0)
check(advanced[5, 5, 7] > 25.0, "surface sensible heat warms local air")
check(np.isfinite(advanced).all() and substeps >= 1,
      "advection-diffusion remains finite and stability-substepped")


print("T4: 60 m domain and non-periodic physical boundary conditions")
config = solver.load_config("microclimate_config.json")
check(config["grid"]["height_above_local_ground_m"] == 60.0,
      "default atmospheric-domain setting is 60 m")
class DummyGround:
    vertices = np.array([[0.0, 0.0, 0.7], [20.0, 0.0, 0.7],
                         [0.0, 20.0, 0.7]])

test_path = np.array([[0.0, 0.0, 1.8], [20.0, 20.0, 1.8]])
_x, _y, test_z, _dx, _dy, _dz = solver.build_grid(
    test_path, DummyGround(), config)
check(test_z[-1] - DummyGround.vertices[:, 2].max() >= 60.0,
      "constructed grid reaches at least 60 m above highest terrain")
boundary = solver.FlowBoundaryConditions(270.0)  # westerly: flow toward +X
check(boundary.flow_axis == 2 and boundary.inlet_side == "low"
      and boundary.outlet_side == "high" and boundary.lateral_axis == 1,
      "westerly wind selects X-low inlet, X-high outlet, and Y slip sides")

shape = (12, 14, 18)
solid = np.zeros(shape, dtype=bool)
solid[0, :, :] = True
solid[2:8, 5:9, 8:10] = True
u = np.full(shape, 2.0); v = np.zeros(shape); w = np.zeros(shape)
w[4:8, 6:8, 7:11] = 0.8
inlet_u, inlet_v = u.copy(), v.copy()
boundary.apply_velocity(u, v, w, inlet_u, inlet_v, solid)
projector = solver.MassConsistentProjector(
    solid, 2.0, 2.0, 2.0, maximum_iterations=1000,
    relative_tolerance=1e-9, boundary=boundary)
pu, pv, pw, before, after = projector.project(u, v, w)
boundary.apply_velocity(
    pu, pv, pw, inlet_u, inlet_v, solid, after_projection=True)
check(after < before * 1e-5,
      "open-boundary pressure projection conserves mass")
left_id = projector.index[5, 5, 0]
right_id = projector.index[5, 5, -1]
check(projector.matrix[left_id, right_id] == 0,
      "Poisson matrix has no periodic opposite-face connection")
check(np.allclose(pv[:, 0, :], 0.0) and np.allclose(pv[:, -1, :], 0.0),
      "both lateral faces enforce zero normal velocity")
check(np.allclose(pu[1:, :, 0][~solid[1:, :, 0]], 2.0),
      "inlet velocity remains prescribed after projection")

temperature = np.full(shape, 25.0)
temperature[5, 6:8, -2] = 30.0
advanced, _ = solver.advance_temperature(
    temperature, np.zeros(shape), np.zeros(shape), np.zeros(shape),
    np.zeros(shape), ~solid, 25.0, 60.0, 2.0, 2.0, 2.0,
    diffusivity=0.5, maximum_cfl=0.65, maximum_anomaly_k=10.0,
    boundary=boundary)
check(np.allclose(advanced[:, :, 0], 25.0),
      "downstream heat does not wrap around to prescribed inlet")
check(np.allclose(advanced[:, 0, :], advanced[:, 1, :])
      and np.allclose(advanced[:, -1, :], advanced[:, -2, :]),
      "lateral temperature boundaries are adiabatic")
check(np.allclose(advanced[-1, :, :], advanced[-2, :, :]),
      "top temperature boundary is open/zero-gradient")


print("T5: ParaView export round-trip")
from types import SimpleNamespace

from paraview_export import read_vtk_appended

with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    x = np.array([0.0, 4.0])
    y = np.array([0.0, 4.0, 8.0])
    z = np.array([1.0, 3.0])
    hours = np.array([6.0, 12.0, 18.0])
    shape = (3, len(z), len(y), len(x))
    rng = np.random.default_rng(7)
    temperature = rng.normal(25.0, 3.0, shape).astype(np.float32)
    u_field = rng.normal(1.0, 0.5, shape).astype(np.float32)
    v_field = rng.normal(0.0, 0.5, shape).astype(np.float32)
    w_field = rng.normal(0.0, 0.2, shape).astype(np.float32)
    fluid = np.ones(shape[1:], dtype=bool)
    fluid[0, 1, 1] = False
    np.savez(directory / AXES_FILE, x_m=x, y_m=y, z_m=z, time_hours=hours)
    np.save(directory / TEMPERATURE_FILE, temperature)
    for name, field in zip(VELOCITY_FILES, (u_field, v_field, w_field)):
        np.save(directory / name, field)
    np.save(directory / FLUID_MASK_FILE, fluid)
    np.save(directory / GROUND_HEIGHT_FILE,
            np.zeros(shape[2:], dtype=np.float32))
    context = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 8.0, 1.0]]),
        faces=np.array([[0, 1, 2]]))
    config = {"output": {"write_paraview": True, "paraview_time_stride": 2}}
    pvd_path = solver.export_paraview(directory, config,
                                      context_meshes={"context_ground": context})
    check(pvd_path.is_file(), "collection (.pvd) file is written")
    listed = [line for line in pvd_path.read_text().splitlines()
              if "DataSet" in line]
    check(len(listed) == 2 and 'timestep="6.000000"' in listed[0]
          and 'timestep="18.000000"' in listed[1],
          "time stride selects frames and stamps local hours")
    frame = read_vtk_appended(directory / "paraview" / "microclimate_000.vtr")
    arrays = frame["arrays"]
    check(np.array_equal(arrays["x_m"], x.astype(np.float32))
          and np.array_equal(arrays["y_m"], y.astype(np.float32))
          and np.array_equal(arrays["z_m"], z.astype(np.float32)),
          "grid coordinates survive the round-trip exactly")
    check(np.array_equal(arrays["air_temperature_C"],
                         temperature[0].ravel(order="C")),
          "temperature PointData is x-fastest flattened (z,y,x) order")
    expected_velocity = np.stack(
        (u_field[0], v_field[0], w_field[0]), axis=-1).reshape(-1, 3)
    check(np.array_equal(arrays["velocity_ms"], expected_velocity),
          "velocity exports as one 3-component vector array")
    check(np.array_equal(arrays["fluid_mask"],
                         fluid.astype(np.uint8).ravel(order="C")),
          "fluid mask allows thresholding solid cells in ParaView")
    mesh_file = read_vtk_appended(
        directory / "paraview" / "context_ground.vtp")
    check(np.array_equal(mesh_file["arrays"]["__points__"],
                         context.vertices.astype(np.float32))
          and np.array_equal(mesh_file["arrays"]["connectivity"], [0, 1, 2]),
          "context geometry mesh survives the round-trip")


print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
