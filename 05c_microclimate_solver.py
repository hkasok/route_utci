#!/usr/bin/env python3
"""Optional diagnostic 3-D temperature and mass-consistent velocity solver.

This is the computationally inexpensive microclimate enhancement used by
TREC-Route between the baseline facet-temperature solution and the final MRT.
It is deliberately not advertised as full CFD.  It couples the existing 1-D
transient facet conduction/energy-balance result to a diagnostic atmospheric
field as follows:

1. Material-specific facet surface temperatures and the shared weather wind
   determine sensible convective heat flux ``q_h = h_c (T_s - T_a)``.
2. Positive sensible flux injects localized thermal plume velocity immediately
   outside route-visible ground, roof and wall facets.
3. The background wind plus plume field is projected through an obstacle-aware
   finite-volume elliptic Poisson solve. Buildings and terrain are impermeable
   cells; the pressure correction produces mass-consistent flow around them at
   diagnostic rather than Navier–Stokes cost.
4. Air temperature is advanced by a conservative first-order upwind
   advection–diffusion step with facet sensible heat as a volumetric source.

The resulting field is spatially and temporally varying and follows the file
contract in ``microclimate_field.py``.  The optional pipeline stage alternates
this solver with stage 05b, so local Ta and velocity update facet convection
before final MRT is regenerated.  With the stage disabled, no file is loaded
and all downstream programs retain the historical uniform WeatherProvider
forcing exactly.

Limitations are explicit: the solver is a diagnostic mass-consistent model,
not RANS/LES; turbulence is represented by an eddy diffusivity; wind direction
is configured because the legacy weather contract supplies speed only; latent
moisture transport is not solved; vegetation drag is not yet represented.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import cg
import trimesh

from microclimate_field import (
    AXES_FILE, FLUID_MASK_FILE, METADATA_FILE, TEMPERATURE_FILE,
    VELOCITY_FILES, GROUND_HEIGHT_FILE)


DEFAULT_CONFIG = {
    "grid": {"horizontal_spacing_m": 8.0, "vertical_spacing_m": 2.0,
             "route_buffer_m": 80.0, "height_above_local_ground_m": 60.0,
             "maximum_cells": 140000},
    "flow": {"wind_direction_from_deg": 270.0,
             "inlet_velocity_ms": None,
             "reference_wind_height_m": 10.0, "roughness_length_m": 0.5,
             "minimum_background_speed_ms": 0.05, "maximum_speed_ms": 15.0,
             "buoyancy_velocity_efficiency": 0.22,
             "poisson_maximum_iterations": 250,
             "poisson_relative_tolerance": 1e-6,
             "maximum_divergence_ratio": 0.05,
             "boundary_conditions": {
                 "inlet": "specified_velocity_and_ambient_temperature",
                 "lateral_sides": "free_slip_adiabatic",
                 "outlet": "pressure_open_zero_gradient",
                 "top": "pressure_open_zero_gradient"}},
    "temperature": {"air_density_kgm3": 1.184,
                    "specific_heat_JkgK": 1006.0,
                    "eddy_diffusivity_m2s": 1.5,
                    "maximum_source_change_K_per_hour": 8.0,
                    "maximum_air_anomaly_K": 12.0, "maximum_cfl": 0.65},
    "output": {"write_diagnostic_figure": True,
               "write_paraview": True, "paraview_time_stride": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve optional TREC-Route diagnostic 3-D microclimate fields")
    parser.add_argument("--buildings-stl", required=True)
    parser.add_argument("--ground-stl", required=True)
    parser.add_argument("--facets-dir", required=True,
                        help="Stage-05a/05b directory containing facets and temperatures")
    parser.add_argument(
        "--urban-radiation-dir", default=None,
        help=("Optional 05d full-surface radiation directory. When supplied, "
              "all urban facets and their radiation-driven surface temperatures "
              "replace the route-visible-only heat-source set."))
    parser.add_argument("--mrt-dir", required=True,
                        help="Preparation directory containing path_xyz.npy and times.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate cached field geometry/routes/times and exit")
    parser.add_argument("--horizontal-spacing-m", type=float, default=None,
                        help="Override config grid spacing (diagnostic/testing)")
    parser.add_argument("--vertical-spacing-m", type=float, default=None)
    parser.add_argument("--maximum-cells", type=int, default=None,
                        help="Override adaptive spatial-cell cap")
    parser.add_argument("--no-diagnostic-figure", action="store_true")
    parser.add_argument("--no-paraview", action="store_true",
                        help="Skip the ParaView (.vtr/.pvd) field export")
    return parser.parse_args()


def _merge(base: dict, update: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> dict:
    config = DEFAULT_CONFIG
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        config = _merge(DEFAULT_CONFIG, supplied)
    for group in ("grid", "flow", "temperature"):
        if group not in config:
            raise ValueError(f"microclimate config missing {group!r}")
    if config["grid"]["horizontal_spacing_m"] <= 0:
        raise ValueError("horizontal grid spacing must be positive")
    if config["grid"]["vertical_spacing_m"] <= 0:
        raise ValueError("vertical grid spacing must be positive")
    if int(config["grid"]["maximum_cells"]) < 1000:
        raise ValueError("maximum_cells must be at least 1000")
    if not 0 < config["temperature"]["maximum_cfl"] <= 1:
        raise ValueError("maximum_cfl must lie in (0, 1]")
    if float(config["grid"]["height_above_local_ground_m"]) <= 0:
        raise ValueError("microclimate height above local ground must be positive")
    inlet_velocity = config["flow"].get("inlet_velocity_ms")
    if inlet_velocity is not None and float(inlet_velocity) < 0:
        raise ValueError("specified inlet velocity cannot be negative")
    return config


def file_identity(path: str | Path) -> dict:
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        first = stream.read(1024 * 1024)
        digest.update(first)
        if stat.st_size > len(first):
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read())
    return {"path": str(source.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest()}


def divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray,
               dx: float, dy: float, dz: float) -> np.ndarray:
    """Cell-centred diagnostic divergence."""
    return (np.gradient(u, dx, axis=2, edge_order=1)
            + np.gradient(v, dy, axis=1, edge_order=1)
            + np.gradient(w, dz, axis=0, edge_order=1))


class FlowBoundaryConditions:
    """Flow-oriented conditions for the rectangular diagnostic domain.

    The dominant horizontal wind component selects one inlet and the opposite
    outlet. The other horizontal pair are the two free-slip sides. This keeps
    the regular ENU grid/file contract while giving every outer face one clear
    physical role for arbitrary configured wind directions.
    """

    def __init__(self, wind_direction_from_deg: float):
        direction_to = (float(wind_direction_from_deg) + 180.0) % 360.0
        theta = np.deg2rad(90.0 - direction_to)
        self.unit_u = float(np.cos(theta))
        self.unit_v = float(np.sin(theta))
        self.flow_axis = 2 if abs(self.unit_u) >= abs(self.unit_v) else 1
        component = self.unit_u if self.flow_axis == 2 else self.unit_v
        self.inlet_side = "low" if component >= 0 else "high"
        self.outlet_side = "high" if self.inlet_side == "low" else "low"
        self.lateral_axis = 1 if self.flow_axis == 2 else 2

    @staticmethod
    def _plane(axis: int, side: str) -> tuple:
        index = [slice(None)] * 3
        index[axis] = 0 if side == "low" else -1
        return tuple(index)

    @staticmethod
    def _adjacent(axis: int, side: str) -> tuple:
        index = [slice(None)] * 3
        index[axis] = 1 if side == "low" else -2
        return tuple(index)

    @property
    def open_faces(self) -> tuple[tuple[int, str], ...]:
        return ((self.flow_axis, self.outlet_side), (0, "high"))

    def describe(self) -> dict:
        axis_name = {1: "y", 2: "x"}
        return {
            "inlet": f"{axis_name[self.flow_axis]}_{self.inlet_side}: prescribed velocity and Ta",
            "outlet": f"{axis_name[self.flow_axis]}_{self.outlet_side}: pressure-open, zero gradient",
            "lateral_sides": f"{axis_name[self.lateral_axis]}_low/high: impermeable free slip",
            "top": "z_high: pressure-open, zero gradient",
            "bottom_and_solids": "impermeable",
        }

    def apply_velocity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                       inlet_u: np.ndarray, inlet_v: np.ndarray,
                       solid: np.ndarray, *, after_projection: bool = False) -> None:
        # Inlet velocity profile is prescribed.
        inlet = self._plane(self.flow_axis, self.inlet_side)
        u[inlet] = inlet_u[inlet]
        v[inlet] = inlet_v[inlet]
        w[inlet] = 0.0
        # Before projection, initialize open faces with a zero-gradient value.
        # Afterwards, retain their corrected normal velocity while keeping only
        # tangential components zero-gradient; otherwise the pressure-open mass
        # flux calculated by the projection would be overwritten here.
        outlet = self._plane(self.flow_axis, self.outlet_side)
        outlet_adjacent = self._adjacent(self.flow_axis, self.outlet_side)
        normal_outlet = u if self.flow_axis == 2 else v
        for value in (u, v, w):
            if not after_projection or value is not normal_outlet:
                value[outlet] = value[outlet_adjacent]
        for value in (u, v):
            value[-1, :, :] = value[-2, :, :]
        if not after_projection:
            w[-1, :, :] = w[-2, :, :]
        # Two side surfaces are free slip: zero normal velocity and zero
        # normal gradient of tangential/vertical components.
        normal = v if self.lateral_axis == 1 else u
        tangential = u if self.lateral_axis == 1 else v
        for side in ("low", "high"):
            plane = self._plane(self.lateral_axis, side)
            adjacent = self._adjacent(self.lateral_axis, side)
            normal[plane] = 0.0
            tangential[plane] = tangential[adjacent]
            w[plane] = w[adjacent]
        u[solid] = v[solid] = w[solid] = 0.0

    def apply_temperature(self, temperature: np.ndarray, baseline_c: float,
                          fluid: np.ndarray) -> None:
        outlet = self._plane(self.flow_axis, self.outlet_side)
        temperature[outlet] = temperature[self._adjacent(
            self.flow_axis, self.outlet_side)]
        for side in ("low", "high"):
            plane = self._plane(self.lateral_axis, side)
            temperature[plane] = temperature[self._adjacent(
                self.lateral_axis, side)]
        temperature[-1, :, :] = temperature[-2, :, :]
        # Apply the prescribed inflow scalar last so edge/corner inlet cells do
        # not accidentally inherit an open/adiabatic neighboring condition.
        inlet = self._plane(self.flow_axis, self.inlet_side)
        temperature[inlet] = baseline_c
        temperature[~fluid] = baseline_c


class MassConsistentProjector:
    """Non-periodic finite-volume Poisson projection with open boundaries."""

    def __init__(self, solid: np.ndarray, dx: float, dy: float, dz: float,
                 maximum_iterations: int = 250, relative_tolerance: float = 1e-6,
                 boundary: FlowBoundaryConditions | None = None):
        self.solid = np.asarray(solid, dtype=bool)
        self.fluid = ~self.solid
        self.spacing = (dz, dy, dx)
        self.maximum_iterations = int(maximum_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.boundary = boundary
        self.index = np.full(self.solid.shape, -1, dtype=np.int64)
        self.index[self.fluid] = np.arange(self.fluid.sum())
        if self.fluid.sum() < 2:
            raise ValueError("microclimate domain must contain at least two fluid cells")

        diagonal = np.zeros(self.fluid.sum(), dtype=float)
        rows, cols, values = [], [], []
        for axis, spacing in enumerate(self.spacing):
            low, high = self._neighbor_slices(axis)
            connected = self.fluid[low] & self.fluid[high]
            cell = self.index[low][connected]
            neighbor = self.index[high][connected]
            coefficient = 1.0 / spacing**2
            diagonal[cell] += coefficient
            diagonal[neighbor] += coefficient
            rows.extend((cell, neighbor))
            cols.extend((neighbor, cell))
            values.extend((np.full(len(cell), -coefficient),
                           np.full(len(cell), -coefficient)))
        # Pressure-open outlet/top: p=0 at a half-cell boundary distance.
        if boundary is not None:
            for axis, side in boundary.open_faces:
                plane = boundary._plane(axis, side)
                cells = self.index[plane][self.fluid[plane]]
                diagonal[cells] += 2.0 / self.spacing[axis]**2
        ids = np.arange(len(diagonal))
        rows.append(ids); cols.append(ids); values.append(diagonal + 1e-12)
        self.matrix = sparse.coo_matrix(
            (np.concatenate(values),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(diagonal), len(diagonal))).tocsr()

    @staticmethod
    def _neighbor_slices(axis: int) -> tuple[tuple, tuple]:
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        return tuple(low), tuple(high)

    def _positive_faces(self, velocity: np.ndarray, axis: int) -> np.ndarray:
        face = np.zeros(self.solid.shape, dtype=float)
        low, high = self._neighbor_slices(axis)
        connected = self.fluid[low] & self.fluid[high]
        values = 0.5 * (velocity[low] + velocity[high])
        face_low = face[low]
        face_low[connected] = values[connected]
        return face

    def _boundary_fluxes(self, velocities: tuple[np.ndarray, np.ndarray, np.ndarray]
                         ) -> list[list[np.ndarray]]:
        fluxes: list[list[np.ndarray]] = []
        for axis, velocity in enumerate(velocities):
            shape = tuple(size for i, size in enumerate(self.solid.shape) if i != axis)
            low_flux, high_flux = np.zeros(shape), np.zeros(shape)
            if self.boundary is not None:
                # Prescribed inlet and initial open outlet normal flux.
                if axis == self.boundary.flow_axis:
                    for side, target in (("low", low_flux), ("high", high_flux)):
                        plane = self.boundary._plane(axis, side)
                        target[:] = np.where(self.fluid[plane], velocity[plane], 0.0)
                # Open top permits vertical outflow/inflow before correction.
                if axis == 0:
                    plane = self.boundary._plane(0, "high")
                    high_flux[:] = np.where(self.fluid[plane], velocity[plane], 0.0)
            fluxes.append([low_flux, high_flux])
        return fluxes

    def _face_divergence(self, faces: tuple[np.ndarray, np.ndarray, np.ndarray],
                         boundary_fluxes: list[list[np.ndarray]]) -> np.ndarray:
        result = np.zeros(self.solid.shape, dtype=float)
        for axis, (face, spacing) in enumerate(zip(faces, self.spacing)):
            low, high = self._neighbor_slices(axis)
            result[low] += face[low] / spacing
            result[high] -= face[low] / spacing
            low_plane = [slice(None)] * 3; low_plane[axis] = 0
            high_plane = [slice(None)] * 3; high_plane[axis] = -1
            result[tuple(low_plane)] -= boundary_fluxes[axis][0] / spacing
            result[tuple(high_plane)] += boundary_fluxes[axis][1] / spacing
        result[self.solid] = 0.0
        return result

    def project(self, u: np.ndarray, v: np.ndarray, w: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        u, v, w = (np.asarray(value, dtype=float).copy()
                   for value in (u, v, w))
        u[self.solid] = v[self.solid] = w[self.solid] = 0.0
        velocities_axis = (w, v, u)
        faces = tuple(self._positive_faces(value, axis)
                      for axis, value in enumerate(velocities_axis))
        boundary_fluxes = self._boundary_fluxes(velocities_axis)
        div = self._face_divergence(faces, boundary_fluxes)
        before = float(np.sqrt(np.mean(div[self.fluid] ** 2)))
        rhs = -div[self.fluid]
        if self.boundary is None:  # closed pure-Neumann compatibility mode
            rhs -= rhs.mean()
        try:
            pressure_vector, info = cg(
                self.matrix, rhs, rtol=self.relative_tolerance, atol=0.0,
                maxiter=self.maximum_iterations)
        except TypeError:
            pressure_vector, info = cg(
                self.matrix, rhs, tol=self.relative_tolerance,
                maxiter=self.maximum_iterations)
        if info < 0:
            raise RuntimeError("microclimate Poisson projection failed")
        pressure = np.zeros(self.solid.shape, dtype=float)
        pressure[self.fluid] = pressure_vector
        corrected = []
        for axis, (face, spacing) in enumerate(zip(faces, self.spacing)):
            low, high = self._neighbor_slices(axis)
            connected = self.fluid[low] & self.fluid[high]
            gradient = (pressure[high] - pressure[low]) / spacing
            face_corrected = np.zeros_like(face)
            target = face_corrected[low]
            target[connected] = face[low][connected] - gradient[connected]
            corrected.append(face_corrected)
        if self.boundary is not None:
            for axis, side in self.boundary.open_faces:
                plane = self.boundary._plane(axis, side)
                pressure_plane = pressure[plane]
                correction = 2.0 * pressure_plane / self.spacing[axis]
                if side == "high":
                    boundary_fluxes[axis][1] += correction
                else:
                    boundary_fluxes[axis][0] -= correction
        div_after = self._face_divergence(tuple(corrected), boundary_fluxes)
        after = float(np.sqrt(np.mean(div_after[self.fluid] ** 2)))

        reconstructed = []
        for axis, face in enumerate(corrected):
            value = np.zeros(self.solid.shape, dtype=float)
            low, high = self._neighbor_slices(axis)
            value[low] += 0.5 * face[low]
            value[high] += 0.5 * face[low]
            low_plane = [slice(None)] * 3; low_plane[axis] = 0
            high_plane = [slice(None)] * 3; high_plane[axis] = -1
            value[tuple(low_plane)] += 0.5 * boundary_fluxes[axis][0]
            value[tuple(high_plane)] += 0.5 * boundary_fluxes[axis][1]
            value[self.solid] = 0.0
            reconstructed.append(value)
        w_out, v_out, u_out = reconstructed
        return u_out, v_out, w_out, before, after


def project_velocity_fft(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                         solid: np.ndarray, dx: float, dy: float, dz: float,
                         passes: int = 3
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Backward-compatible test helper around the obstacle-aware projection."""
    projector = MassConsistentProjector(
        solid, dx, dy, dz, maximum_iterations=max(80, int(passes) * 80))
    return projector.project(u, v, w)


def _upwind_gradient(field: np.ndarray, velocity: np.ndarray,
                     spacing: float, axis: int) -> np.ndarray:
    lower, upper = _axis_neighbors(field, axis)
    backward = (field - lower) / spacing
    forward = (upper - field) / spacing
    return np.where(velocity >= 0, backward, forward)


def _axis_neighbors(field: np.ndarray, axis: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return no-wrap neighbors, using a zero-normal-gradient outer value."""
    lower = field.copy()
    upper = field.copy()
    low, high = MassConsistentProjector._neighbor_slices(axis)
    lower[high] = field[low]
    upper[low] = field[high]
    return lower, upper


def advance_temperature(temperature_c: np.ndarray, u: np.ndarray,
                        v: np.ndarray, w: np.ndarray,
                        source_kps: np.ndarray, fluid: np.ndarray,
                        baseline_c: float, dt_s: float, dx: float, dy: float,
                        dz: float, diffusivity: float, maximum_cfl: float,
                        maximum_anomaly_k: float,
                        boundary: FlowBoundaryConditions | None = None
                        ) -> tuple[np.ndarray, int]:
    """Explicit first-order Eulerian advection–diffusion with stable substeps."""
    speed_cfl = (np.max(np.abs(u)) / dx + np.max(np.abs(v)) / dy
                 + np.max(np.abs(w)) / dz)
    diffusion_cfl = 2.0 * diffusivity * (1 / dx**2 + 1 / dy**2 + 1 / dz**2)
    n_sub = max(1, int(math.ceil(dt_s * (speed_cfl + diffusion_cfl)
                                     / maximum_cfl)))
    sub_dt = dt_s / n_sub
    result = np.asarray(temperature_c, dtype=float).copy()
    for _ in range(n_sub):
        adv = (u * _upwind_gradient(result, u, dx, 2)
               + v * _upwind_gradient(result, v, dy, 1)
               + w * _upwind_gradient(result, w, dz, 0))
        x_lower, x_upper = _axis_neighbors(result, 2)
        y_lower, y_upper = _axis_neighbors(result, 1)
        z_lower, z_upper = _axis_neighbors(result, 0)
        lap = ((x_lower - 2 * result + x_upper) / dx**2
               + (y_lower - 2 * result + y_upper) / dy**2
               + (z_lower - 2 * result + z_upper) / dz**2)
        result += sub_dt * (-adv + diffusivity * lap + source_kps)
        if boundary is None:
            # Historical helper/test behavior (fixed horizontal/top values),
            # now evaluated without periodic wrap-around stencils.
            result[~fluid] = baseline_c
            result[:, :, (0, -1)] = baseline_c
            result[:, (0, -1), :] = baseline_c
            result[-1, :, :] = baseline_c
        else:
            boundary.apply_temperature(result, baseline_c, fluid)
        result = np.clip(result, baseline_c - maximum_anomaly_k,
                         baseline_c + maximum_anomaly_k)
    return result, n_sub


def build_grid(path_xyz: np.ndarray, ground_mesh: trimesh.Trimesh,
               config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    grid = config["grid"]
    dx = dy = float(grid["horizontal_spacing_m"])
    dz = float(grid["vertical_spacing_m"])
    buffer_m = float(grid["route_buffer_m"])
    x0, y0 = np.min(path_xyz[:, :2], axis=0) - buffer_m
    x1, y1 = np.max(path_xyz[:, :2], axis=0) + buffer_m
    ground_z = ground_mesh.vertices[:, 2]
    z0 = math.floor(float(np.min(ground_z)) / dz) * dz
    z1 = float(np.max(ground_z)) + float(grid["height_above_local_ground_m"])
    n_z = int(math.ceil((z1 - z0) / dz))
    z_axis = z0 + np.arange(n_z + 1, dtype=float) * dz

    def axes(spacing_xy: float):
        return (np.arange(x0, x1 + spacing_xy * 0.5, spacing_xy),
                np.arange(y0, y1 + spacing_xy * 0.5, spacing_xy),
                z_axis)
    x, y, z = axes(dx)
    maximum = int(grid["maximum_cells"])
    while len(x) * len(y) * len(z) > maximum:
        factor = math.sqrt(len(x) * len(y) * len(z) / maximum)
        dx = dy = dx * factor * 1.02
        x, y, z = axes(dx)
    if min(len(x), len(y), len(z)) < 3:
        raise ValueError("microclimate grid requires at least three cells per axis")
    return x, y, z, dx, dy, dz


def solid_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray,
               ground_mesh: trimesh.Trimesh,
               building_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Approximate terrain and building occupancy on cell centers."""
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    origins = np.column_stack((xy, np.full(len(xy),
                              float(ground_mesh.bounds[1, 2]) + 100.0)))
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    ground_height_flat = np.full(len(origins), np.nan)
    try:
        locations, ray_index, _triangle = ground_mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
        ground_height_flat[ray_index] = locations[:, 2]
    except Exception as exc:
        print(f"  WARNING: terrain-height ray casting failed ({exc}); using nearest vertices")
    missing = ~np.isfinite(ground_height_flat)
    if missing.any():
        tree = cKDTree(ground_mesh.vertices[:, :2])
        _distance, vertex_index = tree.query(xy[missing])
        ground_height_flat[missing] = ground_mesh.vertices[vertex_index, 2]
        print(f"  Terrain height: nearest-vertex fallback for {missing.sum():,} grid columns")
    ground_height = ground_height_flat.reshape(len(y), len(x))
    solid = z[:, None, None] <= ground_height[None, :, :]

    # Voxel centres are substantially cheaper than point-in-polyhedron testing
    # every grid cell against large campus meshes and preserve the actual STL
    # footprint/height at the diagnostic grid resolution.
    pitch = float(min(np.diff(x).mean(), np.diff(y).mean(), np.diff(z).mean()))
    try:
        points = building_mesh.voxelized(pitch).fill().points
        ix = np.rint((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
        iy = np.rint((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
        iz = np.rint((points[:, 2] - z[0]) / (z[1] - z[0])).astype(int)
        valid = ((ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
                 & (iz >= 0) & (iz < len(z)))
        solid[iz[valid], iy[valid], ix[valid]] = True
    except Exception as exc:
        print(f"  WARNING: building voxelization failed ({exc}); terrain mask only")
    return solid, ground_height


def source_cells(centroids: np.ndarray, normals: np.ndarray,
                 x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 solid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offset = 0.55 * max(x[1] - x[0], y[1] - y[0], z[1] - z[0])
    sample = centroids + normals * offset
    ix = np.rint((sample[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
    iy = np.rint((sample[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
    iz = np.rint((sample[:, 2] - z[0]) / (z[1] - z[0])).astype(int)
    valid = ((ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
             & (iz >= 0) & (iz < len(z)))
    valid_indices = np.where(valid)[0]
    fluid = ~solid[iz[valid], iy[valid], ix[valid]]
    valid_indices = valid_indices[fluid]
    return valid_indices, iz[valid_indices], iy[valid_indices], ix[valid_indices]


def background_velocity(shape: tuple[int, int, int], z: np.ndarray,
                        ground_height: np.ndarray, speed_ms: float,
                        config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = config["flow"]
    direction_to = (float(flow["wind_direction_from_deg"]) + 180.0) % 360.0
    theta = np.deg2rad(90.0 - direction_to)
    roughness = float(flow["roughness_length_m"])
    reference_height = float(flow["reference_wind_height_m"])
    local_height = np.maximum(z[:, None, None] - ground_height[None, :, :], 0.05)
    profile = (np.log((local_height + roughness) / roughness)
               / np.log((reference_height + roughness) / roughness))
    profile = np.clip(profile, 0.0, 1.5)
    speed = max(float(speed_ms), float(flow["minimum_background_speed_ms"])) * profile
    u = np.broadcast_to(speed * np.cos(theta), shape).copy()
    v = np.broadcast_to(speed * np.sin(theta), shape).copy()
    return u, v, np.zeros(shape, dtype=float)


def diagnostic_figure(out_dir: Path, x: np.ndarray, y: np.ndarray,
                      z: np.ndarray, air: np.ndarray, u: np.ndarray,
                      v: np.ndarray, fluid: np.ndarray, time_label: str) -> None:
    iz = int(np.argmin(np.abs(z - (z[0] + 2.0))))
    stride = max(1, int(max(len(x), len(y)) / 35))
    fig, ax = plt.subplots(figsize=(9, 7))
    data = np.where(fluid[iz], air[iz], np.nan)
    image = ax.pcolormesh(x, y, data, shading="nearest", cmap="coolwarm")
    ax.quiver(x[::stride], y[::stride], u[iz, ::stride, ::stride],
              v[iz, ::stride, ::stride], color="k", alpha=0.65, scale=70)
    fig.colorbar(image, ax=ax, label="Air temperature (°C)")
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)",
           title=f"TREC-Route diagnostic microclimate field · {time_label}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "microclimate_plan_view.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def export_paraview(out_dir: Path, config: dict,
                    context_meshes: dict | None = None) -> Path:
    """Write the solved field as a ParaView time series under paraview/.

    Reads the completed .npy/.npz products from ``out_dir`` so the export
    always matches what downstream stages will interpolate.  Each frame is
    a RectilinearGrid (.vtr) whose points are the cell-center axes, with
    air temperature, the (u, v, w) velocity vector, and the fluid mask;
    ``microclimate.pvd`` binds the frames to their local-time hours.
    Optional ``context_meshes`` (name -> trimesh) are written once as .vtp
    so the terrain/buildings can be displayed with the field.
    """
    from paraview_export import write_pvd, write_vtp, write_vtr

    axes = np.load(out_dir / AXES_FILE)
    x, y, z = axes["x_m"], axes["y_m"], axes["z_m"]
    time_hours = axes["time_hours"]
    air = np.load(out_dir / TEMPERATURE_FILE, mmap_mode="r")
    u, v, w = (np.load(out_dir / name, mmap_mode="r")
               for name in VELOCITY_FILES)
    fluid = np.load(out_dir / FLUID_MASK_FILE)
    stride = max(1, int(config["output"].get("paraview_time_stride", 1) or 1))
    target = out_dir / "paraview"
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for it in range(0, len(time_hours), stride):
        velocity = np.stack(
            [np.asarray(u[it]), np.asarray(v[it]), np.asarray(w[it])], axis=-1)
        name = f"microclimate_{it:03d}.vtr"
        write_vtr(target / name, x, y, z, point_data={
            "air_temperature_C": np.asarray(air[it]),
            "velocity_ms": velocity,
            "fluid_mask": fluid})
        entries.append((float(time_hours[it]), name))
    write_pvd(target / "microclimate.pvd", entries)
    for name, mesh in (context_meshes or {}).items():
        write_vtp(target / f"{name}.vtp", np.asarray(mesh.vertices),
                  np.asarray(mesh.faces))
    print(f"  ParaView export: {len(entries)} frame(s), stride {stride} "
          f"-> {target / 'microclimate.pvd'}")
    return target / "microclimate.pvd"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.horizontal_spacing_m is not None:
        config["grid"]["horizontal_spacing_m"] = args.horizontal_spacing_m
    if args.vertical_spacing_m is not None:
        config["grid"]["vertical_spacing_m"] = args.vertical_spacing_m
    if args.maximum_cells is not None:
        config["grid"]["maximum_cells"] = args.maximum_cells
    if args.no_diagnostic_figure:
        config["output"]["write_diagnostic_figure"] = False
    if args.no_paraview:
        config["output"]["write_paraview"] = False
    # Revalidate CLI overrides through the same centralized checks.
    config = _merge(DEFAULT_CONFIG, config)
    if config["grid"]["horizontal_spacing_m"] <= 0 \
            or config["grid"]["vertical_spacing_m"] <= 0:
        raise ValueError("microclimate grid spacing must be positive")
    if int(config["grid"]["maximum_cells"]) < 1000:
        raise ValueError("microclimate maximum_cells must be at least 1000")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facets_dir, mrt_dir = Path(args.facets_dir), Path(args.mrt_dir)
    required = [facets_dir / "facets.npz", facets_dir / "facet_T_matrix_K.npy",
                mrt_dir / "path_xyz.npy", mrt_dir / "times.csv",
                Path(args.buildings_stl), Path(args.ground_stl)]
    urban_dir = Path(args.urban_radiation_dir) if args.urban_radiation_dir else None
    if urban_dir is not None:
        required += [urban_dir / "urban_surface_mesh.npz",
                     urban_dir / "surface_temperature_K.npy",
                     urban_dir / "sensible_heat_flux_Wm2.npy",
                     urban_dir / "urban_radiation_metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing microclimate input(s): " + ", ".join(missing))

    if args.validate_only:
        metadata_path = out_dir / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(f"microclimate metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded = metadata.get("inputs", {})
        recorded_urban = "urban_surface_mesh" in recorded
        if recorded_urban and urban_dir is None:
            raise ValueError(
                "cached microclimate used full-surface urban radiation; "
                "provide --urban-radiation-dir to validate it")
        current = {"buildings_stl": file_identity(args.buildings_stl),
                   "ground_stl": file_identity(args.ground_stl),
                   "facets": file_identity(facets_dir / "facets.npz"),
                   "path_xyz": file_identity(mrt_dir / "path_xyz.npy"),
                   "times": file_identity(mrt_dir / "times.csv")}
        if urban_dir is not None and recorded_urban:
            current.update({
                "urban_surface_mesh": file_identity(urban_dir / "urban_surface_mesh.npz"),
                "urban_surface_temperature": file_identity(
                    urban_dir / "surface_temperature_K.npy"),
                "urban_sensible_heat": file_identity(
                    urban_dir / "sensible_heat_flux_Wm2.npy")})
        mismatches = []
        for name, identity in current.items():
            old = recorded.get(name, {})
            for key in ("size", "mtime_ns", "edge_sha256"):
                if old.get(key) != identity.get(key):
                    mismatches.append(f"{name}.{key}")
        if mismatches:
            raise ValueError("cached microclimate context mismatch: "
                             + ", ".join(mismatches))
        # The output section (figures, ParaView export) never changes the
        # solved field values, so it does not participate in cache identity.
        def _physics_config(candidate: dict) -> dict:
            return {key: value for key, value in (candidate or {}).items()
                    if key != "output"}
        if _physics_config(metadata.get("config")) != _physics_config(config):
            raise ValueError(
                "cached microclimate configuration mismatch; rerun optional step 4")
        # Opening through the shared reader checks axes and all array shapes.
        from microclimate_field import MicroclimateField
        field = MicroclimateField(out_dir)
        print(f"[microclimate_cache_valid] {field.describe()}")
        return 0

    # Metadata is the atomic completeness marker consumed by downstream
    # stages. Remove an older marker before any field array is overwritten so
    # an interrupted optional solve can never be mistaken for a valid cache.
    metadata_path = out_dir / METADATA_FILE
    if metadata_path.exists():
        metadata_path.unlink()

    path_xyz = np.load(mrt_dir / "path_xyz.npy")
    times = pd.read_csv(mrt_dir / "times.csv")
    timestamps = pd.to_datetime(times["time"])
    time_hours = np.array([value.hour + value.minute / 60 + value.second / 3600
                           for value in timestamps], dtype=float)
    if np.any(np.diff(time_hours) <= 0):
        raise ValueError("microclimate requires increasing within-day times.csv")
    dt_s = float(timestamps.diff().dt.total_seconds().dropna().median())
    if not 30 <= dt_s <= 7200:
        raise ValueError(f"invalid microclimate timestep {dt_s:g} s")
    baseline_ta = times["air_temp_C"].to_numpy(float)
    baseline_wind = times["wind_ms"].to_numpy(float)
    if not (np.isfinite(baseline_ta).all() and np.isfinite(baseline_wind).all()
            and np.all(baseline_wind >= 0)):
        raise ValueError("times.csv contains invalid air temperature or wind")

    if urban_dir is not None:
        facets = np.load(urban_dir / "urban_surface_mesh.npz", allow_pickle=True)
        centroids = facets["centroid"].astype(float)
        normals = facets["normal"].astype(float)
        areas = facets["area"].astype(float)
        surface_k = np.load(urban_dir / "surface_temperature_K.npy", mmap_mode="r")
        prescribed_sensible = np.load(
            urban_dir / "sensible_heat_flux_Wm2.npy", mmap_mode="r")
        surface_source_description = "05d full-domain radiation + 1-D conduction"
    else:
        facets = np.load(facets_dir / "facets.npz")
        centroids = facets["centroid"].astype(float)
        normals = facets["normal"].astype(float)
        areas = facets["area"].astype(float)
        surface_k = np.load(facets_dir / "facet_T_matrix_K.npy", mmap_mode="r")
        prescribed_sensible = None
        surface_source_description = "stage-05b route-visible 1-D conduction"
    if surface_k.shape != (len(times), len(centroids)):
        raise ValueError("facet temperature matrix does not match times/facets")
    if prescribed_sensible is not None and prescribed_sensible.shape != surface_k.shape:
        raise ValueError("urban sensible-heat matrix does not match surface temperatures")

    print("Loading diagnostic microclimate geometry...")
    ground_mesh = trimesh.load(args.ground_stl, force="mesh")
    building_mesh = trimesh.load(args.buildings_stl, force="mesh")
    x, y, z, dx, dy, dz = build_grid(path_xyz, ground_mesh, config)
    cells = len(x) * len(y) * len(z)
    print(f"  Grid: {len(x)} x {len(y)} x {len(z)} = {cells:,} cells; "
          f"spacing {dx:.2f} x {dy:.2f} x {dz:.2f} m")
    print(f"  Vertical domain: {z[0]:.2f} to {z[-1]:.2f} m; "
          f"configured top >= {config['grid']['height_above_local_ground_m']:.1f} m "
          "above highest terrain")
    solid, ground_height = solid_mask(x, y, z, ground_mesh, building_mesh)
    fluid = ~solid
    print(f"  Fluid cells: {fluid.sum():,}; solid cells: {solid.sum():,}")
    facet_index, source_z, source_y, source_x = source_cells(
        centroids, normals, x, y, z, solid)
    print(f"  Surface facets coupled to air cells: "
          f"{len(facet_index):,}/{len(centroids):,}")
    if len(facet_index) == 0:
        raise ValueError("no route-visible facets map to fluid microclimate cells")

    shape = (len(times), len(z), len(y), len(x))
    np.savez(out_dir / AXES_FILE, x_m=x, y_m=y, z_m=z,
             time_hours=time_hours)
    np.save(out_dir / FLUID_MASK_FILE, fluid)
    np.save(out_dir / GROUND_HEIGHT_FILE, ground_height.astype(np.float32))
    writers = [np.lib.format.open_memmap(out_dir / name, mode="w+",
                                        dtype=np.float32, shape=shape)
               for name in (TEMPERATURE_FILE,) + VELOCITY_FILES]
    temperature_writer, u_writer, v_writer, w_writer = writers

    rho = float(config["temperature"]["air_density_kgm3"])
    cp = float(config["temperature"]["specific_heat_JkgK"])
    diffusivity = float(config["temperature"]["eddy_diffusivity_m2s"])
    maximum_source = float(
        config["temperature"]["maximum_source_change_K_per_hour"]) / 3600.0
    max_anomaly = float(config["temperature"]["maximum_air_anomaly_K"])
    cell_volume = dx * dy * dz
    g, beta = 9.81, 1.0 / (273.15 + float(np.mean(baseline_ta)))
    buoyancy_efficiency = float(config["flow"]["buoyancy_velocity_efficiency"])
    maximum_speed = float(config["flow"]["maximum_speed_ms"])
    maximum_divergence_ratio = float(config["flow"]["maximum_divergence_ratio"])
    boundary = FlowBoundaryConditions(config["flow"]["wind_direction_from_deg"])
    inlet_override = config["flow"].get("inlet_velocity_ms")
    print("  Boundary conditions:")
    for name, description in boundary.describe().items():
        print(f"    {name}: {description}")
    print("  Inlet speed: " + (f"specified {float(inlet_override):.3f} m/s"
          if inlet_override is not None else "time-varying WeatherProvider wind"))
    projector = MassConsistentProjector(
        solid, dx, dy, dz,
        maximum_iterations=int(config["flow"]["poisson_maximum_iterations"]),
        relative_tolerance=float(config["flow"]["poisson_relative_tolerance"]),
        boundary=boundary)

    air = np.full(fluid.shape, baseline_ta[0], dtype=float)
    rows = []
    start = time.time()
    for it in range(len(times)):
        if it:
            air += baseline_ta[it] - baseline_ta[it - 1]
        inlet_speed = (float(inlet_override) if inlet_override is not None
                       else float(baseline_wind[it]))
        u, v, w = background_velocity(
            fluid.shape, z, ground_height, inlet_speed, config)
        u[solid] = v[solid] = w[solid] = 0.0
        inlet_u, inlet_v = u.copy(), v.copy()

        local_air_at_surface = air[source_z, source_y, source_x]
        local_background_speed = np.sqrt(
            u[source_z, source_y, source_x] ** 2
            + v[source_z, source_y, source_x] ** 2
            + w[source_z, source_y, source_x] ** 2)
        delta_t = (surface_k[it, facet_index].astype(float) - 273.15
                   - local_air_at_surface)
        h_conv = 5.7 + 3.8 * local_background_speed
        if prescribed_sensible is None:
            sensible_w = h_conv * delta_t * areas[facet_index]
        else:
            # 05d evaluated q_h consistently with its radiation-driven surface
            # balance. Radiation itself never heats air directly.
            sensible_w = (prescribed_sensible[it, facet_index].astype(float)
                          * areas[facet_index])
        source = np.zeros(fluid.shape, dtype=float)
        np.add.at(source, (source_z, source_y, source_x),
                  sensible_w / (rho * cp * cell_volume))
        source = np.clip(source, -maximum_source, maximum_source)

        positive = np.maximum(delta_t, 0.0)
        plume = (buoyancy_efficiency
                 * np.sqrt(2.0 * g * beta * positive * max(dz, 1.0)))
        weighted_plume = np.zeros(fluid.shape, dtype=float)
        plume_area = np.zeros(fluid.shape, dtype=float)
        np.add.at(weighted_plume, (source_z, source_y, source_x),
                  plume * areas[facet_index])
        np.add.at(plume_area, (source_z, source_y, source_x), areas[facet_index])
        active = plume_area > 0
        w[active] += weighted_plume[active] / plume_area[active]

        boundary.apply_velocity(u, v, w, inlet_u, inlet_v, solid)
        u, v, w, div_before, div_after = projector.project(u, v, w)
        speed = np.sqrt(u * u + v * v + w * w)
        # One domain-wide limiter preserves the divergence-free projection;
        # cell-wise clipping would create artificial convergence at clip edges.
        scale = min(1.0, maximum_speed / max(float(speed.max()), 1e-12))
        u *= scale; v *= scale; w *= scale
        inlet_u *= scale; inlet_v *= scale
        boundary.apply_velocity(
            u, v, w, inlet_u, inlet_v, solid, after_projection=True)
        speed = np.sqrt(u * u + v * v + w * w)
        divergence_ratio = div_after / max(div_before, 1e-12)
        if divergence_ratio > maximum_divergence_ratio and div_after > 1e-7:
            raise RuntimeError(
                f"mass-consistent projection failed at timestep {it + 1}: "
                f"divergence RMS retained {100 * divergence_ratio:.2f}% "
                f"(limit {100 * maximum_divergence_ratio:.2f}%)")

        air, n_sub = advance_temperature(
            air, u, v, w, source, fluid, baseline_ta[it], dt_s,
            dx, dy, dz, diffusivity,
            float(config["temperature"]["maximum_cfl"]), max_anomaly,
            boundary=boundary)
        if not (np.isfinite(air).all() and np.isfinite(u).all()
                and np.isfinite(v).all() and np.isfinite(w).all()):
            raise RuntimeError(f"non-finite microclimate field at timestep {it}")
        temperature_writer[it] = air.astype(np.float32)
        u_writer[it] = u.astype(np.float32)
        v_writer[it] = v.astype(np.float32)
        w_writer[it] = w.astype(np.float32)
        rows.append({
            "time": times["time"].iloc[it], "baseline_air_temperature_C": baseline_ta[it],
            "field_air_temperature_min_C": float(air[fluid].min()),
            "field_air_temperature_mean_C": float(air[fluid].mean()),
            "field_air_temperature_max_C": float(air[fluid].max()),
            "baseline_wind_ms": baseline_wind[it],
            "inlet_wind_ms": inlet_speed,
            "field_wind_mean_ms": float(speed[fluid].mean()),
            "field_wind_max_ms": float(speed[fluid].max()),
            "updraft_max_ms": float(w[fluid].max()),
            "sensible_heat_source_sum_W": float(sensible_w.sum()),
            "divergence_rms_before_s-1": div_before,
            "divergence_rms_after_s-1": div_after,
            "divergence_retention_ratio": divergence_ratio,
            "temperature_substeps": n_sub,
        })
        if (it + 1) % max(1, len(times) // 12) == 0 or it == len(times) - 1:
            print(f"  microclimate step {it + 1}/{len(times)} -- "
                  f"Ta {air[fluid].min():.1f}..{air[fluid].max():.1f} C, "
                  f"wind max {speed[fluid].max():.1f} m/s, {time.time()-start:.0f}s")

    for writer in writers:
        writer.flush()
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "microclimate_summary_by_time.csv", index=False)
    metadata = {
        "format_version": 1,
        "model": "TREC-Route diagnostic mass-consistent microclimate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "TREC-Route local projected metres; z vertical",
        "array_order": ["time", "z", "y", "x"],
        "shape": list(shape),
        "units": {"air_temperature_C": "degree_Celsius",
                  "velocity_u_ms": "m s-1 local +X",
                  "velocity_v_ms": "m s-1 local +Y",
                  "velocity_w_ms": "m s-1 upward",
                  "ground_height_m": "m in local vertical datum"},
        "humidity": "not solved; shared WeatherProvider RH remains active",
        "method": {"surfaces": surface_source_description,
                   "buoyancy": "sensible-flux plume injection",
                   "mass_consistency": "non-periodic obstacle-aware finite-volume elliptic Poisson projection",
                   "temperature": "non-periodic Eulerian first-order upwind advection-diffusion",
                   "boundary_conditions": boundary.describe()},
        "config": config,
        "inputs": {"buildings_stl": file_identity(args.buildings_stl),
                   "ground_stl": file_identity(args.ground_stl),
                   "facets": file_identity(facets_dir / "facets.npz"),
                   "facet_temperature": file_identity(
                       facets_dir / "facet_T_matrix_K.npy"),
                   "path_xyz": file_identity(mrt_dir / "path_xyz.npy"),
                   "times": file_identity(mrt_dir / "times.csv")},
        "diagnostics": {"maximum_divergence_retention_ratio": float(
                            summary["divergence_retention_ratio"].max()),
                        "temperature_range_C": [float(summary["field_air_temperature_min_C"].min()),
                                                float(summary["field_air_temperature_max_C"].max())],
                        "maximum_wind_speed_ms": float(
                            summary["field_wind_max_ms"].max())},
        "limitations": [
            "diagnostic mass-consistent model, not RANS or LES",
            "configured wind direction because legacy weather supplies speed only",
            "dominant-axis inlet/outlet approximates oblique wind on the fixed ENU grid",
            "constant eddy diffusivity; no prognostic turbulence closure",
            "humidity and vegetation drag are not transported",
        ],
    }
    if urban_dir is not None:
        metadata["inputs"].update({
            "urban_surface_mesh": file_identity(urban_dir / "urban_surface_mesh.npz"),
            "urban_surface_temperature": file_identity(
                urban_dir / "surface_temperature_K.npy"),
            "urban_sensible_heat": file_identity(
                urban_dir / "sensible_heat_flux_Wm2.npy")})
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2),
                                         encoding="utf-8")
    if config["output"].get("write_diagnostic_figure", True):
        diagnostic_figure(out_dir, x, y, z, air, u, v, fluid,
                          str(times["time"].iloc[-1]))
    if config["output"].get("write_paraview", True):
        export_paraview(out_dir, config, context_meshes={
            "context_ground": ground_mesh,
            "context_buildings": building_mesh})
    size_mb = sum(path.stat().st_size for path in out_dir.iterdir()
                  if path.is_file()) / 1e6
    print(f"[microclimate_result] cells={cells} times={len(times)} "
          f"size_mb={size_mb:.1f} output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
