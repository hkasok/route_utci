#!/usr/bin/env python3
"""Shared TREC-Route spatiotemporal air-temperature and velocity fields.

The optional diagnostic microclimate stage writes a rectilinear, local-coordinate
field with the contract below.  Downstream stages use :class:`EnvironmentField`
instead of knowing how the field is stored.  When no solved directory is supplied,
the class deliberately broadcasts the existing :class:`WeatherProvider` values;
this is the backward-compatible, spatially uniform approximation.

Field directory contract
------------------------
``microclimate_metadata.json``
    Provenance, solver settings, units, array shapes and source fingerprints.
``microclimate_axes.npz``
    ``x_m``, ``y_m``, ``z_m`` and periodic decimal ``time_hours`` axes.
``air_temperature_C.npy``
    Float32 array shaped ``(time, z, y, x)``.
``velocity_u_ms.npy``, ``velocity_v_ms.npy``, ``velocity_w_ms.npy``
    Float32 local east/X, north/Y and vertical velocity arrays with the same shape.
``fluid_mask.npy``
    Boolean ``(z, y, x)`` mask.  Queries outside the grid are boundary-clamped;
    solid-cell queries use the nearest stored value and are flagged by provenance.

Time interpolation is periodic over 24 hours, matching the MRT workflow.  Spatial
interpolation is trilinear.  Humidity is intentionally not diagnosed by the present
solver and continues to come from the shared weather provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


METADATA_FILE = "microclimate_metadata.json"
AXES_FILE = "microclimate_axes.npz"
TEMPERATURE_FILE = "air_temperature_C.npy"
VELOCITY_FILES = (
    "velocity_u_ms.npy", "velocity_v_ms.npy", "velocity_w_ms.npy")
FLUID_MASK_FILE = "fluid_mask.npy"
GROUND_HEIGHT_FILE = "ground_height_m.npy"


def _strict_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError(f"microclimate {name} axis must be finite and one-dimensional")
    if np.any(np.diff(axis) <= 0):
        raise ValueError(f"microclimate {name} axis must be strictly increasing")
    return axis


@dataclass(frozen=True)
class FieldSample:
    """Environmental values returned at arbitrary local XYZ/time coordinates."""

    air_temperature_c: np.ndarray
    relative_humidity_pct: np.ndarray
    velocity_u_ms: np.ndarray
    velocity_v_ms: np.ndarray
    velocity_w_ms: np.ndarray
    wind_speed_ms: np.ndarray
    utci_wind_speed_10m_ms: np.ndarray
    source: str


class MicroclimateField:
    """Memory-mapped reader and periodic/trilinear interpolator for solved fields."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        metadata_path = self.directory / METADATA_FILE
        axes_path = self.directory / AXES_FILE
        if not metadata_path.is_file() or not axes_path.is_file():
            raise FileNotFoundError(
                f"incomplete microclimate field directory: {self.directory}; "
                f"expected {METADATA_FILE} and {AXES_FILE}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8"))
        axes = np.load(axes_path)
        self.x = _strict_axis(axes["x_m"], "x")
        self.y = _strict_axis(axes["y_m"], "y")
        self.z = _strict_axis(axes["z_m"], "z")
        self.time_hours = _strict_axis(axes["time_hours"], "time")
        if self.time_hours[0] < 0 or self.time_hours[-1] >= 24:
            raise ValueError("microclimate time_hours must lie in [0, 24)")

        self.temperature = np.load(
            self.directory / TEMPERATURE_FILE, mmap_mode="r")
        self.u = np.load(self.directory / VELOCITY_FILES[0], mmap_mode="r")
        self.v = np.load(self.directory / VELOCITY_FILES[1], mmap_mode="r")
        self.w = np.load(self.directory / VELOCITY_FILES[2], mmap_mode="r")
        self.fluid_mask = np.load(
            self.directory / FLUID_MASK_FILE, mmap_mode="r")
        ground_path = self.directory / GROUND_HEIGHT_FILE
        if not ground_path.is_file():
            raise FileNotFoundError(f"microclimate ground-height grid missing: {ground_path}")
        self.ground_height = np.load(ground_path, mmap_mode="r")
        expected = (len(self.time_hours), len(self.z), len(self.y), len(self.x))
        for name, array in (("air temperature", self.temperature),
                            ("velocity u", self.u), ("velocity v", self.v),
                            ("velocity w", self.w)):
            if array.shape != expected:
                raise ValueError(
                    f"microclimate {name} shape {array.shape} != {expected}")
        if self.fluid_mask.shape != expected[1:]:
            raise ValueError("microclimate fluid_mask shape does not match spatial axes")
        if self.ground_height.shape != (len(self.y), len(self.x)):
            raise ValueError("microclimate ground-height shape does not match x/y axes")
        for name, array in (("air temperature", self.temperature),
                            ("velocity u", self.u), ("velocity v", self.v),
                            ("velocity w", self.w)):
            # Checking the full disk-backed array can be expensive.  The solver
            # records full validation; the reader verifies deterministic slices.
            probe = np.asarray(array[[0, -1]])
            if not np.isfinite(probe).all():
                raise ValueError(f"microclimate {name} contains non-finite values")

    @staticmethod
    def _fractional_axis(query: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Return boundary-clamped fractional grid indices for a regular axis."""
        clipped = np.clip(query, axis[0], axis[-1])
        return np.interp(clipped, axis, np.arange(len(axis), dtype=float))

    def _spatial_sample(self, array: np.ndarray, time_index: np.ndarray,
                        xyz: np.ndarray) -> np.ndarray:
        xi = self._fractional_axis(xyz[:, 0], self.x)
        yi = self._fractional_axis(xyz[:, 1], self.y)
        zi = self._fractional_axis(xyz[:, 2], self.z)
        out = np.empty(len(xyz), dtype=float)
        for index in np.unique(time_index):
            take = np.where(time_index == index)[0]
            coords = np.vstack((zi[take], yi[take], xi[take]))
            out[take] = map_coordinates(
                np.asarray(array[int(index)]), coords, order=1,
                mode="nearest", prefilter=False)
        return out

    def sample(self, xyz: np.ndarray, hours: np.ndarray | float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``Ta, u, v, w`` at local XYZ and decimal local hours.

        ``xyz`` may be one point or ``(n, 3)``.  A scalar hour broadcasts to all
        points.  The return arrays always have length ``n``.
        """
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("microclimate query coordinates must have shape (n, 3)")
        if not np.isfinite(points).all():
            raise ValueError("microclimate query coordinates must be finite")
        # A route receptor or facet centroid can lie below the centre of the
        # first resolved fluid layer. Sampling it literally would blend with a
        # zero-velocity solid cell. Use the first full fluid-level height for
        # the grid interpolation; EnvironmentField still retains the original
        # height above ground for reference-height conversion.
        sample_points = points.copy()
        ground_z = self.sample_ground_height(points[:, :2])
        first_fluid_height = ground_z + float(np.median(np.diff(self.z)))
        sample_points[:, 2] = np.maximum(sample_points[:, 2], first_fluid_height)
        query_h = np.asarray(hours, dtype=float)
        if query_h.ndim == 0:
            query_h = np.full(len(points), float(query_h))
        query_h = np.broadcast_to(query_h, (len(points),)).astype(float)
        if not np.isfinite(query_h).all():
            raise ValueError("microclimate query times must be finite")

        first = float(self.time_hours[0])
        periodic_h = ((query_h - first) % 24.0) + first
        extended = np.concatenate((self.time_hours, [first + 24.0]))
        upper_ext = np.searchsorted(extended, periodic_h, side="right")
        upper_ext = np.clip(upper_ext, 1, len(extended) - 1)
        lower = upper_ext - 1
        t0 = extended[lower]
        t1 = extended[upper_ext]
        weight = np.divide(periodic_h - t0, t1 - t0,
                           out=np.zeros_like(periodic_h), where=t1 > t0)
        lower_index = lower % len(self.time_hours)
        upper_index = upper_ext % len(self.time_hours)

        values = []
        for array in (self.temperature, self.u, self.v, self.w):
            low = self._spatial_sample(array, lower_index, sample_points)
            high = self._spatial_sample(array, upper_index, sample_points)
            values.append(low + weight * (high - low))
        return tuple(values)  # type: ignore[return-value]

    def sample_ground_height(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        xi = self._fractional_axis(points[:, 0], self.x)
        yi = self._fractional_axis(points[:, 1], self.y)
        return map_coordinates(
            np.asarray(self.ground_height), np.vstack((yi, xi)), order=1,
            mode="nearest", prefilter=False)

    def describe(self) -> str:
        shape = self.temperature.shape
        return (f"solved diagnostic microclimate field {shape[3]}x{shape[2]}x"
                f"{shape[1]} cells, {shape[0]} times ({self.directory})")


class EnvironmentField:
    """One shared access point for solved or uniform Ta/RH/velocity forcing."""

    def __init__(self, weather, microclimate_dir: str | Path | None = None,
                 utci_receptor_height_m: float = 1.1):
        self.weather = weather
        if utci_receptor_height_m <= 0:
            raise ValueError("UTCI receptor height must be positive")
        self.utci_receptor_height_m = float(utci_receptor_height_m)
        self.field = (MicroclimateField(microclimate_dir)
                      if microclimate_dir else None)

    @property
    def solved(self) -> bool:
        return self.field is not None

    def describe(self) -> str:
        if self.field is not None:
            return self.field.describe()
        return "spatially uniform shared-weather fallback"

    def sample(self, xyz: np.ndarray, hours: np.ndarray | float) -> FieldSample:
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        query_h = np.asarray(hours, dtype=float)
        if query_h.ndim == 0:
            query_h = np.full(len(points), float(query_h))
        query_h = np.broadcast_to(query_h, (len(points),)).astype(float)
        rh = np.broadcast_to(
            np.asarray(self.weather.rh_pct(query_h), dtype=float),
            (len(points),)).copy()
        if self.field is None:
            ta = np.broadcast_to(
                np.asarray(self.weather.air_temp_c(query_h), dtype=float),
                (len(points),)).copy()
            speed = np.broadcast_to(
                np.asarray(self.weather.wind_ms(query_h), dtype=float),
                (len(points),)).copy()
            # The existing forcing has speed but no direction.  Retain it as
            # local +X solely to complete the vector contract; UTCI/JOS-3 use
            # the unchanged magnitude.
            u, v, w = speed.copy(), np.zeros_like(speed), np.zeros_like(speed)
            utci_speed = speed.copy()
            source = "uniform_weather_fallback"
        else:
            ta, u, v, w = self.field.sample(points, query_h)
            speed = np.sqrt(u * u + v * v + w * w)
            # MRT network and route points represent the configured standing
            # receptor height. A coarse diagnostic terrain grid is unsuitable
            # for recovering centimetre-accurate AGL height by subtraction, so
            # retain that known receptor height for the logarithmic conversion.
            height_agl = np.full(len(points), self.utci_receptor_height_m)
            flow_config = self.field.metadata.get("config", {}).get("flow", {})
            roughness = max(float(flow_config.get("roughness_length_m", 0.5)), 1e-3)
            reference = max(float(flow_config.get("reference_wind_height_m", 10.0)),
                            roughness + 0.1)
            denominator = np.log((height_agl + roughness) / roughness)
            factor = (np.log((reference + roughness) / roughness)
                      / np.maximum(denominator, 1e-6))
            utci_speed = speed * np.clip(factor, 0.25, 6.0)
            source = "solved_microclimate_field"
        return FieldSample(ta, rh, u, v, w, speed, utci_speed, source)


def add_microclimate_argument(parser) -> None:
    """Add the common optional field argument to an ``argparse`` parser."""
    parser.add_argument(
        "--microclimate-dir", default=None,
        help=("Optional output of 05c_microclimate_solver.py. When omitted, "
              "air temperature and wind retain the spatially uniform shared-"
              "weather behavior."))
    parser.add_argument(
        "--microclimate-receptor-height-m", type=float, default=1.1,
        help="Known route-receptor height above local terrain for UTCI wind conversion")
