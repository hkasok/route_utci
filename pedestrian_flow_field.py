#!/usr/bin/env python3
"""Shared reader/sampler for the stage-05e pedestrian-level potential-flow wind.

Stage 05e (pipeline step 3) writes a single steady, 2-D, terrain-referenced
pedestrian-level velocity field into ``run_output/<case>/pedestrian_wind/``.
This module is the only interpolation interface to that field, mirroring the
role ``microclimate_field.py`` plays for the legacy 3-D solver.

Field directory contract
------------------------
``potential_flow_metadata.json``
    Provenance, grid, wind direction, reference speed, solver and
    mass-conservation diagnostics, limitations.
``potential.npy``                 (ny, nx) velocity potential, NaN outside the
                                  solved region (solids and enclosed stagnant
                                  courtyards); velocities there are zero.
``velocity_u.npy``/``velocity_v.npy``/``velocity_speed.npy``/
``velocity_direction.npy``        (ny, nx) cell-centre fields at the saved
                                  reference wind speed. Direction is the
                                  compass bearing the flow is going TOWARD
                                  (deg clockwise from +Y/north).
``ground_z.npy``/``pedestrian_z.npy``
    (ny, nx) local terrain height and terrain-following pedestrian sampling
    height ``z_ground + h_ped``.
``x_coordinates.npy``/``y_coordinates.npy``
    1-D cell-centre axes in local projected metres.
``fluid_mask.npy``                (ny, nx) bool, True where the pedestrian
                                  layer is open air.
``cell_fluid_fraction.npy``       (ny, nx) cut-cell fluid volume fraction.
``face_open_fraction_x.npy``      (ny, nx+1) open fraction of x-normal faces.
``face_open_fraction_y.npy``      (ny+1, nx) open fraction of y-normal faces.

Linearity contract
------------------
The Laplace problem is linear in the inlet speed, so the saved field for
reference speed ``U_ref`` rescales exactly to any other speed ``U`` as
``u(U) = (U / U_ref) * u(U_ref)`` for the same wind direction.  Samplers
therefore accept a per-query ``wind_speed_ms`` and never require re-solving.

Interpolation is bilinear and fluid-aware: solid cells carry zero weight, and
a query surrounded entirely by solid cells falls back to the nearest fluid
cell so route points hugging a wall never blend with in-building zeros.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

METADATA_FILE = "potential_flow_metadata.json"
POTENTIAL_FILE = "potential.npy"
VELOCITY_U_FILE = "velocity_u.npy"
VELOCITY_V_FILE = "velocity_v.npy"
VELOCITY_SPEED_FILE = "velocity_speed.npy"
VELOCITY_DIRECTION_FILE = "velocity_direction.npy"
GROUND_Z_FILE = "ground_z.npy"
PEDESTRIAN_Z_FILE = "pedestrian_z.npy"
X_FILE = "x_coordinates.npy"
Y_FILE = "y_coordinates.npy"
FLUID_MASK_FILE = "fluid_mask.npy"
CELL_FRACTION_FILE = "cell_fluid_fraction.npy"
FACE_FRACTION_X_FILE = "face_open_fraction_x.npy"
FACE_FRACTION_Y_FILE = "face_open_fraction_y.npy"
# Two-direction basis (unit east and north free streams) written by 05e.
# Optional: a field solved before this existed simply has no basis and
# keeps working at its single solved direction.
BASIS_FILE = "direction_basis.npz"


def file_identity(path: str | Path) -> dict:
    """Size/mtime/edge-hash fingerprint (same scheme as the 05c solver)."""
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


class PedestrianFlowField:
    """Reader and fluid-aware bilinear sampler for a solved 05e field."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        metadata_path = self.directory / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"incomplete pedestrian-flow directory: {self.directory}; "
                f"expected {METADATA_FILE}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8"))
        self.x = np.load(self.directory / X_FILE).astype(float)
        self.y = np.load(self.directory / Y_FILE).astype(float)
        if (self.x.ndim != 1 or self.y.ndim != 1 or len(self.x) < 2
                or len(self.y) < 2 or np.any(np.diff(self.x) <= 0)
                or np.any(np.diff(self.y) <= 0)):
            raise ValueError("pedestrian-flow axes must be strictly increasing 1-D")
        self.u = np.load(self.directory / VELOCITY_U_FILE)
        self.v = np.load(self.directory / VELOCITY_V_FILE)
        self.speed = np.load(self.directory / VELOCITY_SPEED_FILE)
        self.ground_z = np.load(self.directory / GROUND_Z_FILE)
        self.pedestrian_z = np.load(self.directory / PEDESTRIAN_Z_FILE)
        self.fluid_mask = np.load(self.directory / FLUID_MASK_FILE).astype(bool)
        expected = (len(self.y), len(self.x))
        for name, array in (("velocity_u", self.u), ("velocity_v", self.v),
                            ("velocity_speed", self.speed),
                            ("ground_z", self.ground_z),
                            ("pedestrian_z", self.pedestrian_z),
                            ("fluid_mask", self.fluid_mask)):
            if array.shape != expected:
                raise ValueError(
                    f"pedestrian-flow {name} shape {array.shape} != {expected}")
        for name, array in (("velocity_u", self.u), ("velocity_v", self.v),
                            ("velocity_speed", self.speed)):
            if not np.isfinite(array).all():
                raise ValueError(f"pedestrian-flow {name} contains non-finite values")
        self.reference_speed_ms = float(self.metadata["reference_wind_speed_ms"])
        if not self.reference_speed_ms > 0:
            raise ValueError("pedestrian-flow reference wind speed must be positive")
        self.pedestrian_height_m = float(self.metadata["pedestrian_height_m"])
        self._fluid_tree = None
        self._fluid_index = None

        # Optional two-direction basis. Absent for a field solved before the
        # basis existed, which then simply keeps working at its one solved
        # direction -- the caller finds out via has_direction_basis().
        self.basis: dict[str, np.ndarray] | None = None
        basis_path = self.directory / BASIS_FILE
        if basis_path.is_file():
            archive = np.load(basis_path)
            required = ("u_east", "v_east", "u_north", "v_north")
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ValueError(
                    f"{basis_path} is missing basis arrays {missing}")
            self.basis = {name: archive[name].astype(float)
                          for name in required}
            for name, array in self.basis.items():
                if array.shape != expected:
                    raise ValueError(
                        f"direction-basis {name} shape {array.shape} != {expected}")
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"direction-basis {name} contains non-finite values")

    def has_direction_basis(self) -> bool:
        """Whether this field can be reconstructed at an arbitrary direction."""
        return self.basis is not None

    def basis_velocity(self, wind_direction_deg: float) -> tuple[np.ndarray,
                                                                 np.ndarray]:
        """Unit-speed (u, v) grids for any wind direction, by superposition.

        The free-stream direction enters the Laplace problem linearly and only
        through the right-hand side, and 05e pins the outlet side across the
        basis pair, so the matrix is identical for both components. Therefore

            u(theta) = cos(theta) u_east + sin(theta) u_north

        is not an interpolation between two solved cases -- it is the exact
        solution of the boundary-value problem for that direction.

        ``wind_direction_deg`` is the meteorological FROM bearing, matching
        ``--wind-direction-deg``; it is converted here to the direction the
        flow travels toward.
        """
        if self.basis is None:
            raise ValueError(
                f"{self.directory} has no direction basis; re-run stage 05e "
                "with --direction-basis on to reconstruct other directions")
        bearing = np.deg2rad(float(wind_direction_deg))
        # Meteorological FROM bearing -> unit vector the flow travels toward.
        east = -np.sin(bearing)
        north = -np.cos(bearing)
        u = east * self.basis["u_east"] + north * self.basis["u_north"]
        v = east * self.basis["v_east"] + north * self.basis["v_north"]
        return u, v

    def describe(self) -> str:
        return (f"pedestrian potential-flow field {len(self.x)}x{len(self.y)} "
                f"cells, wind from "
                f"{self.metadata.get('wind_direction_from_deg', '?')} deg, "
                f"reference {self.reference_speed_ms:g} m/s ({self.directory})")

    # -- interpolation ----------------------------------------------------
    def _corner_indices(self, x: np.ndarray, y: np.ndarray):
        dx = float(np.median(np.diff(self.x)))
        dy = float(np.median(np.diff(self.y)))
        fx = np.clip((x - self.x[0]) / dx, 0.0, len(self.x) - 1.0)
        fy = np.clip((y - self.y[0]) / dy, 0.0, len(self.y) - 1.0)
        i0 = np.clip(np.floor(fx).astype(int), 0, len(self.x) - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, len(self.y) - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        return i0, j0, tx, ty

    def _fluid_lookup(self):
        if self._fluid_tree is None:
            jj, ii = np.nonzero(self.fluid_mask)
            if len(ii) == 0:
                raise ValueError("pedestrian-flow field contains no fluid cells")
            self._fluid_tree = cKDTree(
                np.column_stack((self.x[ii], self.y[jj])))
            self._fluid_index = (jj, ii)
        return self._fluid_tree, self._fluid_index

    def _sample_arrays(self, x: np.ndarray, y: np.ndarray,
                       arrays: tuple[np.ndarray, ...]) -> list[np.ndarray]:
        """Fluid-aware bilinear sample of several (ny, nx) arrays at once."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("query x and y must be matching 1-D arrays")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("pedestrian-flow query coordinates must be finite")
        i0, j0, tx, ty = self._corner_indices(x, y)
        corner_i = (i0, i0 + 1, i0, i0 + 1)
        corner_j = (j0, j0, j0 + 1, j0 + 1)
        corner_w = ((1 - tx) * (1 - ty), tx * (1 - ty),
                    (1 - tx) * ty, tx * ty)
        # Solid cells carry zero interpolation weight so a near-wall query
        # never blends with in-building zeros (fluid-aware bilinear).
        weights = [w * self.fluid_mask[j, i]
                   for w, j, i in zip(corner_w, corner_j, corner_i)]
        total = np.sum(weights, axis=0)
        inside = total > 1e-12
        results = []
        for array in arrays:
            value = np.zeros(len(x), dtype=float)
            for w, j, i in zip(weights, corner_j, corner_i):
                value += w * np.asarray(array, dtype=float)[j, i]
            value[inside] /= total[inside]
            results.append(value)
        if not inside.all():
            tree, (jj, ii) = self._fluid_lookup()
            _dist, nearest = tree.query(
                np.column_stack((x[~inside], y[~inside])))
            for value, array in zip(results, arrays):
                value[~inside] = np.asarray(array, dtype=float)[
                    jj[nearest], ii[nearest]]
        return results

    def sample(self, x: np.ndarray, y: np.ndarray,
               wind_speed_ms: float | np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``u, v, speed, direction_deg`` at route/facet locations.

        ``wind_speed_ms`` (scalar or per-point) rescales the saved
        reference-speed solution through the exact linearity of the Laplace
        problem; ``None`` returns the field at its stored reference speed.
        Direction is recomputed from the interpolated vector (compass bearing
        the flow is going toward); speed is interpolated as a magnitude so
        opposing near-corner vectors do not cancel.
        """
        u, v, speed = self._sample_arrays(x, y, (self.u, self.v, self.speed))
        if wind_speed_ms is not None:
            scale = np.asarray(wind_speed_ms, dtype=float) / self.reference_speed_ms
            if not np.isfinite(scale).all() or np.any(scale < 0):
                raise ValueError("wind_speed_ms must be finite and non-negative")
            u, v, speed = u * scale, v * scale, speed * scale
        direction = (np.degrees(np.arctan2(u, v))) % 360.0
        return u, v, speed, direction

    def amplification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Dimensionless local speed factor ``speed / U_ref`` (unit-wind field)."""
        (speed,) = self._sample_arrays(x, y, (self.speed,))
        return speed / self.reference_speed_ms

    def sample_direction(self, x: np.ndarray, y: np.ndarray,
                         wind_direction_deg: float,
                         wind_speed_ms: float | np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
        """``u, v, speed, direction_deg`` for ANY wind direction and speed.

        Direction comes from the exact two-solve superposition; speed from the
        same linear rescaling the single-direction path uses. The two are
        independent: direction sets the shape of the field, speed sets its
        magnitude, and neither re-solves anything.

        Speed is recomputed as ``hypot(u, v)`` AFTER superposition rather than
        combined from the stored magnitudes. Speeds do not superpose -- adding
        two magnitudes would invent flow where opposing components should
        partly cancel.
        """
        grid_u, grid_v = self.basis_velocity(wind_direction_deg)
        u, v = self._sample_arrays(x, y, (grid_u, grid_v))
        if wind_speed_ms is not None:
            scale = np.asarray(wind_speed_ms, dtype=float)
            if not np.isfinite(scale).all() or np.any(scale < 0):
                raise ValueError("wind_speed_ms must be finite and non-negative")
            u, v = u * scale, v * scale
        speed = np.hypot(u, v)
        direction = np.degrees(np.arctan2(u, v)) % 360.0
        return u, v, speed, direction

    def pedestrian_elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain-following sampling height ``z_ground(x, y) + h_ped``."""
        (z_p,) = self._sample_arrays(x, y, (self.pedestrian_z,))
        return z_p


def facet_wind_speed_matrix(field: PedestrianFlowField, xy: np.ndarray,
                            wind_ms_series: np.ndarray,
                            maximum_amplification: float = 5.0) -> np.ndarray:
    """(n_times, n_points) local wind speed for the surface-energy stage.

    Exploits linearity: one solved unit-normalized field times the shared
    time-varying weather wind gives every facet its local speed without any
    re-solve.  Wall facet centroids sit on the obstacle boundary and roof
    centroids inside footprints; the fluid-aware sampler returns the nearest
    pedestrian-level open-air value there (documented approximation).

    ``maximum_amplification`` is a safety cap on the dimensionless local
    speed factor: inviscid potential flow produces unphysically large slip
    speeds inside nearly closed sub-cell gaps (a documented limitation), and
    those must not inflate the convective coefficient.  The overwhelming
    majority of cells lie far below the cap; it only trims corner/gap
    singularities.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("facet coordinates must be (n, >=2)")
    series = np.asarray(wind_ms_series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all() or np.any(series < 0):
        raise ValueError("wind series must be finite, non-negative and 1-D")
    if not maximum_amplification > 0:
        raise ValueError("maximum_amplification must be positive")
    amplification = np.minimum(
        field.amplification(points[:, 0], points[:, 1]),
        float(maximum_amplification))
    return series[:, None] * amplification[None, :]


def add_pedestrian_flow_argument(parser) -> None:
    """Attach the shared optional stage-05e field argument to a parser."""
    parser.add_argument(
        "--pedestrian-flow-dir", default=None,
        help=("Optional output of 05e_potential_flow.py (pipeline step 3). "
              "When supplied, the surface-energy convection wind becomes the "
              "locally varying pedestrian-level potential-flow speed scaled "
              "by the shared weather wind. When omitted, the historical "
              "spatially uniform weather/reference wind is used unchanged."))
