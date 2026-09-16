#!/usr/bin/env python3
"""Stage 05e -- pedestrian-level 2-D potential-flow wind (pipeline step 3).

Terrain-referenced, mass-conserving 2-D pedestrian-level potential-flow model
with embedded (cut-cell) building boundaries.  It runs BEFORE the MRT /
radiation / surface-energy step and produces the only step-3 product: a
horizontally resolved velocity field on the terrain-following pedestrian
surface

    z_p(x, y) = z_ground(x, y) + h_ped .

Governing equation (fluid domain, plan view):

    u = grad(phi),   div(u) = 0   =>   Laplacian(phi) = 0

with no-penetration on embedded building boundaries

    d(phi)/dn = 0   =>   u . n = 0 .

Discretization: conservative cut-cell finite volume on a regular Cartesian
grid.  Every cell stores a fluid volume fraction ``alpha`` (sub-sampled from
the true 3-D building geometry intersected with the pedestrian surface) and
every face an open fraction ``beta`` sampled ON the geometric face line, so
the building boundary stays geometrically distinct from the Cartesian cell
boundary and 1-2 m sidewalks beside walls are not erased by the mesh spacing.
Per fluid control volume the solver enforces exact discrete conservation

    sum_faces beta_f * A_f * (grad(phi) . n)_f = 0 .

Boundary conditions: the upstream and lateral outer boundaries carry the
prescribed background normal flux ``(U d . n) beta_f A_f`` (an exact far-field
uniform-flow Neumann condition); the single dominant downwind boundary is a
far-field Dirichlet outlet ``phi = U d . x`` that pins the potential and
absorbs the global blockage imbalance; embedded building faces are closed
(``beta = 0`` => ``u . n = 0``).  A uniform free stream satisfies this
discrete system exactly (verify_potential_flow.py TEST A).

Linearity in SPEED: the system is solved once at unit inlet speed and scaled to
the reference wind afterwards; downstream consumers rescale it to any other wind
speed via ``u(U) = U/U_ref * u(U_ref)`` (see pedestrian_flow_field.py).
Multiplying both components by one positive number cannot rotate a vector, so
that rescaling correctly leaves the direction untouched.

Linearity in DIRECTION: a change of wind direction is NOT a rescale, but it is
still linear. The free-stream direction ``d`` enters the discrete system only
through the right-hand side -- ``phi_b = d . (x_face - centre)`` on Dirichlet
faces and ``B = (d . n) beta_f A_f`` on Neumann faces -- while the matrix is
built from the cut-cell fractions alone. With ``--direction-basis on`` (default)
the stage therefore solves TWO extra unit problems, for east and north free
streams, with the outlet side PINNED to the configured run's outlet so that both
carry identical boundary-condition placement. Any direction then follows exactly:

    u(theta) = cos(theta) u_east + sin(theta) u_north

Two components, not four: the southerly solution is exactly minus the northerly
one, so a four-direction basis would be redundant. Verified against direct
oblique solves to 1e-9 relative on a real 220k-cell case (TEST J).

Pinning the outlet is what makes this exact rather than approximate. A
standalone solve at a different direction picks its own downwind outlet, so it
differs slightly near the outer boundary; that difference is a near-boundary
discretisation artifact which shrinks with --route-buffer-m, and it is recorded
in the metadata rather than hidden.

WHAT THIS MODEL IS NOT
----------------------
Potential flow represents blockage and corner acceleration only.  It does NOT
resolve viscous boundary layers, wall shear, turbulence, flow separation,
building wakes, vortex shedding or recirculation: behind a building the
solution recovers smoothly and downstream sheltering is therefore
underestimated.  No heat equation, buoyancy, or transport of any scalar is
solved here -- step 3 is flow only.  Vegetation aerodynamic drag is ignored
in this first implementation (Option A); tree crowns are never treated as
solid obstacles.

Run:
    python3 05e_potential_flow.py \
        --buildings-stl ... --ground-stl ... \
        --polylines-pkl input/<case>/routes/route_polylines.pkl \
        --output-dir run_output/<case>/pedestrian_wind \
        --wind-direction-deg 270 --grid-spacing 2.0 --pedestrian-height 1.1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import json
import pickle
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree
import trimesh

from pedestrian_flow_field import (
    CELL_FRACTION_FILE, FACE_FRACTION_X_FILE, FACE_FRACTION_Y_FILE,
    FLUID_MASK_FILE, GROUND_Z_FILE, METADATA_FILE, PEDESTRIAN_Z_FILE,
    POTENTIAL_FILE, VELOCITY_DIRECTION_FILE, VELOCITY_SPEED_FILE,
    VELOCITY_U_FILE, VELOCITY_V_FILE, X_FILE, Y_FILE, BASIS_FILE,
    file_identity)
from thermal_common import get_intersector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pedestrian-level 2-D potential-flow wind field (step 3)")
    parser.add_argument("--buildings-stl", required=True)
    parser.add_argument("--ground-stl", required=True)
    parser.add_argument("--polylines-pkl", required=True,
                        help="Case route polylines (domain extent source)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-spacing", type=float, default=2.0,
                        help="Cartesian cell size dx = dy, metres (default 2)")
    parser.add_argument("--route-buffer-m", type=float, default=100.0,
                        help="Domain margin around the route bounding box")
    parser.add_argument("--pedestrian-height", type=float, default=1.1,
                        help="Height of the sampling surface above LOCAL "
                             "ground, metres. Default 1.1 m = the project "
                             "Z_HEIGHT pedestrian receptor convention.")
    parser.add_argument("--wind-direction-deg", type=float, default=270.0,
                        help="Meteorological direction the wind blows FROM, "
                             "degrees clockwise from north (270 = westerly)")
    parser.add_argument("--wind-speed-ms", type=float, default=None,
                        help="Reference wind speed the saved field is scaled "
                             "to. Default: mean weather-CSV wind, else 1 m/s. "
                             "Downstream stages rescale linearly, so this "
                             "never forces a re-solve.")
    parser.add_argument("--weather-csv", default=None,
                        help="Optional weather CSV; its mean wind_ms becomes "
                             "the recorded reference speed")
    parser.add_argument("--face-subsamples", type=int, default=4,
                        help="Geometry sub-samples per cell edge for the "
                             "cut-cell alpha/beta fractions (default 4)")
    parser.add_argument("--maximum-cells", type=int, default=600000,
                        help="Auto-coarsen the grid above this cell count")
    parser.add_argument("--solver-tolerance", type=float, default=1e-10,
                        help="Relative CG tolerance for the Laplace solve")
    parser.add_argument("--minimum-fluid-fraction", type=float, default=0.01,
                        help="Cells with less open volume are solid")
    parser.add_argument("--maximum-relative-imbalance", type=float, default=1e-3,
                        help="Reject the solve when the global inlet/outlet "
                             "mass imbalance exceeds this fraction "
                             "(default 1e-3 = 0.1%%)")
    parser.add_argument("--occupancy-batch", type=int, default=200000,
                        help="Ray-cast batch size for geometry sampling")
    parser.add_argument("--project-crs", default=None,
                        help="Recorded in metadata only (case project CRS)")
    parser.add_argument("--direction-basis", choices=["on", "off"], default="on",
                        help="Also solve the two-direction basis (east and "
                             "north unit free streams) so downstream consumers "
                             "can reconstruct ANY wind direction exactly by "
                             "superposition, without re-solving. Two solves "
                             "span the whole circle because the free-stream "
                             "direction enters the system linearly and only "
                             "through the right-hand side; a four-direction "
                             "basis would be exactly redundant. Costs one "
                             "extra Laplace solve.")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip the diagnostic figures")
    parser.add_argument("--no-paraview", action="store_true",
                        help="Skip the ParaView (.vtr/.vtp) field export")
    return parser.parse_args()


def load_route_extent(polylines_pkl: str) -> tuple[np.ndarray, np.ndarray]:
    with open(polylines_pkl, "rb") as stream:
        data = pickle.load(stream)
    polylines = data["polylines"] if isinstance(data, dict) else data
    points = np.vstack([np.asarray(poly, dtype=float)[:, :2]
                        for poly in polylines])
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("route polylines contain no finite 2-D extent")
    return points.min(axis=0), points.max(axis=0)


def build_axes(low: np.ndarray, high: np.ndarray, buffer_m: float,
               spacing: float, maximum_cells: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if spacing <= 0:
        raise ValueError("grid spacing must be positive")
    if buffer_m < 0:
        raise ValueError("route buffer must be non-negative")
    x0, y0 = low - buffer_m
    x1, y1 = high + buffer_m

    def axes(dx: float):
        nx = max(3, int(math.ceil((x1 - x0) / dx)))
        ny = max(3, int(math.ceil((y1 - y0) / dx)))
        x_edges = x0 + np.arange(nx + 1, dtype=float) * dx
        y_edges = y0 + np.arange(ny + 1, dtype=float) * dx
        return x_edges, y_edges

    dx = float(spacing)
    x_edges, y_edges = axes(dx)
    while (len(x_edges) - 1) * (len(y_edges) - 1) > int(maximum_cells):
        factor = math.sqrt((len(x_edges) - 1) * (len(y_edges) - 1)
                           / int(maximum_cells))
        dx *= factor * 1.02
        x_edges, y_edges = axes(dx)
    if dx != float(spacing):
        print(f"  NOTE: grid coarsened to dx = {dx:.3f} m to respect "
              f"--maximum-cells {maximum_cells}")
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_edges, y_edges, x_centres, y_centres, dx


def ground_height_grid(ground_mesh: trimesh.Trimesh, x: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
    """Terrain height at cell centres via vertical ray casting (05c scheme)."""
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    top = float(ground_mesh.bounds[1, 2]) + 100.0
    origins = np.column_stack((xy, np.full(len(xy), top)))
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    heights = np.full(len(origins), np.nan)
    try:
        locations, ray_index, _tri = ground_mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
        heights[ray_index] = locations[:, 2]
    except Exception as exc:
        print(f"  WARNING: terrain ray casting failed ({exc}); "
              "using nearest vertices")
    missing = ~np.isfinite(heights)
    if missing.any():
        tree = cKDTree(ground_mesh.vertices[:, :2])
        _dist, vertex_index = tree.query(xy[missing])
        heights[missing] = ground_mesh.vertices[vertex_index, 2]
        print(f"  Terrain height: nearest-vertex fallback for "
              f"{missing.sum():,} columns")
    return heights.reshape(len(y), len(x))


def interpolate_ground(ground: np.ndarray, x: np.ndarray, y: np.ndarray,
                       qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """Bilinear terrain interpolation with edge clamping."""
    fx = np.clip(np.interp(qx, x, np.arange(len(x), dtype=float)),
                 0, len(x) - 1)
    fy = np.clip(np.interp(qy, y, np.arange(len(y), dtype=float)),
                 0, len(y) - 1)
    i0 = np.clip(np.floor(fx).astype(int), 0, len(x) - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, len(y) - 2)
    tx, ty = fx - i0, fy - j0
    return ((1 - tx) * (1 - ty) * ground[j0, i0]
            + tx * (1 - ty) * ground[j0, i0 + 1]
            + (1 - tx) * ty * ground[j0 + 1, i0]
            + tx * ty * ground[j0 + 1, i0 + 1])


class PedestrianOccupancy:
    """Point-wise solid test at the terrain-following pedestrian surface.

    A location is solid where the 3-D building geometry occupies
    ``z = z_ground(x, y) + h_ped``: a downward vertical ray is cast through
    the building mesh and the crossing count strictly above the pedestrian
    height decides interiority by parity.  This keeps arcades/passages open
    (roof + underside = even crossings) while extruded building bodies read
    as solid, using the actual STL rather than any coarse-grid occupancy.
    """

    def __init__(self, building_mesh: trimesh.Trimesh,
                 ground: np.ndarray, x: np.ndarray, y: np.ndarray,
                 h_ped: float, batch: int):
        self.mesh = building_mesh
        self.intersector = get_intersector(building_mesh, quiet=True)
        self.ground = ground
        self.x, self.y = x, y
        self.h_ped = float(h_ped)
        self.batch = max(1, int(batch))
        pad = 1.0
        self.bbox_low = building_mesh.bounds[0, :2] - pad
        self.bbox_high = building_mesh.bounds[1, :2] + pad
        self.z_top = float(building_mesh.bounds[1, 2]) + 5.0

    def query(self, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        occupied = np.zeros(len(qx), dtype=bool)
        candidate = ((qx >= self.bbox_low[0]) & (qx <= self.bbox_high[0])
                     & (qy >= self.bbox_low[1]) & (qy <= self.bbox_high[1]))
        index = np.where(candidate)[0]
        z_p = (interpolate_ground(self.ground, self.x, self.y,
                                  qx[index], qy[index]) + self.h_ped)
        n_batches = max(1, int(math.ceil(len(index) / self.batch)))
        for number, start in enumerate(range(0, max(len(index), 1),
                                             self.batch)):
            take = index[start:start + self.batch]
            if len(take):
                z_take = z_p[start:start + self.batch]
                origins = np.column_stack(
                    (qx[take], qy[take], np.full(len(take), self.z_top)))
                directions = np.tile([0.0, 0.0, -1.0], (len(take), 1))
                locations, ray_index, _tri = (
                    self.intersector.intersects_location(
                        origins, directions, multiple_hits=True))
                counts = np.zeros(len(take), dtype=np.int64)
                if len(ray_index):
                    above = locations[:, 2] > z_take[ray_index] + 1e-6
                    np.add.at(counts, ray_index[above], 1)
                occupied[take] = (counts % 2) == 1
            print(f"  cut-cell occupancy batch {number + 1}/{n_batches}",
                  flush=True)
        return occupied


def cut_cell_fractions(occupancy: PedestrianOccupancy, x_edges: np.ndarray,
                       y_edges: np.ndarray, dx: float, n_sub: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """alpha (cell fluid fraction), beta_x/beta_y (face open fractions).

    ``alpha`` is sub-sampled on an ``n_sub x n_sub`` lattice inside every
    cell; ``beta`` is sampled ON each geometric face line (with a fixed
    sub-millimetre offset so rays never graze grid-aligned walls), which is
    what keeps the embedded boundary sharp rather than stair-stepped.
    Also returns the fine open/solid lattice for true-outline plotting.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    offsets = (np.arange(n_sub, dtype=float) + 0.5) / n_sub * dx
    eps = 3.7e-4  # keeps face samples off exactly grid-aligned wall planes

    fine_x = (x_edges[:-1, None] + offsets[None, :]).ravel()
    fine_y = (y_edges[:-1, None] + offsets[None, :]).ravel()

    qx, qy = np.meshgrid(fine_x, fine_y, indexing="xy")
    fine_solid = occupancy.query(qx.ravel(), qy.ravel()).reshape(
        len(fine_y), len(fine_x))
    alpha = 1.0 - fine_solid.reshape(ny, n_sub, nx, n_sub).mean(axis=(1, 3))

    face_x_x = np.repeat(x_edges + eps, ny * n_sub)
    face_x_y = np.tile((y_edges[:-1, None] + offsets[None, :]).ravel(), nx + 1)
    solid_fx = occupancy.query(face_x_x, face_x_y).reshape(
        nx + 1, ny, n_sub)
    beta_x = 1.0 - solid_fx.mean(axis=2).T          # (ny, nx+1)

    face_y_y = np.repeat(y_edges + eps, nx * n_sub)
    face_y_x = np.tile((x_edges[:-1, None] + offsets[None, :]).ravel(), ny + 1)
    solid_fy = occupancy.query(face_y_x, face_y_y).reshape(
        ny + 1, nx, n_sub)
    beta_y = 1.0 - solid_fy.mean(axis=2)            # (ny+1, nx)
    return alpha, beta_x, beta_y, fine_solid


def flow_direction_vector(direction_from_deg: float) -> np.ndarray:
    """Unit vector the flow travels TOWARD in local ENU (x east, y north)."""
    direction_to = (float(direction_from_deg) + 180.0) % 360.0
    theta = np.deg2rad(90.0 - direction_to)
    return np.array([np.cos(theta), np.sin(theta)])


def choose_outlet_side(direction: np.ndarray) -> str:
    normals = {"x_low": (-1.0, 0.0), "x_high": (1.0, 0.0),
               "y_low": (0.0, -1.0), "y_high": (0.0, 1.0)}
    return max(normals, key=lambda side: float(np.dot(direction, normals[side])))


def solve_unit_potential(alpha: np.ndarray, beta_x: np.ndarray,
                         beta_y: np.ndarray, x_edges: np.ndarray,
                         y_edges: np.ndarray, dx: float,
                         direction: np.ndarray, outlet_side: str,
                         minimum_fluid_fraction: float,
                         tolerance: float) -> dict:
    """Cut-cell finite-volume Laplace solve at unit inlet speed.

    Returns cell potential, face normal velocities, active/stagnant masks and
    solver diagnostics.  Face conductances use the sampled open fractions;
    interior fluid regions with no open connection to the outer boundary
    (enclosed courtyards) are held stagnant at zero velocity instead of
    entering the (would-be singular) system.

    Well-posedness across disconnected regions: only the fluid component(s)
    reaching the downwind outlet can carry the prescribed background flux out
    of the domain.  A component that touches the outer boundary somewhere else
    (a dead-end pocket opening onto the inlet or a lateral side) would form a
    pure-Neumann block whose prescribed inflow has nowhere to go -- an exactly
    singular, inconsistent system.  Such components therefore take the
    far-field Dirichlet potential ``phi = U d . x`` on every open outer face
    they own, which is the same far-field condition used at the outlet and
    leaves each block symmetric positive definite.
    """
    ny, nx = alpha.shape
    fluid = alpha >= float(minimum_fluid_fraction)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    open_face_x = np.zeros_like(beta_x)
    open_face_x[:, 1:-1] = np.where(
        fluid[:, :-1] & fluid[:, 1:], beta_x[:, 1:-1], 0.0)
    open_face_x[:, 0] = np.where(fluid[:, 0], beta_x[:, 0], 0.0)
    open_face_x[:, -1] = np.where(fluid[:, -1], beta_x[:, -1], 0.0)
    open_face_y = np.zeros_like(beta_y)
    open_face_y[1:-1, :] = np.where(
        fluid[:-1, :] & fluid[1:, :], beta_y[1:-1, :], 0.0)
    open_face_y[0, :] = np.where(fluid[0, :], beta_y[0, :], 0.0)
    open_face_y[-1, :] = np.where(fluid[-1, :], beta_y[-1, :], 0.0)

    # Connected components over interior open faces; components that touch an
    # open outer-boundary face participate in the solve, the rest stagnate.
    cell_id = np.arange(nx * ny).reshape(ny, nx)
    conn_x = open_face_x[:, 1:-1] > 0
    conn_y = open_face_y[1:-1, :] > 0
    rows = np.concatenate([cell_id[:, :-1][conn_x].ravel(),
                           cell_id[:-1, :][conn_y].ravel()])
    cols = np.concatenate([cell_id[:, 1:][conn_x].ravel(),
                           cell_id[1:, :][conn_y].ravel()])
    graph = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(nx * ny, nx * ny))
    _n_comp, labels = connected_components(graph, directed=False)
    labels = labels.reshape(ny, nx)
    boundary_open = np.zeros((ny, nx), dtype=bool)
    boundary_open[:, 0] |= open_face_x[:, 0] > 0
    boundary_open[:, -1] |= open_face_x[:, -1] > 0
    boundary_open[0, :] |= open_face_y[0, :] > 0
    boundary_open[-1, :] |= open_face_y[-1, :] > 0
    reachable_labels = np.unique(labels[fluid & boundary_open])
    active = fluid & np.isin(labels, reachable_labels)
    stagnant = fluid & ~active
    if not active.any():
        raise ValueError("potential-flow domain has no fluid connected to "
                         "the outer boundary")
    # Components that reach the outlet keep the prescribed-flux inlet/lateral
    # condition; the remaining dead-end components are anchored by the
    # far-field potential on all of their open outer faces (see docstring).
    outlet_touch = np.zeros((ny, nx), dtype=bool)
    if outlet_side == "x_low":
        outlet_touch[:, 0] = open_face_x[:, 0] > 0
    elif outlet_side == "x_high":
        outlet_touch[:, -1] = open_face_x[:, -1] > 0
    elif outlet_side == "y_low":
        outlet_touch[0, :] = open_face_y[0, :] > 0
    else:
        outlet_touch[-1, :] = open_face_y[-1, :] > 0
    outlet_labels = np.unique(labels[fluid & outlet_touch])
    far_field = active & ~np.isin(labels, outlet_labels)
    if far_field.any():
        print(f"  {int(far_field.sum()):,} fluid cells lie in dead-end "
              "regions that cannot reach the outlet; they take the far-field "
              "Dirichlet potential on their open outer faces")

    index = np.full((ny, nx), -1, dtype=np.int64)
    index[active] = np.arange(int(active.sum()))
    n_active = int(active.sum())
    diag = np.zeros(n_active)
    rows_a, cols_a, vals_a = [], [], []
    rhs = np.zeros(n_active)
    # Face area per unit depth equals dx for both orientations (dy == dx).
    g_scale = 1.0  # beta * A_f / d = beta * dx / dx

    def couple(cell, neighbour, conductance):
        diag[cell] += conductance
        diag[neighbour] += conductance
        rows_a.extend((cell, neighbour))
        cols_a.extend((neighbour, cell))
        vals_a.extend((-conductance, -conductance))

    # interior x faces
    both = active[:, :-1] & active[:, 1:]
    face_open = open_face_x[:, 1:-1]
    use = both & (face_open > 0)
    couple(index[:, :-1][use], index[:, 1:][use], face_open[use] * g_scale)
    # interior y faces
    both = active[:-1, :] & active[1:, :]
    face_open = open_face_y[1:-1, :]
    use = both & (face_open > 0)
    couple(index[:-1, :][use], index[1:, :][use], face_open[use] * g_scale)

    # outer-boundary faces (unit inlet speed)
    sides = {
        "x_low": {"normal": (-1.0, 0.0), "cells": (slice(None), 0),
                  "beta": open_face_x[:, 0],
                  "face_pos": lambda: np.column_stack(
                      (np.full(ny, x_edges[0]), y_centres))},
        "x_high": {"normal": (1.0, 0.0), "cells": (slice(None), -1),
                   "beta": open_face_x[:, -1],
                   "face_pos": lambda: np.column_stack(
                       (np.full(ny, x_edges[-1]), y_centres))},
        "y_low": {"normal": (0.0, -1.0), "cells": (0, slice(None)),
                  "beta": open_face_y[0, :],
                  "face_pos": lambda: np.column_stack(
                      (x_centres, np.full(nx, y_edges[0])))},
        "y_high": {"normal": (0.0, 1.0), "cells": (-1, slice(None)),
                   "beta": open_face_y[-1, :],
                   "face_pos": lambda: np.column_stack(
                       (x_centres, np.full(nx, y_edges[-1])))},
    }
    centre = np.array([x_edges.mean(), y_edges.mean()])

    def boundary_split(side, spec):
        """Split one outer side into its Dirichlet and Neumann face masks."""
        cells = index[spec["cells"]]
        open_mask = (cells >= 0) & (spec["beta"] > 0)
        if side == outlet_side:
            return cells, open_mask, np.zeros_like(open_mask)
        dirichlet = open_mask & far_field[spec["cells"]]
        return cells, dirichlet, open_mask & ~dirichlet

    for side, spec in sides.items():
        beta = spec["beta"]
        cells, use_dirichlet, use_neumann = boundary_split(side, spec)
        if use_dirichlet.any():
            # Far-field Dirichlet at half-cell distance:
            # phi_b = d . (x_face - centre) for the unit free stream.
            positions = spec["face_pos"]()[use_dirichlet]
            phi_b = (positions - centre) @ direction
            g_b = 2.0 * beta[use_dirichlet] * g_scale
            np.add.at(diag, cells[use_dirichlet], g_b)
            np.add.at(rhs, cells[use_dirichlet], g_b * phi_b)
        if use_neumann.any():
            # Prescribed background normal flux (exact uniform far field):
            # B = (U d . n) beta_f A_f with A_f = dx and unit U.
            outward = float(np.dot(direction, spec["normal"]))
            np.add.at(rhs, cells[use_neumann],
                      outward * beta[use_neumann] * dx)

    ids = np.arange(n_active)
    rows_a.append(ids)
    cols_a.append(ids)
    vals_a.append(diag)
    matrix = sparse.coo_matrix(
        (np.concatenate([np.atleast_1d(v).ravel() for v in vals_a]),
         (np.concatenate([np.atleast_1d(r).ravel() for r in rows_a]),
          np.concatenate([np.atleast_1d(c).ravel() for c in cols_a]))),
        shape=(n_active, n_active)).tocsr()

    print("Solving potential-flow Laplace system "
          f"({n_active:,} unknowns)...", flush=True)
    inverse_diag = 1.0 / np.maximum(matrix.diagonal(), 1e-300)
    preconditioner = sparse.linalg.LinearOperator(
        matrix.shape, matvec=lambda vector: inverse_diag * vector)
    try:
        phi_vec, info = cg(matrix, rhs, rtol=float(tolerance), atol=0.0,
                           maxiter=20000, M=preconditioner)
    except TypeError:  # older scipy signature
        phi_vec, info = cg(matrix, rhs, tol=float(tolerance),
                           maxiter=20000, M=preconditioner)
    solver_used = "scipy.sparse.linalg.cg (Jacobi preconditioner)"
    if info != 0:
        print(f"  CG did not converge (info={info}); "
              "falling back to sparse direct solve")
        phi_vec = spsolve(matrix.tocsc(), rhs)
        solver_used = "scipy.sparse.linalg.spsolve (SuperLU)"
    residual = float(np.linalg.norm(matrix @ phi_vec - rhs)
                     / max(np.linalg.norm(rhs), 1e-300))
    print(f"Potential-flow solve complete: relative residual "
          f"{residual:.3e} ({solver_used})", flush=True)

    if not np.isfinite(phi_vec).all():
        raise RuntimeError(
            "potential-flow solve produced non-finite potential values; the "
            "discrete system is singular or inconsistent. This must never "
            "reach the saved field.")
    phi = np.full((ny, nx), np.nan)
    phi[active] = phi_vec

    # Face normal velocities (unit inlet speed).
    u_face = np.zeros((ny, nx + 1))
    both = active[:, :-1] & active[:, 1:] & (open_face_x[:, 1:-1] > 0)
    u_face[:, 1:-1][both] = ((phi[:, 1:] - phi[:, :-1]) / dx)[both]
    v_face = np.zeros((ny + 1, nx))
    both = active[:-1, :] & active[1:, :] & (open_face_y[1:-1, :] > 0)
    v_face[1:-1, :][both] = ((phi[1:, :] - phi[:-1, :]) / dx)[both]
    for side, spec in sides.items():
        _cells, use_dirichlet, use_neumann = boundary_split(side, spec)
        if side == "x_low":
            target, orient = u_face[:, 0], -1.0
        elif side == "x_high":
            target, orient = u_face[:, -1], 1.0
        elif side == "y_low":
            target, orient = v_face[0, :], -1.0
        else:
            target, orient = v_face[-1, :], 1.0
        if use_dirichlet.any():
            positions = spec["face_pos"]()
            phi_b = (positions - centre) @ direction
            phi_cell = phi[spec["cells"]]
            outward = 2.0 * (phi_b - phi_cell) / dx
            target[use_dirichlet] = (orient * outward)[use_dirichlet]
        if use_neumann.any():
            background = float(
                np.dot(direction, {"x_low": (1.0, 0.0), "x_high": (1.0, 0.0),
                                   "y_low": (0.0, 1.0),
                                   "y_high": (0.0, 1.0)}[side]))
            target[use_neumann] = background
    return {"phi": phi, "u_face": u_face, "v_face": v_face,
            "active": active, "stagnant": stagnant, "fluid": fluid,
            "open_face_x": open_face_x, "open_face_y": open_face_y,
            "solver": solver_used, "residual": residual,
            "n_active": n_active, "n_stagnant": int(stagnant.sum())}


def conservation_diagnostics(solution: dict, dx: float) -> dict:
    """Per-cell flux imbalance and integrated boundary fluxes (unit speed)."""
    flux_x = solution["u_face"] * solution["open_face_x"] * dx
    flux_y = solution["v_face"] * solution["open_face_y"] * dx
    net = (flux_x[:, 1:] - flux_x[:, :-1] + flux_y[1:, :] - flux_y[:-1, :])
    active = solution["active"]
    cell_area = dx * dx
    divergence = np.zeros_like(net)
    divergence[active] = net[active] / cell_area
    boundary_out = np.concatenate((
        -flux_x[:, 0], flux_x[:, -1], -flux_y[0, :], flux_y[-1, :]))
    inlet = float(-boundary_out[boundary_out < 0].sum())
    outlet = float(boundary_out[boundary_out > 0].sum())
    return {
        "max_abs_divergence_per_s": float(np.abs(divergence[active]).max()),
        "rms_divergence_per_s": float(
            np.sqrt(np.mean(divergence[active] ** 2))),
        "max_abs_cell_flux_imbalance_m2s": float(np.abs(net[active]).max()),
        "mean_abs_cell_flux_imbalance_m2s": float(
            np.mean(np.abs(net[active]))),
        "integrated_inlet_flux_m2s": inlet,
        "integrated_outlet_flux_m2s": outlet,
        "relative_global_mass_imbalance": float(
            (inlet - outlet) / max(inlet, 1e-300)),
        "divergence_field": divergence,
    }


def narrow_corridor_report(alpha: np.ndarray, dx: float) -> dict:
    """Warn about corridors at or below ~2 cells: they exist geometrically in
    alpha/beta but their through-flow profile is under-resolved."""
    open_cell = alpha >= 0.5
    narrow = 0
    for grid in (open_cell, open_cell.T):
        padded = np.pad(grid, ((0, 0), (1, 1)), constant_values=False)
        for row in padded:
            edges = np.flatnonzero(np.diff(row.astype(np.int8)))
            for start, stop in zip(edges[::2], edges[1::2]):
                run = stop - start
                # Exclude runs touching the (padded) domain boundary: only
                # gaps bounded by solid on BOTH sides are corridors.
                interior = start > 0 and stop < len(row) - 2
                if interior and run <= 2:
                    narrow += run
    partial_faces = int(((alpha > 0.0) & (alpha < 1.0)).sum())
    subcell = int(((alpha > 0.0) & (alpha < 0.5)).sum())
    if narrow:
        print(f"  WARNING: {narrow} fluid cells lie in corridors <= 2 cells "
              f"({2 * dx:.1f} m) wide; the gap stays open through the "
              "cut-cell fractions but its flow profile is under-resolved. "
              "Reduce --grid-spacing for these paths.")
    if subcell:
        print(f"  WARNING: {subcell} cells are mostly solid but partially "
              "open (sub-cell passages/boundary cells). Passages narrower "
              "than one cell remain open via the face fractions but carry "
              "no resolved flow profile; do not treat their local speeds "
              "as accurate.")
    return {"narrow_corridor_cells": int(narrow),
            "mostly_solid_partially_open_cells": subcell,
            "cells_with_partial_fluid_fraction": partial_faces,
            "corridor_warning_threshold_m": 2.0 * dx}


def cell_velocities(solution: dict) -> tuple[np.ndarray, np.ndarray]:
    """Open-face-weighted cell-centre velocity components (unit speed)."""
    beta_x, beta_y = solution["open_face_x"], solution["open_face_y"]
    u_face, v_face = solution["u_face"], solution["v_face"]
    weight_u = beta_x[:, :-1] + beta_x[:, 1:]
    u = np.where(weight_u > 0,
                 (u_face[:, :-1] * beta_x[:, :-1]
                  + u_face[:, 1:] * beta_x[:, 1:])
                 / np.maximum(weight_u, 1e-300), 0.0)
    weight_v = beta_y[:-1, :] + beta_y[1:, :]
    v = np.where(weight_v > 0,
                 (v_face[:-1, :] * beta_y[:-1, :]
                  + v_face[1:, :] * beta_y[1:, :])
                 / np.maximum(weight_v, 1e-300), 0.0)
    fluid = solution["fluid"]
    u = np.where(fluid & solution["active"], u, 0.0)
    v = np.where(fluid & solution["active"], v, 0.0)
    return u, v


def diagnostic_figures(out_dir: Path, x: np.ndarray, y: np.ndarray,
                       x_edges: np.ndarray, y_edges: np.ndarray,
                       speed: np.ndarray, u: np.ndarray, v: np.ndarray,
                       fluid: np.ndarray, fine_solid: np.ndarray,
                       divergence: np.ndarray, n_sub: int,
                       reference_speed: float) -> None:
    dx = x_edges[1] - x_edges[0]
    fine_x = (x_edges[:-1, None]
              + ((np.arange(n_sub) + 0.5) / n_sub * dx)[None, :]).ravel()
    fine_y = (y_edges[:-1, None]
              + ((np.arange(n_sub) + 0.5) / n_sub * dx)[None, :]).ravel()

    def outlines(ax):
        if fine_solid.any():
            ax.contour(fine_x, fine_y, fine_solid.astype(float),
                       levels=[0.5], colors="k", linewidths=0.7)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    shown = np.where(fluid, speed, np.nan)
    image = ax.pcolormesh(x_edges, y_edges, shown, cmap="viridis",
                          shading="flat")
    stride = max(1, int(max(len(x), len(y)) / 40))
    ax.quiver(x[::stride], y[::stride], u[::stride, ::stride],
              v[::stride, ::stride], color="w", alpha=0.75,
              scale=25 * max(reference_speed, 0.1))
    outlines(ax)
    fig.colorbar(image, ax=ax, label="Pedestrian-level wind speed (m/s)")
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)",
           title="Pedestrian-level potential-flow wind "
                 "(z = ground + h_ped; building outlines at true "
                 "pedestrian-level geometry)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "pedestrian_wind_speed.png", dpi=250,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    shown = np.where(fluid, np.abs(divergence), np.nan)
    image = ax.pcolormesh(x_edges, y_edges, shown, cmap="magma",
                          shading="flat")
    outlines(ax)
    fig.colorbar(image, ax=ax, label="|div(u)| (1/s)")
    ax.set(xlabel="Local X (m)", ylabel="Local Y (m)",
           title="Mass-conservation error of the potential-flow solution")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "pedestrian_wind_mass_error.png", dpi=250,
                bbox_inches="tight")
    plt.close(fig)


def export_paraview(out_dir: Path, x: np.ndarray, y: np.ndarray,
                    u: np.ndarray, v: np.ndarray, speed: np.ndarray,
                    direction_deg: np.ndarray, phi: np.ndarray,
                    fluid: np.ndarray, alpha: np.ndarray,
                    ground: np.ndarray, pedestrian_z: np.ndarray,
                    context_meshes: dict | None = None) -> Path:
    """Write the solved 2-D field as ParaView-readable files under paraview/.

    Two representations of the one steady field:

    * ``pedestrian_wind_surface.vtp`` -- the terrain-following pedestrian
      surface itself: grid vertices at ``(x_i, y_j, z_ground + h_ped)``
      triangulated into a mesh, with the velocity vector (w = 0), speed,
      direction, potential, fluid mask and cut-cell fluid fraction as
      PointData.  This is the primary 3-D view; threshold on ``fluid_mask``
      to open the building footprints, and add Glyph/Stream Tracer filters
      on ``velocity_ms``.
    * ``pedestrian_wind.vtr`` -- the same PointData on the flat 2-D
      rectilinear grid (single z level 0) for plan-view slices/plots.

    ``context_meshes`` (name -> trimesh) are written as .vtp so the ground
    and buildings can be displayed with the field, exactly like the legacy
    05c/05d exports.
    """
    from paraview_export import write_vtp, write_vtr

    target = out_dir / "paraview"
    target.mkdir(parents=True, exist_ok=True)
    nx, ny = len(x), len(y)
    velocity = np.stack([u, v, np.zeros_like(u)], axis=-1)
    point_data = {
        "velocity_ms": velocity,
        "speed_ms": speed,
        "direction_to_deg": direction_deg,
        "potential_m2s": np.nan_to_num(phi, nan=0.0),
        "fluid_mask": fluid,
        "cell_fluid_fraction": alpha,
        "ground_z_m": ground,
        "pedestrian_z_m": pedestrian_z,
    }

    # Terrain-following pedestrian surface: one vertex per cell centre.
    X, Y = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack(
        [X.ravel(), Y.ravel(), pedestrian_z.ravel()])
    index = np.arange(nx * ny).reshape(ny, nx)
    a = index[:-1, :-1].ravel()
    b = index[:-1, 1:].ravel()
    c = index[1:, 1:].ravel()
    d = index[1:, :-1].ravel()
    faces = np.concatenate([np.column_stack([a, b, c]),
                            np.column_stack([a, c, d])])
    write_vtp(target / "pedestrian_wind_surface.vtp", vertices, faces,
              point_data={name: values.reshape(nx * ny, -1)
                          for name, values in point_data.items()})

    # Flat plan-view rectilinear grid (arrays as (z=1, y, x[, ncomp])).
    write_vtr(target / "pedestrian_wind.vtr", x, y, np.zeros(1),
              point_data={name: values[None, ...]
                          for name, values in point_data.items()})

    for name, mesh in (context_meshes or {}).items():
        write_vtp(target / f"{name}.vtp", np.asarray(mesh.vertices),
                  np.asarray(mesh.faces))
    print(f"  ParaView export: pedestrian_wind_surface.vtp + "
          f"pedestrian_wind.vtr -> {target}")
    return target


def resolve_reference_speed(args: argparse.Namespace) -> tuple[float, str]:
    if args.wind_speed_ms is not None:
        if not float(args.wind_speed_ms) > 0:
            raise ValueError("--wind-speed-ms must be positive")
        return float(args.wind_speed_ms), "explicit --wind-speed-ms"
    if args.weather_csv:
        from weather_provider import WeatherProvider
        provider = WeatherProvider(csv_path=args.weather_csv)
        speeds = np.asarray(
            provider.wind_ms(np.arange(0.0, 24.0, 1.0)), dtype=float)
        mean_speed = float(np.mean(speeds))
        if mean_speed > 0:
            return mean_speed, f"mean weather-CSV wind ({args.weather_csv})"
    return 1.0, "default unit reference (no weather wind available)"


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / METADATA_FILE
    if metadata_path.exists():
        metadata_path.unlink()   # never leave a stale completeness marker

    if args.face_subsamples < 1:
        raise ValueError("--face-subsamples must be at least 1")
    if not 0 < args.minimum_fluid_fraction < 1:
        raise ValueError("--minimum-fluid-fraction must lie in (0, 1)")

    reference_speed, reference_source = resolve_reference_speed(args)
    direction = flow_direction_vector(args.wind_direction_deg)
    outlet_side = choose_outlet_side(direction)

    print("Loading pedestrian-flow geometry...")
    ground_mesh = trimesh.load(args.ground_stl, force="mesh")
    building_mesh = trimesh.load(args.buildings_stl, force="mesh")
    low, high = load_route_extent(args.polylines_pkl)
    x_edges, y_edges, x, y, dx = build_axes(
        low, high, args.route_buffer_m, args.grid_spacing, args.maximum_cells)
    nx, ny = len(x), len(y)
    print(f"  Grid: {nx} x {ny} = {nx * ny:,} cells; dx = {dx:.3f} m")
    print(f"  Wind: from {args.wind_direction_deg:.1f} deg at reference "
          f"{reference_speed:.3f} m/s ({reference_source}); "
          f"outlet boundary {outlet_side}")
    print(f"  Pedestrian surface: z_ground(x, y) + "
          f"{args.pedestrian_height:.2f} m (terrain-following)")

    ground = ground_height_grid(ground_mesh, x, y)
    occupancy = PedestrianOccupancy(
        building_mesh, ground, x, y, args.pedestrian_height,
        args.occupancy_batch)
    t0 = time.time()
    alpha, beta_x, beta_y, fine_solid = cut_cell_fractions(
        occupancy, x_edges, y_edges, dx, args.face_subsamples)
    print(f"  Cut-cell fractions complete in {time.time() - t0:.0f}s; "
          f"open area fraction {alpha.mean():.3f}")

    print("Assembling potential-flow system...", flush=True)
    solution = solve_unit_potential(
        alpha, beta_x, beta_y, x_edges, y_edges, dx, direction, outlet_side,
        args.minimum_fluid_fraction, args.solver_tolerance)
    diagnostics = conservation_diagnostics(solution, dx)
    corridor = narrow_corridor_report(alpha, dx)

    # ------------------------------------------------------------------
    # DIRECTION BASIS
    #
    # The free-stream direction d enters the discrete system ONLY through the
    # right-hand side, and it enters linearly:
    #
    #   Dirichlet faces : phi_b = d . (x_face - centre)
    #   Neumann faces   : B     = (d . n) beta_f A_f
    #
    # The matrix itself is built from the cut-cell fractions alone and does not
    # contain d at all. So with the boundary-condition PLACEMENT held fixed --
    # the same outlet side, hence the same Dirichlet/Neumann masks and the same
    # dead-end far-field set -- two orthogonal solves span every wind direction
    # exactly:
    #
    #   phi(theta) = cos(theta) phi_E + sin(theta) phi_N
    #   u(theta)   = cos(theta) u_E   + sin(theta) u_N
    #
    # Two, not four: the southerly solution is just the negative of the
    # northerly one, so a four-direction basis would be exactly redundant.
    #
    # Pinning the outlet side across the pair is what makes this exact rather
    # than approximate. If each basis solve were allowed to pick its own
    # downwind outlet the way a standalone run does, the two would carry
    # different Dirichlet/Neumann placements and different dead-end sets, and
    # superposing them would no longer solve any single boundary-value problem.
    # ------------------------------------------------------------------
    basis = None
    if args.direction_basis == "on":
        print("\nSolving the two-direction basis (east and north unit free "
              f"streams, outlet pinned to {outlet_side})...", flush=True)
        basis_solutions = {}
        for label, basis_direction in (("east", np.array([1.0, 0.0])),
                                       ("north", np.array([0.0, 1.0]))):
            print(f"  basis component: {label}", flush=True)
            basis_solutions[label] = solve_unit_potential(
                alpha, beta_x, beta_y, x_edges, y_edges, dx, basis_direction,
                outlet_side, args.minimum_fluid_fraction,
                args.solver_tolerance)
        basis = {label: cell_velocities(item)
                 for label, item in basis_solutions.items()}
        basis_phi = {label: item["phi"] for label, item in basis_solutions.items()}

        # Self-check: rebuild the CONFIGURED direction from the basis and
        # compare against the direct solve. Because the outlet side is shared,
        # these must agree to solver tolerance. A failure here means the
        # superposition assumption has been broken by some direction-dependent
        # term creeping into the assembly, and it is far better to hear about
        # it now than to silently ship a basis that cannot be superposed.
        rebuilt_u = (direction[0] * basis["east"][0]
                     + direction[1] * basis["north"][0])
        rebuilt_v = (direction[0] * basis["east"][1]
                     + direction[1] * basis["north"][1])
        u_direct, v_direct = cell_velocities(solution)
        scale = max(float(np.nanmax(np.hypot(u_direct, v_direct))), 1e-12)
        basis_error = float(np.nanmax(np.hypot(rebuilt_u - u_direct,
                                               rebuilt_v - v_direct)) / scale)
        print(f"  basis reconstruction of the configured direction: "
              f"max relative velocity error {basis_error:.3e}")
        if basis_error > 1e-6:
            raise RuntimeError(
                f"the two-direction basis does not reproduce the direct solve "
                f"(relative error {basis_error:.3e}). Superposition is only "
                "exact while the boundary-condition placement is identical "
                "across the basis pair and the configured solve; something "
                "direction-dependent has entered the assembly.")
        basis_arrays = {
            "u_east": basis["east"][0], "v_east": basis["east"][1],
            "u_north": basis["north"][0], "v_north": basis["north"][1],
            "phi_east": basis_phi["east"], "phi_north": basis_phi["north"],
        }
        basis_diagnostics = {
            label: conservation_diagnostics(item, dx)
            for label, item in basis_solutions.items()
        }
        for label, item in basis_diagnostics.items():
            imbalance = abs(item["relative_global_mass_imbalance"])
            if not np.isfinite(imbalance) or imbalance > float(
                    args.maximum_relative_imbalance):
                raise RuntimeError(
                    f"the {label} basis component has a mass imbalance of "
                    f"{100 * imbalance:.4f}%, above the "
                    f"{100 * float(args.maximum_relative_imbalance):.4f}% "
                    "tolerance; refusing to write a non-conservative basis.")
        if not all(np.isfinite(array).all()
                   for array in (basis_arrays["u_east"], basis_arrays["v_east"],
                                 basis_arrays["u_north"], basis_arrays["v_north"])):
            raise RuntimeError(
                "the direction basis contains non-finite velocities; refusing "
                "to write it.")
        basis = {"arrays": basis_arrays, "diagnostics": basis_diagnostics,
                 "reconstruction_error": basis_error}

    u_unit, v_unit = cell_velocities(solution)
    u = u_unit * reference_speed
    v = v_unit * reference_speed
    speed = np.hypot(u, v)
    direction_deg = np.degrees(np.arctan2(u, v)) % 360.0
    phi = solution["phi"] * reference_speed
    fluid = solution["fluid"]
    pedestrian_z = ground + float(args.pedestrian_height)

    scaled = {k: diagnostics[k] * reference_speed for k in (
        "max_abs_divergence_per_s", "rms_divergence_per_s",
        "max_abs_cell_flux_imbalance_m2s", "mean_abs_cell_flux_imbalance_m2s",
        "integrated_inlet_flux_m2s", "integrated_outlet_flux_m2s")}
    scaled["relative_global_mass_imbalance"] = diagnostics[
        "relative_global_mass_imbalance"]
    print("Mass-conservation diagnostics (at reference speed):")
    print(f"  max |div(u)| = {scaled['max_abs_divergence_per_s']:.3e} 1/s")
    print(f"  RMS |div(u)| = {scaled['rms_divergence_per_s']:.3e} 1/s")
    print(f"  inlet flux   = {scaled['integrated_inlet_flux_m2s']:.4f} m^2/s")
    print(f"  outlet flux  = {scaled['integrated_outlet_flux_m2s']:.4f} m^2/s")
    print(f"  imbalance    = "
          f"{100 * scaled['relative_global_mass_imbalance']:.4f} %")

    # Validate BEFORE any array is written: a field that fails these checks is
    # unusable downstream, and silently saving it turns a step-3 solver fault
    # into a confusing step-4 crash (or, worse, wrong convection).
    # The potential is intentionally NaN outside the solved region (solids and
    # enclosed stagnant courtyards, which carry zero velocity), so it is
    # checked only where it was actually solved.
    if not all(np.isfinite(array).all()
               for array in (u, v, speed, phi[solution["active"]])):
        raise RuntimeError(
            "potential-flow field contains non-finite values; refusing to "
            "write it. Check the geometry and grid spacing for this case.")
    imbalance = abs(scaled["relative_global_mass_imbalance"])
    if not np.isfinite(imbalance) or imbalance > float(
            args.maximum_relative_imbalance):
        raise RuntimeError(
            f"potential-flow mass imbalance {100 * imbalance:.4f}% exceeds the "
            f"{100 * float(args.maximum_relative_imbalance):.4f}% tolerance "
            f"(solver residual {solution['residual']:.3e}); refusing to write "
            "a non-conservative field.")

    print("Writing pedestrian wind outputs...", flush=True)
    np.save(out_dir / POTENTIAL_FILE, phi.astype(np.float64))
    np.save(out_dir / VELOCITY_U_FILE, u.astype(np.float64))
    np.save(out_dir / VELOCITY_V_FILE, v.astype(np.float64))
    np.save(out_dir / VELOCITY_SPEED_FILE, speed.astype(np.float64))
    np.save(out_dir / VELOCITY_DIRECTION_FILE, direction_deg.astype(np.float64))
    np.save(out_dir / GROUND_Z_FILE, ground.astype(np.float64))
    np.save(out_dir / PEDESTRIAN_Z_FILE, pedestrian_z.astype(np.float64))
    np.save(out_dir / X_FILE, x.astype(np.float64))
    np.save(out_dir / Y_FILE, y.astype(np.float64))
    np.save(out_dir / FLUID_MASK_FILE, fluid)
    np.save(out_dir / CELL_FRACTION_FILE, alpha.astype(np.float64))
    np.save(out_dir / FACE_FRACTION_X_FILE, beta_x.astype(np.float64))
    np.save(out_dir / FACE_FRACTION_Y_FILE, beta_y.astype(np.float64))
    if basis is not None:
        # Stored at UNIT inlet speed, exactly as solved. Downstream applies
        # both the direction weights and the speed scale, so nothing here bakes
        # in a particular wind.
        np.savez_compressed(
            out_dir / BASIS_FILE,
            **{name: array.astype(np.float64)
               for name, array in basis["arrays"].items()})
        print(f"  direction basis written ({BASIS_FILE}); any wind direction "
              "can now be reconstructed without re-solving")

    metadata = {
        "format_version": 1,
        "model": ("terrain-referenced pedestrian-level potential-flow model "
                  "with embedded (cut-cell) building boundaries; "
                  "mass-conserving 2-D Laplace solve, flow only"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "governing_equations": {
            "potential": "Laplacian(phi) = 0 in the pedestrian-level fluid domain",
            "velocity": "u = grad(phi)",
            "building_boundary": "d(phi)/dn = 0  (u . n = 0, no penetration)",
            "inlet_and_lateral": ("prescribed background normal flux "
                                  "(U d . n) beta_f A_f on non-outlet outer "
                                  "boundaries"),
            "outlet": (f"far-field Dirichlet phi = U d . x on the dominant "
                       f"downwind boundary ({outlet_side})"),
        },
        "coordinate_system": {
            "frame": "TREC-Route local projected metres; x east, y north",
            "project_crs": args.project_crs,
            "direction_convention": ("wind_direction_from_deg is the "
                                     "meteorological FROM bearing; the saved "
                                     "velocity_direction is the compass "
                                     "bearing the flow is going TOWARD"),
        },
        "grid": {"spacing_m": dx, "nx": nx, "ny": ny,
                 "route_buffer_m": float(args.route_buffer_m),
                 "face_subsamples": int(args.face_subsamples)},
        "pedestrian_height_m": float(args.pedestrian_height),
        "pedestrian_surface": "z_p(x, y) = z_ground(x, y) + pedestrian_height",
        "wind_direction_from_deg": float(args.wind_direction_deg),
        "reference_wind_speed_ms": reference_speed,
        "reference_wind_source": reference_source,
        "linearity": ("solution scales exactly as u(U) = U/U_ref * u(U_ref) "
                      "for the same direction; solved once at unit speed"),
        "direction_basis": ({
            "available": True,
            "file": BASIS_FILE,
            "components": ["east", "north"],
            "outlet_side": outlet_side,
            "superposition": ("u(theta) = cos(theta) u_east + sin(theta) "
                              "u_north, with theta the compass bearing the "
                              "flow travels TOWARD; combine u and v, then "
                              "recompute speed = hypot(u, v) -- speeds do not "
                              "superpose"),
            "exactness": ("exact for this discretisation: the matrix contains "
                          "no direction term and the outlet side is pinned "
                          "across the pair, so only the right-hand side "
                          "changes and it changes linearly in d"),
            "reconstruction_error_vs_direct_solve": basis["reconstruction_error"],
            "caveat": ("a standalone solve at a DIFFERENT direction would pick "
                       "its own downwind outlet and so differ slightly near "
                       "the outer boundary; the basis holds the outlet fixed, "
                       "which is what makes superposition well posed. The "
                       "difference is a near-boundary discretisation artifact "
                       "that shrinks with --route-buffer-m."),
            "mass_imbalance_by_component": {
                label: float(item["relative_global_mass_imbalance"])
                for label, item in basis["diagnostics"].items()},
        } if basis is not None else {"available": False}),
        "obstacle_treatment": (
            "conservative cut-cell finite volume: per-cell fluid volume "
            "fraction alpha and per-face open fraction beta sub-sampled "
            f"({args.face_subsamples} per edge) from the true 3-D building "
            "STL intersected with the terrain-following pedestrian surface "
            "via vertical-ray parity"),
        "vegetation_treatment": ("ignored (Option A): no aerodynamic drag, "
                                 "tree crowns are never solid obstacles"),
        "solver": {"backend": solution["solver"],
                   "relative_tolerance": float(args.solver_tolerance),
                   "achieved_relative_residual": solution["residual"],
                   "active_cells": solution["n_active"],
                   "stagnant_enclosed_cells": solution["n_stagnant"],
                   "minimum_fluid_fraction": float(
                       args.minimum_fluid_fraction)},
        "mass_conservation": scaled,
        "resolution_warnings": corridor,
        "paraview_export": (None if args.no_paraview else {
            "directory": "paraview/",
            "files": ["pedestrian_wind_surface.vtp (terrain-following "
                      "pedestrian surface with velocity/speed/direction "
                      "PointData)",
                      "pedestrian_wind.vtr (flat plan-view grid, "
                      "same PointData)",
                      "context_ground.vtp", "context_buildings.vtp"]}),
        "inputs": {"buildings_stl": file_identity(args.buildings_stl),
                   "ground_stl": file_identity(args.ground_stl),
                   "polylines_pkl": file_identity(args.polylines_pkl)},
        "limitations": [
            "potential flow: represents blockage and corner acceleration only",
            "no building wakes, separation, vortex shedding or recirculation; "
            "the flow recovers smoothly behind obstacles, so downstream "
            "sheltering is underestimated",
            "no viscous boundary layers or wall shear; near-wall speeds are "
            "inviscid slip values",
            "no turbulence, no heat/temperature transport, no buoyancy "
            "(step 3 is flow only by design)",
            "2-D in plan view: one terrain-following pedestrian level, no "
            "vertical flow structure",
            "vegetation aerodynamic drag ignored in this first implementation",
            "corridors narrower than ~2 cells stay open via cut-cell "
            "fractions but their flow profile is under-resolved",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.no_figure:
        diagnostic_figures(out_dir, x, y, x_edges, y_edges, speed, u, v,
                           fluid & solution["active"], fine_solid,
                           diagnostics["divergence_field"] * reference_speed,
                           args.face_subsamples, reference_speed)
    if not args.no_paraview:
        export_paraview(out_dir, x, y, u, v, speed, direction_deg, phi,
                        fluid, alpha, ground, pedestrian_z,
                        context_meshes={"context_ground": ground_mesh,
                                        "context_buildings": building_mesh})
    print(f"[potential_flow_result] cells={nx * ny} "
          f"active={solution['n_active']} "
          f"max_div={scaled['max_abs_divergence_per_s']:.3e} "
          f"imbalance_pct={100 * scaled['relative_global_mass_imbalance']:.4f} "
          f"output_dir={out_dir}")
    return metadata


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
