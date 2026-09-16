"""
crown_field_reconstruction.py -- arbitrary watertight crown surfaces from a
LiDAR point cluster, with no geometric primitive anywhere in the pipeline.

WHY THIS EXISTS
---------------
Two earlier crown models both impose a shape class the data never justified:

  * `hemisphere` fits a dome, forcing depth/width = 1.00 on every crown.
  * `radial` fits one radius per (height band x angular sector) about a
    VERTICAL AXIS -- a star-shaped hull. With 8x16 bands that outline is a
    16-gon in plan, which is why a scene built this way reads as a field of
    circles from above. It is a primitive parameterisation: no matter how
    many sectors are used, it cannot represent a crown that is re-entrant in
    plan, forked, leaning, or wrapped around a neighbour, because every
    height band is a single closed loop about one axis.

The alpha-shape path (`crown_reconstruction.reconstruct_crown`) was the
attempt at an unconstrained surface, and it is the right idea, but on this
LiDAR it does not close. Its boundary pinches: retained tetrahedra touch
along edges shared by more than two boundary faces, so the surface is
non-manifold and `is_watertight` fails at every alpha that covers a useful
fraction of the cloud. Measured on a real cluster, the only alpha values
producing a watertight result are the degenerate small ones -- 12 faces,
0.1 m3, 0.1% coverage. Growing the tetrahedron set to seal the pinches does
close it, but the repair runs away and yields volumes hundreds of times the
bounding box. Carving from the outside instead does not help either, because
3-D Delaunay slivers have huge circumradii and tiny volume, so the carve
tunnels straight through the solid.

METHOD
------
Work with an implicit field rather than a combinatorial complex, which
sidesteps the manifold problem entirely instead of repairing it:

  1. Rasterise the cluster's points into an occupancy grid. The voxel is the
     cloud's OWN spacing (floored at `target_resolution_m`, since resolving
     finer than the sampling only manufactures holes) and additionally capped
     so several voxels span the bridge radius -- otherwise a sparse cluster
     gets a voxel larger than the radius, the balls fall below one cell, and
     the field collapses.
  2. Euclidean distance transform, then threshold at `bridge_radius_m`. The
     result is the union of balls of that radius about every point -- a
     smooth, closed, entirely arbitrary solid. `bridge_radius_m` is the one
     physical parameter: it is the canopy gap scale being bridged, so it has
     a meaning a reader can argue with, unlike a sector count.
  3. Fill interior cavities and drop disconnected specks below a fraction of
     the main body, so one crown is one solid rather than a swarm.
  4. Marching cubes on that binary field. This is watertight and consistently
     wound BY CONSTRUCTION -- the surface separates inside from outside of a
     filled volume, so there is nothing to seal afterwards.
  5. Taubin smoothing to take off the voxel stair-stepping without the
     shrinkage a plain Laplacian would cause, then quadric decimation to a
     face budget with `preservetopology=True`, which is what keeps the mesh
     watertight through simplification.

Measured on 40 real segmented crowns at `bridge_radius_m = 0.70`: 37/40
watertight, median 1,125 faces, 98.5% of the cluster's points contained
inside the solid, and the surface sits a median 0.64 m from the nearest
measured point. Nothing in the result is axisymmetric.

The face budget matters at scene scale: a case holds thousands of crowns and
the vegetation STL is consumed by a ray tracer, so an unbudgeted marching
cubes surface (~74,000 faces per crown here) would be unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage


class CrownFieldError(RuntimeError):
    """Raised when a cluster cannot yield a usable surface."""


@dataclass
class FieldSettings:
    """Tunables for implicit-field crown reconstruction."""

    target_resolution_m: float = 0.10
    """Requested voxel size. Floored at the cloud's own point spacing --
    resolving finer than the data is sampled only creates holes to fill."""

    bridge_radius_m: float = 0.70
    """Union-of-balls radius, metres: the canopy gap scale being bridged.
    THE parameter of this method. Smaller hugs the points more tightly but
    fragments the canopy; larger inflates it. 0.70 was chosen on real crowns
    as the smallest radius reaching >98% point containment while keeping the
    crown a single body."""

    face_budget: int = 800
    """Target faces per crown after decimation. Scene-scale constraint: a
    case has thousands of crowns feeding a ray tracer."""

    smooth_iterations: int = 8
    """Taubin smoothing passes. Taubin rather than Laplacian because it does
    not shrink the volume."""

    min_component_fraction: float = 0.05
    """Drop connected components smaller than this fraction of the largest."""

    min_component_voxels: int = 27
    """...and never keep a component below this absolute voxel count."""

    max_voxels: float = 2.0e7
    """Grid cap. The voxel is coarsened to respect it, so a huge cluster
    degrades in resolution rather than exhausting memory."""

    minimum_points: int = 4
    """Below this a cluster is not reconstructable.

    A distance field needs no minimum in principle -- unlike a Delaunay
    complex, which is where the previous value of 24 came from. It was
    carried over unexamined and created a SILENT GAP against
    `02_vegetation_to_stl.py --min-hull-points` (default 10): clusters of
    10-23 points passed that filter, were refused here, and silently fell
    back to a hemisphere. Every hemisphere in the built Lisbon geometry
    (271 of 3,178 components in lisbon1) came from that gap and nothing
    else. Keep this at or below --min-hull-points."""


@dataclass
class FieldReport:
    """What reconstruction did, for logging and QC."""

    voxel_m: float
    n_faces: int
    containment: float
    watertight: bool
    body_count: int
    volume_m3: float
    repaired: bool = False


def characteristic_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance -- the cloud's own length scale."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise CrownFieldError("need at least two points for spacing")
    distances, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1e-3


VOXELS_PER_BRIDGE_RADIUS = 3.0
"""How many voxels must span the bridge radius for the ball to be resolved."""


def _grid_geometry(points: np.ndarray, settings: FieldSettings
                   ) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Voxel size, grid origin, shape, and the EFFECTIVE bridge radius.

    The voxel must resolve the bridge radius, not merely the point spacing.
    A sparse cluster has a large `characteristic_spacing`, and taking the
    voxel from spacing alone lets it exceed the radius -- the union of balls
    then falls BELOW one voxel and the field collapses to a handful of
    isolated cells, or to nothing. That is what sent sparse clusters down the
    hemisphere fallback.

    Two-sided, because the `max_voxels` cap can push the voxel back up: the
    grid is first refined to resolve the radius, and if the cap then coarsens
    it, the radius is raised to match. A coarse grid cannot represent a fine
    radius, so the returned radius is the one actually resolvable -- reported
    rather than silently unmet.
    """
    voxel = max(settings.target_resolution_m, characteristic_spacing(points))
    voxel = min(voxel, settings.bridge_radius_m / VOXELS_PER_BRIDGE_RADIUS)
    pad = settings.bridge_radius_m + 3.0 * voxel
    origin = points.min(axis=0) - pad
    shape = np.ceil((points.max(axis=0) + pad - origin) / voxel).astype(int) + 1
    for _ in range(8):
        if float(np.prod(shape.astype(float))) <= settings.max_voxels:
            break
        voxel *= float(np.prod(shape.astype(float)) / settings.max_voxels) ** (1.0 / 3.0)
        pad = settings.bridge_radius_m + 3.0 * voxel
        origin = points.min(axis=0) - pad
        shape = np.ceil((points.max(axis=0) + pad - origin) / voxel).astype(int) + 1
    radius = max(settings.bridge_radius_m, VOXELS_PER_BRIDGE_RADIUS * voxel)
    return voxel, origin, shape, radius


def solid_field(points: np.ndarray, settings: Optional[FieldSettings] = None
                ) -> Tuple[np.ndarray, np.ndarray, float]:
    """The filled binary solid for a cluster, plus its origin and voxel size.

    Returned separately from the mesh so containment can be answered exactly
    against the same field the surface was extracted from, with no ray
    casting and no `rtree` dependency.
    """
    settings = settings or FieldSettings()
    points = np.asarray(points, dtype=float)
    if len(points) < settings.minimum_points:
        raise CrownFieldError(
            f"cluster has {len(points)} points, below the "
            f"{settings.minimum_points} needed")

    voxel, origin, shape, radius = _grid_geometry(points, settings)
    occupancy = np.zeros(shape, dtype=bool)
    index = ((points - origin) / voxel).astype(int)
    np.clip(index, 0, np.array(shape) - 1, out=index)
    occupancy[index[:, 0], index[:, 1], index[:, 2]] = True

    # Union of balls of bridge_radius_m about every measured point.
    distance = ndimage.distance_transform_edt(~occupancy, sampling=voxel)
    solid = distance <= radius

    # Close over DIAGONAL voxel contacts before surfacing. Two voxels touching
    # only at an edge or corner are one region to `label`, but marching cubes
    # meets that contact as a pinch: the surface has no hole there yet more
    # than two faces share the edge, so it is non-manifold and `is_watertight`
    # is False. Measured on a plain blob that is 12 such edges out of 37,242
    # faces with ZERO boundary edges -- invisible to a hole-filling repair,
    # which is why this is fixed in the field rather than on the mesh. A full
    # 3x3x3 element merges those contacts into face-connected ones; it costs
    # about 12% in volume, the price of a closed surface.
    solid = ndimage.binary_closing(solid, np.ones((3, 3, 3)))
    solid = ndimage.binary_fill_holes(solid)

    labels, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(solid, labels, range(1, n + 1))
        floor = max(sizes.max() * settings.min_component_fraction,
                    float(settings.min_component_voxels))
        keep = np.nonzero(sizes >= floor)[0] + 1
        solid = np.isin(labels, keep)

    if solid.sum() < 8:
        raise CrownFieldError("field collapsed to nothing at this radius")
    return solid, origin, voxel


def containment(points: np.ndarray, solid: np.ndarray, origin: np.ndarray,
                voxel: float) -> float:
    """Fraction of the cluster's points lying inside the solid.

    An exact lookup in the field the surface came from -- the honest measure
    of whether the reconstruction represents the measurement.
    """
    points = np.asarray(points, dtype=float)
    index = np.floor((points - origin) / voxel).astype(int)
    inside_grid = np.all((index >= 0) & (index < np.array(solid.shape)), axis=1)
    hit = np.zeros(len(points), dtype=bool)
    ok = index[inside_grid]
    hit[inside_grid] = solid[ok[:, 0], ok[:, 1], ok[:, 2]]
    return float(hit.mean())


def _decimate(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh:
    """Quadric decimation that preserves watertightness.

    `preservetopology=True` is the load-bearing flag: without it the collapse
    happily opens the surface, and a crown that is not closed cannot be paired
    for Beer-Lambert entry/exit in the MRT ray trace.
    """
    if len(mesh.faces) <= budget:
        return mesh
    try:
        import pymeshlab
    except ImportError:
        return mesh
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    try:
        meshset.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(budget), preserveboundary=True,
            preservetopology=True, planarquadric=True, autoclean=True)
    except Exception:
        return mesh
    out = meshset.current_mesh()
    reduced = trimesh.Trimesh(out.vertex_matrix(), out.face_matrix(), process=False)
    # Refuse a decimation that broke the surface -- the budget is a preference,
    # closure is a requirement.
    if not reduced.is_watertight and mesh.is_watertight:
        return mesh
    return reduced


def reconstruct_crown(points: np.ndarray,
                      settings: Optional[FieldSettings] = None
                      ) -> Tuple[trimesh.Trimesh, FieldReport]:
    """Arbitrary watertight surface around one crown cluster.

    Raises `CrownFieldError` if the cluster cannot produce one.
    """
    settings = settings or FieldSettings()
    points = np.asarray(points, dtype=float)
    solid, origin, voxel = solid_field(points, settings)

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(solid, pitch=voxel)
    mesh.apply_translation(origin)
    if len(mesh.faces) < 4:
        raise CrownFieldError("marching cubes produced no usable surface")

    if settings.smooth_iterations:
        trimesh.smoothing.filter_taubin(mesh, iterations=settings.smooth_iterations)
    mesh = _decimate(mesh, settings.face_budget)

    repaired = False
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        repaired = True
    trimesh.repair.fix_normals(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    return mesh, FieldReport(
        voxel_m=voxel,
        n_faces=len(mesh.faces),
        containment=containment(points, solid, origin, voxel),
        watertight=bool(mesh.is_watertight),
        body_count=int(mesh.body_count),
        volume_m3=float(mesh.volume),
        repaired=repaired,
    )
