"""crown_reconstruction.py -- watertight surface reconstruction for LiDAR clusters.

Builds an arbitrary, irregular, manifold, watertight triangular mesh that
tightly encloses a vegetation point cluster. NO geometric primitive is assumed
anywhere: the surface is derived entirely from the points, so a lobed, forked,
leaning or hollow crown comes out lobed, forked, leaning or hollow.

WHY AN ALPHA SHAPE AND NOT POISSON OR BALL-PIVOTING
----------------------------------------------------
Screened Poisson needs consistently oriented normals, which airborne LiDAR does
not provide for a canopy sampled almost entirely from above -- normal estimation
on a one-sided shell flips wherever the local neighbourhood is planar, and the
reconstruction then bulges or self-intersects. Ball-pivoting leaves holes
wherever the sampling is sparser than the ball, and a hole is fatal here (see
CLOSURE below), so it would need a separate sealing pass that reintroduces the
guesswork it was meant to avoid.

The 3D Delaunay alpha shape needs no normals and is CLOSED BY CONSTRUCTION: it
is the boundary of a set of tetrahedra, and the boundary of any cell complex is
a closed surface. That property is not cosmetic here -- the radiation model
computes Beer-Lambert attenuation from the path length through the crown by
pairing ray entries against exits, so an open surface yields a nonsense chord
rather than an obvious visual defect.

(CGAL alpha wrapping via pymeshlab would also be watertight by construction, and
was tried first. ``generate_alpha_wrap`` SEGFAULTS on a bare point cloud in this
environment, which is not survivable in a 3800-crown batch, so the
self-contained SciPy route is used instead.)

THE PIPELINE
------------
1. Voxel downsample at the target resolution. This simultaneously enforces the
   detail floor and bounds the element count -- it is the single most effective
   control over how expensive the result is downstream.
2. Statistical outlier removal on the k-nearest-neighbour mean distance:
   isolated returns from birds, wires and multi-path sit far from their
   neighbours and are dropped before they can drag the surface outward.
3. Radius density filter: points with too few neighbours inside a small ball are
   sparse stragglers, not canopy.
4. ADAPTIVE alpha: alpha starts near the characteristic point spacing and grows
   until the resulting surface is watertight, manifold, and encloses at least
   ``coverage_target`` of the surviving points. Small alpha follows fine
   structure but leaves the shape fragmented; large alpha degenerates toward the
   convex hull. The search takes the smallest alpha that satisfies both
   criteria, which is the tightest defensible enclosure.
5. Manifold sealing: only the largest FACE-CONNECTED component of retained
   tetrahedra is kept, which removes islands and the vertex- or edge-only
   contacts that would otherwise make the boundary non-manifold.
6. Optional decimation to a per-crown face budget, re-validated afterwards.

Every stage is validated, and the caller is told which alpha was used and what
fraction of points the surface encloses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree


class CrownReconstructionError(ValueError):
    """Raised when no valid watertight surface can be produced."""


@dataclass
class ReconstructionSettings:
    """Parameters controlling detail, robustness and cost.

    ``voxel_size`` is the target resolution. 0.10 m captures the structural
    detail asked for while keeping element counts tractable; going finer buys
    detail the downstream radiation model cannot use, because a crown is a
    Beer-Lambert attenuating volume rather than a resolved surface.
    """

    voxel_size: float = 0.10
    # Statistical outlier removal: a point whose mean distance to its k nearest
    # neighbours exceeds mean + ratio*std over the cluster is an outlier.
    outlier_neighbours: int = 12
    outlier_std_ratio: float = 2.0
    # Density filter: minimum neighbours inside density_radius (in voxels).
    density_radius_voxels: float = 3.0
    density_min_neighbours: int = 4
    # Adaptive alpha search, in multiples of the characteristic spacing.
    alpha_start_factor: float = 1.6
    alpha_growth: float = 1.35
    alpha_max_steps: int = 14
    coverage_target: float = 0.95
    # Surface refinement. A LiDAR crown cloud is a near-SHELL: returns come
    # from the canopy surface, so the interior is empty. A solid alpha shape
    # therefore only forms once alpha exceeds the local curvature/thickness
    # scale -- measured on a test crown, no alpha below 2.0 m produced a
    # watertight solid, which would cap surface detail at 2 m however finely
    # the cloud is sampled.
    #
    # Seeding the interior with support points decouples the two: the interior
    # lattice makes small-alpha tetrahedra viable inside the crown, so the
    # boundary can follow the real surface points at ``refine_alpha`` while the
    # solid stays closed. The support points are strictly interior and never
    # appear on the surface.
    # DEFAULT OFF, and the reason is measured rather than assumed. With
    # interior support the surface does close at alpha 0.5-1.0 m, but the
    # resulting boundary is NON-MANIFOLD: at fine alpha the knobbly surface
    # pinches, leaving edges shared by four faces instead of two, and
    # ``is_watertight`` fails (measured: 39k-134k faces, manifold False,
    # measured-point coverage 87-92%). A non-manifold crown would still pair ray
    # entries against exits correctly -- crossings stay even -- but trimesh
    # volume and normal queries become unreliable, so it is not shipped as the
    # default. Enabling it requires a manifold-sealing pass that removes the
    # pinching tetrahedra; the machinery below is left in place for that work.
    refine_surface: bool = False
    refine_alpha_voxels: float = 5.0     # target surface alpha, in voxels
    refine_support_ratio: float = 0.6    # lattice spacing / refine alpha
    # Cost control. None disables decimation.
    max_faces: int | None = 3000
    minimum_points: int = 24

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionReport:
    """What the reconstruction actually did, for provenance and QA."""

    n_input: int
    n_after_downsample: int
    n_after_filtering: int
    alpha_m: float
    characteristic_spacing_m: float
    coverage: float
    n_faces: int
    decimated: bool
    watertight: bool
    refined: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Point-cloud conditioning
# ---------------------------------------------------------------------------
def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """One representative point per occupied voxel (the voxel centroid).

    Enforces the resolution floor and bounds element count in one step. The
    centroid rather than an arbitrary member keeps the surface centred on the
    local point mass instead of jittering onto whichever return came first.
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0:
        raise CrownReconstructionError("voxel size must be positive")
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_points = points[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    groups = np.split(sorted_points, boundaries)
    return np.array([group.mean(axis=0) for group in groups])


def statistical_outlier_filter(points: np.ndarray, neighbours: int = 12,
                               std_ratio: float = 2.0) -> np.ndarray:
    """Drop points whose mean k-NN distance is anomalously large.

    Isolated LiDAR returns -- birds, wires, multi-path -- sit far from their
    neighbours. Left in, they drag the alpha shape outward and inflate the
    crown; this is the standard, established test for them.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= neighbours + 1:
        return points
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=neighbours + 1)
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    keep = mean_distance <= threshold
    return points[keep] if keep.sum() >= 4 else points


def density_filter(points: np.ndarray, radius: float,
                   min_neighbours: int = 4) -> np.ndarray:
    """Drop points with too few neighbours inside ``radius``.

    Complements the statistical test: a small clump of stragglers can have a
    respectable mean k-NN distance among themselves while still not being part
    of the canopy.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= min_neighbours + 1 or radius <= 0:
        return points
    tree = cKDTree(points)
    counts = np.array([len(item) - 1
                       for item in tree.query_ball_point(points, radius)])
    keep = counts >= min_neighbours
    return points[keep] if keep.sum() >= 4 else points


def characteristic_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance -- the cloud's own length scale.

    The alpha search is anchored to this rather than to an absolute constant so
    that a densely sampled crown and a sparsely sampled one each get an alpha
    appropriate to their own sampling.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise CrownReconstructionError("need at least two points for spacing")
    distances, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1e-3


# ---------------------------------------------------------------------------
# Alpha shape
# ---------------------------------------------------------------------------
def _circumradii(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Circumscribed-sphere radius of each tetrahedron.

    Solved as a linear system per tetrahedron rather than by the determinant
    formula: it is the same algebra, vectorises cleanly, and degenerate (flat)
    tetrahedra fall out as a singular system that is flagged rather than
    silently producing a huge radius.
    """
    a = points[tetrahedra[:, 0]]
    matrix = np.stack([points[tetrahedra[:, i]] - a for i in (1, 2, 3)], axis=1)
    squared = np.sum(np.stack(
        [points[tetrahedra[:, i]] ** 2 - a ** 2 for i in (1, 2, 3)], axis=1),
        axis=2)
    radii = np.full(len(tetrahedra), np.inf)
    determinant = np.linalg.det(matrix)
    usable = np.abs(determinant) > 1e-12
    if usable.any():
        centres = np.linalg.solve(matrix[usable], 0.5 * squared[usable])
        radii[usable] = np.linalg.norm(centres - a[usable], axis=1)
    return radii


def _largest_connected_tetrahedra(tetrahedra: np.ndarray,
                                  neighbours: np.ndarray,
                                  keep: np.ndarray) -> np.ndarray:
    """Largest FACE-connected group of retained tetrahedra.

    Discards islands, and with them the vertex-only and edge-only contacts that
    make a boundary non-manifold even when every tetrahedron is individually
    fine. Uses the Delaunay neighbour table, so no adjacency is rebuilt.
    """
    kept = np.flatnonzero(keep)
    if len(kept) == 0:
        return kept
    index = -np.ones(len(tetrahedra), dtype=np.int64)
    index[kept] = np.arange(len(kept))
    parent = np.arange(len(kept))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for local, tet in enumerate(kept):
        for neighbour in neighbours[tet]:
            if neighbour >= 0 and keep[neighbour]:
                other = index[neighbour]
                root_a, root_b = find(local), find(other)
                if root_a != root_b:
                    parent[root_b] = root_a
    roots = np.array([find(i) for i in range(len(kept))])
    labels, counts = np.unique(roots, return_counts=True)
    return kept[roots == labels[np.argmax(counts)]]


def _boundary_faces(tetrahedra: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary of a set of tetrahedra.

    A face shared by two retained tetrahedra is interior; a face belonging to
    exactly one is on the boundary. Orientation is fixed by the tetrahedron's
    opposite vertex, so the result is consistently outward without a separate
    normal-repair pass.
    """
    cells = tetrahedra[selected]
    corners = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    faces = np.concatenate([cells[:, list(triple)] for triple in corners])
    apex = np.concatenate([cells[:, opposite] for opposite in (0, 1, 2, 3)])
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    boundary = counts[inverse] == 1
    return faces[boundary], apex[boundary]


def alpha_solid(points: np.ndarray, alpha: float):
    """Retained tetrahedra of the alpha complex, or None.

    Returned alongside the triangulation so that containment and coverage can be
    answered EXACTLY from the same cell complex -- ``Delaunay.find_simplex``
    tells which tetrahedron holds a query point, which is both faster and more
    robust than a ray-casting containment test, and needs no spatial-index
    package.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 5:
        return None
    try:
        triangulation = Delaunay(points)
    except Exception:
        return None
    tetrahedra = triangulation.simplices
    if len(tetrahedra) == 0:
        return None
    keep = _circumradii(points, tetrahedra) <= alpha
    if not keep.any():
        return None
    selected = _largest_connected_tetrahedra(tetrahedra,
                                             triangulation.neighbors, keep)
    if len(selected) == 0:
        return None
    return triangulation, selected


def mesh_from_solid(points: np.ndarray, triangulation, selected
                    ) -> trimesh.Trimesh | None:
    """Outward-oriented boundary mesh of a retained tetrahedron set."""
    faces, apex = _boundary_faces(triangulation.simplices, selected)
    if len(faces) < 4:
        return None
    a, b, c = (points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]])
    normals = np.cross(b - a, c - a)
    inward = np.einsum("ij,ij->i", normals, points[apex] - a) > 0
    faces = faces.copy()
    faces[inward] = faces[inward][:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 4:
        return None
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def alpha_shape_mesh(points: np.ndarray, alpha: float) -> trimesh.Trimesh | None:
    """Closed alpha-shape surface, or None if this alpha yields nothing usable."""
    solid = alpha_solid(points, alpha)
    if solid is None:
        return None
    return mesh_from_solid(np.asarray(points, dtype=float), *solid)


def solid_coverage(triangulation, selected, n_points: int) -> float:
    """Fraction of input points incorporated into the retained solid.

    Every point of an alpha complex is either a vertex of a retained
    tetrahedron -- hence on or inside the surface -- or excluded from it. That
    is an exact combinatorial test, so no ray casting, distance query or
    tolerance is involved.
    """
    if n_points <= 0:
        return 1.0
    used = np.unique(triangulation.simplices[selected])
    return float(len(used) / n_points)


def points_inside_solid(triangulation, selected, query: np.ndarray) -> np.ndarray:
    """Boolean mask of query points falling inside the retained solid."""
    if len(query) == 0:
        return np.zeros(0, dtype=bool)
    located = triangulation.find_simplex(query)
    retained = np.zeros(len(triangulation.simplices), dtype=bool)
    retained[selected] = True
    inside = np.zeros(len(query), dtype=bool)
    valid = located >= 0
    inside[valid] = retained[located[valid]]
    return inside


def interior_support_points(points: np.ndarray, triangulation, selected,
                            spacing: float, maximum: int = 200000) -> np.ndarray:
    """Lattice points strictly inside the retained solid.

    Without them a shell-sampled crown admits no solid at fine alpha and the
    reconstruction is forced coarse -- on a test crown nothing below alpha 2.0 m
    closed. With them the boundary is still set entirely by the MEASURED points;
    these sit inside and cannot become surface vertices.

    Containment comes from ``Delaunay.find_simplex`` on the same triangulation,
    so it is exact and needs no ray casting or spatial-index package.
    """
    if spacing <= 0:
        return np.empty((0, 3))
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    axes = [np.arange(low[i] + 0.5 * spacing, high[i], spacing) for i in range(3)]
    if any(len(axis) == 0 for axis in axes):
        return np.empty((0, 3))
    if int(np.prod([float(len(axis)) for axis in axes])) > maximum:
        return np.empty((0, 3))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[points_inside_solid(triangulation, selected, grid)]


def reconstruct_crown(points: np.ndarray,
                      settings: ReconstructionSettings | None = None
                      ) -> tuple[trimesh.Trimesh, ReconstructionReport]:
    """Watertight, manifold, tight-fitting surface around one point cluster."""
    settings = settings or ReconstructionSettings()
    raw = np.asarray(points, dtype=float)
    if len(raw) < settings.minimum_points:
        raise CrownReconstructionError(
            f"cluster has {len(raw)} points, below the {settings.minimum_points} "
            "needed for a reconstruction")

    reduced = voxel_downsample(raw, settings.voxel_size)
    filtered = statistical_outlier_filter(reduced, settings.outlier_neighbours,
                                          settings.outlier_std_ratio)
    filtered = density_filter(
        filtered, settings.density_radius_voxels * settings.voxel_size,
        settings.density_min_neighbours)
    if len(filtered) < 8:
        raise CrownReconstructionError(
            "too few points survive conditioning to reconstruct a surface")

    spacing = characteristic_spacing(filtered)

    # ADAPTIVE ALPHA: smallest alpha giving a watertight, manifold solid that
    # incorporates at least coverage_target of the conditioned points. Too small
    # and the complex fragments; too large and it degenerates to the convex hull.
    alpha = settings.alpha_start_factor * spacing
    best = None
    best_alpha = alpha
    best_coverage = 0.0
    best_solid = None
    for _ in range(settings.alpha_max_steps):
        solid = alpha_solid(filtered, alpha)
        if solid is not None:
            triangulation, selected = solid
            candidate = mesh_from_solid(filtered, triangulation, selected)
            if candidate is not None and candidate.is_watertight \
                    and candidate.is_winding_consistent and candidate.volume > 0:
                coverage = solid_coverage(triangulation, selected, len(filtered))
                if coverage > best_coverage:
                    best = candidate
                    best_alpha = alpha
                    best_coverage = coverage
                    best_solid = solid
                if coverage >= settings.coverage_target:
                    break
        alpha *= settings.alpha_growth
    if best is None:
        raise CrownReconstructionError(
            "no alpha produced a watertight surface for this cluster")

    # SURFACE REFINEMENT. A LiDAR crown cloud is a near-shell, so a solid only
    # closes once alpha exceeds the crown's curvature scale -- which would cap
    # surface detail there no matter how finely the cloud is sampled. Seeding
    # the interior of the coarse solid decouples the two: the support lattice
    # makes small-alpha tetrahedra viable inside, so the boundary can follow the
    # measured surface at the target resolution while staying closed.
    refined_alpha = None
    if settings.refine_surface and best_solid is not None:
        target_alpha = max(settings.refine_alpha_voxels * settings.voxel_size,
                           2.5 * spacing)
        if target_alpha < best_alpha:
            support = interior_support_points(
                filtered, best_solid[0], best_solid[1],
                settings.refine_support_ratio * target_alpha)
            if len(support):
                combined = np.vstack([filtered, support])
                solid = alpha_solid(combined, target_alpha)
                if solid is not None:
                    candidate = mesh_from_solid(combined, *solid)
                    if candidate is not None and candidate.is_watertight \
                            and candidate.is_winding_consistent \
                            and candidate.volume > 0:
                        # Coverage is measured on the MEASURED points only; the
                        # support lattice must not be allowed to flatter it.
                        inside = points_inside_solid(solid[0], solid[1], filtered)
                        coverage = float(inside.mean())
                        if coverage >= settings.coverage_target:
                            best = candidate
                            refined_alpha = target_alpha
                            best_alpha = target_alpha
                            best_coverage = coverage

    decimated = False
    if settings.max_faces and len(best.faces) > settings.max_faces:
        reduced_mesh = _decimate(best, settings.max_faces)
        if reduced_mesh is not None and reduced_mesh.is_watertight \
                and reduced_mesh.volume > 0:
            best = reduced_mesh
            decimated = True

    if not best.is_watertight:
        raise CrownReconstructionError(
            "reconstruction produced a non-watertight surface")

    return best, ReconstructionReport(
        n_input=int(len(raw)), n_after_downsample=int(len(reduced)),
        n_after_filtering=int(len(filtered)), alpha_m=float(best_alpha),
        characteristic_spacing_m=float(spacing), coverage=float(best_coverage),
        n_faces=int(len(best.faces)), decimated=decimated,
        watertight=bool(best.is_watertight),
        refined=refined_alpha is not None)


def _decimate(mesh: trimesh.Trimesh, target_faces: int):
    """Quadric edge-collapse decimation, watertightness preserved.

    Tries pymeshlab, which is present here, and gives up quietly rather than
    returning a broken mesh -- a crown slightly over budget is far better than
    one with a hole in it.
    """
    try:
        import pymeshlab
    except ImportError:
        return None
    try:
        meshset = pymeshlab.MeshSet()
        meshset.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
        meshset.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces), preservetopology=True,
            preserveboundary=True, planarquadric=True)
        result = meshset.current_mesh()
        out = trimesh.Trimesh(vertices=result.vertex_matrix(),
                              faces=result.face_matrix(), process=True)
        return out if len(out.faces) >= 4 else None
    except Exception:
        return None
