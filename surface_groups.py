"""surface_groups.py -- group triangles into logical physical surfaces.

A classified STL is a soup of triangles. "The east wall of building 12" is the
thing a material belongs to, not any one of the several hundred triangles that
happen to tile it. This module turns the soup into named surface groups so that
material assignment, provenance and QA all operate at the level a human can
actually check.

MERGE RULE
----------
Two adjacent triangles join the same group only when ALL of these hold:

1. they share an edge (spatial adjacency -- not merely nearness);
2. same parent object;
3. same surface type (wall / roof / ground);
4. same material candidate key;
5. their normals agree within ``coplanar_angle_deg``;
6. each centroid lies within ``coplanar_distance_m`` of the other's plane.

Conditions 2-4 are what stop the merge being purely geometric. Two coplanar
triangles belonging to different buildings, or to the same building but carrying
different material evidence, stay apart no matter how flat the join is --
numerical coplanarity is not physical identity.

Conditions 5 and 6 together are what stop a curved surface collapsing into one
patch. Angle alone would let a cylinder tile round in small steps and merge end
to end; requiring each triangle to sit near the other's PLANE breaks that walk,
because the accumulated offset grows even while each successive angle stays
small.

The grouping is a preprocessing step. It runs once, its result is cached in the
facet arrays, and no radiation timestep ever re-derives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_COPLANAR_ANGLE_DEG = 15.0
DEFAULT_COPLANAR_DISTANCE_M = 0.25


class _UnionFind:
    """Union-find with path halving and union by size."""

    __slots__ = ("parent", "size")

    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.size = np.ones(count, dtype=np.int64)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return int(item)

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


@dataclass
class SurfaceGroup:
    """One logical physical surface."""

    surface_group_id: str
    parent_object_id: str
    surface_type: str
    face_indices: np.ndarray
    area_m2: float
    mean_normal: tuple[float, float, float]
    centroid: tuple[float, float, float]
    material_key: str
    orientation: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "surface_group_id": self.surface_group_id,
            "parent_object_id": self.parent_object_id,
            "surface_type": self.surface_type,
            "orientation": self.orientation,
            "n_faces": int(len(self.face_indices)),
            "area_m2": float(self.area_m2),
            "mean_normal_x": float(self.mean_normal[0]),
            "mean_normal_y": float(self.mean_normal[1]),
            "mean_normal_z": float(self.mean_normal[2]),
            "centroid_x": float(self.centroid[0]),
            "centroid_y": float(self.centroid[1]),
            "centroid_z": float(self.centroid[2]),
        }


def compass_orientation(normal: Sequence[float],
                        roof_normal_z: float = 0.7) -> str:
    """Human-readable orientation, so a group id means something on sight.

    ``building_12_east_wall`` is checkable by a person; ``group_8137`` is not.
    """
    nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
    if nz >= roof_normal_z:
        return "roof"
    if nz <= -roof_normal_z:
        return "underside"
    if abs(nx) < 1e-12 and abs(ny) < 1e-12:
        return "flat"
    bearing = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    sectors = ["north", "northeast", "east", "southeast",
               "south", "southwest", "west", "northwest"]
    return sectors[int(((bearing + 22.5) % 360.0) // 45.0)]


DEFAULT_WELD_TOLERANCE_M = 1e-6


def weld_face_indices(vertices: np.ndarray, faces: np.ndarray,
                      tolerance_m: float = DEFAULT_WELD_TOLERANCE_M
                      ) -> np.ndarray:
    """Re-index faces onto de-duplicated vertices, preserving face order.

    THIS IS NOT OPTIONAL FOR STL INPUT. The STL format stores three independent
    vertex triples per triangle, so a mesh loaded with ``process=False`` has no
    two triangles sharing a vertex index -- and any edge-adjacency test then
    reports that nothing touches anything. Grouping such a mesh yields exactly
    one group per triangle, which looks like it worked and is worthless.

    The pipeline loads meshes with ``process=False`` on purpose, because face
    ORDER and COUNT are the contract that ties facets back to the terrain
    material map. So this welds only the vertex INDICES used for adjacency; the
    returned array has the same shape and row order as ``faces``, and the
    caller's geometry arrays stay valid.

    Coordinates are quantised to ``tolerance_m`` before matching, so vertices
    that differ only in float noise are recognised as the same point.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces
    if tolerance_m <= 0.0:
        raise ValueError("weld tolerance must be positive")
    quantised = np.round(vertices / float(tolerance_m)).astype(np.int64)
    _, inverse = np.unique(quantised, axis=0, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64)[faces]


def face_adjacency_pairs(faces: np.ndarray) -> np.ndarray:
    """Pairs of face indices sharing an edge.

    Written directly rather than taken from trimesh so that grouping works on
    plain arrays -- the callers already hold faces/vertices and should not have
    to keep a mesh object alive just to ask which triangles touch.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    owners = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    left = owners[:-1][same]
    right = owners[1:][same]
    keep = left != right
    return np.column_stack([left[keep], right[keep]])


def build_surface_groups(
        vertices: np.ndarray,
        faces: np.ndarray,
        parent_object_id: Sequence[Any],
        surface_type: Sequence[str],
        material_key: Sequence[str],
        *,
        centroids: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        areas: np.ndarray | None = None,
        coplanar_angle_deg: float = DEFAULT_COPLANAR_ANGLE_DEG,
        coplanar_distance_m: float = DEFAULT_COPLANAR_DISTANCE_M,
        roof_normal_z: float = 0.7,
        id_prefix: str = "surface",
        weld_tolerance_m: float = DEFAULT_WELD_TOLERANCE_M,
) -> tuple[np.ndarray, list[SurfaceGroup]]:
    """Group faces into surface patches.

    Returns ``(group_index_per_face, groups)``. ``group_index_per_face`` indexes
    into ``groups``, so every face belongs to exactly one group and the manifest
    can never double-count or miss a surface.
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=float)
    n_faces = len(faces)
    parent_object_id = np.asarray(parent_object_id).astype(str)
    surface_type = np.asarray(surface_type).astype(str)
    material_key = np.asarray(material_key).astype(str)
    for name, array in (("parent_object_id", parent_object_id),
                        ("surface_type", surface_type),
                        ("material_key", material_key)):
        if len(array) != n_faces:
            raise ValueError(f"{name} has {len(array)} entries for {n_faces} faces")
    if n_faces == 0:
        return np.empty(0, dtype=np.int64), []

    triangles = vertices[faces]
    if centroids is None:
        centroids = triangles.mean(axis=1)
    else:
        centroids = np.asarray(centroids, dtype=float)
    if normals is None or areas is None:
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
        magnitude = np.linalg.norm(cross, axis=1)
        computed_areas = 0.5 * magnitude
        computed_normals = cross / np.maximum(magnitude, 1e-30)[:, None]
        if normals is None:
            normals = computed_normals
        if areas is None:
            areas = computed_areas
    normals = np.asarray(normals, dtype=float)
    areas = np.asarray(areas, dtype=float)
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-30)[:, None]

    cosine_tolerance = float(np.cos(np.deg2rad(coplanar_angle_deg)))
    # Weld before asking what touches what -- see weld_face_indices.
    pairs = face_adjacency_pairs(
        weld_face_indices(vertices, faces, weld_tolerance_m))
    union = _UnionFind(n_faces)
    if len(pairs):
        left, right = pairs[:, 0], pairs[:, 1]
        same_label = (
            (parent_object_id[left] == parent_object_id[right])
            & (surface_type[left] == surface_type[right])
            & (material_key[left] == material_key[right]))
        alignment = np.einsum("ij,ij->i", unit[left], unit[right])
        aligned = alignment >= cosine_tolerance
        # Symmetric plane test: each centroid must lie near the OTHER's plane.
        # Testing one direction only would let a fan of thin slivers creep
        # around a curve, because the reference plane would keep moving.
        offset = centroids[right] - centroids[left]
        distance_left = np.abs(np.einsum("ij,ij->i", offset, unit[left]))
        distance_right = np.abs(np.einsum("ij,ij->i", offset, unit[right]))
        coplanar = ((distance_left <= coplanar_distance_m)
                    & (distance_right <= coplanar_distance_m))
        mergeable = same_label & aligned & coplanar
        for index in np.flatnonzero(mergeable):
            union.union(int(left[index]), int(right[index]))

    roots = np.array([union.find(index) for index in range(n_faces)],
                     dtype=np.int64)
    unique_roots, group_index = np.unique(roots, return_inverse=True)

    groups: list[SurfaceGroup] = []
    order = np.argsort(group_index, kind="stable")
    boundaries = np.searchsorted(group_index[order], np.arange(len(unique_roots) + 1))
    counters: dict[tuple[str, str], int] = {}
    for slot in range(len(unique_roots)):
        member_faces = order[boundaries[slot]:boundaries[slot + 1]]
        group_area = float(areas[member_faces].sum())
        weights = areas[member_faces]
        weight_total = float(weights.sum())
        if weight_total <= 0.0:
            weights = np.ones_like(weights)
            weight_total = float(weights.sum())
        mean_normal = (unit[member_faces] * weights[:, None]).sum(axis=0) / weight_total
        norm = float(np.linalg.norm(mean_normal))
        mean_normal = mean_normal / norm if norm > 1e-12 else unit[member_faces[0]]
        group_centroid = ((centroids[member_faces] * weights[:, None]).sum(axis=0)
                          / weight_total)
        parent = str(parent_object_id[member_faces[0]])
        stype = str(surface_type[member_faces[0]])
        orientation = compass_orientation(mean_normal, roof_normal_z)
        key = (parent, stype)
        counters[key] = counters.get(key, 0) + 1
        if stype == "ground":
            group_id = (f"{id_prefix}_{parent}_{material_key[member_faces[0]]}"
                        f"_{counters[key]:04d}")
        else:
            group_id = f"{parent}_{orientation}_{stype}_{counters[key]:03d}"
        groups.append(SurfaceGroup(
            surface_group_id=group_id,
            parent_object_id=parent,
            surface_type=stype,
            face_indices=member_faces,
            area_m2=group_area,
            mean_normal=(float(mean_normal[0]), float(mean_normal[1]),
                         float(mean_normal[2])),
            centroid=(float(group_centroid[0]), float(group_centroid[1]),
                      float(group_centroid[2])),
            material_key=str(material_key[member_faces[0]]),
            orientation=orientation,
        ))
    return group_index, groups


def connected_object_labels(faces: np.ndarray, n_faces: int | None = None,
                            vertices: np.ndarray | None = None,
                            weld_tolerance_m: float = DEFAULT_WELD_TOLERANCE_M
                            ) -> np.ndarray:
    """Label edge-connected components of a face set -- i.e. separate objects.

    A merged building STL holds every building in one mesh; its connected
    components are the individual buildings, which is what ``parent_object_id``
    needs to be so that one building's wall can never merge into its neighbour's.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if vertices is not None:
        faces = weld_face_indices(vertices, faces, weld_tolerance_m)
    count = int(n_faces if n_faces is not None else len(faces))
    union = _UnionFind(count)
    for left, right in face_adjacency_pairs(faces):
        union.union(int(left), int(right))
    roots = np.array([union.find(index) for index in range(count)],
                     dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def group_summary(groups: Iterable[SurfaceGroup]) -> dict[str, Any]:
    groups = list(groups)
    by_type: dict[str, dict[str, float]] = {}
    for group in groups:
        record = by_type.setdefault(group.surface_type,
                                    {"count": 0.0, "area_m2": 0.0})
        record["count"] += 1
        record["area_m2"] += group.area_m2
    return {
        "n_groups": len(groups),
        "total_area_m2": float(sum(group.area_m2 for group in groups)),
        "by_surface_type": by_type,
        "median_faces_per_group": float(np.median(
            [len(group.face_indices) for group in groups])) if groups else 0.0,
    }
