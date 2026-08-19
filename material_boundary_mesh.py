"""Create a terrain mesh whose triangle edges conform to OSM materials.

The normal OSM ground stage deliberately leaves the source terrain geometry
unchanged and assigns one material ID per existing face.  The optional
full-surface radiation/microclimate stage needs an exact interface instead:
only triangles crossed by a material boundary are clipped in plan, triangulated
again, and lifted onto the original 3-D triangle plane.  Away from boundaries,
the source facets are retained byte-for-byte.

STL cannot retain material IDs, so the authoritative companion file is
``ground_material_conforming.npz``.  It stores the final vertices/faces,
material ID and original parent-face ID.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import triangulate, unary_union
from shapely.strtree import STRtree


def _lift_xy_to_triangle(xy: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Lift plan coordinates to the plane of a non-vertical terrain triangle."""
    tri = np.asarray(triangle, dtype=float)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if abs(normal[2]) < 1.0e-10:
        raise ValueError("terrain triangle is vertical and cannot be partitioned in plan")
    z = tri[0, 2] - (normal[0] * (xy[:, 0] - tri[0, 0])
                     + normal[1] * (xy[:, 1] - tri[0, 1])) / normal[2]
    return np.column_stack([xy, z])


def _triangulate_piece(piece, parent_triangle: np.ndarray,
                       area_tolerance: float) -> list[np.ndarray]:
    """Constrained-by-filtering triangulation of a clipped plan polygon."""
    if piece.is_empty or piece.area <= area_tolerance:
        return []
    polygons = ([piece] if piece.geom_type == "Polygon"
                else [g for g in getattr(piece, "geoms", []) if g.geom_type == "Polygon"])
    output: list[np.ndarray] = []
    for polygon in polygons:
        try:
            vertices_2d, faces_2d = trimesh.creation.triangulate_polygon(
                polygon, engine="earcut")
            candidates = [np.asarray(vertices_2d)[face]
                          for face in np.asarray(faces_2d, dtype=np.int64)]
        except (ImportError, ValueError):
            # Functional fallback for environments without mapbox-earcut.
            # It is exact for convex pieces; a post-remesh area guard rejects
            # any concave case where unconstrained Delaunay filtering loses area.
            candidates = [np.asarray(candidate.exterior.coords[:3], dtype=float)
                          for candidate in triangulate(polygon)
                          if candidate.area > area_tolerance
                          and polygon.covers(candidate)]
        for coords in candidates:
            edge_a, edge_b = coords[1] - coords[0], coords[2] - coords[0]
            plan_double_area = abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
            if 0.5 * plan_double_area <= area_tolerance:
                continue
            lifted = _lift_xy_to_triangle(coords, parent_triangle)
            parent_normal = np.cross(parent_triangle[1] - parent_triangle[0],
                                     parent_triangle[2] - parent_triangle[0])
            new_normal = np.cross(lifted[1] - lifted[0], lifted[2] - lifted[0])
            if np.dot(parent_normal, new_normal) < 0:
                lifted[[1, 2]] = lifted[[2, 1]]
            output.append(lifted)
    return output


def build_material_conforming_ground(
        ground_stl: str | Path, material_directory: str | Path,
        output_directory: str | Path, *, tolerance_m: float = 0.01,
        area_tolerance_m2: float = 1.0e-8) -> dict:
    """Partition terrain triangles at the final OSM material boundaries."""
    ground_stl = Path(ground_stl)
    material_directory = Path(material_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(ground_stl, force="mesh", process=False)
    triangles = np.asarray(mesh.triangles, dtype=float)
    if not len(triangles):
        raise ValueError("ground mesh contains no triangles")
    # Vertical edge-wall triangles occur legitimately in terrain/water STLs.
    # They have no 2-D plan area, so an OSM footprint cannot subdivide them;
    # retain them unchanged with their existing face material. Only facets
    # with meaningful plan area participate in polygon clipping.
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    plan_area = 0.5 * np.abs(cross[:, 2])
    partitionable = plan_area > area_tolerance_m2

    gpkg = material_directory / "osm_ground_materials.gpkg"
    catalog_path = material_directory / "ground_material_catalog.json"
    face_map_path = material_directory / "ground_face_materials.npz"
    for required in (gpkg, catalog_path, face_map_path):
        if not required.is_file():
            raise FileNotFoundError(f"required OSM material artifact missing: {required}")
    frame = gpd.read_file(gpkg, layer="final_ground_materials")
    if frame.crs is None or getattr(frame.crs, "is_geographic", False):
        raise ValueError("material-boundary partition requires the scene projected metric CRS")
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()].copy()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    material_names = list(catalog["material_names"])
    name_to_id = {name: index for index, name in enumerate(material_names)}
    unknown = sorted(set(frame["assigned_material"]) - set(name_to_id))
    if unknown:
        raise ValueError(f"material polygons reference unknown catalog names: {unknown}")
    old_ids = np.load(face_map_path)["material_id"].astype(np.int32)
    if old_ids.shape != (len(triangles),):
        raise ValueError("ground material face map does not match ground STL")

    # Index individual exterior/interior rings. Querying one dissolved campus-
    # scale MultiLineString for every terrain face is correct but needlessly
    # expensive because GEOS repeatedly traverses the full boundary graph.
    boundary_geometry: list[object] = []
    for geometry in frame.geometry:
        parts = ([geometry] if geometry.geom_type == "Polygon"
                 else [part for part in getattr(geometry, "geoms", [])
                       if part.geom_type == "Polygon"])
        for polygon in parts:
            boundary_geometry.append(polygon.exterior)
            boundary_geometry.extend(list(polygon.interiors))
    boundary_tree = STRtree(boundary_geometry)
    try:
        import shapely
        boundary_faces: list[np.ndarray] = []
        for start in range(0, len(triangles), 50_000):
            stop = min(start + 50_000, len(triangles))
            ids = np.flatnonzero(partitionable[start:stop]) + start
            if not len(ids):
                continue
            footprints = shapely.polygons(triangles[ids, :, :2])
            pairs = np.asarray(boundary_tree.query(
                footprints, predicate="intersects"), dtype=np.int64)
            if pairs.size:
                boundary_faces.append(ids[np.unique(pairs[0])])
        boundary_ids = (np.concatenate(boundary_faces) if boundary_faces
                        else np.empty(0, dtype=np.int64))
    except (ImportError, AttributeError):
        boundary = unary_union(boundary_geometry)
        boundary_ids = np.asarray([
            i for i, triangle in enumerate(triangles)
            if partitionable[i] and Polygon(triangle[:, :2]).intersects(boundary)
        ], dtype=np.int64)
    boundary_set = set(boundary_ids.tolist())

    # Index individual polygon components rather than the large dissolved
    # multipolygons stored in the GeoPackage. This is essential for campus-
    # scale cases: testing every boundary facet against an all-domain generic-
    # ground multipolygon is correct but prohibitively slow.
    component_geometry: list[object] = []
    component_material: list[int] = []
    for row in frame.itertuples():
        geometry = row.geometry
        parts = ([geometry] if geometry.geom_type == "Polygon"
                 else [part for part in getattr(geometry, "geoms", [])
                       if part.geom_type == "Polygon"])
        component_geometry.extend(parts)
        component_material.extend(
            [name_to_id[row.assigned_material]] * len(parts))
    component_tree = STRtree(component_geometry)
    component_by_identity = {id(geometry): index
                             for index, geometry in enumerate(component_geometry)}

    final_triangles: list[np.ndarray] = []
    final_material: list[int] = []
    final_parent: list[int] = []
    split_source_faces = 0
    for face_id, triangle in enumerate(triangles):
        if face_id not in boundary_set:
            final_triangles.append(triangle)
            final_material.append(int(old_ids[face_id]))
            final_parent.append(face_id)
            continue
        footprint = Polygon(triangle[:, :2])
        pieces: list[tuple[int, object]] = []
        try:
            candidates = component_tree.query(footprint, predicate="intersects")
        except TypeError:  # Shapely 1.x has no predicate argument.
            candidates = component_tree.query(footprint)
        candidates = np.asarray(candidates)
        if candidates.dtype.kind in "iu":  # Shapely 2.x returns indices.
            component_ids = candidates.astype(int).tolist()
        else:  # Shapely 1.x returns the geometry objects.
            component_ids = [component_by_identity[id(geometry)]
                             for geometry in candidates.tolist()
                             if id(geometry) in component_by_identity]
        for component_id in component_ids:
            geometry = component_geometry[component_id]
            if not footprint.intersects(geometry):
                continue
            clipped = footprint.intersection(geometry)
            if not clipped.is_empty and clipped.area > area_tolerance_m2:
                pieces.append((component_material[component_id], clipped))
        made = 0
        for material_id, piece in pieces:
            for new_triangle in _triangulate_piece(piece, triangle, area_tolerance_m2):
                final_triangles.append(new_triangle)
                final_material.append(material_id)
                final_parent.append(face_id)
                made += 1
        if made == 0:
            # Numerically touching a boundary without a meaningful split.
            final_triangles.append(triangle)
            final_material.append(int(old_ids[face_id]))
            final_parent.append(face_id)
        elif made > 1:
            split_source_faces += 1

    tri_array = np.asarray(final_triangles, dtype=float)
    material_id = np.asarray(final_material, dtype=np.int16)
    parent_id = np.asarray(final_parent, dtype=np.int64)
    # Merge coincident vertices only after exact material clipping.
    flat = tri_array.reshape(-1, 3)
    # Polygon-operation vertices that should be shared differ only at floating
    # roundoff scale. Using the centimetre-scale geometry tolerance here can
    # merge physically distinct nearby material vertices and measurably alter
    # surface area on a large domain. Keep topology deduplication sub-micrometre.
    vertex_merge_tolerance_m = min(float(tolerance_m), 1.0e-7)
    quantized = np.round(flat / vertex_merge_tolerance_m).astype(np.int64)
    _, unique_index, inverse = np.unique(quantized, axis=0,
                                         return_index=True, return_inverse=True)
    vertices = flat[unique_index]
    faces = inverse.reshape(-1, 3)
    final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    nondegenerate = final_mesh.area_faces > area_tolerance_m2
    if not nondegenerate.all():
        final_mesh.update_faces(nondegenerate)
        material_id = material_id[nondegenerate]
        parent_id = parent_id[nondegenerate]
        final_mesh.remove_unreferenced_vertices()

    source_area = float(mesh.area)
    final_area = float(final_mesh.area)
    relative_error = abs(final_area - source_area) / max(source_area, 1e-12)
    if relative_error > 5.0e-5:
        raise RuntimeError(
            "material-conforming remesh failed area conservation: "
            f"source={source_area:.6f} m2, final={final_area:.6f} m2, "
            f"relative={relative_error:.3e}")
    if len(material_id) != len(final_mesh.faces):
        raise RuntimeError("material ID count differs from conforming mesh faces")

    final_mesh.export(output_directory / "ground_material_conforming.stl")
    np.savez_compressed(
        output_directory / "ground_material_conforming.npz",
        vertices=np.asarray(final_mesh.vertices), faces=np.asarray(final_mesh.faces),
        material_id=material_id, parent_face_id=parent_id)
    (output_directory / "ground_material_catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8")
    report = {
        "method": "OSM polygon clipping and boundary-conforming terrain retriangulation",
        "source_faces": int(len(mesh.faces)),
        "preserved_nonplanar_edge_faces": int((~partitionable).sum()),
        "boundary_touched_source_faces": int(len(boundary_ids)),
        "actually_split_source_faces": int(split_source_faces),
        "final_faces": int(len(final_mesh.faces)),
        "source_surface_area_m2": source_area,
        "final_surface_area_m2": final_area,
        "surface_area_error_m2": final_area - source_area,
        "relative_surface_area_error": relative_error,
        "vertex_merge_tolerance_m": vertex_merge_tolerance_m,
        "material_face_counts": {
            name: int(np.sum(material_id == index))
            for index, name in enumerate(material_names)},
        "terrain_elevation_preserved": True,
        "route_geometry_modified": False,
    }
    (output_directory / "material_conforming_mesh_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
