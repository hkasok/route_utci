#!/usr/bin/env python3
"""prepare_surface_materials.py -- hierarchical surface-material classification.

Preprocessing stage. Runs ONCE per case, before any radiation work, and writes
a cached per-face material assignment that every downstream stage reads. No
radiation timestep ever re-classifies anything.

WHAT IT DOES
------------
1. Splits the building mesh into individual buildings (edge-connected
   components) and each building into wall and roof faces.
2. Matches each building to its OSM polygon so its tags are available.
3. Classifies facade and roof SEPARATELY through ``material_classification``.
4. Classifies ground, reusing the existing OSM ground material map when one is
   present, and optionally refining it with a classified imagery raster.
5. Groups faces into logical surface patches via ``surface_groups``.
6. Writes the manifest, the coverage summary and the QC products.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not touch the radiation framework, the reflection model, or run_output
conventions. Its whole output contract to the rest of the pipeline is one
``surface_material_assignment.npz`` per mesh, which 05a consumes in the same
place it already consumes the ground material map.

BACKWARD COMPATIBILITY
----------------------
Everything is optional. Without imagery and without overrides the hierarchy
degrades to OSM direct > OSM inferred > default, which is a strict superset of
what the pipeline did before: ground keeps its existing classification, and
walls/roofs gain classification where they previously had none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh

import material_classification as mc
import material_library
import surface_groups as sg
from material_library import (CATEGORY_FACADE, CATEGORY_GROUND, CATEGORY_ROOF,
                              CATEGORY_WATER)
from osm_ground_materials import (GROUND_FACE_MATERIAL_MAP,
                                  GROUND_MATERIAL_CATALOG, normalize_tag)

ASSIGNMENT_FILE = mc.SURFACE_ASSIGNMENT_FILE
MANIFEST_CSV = "material_assignment_manifest.csv"
MANIFEST_JSON = "material_assignment_manifest.json"
SUMMARY_JSON = "material_classification_summary.json"
LIBRARY_CSV = "material_property_library.csv"
LOW_CONFIDENCE_CSV = "low_confidence_surfaces.csv"
DEFAULT_SURFACES_CSV = "default_material_surfaces.csv"

MESH_ROLES = ("ground", "buildings", "vegetation")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Hierarchical surface-material classification (preprocessing)")
    parser.add_argument("--ground-stl", required=True)
    parser.add_argument("--buildings-stl", default=None)
    parser.add_argument("--vegetation-stl", default=None)
    parser.add_argument("--output-dir", required=True,
                        help="Written inside the case's existing material "
                             "preprocessing subtree; no new output root.")
    parser.add_argument("--osm-features", default=None,
                        help="GeoPackage/GeoJSON of raw OSM features, used for "
                             "building tags. Without it, buildings fall back to "
                             "explicit generic classes at default confidence.")
    parser.add_argument("--osm-layer", default="raw_complete_osm_features")
    parser.add_argument("--ground-material-dir", default=None,
                        help="Directory holding ground_face_materials.npz and "
                             "ground_material_catalog.json from the existing "
                             "OSM ground stage. Reused rather than recomputed.")
    parser.add_argument("--imagery-raster", default=None,
                        help="OPTIONAL classified raster. Applied to ground and "
                             "roofs only -- never to vertical facades.")
    parser.add_argument("--imagery-legend", default=None,
                        help="Pixel-value -> class-name JSON; defaults to the "
                             "raster path with a .classes.json suffix.")
    parser.add_argument("--overrides", default=None,
                        help="OPTIONAL material_overrides.csv. Highest priority.")
    parser.add_argument("--local-origin-x", type=float, default=0.0)
    parser.add_argument("--local-origin-y", type=float, default=0.0)
    parser.add_argument("--project-crs", default=None)
    parser.add_argument("--coplanar-angle-deg", type=float,
                        default=sg.DEFAULT_COPLANAR_ANGLE_DEG,
                        help="Maximum normal disagreement for two adjacent "
                             "triangles to join one surface group.")
    parser.add_argument("--coplanar-distance-m", type=float,
                        default=sg.DEFAULT_COPLANAR_DISTANCE_M,
                        help="Maximum out-of-plane offset for the same test. "
                             "Together with the angle this is what stops a "
                             "curved surface collapsing into one patch.")
    parser.add_argument("--roof-normal-z", type=float, default=0.7,
                        help="Upward normal component above which a building "
                             "face is a roof rather than a wall. Matches 05a.")
    parser.add_argument("--default-area-warn-fraction", type=float, default=0.20,
                        help="Warn when more than this fraction of classified "
                             "area falls back to a default class. Warning only; "
                             "never fails the run.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the QC maps.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# OSM building tags
# ---------------------------------------------------------------------------
def load_building_tags(path: str, layer: str, local_origin_x: float,
                       local_origin_y: float, project_crs: str | None):
    """Building polygons in the mesh's local metric frame, with their tags.

    Returns ``(geometries, tag_dicts)``. Geometries are shapely polygons already
    translated into the same frame the STL uses, so a mesh centroid can be
    tested against them directly.
    """
    import geopandas as gpd
    from shapely import affinity

    frame = gpd.read_file(path, layer=layer)
    if "building" not in frame.columns:
        return [], []
    frame = frame[frame["building"].notna()].copy()
    if frame.empty:
        return [], []
    if project_crs and frame.crs is not None:
        try:
            frame = frame.to_crs(project_crs)
        except Exception as error:  # pragma: no cover - CRS availability
            print(f"  WARNING: could not reproject OSM buildings ({error}); "
                  "using their stored CRS")
    if local_origin_x or local_origin_y:
        frame["geometry"] = frame.geometry.apply(
            lambda geom: affinity.translate(geom, -local_origin_x, -local_origin_y))
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    tag_columns = [column for column in frame.columns if column != "geometry"]
    geometries = list(frame.geometry.values)
    tags = frame[tag_columns].to_dict(orient="records")
    return geometries, tags


def match_buildings_to_polygons(object_centroids: np.ndarray, geometries,
                                tags: list[dict]) -> list[dict | None]:
    """Attach OSM tags to each mesh building by centroid containment.

    Containment rather than nearest-neighbour: a building whose footprint no
    polygon covers has genuinely unknown tags, and inventing them from whatever
    happens to be nearby would manufacture confident-looking provenance out of
    nothing.
    """
    if not geometries:
        return [None] * len(object_centroids)
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    tree = STRtree(geometries)
    matched: list[dict | None] = []
    for centroid in object_centroids:
        point = Point(float(centroid[0]), float(centroid[1]))
        candidates = tree.query(point)
        chosen = None
        for index in np.atleast_1d(candidates):
            index = int(index)
            if geometries[index].contains(point):
                chosen = tags[index]
                break
        matched.append(chosen)
    return matched


# ---------------------------------------------------------------------------
# Per-mesh classification
# ---------------------------------------------------------------------------
def classify_building_mesh(mesh, building_tags, args, overrides, imagery):
    """Classify a building mesh into wall and roof surface groups."""
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    centroids = np.asarray(mesh.triangles_center, dtype=float)

    print("  splitting the building mesh into individual buildings...")
    object_labels = sg.connected_object_labels(
        faces, len(faces), vertices=np.asarray(mesh.vertices, dtype=float))
    n_objects = int(object_labels.max()) + 1 if len(object_labels) else 0
    print(f"    {n_objects:,} building objects from {len(faces):,} triangles")

    object_centroids = np.zeros((n_objects, 3), dtype=float)
    for index in range(n_objects):
        selection = object_labels == index
        weights = areas[selection]
        total = weights.sum()
        if total <= 0:
            object_centroids[index] = centroids[selection].mean(axis=0)
        else:
            object_centroids[index] = (
                centroids[selection] * weights[:, None]).sum(axis=0) / total

    matched_tags = match_buildings_to_polygons(object_centroids, *building_tags)
    n_matched = sum(1 for item in matched_tags if item)
    print(f"    {n_matched:,}/{n_objects:,} matched to an OSM building polygon")

    is_roof = normals[:, 2] > args.roof_normal_z
    surface_type = np.where(is_roof, "roof", "wall")
    parent_ids = np.array([f"building_{label:05d}" for label in object_labels])

    # One classification per (building, wall/roof) -- not per triangle. Roof and
    # facade are resolved independently, so building:material never leaks onto
    # the roof and roof:material never leaks onto the facade.
    assignments: dict[tuple[int, str], mc.Assignment] = {}
    roof_imagery: dict[int, str | None] = {}
    if imagery is not None:
        roof_faces = np.flatnonzero(is_roof)
        if len(roof_faces):
            classes = imagery.class_at(centroids[roof_faces, 0],
                                       centroids[roof_faces, 1])
            for label in range(n_objects):
                member = roof_faces[object_labels[roof_faces] == label]
                if not len(member):
                    continue
                names = [classes[position] for position, face in
                         enumerate(roof_faces) if object_labels[face] == label]
                names = [name for name in names if name]
                if names:
                    values, counts = np.unique(names, return_counts=True)
                    roof_imagery[label] = str(values[np.argmax(counts)])

    for label in range(n_objects):
        tags = matched_tags[label] or {}
        parent = f"building_{label:05d}"
        for stype, category in (("wall", CATEGORY_FACADE), ("roof", CATEGORY_ROOF)):
            manual = overrides.get(parent) or overrides.get(f"{parent}_{stype}")
            if manual and material_library.get(manual).category != category:
                manual = None
            if stype == "wall":
                assignments[(label, stype)] = mc.classify_facade(
                    tags, manual_material=manual, source_object_id=parent)
            else:
                assignments[(label, stype)] = mc.classify_roof(
                    tags, imagery_class=roof_imagery.get(label),
                    manual_material=manual, source_object_id=parent)

    material_key = np.array([
        assignments[(int(object_labels[index]), surface_type[index])].material_class
        for index in range(len(faces))])

    print("  building surface groups...")
    group_index, groups = sg.build_surface_groups(
        mesh.vertices, faces, parent_ids, surface_type, material_key,
        centroids=centroids, normals=normals, areas=areas,
        coplanar_angle_deg=args.coplanar_angle_deg,
        coplanar_distance_m=args.coplanar_distance_m,
        roof_normal_z=args.roof_normal_z, id_prefix="building")
    group_assignments = [
        assignments[(int(object_labels[group.face_indices[0]]),
                     group.surface_type)]
        for group in groups]
    return group_index, groups, group_assignments


def classify_ground_mesh(mesh, args, overrides, imagery):
    """Classify the terrain mesh, reusing the existing OSM ground map."""
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    centroids = np.asarray(mesh.triangles_center, dtype=float)
    n_faces = len(faces)

    osm_material = np.full(n_faces, None, dtype=object)
    osm_source = np.full(n_faces, None, dtype=object)
    if args.ground_material_dir:
        directory = Path(args.ground_material_dir)
        map_path = directory / GROUND_FACE_MATERIAL_MAP
        catalog_path = directory / GROUND_MATERIAL_CATALOG
        if not map_path.is_file() or not catalog_path.is_file():
            raise FileNotFoundError(
                f"--ground-material-dir must contain {GROUND_FACE_MATERIAL_MAP} "
                f"and {GROUND_MATERIAL_CATALOG}")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        material_id = np.load(map_path)["material_id"]
        if len(material_id) != n_faces:
            raise ValueError(
                f"ground material map has {len(material_id)} faces but the "
                f"terrain mesh has {n_faces}; regenerate it for this mesh")
        names = np.asarray(catalog["material_names"], dtype=object)
        osm_material = names[material_id]
        # The existing stage records provenance per FEATURE, not per face; the
        # per-face map only carries the winning material. Faces holding the
        # explicit generic fallback are therefore the unclassified ones, and
        # everything else came from a matched OSM feature.
        explicit = catalog.get("face_material_source")
        if explicit is not None and len(explicit) == n_faces:
            osm_source = np.asarray(explicit, dtype=object)
        else:
            osm_source = np.where(osm_material == "generic_ground",
                                  None, "class_default").astype(object)

    imagery_classes: list[str | None] = [None] * n_faces
    if imagery is not None:
        imagery_classes = imagery.class_at(centroids[:, 0], centroids[:, 1])

    print("  classifying terrain faces...")
    # Classification is per unique (osm material, osm source, imagery class,
    # override) COMBINATION, not per face: a million-face terrain has only a
    # handful of distinct evidence states, and re-running the resolver per face
    # would be pure waste in a stage that is supposed to be cheap.
    keys = np.array([
        f"{osm_material[i]}|{osm_source[i]}|{imagery_classes[i]}"
        for i in range(n_faces)], dtype=object)
    unique_keys, inverse = np.unique(keys.astype(str), return_inverse=True)
    resolved: list[mc.Assignment] = []
    generic = material_library.DEFAULT_MATERIAL_BY_CATEGORY[CATEGORY_GROUND]
    for key in unique_keys:
        material_text, source_text, imagery_text = key.split("|")
        material_value = None if material_text in {"None", ""} else material_text
        imagery_value = None if imagery_text in {"None", ""} else imagery_text
        if material_value == generic:
            # The ground map's own fallback is a DEFAULT, not an inference. The
            # existing stage stamps unmatched terrain with the generic class, and
            # passing that through as "inferred" would have inflated the inferred
            # share by the whole uncovered area and hidden how much of the domain
            # is really unclassified.
            material_value = None
        resolved.append(mc.classify_ground(
            osm_material=material_value,
            osm_material_source=None if source_text in {"None", ""} else source_text,
            osm_source_tag=None,
            imagery_class=imagery_value,
        ))
    print(f"    {len(unique_keys)} distinct evidence states over "
          f"{n_faces:,} terrain faces")
    face_assignment = [resolved[index] for index in inverse]
    material_key = np.array([item.material_class for item in face_assignment])
    parent_ids = np.full(n_faces, "terrain", dtype=object)
    surface_type = np.full(n_faces, "ground", dtype=object)

    print("  building surface groups...")
    group_index, groups = sg.build_surface_groups(
        mesh.vertices, faces, parent_ids, surface_type, material_key,
        centroids=centroids, normals=normals, areas=areas,
        coplanar_angle_deg=args.coplanar_angle_deg,
        coplanar_distance_m=args.coplanar_distance_m,
        roof_normal_z=args.roof_normal_z, id_prefix="ground")
    group_assignments = []
    for group in groups:
        assignment = face_assignment[int(group.face_indices[0])]
        manual = overrides.get(group.surface_group_id)
        if manual and material_library.get(manual).category in {CATEGORY_GROUND,
                                                                CATEGORY_WATER}:
            assignment = mc.Assignment(
                material_class=manual, material_source=mc.SOURCE_MANUAL,
                confidence=mc.SOURCE_CONFIDENCE[mc.SOURCE_MANUAL],
                category=material_library.get(manual).category,
                source_tag="manual_override",
                source_object_id=group.surface_group_id)
        group_assignments.append(assignment)
    return group_index, groups, group_assignments


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def manifest_frame(groups, assignments, mesh_role: str) -> pd.DataFrame:
    rows = []
    for group, assignment in zip(groups, assignments):
        material = material_library.get(assignment.material_class)
        record = group.as_record()
        record.update({
            "mesh_role": mesh_role,
            "material_class": assignment.material_class,
            "material_category": material.category,
            "material_source": assignment.material_source,
            "confidence": assignment.confidence,
            "source_tag": assignment.source_tag,
            "imagery_class": assignment.imagery_class,
            "fallback_reason": assignment.fallback_reason,
            "albedo": material.albedo,
            "albedo_min": material.albedo_range[0],
            "albedo_max": material.albedo_range[1],
            "shortwave_absorptivity": material.shortwave_absorptivity,
            "emissivity": material.emissivity,
            "emissivity_min": material.emissivity_range[0],
            "emissivity_max": material.emissivity_range[1],
            "thermal_conductivity_WmK": material.thermal_conductivity_WmK,
            "density_kgm3": material.density_kgm3,
            "specific_heat_JkgK": material.specific_heat_JkgK,
            "volumetric_heat_capacity_Jm3K": material.volumetric_heat_capacity_Jm3K,
            "evaporative_efficiency": material.evaporative_efficiency,
            "rejected_candidates": "; ".join(
                f"{item['material_class']}({item['source']})"
                for item in assignment.rejected),
        })
        rows.append(record)
    return pd.DataFrame(rows)


def write_qc_figures(manifest: pd.DataFrame, output: Path) -> None:
    """Classification, confidence and provenance maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ground = manifest[manifest["surface_type"].eq("ground")]
    if ground.empty:
        return
    x = ground["centroid_x"].to_numpy(float)
    y = ground["centroid_y"].to_numpy(float)
    size = np.clip(ground["area_m2"].to_numpy(float) * 0.25, 1.0, 40.0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    classes = ground["material_class"].astype(str)
    codes, names = pd.factorize(classes)
    scatter = axes[0].scatter(x, y, c=codes, s=size, cmap="tab20", linewidths=0)
    axes[0].set_title("Material class")
    handles = [plt.Line2D([], [], marker="o", ls="", color=scatter.cmap(
        scatter.norm(index)), label=name)
        for index, name in enumerate(names)]
    axes[0].legend(handles=handles, fontsize=6, loc="upper right", ncol=2)

    confidence = axes[1].scatter(x, y, c=ground["confidence"].to_numpy(float),
                                 s=size, cmap="viridis", vmin=0.0, vmax=1.0,
                                 linewidths=0)
    axes[1].set_title("Classification confidence")
    fig.colorbar(confidence, ax=axes[1], fraction=0.046)

    sources = ground["material_source"].astype(str)
    source_codes, source_names = pd.factorize(sources)
    provenance = axes[2].scatter(x, y, c=source_codes, s=size, cmap="Set1",
                                 linewidths=0)
    axes[2].set_title("Provenance")
    handles = [plt.Line2D([], [], marker="o", ls="", color=provenance.cmap(
        provenance.norm(index)), label=name)
        for index, name in enumerate(source_names)]
    axes[2].legend(handles=handles, fontsize=7, loc="upper right")

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.suptitle("Ground material classification QC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"material_classification_qc.{suffix}", dpi=140,
                    bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HIERARCHICAL SURFACE-MATERIAL CLASSIFICATION")
    print("=" * 70)
    library_summary = material_library.validate_library()
    print(f"Material library: {library_summary['n_materials']} materials "
          f"{library_summary['by_category']}")
    print("Priority: manual > osm_direct > imagery > osm_inferred > "
          "colour_hint > default")

    overrides: dict[str, str] = {}
    if args.overrides:
        overrides = mc.load_material_overrides(args.overrides)
        print(f"Manual overrides: {len(overrides)} entries from {args.overrides}")

    imagery = None
    if args.imagery_raster:
        imagery = mc.load_imagery_classes(args.imagery_raster, args.imagery_legend)
        print(f"Imagery: {args.imagery_raster} "
              f"({len(imagery.class_names)} classes); applied to ground and "
              "roofs only, never to vertical facades")
    else:
        print("Imagery: none (optional input); hierarchy is "
              "OSM direct > OSM inferred > default")

    building_tags: tuple[list, list] = ([], [])
    if args.osm_features and args.buildings_stl:
        building_tags = load_building_tags(
            args.osm_features, args.osm_layer, args.local_origin_x,
            args.local_origin_y, args.project_crs)
        print(f"OSM building polygons: {len(building_tags[0])}")

    manifests: list[pd.DataFrame] = []
    assignment_payload: dict[str, Any] = {}

    stl_by_role = {"ground": args.ground_stl, "buildings": args.buildings_stl}
    for role, path in stl_by_role.items():
        if not path or not Path(path).is_file():
            continue
        print(f"\n[{role}] {path}")
        mesh = trimesh.load(path, process=False, force="mesh")
        print(f"  {len(mesh.faces):,} triangles")
        if role == "ground":
            group_index, groups, assignments = classify_ground_mesh(
                mesh, args, overrides, imagery)
        else:
            group_index, groups, assignments = classify_building_mesh(
                mesh, building_tags, args, overrides, imagery)
        print(f"  {len(groups):,} surface groups")
        frame = manifest_frame(groups, assignments, role)
        manifests.append(frame)
        material_names = [item.material_class for item in assignments]
        catalog = sorted(set(material_names))
        lookup = {name: index for index, name in enumerate(catalog)}
        assignment_payload[f"{role}_group_index"] = group_index.astype(np.int32)
        assignment_payload[f"{role}_group_material_id"] = np.array(
            [lookup[name] for name in material_names], dtype=np.int32)
        assignment_payload[f"{role}_material_names"] = np.array(catalog, dtype="U64")
        assignment_payload[f"{role}_group_source"] = np.array(
            [item.material_source for item in assignments], dtype="U24")
        assignment_payload[f"{role}_group_confidence"] = np.array(
            [item.confidence for item in assignments], dtype=np.float32)
        assignment_payload[f"{role}_group_id"] = np.array(
            [group.surface_group_id for group in groups], dtype="U80")

    if not manifests:
        print("No meshes to classify.", file=sys.stderr)
        return 2

    manifest = pd.concat(manifests, ignore_index=True)
    duplicated = manifest["surface_group_id"].duplicated()
    if duplicated.any():
        raise RuntimeError(
            "surface group ids are not unique: "
            f"{manifest.loc[duplicated, 'surface_group_id'].head().tolist()}")
    manifest.to_csv(output / MANIFEST_CSV, index=False)
    manifest.to_json(output / MANIFEST_JSON, orient="records", indent=1)
    np.savez_compressed(output / ASSIGNMENT_FILE, **assignment_payload)
    pd.DataFrame(material_library.manifest_records()).to_csv(
        output / LIBRARY_CSV, index=False)

    low = manifest[manifest["confidence"] < mc.LOW_CONFIDENCE_THRESHOLD]
    low.sort_values("area_m2", ascending=False).to_csv(
        output / LOW_CONFIDENCE_CSV, index=False)
    defaults = manifest[manifest["material_source"].eq(mc.SOURCE_DEFAULT)]
    defaults.sort_values("area_m2", ascending=False).to_csv(
        output / DEFAULT_SURFACES_CSV, index=False)

    total_area = float(manifest["area_m2"].sum())
    by_source = manifest.groupby("material_source")["area_m2"].sum()
    by_material = manifest.groupby("material_class")["area_m2"].sum()
    by_type = manifest.groupby("surface_type")["area_m2"].sum()
    default_fraction = float(by_source.get(mc.SOURCE_DEFAULT, 0.0) / max(total_area, 1e-9))
    summary = {
        "n_surface_groups": int(len(manifest)),
        "total_area_m2": total_area,
        "area_by_source_m2": by_source.to_dict(),
        "fraction_by_source": (by_source / max(total_area, 1e-9)).to_dict(),
        "area_by_material_m2": by_material.to_dict(),
        "fraction_by_material": (by_material / max(total_area, 1e-9)).to_dict(),
        "area_by_surface_type_m2": by_type.to_dict(),
        "default_area_fraction": default_fraction,
        "low_confidence_area_fraction": float(
            low["area_m2"].sum() / max(total_area, 1e-9)),
        "mean_confidence_area_weighted": float(
            (manifest["confidence"] * manifest["area_m2"]).sum()
            / max(total_area, 1e-9)),
        "classification_priority": ["manual_override", "osm_direct", "imagery",
                                    "osm_inferred", "colour_hint", "default"],
        "imagery_used": bool(args.imagery_raster),
        "imagery_allowed_categories": sorted(mc.IMAGERY_ALLOWED_CATEGORIES),
        "overrides_used": len(overrides),
        "grouping": {"coplanar_angle_deg": args.coplanar_angle_deg,
                     "coplanar_distance_m": args.coplanar_distance_m,
                     "roof_normal_z": args.roof_normal_z},
        "material_library": library_summary,
    }
    (output / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n",
                                       encoding="utf-8")

    if not args.no_figures:
        try:
            write_qc_figures(manifest, output)
        except Exception as error:  # pragma: no cover - plotting is diagnostic
            print(f"  WARNING: QC figures skipped ({error})")

    print("\n" + "=" * 70)
    print("CLASSIFICATION COVERAGE")
    print("=" * 70)
    print(f"Surface groups : {len(manifest):,}")
    print(f"Total area     : {total_area:,.0f} m2")
    print("\nBy provenance:")
    for source, area in by_source.sort_values(ascending=False).items():
        print(f"  {source:16s} {area / total_area:6.1%}  ({area:,.0f} m2)")
    print("\nTop material classes by area:")
    for name, area in by_material.sort_values(ascending=False).head(12).items():
        print(f"  {name:26s} {area / total_area:6.1%}  ({area:,.0f} m2)")
    print("\nBy surface type:")
    for name, area in by_type.sort_values(ascending=False).items():
        print(f"  {name:16s} {area / total_area:6.1%}  ({area:,.0f} m2)")
    print(f"\nArea-weighted mean confidence: "
          f"{summary['mean_confidence_area_weighted']:.2f}")
    if default_fraction > args.default_area_warn_fraction:
        print(f"\nWARNING: {default_fraction:.1%} of classified area uses a "
              f"DEFAULT material class, above the "
              f"{args.default_area_warn_fraction:.0%} threshold. This is a "
              f"warning, not a failure. See {DEFAULT_SURFACES_CSV} for the "
              "largest offenders; supplying imagery or material overrides is "
              "the usual remedy.")
    print(f"\nWritten to {output}")
    print(f"[surface_materials] groups={len(manifest)} "
          f"area_m2={total_area:.0f} default_fraction={default_fraction:.3f} "
          f"output_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
