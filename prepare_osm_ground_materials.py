"""Build the OSM-derived ground-material branch for TREC-Route.

The authoritative routing graph is read-only.  This command consumes its
projected edge export, constructs material polygons, and assigns material IDs
to the existing terrain faces without changing their geometry or elevations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely import affinity
from pyproj import CRS
from shapely.geometry import MultiPoint

from osm_ground_materials import (
    GROUND_FACE_MATERIAL_MAP, GROUND_MATERIAL_CATALOG,
    assign_materials_boundary_aware, assign_materials_to_face_centroids,
    classify_osm_feature,
    load_osm_ground_config, normalize_tag, repair_polygon,
    resolve_feature_material_overlaps, resolve_feature_width,
    validate_projected_crs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify OSM transportation surfaces as terrain materials; routing is unchanged")
    parser.add_argument("--osm-features", required=True,
                        help="Projected OSM edge/polygon GeoJSON or GeoPackage")
    parser.add_argument("--osm-layer", default="raw_complete_osm_features",
                        help="GeoPackage layer containing cached complete OSM features")
    parser.add_argument("--ground-mesh", required=True,
                        help="Existing terrain STL/mesh whose face IDs are classified")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="osm_ground_materials.json")
    parser.add_argument("--routes-pkl", default=None,
                        help="Existing route polylines, used only as a diagnostic overlay")
    parser.add_argument("--buildings-mesh", default=None)
    parser.add_argument("--vegetation-mesh", default=None)
    parser.add_argument("--local-origin-x", type=float, default=0.0)
    parser.add_argument("--local-origin-y", type=float, default=0.0)
    parser.add_argument("--overrides", default=None,
                        help="Optional JSON, CSV, or GeoPackage overrides keyed by OSM ID")
    return parser.parse_args()


def _scalar_osm_id(value: object) -> str:
    tag = normalize_tag(value)
    return tag or "unknown"


def _override_flag(value: object) -> bool:
    """Parse override booleans consistently across JSON, CSV, and GPKG."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "include", "included"}:
            return True
        if normalized in {"false", "no", "0", "exclude", "excluded", ""}:
            return False
        raise ValueError(f"invalid override boolean: {value!r}")
    return bool(value)


def _mesh_signature(mesh: trimesh.Trimesh) -> dict[str, object]:
    payload = np.asarray(mesh.faces, dtype=np.int64).tobytes()
    payload += np.asarray(mesh.vertices, dtype=np.float64).tobytes()
    return {
        "n_faces": int(len(mesh.faces)),
        "n_vertices": int(len(mesh.vertices)),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
        "sha256_vertices_faces": hashlib.sha256(payload).hexdigest(),
    }


def _load_overrides(config: dict, override_path: str | None,
                    target_crs: CRS) -> dict[str, dict]:
    overrides = dict(config.get("manual_overrides", {}))
    if override_path:
        path = Path(override_path)
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as stream:
                overrides.update(json.load(stream))
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if "osm_id" not in frame:
                raise ValueError("CSV override file requires an osm_id column")
            for _, row in frame.iterrows():
                overrides[str(row["osm_id"])] = {
                    key: value for key, value in row.items()
                    if key != "osm_id" and pd.notna(value)}
        elif path.suffix.lower() == ".gpkg":
            frame = gpd.read_file(path)
            if "osm_id" not in frame:
                raise ValueError("GeoPackage override file requires an osm_id field")
            if frame.crs is None:
                raise ValueError("GeoPackage override file requires a declared CRS")
            if not CRS.from_user_input(frame.crs).equals(target_crs):
                frame = frame.to_crs(target_crs)
            for _, row in frame.iterrows():
                record = {key: value for key, value in row.items()
                          if key not in {"osm_id", "geometry"} and pd.notna(value)}
                if row.geometry is not None and not row.geometry.is_empty:
                    record["geometry"] = row.geometry
                overrides[str(row["osm_id"])] = record
        else:
            raise ValueError("override file must be JSON, CSV, or GeoPackage")
    return {str(key): value for key, value in overrides.items()}


def _deduplicate_directed_edges(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove exact duplicate geometries while retaining distinct OSM elements."""
    seen: set[tuple[str, str, bytes]] = set()
    keep = []
    for index, row in frame.iterrows():
        geom = row.geometry
        normalized = geom.normalize() if geom is not None else geom
        key = (str(row.get("osm_type", "way")),
               _scalar_osm_id(row.get("osm_id", row.get("osmid"))),
               normalized.wkb if normalized is not None else b"")
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return frame.loc[keep].copy().reset_index(drop=True)


def classify_features(frame: gpd.GeoDataFrame, config: dict,
                      overrides: dict[str, dict]) -> gpd.GeoDataFrame:
    """Classify, width-resolve, and buffer a copy of OSM features."""
    records = []
    for _, row in frame.iterrows():
        tags = row.to_dict()
        raw_tags = tags.get("all_tags_json")
        if isinstance(raw_tags, str):
            try:
                tags.update(json.loads(raw_tags))
            except json.JSONDecodeError:
                pass
        osm_id = _scalar_osm_id(tags.get("osm_id", tags.get("osmid")))
        osm_type = str(tags.get("osm_type", "way"))
        override = overrides.get(osm_id, {})
        classified = classify_osm_feature(tags, config)
        include_override = next((override[key] for key in
                                 ("include", "inclusion", "included_for_ground_material")
                                 if key in override), None)
        explicit_exclusion = any(
            _override_flag(override[key]) for key in ("exclude", "exclusion")
            if key in override)
        if explicit_exclusion:
            include_override = False
        if include_override is not None:
            include_override = _override_flag(include_override)
            classified["included_for_ground_material"] = include_override
            if not include_override:
                classified["rejection_reason"] = "manual_exclusion"
        for source_keys, target_key in (
            (("surface_class", "class", "assigned_surface_class"),
             "assigned_surface_class"),
            (("material", "assigned_material"), "assigned_material"),
        ):
            source_key = next((key for key in source_keys if key in override), None)
            if source_key is not None:
                classified[target_key] = override[source_key]
                classified["material_source"] = "manual_override"
                classified["included_for_ground_material"] = True
                classified["rejection_reason"] = None
        if explicit_exclusion:
            classified["included_for_ground_material"] = False
            classified["rejection_reason"] = "manual_exclusion"
        geom = override.get("geometry", row.geometry)
        is_polygon = geom is not None and geom.geom_type in {"Polygon", "MultiPolygon"}
        width, width_source = resolve_feature_width(
            tags, classified.get("assigned_surface_class") or "",
            classified.get("feature_subcategory") or "", config,
            is_polygon=is_polygon)
        override_width = next((override[key] for key in ("width_m", "width")
                               if key in override), None)
        if override_width is not None:
            width = float(override_width)
            width_source = "manual_override"
        width_raw = next((tags.get(key) for key in (
            "width", "est_width", "sidewalk:width",
            "sidewalk:left:width", "sidewalk:right:width")
                          if tags.get(key) is not None), None)
        polygon = None
        if classified["included_for_ground_material"]:
            if classified.get("assigned_material") not in config["materials"]:
                classified["included_for_ground_material"] = False
                classified["rejection_reason"] = "manual_or_classified_material_not_in_database"
            elif geom is None or geom.is_empty:
                classified["included_for_ground_material"] = False
                classified["rejection_reason"] = "empty_geometry"
            elif is_polygon:
                polygon = repair_polygon(geom, config["geometry_tolerance_m"])
            elif width is None or not np.isfinite(width) or width <= 0:
                classified["included_for_ground_material"] = False
                classified["rejection_reason"] = "no_valid_width_or_fallback"
            elif geom.geom_type in {"LineString", "MultiLineString"}:
                polygon = repair_polygon(
                    geom.buffer(width / 2.0, cap_style="flat", join_style="round"),
                    config["geometry_tolerance_m"])
            else:
                classified["included_for_ground_material"] = False
                classified["rejection_reason"] = f"unsupported_geometry:{geom.geom_type}"
        if polygon is not None and polygon.area < config["minimum_polygon_area_m2"]:
            polygon = None
            classified["included_for_ground_material"] = False
            classified["rejection_reason"] = "below_minimum_polygon_area"
        record = tags.copy()
        for key in ("highway", "footway", "service", "surface", "access",
                    "area", "area:highway"):
            record.setdefault(key, None)
        record.update(classified)
        record.update({
            "osm_id": osm_id,
            "osm_type": osm_type,
            "width_raw": None if width_raw is None else str(width_raw),
            "width_m": width,
            "width_source": width_source,
            "overlap_priority_override": override.get("overlap_priority", 0),
            "width_uncertainty": width_source in {"class_default", "lane_inference"},
            "override_applied": bool(override),
            "override_comments": override.get("comments", override.get("comment")),
            "used_for_route_graph": False,
            "material_polygon": polygon,
        })
        records.append(record)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)


def _write_layer(frame: gpd.GeoDataFrame, path: Path, layer: str,
                 written: list[dict[str, object]]) -> None:
    if frame.empty:
        written.append({"layer": layer, "feature_count": 0, "written": False})
        return
    safe = frame.copy()
    for column in safe.columns:
        if column == safe.geometry.name:
            continue
        safe[column] = safe[column].map(
            lambda value: json.dumps(
                value,
                default=lambda item: (item.tolist() if isinstance(item, np.ndarray)
                                      else str(item)),
            ) if isinstance(value, (list, tuple, dict, np.ndarray)) else value)
    safe.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
    written.append({"layer": layer, "feature_count": len(safe), "written": True})


def _diagnostic_plot(output: Path, material_polygons: dict[str, object],
                     routes_path: str | None, buildings_path: str | None,
                     vegetation_path: str | None) -> None:
    colors = {
        "generic_ground": "#d9d2b6", "asphalt_road": "#4d4d4d",
        "asphalt_pedestrian_path": "#8c6d5a", "concrete_sidewalk": "#d9d9d9",
        "paving_stone_path": "#c7a76d", "unpaved_path": "#b58b55",
        "pedestrian_crossing": "#f2f2f2",
        "concrete_road": "#969696", "asphalt_pedestrian": "#8c6d5a",
        "concrete_pedestrian": "#e0e0e0",
        "paving_stone_pedestrian": "#c7a76d", "pedestrian_plaza": "#dfc27d",
        "asphalt_parking": "#636363", "concrete_parking": "#bdbdbd",
        "gravel_parking": "#b8a47a", "grass_lawn": "#78c679",
        "artificial_turf": "#31a354", "sports_surface": "#fdae6b",
        "playground_surface": "#f768a1", "bare_ground": "#b58b55",
        "water": "#6baed6",
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    legend_handles = []
    for index, name in enumerate(material_polygons):
        geom = material_polygons.get(name)
        if geom is None or geom.is_empty:
            continue
        color = colors.get(name, plt.get_cmap("tab20")(index % 20))
        gpd.GeoSeries([geom]).plot(ax=ax, color=color, edgecolor="none",
                                  alpha=0.90)
        legend_handles.append(Patch(facecolor=color, label=name))
    if buildings_path and Path(buildings_path).is_file():
        mesh = trimesh.load(buildings_path, force="mesh")
        edges = np.asarray(mesh.vertices)[np.asarray(mesh.edges_unique)][:, :, :2]
        if len(edges) > 30000:
            edges = edges[::int(np.ceil(len(edges) / 30000))]
        ax.add_collection(LineCollection(edges, colors="#7f0000", linewidths=0.25,
                                         alpha=0.35))
        legend_handles.append(Line2D([], [], color="#7f0000", lw=1,
                                     label="buildings"))
    if vegetation_path and Path(vegetation_path).is_file():
        mesh = trimesh.load(vegetation_path, force="mesh")
        points = np.asarray(mesh.triangles_center)[:, :2]
        if len(points) > 5000:
            points = points[::int(np.ceil(len(points) / 5000))]
        ax.scatter(points[:, 0], points[:, 1], s=1, color="#238b45", alpha=0.2,
                   label="vegetation")
        legend_handles.append(Line2D([], [], color="#238b45", marker=".",
                                     linestyle="none", label="vegetation"))
    if routes_path and Path(routes_path).is_file():
        with open(routes_path, "rb") as stream:
            payload = pickle.load(stream)
        lines = payload.get("polylines", payload) if isinstance(payload, dict) else payload
        for i, coords in enumerate(lines):
            xy = np.asarray(coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#0868ac", lw=1.1,
                    label="existing route centerlines" if i == 0 else None)
        legend_handles.append(Line2D([], [], color="#0868ac", lw=1.1,
                                     label="existing route centerlines"))
    ax.set_aspect("equal")
    ax.set_xlabel("Projected easting (m)")
    ax.set_ylabel("Projected northing (m)")
    ax.set_title("TREC-Route OSM ground materials (routing geometry unchanged)")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_osm_ground_config(args.config)
    if not config["enabled"]:
        print("OSM ground materials disabled; no surface branch generated")
        return
    source_path = Path(args.osm_features)
    layer = args.osm_layer if source_path.suffix.lower() == ".gpkg" else None
    try:
        frame = gpd.read_file(source_path, layer=layer)
    except Exception:
        # Backward compatibility with older single-layer GeoPackages.
        frame = gpd.read_file(source_path)
    if frame.crs is None:
        raise ValueError("OSM features require a declared CRS")
    configured_crs = config.get("projected_crs", "auto")
    if configured_crs == "auto":
        crs = validate_projected_crs(frame.crs)
    else:
        crs = validate_projected_crs(configured_crs)
        if CRS.from_user_input(frame.crs).is_geographic or CRS_mismatch(
                CRS.from_user_input(frame.crs), crs):
            frame = frame.to_crs(crs)
    if args.local_origin_x or args.local_origin_y:
        frame = frame.copy()
        frame.geometry = frame.geometry.map(
            lambda geom: affinity.translate(geom, -args.local_origin_x, -args.local_origin_y))
    original_count = len(frame)
    unique = _deduplicate_directed_edges(frame)
    overrides = _load_overrides(config, args.overrides, crs)
    classified = classify_features(unique, config, overrides)

    mesh = trimesh.load(args.ground_mesh, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("ground mesh must contain triangular faces")
    ground_polygon = repair_polygon(MultiPoint(np.asarray(mesh.vertices)[:, :2]).convex_hull)
    # Preserve original OSM geometries in the raw cache, but clip physical
    # material polygons to the exact terrain domain here.
    classified["material_polygon"] = classified["material_polygon"].map(
        lambda geom: repair_polygon(geom.intersection(ground_polygon), 0.0)
        if geom is not None and not geom.is_empty else geom)
    empty_after_clip = classified["included_for_ground_material"] & classified[
        "material_polygon"].map(lambda geom: geom is None or geom.is_empty)
    classified.loc[empty_after_clip, "included_for_ground_material"] = False
    classified.loc[empty_after_clip, "rejection_reason"] = "outside_simulation_domain"
    included = classified[classified["included_for_ground_material"]].copy()
    overlap_records = []
    for _, row in included.iterrows():
        overlap_records.append({
            "geometry": row["material_polygon"],
            "assigned_surface_class": row["assigned_surface_class"],
            "assigned_material": row["assigned_material"],
            "material_source": row["material_source"],
            "overlap_priority_override": row["overlap_priority_override"],
        })
    material_polygons, overlap_before, overlap_after = resolve_feature_material_overlaps(
        overlap_records, ground_polygon, config["overlap_priority"],
        config["minimum_polygon_area_m2"])
    material_names = list(config["materials"])
    centroids = np.asarray(mesh.triangles_center)[:, :2]
    if config.get("boundary_aware_face_assignment", True):
        face_material_id, boundary_stats = assign_materials_boundary_aware(
            np.asarray(mesh.triangles)[:, :, :2], material_polygons,
            material_names, config["material_priority"],
            float(config["minimum_face_overlap_fraction"]))
        classification_method = (
            "centroid assignment plus boundary-aware triangle-polygon plan-area overlap")
    else:
        face_material_id = assign_materials_to_face_centroids(
            centroids, material_polygons, material_names)
        boundary_stats = {"faces_changed_from_centroid_assignment": 0}
        classification_method = "existing terrain face centroid in mutually exclusive OSM material polygons"
    if len(face_material_id) != len(mesh.faces):
        raise RuntimeError("ground material map does not cover every terrain face")
    np.savez_compressed(out_dir / GROUND_FACE_MATERIAL_MAP, material_id=face_material_id)
    catalog = {
        "enabled": True,
        "affect_route_connectivity": False,
        "crs": crs.to_string(),
        "ground_mesh": str(Path(args.ground_mesh).resolve()),
        "ground_mesh_signature": _mesh_signature(mesh),
        "material_names": material_names,
        "materials": config["materials"],
        "classification_method": classification_method,
        "boundary_assignment": boundary_stats,
        "terrain_geometry_modified": False,
    }
    (out_dir / GROUND_MATERIAL_CATALOG).write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    gpkg = out_dir / "osm_ground_materials.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    written: list[dict[str, object]] = []
    base_fields = [column for column in (
        "osm_id", "osm_type", "highway", "footway", "service", "surface",
        "sidewalk", "cycleway", "landuse", "landcover", "natural", "leisure",
        "amenity", "sport", "parking", "parking:surface", "water", "waterway",
        "building", "building:part", "bridge", "tunnel", "layer", "covered",
        "all_tags_json",
        "width_raw", "width_m", "width_source", "assigned_surface_class",
        "feature_subcategory",
        "assigned_material", "classification_source", "material_source",
        "uncertainty_flag", "missing_surface_tag", "width_uncertainty",
        "override_applied", "override_comments", "included_for_ground_material",
        "used_for_route_graph", "rejection_reason", "geometry")
                   if column in classified.columns]
    _write_layer(frame, gpkg, "raw_complete_osm_features", written)
    _write_layer(classified[base_fields], gpkg, "classified_features", written)
    _write_layer(classified[classified["assigned_surface_class"].eq("vehicle_road")][base_fields],
                 gpkg, "vehicle_road_centerlines", written)
    _write_layer(classified[classified["assigned_surface_class"].isin(
        ["sidewalk", "pedestrian_path", "pedestrian_crossing", "pedestrian_plaza"])][base_fields],
                 gpkg, "pedestrian_centerlines", written)
    rejected = classified[~classified["included_for_ground_material"]][base_fields]
    unclassified = rejected[rejected["rejection_reason"].eq(
        "not_a_supported_transportation_surface")]
    _write_layer(unclassified, gpkg, "unclassified_features", written)
    _write_layer(rejected, gpkg, "rejected_features", written)
    polygon_rows = []
    for _, row in included.iterrows():
        rec = {key: row.get(key) for key in base_fields if key != "geometry"}
        rec["geometry"] = row["material_polygon"]
        polygon_rows.append(rec)
    buffered = gpd.GeoDataFrame(polygon_rows, geometry="geometry", crs=frame.crs)
    if not buffered.empty:
        _write_layer(buffered[buffered["assigned_surface_class"].eq("vehicle_road")],
                     gpkg, "buffered_vehicle_roads", written)
        _write_layer(buffered[buffered["assigned_surface_class"].isin(["sidewalk", "pedestrian_path"])],
                     gpkg, "buffered_pedestrian", written)
        _write_layer(buffered[buffered["assigned_surface_class"].eq("pedestrian_crossing")],
                     gpkg, "crossings", written)
        _write_layer(buffered[buffered["assigned_surface_class"].eq("pedestrian_plaza")],
                     gpkg, "pedestrian_plazas", written)
        for surface_class, layer_name in (
                ("parking", "parking_surfaces"),
                ("grass_area", "grass_polygons"),
                ("sports_surface", "sports_surfaces"),
                ("playground_surface", "playground_surfaces"),
                ("water", "water_surfaces"),
                ("bare_ground", "bare_ground_surfaces")):
            _write_layer(buffered[buffered["assigned_surface_class"].eq(surface_class)],
                         gpkg, layer_name, written)
    final_rows = [{"assigned_material": name, "geometry": geom}
                  for name, geom in material_polygons.items() if not geom.is_empty]
    final = gpd.GeoDataFrame(final_rows, geometry="geometry", crs=frame.crs)
    _write_layer(final, gpkg, "final_ground_materials", written)
    _write_layer(final[final["assigned_material"].eq("generic_ground")],
                 gpkg, "remaining_generic_ground", written)
    pd.DataFrame(written).to_csv(out_dir / "gpkg_layer_inventory.csv", index=False)

    face_area = np.asarray(mesh.area_faces)
    area_by_material = {
        name: float(face_area[face_material_id == index].sum())
        for index, name in enumerate(material_names)
    }
    total_face_area = float(face_area.sum())
    polygon_area = sum(geom.area for geom in material_polygons.values())
    feature_counts = Counter(included["assigned_surface_class"].dropna())
    highway_counts = Counter(included["highway"].map(normalize_tag).dropna())
    material_feature_counts = Counter(included["assigned_material"].dropna())
    centerline_length = defaultdict(float)
    for _, row in included.iterrows():
        centerline_length[str(row["assigned_surface_class"])] += float(row.geometry.length)
    width_sources = included["width_source"].value_counts().to_dict()
    n_included = len(included)
    explicit_width_count = sum(width_sources.get(k, 0) for k in
                               ("width", "est_width", "sidewalk_width"))
    fallback_width_count = width_sources.get("class_default", 0)
    summary_rows = [
        ("input_directed_features", original_count, "count"),
        ("unique_transport_features", len(unique), "count"),
        ("included_features", n_included, "count"),
        ("unclassified_or_rejected_features", len(rejected), "count"),
        ("explicit_width_features", explicit_width_count, "count"),
        ("fallback_width_features", fallback_width_count, "count"),
        ("explicit_width_pct", 100.0 * explicit_width_count /
         max(1, n_included), "percent"),
        ("fallback_width_pct", 100.0 * fallback_width_count /
         max(1, n_included), "percent"),
        ("overlap_area_before_resolution", overlap_before, "m2"),
        ("overlap_area_after_resolution", overlap_after, "m2"),
        ("remaining_ground_plan_area", material_polygons["generic_ground"].area, "m2"),
        ("ground_plan_area", ground_polygon.area, "m2"),
        ("polygon_area_conservation_error", polygon_area - ground_polygon.area, "m2"),
        ("terrain_surface_area", total_face_area, "m2"),
        ("face_area_conservation_error", sum(area_by_material.values()) - total_face_area, "m2"),
        ("boundary_faces_changed_from_centroid_assignment",
         boundary_stats.get("faces_changed_from_centroid_assignment", 0), "count"),
    ]
    summary_rows += [(f"feature_count:{key}", value, "count")
                     for key, value in sorted(feature_counts.items())]
    summary_rows += [(f"highway_count:{key}", value, "count")
                     for key, value in sorted(highway_counts.items())]
    summary_rows += [(f"material_feature_count:{key}", value, "count")
                     for key, value in sorted(material_feature_counts.items())]
    material_sources = included["material_source"].value_counts().to_dict()
    summary_rows += [(f"material_source_count:{key}", value, "count")
                     for key, value in sorted(material_sources.items())]
    summary_rows += [
        ("missing_surface_tag_features", int(included["missing_surface_tag"].sum()), "count"),
        ("uncertain_material_features", int(included["uncertainty_flag"].sum()), "count"),
    ]
    summary_rows += [(f"centerline_length:{key}", value, "m")
                     for key, value in sorted(centerline_length.items())]
    summary_rows += [(f"face_surface_area:{key}", value, "m2")
                     for key, value in area_by_material.items()]
    summary_rows += [(f"terrain_triangle_count:{name}",
                      int(np.sum(face_material_id == index)), "count")
                     for index, name in enumerate(material_names)]
    summary_rows += [(f"polygon_plan_area:{key}", geom.area, "m2")
                     for key, geom in material_polygons.items()]
    pd.DataFrame(summary_rows, columns=["metric", "value", "unit"]).to_csv(
        out_dir / "ground_material_summary.csv", index=False)
    classified[classified["included_for_ground_material"]
               & classified["missing_surface_tag"]][base_fields].drop(
                   columns="geometry", errors="ignore").to_csv(
                       out_dir / "missing_surface_tags.csv", index=False)
    _diagnostic_plot(out_dir / "ground_material_plan",
                     material_polygons, args.routes_pkl,
                     args.buildings_mesh, args.vegetation_mesh)
    print(f"OSM material features: {original_count} directed -> {len(unique)} unique; "
          f"{n_included} included")
    print(f"Terrain faces classified: {len(mesh.faces):,}; materials: "
          f"{dict(Counter(material_names[i] for i in face_material_id))}")
    print(f"Area conservation errors: polygons {polygon_area-ground_polygon.area:.3e} m2; "
          f"faces {sum(area_by_material.values())-total_face_area:.3e} m2")
    print(f"[osm_ground_materials] output_dir={out_dir} route_connectivity_changed=false")


def CRS_mismatch(first, second) -> bool:
    """Small named helper kept separate for unit testing."""
    return not first.equals(second)


if __name__ == "__main__":
    main()
