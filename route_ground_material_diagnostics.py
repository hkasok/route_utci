"""Trace exact TREC-Route receptors to OSM and terrain material diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

from radiant_flux_contributions import canonical_lw_source, canonical_sw_source
from thermal_common import get_intersector


PEDESTRIAN_CLASSES = {
    "sidewalk", "pedestrian_path", "pedestrian_crossing", "pedestrian_plaza",
}
PEDESTRIAN_MATERIALS = {
    "asphalt_pedestrian", "concrete_pedestrian", "paving_stone_pedestrian",
    "pedestrian_crossing", "pedestrian_plaza",
    # Legacy cache compatibility.
    "asphalt_pedestrian_path", "concrete_sidewalk", "paving_stone_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route-point ground-material audit")
    parser.add_argument("--mrt-dir", required=True)
    parser.add_argument("--thermal-dir", required=True)
    parser.add_argument("--ground-mesh", required=True)
    parser.add_argument("--ground-material-dir", required=True)
    parser.add_argument("--departure-hour", type=float, required=True)
    parser.add_argument("--walking-speed", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _periodic_interp(hours: np.ndarray, values: np.ndarray,
                     target_hours: np.ndarray, column_indices: np.ndarray) -> np.ndarray:
    result = np.empty(len(target_hours), dtype=float)
    for i, (hour, column) in enumerate(zip(np.mod(target_hours, 24.0), column_indices)):
        result[i] = np.interp(hour, hours, values[:, column], period=24.0)
    return result


def _time_hours(times_csv: Path) -> np.ndarray:
    times = pd.to_datetime(pd.read_csv(times_csv)["time"])
    return (times.dt.hour + times.dt.minute / 60.0
            + times.dt.second / 3600.0).to_numpy(dtype=float)


def _inside_feature(point, feature) -> bool:
    geometry = feature.geometry
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return bool(geometry.covers(point))
    width = feature.get("width_m")
    return bool(pd.notna(width) and geometry.distance(point) <= float(width) / 2.0 + 1.0e-8)


def main() -> None:
    args = parse_args()
    if args.walking_speed <= 0:
        raise ValueError("walking speed must be positive")
    mrt_dir = Path(args.mrt_dir)
    thermal_dir = Path(args.thermal_dir)
    material_dir = Path(args.ground_material_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.load(mrt_dir / "path_xyz.npy")
    segment = np.load(mrt_dir / "path_segment_id.npy").astype(int)
    if len(xyz) != len(segment):
        raise ValueError("path_xyz and path_segment_id lengths differ")
    route_id = segment + 1
    route_index = np.empty(len(xyz), dtype=int)
    distance = np.empty(len(xyz), dtype=float)
    for rid in np.unique(route_id):
        indices = np.flatnonzero(route_id == rid)
        route_index[indices] = np.arange(len(indices))
        increments = np.linalg.norm(np.diff(xyz[indices, :2], axis=0), axis=1)
        distance[indices] = np.r_[0.0, np.cumsum(increments)]
    arrival_hour = args.departure_hour + distance / args.walking_speed / 3600.0

    catalog = json.loads((material_dir / "ground_material_catalog.json").read_text())
    names = np.asarray(catalog["material_names"], dtype=str)
    face_material = np.load(material_dir / "ground_face_materials.npz")["material_id"]
    ground = trimesh.load(args.ground_mesh, force="mesh", process=False)
    hit_face = get_intersector(ground).intersects_first(
        xyz + np.array([0.0, 0.0, 0.05]),
        np.tile([0.0, 0.0, -1.0], (len(xyz), 1)))
    downward_method = np.full(len(xyz), "vertical_ray", dtype=object)
    terrain_fallback_distance = np.zeros(len(xyz), dtype=float)
    misses = hit_face < 0
    if np.any(misses):
        nearest_distance, nearest_face = cKDTree(
            np.asarray(ground.triangles_center)[:, :2]).query(xyz[misses, :2])
        hit_face[misses] = nearest_face.astype(hit_face.dtype)
        downward_method[misses] = "nearest_face_fallback_after_vertical_miss"
        terrain_fallback_distance[misses] = nearest_distance
    downward_material = names[face_material[hit_face]]
    properties = catalog["materials"]

    gpkg = material_dir / "osm_ground_materials.gpkg"
    features = gpd.read_file(gpkg, layer="classified_features")
    included = features[features["included_for_ground_material"].fillna(False).astype(bool)].copy()
    transport = included[included["assigned_surface_class"].isin(
        ["vehicle_road", *sorted(PEDESTRIAN_CLASSES)])
        & included.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    points = gpd.GeoDataFrame(
        {"global_index": np.arange(len(xyz))},
        geometry=gpd.points_from_xy(xyz[:, 0], xyz[:, 1]), crs=features.crs)
    nearest = gpd.sjoin_nearest(
        points, transport[["geometry"]], how="left", distance_col="nearest_distance_m")
    nearest = nearest.sort_values(["global_index", "nearest_distance_m"], kind="mergesort")
    nearest = nearest.drop_duplicates("global_index").set_index("global_index").reindex(
        np.arange(len(xyz)))
    nearest_rows = transport.loc[nearest["index_right"].astype(int)].reset_index(drop=True)
    inside = np.array([
        _inside_feature(point, feature)
        for point, (_, feature) in zip(points.geometry, nearest_rows.iterrows())], dtype=bool)

    facet_surface_c = np.full(len(xyz), np.nan)
    facets_path = thermal_dir / "facets.npz"
    temperature_path = thermal_dir / "facet_T_matrix_K.npy"
    hours = _time_hours(mrt_dir / "times.csv")
    if facets_path.is_file() and temperature_path.is_file():
        facets = np.load(facets_path)
        ground_facets = np.flatnonzero(facets["mesh_id"] == 1)
        face_to_facet = {int(facets["face_id"][idx]): int(idx) for idx in ground_facets}
        facet_index = np.array([face_to_facet.get(int(face), -1) for face in hit_face])
        available = facet_index >= 0
        temperatures = np.load(temperature_path, mmap_mode="r")
        facet_surface_c[available] = _periodic_interp(
            hours, temperatures, arrival_hour[available], facet_index[available]) - 273.15
    else:
        facet_index = np.full(len(xyz), -1, dtype=int)

    reflected = np.full(len(xyz), np.nan)
    emitted = np.full(len(xyz), np.nan)
    archive_path = mrt_dir / "radiant_flux_contributions.npz"
    if archive_path.is_file():
        archive = np.load(archive_path)
        for material in np.unique(downward_material):
            mask = downward_material == material
            sw_key = canonical_sw_source(material)
            lw_key = canonical_lw_source(material)
            if sw_key in archive.files:
                reflected[mask] = _periodic_interp(
                    hours, archive[sw_key], arrival_hour[mask], np.flatnonzero(mask))
            if lw_key in archive.files:
                emitted[mask] = _periodic_interp(
                    hours, archive[lw_key], arrival_hour[mask], np.flatnonzero(mask))

    assigned_nearest = nearest_rows["assigned_material"].astype(str).to_numpy()
    nearest_class = nearest_rows["assigned_surface_class"].astype(str).to_numpy()
    warning = np.full(len(xyz), "", dtype=object)
    path_generic = (np.isin(nearest_class, list(PEDESTRIAN_CLASSES)) & inside
                    & ~np.isin(downward_material, list(PEDESTRIAN_MATERIALS)))
    warning[path_generic] = "pedestrian_centerline_or_footprint_hits_non_pedestrian_material"
    warning[misses] = np.where(
        warning[misses] == "", "vertical_downward_ray_missed_terrain_nearest_face_fallback",
        warning[misses] + ";vertical_downward_ray_missed_terrain_nearest_face_fallback")
    missing_facet = facet_index < 0
    warning[missing_facet] = np.where(
        warning[missing_facet] == "", "downward_face_not_in_route_visible_thermal_facets",
        warning[missing_facet] + ";downward_face_not_in_route_visible_thermal_facets")

    frame = pd.DataFrame({
        "route_id": route_id,
        "route_point_index": route_index,
        "global_receptor_index": np.arange(len(xyz)),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "distance_along_route_m": distance,
        "arrival_hour": np.mod(arrival_hour, 24.0),
        "nearest_osm_id": nearest_rows["osm_id"].astype(str).to_numpy(),
        "nearest_osm_type": nearest_rows["osm_type"].astype(str).to_numpy(),
        "nearest_feature_class": nearest_class,
        "nearest_feature_subcategory": nearest_rows["feature_subcategory"].astype(str).to_numpy(),
        "nearest_original_surface": nearest_rows["surface"].to_numpy(),
        "nearest_assigned_material": assigned_nearest,
        "nearest_material_source": nearest_rows["material_source"].astype(str).to_numpy(),
        "nearest_width_m": pd.to_numeric(nearest_rows["width_m"], errors="coerce").to_numpy(),
        "nearest_width_source": nearest_rows["width_source"].astype(str).to_numpy(),
        "distance_to_nearest_feature_centerline_m": nearest["nearest_distance_m"].to_numpy(),
        "inside_nearest_buffered_footprint": inside,
        "downward_hit_triangle_id": hit_face,
        "downward_hit_method": downward_method,
        "terrain_fallback_horizontal_distance_m": terrain_fallback_distance,
        "downward_hit_material": downward_material,
        "downward_material_albedo": [properties[name]["albedo"] for name in downward_material],
        "downward_material_emissivity": [properties[name]["emissivity"] for name in downward_material],
        "downward_surface_temperature_C": facet_surface_c,
        "sw_reflected_from_downward_material_absorbed_Wm2": reflected,
        "lw_from_downward_material_absorbed_Wm2": emitted,
        "classification_warning": warning,
    })
    frame.sort_values(["route_id", "route_point_index"], kind="mergesort").to_csv(
        output, index=False)
    # Append the exact receptor locations and downward-hit classification to
    # the reusable material GeoPackage.  This is diagnostic only and never
    # feeds route construction or graph connectivity.
    gpkg_frame = gpd.GeoDataFrame(
        frame.copy(), geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=features.crs)
    gpkg_frame.to_file(
        gpkg, layer="route_points_ground_materials", driver="GPKG",
        engine="pyogrio")
    summary = (frame.groupby(["route_id", "downward_hit_material"], sort=True)
               .size().rename("point_count").reset_index())
    totals = summary.groupby("route_id")["point_count"].transform("sum")
    summary["route_point_fraction_percent"] = 100.0 * summary["point_count"] / totals
    summary.to_csv(output.with_name("route_ground_material_coverage_summary.csv"), index=False)
    print(f"Wrote {len(frame):,} route receptor diagnostics: {output}")
    print(f"Wrote route_points_ground_materials layer: {gpkg}")
    print(f"Classification warnings: {int(np.sum(warning != ''))}")


if __name__ == "__main__":
    main()
