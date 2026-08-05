"""Download/cache complete OSM physical-surface features for FIU MMC.

This is an independent material-data branch. It never opens the authoritative
routing graph for writing and never constructs routes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import trimesh
from pyproj import CRS
from shapely.geometry import MultiPoint

from osm_ground_materials import load_osm_ground_config


QUERY_TAG_KEYS = [
    "highway", "area:highway", "landuse", "landcover", "natural",
    "leisure", "amenity", "sport", "parking", "water", "waterway",
    "building", "building:part",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache complete FIU MMC OSM features for ground materials")
    parser.add_argument("--ground-mesh", required=True,
                        help="Terrain mesh defining the exact simulation domain")
    parser.add_argument("--config", default="osm_ground_materials.json")
    parser.add_argument("--output-file", default=None,
                        help="Override configured raw cache GeoPackage")
    parser.add_argument("--input-file", default=None,
                        help="Use a supplied .gpkg/.geojson/.osm/.xml instead of downloading")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _cache_path(config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    name = Path(config["raw_cache_file"])
    return name if name.is_absolute() else Path(config["cache_directory"]) / name


def _domain(mesh_path: str, projected_crs: str, buffer_m: float):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("ground mesh must contain triangular faces")
    polygon = MultiPoint(np.asarray(mesh.vertices)[:, :2]).convex_hull.buffer(buffer_m)
    projected = gpd.GeoSeries([polygon], crs=projected_crs)
    return polygon, projected.to_crs(4326).iloc[0], mesh


def _make_tags_json(row: pd.Series) -> str:
    tags = {}
    skip = {"geometry", "osm_type", "osm_id", "all_tags_json"}
    for key, value in row.items():
        if key in skip or value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (list, tuple, set)):
            value = list(value)
        elif isinstance(value, np.generic):
            value = value.item()
        tags[str(key)] = value
    return json.dumps(tags, sort_keys=True, default=str)


def normalize_feature_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Expose OSM identity columns and make all tag values GeoPackage-safe."""
    frame = frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    else:
        frame = frame.reset_index(drop=False)
    rename = {}
    if "element_type" in frame.columns:
        rename["element_type"] = "osm_type"
    elif "element" in frame.columns:
        rename["element"] = "osm_type"
    if "osmid" in frame.columns and "osm_id" not in frame.columns:
        rename["osmid"] = "osm_id"
    elif "id" in frame.columns and "osm_id" not in frame.columns:
        rename["id"] = "osm_id"
    frame = frame.rename(columns=rename)
    if "osm_type" not in frame:
        frame["osm_type"] = "feature"
    if "osm_id" not in frame:
        frame["osm_id"] = np.arange(len(frame), dtype=np.int64)
    frame["all_tags_json"] = frame.apply(_make_tags_json, axis=1)
    for column in frame.columns:
        if column == frame.geometry.name:
            continue
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, default=str)
                if isinstance(value, (list, tuple, dict, set, np.ndarray)) else value)
    frame["osm_type"] = frame["osm_type"].astype(str)
    frame["osm_id"] = frame["osm_id"].astype(str)
    return gpd.GeoDataFrame(frame, geometry=frame.geometry.name, crs=frame.crs)


def feature_inventory(frame: gpd.GeoDataFrame) -> dict[str, Any]:
    """Return compact identity, geometry, and retained-tag counts."""
    return {
        "osm_type_counts": {
            str(key): int(value)
            for key, value in frame["osm_type"].value_counts(dropna=False).items()
        },
        "geometry_type_counts": {
            str(key): int(value)
            for key, value in frame.geometry.geom_type.value_counts().items()
        },
        "retained_tag_non_null_counts": {
            key: int(frame[key].notna().sum()) if key in frame else 0
            for key in QUERY_TAG_KEYS + [
                "surface", "width", "est_width", "sidewalk", "footway",
                "service", "area", "parking:surface", "covered", "bridge",
                "tunnel", "layer",
            ]
        },
    }


def load_user_input(path: Path) -> gpd.GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        return gpd.read_file(path)
    if suffix in {".osm", ".xml"}:
        return ox.features.features_from_xml(path, tags={key: True for key in QUERY_TAG_KEYS})
    if suffix == ".pbf":
        try:
            from pyrosm import OSM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                ".pbf input requires pyrosm; install pyrosm or supply GPKG/GeoJSON/OSM XML") from exc
        osm = OSM(str(path))
        frames = [osm.get_data_by_custom_criteria(
            custom_filter={key: True}, filter_type="keep", keep_nodes=True,
            keep_ways=True, keep_relations=True) for key in QUERY_TAG_KEYS]
        frames = [item for item in frames if item is not None and not item.empty]
        if not frames:
            raise RuntimeError(f"no relevant OSM features found in {path}")
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    raise ValueError(f"unsupported OSM input format: {path.suffix}")


def write_overpass_query(path: Path, polygon_wgs84) -> None:
    west, south, east, north = polygon_wgs84.bounds
    key_regex = "|".join(QUERY_TAG_KEYS).replace(":", "\\:")
    query = (
        "[out:xml][timeout:180];\n"
        "(\n"
        f"  nwr[~\"^({key_regex})$\"~\".\"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});\n"
        ");\n"
        "(._;>;);\n"
        "out body;\n")
    path.write_text(query, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_osm_ground_config(args.config)
    projected_crs = config.get("projected_crs")
    if projected_crs in {None, "auto"} or CRS.from_user_input(projected_crs).is_geographic:
        raise ValueError("complete OSM download requires a configured projected metric CRS")
    cache = _cache_path(config, args.output_file)
    cache.parent.mkdir(parents=True, exist_ok=True)
    domain_projected, domain_wgs84, mesh = _domain(
        args.ground_mesh, projected_crs, float(config["domain_buffer_m"]))
    query_path = cache.with_suffix(".overpass.ql")
    metadata_path = cache.with_suffix(".metadata.json")
    write_overpass_query(query_path, domain_wgs84)

    supplied = args.input_file or config.get("input_file")
    if supplied:
        source = Path(supplied)
        if not source.is_file():
            raise FileNotFoundError(f"configured complete OSM input does not exist: {source}")
        frame = load_user_input(source)
        source_description = f"user_file:{source.resolve()}"
    elif cache.is_file() and not args.force_download:
        frame = gpd.read_file(cache, layer="raw_complete_osm_features")
        inventory = feature_inventory(frame)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(inventory)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Using cached complete OSM features: {cache}")
        print(f"[osm_complete] features={len(frame)} cache={cache} downloaded=false")
        return
    else:
        if not config.get("download_if_missing", True):
            raise FileNotFoundError(
                f"complete OSM cache missing: {cache}. Supply --input-file or run the "
                f"query saved at {query_path}")
        tags = {key: True for key in QUERY_TAG_KEYS}
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 180
        try:
            frame = ox.features.features_from_polygon(domain_wgs84, tags)
        except Exception as exc:
            raise RuntimeError(
                f"complete OSM download failed and no cache is available. Use {query_path} "
                f"to obtain data for bounds {domain_wgs84.bounds}, then pass --input-file") from exc
        source_description = "OSMnx.features_from_polygon/Overpass"

    if frame.empty:
        raise RuntimeError("complete OSM feature source returned no relevant features")
    if frame.crs is None:
        raise ValueError("complete OSM features have no coordinate reference system")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame = normalize_feature_frame(frame)
    frame.to_file(cache, layer="raw_complete_osm_features", driver="GPKG")
    gpd.GeoDataFrame(
        {"domain_buffer_m": [float(config["domain_buffer_m"])],
         "geometry": [domain_projected]}, crs=projected_crs,
    ).to_file(cache, layer="query_domain_projected", driver="GPKG")
    gpd.GeoDataFrame(
        {"geometry": [domain_wgs84]}, crs="EPSG:4326",
    ).to_file(cache, layer="query_domain_wgs84", driver="GPKG")

    inventory = feature_inventory(frame)
    metadata = {
        "source": source_description,
        "retrieval_utc": datetime.now(timezone.utc).isoformat(),
        "cache_file": str(cache.resolve()),
        "query_tag_keys": QUERY_TAG_KEYS,
        "query_bounds_wgs84": list(domain_wgs84.bounds),
        "domain_buffer_m": float(config["domain_buffer_m"]),
        "projected_crs": projected_crs,
        "ground_mesh": str(Path(args.ground_mesh).resolve()),
        "ground_mesh_faces": int(len(mesh.faces)),
        "feature_count": int(len(frame)),
        **inventory,
        "preserve_all_tags": True,
        "preserve_relations": True,
        "overpass_query_file": str(query_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cached {len(frame):,} complete OSM features at {cache}")
    print(f"Geometry types: {inventory['geometry_type_counts']}")
    print(f"[osm_complete] features={len(frame)} cache={cache} downloaded=true")


if __name__ == "__main__":
    main()
