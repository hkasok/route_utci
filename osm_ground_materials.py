"""OSM transportation surfaces for TREC-Route ground materials.

This module is deliberately independent of the routing graph.  It reads a
projected copy of OSM transportation geometries, classifies and buffers them,
resolves overlaps, and assigns one material identifier to every existing
terrain face.  It never adds/removes graph nodes or edges and never decides
whether a feature is walkable.
"""

from __future__ import annotations

import ast
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import CRS
from shapely import area as shapely_area
from shapely import contains_xy, intersection as shapely_intersection, make_valid
from shapely import polygons as shapely_polygons
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPolygon, Polygon)
from shapely.ops import unary_union


VEHICLE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "unclassified", "living_street",
    "service",
}
PEDESTRIAN_HIGHWAYS = {"footway", "path", "pedestrian", "steps"}
UNPAVED_SURFACES = {
    "compacted", "fine_gravel", "gravel", "dirt", "ground", "earth",
    "soil", "sand",
}
GROUND_FACE_MATERIAL_MAP = "ground_face_materials.npz"
GROUND_MATERIAL_CATALOG = "ground_material_catalog.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "affect_route_connectivity": False,
    "source_mode": "auto",
    "input_file": None,
    "download_if_missing": True,
    "cache_download": True,
    "domain_buffer_m": 50.0,
    "cache_directory": "data/osm",
    "raw_cache_file": "fiu_mmc_complete_osm.gpkg",
    "preserve_all_tags": True,
    "preserve_relations": True,
    "record_download_date": True,
    "include_vehicle_roads": True,
    "include_pedestrian_surfaces": True,
    "include_crossings": True,
    "include_pedestrian_plazas": True,
    "include_service_roads": True,
    "include_parking_aisles": True,
    "include_landcover": True,
    "include_parking_areas": True,
    "include_sports_surfaces": True,
    "include_water": True,
    "projected_crs": "auto",
    "minimum_polygon_area_m2": 0.25,
    "geometry_tolerance_m": 0.02,
    "boundary_aware_face_assignment": True,
    "minimum_face_overlap_fraction": 0.05,
    "infer_width_from_lanes": False,
    "lane_width_m": 3.2,
    "fallback_widths_m": {
        "motorway": 14.0, "motorway_link": 7.0, "trunk": 12.0,
        "trunk_link": 7.0, "primary": 10.0, "primary_link": 6.5,
        "secondary": 8.0, "secondary_link": 6.0, "tertiary": 7.0,
        "tertiary_link": 5.5, "residential": 6.0,
        "unclassified": 5.5, "living_street": 5.0, "service": 4.0,
        "parking_aisle": 3.5, "driveway": 3.0, "alley": 3.5,
        "footway": 2.0, "sidewalk": 1.8, "path": 2.0,
        "cycleway": 2.5, "crossing": 3.0, "steps": 2.0,
        "pedestrian": 4.0, "pedestrian_plaza": 8.0,
    },
    "default_material_by_osm_class": {
        "vehicle_road": "asphalt_road",
        "concrete_road": "concrete_road",
        "sidewalk": "concrete_pedestrian",
        "footway": "concrete_pedestrian",
        "pedestrian_path": "asphalt_pedestrian",
        "mixed_use_path": "asphalt_pedestrian",
        "crossing": "pedestrian_crossing",
        "pedestrian_plaza": "pedestrian_plaza",
        "steps": "concrete_pedestrian",
        "unpaved_path": "bare_ground",
        "parking": "asphalt_parking",
        "parking_aisle": "asphalt_parking",
        "grass_area": "grass_lawn",
        "sports_area": "sports_surface",
        "playground": "playground_surface",
        "bare_ground": "bare_ground",
        "water": "water",
    },
    "overlap_priority": [
        "water", "pedestrian_crossing", "pedestrian_plaza", "sidewalk",
        "pedestrian_path", "vehicle_road", "parking", "sports_surface",
        "playground_surface", "grass_area", "bare_ground", "generic_ground",
    ],
    "material_priority": [
        "water", "pedestrian_crossing", "pedestrian_plaza",
        "concrete_pedestrian", "asphalt_pedestrian", "paving_stone_pedestrian",
        "concrete_road", "asphalt_road", "concrete_parking", "asphalt_parking",
        "gravel_parking", "artificial_turf", "sports_surface",
        "playground_surface", "grass_lawn", "bare_ground", "generic_ground",
    ],
    "materials": {
        # Assumed, configurable defaults; not site measurements.  C is also
        # provided explicitly for direct use by the existing 1-D solver.
        "generic_ground": {
            "albedo": 0.18, "emissivity": 0.95,
            "thermal_conductivity": 1.00, "density": 2000.0,
            "specific_heat": 1000.0, "k": 1.00, "C": 2.00e6,
            "depth": 0.50, "n_layers": 8, "bottom_bc": "fixed",
            "roughness_m": 0.03,
        },
        "asphalt_road": {
            "albedo": 0.12, "emissivity": 0.95,
            "thermal_conductivity": 0.75, "density": 2300.0,
            "specific_heat": 920.0, "k": 0.75, "C": 2.116e6,
            "depth": 0.40, "n_layers": 8, "bottom_bc": "fixed",
            "roughness_m": 0.002,
        },
        "asphalt_pedestrian_path": {
            "albedo": 0.14, "emissivity": 0.95,
            "thermal_conductivity": 0.75, "density": 2250.0,
            "specific_heat": 920.0, "k": 0.75, "C": 2.070e6,
            "depth": 0.30, "n_layers": 7, "bottom_bc": "fixed",
            "roughness_m": 0.002,
        },
        "concrete_sidewalk": {
            "albedo": 0.30, "emissivity": 0.94,
            "thermal_conductivity": 1.40, "density": 2300.0,
            "specific_heat": 880.0, "k": 1.40, "C": 2.024e6,
            "depth": 0.25, "n_layers": 7, "bottom_bc": "fixed",
            "roughness_m": 0.003,
        },
        "paving_stone_path": {
            "albedo": 0.24, "emissivity": 0.94,
            "thermal_conductivity": 1.10, "density": 2200.0,
            "specific_heat": 840.0, "k": 1.10, "C": 1.848e6,
            "depth": 0.25, "n_layers": 7, "bottom_bc": "fixed",
            "roughness_m": 0.006,
        },
        "unpaved_path": {
            "albedo": 0.20, "emissivity": 0.95,
            "thermal_conductivity": 0.80, "density": 1800.0,
            "specific_heat": 1000.0, "k": 0.80, "C": 1.80e6,
            "depth": 0.40, "n_layers": 8, "bottom_bc": "fixed",
            "roughness_m": 0.015,
        },
        "pedestrian_crossing": {
            "albedo": 0.26, "emissivity": 0.94,
            "thermal_conductivity": 1.20, "density": 2250.0,
            "specific_heat": 880.0, "k": 1.20, "C": 1.98e6,
            "depth": 0.25, "n_layers": 7, "bottom_bc": "fixed",
            "roughness_m": 0.004,
        },
    },
    "manual_overrides": {},
}

# Additional complete-feature materials. Values are configurable assumed
# defaults, not OSM measurements; OSM supplies classification, not physics.
DEFAULT_CONFIG["materials"].update({
    "concrete_road": {"albedo": 0.28, "emissivity": 0.94, "k": 1.40,
                      "C": 2.024e6, "depth": 0.40, "n_layers": 8,
                      "bottom_bc": "fixed", "roughness_m": 0.003},
    "asphalt_pedestrian": {"albedo": 0.14, "emissivity": 0.95, "k": 0.75,
                           "C": 2.070e6, "depth": 0.30, "n_layers": 7,
                           "bottom_bc": "fixed", "roughness_m": 0.002},
    "concrete_pedestrian": {"albedo": 0.30, "emissivity": 0.94, "k": 1.40,
                            "C": 2.024e6, "depth": 0.25, "n_layers": 7,
                            "bottom_bc": "fixed", "roughness_m": 0.003},
    "paving_stone_pedestrian": {"albedo": 0.24, "emissivity": 0.94, "k": 1.10,
                                "C": 1.848e6, "depth": 0.25, "n_layers": 7,
                                "bottom_bc": "fixed", "roughness_m": 0.006},
    "pedestrian_plaza": {"albedo": 0.24, "emissivity": 0.94, "k": 1.10,
                         "C": 1.848e6, "depth": 0.30, "n_layers": 7,
                         "bottom_bc": "fixed", "roughness_m": 0.006},
    "asphalt_parking": {"albedo": 0.12, "emissivity": 0.95, "k": 0.75,
                        "C": 2.116e6, "depth": 0.40, "n_layers": 8,
                        "bottom_bc": "fixed", "roughness_m": 0.002},
    "concrete_parking": {"albedo": 0.28, "emissivity": 0.94, "k": 1.40,
                         "C": 2.024e6, "depth": 0.40, "n_layers": 8,
                         "bottom_bc": "fixed", "roughness_m": 0.003},
    "gravel_parking": {"albedo": 0.22, "emissivity": 0.95, "k": 0.80,
                       "C": 1.800e6, "depth": 0.40, "n_layers": 8,
                       "bottom_bc": "fixed", "roughness_m": 0.015},
    "grass_lawn": {"albedo": 0.23, "emissivity": 0.96, "k": 0.60,
                   "C": 1.820e6, "depth": 0.50, "n_layers": 8,
                   "bottom_bc": "fixed", "roughness_m": 0.03},
    "artificial_turf": {"albedo": 0.18, "emissivity": 0.95, "k": 0.30,
                        "C": 1.540e6, "depth": 0.15, "n_layers": 6,
                        "bottom_bc": "fixed", "roughness_m": 0.012},
    "sports_surface": {"albedo": 0.20, "emissivity": 0.95, "k": 0.50,
                       "C": 1.760e6, "depth": 0.20, "n_layers": 6,
                       "bottom_bc": "fixed", "roughness_m": 0.008},
    "playground_surface": {"albedo": 0.20, "emissivity": 0.95, "k": 0.35,
                           "C": 1.560e6, "depth": 0.15, "n_layers": 6,
                           "bottom_bc": "fixed", "roughness_m": 0.01},
    "bare_ground": {"albedo": 0.20, "emissivity": 0.95, "k": 0.80,
                    "C": 1.800e6, "depth": 0.50, "n_layers": 8,
                    "bottom_bc": "fixed", "roughness_m": 0.02},
    "water": {"albedo": 0.08, "emissivity": 0.98, "k": 0.60,
              "C": 4.17164e6, "depth": 1.0, "n_layers": 10,
              "bottom_bc": "fixed", "roughness_m": 0.0002},
})


SURFACE_TO_MATERIAL = {
    "asphalt": {"vehicle": "asphalt_road", "pedestrian": "asphalt_pedestrian",
                "parking": "asphalt_parking", "area": "sports_surface"},
    "concrete": {"vehicle": "concrete_road", "pedestrian": "concrete_pedestrian",
                 "parking": "concrete_parking", "area": "sports_surface"},
    "concrete:plates": {"vehicle": "concrete_road", "pedestrian": "concrete_pedestrian",
                        "parking": "concrete_parking", "area": "sports_surface"},
    "paving_stones": {"vehicle": "paving_stone_pedestrian",
                      "pedestrian": "paving_stone_pedestrian",
                      "parking": "paving_stone_pedestrian", "area": "paving_stone_pedestrian"},
    "sett": "paving_stone_pedestrian", "cobblestone": "paving_stone_pedestrian",
    "bricks": "paving_stone_pedestrian", "wood": "paving_stone_pedestrian",
    "metal": "sports_surface", "artificial_turf": "artificial_turf",
    "tartan": "sports_surface", "rubber": "playground_surface",
    "grass": "grass_lawn",
    "compacted": "bare_ground", "fine_gravel": "bare_ground",
    "gravel": {"parking": "gravel_parking", "vehicle": "bare_ground",
               "pedestrian": "bare_ground", "area": "bare_ground"},
    "dirt": "bare_ground", "ground": "bare_ground", "earth": "bare_ground",
    "soil": "bare_ground", "sand": "bare_ground",
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_osm_ground_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the centralized JSON configuration and validate its invariants."""
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as stream:
            _deep_update(config, json.load(stream))
    if config.get("affect_route_connectivity") is not False:
        raise ValueError("osm_ground_materials.affect_route_connectivity must remain false")
    required = set(DEFAULT_CONFIG["materials"])
    missing = required - set(config["materials"])
    if missing:
        raise ValueError(f"OSM material configuration is missing: {sorted(missing)}")
    for name, mat in config["materials"].items():
        for prop in ("albedo", "emissivity", "k", "C", "depth", "n_layers"):
            if prop not in mat:
                raise ValueError(f"material {name!r} is missing {prop!r}")
        if not (0 <= float(mat["albedo"]) <= 1):
            raise ValueError(f"material {name!r} has invalid albedo")
        if not (0 < float(mat["emissivity"]) <= 1):
            raise ValueError(f"material {name!r} has invalid emissivity")
    return config


def normalize_tag(value: Any) -> str | None:
    """Normalize scalar/list-valued OSM tags to a lower-case scalar."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return normalize_tag(value[0]) if len(value) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text.replace(" ", ", ") if "," not in text else text)
            if isinstance(parsed, (list, tuple)) and parsed:
                return normalize_tag(parsed[0])
        except (ValueError, SyntaxError):
            match = re.search(r"['\"]([^'\"]+)['\"]", text)
            if match:
                return match.group(1).strip().lower()
    return text.lower()


def parse_width_m(value: Any) -> float | None:
    """Parse common OSM width syntax; return ``None`` for malformed values.

    Semicolon-separated alternatives are parsed independently and their
    median is used, avoiding a crash or an arbitrary concatenation.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [parse_width_m(item) for item in value]
        parsed = [item for item in parsed if item is not None]
        return float(np.median(parsed)) if parsed else None
    text = str(value).strip().lower().replace(",", ".")
    if not text:
        return None
    if ";" in text:
        parsed = [parse_width_m(part) for part in text.split(";")]
        parsed = [item for item in parsed if item is not None]
        return float(np.median(parsed)) if parsed else None
    feet_inches = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)?\s*(?:\"|in)?\s*", text)
    if feet_inches:
        feet = float(feet_inches.group(1))
        inches = float(feet_inches.group(2) or 0.0)
        result = feet * 0.3048 + inches * 0.0254
        return result if result > 0 else None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres|ft|feet|foot|in|inch|inches)?\s*", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or "m"
    if unit in {"ft", "feet", "foot"}:
        number *= 0.3048
    elif unit in {"in", "inch", "inches"}:
        number *= 0.0254
    return number if np.isfinite(number) and number > 0 else None


def validate_projected_crs(crs: Any) -> CRS:
    """Reject missing/geographic CRS before any metric buffering."""
    if crs is None:
        raise ValueError("OSM features need an explicit projected metric CRS")
    parsed = CRS.from_user_input(crs)
    if parsed.is_geographic:
        raise ValueError(f"Refusing to buffer OSM features in geographic CRS {parsed}")
    axes = parsed.axis_info
    if axes and any((axis.unit_name or "").lower() not in {"metre", "meter", "metres", "meters"} for axis in axes[:2]):
        raise ValueError(f"OSM buffering CRS must use meters, got {parsed}")
    return parsed


def classify_osm_feature(tags: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Classify one OSM transportation or physical land-cover feature."""
    highway = normalize_tag(tags.get("highway"))
    area_highway = normalize_tag(tags.get("area:highway"))
    footway = normalize_tag(tags.get("footway"))
    service = normalize_tag(tags.get("service"))
    surface = normalize_tag(tags.get("surface"))
    landuse = normalize_tag(tags.get("landuse"))
    landcover = normalize_tag(tags.get("landcover"))
    natural = normalize_tag(tags.get("natural"))
    leisure = normalize_tag(tags.get("leisure"))
    amenity = normalize_tag(tags.get("amenity"))
    sport = normalize_tag(tags.get("sport"))
    parking = normalize_tag(tags.get("parking"))
    water = normalize_tag(tags.get("water"))
    waterway = normalize_tag(tags.get("waterway"))
    area = normalize_tag(tags.get("area")) == "yes"
    foot = normalize_tag(tags.get("foot"))
    covered = normalize_tag(tags.get("covered")) == "yes"
    bridge = normalize_tag(tags.get("bridge")) not in {None, "no"}
    tunnel = normalize_tag(tags.get("tunnel")) not in {None, "no"}
    try:
        layer = float(normalize_tag(tags.get("layer")) or 0)
    except ValueError:
        layer = 0.0

    result = {
        "included_for_ground_material": False,
        "assigned_surface_class": None,
        "assigned_material": None,
        "material_source": None,
        "classification_source": None,
        "uncertainty_flag": False,
        "missing_surface_tag": surface is None,
        "feature_subcategory": None,
        "rejection_reason": None,
    }
    if bridge or tunnel or layer != 0:
        result["rejection_reason"] = "non_ground_level_bridge_tunnel_or_layer"
        return result
    if normalize_tag(tags.get("building")) or normalize_tag(tags.get("building:part")):
        result["rejection_reason"] = "building_context_not_ground_material"
        return result
    if area_highway in {"footway", "pedestrian", "path"}:
        highway, area = area_highway, True
    is_crossing = footway == "crossing" or normalize_tag(tags.get("crossing")) not in {None, "no"}
    if is_crossing:
        surface_class, subcategory = "pedestrian_crossing", "pedestrian_crossing"
    elif highway == "pedestrian" and area:
        surface_class, subcategory = "pedestrian_plaza", "pedestrian_plaza"
    # A sidewalk tag on a VEHICLE-road centerline describes adjacent pavement;
    # it must not turn the whole carriageway into concrete. Explicit footway
    # geometries remain the surface source until parallel offset generation is
    # requested in a future, separately validated extension.
    elif footway == "sidewalk" or (
            highway == "footway"
            and normalize_tag(tags.get("sidewalk")) in {"both", "left", "right", "yes"}):
        surface_class, subcategory = "sidewalk", "sidewalk"
    elif highway == "steps":
        surface_class, subcategory = "pedestrian_path", "steps"
    elif highway == "cycleway" and foot not in {"no", "private"}:
        surface_class, subcategory = "pedestrian_path", "mixed_use_path"
    elif highway in PEDESTRIAN_HIGHWAYS:
        surface_class = "pedestrian_path"
        subcategory = "independent_footway" if highway == "footway" else "pedestrian_path"
    elif highway in VEHICLE_HIGHWAYS:
        surface_class, subcategory = "vehicle_road", service or highway
    elif natural == "water" or water is not None or (waterway is not None and area):
        surface_class, subcategory = "water", water or waterway or "water"
    elif amenity == "parking" or parking is not None:
        surface_class, subcategory = "parking", parking or "parking"
    elif leisure == "playground":
        surface_class, subcategory = "playground_surface", "playground"
    elif leisure in {"pitch", "sports_centre", "track"} or sport is not None:
        surface_class, subcategory = "sports_surface", sport or leisure or "sports"
    elif (landuse in {"grass", "meadow", "recreation_ground", "village_green"}
          or landcover in {"grass", "grassland", "meadow"}
          or natural == "grassland" or leisure in {"park", "garden"}
          or surface == "grass"):
        surface_class, subcategory = "grass_area", landuse or landcover or natural or leisure or "grass"
    elif (surface in UNPAVED_SURFACES
          or natural in {"sand", "bare_rock", "scree", "shingle"}):
        surface_class, subcategory = "bare_ground", surface or natural or "bare_ground"
    else:
        result["rejection_reason"] = "not_a_supported_transportation_surface"
        return result

    if surface_class == "vehicle_road" and not config["include_vehicle_roads"]:
        result["rejection_reason"] = "vehicle_roads_disabled"
        return result
    if surface_class == "vehicle_road" and highway == "service":
        if not config["include_service_roads"]:
            result["rejection_reason"] = "service_roads_disabled"
            return result
        if service == "parking_aisle" and not config["include_parking_aisles"]:
            result["rejection_reason"] = "parking_aisles_disabled"
            return result
    if surface_class in {"sidewalk", "pedestrian_path"} and not config["include_pedestrian_surfaces"]:
        result["rejection_reason"] = "pedestrian_surfaces_disabled"
        return result
    if surface_class == "pedestrian_crossing" and not config["include_crossings"]:
        result["rejection_reason"] = "crossings_disabled"
        return result
    if surface_class == "pedestrian_plaza" and not config["include_pedestrian_plazas"]:
        result["rejection_reason"] = "pedestrian_plazas_disabled"
        return result
    if surface_class in {"grass_area", "bare_ground"} and not config["include_landcover"]:
        result["rejection_reason"] = "landcover_disabled"
        return result
    if surface_class == "parking" and not config["include_parking_areas"]:
        result["rejection_reason"] = "parking_areas_disabled"
        return result
    if surface_class in {"sports_surface", "playground_surface"} and not config["include_sports_surfaces"]:
        result["rejection_reason"] = "sports_surfaces_disabled"
        return result
    if surface_class == "water" and not config["include_water"]:
        result["rejection_reason"] = "water_disabled"
        return result

    if surface_class == "vehicle_road" and service == "parking_aisle":
        context = "parking"
    elif surface_class == "vehicle_road":
        context = "vehicle"
    elif surface_class in {"sidewalk", "pedestrian_path", "pedestrian_crossing",
                           "pedestrian_plaza"}:
        context = "pedestrian"
    elif surface_class == "parking":
        context = "parking"
    else:
        context = "area"
    explicit_surface = surface
    if context == "parking":
        explicit_surface = normalize_tag(tags.get("parking:surface")) or surface
    elif context == "pedestrian":
        explicit_surface = (
            normalize_tag(tags.get("sidewalk:surface"))
            or normalize_tag(tags.get("sidewalk:left:surface"))
            or normalize_tag(tags.get("sidewalk:right:surface"))
            or surface)
    if explicit_surface in SURFACE_TO_MATERIAL:
        mapped = SURFACE_TO_MATERIAL[explicit_surface]
        material = mapped.get(context, mapped.get("area")) if isinstance(mapped, dict) else mapped
        material_source = "explicit_surface_tag"
    else:
        if surface_class == "water":
            material = config["default_material_by_osm_class"]["water"]
        elif surface_class == "grass_area":
            material = config["default_material_by_osm_class"]["grass_area"]
        elif surface_class == "bare_ground":
            material = config["default_material_by_osm_class"]["bare_ground"]
        elif surface_class == "parking":
            material = config["default_material_by_osm_class"]["parking"]
        elif surface_class == "sports_surface":
            material = config["default_material_by_osm_class"]["sports_area"]
        elif surface_class == "playground_surface":
            material = config["default_material_by_osm_class"]["playground"]
        elif surface in UNPAVED_SURFACES:
            material = config["default_material_by_osm_class"]["unpaved_path"]
        elif surface_class == "pedestrian_crossing":
            material = config["default_material_by_osm_class"]["crossing"]
        elif surface_class == "pedestrian_plaza":
            material = config["default_material_by_osm_class"]["pedestrian_plaza"]
        elif surface_class == "sidewalk":
            material = config["default_material_by_osm_class"]["sidewalk"]
        elif subcategory == "steps":
            material = config["default_material_by_osm_class"]["steps"]
        elif surface_class == "pedestrian_path":
            key = "footway" if highway == "footway" else "pedestrian_path"
            material = config["default_material_by_osm_class"][key]
        elif surface_class == "vehicle_road" and service == "parking_aisle":
            material = config["default_material_by_osm_class"]["parking_aisle"]
        else:
            material = config["default_material_by_osm_class"]["vehicle_road"]
        material_source = "class_default"

    if material not in config["materials"]:
        result["rejection_reason"] = f"unknown_material:{material}"
        return result
    result.update({
        "included_for_ground_material": True,
        "assigned_surface_class": surface_class,
        "assigned_material": material,
        "material_source": material_source,
        "classification_source": "explicit_osm_tags",
        "uncertainty_flag": material_source != "explicit_surface_tag",
        "feature_subcategory": ("covered_walkway" if covered else subcategory),
    })
    return result


def resolve_feature_width(tags: dict[str, Any], surface_class: str,
                          subcategory: str, config: dict[str, Any],
                          is_polygon: bool = False) -> tuple[float | None, str]:
    """Resolve feature width using the documented priority order."""
    if is_polygon:
        return None, "polygon"
    for key, source in (
        ("width", "width"), ("est_width", "est_width"),
        ("sidewalk:width", "sidewalk_width"),
        ("sidewalk:left:width", "sidewalk_width"),
        ("sidewalk:right:width", "sidewalk_width"),
    ):
        parsed = parse_width_m(tags.get(key))
        if parsed is not None:
            return parsed, source
    if config["infer_width_from_lanes"]:
        lanes = parse_width_m(tags.get("lanes"))
        if lanes is not None and lanes >= 1:
            return lanes * float(config["lane_width_m"]), "lane_inference"
    highway = normalize_tag(tags.get("highway")) or "path"
    service = normalize_tag(tags.get("service"))
    if surface_class == "pedestrian_crossing":
        key = "crossing"
    elif surface_class == "pedestrian_plaza":
        key = "pedestrian_plaza"
    elif surface_class == "sidewalk":
        key = "sidewalk"
    elif service in {"parking_aisle", "driveway", "alley"}:
        key = service
    elif subcategory == "mixed_use_path":
        key = "cycleway"
    else:
        key = highway
    fallback = config["fallback_widths_m"].get(key)
    if fallback is None:
        return None, "class_default"
    return float(fallback), "class_default"


def repair_polygon(geometry: Any, tolerance_m: float = 0.02) -> Any:
    """Repair polygon topology without offsetting terrain vertically."""
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    fixed = make_valid(geometry)
    if tolerance_m > 0:
        fixed = fixed.buffer(tolerance_m).buffer(-tolerance_m)
    fixed = make_valid(fixed)
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    polygon_parts = []
    for part in getattr(fixed, "geoms", []):
        if isinstance(part, Polygon):
            polygon_parts.append(part)
        elif isinstance(part, MultiPolygon):
            polygon_parts.extend(part.geoms)
    return (unary_union(polygon_parts) if polygon_parts
            else GeometryCollection())


def resolve_material_overlaps(polygons_by_class: dict[str, list[Any]],
                              ground_polygon: Any,
                              priority: Iterable[str],
                              minimum_area_m2: float = 0.25) -> tuple[dict[str, Any], float, float]:
    """Create a deterministic, mutually exclusive ground partition."""
    unions = {
        name: repair_polygon(unary_union(geoms), 0.0)
        for name, geoms in polygons_by_class.items() if geoms
    }
    names = [name for name in priority if name != "generic_ground"]
    overlap_before = 0.0
    for i, name in enumerate(names):
        if name not in unions:
            continue
        for other in names[i + 1:]:
            if other in unions:
                overlap_before += unions[name].intersection(unions[other]).area
    remaining = repair_polygon(ground_polygon, 0.0)
    resolved: dict[str, Any] = {}
    for name in names:
        geom = repair_polygon(
            unions.get(name, GeometryCollection()).intersection(remaining), 0.0)
        if not geom.is_empty and geom.area >= minimum_area_m2:
            resolved[name] = geom
            remaining = repair_polygon(remaining.difference(geom), 0.0)
    resolved["generic_ground"] = remaining
    overlap_after = 0.0
    values = list(resolved.values())
    for i, geom in enumerate(values):
        for other in values[i + 1:]:
            overlap_after += geom.intersection(other).area
    return resolved, float(overlap_before), float(overlap_after)


def resolve_feature_material_overlaps(records: list[dict[str, Any]],
                                      ground_polygon: Any,
                                      priority: Iterable[str],
                                      minimum_area_m2: float = 0.25) -> tuple[dict[str, Any], float, float]:
    """Resolve classified feature polygons into material-specific regions.

    Surface-class priority controls intersections; within one class, explicit
    ``surface=*`` assignments precede inferred defaults and material name is a
    stable final tie-breaker.  The returned material geometries plus generic
    ground partition the original ground polygon.
    """
    priority_list = [item for item in priority if item != "generic_ground"]
    rank = {name: index for index, name in enumerate(priority_list)}
    grouped: dict[tuple[str, str, str, float], list[Any]] = {}
    for record in records:
        geom = record.get("geometry")
        if geom is None or geom.is_empty:
            continue
        surface_class = str(record["assigned_surface_class"])
        material = str(record["assigned_material"])
        source = str(record.get("material_source", "class_default"))
        override = float(record.get("overlap_priority_override", 0.0) or 0.0)
        grouped.setdefault((surface_class, material, source, override), []).append(geom)
    items = []
    for (surface_class, material, source, override), geoms in grouped.items():
        geom = repair_polygon(unary_union(geoms), 0.0)
        source_rank = 0 if source in {"manual_override", "explicit_surface_tag"} else 1
        items.append((rank.get(surface_class, len(rank)), -override,
                      source_rank, material, geom))
    overlap_before = 0.0
    raw = [item[-1] for item in items]
    for i, geom in enumerate(raw):
        for other in raw[i + 1:]:
            overlap_before += geom.intersection(other).area
    remaining = repair_polygon(ground_polygon, 0.0)
    by_material: dict[str, list[Any]] = {}
    for _, _, _, material, geom in sorted(items, key=lambda item: item[:4]):
        assigned = repair_polygon(geom.intersection(remaining), 0.0)
        if assigned.is_empty or assigned.area < minimum_area_m2:
            continue
        by_material.setdefault(material, []).append(assigned)
        remaining = repair_polygon(remaining.difference(assigned), 0.0)
    resolved = {
        material: repair_polygon(unary_union(geoms), 0.0)
        for material, geoms in by_material.items()
    }
    resolved["generic_ground"] = remaining
    values = list(resolved.values())
    overlap_after = sum(
        values[i].intersection(values[j]).area
        for i in range(len(values)) for j in range(i + 1, len(values))
    )
    return resolved, float(overlap_before), float(overlap_after)


def assign_materials_to_face_centroids(centroids_xy: np.ndarray,
                                       material_polygons: dict[str, Any],
                                       material_names: list[str]) -> np.ndarray:
    """Assign exactly one material ID to every existing terrain face."""
    xy = np.asarray(centroids_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("centroids_xy must be a finite (n,2) array")
    generic_id = material_names.index("generic_ground")
    ids = np.full(len(xy), generic_id, dtype=np.int16)
    # The polygons are already mutually exclusive. Generic is initialized
    # first; classified materials overwrite it only where their region covers
    # a face representative point.
    for name, geom in material_polygons.items():
        if name == "generic_ground" or geom.is_empty:
            continue
        ids[contains_xy(geom, xy[:, 0], xy[:, 1])] = material_names.index(name)
    return ids


def assign_materials_boundary_aware(
        triangles_xy: np.ndarray, material_polygons: dict[str, Any],
        material_names: list[str], material_priority: Iterable[str],
        minimum_overlap_fraction: float = 0.05,
        ) -> tuple[np.ndarray, dict[str, int]]:
    """Classify terrain faces while preserving narrow material polygons.

    Centroids provide the fast complete assignment. Faces near a material
    boundary are then tested by actual plan-area intersection. Specific
    materials overwrite broader classes when at least ``minimum_overlap_fraction``
    of a triangle is covered. The mesh vertices/elevations remain unchanged.
    """
    triangles = np.asarray(triangles_xy, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("triangles_xy must have shape (n_faces, 3, 2)")
    if not (0 < minimum_overlap_fraction <= 1):
        raise ValueError("minimum_overlap_fraction must be in (0,1]")
    centers = triangles.mean(axis=1)
    ids = assign_materials_to_face_centroids(centers, material_polygons, material_names)
    initial = ids.copy()
    max_span = float(np.max(np.ptp(triangles, axis=1))) + 1.0e-6
    rank = [name for name in material_priority
            if name != "generic_ground" and name in material_polygons
            and name in material_names]
    # Process broad/low-priority classes first so specific/high-priority
    # classes overwrite boundary-spanning triangles last.
    for name in reversed(rank):
        geometry = material_polygons[name]
        if geometry is None or geometry.is_empty:
            continue
        candidate = np.flatnonzero(contains_xy(
            geometry.buffer(max_span), centers[:, 0], centers[:, 1]))
        if not len(candidate):
            continue
        footprints = shapely_polygons(triangles[candidate])
        footprint_area = np.asarray(shapely_area(footprints), dtype=float)
        overlap_area = np.asarray(
            shapely_area(shapely_intersection(footprints, geometry)), dtype=float)
        fraction = np.divide(overlap_area, footprint_area,
                             out=np.zeros_like(overlap_area), where=footprint_area > 0)
        ids[candidate[fraction >= minimum_overlap_fraction]] = material_names.index(name)
    changed = ids != initial
    stats = {
        "faces_changed_from_centroid_assignment": int(changed.sum()),
        "non_generic_faces_after_boundary_assignment": int(
            np.sum(ids != material_names.index("generic_ground"))),
    }
    return ids.astype(np.int16), stats


def material_table_for_thermal(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return OSM ground materials in the existing thermal material schema."""
    return {name: dict(values) for name, values in config["materials"].items()}
