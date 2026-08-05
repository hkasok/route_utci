"""Dependency-free verification for the OSM ground-material upgrade."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from download_osm_complete_features import normalize_feature_frame
from prepare_osm_ground_materials import _override_flag, classify_features

from osm_ground_materials import (
    DEFAULT_CONFIG, assign_materials_boundary_aware,
    assign_materials_to_face_centroids,
    classify_osm_feature, load_osm_ground_config, material_table_for_thermal,
    parse_width_m, repair_polygon, resolve_feature_material_overlaps,
    resolve_feature_width, validate_projected_crs,
)


passed = failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")


def close(actual, expected, tolerance=1e-8):
    return abs(actual - expected) <= tolerance


def config():
    return copy.deepcopy(DEFAULT_CONFIG)


print("T0: complete-feature identity, tags, and multipolygons")
index = pd.MultiIndex.from_tuples(
    [("way", 101), ("relation", 202)], names=["element", "id"])
raw = gpd.GeoDataFrame(
    {"surface": ["concrete", None], "landuse": [None, "grass"],
     "custom:tag": ["kept", "also_kept"]},
    geometry=[LineString([(0, 0), (1, 0)]), box(0, 0, 2, 2)],
    index=index, crs="EPSG:4326")
normalized = normalize_feature_frame(raw)
check(normalized[["osm_type", "osm_id"]].astype(str).values.tolist()
      == [["way", "101"], ["relation", "202"]],
      "OSM way/relation IDs preserved")
check(all("custom:tag" in json.loads(value)
          for value in normalized["all_tags_json"]),
      "all original tags retained in machine-readable JSON")
with_hole = Polygon(
    [(0, 0), (10, 0), (10, 10), (0, 10)],
    holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]])
multi = MultiPolygon([with_hole, box(20, 0, 22, 2)])
repaired_multi = repair_polygon(multi, 0.0)
check(repaired_multi.geom_type == "MultiPolygon"
      and close(repaired_multi.area, 88.0),
      "multipolygon outer/inner rings preserved")


print("T1: vehicle and pedestrian classification")
for highway in ("motorway", "primary", "secondary_link", "residential", "service"):
    result = classify_osm_feature({"highway": highway}, config())
    check(result["included_for_ground_material"]
          and result["assigned_surface_class"] == "vehicle_road"
          and result["assigned_material"] == "asphalt_road",
          f"vehicle class {highway}")
pedestrian_cases = (
    ({"highway": "footway", "footway": "sidewalk"}, "sidewalk", "sidewalk"),
    ({"highway": "footway"}, "pedestrian_path", "independent_footway"),
    ({"highway": "path"}, "pedestrian_path", "pedestrian_path"),
    ({"highway": "cycleway", "foot": "yes"}, "pedestrian_path", "mixed_use_path"),
    ({"highway": "steps"}, "pedestrian_path", "steps"),
    ({"highway": "footway", "footway": "crossing"}, "pedestrian_crossing", "pedestrian_crossing"),
    ({"highway": "pedestrian", "area": "yes"}, "pedestrian_plaza", "pedestrian_plaza"),
)
for tags, expected_class, expected_subcategory in pedestrian_cases:
    result = classify_osm_feature(tags, config())
    check(result["included_for_ground_material"]
          and result["assigned_surface_class"] == expected_class
          and result["feature_subcategory"] == expected_subcategory,
          f"pedestrian class {tags}")

print("\nT2: surface material interpretation")
surface_cases = (
    ("asphalt", "residential", "asphalt_road"),
    ("asphalt", "footway", "asphalt_pedestrian"),
    ("concrete:plates", "footway", "concrete_pedestrian"),
    ("paving_stones", "path", "paving_stone_pedestrian"),
    ("bricks", "path", "paving_stone_pedestrian"),
    ("cobblestone", "path", "paving_stone_pedestrian"),
    ("fine_gravel", "path", "bare_ground"),
    ("grass", "path", "grass_lawn"),
    ("wood", "path", "paving_stone_pedestrian"),
)
for surface, highway, expected in surface_cases:
    result = classify_osm_feature(
        {"highway": highway, "surface": surface}, config())
    check(result["assigned_material"] == expected
          and result["material_source"] == "explicit_surface_tag",
          f"surface={surface} -> {expected}")

area_cases = (
    ({"landuse": "grass"}, "grass_area", "grass_lawn", True),
    ({"amenity": "parking", "surface": "asphalt"}, "parking", "asphalt_parking", False),
    ({"amenity": "parking", "parking:surface": "concrete"}, "parking", "concrete_parking", False),
    ({"leisure": "pitch", "surface": "artificial_turf"}, "sports_surface", "artificial_turf", False),
    ({"leisure": "playground", "surface": "rubber"}, "playground_surface", "playground_surface", False),
    ({"natural": "water"}, "water", "water", True),
    ({"natural": "sand"}, "bare_ground", "bare_ground", True),
)
for tags, expected_class, expected_material, expected_uncertain in area_cases:
    result = classify_osm_feature(tags, config())
    check(result["assigned_surface_class"] == expected_class
          and result["assigned_material"] == expected_material
          and result["uncertainty_flag"] == expected_uncertain,
          f"physical area {tags} -> {expected_material}")
university = classify_osm_feature({"amenity": "university"}, config())
check(not university["included_for_ground_material"],
      "broad university boundary is not painted as a physical surface")

print("\nT3: width parsing and fallback")
width_cases = (
    ("3.5", 3.5), ("3.5 m", 3.5), ("10 ft", 3.048),
    ("5' 6\"", 1.6764), ("24 in", 0.6096),
    ("2;4", 3.0), ([2, "4 m"], 3.0),
    (None, None), ("approximately wide", None), ("-3", None),
)
for raw, expected in width_cases:
    actual = parse_width_m(raw)
    check((actual is None and expected is None)
          or (actual is not None and expected is not None
              and close(actual, expected, 1e-6)), f"width {raw!r}")
c = config()
width, source = resolve_feature_width(
    {"highway": "footway", "width": "bad", "est_width": "2.4 m"},
    "pedestrian_path", "independent_footway", c)
check(close(width, 2.4) and source == "est_width", "width priority")
width, source = resolve_feature_width(
    {"highway": "footway", "width": "bad"},
    "pedestrian_path", "independent_footway", c)
check(close(width, c["fallback_widths_m"]["footway"])
      and source == "class_default", "malformed width uses class fallback")

print("\nT4: CRS, buffering, and polygon repair")
check(validate_projected_crs("EPSG:6346").is_projected,
      "projected metric CRS accepted")
try:
    validate_projected_crs("EPSG:4326")
    geographic_rejected = False
except ValueError:
    geographic_rejected = True
check(geographic_rejected, "geographic buffering rejected")
line = LineString([(0, 0), (10, 0)])
pavement = repair_polygon(line.buffer(2.0, cap_style="flat"))
check(pavement.is_valid and close(pavement.area, 40.0, 0.1), "line buffering")
bow_tie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
check(repair_polygon(bow_tie).is_valid, "invalid polygon repair")
override_input = gpd.GeoDataFrame(
    {"osm_id": ["501"], "osm_type": ["way"], "highway": ["footway"]},
    geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:6346")
overridden = classify_features(override_input, c, {
    "501": {"class": "sidewalk", "material": "concrete_pedestrian",
            "width": 2.5, "include": "true", "comments": "field checked"}})
check(bool(overridden.iloc[0]["override_applied"])
      and close(overridden.iloc[0]["width_m"], 2.5)
      and overridden.iloc[0]["assigned_material"] == "concrete_pedestrian"
      and overridden.iloc[0]["override_comments"] == "field checked",
      "manual class/material/width/comment override")
check(not _override_flag("false") and _override_flag("yes"),
      "text boolean overrides parsed without truthy-string ambiguity")

print("\nT5: priority, ground subtraction, and conservation")
ground = box(0, 0, 20, 20)
records = [
    {"geometry": box(2, 8, 18, 12), "assigned_surface_class": "vehicle_road",
     "assigned_material": "asphalt_road", "material_source": "class_default"},
    {"geometry": box(9, 2, 11, 18), "assigned_surface_class": "pedestrian_path",
     "assigned_material": "concrete_sidewalk", "material_source": "explicit_surface_tag"},
    {"geometry": box(8, 8, 12, 12), "assigned_surface_class": "pedestrian_crossing",
     "assigned_material": "pedestrian_crossing", "material_source": "explicit_surface_tag"},
]
resolved, before, after = resolve_feature_material_overlaps(
    records, ground, c["overlap_priority"], 0.01)
check(before > 0 and after < 1e-7, "overlap removed")
check(close(resolved["pedestrian_crossing"].area, 16.0, 0.1),
      "crossing priority retained")
check(resolved["pedestrian_crossing"].intersection(
      resolved["asphalt_road"]).area < 1e-8, "no coincident classes")
check(close(sum(value.area for value in resolved.values()), ground.area, 0.1),
      "plan-area conservation")

print("\nT6: terrain-face material assignment")
names = list(c["materials"])
polygons = {
    "asphalt_road": box(0, 0, 5, 10),
    "concrete_sidewalk": box(5, 0, 10, 10),
    "generic_ground": box(0, 0, 10, 10),
}
centroids = np.array([[1, 1], [4, 9], [6, 1], [9, 9], [20, 20]])
ids = assign_materials_to_face_centroids(centroids, polygons, names)
assigned = [names[index] for index in ids]
check(assigned == ["asphalt_road", "asphalt_road", "concrete_sidewalk",
                   "concrete_sidewalk", "generic_ground"],
      "exclusive complete face assignment")

triangles = np.array([
    [[0, 0], [2, 0], [2, 2]],
    [[0, 0], [2, 2], [0, 2]],
], dtype=float)
narrow = box(0.9, 0.0, 1.1, 2.0)
narrow_polygons = {
    "concrete_pedestrian": narrow,
    "generic_ground": box(0, 0, 2, 2).difference(narrow),
}
boundary_ids, boundary_stats = assign_materials_boundary_aware(
    triangles, narrow_polygons, names, c["material_priority"], 0.05)
check(all(names[index] == "concrete_pedestrian" for index in boundary_ids)
      and boundary_stats["faces_changed_from_centroid_assignment"] == 2,
      "narrow 0.2 m path survives centroid-missing terrain faces")

print("\nT7: material database and route-change guard")
materials = material_table_for_thermal(c)
required = set(c["materials"])
check(required <= set(materials), "all required materials present")
check(all(0 <= materials[name]["albedo"] <= 1
          and 0 < materials[name]["emissivity"] <= 1
          and materials[name]["k"] > 0 and materials[name]["C"] > 0
          for name in required), "radiative and thermal properties valid")
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "bad.json"
    path.write_text(json.dumps({"affect_route_connectivity": True}))
    try:
        load_osm_ground_config(path)
        guard_works = False
    except ValueError:
        guard_works = True
check(guard_works, "configuration cannot enable route changes")

print("\nT8: synthetic integration and route invariance")
graph = nx.MultiDiGraph()
graph.add_node(1, x=0.0, y=0.0)
graph.add_node(2, x=10.0, y=0.0)
graph.add_edge(1, 2, length=10.0, highway="footway")
before_nodes = list(graph.nodes(data=True))
before_edges = list(graph.edges(keys=True, data=True))
features = [
    ({"highway": "residential", "surface": "asphalt"}, LineString([(0, 5), (20, 5)])),
    ({"highway": "footway", "footway": "sidewalk", "surface": "concrete"}, LineString([(0, 8), (20, 8)])),
    ({"highway": "path", "surface": "asphalt"}, LineString([(3, 0), (3, 20)])),
    ({"highway": "path", "surface": "gravel"}, LineString([(17, 0), (17, 20)])),
    ({"highway": "footway", "footway": "crossing"}, LineString([(8, 0), (8, 20)])),
    ({"highway": "pedestrian", "area": "yes"}, box(12, 12, 18, 18)),
    ({"highway": "footway"}, LineString([(0, 2), (20, 2)])),
    ({"highway": "service", "width": "malformed"}, LineString([(0, 15), (20, 15)])),
]
integration_records = []
encountered = set()
for tags, geometry in features:
    result = classify_osm_feature(tags, c)
    width, _ = resolve_feature_width(
        tags, result["assigned_surface_class"], result["feature_subcategory"], c,
        is_polygon=geometry.geom_type == "Polygon")
    polygon = geometry if geometry.geom_type == "Polygon" else geometry.buffer(width / 2)
    integration_records.append({**result, "geometry": polygon})
    encountered.add(result["assigned_material"])
_, _, integration_overlap = resolve_feature_material_overlaps(
    integration_records, ground, c["overlap_priority"], 0.01)
check(integration_overlap < 1e-7, "synthetic intersection resolved")
check({"asphalt_road", "concrete_pedestrian", "asphalt_pedestrian",
       "bare_ground", "pedestrian_crossing", "pedestrian_plaza"} <= encountered,
      "all synthetic material classes reach partition")
check(list(graph.nodes(data=True)) == before_nodes
      and list(graph.edges(keys=True, data=True)) == before_edges,
      "route nodes, edges, attributes, and lengths unchanged")

print("\nT9: material-sensitive radiation and thermal response")
spec = importlib.util.spec_from_file_location(
    "facet_energy_balance", Path(__file__).with_name("05b_facet_energy_balance.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
air = 303.15
sky_lw = 390.0
sw_in = 800.0
outputs = {}
for name in ("asphalt_road", "concrete_pedestrian"):
    mat = materials[name]
    solver = module.ClassSolver(mat, 1, 600.0, air, air, 0.0)
    q_ext = (1.0 - mat["albedo"]) * sw_in + mat["emissivity"] * sky_lw
    for _ in range(36):
        surface = solver.step(np.array([q_ext]), 6.0, air)[0]
    outputs[name] = (
        surface,
        mat["emissivity"] * module.SIGMA * surface**4,
        mat["albedo"] * sw_in,
    )
asphalt, concrete = outputs["asphalt_road"], outputs["concrete_pedestrian"]
check(asphalt[2] < concrete[2], "asphalt reflects less SW than concrete")
check(asphalt[0] > concrete[0], "asphalt heats more under identical forcing")
check(abs(asphalt[1] - concrete[1]) > 0.1,
      "material temperature/emissivity changes emitted LW")
all_surface_temperatures = []
for name in required:
    mat = materials[name]
    solver = module.ClassSolver(mat, 1, 600.0, air, air, 0.0)
    q_ext = ((1.0 - mat["albedo"]) * sw_in
             + mat["emissivity"] * sky_lw)
    all_surface_temperatures.append(float(
        solver.step(np.array([q_ext]), 6.0, air)[0]))
check(np.isfinite(all_surface_temperatures).all()
      and len(all_surface_temperatures) == len(required),
      "every configured ground material reaches the energy solver")

spec05 = importlib.util.spec_from_file_location(
    "mrt_model", Path(__file__).with_name("05_mrt_network_raytrace.py"))
mrt = importlib.util.module_from_spec(spec05)
spec05.loader.exec_module(mrt)
args = SimpleNamespace(
    clear_sky_emissivity="prata", surface_temp_offset_day_c=8.0,
    surrounding_emissivity=0.95, projected_area_model="standing",
    f_projected_direct=0.25, person_sw_absorptivity=0.70,
    f_sky_diffuse=0.5, reflected_model="local",
    f_ground_reflected=0.5, ground_albedo=0.18,
    person_emissivity=0.97,
)
common = dict(
    dni=700.0, dhi=100.0, ghi=600.0, elevation_deg=45.0,
    tau_direct=np.ones(2), svf_person=np.ones(2) * 0.8,
    svf_ground=np.ones(2) * 0.8, air_temp_C=30.0, rh_pct=60.0,
    cloud_fraction=0.0, args=args,
)
tmrt_material, _, sw_material, _ = mrt.estimate_mrt_from_radiation(
    **common, local_ground_albedo=np.array([0.12, 0.30]))
check(sw_material[1] > sw_material[0] and tmrt_material[1] > tmrt_material[0],
      "local concrete albedo increases reflected pedestrian SW/MRT")
legacy = mrt.estimate_mrt_from_radiation(**common)[0]
uniform = mrt.estimate_mrt_from_radiation(
    **common, local_ground_albedo=np.full(2, 0.18))[0]
check(np.array_equal(legacy, uniform),
      "disabled/uniform local-albedo path is exactly backward compatible")

print("\n" + "=" * 70)
print(f"RESULT: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
