#!/usr/bin/env python3
"""verify_material_classification.py -- suite for the material subsystem.

Covers material_library.py (the authoritative property database),
material_classification.py (the hierarchy, OSM rules, imagery and overrides),
surface_groups.py (triangle grouping) and the manifest contract.

Tests A-H from the upgrade specification are labelled inline.

Run: python3 verify_material_classification.py   (exits nonzero on failure)
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

import material_classification as mc
import material_library as ml
import surface_groups as sg

HERE = Path(__file__).resolve().parent
passed = 0
failed = 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(condition: bool, description: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {description}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {description}" + (f" ({detail})" if detail else ""))


print("=" * 70)
print("SURFACE-MATERIAL CLASSIFICATION VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------------------
print("\nL1: the material library is internally consistent")
summary = ml.validate_library()
check(summary["n_materials"] > 30, "library is populated",
      f"{summary['n_materials']} materials")
for name, material in ml.MATERIAL_LIBRARY.items():
    if abs(material.shortwave_absorptivity - (1.0 - material.albedo)) > 1e-12:
        check(False, f"{name}: absorptivity = 1 - albedo")
        break
else:
    check(True, "shortwave_absorptivity == 1 - albedo for EVERY material")
for name, material in ml.MATERIAL_LIBRARY.items():
    product = material.density_kgm3 * material.specific_heat_JkgK
    if abs(material.legacy_entry()["C"] - product) > 1e-6 * product:
        check(False, f"{name}: solver C == density * specific heat")
        break
else:
    check(True, "solver volumetric heat capacity == density * specific heat")
for name, material in ml.MATERIAL_LIBRARY.items():
    low, high = material.albedo_range
    lowe, highe = material.emissivity_range
    if not (low <= material.albedo <= high and lowe <= material.emissivity <= highe):
        check(False, f"{name}: nominal inside its declared range")
        break
else:
    check(True, "every nominal albedo/emissivity lies inside its own range")
check(all(material.source_reference for material in ml.MATERIAL_LIBRARY.values()),
      "every material carries a source reference")

# Uncertainty ranges must be available for a later sensitivity study.
widths = [material.albedo_range[1] - material.albedo_range[0]
          for material in ml.MATERIAL_LIBRARY.values()]
check(min(widths) > 0.0, "every albedo range is a non-degenerate interval",
      f"narrowest {min(widths):.2f}")

# An inconsistent library must be REJECTED, not tolerated.
broken = dict(ml.MATERIAL_LIBRARY)
broken["bad"] = ml.with_overrides("asphalt_road", name="bad", albedo=0.9)
try:
    ml.validate_library(broken)
    check(False, "an albedo outside its declared range is rejected")
except ml.MaterialLibraryError:
    check(True, "an albedo outside its declared range is rejected")
try:
    ml.get("no_such_material")
    check(False, "an unknown material name raises rather than defaulting")
except ml.MaterialLibraryError:
    check(True, "an unknown material name raises rather than defaulting")

# ---------------------------------------------------------------------------
print("\nL2: ONE database -- the legacy tables are derived, not duplicated")
osm = load_module(HERE / "osm_ground_materials.py", "osm_check")
thermal = load_module(HERE / "thermal_common.py", "thermal_check")
mismatch = []
for name, entry in osm.DEFAULT_CONFIG["materials"].items():
    material = ml.get(name)
    if abs(entry["albedo"] - material.albedo) > 1e-12:
        mismatch.append(f"{name} albedo")
    if abs(entry["emissivity"] - material.emissivity) > 1e-12:
        mismatch.append(f"{name} emissivity")
check(not mismatch, "the OSM ground config resolves to library values",
      f"{len(osm.DEFAULT_CONFIG['materials'])} materials")
mismatch = []
for name, entry in thermal.DEFAULT_MATERIALS.items():
    material = ml.get(name)
    for key, value in (("albedo", material.albedo),
                       ("emissivity", material.emissivity),
                       ("k", material.thermal_conductivity_WmK),
                       ("C", material.volumetric_heat_capacity_Jm3K)):
        if abs(entry[key] - value) > 1e-9 * max(abs(value), 1.0):
            mismatch.append(f"{name}.{key}")
check(not mismatch, "the thermal-common table resolves to library values",
      f"{len(thermal.DEFAULT_MATERIALS)} materials")
# Historical values must be preserved exactly, or every existing run changes.
for name, expected in (("ground", (1.00, 2.0e6, 0.18, 0.95)),
                       ("wall", (1.40, 1.8e6, 0.30, 0.90)),
                       ("roof", (1.00, 1.6e6, 0.15, 0.92))):
    entry = thermal.DEFAULT_MATERIALS[name]
    ok = (abs(entry["k"] - expected[0]) < 1e-12
          and abs(entry["C"] - expected[1]) < 1e-3
          and abs(entry["albedo"] - expected[2]) < 1e-12
          and abs(entry["emissivity"] - expected[3]) < 1e-12)
    check(ok, f"legacy '{name}' keeps its exact historical properties")

# ---------------------------------------------------------------------------
print("\nTEST A: an explicit OSM material tag resolves directly, high confidence")
assignment = mc.classify_ground(osm_material="asphalt_road",
                                osm_material_source="explicit_surface_tag",
                                osm_source_tag="surface=asphalt")
check(assignment.material_class == "asphalt_road",
      "surface=asphalt resolves to asphalt", assignment.material_class)
check(assignment.material_source == mc.SOURCE_OSM_DIRECT,
      "provenance is osm_direct", assignment.material_source)
check(assignment.confidence >= 0.9, "confidence is high",
      f"{assignment.confidence}")
check(assignment.source_tag == "surface=asphalt", "the source tag is recorded")
facade = mc.classify_facade({"building": "yes", "building:material": "brick"})
check(facade.material_class == "brick_facade"
      and facade.material_source == mc.SOURCE_OSM_DIRECT,
      "building:material=brick resolves directly too", facade.material_class)

print("\nTEST B: no surface tag but a highway resolves by INFERENCE, not directly")
assignment = mc.classify_ground(osm_material="asphalt_road",
                                osm_material_source="class_default")
check(assignment.material_source == mc.SOURCE_OSM_INFERRED,
      "provenance is osm_inferred, NOT osm_direct", assignment.material_source)
check(assignment.confidence < mc.SOURCE_CONFIDENCE[mc.SOURCE_OSM_DIRECT],
      "inferred confidence is strictly below direct",
      f"{assignment.confidence} < {mc.SOURCE_CONFIDENCE[mc.SOURCE_OSM_DIRECT]}")
check(assignment.fallback_reason is not None,
      "the inference records WHY it was an inference")
# The real OSM classifier must agree that this is the inferred path.
classified = osm.classify_osm_feature(
    {"highway": "residential"}, osm.load_osm_ground_config())
check(classified["material_source"] == "class_default",
      "the OSM feature classifier reports highway-without-surface as a default",
      classified["material_source"])

print("\nTEST C: imagery says concrete, OSM says asphalt -- OSM DIRECT wins")
assignment = mc.classify_ground(osm_material="asphalt_road",
                                osm_material_source="explicit_surface_tag",
                                osm_source_tag="surface=asphalt",
                                imagery_class="concrete")
check(assignment.material_class == "asphalt_road",
      "the direct OSM tag wins over imagery", assignment.material_class)
check(assignment.material_source == mc.SOURCE_OSM_DIRECT, "provenance is osm_direct")
check(any(item["source"] == mc.SOURCE_IMAGERY for item in assignment.rejected),
      "the imagery candidate is RECORDED as rejected, not silently dropped")
# But imagery must beat a mere inference.
assignment = mc.classify_ground(osm_material="asphalt_road",
                                osm_material_source="class_default",
                                imagery_class="grass")
check(assignment.material_source == mc.SOURCE_IMAGERY,
      "imagery outranks an OSM inference", assignment.material_source)
check(mc.SOURCE_PRIORITY[mc.SOURCE_OSM_DIRECT]
      > mc.SOURCE_PRIORITY[mc.SOURCE_IMAGERY]
      > mc.SOURCE_PRIORITY[mc.SOURCE_OSM_INFERRED]
      > mc.SOURCE_PRIORITY[mc.SOURCE_DEFAULT],
      "the documented priority order holds numerically")

print("\nTEST D: a manual override beats everything")
assignment = mc.classify_ground(osm_material="asphalt_road",
                                osm_material_source="explicit_surface_tag",
                                imagery_class="grass",
                                manual_material="concrete_pedestrian")
check(assignment.material_class == "concrete_pedestrian",
      "the manual override wins over a direct OSM tag AND imagery")
check(assignment.material_source == mc.SOURCE_MANUAL
      and assignment.confidence == 1.0,
      "manual provenance and confidence are recorded")
facade = mc.classify_facade({"building:material": "brick"},
                            manual_material="stone_facade")
check(facade.material_class == "stone_facade",
      "the same holds for facades", facade.material_class)

print("\nTEST E: building:material and roof:material stay SEPARATE")
tags = {"building": "yes", "building:material": "stone",
        "roof:material": "roof_tiles"}
facade = mc.classify_facade(tags)
roof = mc.classify_roof(tags)
check(facade.material_class == "stone_facade",
      "the facade takes building:material", facade.material_class)
check(roof.material_class == "tile_roof",
      "the roof takes roof:material", roof.material_class)
check(ml.get(facade.material_class).category == ml.CATEGORY_FACADE
      and ml.get(roof.material_class).category == ml.CATEGORY_ROOF,
      "each lands in its own material category")
# building:material must NOT leak onto a roof that has no roof tag.
roof_only = mc.classify_roof({"building": "yes", "building:material": "stone"})
check(roof_only.material_class == "generic_roof"
      and roof_only.material_source == mc.SOURCE_DEFAULT,
      "building:material does NOT become the roof material",
      roof_only.material_class)
facade_only = mc.classify_facade({"building": "yes", "roof:material": "metal"})
check(facade_only.material_class == "generic_building_facade",
      "roof:material does NOT become the facade material",
      facade_only.material_class)

print("\nTEST F: an unknown building gets an EXPLICIT default at low confidence")
assignment = mc.classify_facade({"building": "yes"})
check(assignment.material_class == "generic_building_facade",
      "building=yes yields a named generic class, not concrete",
      assignment.material_class)
check(assignment.material_source == mc.SOURCE_DEFAULT,
      "provenance says default")
check(assignment.confidence <= 0.3, "confidence is low", f"{assignment.confidence}")
check(assignment.fallback_reason and "building:material" in assignment.fallback_reason,
      "the fallback reason names what was missing", assignment.fallback_reason)
check(assignment.is_default() and assignment.is_low_confidence(),
      "the assignment reports itself as a low-confidence default")
# No silent defaults anywhere: every default class is a real library entry.
for category, name in ml.DEFAULT_MATERIAL_BY_CATEGORY.items():
    if name not in ml.MATERIAL_LIBRARY:
        check(False, f"default for {category} exists in the library")
        break
else:
    check(True, "every category's default is a NAMED library material")

print("\nTEST G: one material yields one set of properties everywhere")
for name in ("asphalt_road", "grass_lawn", "paving_stone_pedestrian", "water"):
    material = ml.get(name)
    osm_entry = osm.DEFAULT_CONFIG["materials"][name]
    thermal_entry = thermal.DEFAULT_MATERIALS[name]
    same = (abs(osm_entry["albedo"] - material.albedo) < 1e-12
            and abs(thermal_entry["albedo"] - material.albedo) < 1e-12
            and abs(osm_entry["emissivity"] - material.emissivity) < 1e-12
            and abs(thermal_entry["emissivity"] - material.emissivity) < 1e-12
            and abs(thermal_entry["k"] - material.thermal_conductivity_WmK) < 1e-12
            and abs(thermal_entry["C"] - material.volumetric_heat_capacity_Jm3K) < 1.0)
    check(same, f"{name}: radiation and energy modules see identical properties",
          f"albedo {material.albedo}, eps {material.emissivity}")
# The absorbing side and the reflecting side must add to one.
material = ml.get("asphalt_road")
check(abs(material.albedo + material.shortwave_absorptivity - 1.0) < 1e-12,
      "reflected + absorbed shortwave = 1 for the same surface")

print("\nTEST H: manifest coverage -- every group appears exactly once")
prep = load_module(HERE / "prepare_surface_materials.py", "prep_check")
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5],
                  [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3],
                  [3, 7, 4], [3, 4, 0]])
group_index, groups = sg.build_surface_groups(
    vertices, faces, ["b1"] * 12, ["wall"] * 12, ["wall"] * 12)
assignments = [mc.classify_facade({"building": "yes"}) for _ in groups]
frame = prep.manifest_frame(groups, assignments, "buildings")
check(len(frame) == len(groups), "one manifest row per surface group",
      f"{len(frame)} rows, {len(groups)} groups")
check(frame["surface_group_id"].nunique() == len(frame),
      "surface group ids are unique")
check(set(np.unique(group_index)) == set(range(len(groups))),
      "every group index refers to a real group")
covered = np.zeros(len(faces), dtype=int)
for group in groups:
    covered[group.face_indices] += 1
check(np.all(covered == 1),
      "every face belongs to exactly one group -- no gaps, no double counting")
required = {"surface_group_id", "parent_object_id", "surface_type",
            "material_class", "material_source", "confidence", "area_m2",
            "albedo", "emissivity", "thermal_conductivity_WmK", "density_kgm3",
            "specific_heat_JkgK", "albedo_min", "albedo_max",
            "emissivity_min", "emissivity_max", "mean_normal_z",
            "centroid_x", "fallback_reason", "imagery_class"}
missing = required - set(frame.columns)
check(not missing, "the manifest carries every required column",
      f"{len(frame.columns)} columns")
check(abs(float(frame["area_m2"].sum()) - 6.0) < 1e-9,
      "manifest area totals the true surface area", f"{frame['area_m2'].sum():.3f}")

# ---------------------------------------------------------------------------
print("\nG1: geometric grouping")
check(len(groups) == 6, "a unit cube becomes its six faces", f"{len(groups)}")
orientations = {group.orientation for group in groups}
check({"north", "south", "east", "west", "roof"} <= orientations,
      "groups are named by compass orientation, so ids are human-checkable",
      f"{sorted(orientations)}")
segments = 24
angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
cylinder_vertices = np.vstack([
    np.column_stack([np.cos(angles), np.sin(angles), np.zeros(segments)]),
    np.column_stack([np.cos(angles), np.sin(angles), np.ones(segments)])])
cylinder_faces = []
for index in range(segments):
    nxt = (index + 1) % segments
    cylinder_faces += [[index, nxt, segments + index],
                       [nxt, segments + nxt, segments + index]]
cylinder_faces = np.array(cylinder_faces)
_, curved = sg.build_surface_groups(
    cylinder_vertices, cylinder_faces, ["cyl"] * len(cylinder_faces),
    ["wall"] * len(cylinder_faces), ["wall"] * len(cylinder_faces))
check(len(curved) > 1, "a curved surface is NOT over-merged into one patch",
      f"{len(curved)} groups from {segments} segments")

# Different parents must never merge, however coplanar.
split_parents = ["a"] * 6 + ["b"] * 6
_, parent_groups = sg.build_surface_groups(
    vertices, faces, split_parents, ["wall"] * 12, ["wall"] * 12)
check(all(len({split_parents[i] for i in group.face_indices}) == 1
          for group in parent_groups),
      "no group ever spans two parent objects")
_, material_groups = sg.build_surface_groups(
    vertices, faces, ["b1"] * 12, ["wall"] * 12,
    ["brick_facade"] * 6 + ["stone_facade"] * 6)
check(all(len({("brick_facade" if i < 6 else "stone_facade")
               for i in group.face_indices}) == 1
          for group in material_groups),
      "coplanar triangles with DIFFERENT materials stay separate")

# Welding: the STL trap that made grouping meaningless.
unwelded_vertices = vertices[faces].reshape(-1, 3)
unwelded_faces = np.arange(len(unwelded_vertices)).reshape(-1, 3)
_, unwelded_groups = sg.build_surface_groups(
    unwelded_vertices, unwelded_faces, ["b1"] * 12, ["wall"] * 12, ["wall"] * 12)
check(len(unwelded_groups) == 6,
      "an STL-style mesh with duplicated vertices still groups correctly -- "
      "without welding this silently yields one group per triangle",
      f"{len(unwelded_groups)} groups")
welded = sg.weld_face_indices(unwelded_vertices, unwelded_faces)
check(welded.shape == unwelded_faces.shape,
      "welding preserves face count and order, so face indices stay valid")

# ---------------------------------------------------------------------------
print("\nI1: imagery is refused for vertical facades")
check(ml.CATEGORY_FACADE not in mc.IMAGERY_ALLOWED_CATEGORIES,
      "facades are excluded from imagery by policy")
check(ml.CATEGORY_ROOF in mc.IMAGERY_ALLOWED_CATEGORIES
      and ml.CATEGORY_GROUND in mc.IMAGERY_ALLOWED_CATEGORIES,
      "roofs and ground accept imagery, where nadir view is defensible")
sneaky = [mc.Candidate("light_facade", mc.SOURCE_IMAGERY),
          mc.Candidate("generic_building_facade", mc.SOURCE_DEFAULT)]
result = mc.resolve(sneaky, ml.CATEGORY_FACADE)
check(result.material_source == mc.SOURCE_DEFAULT,
      "an imagery candidate handed to a facade is dropped, not honoured",
      result.material_source)
roof = mc.classify_roof({"building": "yes"}, imagery_class="roof_light")
check(roof.material_class == "light_roof"
      and roof.material_source == mc.SOURCE_IMAGERY,
      "the same imagery IS honoured for a roof", roof.material_class)

print("\nI2: colour is an albedo hint only, and ranks below type inference")
check(mc.SOURCE_PRIORITY[mc.SOURCE_COLOUR_HINT]
      < mc.SOURCE_PRIORITY[mc.SOURCE_OSM_INFERRED],
      "colour ranks below object-type inference")
white = mc.classify_roof({"building": "yes", "roof:colour": "white"})
check(white.material_class == "light_roof"
      and white.material_source == mc.SOURCE_COLOUR_HINT,
      "a white roof becomes a high-albedo generic roof", white.material_class)
check(ml.get("light_roof").albedo > ml.get("dark_roof").albedo,
      "light and dark colour classes differ in albedo as intended",
      f"{ml.get('light_roof').albedo} vs {ml.get('dark_roof').albedo}")
check(white.fallback_reason and "does not identify the material" in white.fallback_reason,
      "the record states that colour does not identify the material")
check(abs(mc.colour_luminance("#ffffff") - 1.0) < 1e-9
      and abs(mc.colour_luminance("#000000")) < 1e-9,
      "hex colours are parsed")
check(mc.colour_luminance("nonsense_colour") is None,
      "an unparseable colour yields no hint rather than a guess")
mid = mc.classify_facade({"building": "yes", "building:colour": "grey"})
check(mid.material_source == mc.SOURCE_DEFAULT,
      "a mid-tone colour produces NO hint, since it adds nothing to the generic",
      mid.material_class)

print("\nI3: water and vegetation are never turned into pavement")
water = mc.classify_ground(osm_material="water",
                           osm_material_source="explicit_surface_tag",
                           imagery_class="asphalt")
check(water.material_class == "water",
      "an asphalt imagery pixel cannot override open water", water.material_class)
check(ml.get("water").category == ml.CATEGORY_WATER,
      "water has its own dedicated category")
check(ml.get("tree_canopy").category == ml.CATEGORY_VEGETATION,
      "vegetation has its own category and is not a ground material")
try:
    mc.resolve([mc.Candidate("grass_lawn", mc.SOURCE_OSM_DIRECT)],
               ml.CATEGORY_FACADE)
    check(False, "a ground material offered for a facade is rejected")
except mc.MaterialClassificationError:
    check(True, "a ground material offered for a facade is rejected")

print("\nI4: manual override files")
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "material_overrides.csv"
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["surface_group_id", "parent_object_id", "material_class"])
        writer.writerow(["building_00001_south_wall_003", "", "brick_facade"])
        writer.writerow(["", "building_00002", "metal_facade"])
    overrides = mc.load_material_overrides(path)
    check(overrides == {"building_00001_south_wall_003": "brick_facade",
                        "building_00002": "metal_facade"},
          "overrides load by group id or by parent object id", str(overrides))
    bad = Path(directory) / "bad.csv"
    bad.write_text("surface_group_id,material_class\ng1,not_a_material\n",
                   encoding="utf-8")
    try:
        mc.load_material_overrides(bad)
        check(False, "an unknown material in an override file is rejected at load")
    except mc.MaterialClassificationError:
        check(True, "an unknown material in an override file is rejected at load")

print("\nI5: coverage statistics")
assignments = [
    mc.classify_ground(osm_material="asphalt_road",
                       osm_material_source="explicit_surface_tag"),
    mc.classify_ground(osm_material="grass_lawn",
                       osm_material_source="class_default"),
    mc.classify_facade({"building": "yes"}),
]
stats = mc.coverage_summary(assignments, [100.0, 100.0, 200.0])
check(abs(stats["total_area_m2"] - 400.0) < 1e-9, "total area is summed")
check(abs(stats["fraction_by_source"][mc.SOURCE_DEFAULT] - 0.5) < 1e-9,
      "the default fraction is area-weighted, not count-weighted",
      f"{stats['fraction_by_source'][mc.SOURCE_DEFAULT]:.2f}")
check(abs(sum(stats["fraction_by_source"].values()) - 1.0) < 1e-9,
      "provenance fractions sum to one")
check(abs(stats["default_area_fraction"] - 0.5) < 1e-9,
      "the default-area fraction is reported for the QA threshold")

print("\nI6: backward compatibility -- no imagery, no overrides")
plain = mc.classify_ground(osm_material="asphalt_road",
                           osm_material_source="explicit_surface_tag")
check(plain.material_class == "asphalt_road",
      "the hierarchy degrades cleanly to OSM direct > inferred > default")
check(mc.load_imagery_classes.__doc__ and "optional"
      in mc.load_imagery_classes.__doc__.lower(),
      "imagery is documented as optional")
facets = load_module(HERE / "05a_thermal_facets_select.py", "facets_check")
check(hasattr(facets, "SURFACE_ASSIGNMENT_FILE"),
      "05a knows the cached assignment filename")
source = (HERE / "05a_thermal_facets_select.py").read_text(encoding="utf-8")
check("if args.surface_material_dir:" in source,
      "05a applies the assignment only when one is supplied, so existing runs "
      "are unaffected")

print("\n" + "=" * 70)
print(f"RESULT: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
