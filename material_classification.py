"""material_classification.py -- the hierarchical material resolver.

Every material-identity rule in TREC-Route lives here. Physics lives in
``material_library.py``; this module decides only WHICH material a surface is,
and records how confident it is and why.

THE HIERARCHY
-------------
Higher priority wins outright. A lower-priority source never overrides a higher
one, but it is still recorded as a rejected candidate so the manifest can show
what the alternative would have been.

    1. MANUAL OVERRIDE   an operator said so
    2. OSM DIRECT        an explicit material/surface tag on the object
    3. IMAGERY           a classified raster overlapping the surface
    4. OSM INFERRED      object type or land use implies a likely material
    5. DEFAULT           nothing usable; a NAMED generic class

CONFIDENCE SCHEME
-----------------
Confidence is a property of the EVIDENCE, not of the material. It is not a
probability and nothing in the pipeline consumes it numerically yet; it exists
so QA can sort by it and a later sensitivity study can weight by it.

    1.00  manual override        a human asserted this surface's material
    0.95  osm_direct             an explicit material tag (surface=asphalt,
                                 building:material=brick). Not 1.0: OSM tags can
                                 be stale or wrong, and the tagger was not
                                 measuring optical properties.
    0.80  imagery                a classified raster pixel majority
    0.60  osm_inferred           object type implies material (highway=* with no
                                 surface tag is very probably asphalt)
    0.45  colour_hint            only a COLOUR is known. Colour constrains
                                 albedo but says nothing about conductivity or
                                 heat capacity, so this sits below type
                                 inference on purpose.
    0.30  default                no usable evidence at all

WHAT A WEAK TAG DOES NOT MEAN
-----------------------------
``building=yes`` is not evidence of concrete. It establishes that the object is
a building -- an object type, not a material. It therefore yields
``generic_building_facade`` at DEFAULT confidence with an explicit
``fallback_reason``, never a confident masonry class. The same holds for
``building=house``, ``building=apartments`` and friends: they constrain the
building's use, not its cladding. In the Lisbon cases this matters a great deal,
because 84% of buildings carry ``building=yes`` and only 1% carry
``building:material``.

AERIAL IMAGERY AND VERTICAL FACADES
-----------------------------------
Imagery is accepted for GROUND and ROOF surfaces, and REFUSED for facades. A
nadir orthophoto sees roofs and ground; it does not see walls, and any pixel it
appears to offer for a wall is really the roof or the ground beside it. Feeding
that to a facade would be worse than admitting ignorance, so the facade chain is

    manual > building:material > facade-specific external label > building-type
    inference > colour hint > generic facade

with imagery absent by construction. ``IMAGERY_ALLOWED_CATEGORIES`` enforces it
and ``classify_facade`` never consults a raster.

ROOF VERSUS WALL
----------------
They are classified separately and never share an assignment.
``building:material`` describes the FACADE and is not promoted to the roof;
``roof:material`` describes the roof and is not promoted to the facade. A
building with ``building:material=stone`` and no roof tag gets a stone facade
and a generic roof, each with its own provenance.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import material_library
from material_library import (CATEGORY_FACADE, CATEGORY_GROUND, CATEGORY_ROOF,
                              CATEGORY_VEGETATION, CATEGORY_WATER)
from osm_ground_materials import normalize_tag

# ---------------------------------------------------------------------------
# Sources, priorities and confidences
# ---------------------------------------------------------------------------
SOURCE_MANUAL = "manual_override"
SOURCE_OSM_DIRECT = "osm_direct"
SOURCE_IMAGERY = "imagery"
SOURCE_OSM_INFERRED = "osm_inferred"
SOURCE_COLOUR_HINT = "colour_hint"
SOURCE_DEFAULT = "default"

# Larger wins. Kept as an explicit table rather than list order so a new source
# can be slotted in without silently renumbering the others.
SOURCE_PRIORITY: dict[str, int] = {
    SOURCE_MANUAL: 100,
    SOURCE_OSM_DIRECT: 80,
    SOURCE_IMAGERY: 60,
    SOURCE_OSM_INFERRED: 40,
    SOURCE_COLOUR_HINT: 30,
    SOURCE_DEFAULT: 10,
}

SOURCE_CONFIDENCE: dict[str, float] = {
    SOURCE_MANUAL: 1.00,
    SOURCE_OSM_DIRECT: 0.95,
    SOURCE_IMAGERY: 0.80,
    SOURCE_OSM_INFERRED: 0.60,
    SOURCE_COLOUR_HINT: 0.45,
    SOURCE_DEFAULT: 0.30,
}

# Imagery is only geometrically defensible looking DOWN. See the module docstring.
IMAGERY_ALLOWED_CATEGORIES = frozenset({CATEGORY_GROUND, CATEGORY_ROOF,
                                        CATEGORY_WATER})

LOW_CONFIDENCE_THRESHOLD = 0.5

# Filename of the cached per-face assignment written by
# prepare_surface_materials.py and read by 05a. Declared here, in the light
# module, so a consumer does not have to import the whole preprocessing stage
# (and its pandas/trimesh dependencies) just to name the file.
SURFACE_ASSIGNMENT_FILE = "surface_material_assignment.npz"


class MaterialClassificationError(ValueError):
    """Raised when a rule would produce an unusable assignment."""


@dataclass(frozen=True)
class Candidate:
    """One source's opinion about a surface's material."""

    material_class: str
    source: str
    source_tag: str | None = None
    imagery_class: str | None = None
    fallback_reason: str | None = None
    confidence: float | None = None

    def resolved_confidence(self) -> float:
        if self.confidence is not None:
            return float(self.confidence)
        return SOURCE_CONFIDENCE[self.source]

    def priority(self) -> int:
        return SOURCE_PRIORITY[self.source]


@dataclass
class Assignment:
    """The winning candidate plus everything needed to audit the decision."""

    material_class: str
    material_source: str
    confidence: float
    category: str
    source_tag: str | None = None
    imagery_class: str | None = None
    fallback_reason: str | None = None
    source_object_id: Any = None
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def is_default(self) -> bool:
        return self.material_source == SOURCE_DEFAULT

    def is_low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["rejected_candidates"] = "; ".join(
            f"{item['material_class']}({item['source']})"
            for item in self.rejected)
        record.pop("rejected")
        return record


def resolve(candidates: Iterable[Candidate], category: str,
            source_object_id: Any = None) -> Assignment:
    """Pick the highest-priority candidate; record the rest as rejected.

    Candidates naming a material outside ``category`` are discarded before the
    contest. That guard is what stops a land-use rule from making a wall out of
    grass, or an imagery pixel from turning a roof into asphalt paving.
    """
    if category not in material_library.ALL_CATEGORIES:
        raise MaterialClassificationError(f"unknown category {category!r}")
    usable: list[Candidate] = []
    for candidate in candidates:
        if candidate is None:
            continue
        material = material_library.get(candidate.material_class)
        if material.category != category:
            raise MaterialClassificationError(
                f"rule produced {candidate.material_class!r} "
                f"(category {material.category}) for a {category} surface")
        if candidate.source == SOURCE_IMAGERY and category not in IMAGERY_ALLOWED_CATEGORIES:
            # Defence in depth: classify_facade never offers one, but a caller
            # assembling candidates by hand must not be able to sneak one in.
            continue
        usable.append(candidate)
    if not usable:
        raise MaterialClassificationError(
            f"no candidate for a {category} surface; every classifier must "
            "supply at least an explicit named default")
    usable.sort(key=lambda item: (-item.priority(), -item.resolved_confidence()))
    winner = usable[0]
    return Assignment(
        material_class=winner.material_class,
        material_source=winner.source,
        confidence=winner.resolved_confidence(),
        category=category,
        source_tag=winner.source_tag,
        imagery_class=winner.imagery_class,
        fallback_reason=winner.fallback_reason,
        source_object_id=source_object_id,
        rejected=[{"material_class": item.material_class, "source": item.source,
                   "confidence": item.resolved_confidence(),
                   "source_tag": item.source_tag}
                  for item in usable[1:]],
    )


# ---------------------------------------------------------------------------
# OSM tag tables -- ONE place, per the "do not scatter rules" requirement
# ---------------------------------------------------------------------------
# building:material / material -> facade class. Values are OSM's own vocabulary.
FACADE_MATERIAL_TAGS: dict[str, str] = {
    "brick": "brick_facade", "brick_block": "brick_facade",
    "clay": "brick_facade",
    "stone": "stone_facade", "limestone": "stone_facade",
    "sandstone": "stone_facade", "granite": "stone_facade",
    "marble": "stone_facade", "masonry": "stone_facade",
    "concrete": "concrete_facade", "reinforced_concrete": "concrete_facade",
    "cement_block": "concrete_facade", "cement": "concrete_facade",
    "plaster": "plaster_facade", "stucco": "plaster_facade",
    "render": "rendered_facade", "rendered": "rendered_facade",
    "wood": "wood_facade", "timber": "wood_facade",
    "timber_framing": "wood_facade",
    "metal": "metal_facade", "steel": "metal_facade",
    "aluminium": "metal_facade", "aluminum": "metal_facade",
    "copper": "metal_facade", "zinc": "metal_facade",
    "glass": "glass_facade", "mirror": "glass_facade",
}

# roof:material -> roof class.
ROOF_MATERIAL_TAGS: dict[str, str] = {
    "roof_tiles": "tile_roof", "tile": "tile_roof", "tiles": "tile_roof",
    "clay": "tile_roof", "terracotta": "tile_roof", "slate": "tile_roof",
    "shingle": "tile_roof", "asbestos": "tile_roof",
    "metal": "metal_roof", "steel": "metal_roof", "copper": "metal_roof",
    "zinc": "metal_roof", "tin": "metal_roof", "aluminium": "metal_roof",
    "aluminum": "metal_roof", "corrugated_iron": "metal_roof",
    "concrete": "concrete_roof", "reinforced_concrete": "concrete_roof",
    "stone": "concrete_roof",
    "tar_paper": "membrane_roof", "bitumen": "membrane_roof",
    "roofing_felt": "membrane_roof", "membrane": "membrane_roof",
    "eternit": "membrane_roof", "plastic": "membrane_roof",
    "gravel": "gravel_roof",
    "grass": "green_roof", "green": "green_roof", "vegetation": "green_roof",
    "glass": "glass_roof",
    "wood": "tile_roof",
}

# Named colours -> relative luminance in [0, 1]. Only used as an ALBEDO hint.
NAMED_COLOUR_LUMINANCE: dict[str, float] = {
    "white": 1.00, "ivory": 0.94, "cream": 0.92, "beige": 0.85,
    "silver": 0.75, "lightgrey": 0.75, "lightgray": 0.75, "light_grey": 0.75,
    "yellow": 0.80, "sand": 0.76, "pink": 0.75, "lightblue": 0.72,
    "orange": 0.58, "grey": 0.50, "gray": 0.50, "tan": 0.60,
    "green": 0.42, "lightgreen": 0.62, "red": 0.35, "terracotta": 0.42,
    "brown": 0.30, "blue": 0.25, "darkgrey": 0.28, "darkgray": 0.28,
    "dark_grey": 0.28, "maroon": 0.22, "darkgreen": 0.20, "navy": 0.15,
    "black": 0.03,
}
LIGHT_COLOUR_LUMINANCE = 0.60
DARK_COLOUR_LUMINANCE = 0.35

# Building type -> facade class where the TYPE genuinely implies construction.
# Deliberately short. Most building=* values describe use, not fabric, and must
# not be promoted to a material.
BUILDING_TYPE_FACADE: dict[str, str] = {
    "greenhouse": "glass_facade",
    "hangar": "metal_facade",
    "industrial": "metal_facade",
    "warehouse": "metal_facade",
    "shed": "metal_facade",
    "garage": "metal_facade",
    "garages": "metal_facade",
    "carport": "metal_facade",
    "cabin": "wood_facade",
    "hut": "wood_facade",
    "static_caravan": "metal_facade",
}

# Building type -> roof class, on the same restrictive principle.
BUILDING_TYPE_ROOF: dict[str, str] = {
    "greenhouse": "glass_roof",
    "hangar": "metal_roof",
    "industrial": "metal_roof",
    "warehouse": "metal_roof",
    "shed": "metal_roof",
    "garage": "metal_roof",
    "garages": "metal_roof",
    "carport": "metal_roof",
    "house": "tile_roof",
    "detached": "tile_roof",
    "semidetached_house": "tile_roof",
    "terrace": "tile_roof",
    "bungalow": "tile_roof",
}

# roof:shape values implying a flat roof, where a membrane/gravel build-up is
# more likely than tiles. Shape is weak evidence and stays at inferred level.
FLAT_ROOF_SHAPES = frozenset({"flat"})

# Imagery raster class name -> material. Names are the ones a preparer is asked
# to use in the sidecar JSON; unknown names are rejected loudly rather than
# guessed at.
IMAGERY_CLASS_TO_MATERIAL: dict[str, str] = {
    "asphalt": "asphalt_road",
    "asphalt_road": "asphalt_road",
    "asphalt_pedestrian": "asphalt_pedestrian",
    "concrete": "concrete_pedestrian",
    "paving": "paving_stone_pedestrian",
    "paving_stones": "paving_stone_pedestrian",
    "gravel": "gravel_parking",
    "bare_ground": "bare_ground",
    "soil": "bare_ground",
    "sand": "bare_ground",
    "grass": "grass_lawn",
    "vegetation": "grass_lawn",
    "water": "water",
    "roof_light": "light_roof",
    "roof_dark": "dark_roof",
    "roof_tile": "tile_roof",
    "roof_metal": "metal_roof",
    "roof_gravel": "gravel_roof",
    "roof_green": "green_roof",
}


def colour_luminance(value: Any) -> float | None:
    """Relative luminance of an OSM colour tag, or None if unparseable.

    Accepts a named colour or a ``#rrggbb`` / ``#rgb`` hex triple. The hex path
    uses the Rec. 709 luma weights, which track perceived lightness far better
    than a plain channel mean -- a saturated blue and a saturated yellow have
    very different albedo and identical mean channel values.
    """
    text = normalize_tag(value)
    if text is None:
        return None
    text = text.strip().replace(" ", "")
    if text in NAMED_COLOUR_LUMINANCE:
        return NAMED_COLOUR_LUMINANCE[text]
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(character * 2 for character in digits)
        if len(digits) == 6:
            try:
                red = int(digits[0:2], 16) / 255.0
                green = int(digits[2:4], 16) / 255.0
                blue = int(digits[4:6], 16) / 255.0
            except ValueError:
                return None
            return 0.2126 * red + 0.7152 * green + 0.0722 * blue
    # "light_blue" / "dark_green" style compounds.
    if text.startswith("light") or text.startswith("dark"):
        base = text.replace("light", "", 1).replace("dark", "", 1).strip("_-")
        if base in NAMED_COLOUR_LUMINANCE:
            shift = 0.2 if text.startswith("light") else -0.2
            return min(1.0, max(0.0, NAMED_COLOUR_LUMINANCE[base] + shift))
    return None


def _colour_candidate(colour: Any, light_class: str, dark_class: str,
                      tag_name: str) -> Candidate | None:
    luminance = colour_luminance(colour)
    if luminance is None:
        return None
    if luminance >= LIGHT_COLOUR_LUMINANCE:
        material = light_class
    elif luminance <= DARK_COLOUR_LUMINANCE:
        material = dark_class
    else:
        # Mid-tone: the colour tells us nothing useful about albedo beyond the
        # generic value, so do not manufacture a hint from it.
        return None
    return Candidate(
        material_class=material, source=SOURCE_COLOUR_HINT,
        source_tag=f"{tag_name}={normalize_tag(colour)}",
        fallback_reason=(f"colour luminance {luminance:.2f} used as an ALBEDO "
                         "hint only; colour does not identify the material"))


def classify_facade(tags: Mapping[str, Any],
                    external_label: str | None = None,
                    manual_material: str | None = None,
                    source_object_id: Any = None) -> Assignment:
    """Classify a building FACADE. Aerial imagery is never consulted here."""
    candidates: list[Candidate] = []
    if manual_material:
        candidates.append(Candidate(manual_material, SOURCE_MANUAL,
                                    source_tag="manual_override"))

    material_tag = (normalize_tag(tags.get("building:material"))
                    or normalize_tag(tags.get("material")))
    if material_tag:
        mapped = FACADE_MATERIAL_TAGS.get(material_tag)
        if mapped:
            candidates.append(Candidate(
                mapped, SOURCE_OSM_DIRECT,
                source_tag=f"building:material={material_tag}"))

    if external_label:
        # A facade-specific external label (street-level survey, facade imagery
        # prepared elsewhere). Trusted at imagery level but NOT tagged as
        # imagery, because it did not come from a nadir raster.
        mapped = FACADE_MATERIAL_TAGS.get(normalize_tag(external_label) or "",
                                          external_label)
        if mapped in material_library.MATERIAL_LIBRARY:
            candidates.append(Candidate(
                mapped, SOURCE_OSM_DIRECT,
                source_tag=f"facade_label={external_label}",
                confidence=0.85,
                fallback_reason="external facade-specific label"))

    building = normalize_tag(tags.get("building"))
    inferred = BUILDING_TYPE_FACADE.get(building or "")
    if inferred:
        candidates.append(Candidate(
            inferred, SOURCE_OSM_INFERRED, source_tag=f"building={building}",
            fallback_reason="construction implied by building type"))

    colour = _colour_candidate(tags.get("building:colour")
                               or tags.get("building:color"),
                               "light_facade", "dark_facade", "building:colour")
    if colour:
        candidates.append(colour)

    reason = ("building tag present but no building:material, no facade label "
              "and no usable colour")
    if not building:
        reason = "no building tag; facade classified by geometry alone"
    candidates.append(Candidate(
        material_library.DEFAULT_MATERIAL_BY_CATEGORY[CATEGORY_FACADE],
        SOURCE_DEFAULT,
        source_tag=f"building={building}" if building else None,
        fallback_reason=reason))
    return resolve(candidates, CATEGORY_FACADE, source_object_id)


def classify_roof(tags: Mapping[str, Any],
                  imagery_class: str | None = None,
                  manual_material: str | None = None,
                  source_object_id: Any = None) -> Assignment:
    """Classify a building ROOF. Nadir imagery IS admissible here."""
    candidates: list[Candidate] = []
    if manual_material:
        candidates.append(Candidate(manual_material, SOURCE_MANUAL,
                                    source_tag="manual_override"))

    material_tag = normalize_tag(tags.get("roof:material"))
    if material_tag:
        mapped = ROOF_MATERIAL_TAGS.get(material_tag)
        if mapped:
            candidates.append(Candidate(
                mapped, SOURCE_OSM_DIRECT,
                source_tag=f"roof:material={material_tag}"))

    if imagery_class:
        mapped = IMAGERY_CLASS_TO_MATERIAL.get(normalize_tag(imagery_class) or "")
        if mapped and material_library.get(mapped).category == CATEGORY_ROOF:
            candidates.append(Candidate(mapped, SOURCE_IMAGERY,
                                        imagery_class=imagery_class))

    building = normalize_tag(tags.get("building"))
    inferred = BUILDING_TYPE_ROOF.get(building or "")
    if inferred:
        candidates.append(Candidate(
            inferred, SOURCE_OSM_INFERRED, source_tag=f"building={building}",
            fallback_reason="roof build-up implied by building type"))

    shape = normalize_tag(tags.get("roof:shape"))
    if shape in FLAT_ROOF_SHAPES:
        candidates.append(Candidate(
            "membrane_roof", SOURCE_OSM_INFERRED,
            source_tag=f"roof:shape={shape}", confidence=0.50,
            fallback_reason=("flat roofs are usually membrane or ballasted; "
                             "shape is weaker evidence than material")))

    colour = _colour_candidate(tags.get("roof:colour") or tags.get("roof:color"),
                               "light_roof", "dark_roof", "roof:colour")
    if colour:
        candidates.append(colour)

    reason = "no roof:material, no roof imagery class and no usable roof colour"
    if not building:
        reason = "no building tag; roof classified by geometry alone"
    candidates.append(Candidate(
        material_library.DEFAULT_MATERIAL_BY_CATEGORY[CATEGORY_ROOF],
        SOURCE_DEFAULT,
        source_tag=f"building={building}" if building else None,
        fallback_reason=reason))
    return resolve(candidates, CATEGORY_ROOF, source_object_id)


def classify_ground(osm_material: str | None = None,
                    osm_material_source: str | None = None,
                    osm_source_tag: str | None = None,
                    imagery_class: str | None = None,
                    manual_material: str | None = None,
                    source_object_id: Any = None) -> Assignment:
    """Classify a GROUND surface.

    ``osm_material`` / ``osm_material_source`` come straight from the existing
    ``osm_ground_materials.classify_osm_feature``, whose own vocabulary is
    mapped onto this module's hierarchy rather than duplicated:

        explicit_surface_tag -> osm_direct   (surface=asphalt and friends)
        class_default        -> osm_inferred (highway=* with no surface tag)
    """
    candidates: list[Candidate] = []
    if manual_material:
        candidates.append(Candidate(manual_material, SOURCE_MANUAL,
                                    source_tag="manual_override"))
    if osm_material:
        material = material_library.get(osm_material)
        if osm_material_source == "explicit_surface_tag":
            candidates.append(Candidate(
                osm_material, SOURCE_OSM_DIRECT,
                source_tag=osm_source_tag or "surface=*"))
        else:
            candidates.append(Candidate(
                osm_material, SOURCE_OSM_INFERRED,
                source_tag=osm_source_tag,
                fallback_reason=("material implied by the OSM feature class; "
                                 "no explicit surface tag")))
        category = material.category
    else:
        category = CATEGORY_GROUND

    if imagery_class:
        mapped = IMAGERY_CLASS_TO_MATERIAL.get(normalize_tag(imagery_class) or "")
        if mapped and material_library.get(mapped).category in {CATEGORY_GROUND,
                                                                CATEGORY_WATER}:
            candidates.append(Candidate(mapped, SOURCE_IMAGERY,
                                        imagery_class=imagery_class))
            if not osm_material:
                category = material_library.get(mapped).category

    candidates.append(Candidate(
        material_library.DEFAULT_MATERIAL_BY_CATEGORY[CATEGORY_GROUND]
        if category != CATEGORY_WATER else "water",
        SOURCE_DEFAULT,
        fallback_reason="terrain not covered by any OSM feature or imagery class"))
    # Water and vegetation must never be resolved into a pavement class, so the
    # contest runs inside the category the evidence established.
    usable = [item for item in candidates
              if material_library.get(item.material_class).category == category]
    return resolve(usable, category, source_object_id)


# ---------------------------------------------------------------------------
# Optional inputs
# ---------------------------------------------------------------------------
def load_material_overrides(path: str | Path) -> dict[str, str]:
    """Read ``material_overrides.csv`` into ``{key: material_class}``.

    Accepted key columns, in the order they are looked for:
    ``surface_group_id``, then ``parent_object_id``, then ``object_id``. The
    first non-empty one on a row is the key, so a file may mix whole-object
    overrides with single-group ones.

    Unknown material names are rejected here rather than at use, so a typo in an
    override file fails immediately and visibly instead of quietly not applying.
    """
    path = Path(path)
    overrides: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise MaterialClassificationError(f"{path}: no header row")
        if "material_class" not in reader.fieldnames:
            raise MaterialClassificationError(
                f"{path}: needs a 'material_class' column")
        for line, row in enumerate(reader, start=2):
            material = (row.get("material_class") or "").strip()
            if not material:
                continue
            if material not in material_library.MATERIAL_LIBRARY:
                raise MaterialClassificationError(
                    f"{path} line {line}: unknown material {material!r}")
            key = None
            for column in ("surface_group_id", "parent_object_id", "object_id"):
                value = (row.get(column) or "").strip()
                if value:
                    key = value
                    break
            if key is None:
                raise MaterialClassificationError(
                    f"{path} line {line}: needs surface_group_id, "
                    "parent_object_id or object_id")
            overrides[key] = material
    return overrides


@dataclass
class ImageryClassMap:
    """A classified raster plus the pixel-value -> class-name mapping.

    Deliberately thin. This project does not do image segmentation; it consumes
    a raster somebody else has already classified. The sidecar JSON is

        {"classes": {"1": "asphalt", "2": "grass", "3": "roof_light"},
         "nodata": 0}
    """

    values: Any                      # 2-D integer array
    transform: Any                   # affine transform, raster -> world
    class_names: dict[int, str]
    nodata: int | None = None
    crs: str | None = None

    def class_at(self, x: Any, y: Any) -> list[str | None]:
        """Nearest-pixel class name for each world coordinate."""
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        inverse = ~self.transform
        columns, rows = inverse * (x, y)
        columns = np.floor(np.asarray(columns)).astype(int)
        rows = np.floor(np.asarray(rows)).astype(int)
        height, width = self.values.shape
        inside = ((rows >= 0) & (rows < height)
                  & (columns >= 0) & (columns < width))
        out: list[str | None] = []
        for index, is_inside in enumerate(np.atleast_1d(inside)):
            if not is_inside:
                out.append(None)
                continue
            value = int(self.values[rows[index], columns[index]])
            if self.nodata is not None and value == self.nodata:
                out.append(None)
                continue
            out.append(self.class_names.get(value))
        return out


def load_imagery_classes(raster_path: str | Path,
                         legend_path: str | Path | None = None
                         ) -> ImageryClassMap:
    """Load a classified raster. Requires rasterio; imagery stays OPTIONAL.

    The import is local so that a machine without rasterio can still run the
    whole pipeline -- imagery is an optional input and must never become a hard
    dependency of ordinary runs.
    """
    import json

    try:
        import rasterio
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MaterialClassificationError(
            "imagery classification needs the optional 'rasterio' package; "
            "omit --imagery-raster to run without it") from error

    raster_path = Path(raster_path)
    if legend_path is None:
        legend_path = raster_path.with_suffix(".classes.json")
    legend_path = Path(legend_path)
    if not legend_path.is_file():
        raise MaterialClassificationError(
            f"imagery legend not found: {legend_path}. It must map pixel values "
            "to class names, e.g. {\"classes\": {\"1\": \"asphalt\"}, \"nodata\": 0}")
    legend = json.loads(legend_path.read_text(encoding="utf-8"))
    class_names = {int(key): str(value)
                   for key, value in legend.get("classes", {}).items()}
    unknown = sorted(set(class_names.values()) - set(IMAGERY_CLASS_TO_MATERIAL))
    if unknown:
        raise MaterialClassificationError(
            f"imagery legend names classes this project cannot map: {unknown}; "
            f"known classes: {sorted(IMAGERY_CLASS_TO_MATERIAL)}")
    with rasterio.open(raster_path) as dataset:
        values = dataset.read(1)
        transform = dataset.transform
        crs = str(dataset.crs) if dataset.crs else None
        nodata = legend.get("nodata", dataset.nodata)
    return ImageryClassMap(values=values, transform=transform,
                           class_names=class_names,
                           nodata=None if nodata is None else int(nodata),
                           crs=crs)


def coverage_summary(assignments: Iterable[Assignment],
                     areas: Iterable[float]) -> dict[str, Any]:
    """Area-weighted classification coverage, by source and by material."""
    by_source: dict[str, float] = {}
    by_material: dict[str, float] = {}
    by_category: dict[str, float] = {}
    total = 0.0
    count = 0
    low_confidence_area = 0.0
    for assignment, area in zip(assignments, areas):
        area = float(area)
        total += area
        count += 1
        by_source[assignment.material_source] = (
            by_source.get(assignment.material_source, 0.0) + area)
        by_material[assignment.material_class] = (
            by_material.get(assignment.material_class, 0.0) + area)
        by_category[assignment.category] = (
            by_category.get(assignment.category, 0.0) + area)
        if assignment.is_low_confidence():
            low_confidence_area += area
    denominator = total if total > 0 else 1.0
    return {
        "n_surface_groups": count,
        "total_area_m2": total,
        "area_by_source_m2": by_source,
        "area_by_material_m2": by_material,
        "area_by_category_m2": by_category,
        "fraction_by_source": {key: value / denominator
                               for key, value in by_source.items()},
        "fraction_by_material": {key: value / denominator
                                 for key, value in by_material.items()},
        "default_area_fraction": by_source.get(SOURCE_DEFAULT, 0.0) / denominator,
        "low_confidence_area_fraction": low_confidence_area / denominator,
    }


if __name__ == "__main__":  # pragma: no cover
    print("classification priority (higher wins):")
    for name, priority in sorted(SOURCE_PRIORITY.items(),
                                 key=lambda item: -item[1]):
        print(f"  {priority:4d}  {name:16s} confidence {SOURCE_CONFIDENCE[name]:.2f}")
    print(f"\nimagery admissible for: {sorted(IMAGERY_ALLOWED_CATEGORIES)}")
    print(f"facade material tags: {len(FACADE_MATERIAL_TAGS)}")
    print(f"roof material tags:   {len(ROOF_MATERIAL_TAGS)}")
