"""material_library.py -- the authoritative surface-material property database.

ONE PLACE. Every albedo, emissivity, conductivity, density and specific heat
used anywhere in TREC-Route resolves here. Before this module the ground classes
lived in ``osm_ground_materials.DEFAULT_CONFIG["materials"]`` while walls and
roofs lived in ``thermal_common.DEFAULT_MATERIALS``, and each stage reached into
whichever it happened to import. Two tables for one physical question is how a
surface ends up absorbing as though its albedo were 0.12 while reflecting as
though it were 0.20 -- which this project has already been bitten by once, and
which is a violation of energy conservation rather than a tuning disagreement.

Both of those tables are now DERIVED from this one (see ``legacy_material_table``
and its callers), so the old import paths keep working and keep returning the
same numbers.

WHAT A RECORD HOLDS
-------------------
Optical and thermal properties, plus the two things that make them auditable:

* a plausible RANGE for albedo and emissivity. Literature values for "concrete"
  span 0.2-0.4 depending on age, mix and soiling; carrying only the midpoint
  silently discards that. The range is stored so a later sensitivity study can
  use it -- this module does not itself propagate uncertainty.
* a ``source_reference`` naming where the value came from.

CONSISTENCY RULES ENFORCED BY ``validate_library``
--------------------------------------------------
1. ``shortwave_absorptivity == 1 - albedo``. Opaque surfaces only; this project
   does not model transmission through facets, so there is no third channel for
   the balance to hide in.
2. ``volumetric_heat_capacity == density * specific_heat``. The 1-D solver in
   05b consumes ``C`` directly, and a record that carries an independent ``C``
   alongside inconsistent density/specific-heat would let the storage term and
   the reported material description disagree.
3. Every nominal value lies inside its own declared range.
4. Emissivity in (0, 1], albedo in [0, 1], evaporative efficiency in [0, 1],
   all thermal properties strictly positive.

ADDING A MATERIAL
-----------------
Add it here, give it a range and a reference, and it becomes available to
classification, radiation, radiosity and the surface energy balance at once.
Do not add material constants to a stage script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

# Categories are coarse groupings used for reporting and for deciding which
# classification rules may produce a material (a facade rule must not be able to
# emit a water class, for instance).
CATEGORY_GROUND = "ground"
CATEGORY_FACADE = "facade"
CATEGORY_ROOF = "roof"
CATEGORY_WATER = "water"
CATEGORY_VEGETATION = "vegetation"

ALL_CATEGORIES = (CATEGORY_GROUND, CATEGORY_FACADE, CATEGORY_ROOF,
                  CATEGORY_WATER, CATEGORY_VEGETATION)


@dataclass(frozen=True)
class Material:
    """One surface material and everything the pipeline needs to know about it.

    ``depth_m``/``n_layers``/``bottom_bc``/``insulation_R_m2K_W`` describe the
    1-D substrate column 05b integrates. ``bottom_bc`` is ``"fixed"`` for ground
    (a deep-soil temperature) and ``"interior"`` for building envelopes (a
    conditioned interior behind insulation).
    """

    name: str
    category: str
    albedo: float
    emissivity: float
    thermal_conductivity_WmK: float
    density_kgm3: float
    specific_heat_JkgK: float
    depth_m: float
    n_layers: int
    bottom_bc: str = "fixed"
    roughness_m: float = 0.01
    evaporative_efficiency: float = 0.0
    insulation_R_m2K_W: float | None = None
    albedo_range: tuple[float, float] = (0.0, 1.0)
    emissivity_range: tuple[float, float] = (0.8, 1.0)
    source_reference: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shortwave_absorptivity(self) -> float:
        """Opaque-surface absorptivity. Never stored independently -- storing it
        would create a second number that could drift from the albedo."""
        return 1.0 - self.albedo

    @property
    def volumetric_heat_capacity_Jm3K(self) -> float:
        return self.density_kgm3 * self.specific_heat_JkgK

    def legacy_entry(self) -> dict[str, Any]:
        """The dict shape the existing solver and config code already expect.

        ``k`` and ``C`` are the names 05b's ClassSolver uses; they are emitted
        here rather than stored separately so they cannot drift from the
        conductivity/density/specific-heat the manifest reports.
        """
        entry: dict[str, Any] = {
            "albedo": self.albedo,
            "emissivity": self.emissivity,
            "thermal_conductivity": self.thermal_conductivity_WmK,
            "density": self.density_kgm3,
            "specific_heat": self.specific_heat_JkgK,
            "k": self.thermal_conductivity_WmK,
            "C": self.volumetric_heat_capacity_Jm3K,
            "depth": self.depth_m,
            "n_layers": self.n_layers,
            "bottom_bc": self.bottom_bc,
            "roughness_m": self.roughness_m,
            "evaporative_efficiency": self.evaporative_efficiency,
        }
        if self.insulation_R_m2K_W is not None:
            entry["insulation_R_m2K_W"] = self.insulation_R_m2K_W
        return entry

    def manifest_entry(self) -> dict[str, Any]:
        """The full auditable record, for the material manifest and QC."""
        return {
            "material_name": self.name,
            "category": self.category,
            "shortwave_albedo": self.albedo,
            "shortwave_albedo_min": self.albedo_range[0],
            "shortwave_albedo_max": self.albedo_range[1],
            "shortwave_absorptivity": self.shortwave_absorptivity,
            "longwave_emissivity": self.emissivity,
            "longwave_emissivity_min": self.emissivity_range[0],
            "longwave_emissivity_max": self.emissivity_range[1],
            "thermal_conductivity_WmK": self.thermal_conductivity_WmK,
            "density_kgm3": self.density_kgm3,
            "specific_heat_JkgK": self.specific_heat_JkgK,
            "volumetric_heat_capacity_Jm3K": self.volumetric_heat_capacity_Jm3K,
            "substrate_depth_m": self.depth_m,
            "substrate_layers": self.n_layers,
            "bottom_boundary": self.bottom_bc,
            "roughness_m": self.roughness_m,
            "evaporative_efficiency": self.evaporative_efficiency,
            "source_reference": self.source_reference,
            "notes": self.notes,
        }


def _m(name, category, albedo, emissivity, k, density, specific_heat,
       depth, n_layers, **kwargs) -> Material:
    return Material(name=name, category=category, albedo=albedo,
                    emissivity=emissivity, thermal_conductivity_WmK=k,
                    density_kgm3=density, specific_heat_JkgK=specific_heat,
                    depth_m=depth, n_layers=n_layers, **kwargs)


# ---------------------------------------------------------------------------
# GROUND
#
# These reproduce the values the project already ran with, to the digit -- the
# density/specific-heat pairs are chosen so their product equals the volumetric
# heat capacity each class already used, so moving the table here changes no
# result. Ranges and references are new.
# ---------------------------------------------------------------------------
_GROUND: tuple[Material, ...] = (
    _m("generic_ground", CATEGORY_GROUND, 0.18, 0.95, 1.00, 2000.0, 1000.0,
       0.50, 8, roughness_m=0.03, evaporative_efficiency=0.05,
       albedo_range=(0.10, 0.30), emissivity_range=(0.90, 0.97),
       source_reference="Oke, Boundary Layer Climates, mixed urban ground",
       notes="Explicit named fallback for terrain no rule could classify."),
    # Portuguese calcada: hand-set limestone setts, often with basalt inlay,
    # bedded on sand over a compacted base. It is the dominant pedestrian
    # surface in historic Lisbon, and differs from the concrete/asphalt classes
    # that OSM defaults assign to a footway in two ways that matter thermally:
    # a much higher albedo (the limestone is near-white) and a markedly higher
    # thermal admittance sqrt(k*rho*c) ~ 2200 against ~1250-1680, which damps
    # the diurnal surface-temperature swing.
    _m("calcada_limestone", CATEGORY_GROUND, 0.38, 0.94, 2.10, 2600.0, 900.0,
       0.50, 8, roughness_m=0.006, evaporative_efficiency=0.0,
       albedo_range=(0.30, 0.50), emissivity_range=(0.90, 0.96),
       source_reference="Limestone thermal properties, Clauser & Huenges (1995); "
                        "albedo of light natural stone paving, Oke (1987)",
       notes="Portuguese calcada; light limestone setts. Use as the regional "
             "pedestrian-surface default for Lisbon rather than concrete."),
    _m("asphalt_road", CATEGORY_GROUND, 0.12, 0.95, 0.75, 2300.0, 920.0,
       0.40, 8, roughness_m=0.002,
       albedo_range=(0.05, 0.20), emissivity_range=(0.90, 0.98),
       source_reference="Oke (1987); ASHRAE Fundamentals asphalt paving",
       notes="Fresh asphalt near 0.05, weathered near 0.20."),
    _m("concrete_road", CATEGORY_GROUND, 0.28, 0.94, 1.40, 2300.0, 880.0,
       0.40, 8, roughness_m=0.003,
       albedo_range=(0.17, 0.40), emissivity_range=(0.88, 0.97),
       source_reference="ASHRAE Fundamentals; Taha (1997) urban albedo"),
    _m("asphalt_pedestrian", CATEGORY_GROUND, 0.14, 0.95, 0.75, 2250.0, 920.0,
       0.30, 7, roughness_m=0.002,
       albedo_range=(0.08, 0.22), emissivity_range=(0.90, 0.98),
       source_reference="Oke (1987), asphalt footway"),
    _m("concrete_pedestrian", CATEGORY_GROUND, 0.30, 0.94, 1.40, 2300.0, 880.0,
       0.25, 7, roughness_m=0.003,
       albedo_range=(0.17, 0.40), emissivity_range=(0.88, 0.97),
       source_reference="ASHRAE Fundamentals, concrete slab"),
    # Thermal admittance mu = sqrt(k*C*omega) = 17.0 W/m2/K, raised from the
    # 12.2 these classes previously carried. Evidence, in order of weight:
    #
    #  * LITERATURE. Lisbon paving is calcada limestone sett. Solid limestone is
    #    k ~ 2.0-2.2, mu ~ 18-19. The value here sits just below that because
    #    setts are bedded on sand, so the composite over one damping depth is
    #    less conductive than the stone alone.
    #  * MEASURED DIURNAL RANGE. Inverting the surface excess over air through
    #    the harmonic surface balance gives mu ~ 20 (k ~ 3.0) on lisbon4, the
    #    only one of the six campaigns whose day/night pair is internally
    #    consistent. Two independent routes landing near mu ~ 18-20 is the
    #    reason for moving at all. The other five cases demand k = 24-48, which
    #    is metal -- their day and night walks are 19-66 days apart with
    #    different antecedent storage, and their measured LW_up cannot close its
    #    own energy budget, so they constrain nothing.
    #  * NOT a fit. lisbon4's 3.04 is deliberately NOT adopted: one case, and
    #    the case whose model side the free-stream convection change slightly
    #    over-corrected.
    #
    # Substrate deepened to 0.60 m because the damping depth rises with k:
    # sqrt(2k/(C*omega)) = 0.16 m here, so 0.25 m would be only 1.6 damping
    # depths above a fixed-temperature boundary, close enough to anchor the
    # diurnal wave artificially.
    #
    # KNOWN TRADE-OFF: raising admittance cools the day and WARMS the night.
    # Night longwave-up was already running about 12 W/m2 cold, so this improves
    # the daytime heat-risk answer the project exists to produce and costs a
    # little at night. That is a deliberate, recorded choice, not an oversight.
    _m("paving_stone_pedestrian", CATEGORY_GROUND, 0.24, 0.94, 1.90, 2400.0,
       875.0, 0.60, 9, roughness_m=0.006,
       albedo_range=(0.12, 0.45), emissivity_range=(0.88, 0.96),
       source_reference=("ASHRAE Fundamentals limestone; Doulos et al. (2004) "
                         "pavers; admittance supported by the lisbon4 diurnal "
                         "surface-excess inversion (mu ~ 20)"),
       notes=("Wide albedo range: light limestone setts sit far above dark "
              "granite. Effective conductivity is that of the sett-plus-bed "
              "composite, not of solid stone, which is why k sits below the "
              "2.0-2.2 of solid limestone.")),
    _m("pedestrian_plaza", CATEGORY_GROUND, 0.24, 0.94, 1.90, 2400.0, 875.0,
       0.60, 9, roughness_m=0.006,
       albedo_range=(0.12, 0.45), emissivity_range=(0.88, 0.96),
       source_reference="As paving_stone_pedestrian"),
    _m("pedestrian_crossing", CATEGORY_GROUND, 0.26, 0.94, 1.20, 2250.0, 880.0,
       0.25, 7, roughness_m=0.004,
       albedo_range=(0.15, 0.40), emissivity_range=(0.88, 0.97),
       source_reference="Painted asphalt, area-weighted paint and substrate",
       notes="Higher albedo than plain asphalt because of the white marking."),
    _m("asphalt_parking", CATEGORY_GROUND, 0.12, 0.95, 0.75, 2300.0, 920.0,
       0.40, 8, roughness_m=0.002,
       albedo_range=(0.05, 0.20), emissivity_range=(0.90, 0.98),
       source_reference="As asphalt_road"),
    _m("concrete_parking", CATEGORY_GROUND, 0.28, 0.94, 1.40, 2300.0, 880.0,
       0.40, 8, roughness_m=0.003,
       albedo_range=(0.17, 0.40), emissivity_range=(0.88, 0.97),
       source_reference="As concrete_road"),
    _m("gravel_parking", CATEGORY_GROUND, 0.22, 0.95, 0.80, 1800.0, 1000.0,
       0.40, 8, roughness_m=0.015,
       albedo_range=(0.12, 0.35), emissivity_range=(0.90, 0.97),
       source_reference="Oke (1987), gravel"),
    _m("grass_lawn", CATEGORY_GROUND, 0.23, 0.96, 0.60, 1820.0, 1000.0,
       0.50, 8, roughness_m=0.03, evaporative_efficiency=0.70,
       albedo_range=(0.16, 0.27), emissivity_range=(0.93, 0.99),
       source_reference="Oke (1987) short grass; Campbell & Norman (1998)",
       notes="Evaporative efficiency is the dominant control, not albedo."),
    _m("artificial_turf", CATEGORY_GROUND, 0.18, 0.95, 0.30, 1400.0, 1100.0,
       0.15, 6, roughness_m=0.012,
       albedo_range=(0.08, 0.25), emissivity_range=(0.90, 0.97),
       source_reference="Synthetic turf, manufacturer data",
       notes="Dry by construction; no evaporative cooling, unlike real grass."),
    _m("sports_surface", CATEGORY_GROUND, 0.20, 0.95, 0.50, 1600.0, 1100.0,
       0.20, 6, roughness_m=0.008,
       albedo_range=(0.10, 0.35), emissivity_range=(0.90, 0.97),
       source_reference="Generic bound sports surface"),
    _m("playground_surface", CATEGORY_GROUND, 0.20, 0.95, 0.35, 1300.0, 1200.0,
       0.15, 6, roughness_m=0.01,
       albedo_range=(0.08, 0.35), emissivity_range=(0.90, 0.97),
       source_reference="Bonded rubber safety surfacing"),
    _m("bare_ground", CATEGORY_GROUND, 0.20, 0.95, 0.80, 1800.0, 1000.0,
       0.50, 8, roughness_m=0.02, evaporative_efficiency=0.20,
       albedo_range=(0.10, 0.35), emissivity_range=(0.90, 0.97),
       source_reference="Oke (1987) dry bare soil"),
    _m("unpaved_path", CATEGORY_GROUND, 0.20, 0.95, 0.80, 1800.0, 1000.0,
       0.40, 8, roughness_m=0.015, evaporative_efficiency=0.20,
       albedo_range=(0.10, 0.35), emissivity_range=(0.90, 0.97),
       source_reference="Compacted unpaved track"),
    # Legacy aliases kept so older configs and manifests still resolve.
    _m("asphalt_pedestrian_path", CATEGORY_GROUND, 0.14, 0.95, 0.75, 2250.0,
       920.0, 0.30, 7, roughness_m=0.002,
       albedo_range=(0.08, 0.22), emissivity_range=(0.90, 0.98),
       source_reference="Legacy name for asphalt_pedestrian"),
    _m("concrete_sidewalk", CATEGORY_GROUND, 0.30, 0.94, 1.40, 2300.0, 880.0,
       0.25, 7, roughness_m=0.003,
       albedo_range=(0.17, 0.40), emissivity_range=(0.88, 0.97),
       source_reference="Legacy name for concrete_pedestrian"),
    _m("paving_stone_path", CATEGORY_GROUND, 0.24, 0.94, 1.90, 2400.0, 875.0,
       0.60, 9, roughness_m=0.006,
       albedo_range=(0.12, 0.45), emissivity_range=(0.88, 0.96),
       source_reference="Legacy name for paving_stone_pedestrian"),
    # The uniform-ground class used when the OSM surface branch is disabled.
    # Identical to generic_ground; kept under its historical name because
    # existing thermal folders and manifests refer to it.
    _m("ground", CATEGORY_GROUND, 0.18, 0.95, 1.00, 2000.0, 1000.0, 0.50, 8,
       roughness_m=0.03, evaporative_efficiency=0.05,
       albedo_range=(0.10, 0.30), emissivity_range=(0.90, 0.97),
       source_reference="Legacy uniform-ground class",
       notes="Backward-compatible uniform ground; same physics as generic_ground."),
)

# ---------------------------------------------------------------------------
# WATER
# ---------------------------------------------------------------------------
_WATER: tuple[Material, ...] = (
    _m("water", CATEGORY_WATER, 0.08, 0.98, 0.60, 1000.0, 4171.64, 1.0, 10,
       roughness_m=0.0002, evaporative_efficiency=1.00,
       albedo_range=(0.03, 0.20), emissivity_range=(0.95, 0.99),
       source_reference="Oke (1987); open water, sun-angle dependent albedo",
       notes=("Albedo rises steeply at low solar elevation; the nominal is a "
              "midday value. Modelled as an opaque slab with a large heat "
              "capacity, not as a transmitting medium.")),
)

# ---------------------------------------------------------------------------
# BUILDING FACADES
#
# ``wall`` is retained as the generic default under its historical name so
# existing thermal folders keep resolving; ``generic_building_facade`` is its
# alias for new, explicitly-named classification output. Both carry the exact
# values the project already used.
# ---------------------------------------------------------------------------
_FACADE: tuple[Material, ...] = (
    _m("wall", CATEGORY_FACADE, 0.30, 0.90, 1.40, 1800.0, 1000.0, 0.25, 6,
       bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.005,
       albedo_range=(0.15, 0.50), emissivity_range=(0.85, 0.95),
       source_reference="Project default mixed urban facade",
       notes="Historical generic facade name; kept for backward compatibility."),
    _m("generic_building_facade", CATEGORY_FACADE, 0.30, 0.90, 1.40, 1800.0,
       1000.0, 0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5,
       roughness_m=0.005,
       albedo_range=(0.15, 0.50), emissivity_range=(0.85, 0.95),
       source_reference="Project default mixed urban facade",
       notes=("Explicit named default for a building with no usable material "
              "evidence. Same physics as 'wall'.")),
    _m("rendered_facade", CATEGORY_FACADE, 0.35, 0.91, 0.90, 1600.0, 1000.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.003,
       albedo_range=(0.20, 0.60), emissivity_range=(0.87, 0.95),
       source_reference="Painted render/stucco, ASHRAE Fundamentals"),
    _m("plaster_facade", CATEGORY_FACADE, 0.35, 0.91, 0.90, 1600.0, 1000.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.003,
       albedo_range=(0.20, 0.60), emissivity_range=(0.87, 0.95),
       source_reference="As rendered_facade"),
    _m("brick_facade", CATEGORY_FACADE, 0.28, 0.93, 0.85, 1900.0, 840.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.006,
       albedo_range=(0.15, 0.40), emissivity_range=(0.90, 0.96),
       source_reference="ASHRAE Fundamentals, fired clay brick"),
    _m("stone_facade", CATEGORY_FACADE, 0.32, 0.92, 2.00, 2400.0, 900.0,
       0.30, 7, bottom_bc="interior", insulation_R_m2K_W=1.0, roughness_m=0.006,
       albedo_range=(0.15, 0.50), emissivity_range=(0.88, 0.95),
       source_reference="ASHRAE Fundamentals, limestone/granite masonry",
       notes="Heavier and more conductive than render; less insulated behind."),
    _m("concrete_facade", CATEGORY_FACADE, 0.25, 0.92, 1.40, 2300.0, 880.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.004,
       albedo_range=(0.15, 0.40), emissivity_range=(0.88, 0.96),
       source_reference="ASHRAE Fundamentals, precast concrete panel"),
    _m("metal_facade", CATEGORY_FACADE, 0.45, 0.35, 45.0, 7800.0, 480.0,
       0.02, 4, bottom_bc="interior", insulation_R_m2K_W=2.0, roughness_m=0.001,
       albedo_range=(0.25, 0.65), emissivity_range=(0.10, 0.60),
       source_reference="Bare/coated sheet metal cladding",
       notes=("LOW EMISSIVITY is the defining property of bare metal and the "
              "reason it must not be lumped with masonry: it emits far less "
              "longwave at the same temperature.")),
    _m("glass_facade", CATEGORY_FACADE, 0.20, 0.88, 1.00, 2500.0, 840.0,
       0.02, 4, bottom_bc="interior", insulation_R_m2K_W=0.8, roughness_m=0.0005,
       albedo_range=(0.10, 0.45), emissivity_range=(0.83, 0.94),
       source_reference="Architectural glazing, normal-incidence reflectance",
       notes=("Treated as an OPAQUE surface with an effective albedo. This "
              "project does not model transmission, so a glazed facade cannot "
              "pass shortwave into the building interior.")),
    _m("wood_facade", CATEGORY_FACADE, 0.30, 0.90, 0.16, 600.0, 1600.0,
       0.10, 5, bottom_bc="interior", insulation_R_m2K_W=1.8, roughness_m=0.004,
       albedo_range=(0.15, 0.45), emissivity_range=(0.85, 0.95),
       source_reference="ASHRAE Fundamentals, softwood cladding"),
    _m("light_facade", CATEGORY_FACADE, 0.45, 0.90, 1.20, 1800.0, 1000.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.004,
       albedo_range=(0.35, 0.60), emissivity_range=(0.85, 0.95),
       source_reference="Colour-hint class: light-coloured facade",
       notes=("Reached from a COLOUR tag only. Colour constrains albedo, not "
              "material identity, so the thermal properties stay generic.")),
    _m("dark_facade", CATEGORY_FACADE, 0.18, 0.90, 1.20, 1800.0, 1000.0,
       0.25, 6, bottom_bc="interior", insulation_R_m2K_W=1.5, roughness_m=0.004,
       albedo_range=(0.08, 0.28), emissivity_range=(0.85, 0.95),
       source_reference="Colour-hint class: dark-coloured facade",
       notes="Reached from a COLOUR tag only; see light_facade."),
)

# ---------------------------------------------------------------------------
# ROOFS
# ---------------------------------------------------------------------------
_ROOF: tuple[Material, ...] = (
    _m("roof", CATEGORY_ROOF, 0.15, 0.92, 1.00, 1600.0, 1000.0, 0.25, 6,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.005,
       albedo_range=(0.08, 0.35), emissivity_range=(0.88, 0.96),
       source_reference="Project default mixed urban roof",
       notes="Historical generic roof name; kept for backward compatibility."),
    _m("generic_roof", CATEGORY_ROOF, 0.15, 0.92, 1.00, 1600.0, 1000.0, 0.25, 6,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.005,
       albedo_range=(0.08, 0.35), emissivity_range=(0.88, 0.96),
       source_reference="Project default mixed urban roof",
       notes=("Explicit named default for a roof with no usable material "
              "evidence. Same physics as 'roof'.")),
    _m("tile_roof", CATEGORY_ROOF, 0.20, 0.93, 0.85, 1900.0, 840.0, 0.20, 6,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.01,
       albedo_range=(0.10, 0.40), emissivity_range=(0.88, 0.96),
       source_reference="Clay/concrete roof tiles, ASHRAE Fundamentals",
       notes="Terracotta sits low in the range; light concrete tile higher."),
    _m("metal_roof", CATEGORY_ROOF, 0.45, 0.35, 45.0, 7800.0, 480.0, 0.02, 4,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.001,
       albedo_range=(0.25, 0.70), emissivity_range=(0.10, 0.60),
       source_reference="Sheet-metal roofing",
       notes=("Low emissivity plus negligible thermal mass: a metal roof both "
              "runs hot in sun and cools fast, and radiates less than masonry "
              "does at the same temperature.")),
    _m("membrane_roof", CATEGORY_ROOF, 0.15, 0.92, 0.20, 1100.0, 1500.0,
       0.05, 4, bottom_bc="interior", insulation_R_m2K_W=2.8, roughness_m=0.002,
       albedo_range=(0.05, 0.30), emissivity_range=(0.88, 0.96),
       source_reference="Bitumen/single-ply flat-roof membrane"),
    _m("concrete_roof", CATEGORY_ROOF, 0.25, 0.92, 1.40, 2300.0, 880.0,
       0.20, 6, bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.004,
       albedo_range=(0.15, 0.40), emissivity_range=(0.88, 0.96),
       source_reference="ASHRAE Fundamentals, concrete deck roof"),
    _m("gravel_roof", CATEGORY_ROOF, 0.20, 0.93, 0.80, 1800.0, 1000.0,
       0.12, 5, bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.015,
       albedo_range=(0.12, 0.32), emissivity_range=(0.90, 0.96),
       source_reference="Ballasted flat roof"),
    _m("green_roof", CATEGORY_ROOF, 0.22, 0.96, 0.60, 1500.0, 1200.0,
       0.25, 7, bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.03,
       evaporative_efficiency=0.60,
       albedo_range=(0.15, 0.28), emissivity_range=(0.93, 0.99),
       source_reference="Extensive green roof, sedum substrate",
       notes="Evaporative cooling is the dominant term, as for grass_lawn."),
    _m("glass_roof", CATEGORY_ROOF, 0.20, 0.88, 1.00, 2500.0, 840.0, 0.02, 4,
       bottom_bc="interior", insulation_R_m2K_W=0.8, roughness_m=0.0005,
       albedo_range=(0.10, 0.45), emissivity_range=(0.83, 0.94),
       source_reference="Glazed roof/atrium; opaque approximation as glass_facade"),
    _m("light_roof", CATEGORY_ROOF, 0.55, 0.92, 1.00, 1600.0, 1000.0, 0.25, 6,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.005,
       albedo_range=(0.40, 0.75), emissivity_range=(0.88, 0.96),
       source_reference="Colour-hint class: white/light roof (cool roof)",
       notes=("Reached from a COLOUR tag only. Colour constrains albedo, not "
              "material identity, so thermal properties stay generic.")),
    _m("dark_roof", CATEGORY_ROOF, 0.10, 0.92, 1.00, 1600.0, 1000.0, 0.25, 6,
       bottom_bc="interior", insulation_R_m2K_W=2.5, roughness_m=0.005,
       albedo_range=(0.04, 0.20), emissivity_range=(0.88, 0.96),
       source_reference="Colour-hint class: dark roof",
       notes="Reached from a COLOUR tag only; see light_roof."),
)

# ---------------------------------------------------------------------------
# VEGETATION
#
# Canopy is NOT a facet material in this pipeline: 05/05a treat vegetation as a
# transmitting medium with a leaf-area-density extinction coefficient, and its
# longwave contribution is taken at air temperature. The entry exists so that
# classification can name vegetation explicitly and refuse to fall through to a
# pavement class, and so the manifest can report it -- not so that leaves get a
# 1-D conduction column.
# ---------------------------------------------------------------------------
_VEGETATION: tuple[Material, ...] = (
    _m("tree_canopy", CATEGORY_VEGETATION, 0.18, 0.97, 0.30, 700.0, 2500.0,
       0.10, 4, roughness_m=0.10, evaporative_efficiency=0.80,
       albedo_range=(0.10, 0.25), emissivity_range=(0.94, 0.99),
       source_reference="Oke (1987) deciduous canopy; Campbell & Norman (1998)",
       notes=("Handled by the canopy transmission model, NOT by the facet "
              "energy balance. Listed so vegetation is never silently "
              "classified as ground.")),
)


MATERIAL_LIBRARY: dict[str, Material] = {
    material.name: material
    for material in (_GROUND + _WATER + _FACADE + _ROOF + _VEGETATION)
}

# Classes that mean "we could not identify this surface". They exist so that an
# unclassifiable surface still gets a NAMED material with recorded provenance
# rather than silently inheriting arbitrary numbers.
DEFAULT_MATERIAL_BY_CATEGORY: dict[str, str] = {
    CATEGORY_GROUND: "generic_ground",
    CATEGORY_FACADE: "generic_building_facade",
    CATEGORY_ROOF: "generic_roof",
    CATEGORY_WATER: "water",
    CATEGORY_VEGETATION: "tree_canopy",
}

# Historical names that the rest of the pipeline still writes into facets.npz
# when no classification has run. Mapped so the library can answer for them.
LEGACY_CATEGORY_DEFAULTS: dict[str, str] = {
    CATEGORY_GROUND: "ground",
    CATEGORY_FACADE: "wall",
    CATEGORY_ROOF: "roof",
}


class MaterialLibraryError(ValueError):
    """Raised when the library or a lookup is internally inconsistent."""


def get(name: str) -> Material:
    """Resolve one material by name, failing loudly on an unknown class.

    Deliberately not forgiving. A typo that silently returned a default would
    hand a surface arbitrary optical properties with no trace in the manifest,
    which is precisely the failure mode this module exists to prevent.
    """
    try:
        return MATERIAL_LIBRARY[name]
    except KeyError:
        raise MaterialLibraryError(
            f"unknown material {name!r}; known materials: "
            f"{sorted(MATERIAL_LIBRARY)}") from None


def names_in_category(category: str) -> list[str]:
    if category not in ALL_CATEGORIES:
        raise MaterialLibraryError(f"unknown category {category!r}")
    return sorted(name for name, material in MATERIAL_LIBRARY.items()
                  if material.category == category)


def albedo(name: str) -> float:
    return get(name).albedo


def emissivity(name: str) -> float:
    return get(name).emissivity


def shortwave_absorptivity(name: str) -> float:
    return get(name).shortwave_absorptivity


def legacy_material_table(categories: Iterable[str] | None = None
                          ) -> dict[str, dict[str, Any]]:
    """The legacy ``{name: {albedo, emissivity, k, C, ...}}`` mapping.

    This is what ``osm_ground_materials.DEFAULT_CONFIG['materials']`` and
    ``thermal_common.DEFAULT_MATERIALS`` are now built from, so there is exactly
    one place where a number lives even though two historical import paths
    still resolve.
    """
    wanted = set(categories) if categories is not None else set(ALL_CATEGORIES)
    return {name: material.legacy_entry()
            for name, material in MATERIAL_LIBRARY.items()
            if material.category in wanted}


def validate_library(library: dict[str, Material] | None = None) -> dict[str, Any]:
    """Enforce the consistency rules documented at the top of this module."""
    library = MATERIAL_LIBRARY if library is None else library
    problems: list[str] = []
    for name, material in library.items():
        if material.name != name:
            problems.append(f"{name}: keyed under a different name than it carries")
        if material.category not in ALL_CATEGORIES:
            problems.append(f"{name}: unknown category {material.category!r}")
        if not 0.0 <= material.albedo <= 1.0:
            problems.append(f"{name}: albedo {material.albedo} outside [0, 1]")
        if not 0.0 < material.emissivity <= 1.0:
            problems.append(f"{name}: emissivity {material.emissivity} outside (0, 1]")
        if not 0.0 <= material.evaporative_efficiency <= 1.0:
            problems.append(f"{name}: evaporative efficiency outside [0, 1]")
        for label, value in (
                ("thermal conductivity", material.thermal_conductivity_WmK),
                ("density", material.density_kgm3),
                ("specific heat", material.specific_heat_JkgK),
                ("substrate depth", material.depth_m),
                ("roughness", material.roughness_m)):
            if value <= 0.0:
                problems.append(f"{name}: {label} must be positive, got {value}")
        if material.n_layers < 2:
            problems.append(f"{name}: needs at least 2 substrate layers")
        # Rule 1: opaque-surface energy conservation.
        if abs(material.shortwave_absorptivity - (1.0 - material.albedo)) > 1e-12:
            problems.append(f"{name}: absorptivity does not equal 1 - albedo")
        # Rule 2: the solver's C must be the density-specific-heat product.
        legacy = material.legacy_entry()
        product = material.density_kgm3 * material.specific_heat_JkgK
        if abs(legacy["C"] - product) > 1e-6 * max(product, 1.0):
            problems.append(
                f"{name}: volumetric heat capacity {legacy['C']} disagrees with "
                f"density*specific_heat {product}")
        if abs(legacy["k"] - material.thermal_conductivity_WmK) > 1e-12:
            problems.append(f"{name}: solver k disagrees with conductivity")
        # Rule 3: nominal inside its own range.
        low, high = material.albedo_range
        if not low <= material.albedo <= high:
            problems.append(
                f"{name}: albedo {material.albedo} outside its range {low}-{high}")
        if low < 0.0 or high > 1.0 or low > high:
            problems.append(f"{name}: albedo range {low}-{high} is not a valid interval")
        low, high = material.emissivity_range
        if not low <= material.emissivity <= high:
            problems.append(
                f"{name}: emissivity {material.emissivity} outside its range "
                f"{low}-{high}")
        if low <= 0.0 or high > 1.0 or low > high:
            problems.append(f"{name}: emissivity range {low}-{high} is not valid")
        if material.bottom_bc not in {"fixed", "interior"}:
            problems.append(f"{name}: unknown bottom boundary {material.bottom_bc!r}")
        if material.bottom_bc == "interior" and material.insulation_R_m2K_W is None:
            problems.append(f"{name}: interior boundary needs an insulation R")
        if not material.source_reference:
            problems.append(f"{name}: missing source_reference")
    for category, default_name in DEFAULT_MATERIAL_BY_CATEGORY.items():
        if default_name not in library:
            problems.append(f"category {category}: default {default_name!r} missing")
        elif library[default_name].category != category:
            problems.append(
                f"category {category}: default {default_name!r} is not in it")
    if problems:
        raise MaterialLibraryError(
            "material library is inconsistent:\n  " + "\n  ".join(problems))
    return {
        "n_materials": len(library),
        "by_category": {category: len(names_in_category(category))
                        for category in ALL_CATEGORIES},
    }


def with_overrides(material_name: str, **overrides: Any) -> Material:
    """A copy of one material with fields replaced, for sensitivity work.

    The lookup argument is ``material_name`` rather than ``name`` so that a
    caller can also override the record's own ``name`` field without the two
    colliding.
    """
    return replace(get(material_name), **overrides)


def manifest_records() -> list[dict[str, Any]]:
    """Every material as a flat auditable record, sorted for stable output."""
    return [MATERIAL_LIBRARY[name].manifest_entry()
            for name in sorted(MATERIAL_LIBRARY)]


# Fail at import time rather than mid-run: an inconsistent library would
# otherwise only surface as a strange surface temperature hours into a solve.
validate_library()


if __name__ == "__main__":  # pragma: no cover - convenience listing
    summary = validate_library()
    print(f"material library: {summary['n_materials']} materials")
    for category in ALL_CATEGORIES:
        names = names_in_category(category)
        print(f"\n{category} ({len(names)}):")
        for name in names:
            material = get(name)
            print(f"  {name:26s} albedo {material.albedo:5.2f} "
                  f"[{material.albedo_range[0]:.2f}-{material.albedo_range[1]:.2f}]  "
                  f"eps {material.emissivity:5.2f}  "
                  f"k {material.thermal_conductivity_WmK:6.2f}  "
                  f"C {material.volumetric_heat_capacity_Jm3K:9.3e}")
