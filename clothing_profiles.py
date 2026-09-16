"""
clothing_profiles.py -- clothing ensembles for the JOS-3 route thermal-strain
stage (09), selectable manually or chosen automatically from climate, time of
day and weather.

WHY THIS EXISTS
---------------
JOS-3 exposes a PER-SEGMENT clothing insulation array (``model.clo``, one
value per body segment) and defaults every segment to 0 clo -- an unclothed
subject.  Stage 09 previously left that default in place, so every route was
walked naked.  That is not a neutral assumption: on a cool, windy night it
exaggerates extremity cooling and can even drive whole-body core temperature
DOWN while walking, and in strong sun it removes the shading that real
clothing gives the skin.

WHAT IS MODELLED (and what is not)
----------------------------------
Each ensemble below is a static per-segment insulation distribution.  JOS-3
applies it as dry thermal resistance between skin and environment.  This
module does NOT model:

  * evaporative resistance of clothing (moisture permeability, i_m / R_et)
    separately -- JOS-3 derives its own vapour resistance from clo;
  * wind- or motion-driven reduction of clothing insulation (pumping);
  * wet clothing after rain, or solar transmission through fabric;
  * behavioural change DURING the walk -- one ensemble is fixed per route,
    which is the realistic case for a 20-60 minute walk.

Anyone needing those effects should treat these values as the dry-insulation
starting point, not as a complete clothing physics model.

VALUES AND THEIR STATUS -- READ THIS BEFORE CITING
---------------------------------------------------
The whole-body totals are aimed at the familiar garment-ensemble magnitudes
used in comfort standards (ASHRAE 55 / ISO 9920 style): walking shorts plus
short-sleeve shirt ~0.36 clo, trousers plus short-sleeve shirt ~0.57 clo,
trousers plus long-sleeve shirt ~0.61 clo, adding a jacket ~1.0 clo, heavy
winter dress ~1.5-2 clo.

The PER-SEGMENT split within each ensemble, and the temperature thresholds
used by the automatic selector, are TRANSPARENT PROJECT CONVENTIONS -- they
are engineering estimates chosen so that covered segments carry the garment's
insulation and bare segments (head, hands, often lower legs in summer) carry
none.  They are NOT copied coefficients from a specific published table, and
this module deliberately does not claim otherwise.

The CONCEPT of choosing clothing from outdoor temperature is standard in
outdoor thermal-comfort work: UTCI itself is defined with a temperature-
adaptive clothing model (Havenith et al. 2012, Int J Biometeorol), and
outdoor-temperature clothing regressions are used in adaptive comfort
practice (e.g. Schiavon & Lee 2013).  This module follows that concept with
its own explicit, overridable table rather than reproducing either model.

Every number here can be replaced from a case without touching code, via
``--clothing-config`` (see ``load_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

# JOS-3 segment order (pythermalcomfort JOS3.body_names).
JOS3_SEGMENTS = (
    "head", "neck", "chest", "back", "pelvis",
    "left_shoulder", "left_arm", "left_hand",
    "right_shoulder", "right_arm", "right_hand",
    "left_thigh", "left_leg", "left_foot",
    "right_thigh", "right_leg", "right_foot",
)

# Body-part groups -> the JOS-3 segments they cover.  Ensembles are written
# against these groups so a garment description stays readable.
_GROUPS = {
    "head": ("head",),
    "neck": ("neck",),
    "torso": ("chest", "back", "pelvis"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "arm": ("left_arm", "right_arm"),
    "hand": ("left_hand", "right_hand"),
    "thigh": ("left_thigh", "right_thigh"),
    "leg": ("left_leg", "right_leg"),
    "foot": ("left_foot", "right_foot"),
}


# JOS-3 segment body-surface-area fractions, in JOS3_SEGMENTS order. These are
# fixed ratios in the model (identical for every height/weight/age/sex, checked
# against male/female/child subjects), so they can be used to normalise an
# ensemble deterministically without constructing a subject.
SEGMENT_BSA_FRACTION = np.array([
    0.05889, 0.01552, 0.09368, 0.08619, 0.11831,
    0.05139, 0.03373, 0.02677,
    0.05139, 0.03373, 0.02677,
    0.11188, 0.05996, 0.02998,
    0.11188, 0.05996, 0.02998,
])


@dataclass(frozen=True)
class ClothingEnsemble:
    """One wearable outfit expressed as per-body-part dry insulation (clo).

    ``parts`` sets the SHAPE of the distribution (which segments the garments
    actually cover, and in what proportion); ``approximate_total_clo`` sets its
    MAGNITUDE. ``segment_clo`` rescales the shape so the area-weighted whole-
    body insulation equals that standard ensemble value -- bare head and hands
    would otherwise drag the whole-body mean ~15% below the familiar
    comfort-standard number for the same outfit.
    """

    key: str
    label: str
    parts: dict           # body-part group -> relative local clo
    description: str
    approximate_total_clo: float   # whole-body value the ensemble is anchored to

    def raw_segment_clo(self) -> np.ndarray:
        """Expand ``parts`` to the 17-element JOS-3 array, without scaling."""
        values = {}
        for group, clo in self.parts.items():
            if group not in _GROUPS:
                raise ValueError(
                    f"clothing ensemble '{self.key}' names unknown body part "
                    f"'{group}'. Valid: {', '.join(sorted(_GROUPS))}")
            if not np.isfinite(clo) or clo < 0:
                raise ValueError(
                    f"clothing ensemble '{self.key}' part '{group}' has an "
                    f"invalid clo value {clo!r}")
            for segment in _GROUPS[group]:
                values[segment] = float(clo)
        return np.array([values.get(name, 0.0) for name in JOS3_SEGMENTS],
                        dtype=float)

    def segment_clo(self) -> np.ndarray:
        """Per-segment clo anchored to the ensemble's whole-body total."""
        raw = self.raw_segment_clo()
        raw_total = float(np.sum(raw * SEGMENT_BSA_FRACTION))
        target = float(self.approximate_total_clo)
        if raw_total <= 0 or not np.isfinite(target) or target <= 0:
            return raw            # 'nude', or an ensemble with no declared total
        return raw * (target / raw_total)


# ---------------------------------------------------------------------------
# Ensembles, coolest-wearing (least clothing) first.
# ---------------------------------------------------------------------------
ENSEMBLES = {
    "nude": ClothingEnsemble(
        key="nude", label="Unclothed (JOS-3 default)",
        parts={},
        description="No clothing on any segment. This is JOS-3's own default "
                    "and reproduces stage-09 results from before clothing was "
                    "modelled. Keep for backward comparison only -- it is not "
                    "a realistic pedestrian.",
        approximate_total_clo=0.0),

    "beachwear": ClothingEnsemble(
        key="beachwear", label="Beachwear (swimwear + sandals)",
        parts={"torso": 0.08, "thigh": 0.12, "foot": 0.12},
        description="Minimal cover for very hot conditions or waterfront "
                    "settings; head, arms and lower legs fully exposed.",
        approximate_total_clo=0.10),

    "hot_minimal": ClothingEnsemble(
        key="hot_minimal", label="Very hot (singlet, shorts, sandals)",
        parts={"torso": 0.35, "shoulder": 0.10, "thigh": 0.35, "foot": 0.20},
        description="Sleeveless top and shorts. Shoulders barely covered, "
                    "arms and lower legs bare.",
        approximate_total_clo=0.25),

    "summer_light": ClothingEnsemble(
        key="summer_light", label="Summer light (T-shirt, shorts, trainers)",
        parts={"torso": 0.50, "shoulder": 0.45, "arm": 0.12,
               "thigh": 0.40, "leg": 0.02, "foot": 0.30},
        description="The familiar warm-weather walking outfit; short sleeves "
                    "cover the shoulder and upper arm only.",
        approximate_total_clo=0.36),

    "summer_trousers": ClothingEnsemble(
        key="summer_trousers", label="Warm (short sleeves + long trousers)",
        parts={"torso": 0.55, "shoulder": 0.50, "arm": 0.15,
               "thigh": 0.70, "leg": 0.60, "foot": 0.35},
        description="Short-sleeve shirt with full-length trousers and closed "
                    "shoes; typical summer workwear or campus dress.",
        approximate_total_clo=0.55),

    "mild_longsleeve": ClothingEnsemble(
        key="mild_longsleeve", label="Mild (long sleeves + trousers)",
        parts={"torso": 0.62, "shoulder": 0.60, "arm": 0.50,
               "thigh": 0.72, "leg": 0.65, "foot": 0.38},
        description="Long-sleeve shirt and trousers; arms now insulated, "
                    "head and hands still bare.",
        approximate_total_clo=0.61),

    "cool_layer": ClothingEnsemble(
        key="cool_layer", label="Cool (sweater or light jacket)",
        parts={"neck": 0.20, "torso": 1.00, "shoulder": 0.95, "arm": 0.80,
               "thigh": 0.80, "leg": 0.70, "foot": 0.42},
        description="An added mid layer over shirt and trousers -- the usual "
                    "evening or shoulder-season outfit.",
        approximate_total_clo=0.90),

    "cold_jacket": ClothingEnsemble(
        key="cold_jacket", label="Cold (insulated jacket, hat, light gloves)",
        parts={"head": 0.25, "neck": 0.50, "torso": 1.40, "shoulder": 1.30,
               "arm": 1.00, "hand": 0.20, "thigh": 0.90, "leg": 0.85,
               "foot": 0.50},
        description="Winter coat with hat and thin gloves; head, neck and "
                    "hands are protected, which strongly changes extremity "
                    "temperatures relative to bare skin.",
        approximate_total_clo=1.20),

    "winter_heavy": ClothingEnsemble(
        key="winter_heavy", label="Severe cold (heavy winter dress)",
        parts={"head": 0.60, "neck": 0.90, "torso": 2.20, "shoulder": 2.00,
               "arm": 1.60, "hand": 0.80, "thigh": 1.40, "leg": 1.30,
               "foot": 0.80},
        description="Heavy coat, insulated gloves and boots for sub-freezing "
                    "walking.",
        approximate_total_clo=1.80),
}


# ---------------------------------------------------------------------------
# Automatic selection: climate + time of day + weather -> ensemble.
# ---------------------------------------------------------------------------
# Thresholds act on a 'dressing temperature': the air temperature the walker
# effectively dresses for. Each entry is (minimum dressing temperature C,
# ensemble key), evaluated warmest-first.
DEFAULT_SELECTION = {
    "thresholds_c": [
        [31.0, "hot_minimal"],
        [26.0, "summer_light"],
        [22.0, "summer_trousers"],
        [17.0, "mild_longsleeve"],
        [11.0, "cool_layer"],
        [4.0, "cold_jacket"],
        [-100.0, "winter_heavy"],
    ],
    # Time of day: people leaving after dark carry an extra layer for the
    # cooler return, so night lowers the dressing temperature.
    "night_offset_c": 1.5,
    # Weather: wind above this speed prompts a windbreak layer.
    "wind_reference_ms": 3.0,
    "wind_offset_c_per_ms": 0.4,
    "maximum_wind_offset_c": 4.0,
    # Climate acclimatization: residents of hot climates dress lighter at the
    # same temperature than residents of cold ones. Positive = dresses lighter.
    "climate_offset_c": {
        "tropical": 2.0,
        "subtropical": 1.0,
        "temperate": 0.0,
        "continental": -1.0,
        "cold": -2.0,
    },
}


def load_config(path: str | Path | None) -> dict:
    """Return the selection configuration, optionally overridden from JSON.

    The JSON may replace any subset of :data:`DEFAULT_SELECTION` and may add
    or redefine ensembles under an ``"ensembles"`` key, each given as
    ``{"label": ..., "parts": {body_part: clo}, "description": ...}``.
    """
    config = json.loads(json.dumps(DEFAULT_SELECTION))
    if not path:
        return config
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in supplied.items():
        if key == "ensembles":
            for name, spec in value.items():
                ENSEMBLES[name] = ClothingEnsemble(
                    key=name, label=spec.get("label", name),
                    parts=spec.get("parts", {}),
                    description=spec.get("description", "case-supplied ensemble"),
                    approximate_total_clo=float(
                        spec.get("approximate_total_clo", float("nan"))))
        elif key == "climate_offset_c" and isinstance(value, dict):
            config["climate_offset_c"].update(value)
        else:
            config[key] = value
    thresholds = config["thresholds_c"]
    if not thresholds or any(len(item) != 2 for item in thresholds):
        raise ValueError("clothing thresholds_c must be [temperature, ensemble] pairs")
    for _t, name in thresholds:
        if name not in ENSEMBLES:
            raise ValueError(f"clothing threshold names unknown ensemble '{name}'")
    if sorted((float(t) for t, _n in thresholds), reverse=True) != \
            [float(t) for t, _n in thresholds]:
        raise ValueError("clothing thresholds_c must be ordered warmest first")
    return config


def dressing_temperature(air_temp_c: float, *, is_daytime: bool,
                         wind_ms: float, climate: str, config: dict
                         ) -> tuple[float, dict]:
    """Air temperature adjusted for time of day, wind and climate habit."""
    if not np.isfinite(air_temp_c):
        raise ValueError("clothing selection needs a finite air temperature")
    if not np.isfinite(wind_ms) or wind_ms < 0:
        raise ValueError("clothing selection needs a finite, non-negative wind speed")
    climate_offsets = config["climate_offset_c"]
    if climate not in climate_offsets:
        raise ValueError(
            f"unknown clothing climate '{climate}'. "
            f"Choices: {', '.join(sorted(climate_offsets))}")
    night = 0.0 if is_daytime else float(config["night_offset_c"])
    wind_excess = max(0.0, float(wind_ms) - float(config["wind_reference_ms"]))
    wind = min(wind_excess * float(config["wind_offset_c_per_ms"]),
               float(config["maximum_wind_offset_c"]))
    climate_offset = float(climate_offsets[climate])
    value = float(air_temp_c) - night - wind + climate_offset
    return value, {"air_temperature_c": float(air_temp_c),
                   "night_offset_c": -night,
                   "wind_offset_c": -wind,
                   "climate_offset_c": climate_offset,
                   "dressing_temperature_c": value}


def select_ensemble(air_temp_c: float, *, is_daytime: bool, wind_ms: float,
                    climate: str = "temperate", config: dict | None = None
                    ) -> tuple[ClothingEnsemble, dict]:
    """Choose an ensemble from climate, time of day and weather.

    Returns the ensemble and a provenance dict recording every term that led
    to the choice, so a result can always be traced back to its clothing.
    """
    config = config or load_config(None)
    value, terms = dressing_temperature(
        air_temp_c, is_daytime=is_daytime, wind_ms=wind_ms,
        climate=climate, config=config)
    for threshold, name in config["thresholds_c"]:
        if value >= float(threshold):
            chosen = ENSEMBLES[name]
            break
    else:                                   # pragma: no cover - guarded above
        chosen = ENSEMBLES[config["thresholds_c"][-1][1]]
    terms.update({"selection": "auto", "climate": climate,
                  "is_daytime": bool(is_daytime), "wind_ms": float(wind_ms),
                  "ensemble": chosen.key, "ensemble_label": chosen.label})
    return chosen, terms


def resolve(spec: str, *, air_temp_c: float, is_daytime: bool, wind_ms: float,
            climate: str = "temperate", config: dict | None = None
            ) -> tuple[np.ndarray, dict]:
    """Resolve a user clothing request into a per-segment clo array.

    ``spec`` is ``"auto"`` (choose from conditions), a named ensemble, or a
    number for a uniform clo value on every segment.
    """
    config = config or load_config(None)
    text = str(spec).strip().lower()
    if text == "auto":
        ensemble, provenance = select_ensemble(
            air_temp_c, is_daytime=is_daytime, wind_ms=wind_ms,
            climate=climate, config=config)
        return ensemble.segment_clo(), provenance
    if text in ENSEMBLES:
        ensemble = ENSEMBLES[text]
        return ensemble.segment_clo(), {
            "selection": "named_ensemble", "ensemble": ensemble.key,
            "ensemble_label": ensemble.label,
            "air_temperature_c": float(air_temp_c),
            "is_daytime": bool(is_daytime), "wind_ms": float(wind_ms)}
    try:
        uniform = float(text)
    except ValueError:
        raise ValueError(
            f"unknown clothing specification '{spec}'. Use 'auto', a number "
            f"of clo, or one of: {', '.join(ENSEMBLES)}") from None
    if not np.isfinite(uniform) or uniform < 0:
        raise ValueError("uniform clothing clo must be finite and non-negative")
    return (np.full(len(JOS3_SEGMENTS), uniform, dtype=float),
            {"selection": "uniform_clo", "ensemble": f"uniform_{uniform:g}clo",
             "ensemble_label": f"Uniform {uniform:g} clo on every segment",
             "air_temperature_c": float(air_temp_c),
             "is_daytime": bool(is_daytime), "wind_ms": float(wind_ms)})


def whole_body_clo(segment_clo: np.ndarray, bsa: np.ndarray) -> float:
    """Body-surface-area-weighted whole-body insulation, for reporting."""
    segment_clo = np.asarray(segment_clo, dtype=float)
    bsa = np.asarray(bsa, dtype=float)
    if segment_clo.shape != bsa.shape:
        raise ValueError("clothing and BSA arrays must have matching shape")
    return float(np.sum(segment_clo * bsa) / np.sum(bsa))


def add_clothing_arguments(parser) -> None:
    """Attach the shared clothing CLI flags."""
    parser.add_argument(
        "--clothing", default="auto",
        help="Clothing for the JOS-3 walker: 'auto' (default; chosen from "
             "the walk's own air temperature, day/night and wind), a named "
             "ensemble (" + ", ".join(ENSEMBLES) + "), or a number read as "
             "a uniform clo on every segment. 'nude' reproduces the "
             "pre-clothing behaviour.")
    parser.add_argument(
        "--clothing-climate", default="temperate",
        choices=sorted(DEFAULT_SELECTION["climate_offset_c"]),
        help="Climate the walker is habituated to; shifts the automatic "
             "selection because hot-climate residents dress lighter at the "
             "same temperature (default: temperate).")
    parser.add_argument(
        "--clothing-config", default=None,
        help="Optional JSON overriding the clothing thresholds, offsets, or "
             "the ensembles themselves (see clothing_profiles.load_config).")
