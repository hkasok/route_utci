"""radiometer_constraints.py -- measurement-constrained radiative layer.

Adds ONE new source at the top of the material/radiation evidence hierarchy: the
four-component radiometer carried along the Lisbon routes. It does not replace,
alter or duplicate the material-classification system in
``material_classification.py`` -- that remains the source of material IDENTITY
and of every property. This module supplies route-local RADIATIVE BEHAVIOUR
where it was actually measured.

    radiometer constraint > manual > osm_direct > imagery > osm_inferred >
    colour_hint > default

WHAT A RADIOMETER MEASURES, AND WHAT IT DOES NOT
------------------------------------------------
The four channels are planar hemispherical irradiances at the sensor:

    SW_down   upward-facing pyranometer   upper-hemisphere shortwave
    SW_up     downward-facing pyranometer lower-hemisphere shortwave
    LW_down   upward-facing pyrgeometer   upper-hemisphere longwave
    LW_up     downward-facing pyrgeometer lower-hemisphere longwave

None of these is a material property.

``SW_up / SW_down`` is NOT the albedo of any one surface. The downward sensor
integrates a cosine-weighted footprint several metres across that may span
asphalt, kerb, grass and a strip of shadow at once, so the ratio is an
EFFECTIVE LOCAL REFLECTANCE of whatever mixture happened to lie beneath the
cart. It is named ``effective_lower_SW_reflectance`` here for exactly that
reason.

``LW_up`` is NOT an emissivity, and cannot become one without an independent
surface temperature: a single measured radiosity ``eps*sigma*Ts^4 +
(1-eps)*LW_down`` has two unknowns. It is therefore kept as what it is, an
EFFECTIVE LOWER-HEMISPHERE RADIOSITY. The optional emissivity diagnostic below
requires a surface temperature to be supplied and is never a property source.

HETEROGENEITY IS THE POINT
--------------------------
These series vary by hundreds of W/m2 within a single walk as the cart crosses
sun/shade boundaries. Collapsing a route to one mean albedo would destroy the
very signal the measurements were taken to capture, so every constraint is
stored per sample, keyed by distance along route and timestamp, and the
interpolation onto model receptors is deliberately transition-preserving.

CONSTRAINED IS NOT VALIDATED
----------------------------
A sample used to constrain the model cannot also serve as independent
validation of it. Every row carries a ``usage`` label -- ``constraint``,
``validation`` or ``unconstrained`` -- and the metadata records which routes
were used for which, so a comparison can never quietly become circular.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SOURCE DATA CONTRACT
#
# Established by inspecting all twelve Lisbon route files rather than assumed.
# Every file carries these names; units are W/m2 throughout, confirmed both by
# magnitude and by the net-radiation identity NR = SWin - SWout + LWin - LWout.
# ---------------------------------------------------------------------------
MEASURED_COLUMN_MAP: dict[str, str] = {
    "SWin": "SW_down_measured_Wm2",
    "SWout": "SW_up_measured_Wm2",
    "LWin": "LW_down_measured_Wm2",
    "LWout": "LW_up_measured_Wm2",
}
NET_RADIATION_COLUMN = "NR"
# x_local_m / y_local_m are in the SAME metric frame as the STL meshes and as
# stage-05's path_xyz, which is what makes georeferencing a lookup rather than a
# transformation. x_proj_m / y_proj_m are the EPSG:3763 projected pair.
LOCAL_X_COLUMN = "x_local_m"
LOCAL_Y_COLUMN = "y_local_m"
TIMESTAMP_COLUMN = "timestamp_utc_refined"
LOCAL_TIMESTAMP_COLUMN = "timestamp_local_refined"
SEQUENCE_COLUMN = "seq"

CONSTRAINT_SOURCE = "four_component_radiometer"

# Per-channel provenance labels (specification section 17).
SOURCE_RADIOMETER = "four_component_radiometer"
SOURCE_MODELLED_SURFACE = "modeled_surface"
SOURCE_MODELLED_SKY = "modeled_sky"

QC_VALID = "valid"
QC_QUESTIONABLE = "questionable"
QC_INVALID = "invalid"

# Confidence of a measurement-derived constraint. Above every classification
# source in material_classification.SOURCE_CONFIDENCE (manual override is 1.00),
# because this is an instrument reading of the actual radiative environment
# rather than an inference about what the surface is made of. Not 1.0: the
# Lisbon pyranometer is demonstrably tilt-affected, so a measured value is
# authoritative about the sensor, not perfect about the world.
CONSTRAINT_CONFIDENCE = 0.98
QUESTIONABLE_CONFIDENCE = 0.60


@dataclass
class QualityControlConfig:
    """Physical QC limits.

    Deliberately loose. The instrument's own range, not an opinion about what
    the site should look like, sets these. Aggressive rejection would discard
    the very excursions -- deep shade, sunlit plaza -- that make the series
    worth having. Anything outside the plausible band is flagged
    ``questionable`` and kept; only physically impossible values are
    ``invalid``.
    """

    # A pyranometer's nighttime thermal offset is genuinely a few W/m2 negative.
    # The Lisbon night routes sit at -1.7 to -3.6 W/m2 mean, so a small negative
    # reading is normal instrument behaviour and must not be thrown away.
    shortwave_minimum_Wm2: float = -20.0
    shortwave_maximum_Wm2: float = 1600.0
    # A reading between -this and 0 is the documented nighttime thermal offset
    # and counts as VALID. The Lisbon night routes average -1.7 to -3.6 W/m2,
    # which is the instrument behaving normally, not a fault. Flagging it
    # "questionable" would mark every night sample suspect for no reason.
    shortwave_night_offset_tolerance_Wm2: float = 10.0
    # A horizontal pyranometer at this latitude cannot exceed roughly this in
    # clear sky. Higher readings indicate the cart-mounted sensor tilting toward
    # the sun. Flagged questionable, NEVER silently corrected -- the excursion
    # is evidence about the instrument and must stay visible.
    shortwave_down_plausible_max_Wm2: float = 1100.0
    longwave_minimum_Wm2: float = 100.0
    longwave_maximum_Wm2: float = 700.0
    minimum_sw_down_for_reflectance_Wm2: float = 50.0
    reflectance_minimum: float = 0.0
    reflectance_maximum: float = 1.0
    net_radiation_closure_tolerance_Wm2: float = 25.0

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterpolationConfig:
    """How measurements are carried onto model receptor points.

    ``nearest`` is the DEFAULT and it is a deliberate choice. Linear
    interpolation across a sun/shade boundary invents intermediate irradiances
    that the instrument never saw and that no surface produced; nearest-sample
    assignment preserves the measured transition exactly, at the cost of a
    step. Since preserving heterogeneity is the whole purpose of this layer,
    the step is the lesser evil. ``linear`` and ``distance_weighted`` are
    available for smoothly varying channels such as LW_down.
    """

    method: str = "nearest"
    max_gap_s: float = 60.0
    max_gap_m: float = 25.0
    distance_weight_power: float = 2.0
    neighbours: int = 3

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


VALID_METHODS = ("nearest", "linear", "distance_weighted")


class RadiometerConstraintError(ValueError):
    """Raised when the measurement contract is not met."""


# ---------------------------------------------------------------------------
# Loading and quality control
# ---------------------------------------------------------------------------
def load_route_measurements(path: str | Path, case_id: str | None = None
                            ) -> pd.DataFrame:
    """Read one route measurement file into the normalised constraint schema.

    The raw file is never modified; this returns a NEW frame carrying the
    observed fluxes under explicit names. The original columns are preserved
    alongside so nothing is lost.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    missing = sorted(set(MEASURED_COLUMN_MAP) - set(raw.columns))
    if missing:
        raise RadiometerConstraintError(
            f"{path}: missing radiometer columns {missing}; expected "
            f"{sorted(MEASURED_COLUMN_MAP)}")
    for column in (LOCAL_X_COLUMN, LOCAL_Y_COLUMN, TIMESTAMP_COLUMN):
        if column not in raw.columns:
            raise RadiometerConstraintError(f"{path}: missing {column!r}")

    frame = pd.DataFrame(index=raw.index)
    frame["case_id"] = case_id or path.parts[-3]
    frame["route_id"] = raw["route_id"].astype(int) if "route_id" in raw else -1
    frame["seq"] = raw[SEQUENCE_COLUMN] if SEQUENCE_COLUMN in raw else raw.index
    frame["timestamp_utc"] = pd.to_datetime(raw[TIMESTAMP_COLUMN],
                                            format="ISO8601", utc=True)
    if LOCAL_TIMESTAMP_COLUMN in raw.columns:
        frame["timestamp_local"] = raw[LOCAL_TIMESTAMP_COLUMN]
    frame["x"] = raw[LOCAL_X_COLUMN].astype(float)
    frame["y"] = raw[LOCAL_Y_COLUMN].astype(float)
    for source, target in MEASURED_COLUMN_MAP.items():
        frame[target] = raw[source].astype(float)
    if NET_RADIATION_COLUMN in raw.columns:
        frame["net_radiation_measured_Wm2"] = raw[NET_RADIATION_COLUMN].astype(float)
    frame["source_file"] = str(path)

    step = np.hypot(np.diff(frame["x"].to_numpy()), np.diff(frame["y"].to_numpy()))
    frame["distance_along_route_m"] = np.concatenate([[0.0], np.cumsum(step)])
    elapsed = (frame["timestamp_utc"] - frame["timestamp_utc"].iloc[0])
    frame["elapsed_s"] = elapsed.dt.total_seconds()
    return frame


def apply_quality_control(frame: pd.DataFrame,
                          config: QualityControlConfig | None = None
                          ) -> pd.DataFrame:
    """Flag every sample ``valid`` / ``questionable`` / ``invalid``.

    Per channel and overall. Nothing is dropped and nothing is clipped: a
    flagged row still carries its raw value, so a later analysis can decide for
    itself. Clipping here would hide the tilt problem rather than report it.
    """
    config = config or QualityControlConfig()
    frame = frame.copy()
    channels = {
        "SW_down": ("SW_down_measured_Wm2", config.shortwave_minimum_Wm2,
                    config.shortwave_maximum_Wm2),
        "SW_up": ("SW_up_measured_Wm2", config.shortwave_minimum_Wm2,
                  config.shortwave_maximum_Wm2),
        "LW_down": ("LW_down_measured_Wm2", config.longwave_minimum_Wm2,
                    config.longwave_maximum_Wm2),
        "LW_up": ("LW_up_measured_Wm2", config.longwave_minimum_Wm2,
                  config.longwave_maximum_Wm2),
    }
    overall = pd.Series(QC_VALID, index=frame.index, dtype=object)
    for label, (column, low, high) in channels.items():
        values = frame[column].to_numpy(float)
        flag = np.full(len(values), QC_VALID, dtype=object)
        finite = np.isfinite(values)
        flag[~finite] = QC_INVALID
        outside = finite & ((values < low) | (values > high))
        flag[outside] = QC_INVALID
        if label.startswith("SW"):
            # A small negative reading is the known nighttime thermal offset and
            # stays VALID; only a larger negative excursion is suspect.
            suspect = finite & (values < -config.shortwave_night_offset_tolerance_Wm2)
            flag[suspect & (flag == QC_VALID)] = QC_QUESTIONABLE
            if label == "SW_down":
                tilted = finite & (values > config.shortwave_down_plausible_max_Wm2)
                flag[tilted & (flag == QC_VALID)] = QC_QUESTIONABLE
        else:
            # Longwave far from any terrestrial blackbody is suspect but not
            # impossible; keep and flag.
            suspect = finite & ((values < 150.0) | (values > 650.0))
            flag[suspect & (flag == QC_VALID)] = QC_QUESTIONABLE
        frame[f"qc_{label}"] = flag
        overall = np.where(flag == QC_INVALID, QC_INVALID,
                           np.where((flag == QC_QUESTIONABLE)
                                    & (overall != QC_INVALID),
                                    QC_QUESTIONABLE, overall))
        overall = pd.Series(overall, index=frame.index, dtype=object)

    if "net_radiation_measured_Wm2" in frame.columns:
        closure = (frame["SW_down_measured_Wm2"] - frame["SW_up_measured_Wm2"]
                   + frame["LW_down_measured_Wm2"] - frame["LW_up_measured_Wm2"]
                   - frame["net_radiation_measured_Wm2"])
        frame["net_radiation_closure_Wm2"] = closure
        frame["qc_net_closure"] = np.where(
            closure.abs() <= config.net_radiation_closure_tolerance_Wm2,
            QC_VALID, QC_QUESTIONABLE)
    frame["qc_flag"] = overall
    frame["confidence"] = np.where(
        overall == QC_VALID, CONSTRAINT_CONFIDENCE,
        np.where(overall == QC_QUESTIONABLE, QUESTIONABLE_CONFIDENCE, 0.0))
    frame["source"] = CONSTRAINT_SOURCE
    return frame


def effective_lower_sw_reflectance(sw_down: Any, sw_up: Any,
                                   config: QualityControlConfig | None = None
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Effective lower-hemisphere shortwave reflectance, and its flag.

    ``alpha_eff = SW_up / SW_down``, computed ONLY where SW_down is large
    enough for the ratio to mean anything. Near dawn, at night, or in deep
    shade the denominator collapses and the ratio explodes or goes negative;
    returning a large number there would be worse than returning nothing, so
    those samples come back as NaN with an explicit flag.

    Values outside [0, 1] are returned AS COMPUTED and flagged invalid rather
    than clipped -- a ratio above one is evidence of something real (a tilted
    sensor, or sunlit surroundings reflecting into a shaded downward sensor),
    and silently clamping it to 1.0 would erase that evidence.

    This is an effective LOCAL reflectance of the sensor's mixed footprint, not
    the intrinsic albedo of any material.
    """
    config = config or QualityControlConfig()
    down = np.asarray(sw_down, dtype=float)
    up = np.asarray(sw_up, dtype=float)
    flag = np.full(down.shape, QC_VALID, dtype=object)
    usable = (np.isfinite(down) & np.isfinite(up)
              & (down >= config.minimum_sw_down_for_reflectance_Wm2))
    reflectance = np.full(down.shape, np.nan, dtype=float)
    reflectance[usable] = up[usable] / down[usable]
    flag[~usable] = QC_INVALID
    out_of_range = usable & ((reflectance < config.reflectance_minimum)
                             | (reflectance > config.reflectance_maximum))
    flag[out_of_range] = QC_INVALID
    return reflectance, flag


def optional_emissivity_diagnostic(lw_up: Any, lw_down: Any,
                                   surface_temperature_C: Any,
                                   sigma: float = 5.670374419e-8) -> np.ndarray:
    """DIAGNOSTIC ONLY: emissivity implied by a measured radiosity.

    Requires an INDEPENDENT surface temperature. Inverting

        LW_up = eps*sigma*Ts^4 + (1 - eps)*LW_down

    for eps is only possible once Ts is known from somewhere else; with LW_up
    alone the equation has two unknowns. This is never a property source and
    never feeds the material library -- it exists so that a future study with
    thermal-camera surface temperatures can check the assumed emissivities.
    """
    up = np.asarray(lw_up, dtype=float)
    down = np.asarray(lw_down, dtype=float)
    temperature = np.asarray(surface_temperature_C, dtype=float)
    emitted = sigma * (temperature + 273.15) ** 4
    denominator = emitted - down
    result = np.full(up.shape, np.nan, dtype=float)
    usable = np.isfinite(denominator) & (np.abs(denominator) > 1.0)
    result[usable] = (up[usable] - down[usable]) / denominator[usable]
    result[usable & ((result < 0.0) | (result > 1.0))] = np.nan
    return result


def build_constraints(frame: pd.DataFrame,
                      sensor_height_m: float,
                      qc_config: QualityControlConfig | None = None,
                      ground_z: Any = None,
                      usage: str = "constraint") -> pd.DataFrame:
    """Assemble the route-local radiative constraint table for one route."""
    qc_config = qc_config or QualityControlConfig()
    constraints = apply_quality_control(frame, qc_config)
    reflectance, reflectance_flag = effective_lower_sw_reflectance(
        constraints["SW_down_measured_Wm2"], constraints["SW_up_measured_Wm2"],
        qc_config)
    constraints["effective_lower_SW_reflectance"] = reflectance
    constraints["qc_effective_reflectance"] = reflectance_flag
    # LW_up and LW_down are carried through as what they physically are. No
    # emissivity is inferred; see the module docstring.
    constraints["effective_lower_LW_radiosity_Wm2"] = (
        constraints["LW_up_measured_Wm2"])
    constraints["effective_upper_LW_irradiance_Wm2"] = (
        constraints["LW_down_measured_Wm2"])
    if ground_z is None:
        constraints["local_ground_z"] = np.nan
        constraints["sensor_z"] = np.nan
    else:
        ground = np.asarray(ground_z, dtype=float)
        constraints["local_ground_z"] = ground
        constraints["sensor_z"] = ground + float(sensor_height_m)
    constraints["sensor_height_m"] = float(sensor_height_m)
    constraints["usage"] = usage
    return constraints


# ---------------------------------------------------------------------------
# Association with model receptor points
# ---------------------------------------------------------------------------
CONSTRAINED_CHANNELS = ("SW_down", "SW_up", "LW_down", "LW_up")


def interpolate_to_route(constraints: pd.DataFrame,
                         target_distance_m: Any,
                         target_elapsed_s: Any = None,
                         config: InterpolationConfig | None = None
                         ) -> pd.DataFrame:
    """Carry the measured channels onto model receptor points.

    Association is by DISTANCE ALONG ROUTE, with an optional simultaneous
    tolerance on elapsed time. A receptor further than ``max_gap_m`` from the
    nearest valid sample -- or, when times are supplied, further than
    ``max_gap_s`` from it -- gets NOTHING, and the caller falls back to the
    material hierarchy. Bridging a 400 m gap (lisbon4's night route really has
    one) would be extrapolation dressed as measurement.
    """
    config = config or InterpolationConfig()
    if config.method not in VALID_METHODS:
        raise RadiometerConstraintError(
            f"unknown interpolation method {config.method!r}; "
            f"expected one of {VALID_METHODS}")
    target_distance = np.asarray(target_distance_m, dtype=float)
    n_targets = target_distance.size
    usable = constraints[constraints["qc_flag"] != QC_INVALID]
    result = pd.DataFrame(index=np.arange(n_targets))
    result["distance_along_route_m"] = target_distance
    for channel in CONSTRAINED_CHANNELS:
        result[f"{channel}_constrained_Wm2"] = np.nan
        result[f"{channel}_source"] = ""
    result["effective_lower_SW_reflectance"] = np.nan
    result["constraint_confidence"] = 0.0
    result["constraint_gap_m"] = np.nan
    result["constraint_gap_s"] = np.nan
    result["constraint_available"] = False
    if usable.empty or n_targets == 0:
        return result

    source_distance = usable["distance_along_route_m"].to_numpy(float)
    order = np.argsort(source_distance)
    source_distance = source_distance[order]
    positions = np.searchsorted(source_distance, target_distance)
    left = np.clip(positions - 1, 0, len(source_distance) - 1)
    right = np.clip(positions, 0, len(source_distance) - 1)
    gap_left = np.abs(target_distance - source_distance[left])
    gap_right = np.abs(source_distance[right] - target_distance)
    nearest_slot = np.where(gap_left <= gap_right, left, right)
    gap_m = np.minimum(gap_left, gap_right)

    gap_s = np.full(n_targets, np.nan)
    if target_elapsed_s is not None and "elapsed_s" in usable.columns:
        source_elapsed = usable["elapsed_s"].to_numpy(float)[order]
        gap_s = np.abs(np.asarray(target_elapsed_s, dtype=float)
                       - source_elapsed[nearest_slot])

    within = gap_m <= config.max_gap_m
    if target_elapsed_s is not None:
        within &= (~np.isfinite(gap_s)) | (gap_s <= config.max_gap_s)

    ordered = usable.iloc[order].reset_index(drop=True)
    for channel in CONSTRAINED_CHANNELS:
        column = f"{channel}_measured_Wm2"
        values = ordered[column].to_numpy(float)
        channel_flag = ordered[f"qc_{channel}"].to_numpy()
        if config.method == "nearest":
            picked = values[nearest_slot]
            picked_ok = channel_flag[nearest_slot] != QC_INVALID
        elif config.method == "linear":
            picked = np.interp(target_distance, source_distance, values)
            picked_ok = ((channel_flag[left] != QC_INVALID)
                         & (channel_flag[right] != QC_INVALID))
        else:
            weight_left = 1.0 / np.maximum(gap_left, 1e-6) ** config.distance_weight_power
            weight_right = 1.0 / np.maximum(gap_right, 1e-6) ** config.distance_weight_power
            picked = ((values[left] * weight_left + values[right] * weight_right)
                      / (weight_left + weight_right))
            picked_ok = ((channel_flag[left] != QC_INVALID)
                         & (channel_flag[right] != QC_INVALID))
        keep = within & picked_ok
        result.loc[keep, f"{channel}_constrained_Wm2"] = picked[keep]
        result.loc[keep, f"{channel}_source"] = SOURCE_RADIOMETER

    reflectance = ordered["effective_lower_SW_reflectance"].to_numpy(float)
    reflectance_ok = (ordered["qc_effective_reflectance"].to_numpy()
                      != QC_INVALID)
    keep = within & reflectance_ok[nearest_slot]
    result.loc[keep, "effective_lower_SW_reflectance"] = (
        reflectance[nearest_slot][keep])
    confidence = ordered["confidence"].to_numpy(float)
    result.loc[within, "constraint_confidence"] = confidence[nearest_slot][within]
    result["constraint_gap_m"] = gap_m
    result["constraint_gap_s"] = gap_s
    result["constraint_available"] = within
    result["interpolation_method"] = config.method
    return result


def resolve_flux_sources(constrained: pd.DataFrame,
                         modelled: Mapping[str, Any],
                         material_source: Any = None) -> pd.DataFrame:
    """The section-15 resolver: measurement first, model otherwise.

    Returns one resolved value and one PROVENANCE LABEL per channel per
    receptor. The rule is flatly stated: where a valid radiometer constraint
    exists it is used; everywhere else the modelled flux stands, carrying the
    label of whatever material evidence produced it.

    ``material_source`` is passed through UNCHANGED and reported separately.
    A measurement constrains the radiative behaviour of a location; it does not
    make the asphalt underneath stop being asphalt, and erasing the
    classification would throw away information the measurement never
    contradicted.
    """
    resolved = pd.DataFrame(index=constrained.index)
    for channel in CONSTRAINED_CHANNELS:
        measured = constrained[f"{channel}_constrained_Wm2"].to_numpy(float)
        model = np.asarray(modelled[channel], dtype=float)
        if model.shape != measured.shape:
            raise RadiometerConstraintError(
                f"{channel}: modelled array {model.shape} does not match "
                f"{measured.shape} receptors")
        have = np.isfinite(measured)
        resolved[f"{channel}_resolved_Wm2"] = np.where(have, measured, model)
        # Downward-looking channels see surfaces; upward-looking ones see sky
        # plus whatever fills the rest of the hemisphere.
        fallback = (SOURCE_MODELLED_SURFACE if channel.endswith("_up")
                    else SOURCE_MODELLED_SKY)
        resolved[f"{channel}_source"] = np.where(have, SOURCE_RADIOMETER,
                                                 fallback)
        resolved[f"{channel}_measured_Wm2"] = measured
        resolved[f"{channel}_modelled_Wm2"] = model
    resolved["radiative_constraint_source"] = np.where(
        constrained["constraint_available"].to_numpy(bool),
        CONSTRAINT_SOURCE, "material_hierarchy")
    if material_source is not None:
        # Material identity is preserved alongside, never replaced.
        resolved["material_source"] = np.asarray(material_source)
    return resolved


# ---------------------------------------------------------------------------
# Surface-group association (section 10)
# ---------------------------------------------------------------------------
def visible_group_weights(footprint_weights: Any, group_ids: Sequence[str],
                          top_n: int = 6) -> list[dict[str, float]]:
    """Which surface groups the downward sensor actually sees, and in what share.

    Built on the SAME cosine footprint kernel stage 05 uses for its emulated
    downward radiometer, so the weights describe the real instrument view
    rather than the nearest triangle. Assigning a measurement to one triangle
    would be wrong in exactly the way the footprint work established: at 1 m
    height, half the signal comes from beyond a 1 m radius.

    Recorded for a later, more rigorous inversion; nothing consumes it yet.
    """
    import scipy.sparse as sp

    matrix = footprint_weights
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=float))
    group_ids = np.asarray(group_ids)
    out: list[dict[str, float]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns = matrix.indices[start:end]
        weights = matrix.data[start:end]
        if not len(weights):
            out.append({})
            continue
        totals: dict[str, float] = {}
        for column, weight in zip(columns, weights):
            name = str(group_ids[column])
            totals[name] = totals.get(name, 0.0) + float(weight)
        total = sum(totals.values())
        if total <= 0:
            out.append({})
            continue
        ranked = sorted(totals.items(), key=lambda item: -item[1])[:top_n]
        out.append({name: weight / total for name, weight in ranked})
    return out


# ---------------------------------------------------------------------------
# Validation statistics
# ---------------------------------------------------------------------------
def channel_statistics(measured: Any, modelled: Any) -> dict[str, float]:
    """Bias, MAE, RMSE, r and the ordinary-least-squares fit for one channel."""
    measured = np.asarray(measured, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(modelled)
    if valid.sum() < 3:
        return {"n": int(valid.sum())}
    measured, modelled = measured[valid], modelled[valid]
    residual = modelled - measured
    record = {
        "n": int(valid.sum()),
        "measured_mean": float(measured.mean()),
        "modelled_mean": float(modelled.mean()),
        "measured_sd": float(measured.std()),
        "modelled_sd": float(modelled.std()),
        "bias_Wm2": float(residual.mean()),
        "mae_Wm2": float(np.abs(residual).mean()),
        "rmse_Wm2": float(np.sqrt(np.mean(residual ** 2))),
    }
    if measured.std() > 0 and modelled.std() > 0:
        record["pearson_r"] = float(np.corrcoef(measured, modelled)[0, 1])
        slope, intercept = np.polyfit(measured, modelled, 1)
        record["slope"] = float(slope)
        record["intercept_Wm2"] = float(intercept)
    return record


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
