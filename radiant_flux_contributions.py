"""Absorbed radiant-flux bookkeeping and ranked route-result figures.

All quantities in this module are whole-body absorbed radiant fluxes in
W m-2.  Mean radiant temperature is deliberately retained only as a nonlinear
diagnostic reconstructed from the total flux; it is never decomposed into
additive temperature bands.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTRIBUTION_ARCHIVE = "radiant_flux_contributions.npz"
CONTRIBUTION_METADATA = "radiant_flux_contributions_metadata.json"

PRIMARY_COLUMNS = [
    "sw_direct_absorbed_Wm2",
    "sw_diffuse_sky_absorbed_Wm2",
    "sw_reflected_total_absorbed_Wm2",
    "lw_sky_absorbed_Wm2",
    "lw_surface_total_absorbed_Wm2",
]

SW_SOURCE_COLUMNS = [
    "sw_reflected_asphalt_road_absorbed_Wm2",
    "sw_reflected_asphalt_path_absorbed_Wm2",
    "sw_reflected_concrete_sidewalk_absorbed_Wm2",
    "sw_reflected_paving_stones_absorbed_Wm2",
    "sw_reflected_generic_ground_absorbed_Wm2",
    "sw_reflected_grass_absorbed_Wm2",
    "sw_reflected_building_wall_absorbed_Wm2",
    "sw_reflected_roof_absorbed_Wm2",
    "sw_reflected_vegetation_absorbed_Wm2",
    "sw_reflected_water_absorbed_Wm2",
    "sw_reflected_other_absorbed_Wm2",
    "sw_reflected_concrete_road_absorbed_Wm2",
    "sw_reflected_asphalt_parking_absorbed_Wm2",
    "sw_reflected_concrete_parking_absorbed_Wm2",
    "sw_reflected_gravel_parking_absorbed_Wm2",
    "sw_reflected_artificial_turf_absorbed_Wm2",
    "sw_reflected_sports_surface_absorbed_Wm2",
    "sw_reflected_playground_surface_absorbed_Wm2",
    "sw_reflected_bare_ground_absorbed_Wm2",
    "sw_reflected_pedestrian_crossing_absorbed_Wm2",
    "sw_reflected_pedestrian_plaza_absorbed_Wm2",
]

LW_SOURCE_COLUMNS = [
    "lw_asphalt_road_absorbed_Wm2",
    "lw_asphalt_path_absorbed_Wm2",
    "lw_concrete_sidewalk_absorbed_Wm2",
    "lw_paving_stones_absorbed_Wm2",
    "lw_generic_ground_absorbed_Wm2",
    "lw_grass_absorbed_Wm2",
    "lw_building_wall_absorbed_Wm2",
    "lw_roof_absorbed_Wm2",
    "lw_tree_canopy_absorbed_Wm2",
    "lw_tree_trunk_absorbed_Wm2",
    "lw_water_absorbed_Wm2",
    "lw_other_surface_absorbed_Wm2",
    "lw_concrete_road_absorbed_Wm2",
    "lw_asphalt_parking_absorbed_Wm2",
    "lw_concrete_parking_absorbed_Wm2",
    "lw_gravel_parking_absorbed_Wm2",
    "lw_artificial_turf_absorbed_Wm2",
    "lw_sports_surface_absorbed_Wm2",
    "lw_playground_surface_absorbed_Wm2",
    "lw_bare_ground_absorbed_Wm2",
    "lw_pedestrian_crossing_absorbed_Wm2",
    "lw_pedestrian_plaza_absorbed_Wm2",
]

TOTAL_COLUMNS = [
    "sw_total_absorbed_Wm2",
    "lw_total_absorbed_Wm2",
    "total_absorbed_radiant_flux_Wm2",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "record_primary_mechanisms": True,
    "record_material_resolved_sources": True,
    "plot_mode": "primary",
    "sort_direction": "descending",
    "sorting_variable": "each_contribution_independently",
    "plot_style": "independently_ranked_lines",
    "separate_figure_per_route": True,
    "combined_multi_panel_figure": True,
    "shared_y_axis": True,
    "separate_color_mode": "ranked_emphasis",
    "combined_color_mode": "fixed_physical_categories",
    "largest_contribution_color": "#d62728",
    "surface_longwave_classification_plot": {
        "enabled": True,
        "sort_direction": "descending",
        "separate_figure_per_route": True,
        "combined_multi_panel_figure": True,
        "shared_y_axis": True,
        "minimum_mean_absorbed_flux_Wm2": 1.0e-8,
    },
    "fixed_category_colors": {
        "sw_direct_absorbed_Wm2": "#e31a1c",
        "sw_diffuse_sky_absorbed_Wm2": "#fdbf6f",
        "sw_reflected_total_absorbed_Wm2": "#9467bd",
        "lw_sky_absorbed_Wm2": "#6baed6",
        "lw_surface_total_absorbed_Wm2": "#756bb1",
        "lw_asphalt_road_absorbed_Wm2": "#4d4d4d",
        "lw_asphalt_path_absorbed_Wm2": "#a65628",
        "lw_concrete_sidewalk_absorbed_Wm2": "#e69f00",
        "lw_paving_stones_absorbed_Wm2": "#cc79a7",
        "lw_generic_ground_absorbed_Wm2": "#009e73",
        "lw_grass_absorbed_Wm2": "#7cae00",
        "lw_building_wall_absorbed_Wm2": "#0072b2",
        "lw_roof_absorbed_Wm2": "#6a3d9a",
        "lw_tree_canopy_absorbed_Wm2": "#33a02c",
        "lw_tree_trunk_absorbed_Wm2": "#8c510a",
        "lw_water_absorbed_Wm2": "#00bfc4",
        "lw_other_surface_absorbed_Wm2": "#999999",
        "Other": "#969696",
    },
    "material_resolved": {
        "maximum_categories": 16,
        "minimum_mean_fraction_percent": 0.0,
        "minimum_mean_absorbed_flux_Wm2": 1.0e-8,
        "group_remaining_as_other": True,
    },
    "validation": {
        "absolute_tolerance_Wm2": 1.0e-4,
        "relative_tolerance": 1.0e-6,
        "mrt_absolute_tolerance_C": 2.0e-4,
    },
    "export": {
        "receptor_csv": True,
        "route_summary_csv": True,
        "pdf": True,
        "png": True,
        "dpi": 300,
    },
}


DISPLAY_LABELS = {
    "sw_direct_absorbed_Wm2": "Direct shortwave",
    "sw_diffuse_sky_absorbed_Wm2": "Diffuse-sky shortwave",
    "sw_reflected_total_absorbed_Wm2": "Reflected shortwave",
    "lw_sky_absorbed_Wm2": "Atmospheric longwave",
    "lw_surface_total_absorbed_Wm2": "Surface longwave",
    "sw_reflected_asphalt_road_absorbed_Wm2": "SW reflected: asphalt road",
    "sw_reflected_asphalt_path_absorbed_Wm2": "SW reflected: asphalt path",
    "sw_reflected_concrete_sidewalk_absorbed_Wm2": "SW reflected: concrete sidewalk/path",
    "sw_reflected_paving_stones_absorbed_Wm2": "SW reflected: paving stones",
    "sw_reflected_generic_ground_absorbed_Wm2": "SW reflected: generic ground",
    "sw_reflected_grass_absorbed_Wm2": "SW reflected: grass/low vegetation",
    "sw_reflected_building_wall_absorbed_Wm2": "SW reflected: building wall",
    "sw_reflected_roof_absorbed_Wm2": "SW reflected: roof",
    "sw_reflected_vegetation_absorbed_Wm2": "SW reflected: vegetation",
    "sw_reflected_water_absorbed_Wm2": "SW reflected: water",
    "sw_reflected_other_absorbed_Wm2": "SW reflected: other",
    "sw_reflected_concrete_road_absorbed_Wm2": "SW reflected: concrete road",
    "sw_reflected_asphalt_parking_absorbed_Wm2": "SW reflected: asphalt parking",
    "sw_reflected_concrete_parking_absorbed_Wm2": "SW reflected: concrete parking",
    "sw_reflected_gravel_parking_absorbed_Wm2": "SW reflected: gravel parking",
    "sw_reflected_artificial_turf_absorbed_Wm2": "SW reflected: artificial turf",
    "sw_reflected_sports_surface_absorbed_Wm2": "SW reflected: sports surface",
    "sw_reflected_playground_surface_absorbed_Wm2": "SW reflected: playground",
    "sw_reflected_bare_ground_absorbed_Wm2": "SW reflected: bare ground",
    "sw_reflected_pedestrian_crossing_absorbed_Wm2": "SW reflected: crossing",
    "sw_reflected_pedestrian_plaza_absorbed_Wm2": "SW reflected: pedestrian plaza",
    "lw_asphalt_road_absorbed_Wm2": "LW surface: asphalt road",
    "lw_asphalt_path_absorbed_Wm2": "LW surface: asphalt path",
    "lw_concrete_sidewalk_absorbed_Wm2": "LW surface: concrete sidewalk/path",
    "lw_paving_stones_absorbed_Wm2": "LW surface: paving stones",
    "lw_generic_ground_absorbed_Wm2": "LW surface: generic ground",
    "lw_grass_absorbed_Wm2": "LW surface: grass/low vegetation",
    "lw_building_wall_absorbed_Wm2": "LW surface: building wall",
    "lw_roof_absorbed_Wm2": "LW surface: roof",
    "lw_tree_canopy_absorbed_Wm2": "LW surface: tree canopy",
    "lw_tree_trunk_absorbed_Wm2": "LW surface: tree trunk",
    "lw_water_absorbed_Wm2": "LW surface: water",
    "lw_other_surface_absorbed_Wm2": "LW surface: other",
    "lw_concrete_road_absorbed_Wm2": "LW surface: concrete road",
    "lw_asphalt_parking_absorbed_Wm2": "LW surface: asphalt parking",
    "lw_concrete_parking_absorbed_Wm2": "LW surface: concrete parking",
    "lw_gravel_parking_absorbed_Wm2": "LW surface: gravel parking",
    "lw_artificial_turf_absorbed_Wm2": "LW surface: artificial turf",
    "lw_sports_surface_absorbed_Wm2": "LW surface: sports surface",
    "lw_playground_surface_absorbed_Wm2": "LW surface: playground",
    "lw_bare_ground_absorbed_Wm2": "LW surface: bare ground",
    "lw_pedestrian_crossing_absorbed_Wm2": "LW surface: crossing",
    "lw_pedestrian_plaza_absorbed_Wm2": "LW surface: pedestrian plaza",
    "Other": "Other",
}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_contribution_config(path: str | Path | None = None,
                             *, default_enabled: bool = False) -> dict[str, Any]:
    """Load contribution configuration.

    A missing path defaults to disabled so legacy direct stage invocations do
    not allocate contribution matrices. ``start.sh`` passes the repository
    configuration explicitly and therefore enables the result branch.
    """
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        config["enabled"] = bool(default_enabled)
    else:
        with open(path, encoding="utf-8") as stream:
            _deep_update(config, json.load(stream))
    resolved_modes = {"surface_longwave_resolved", "material_resolved"}
    if config["plot_mode"] not in {"primary", *resolved_modes}:
        raise ValueError(
            "radiant-flux plot_mode must be primary, surface_longwave_resolved, "
            "or material_resolved")
    if config["sort_direction"] != "descending":
        raise ValueError("ranked radiant-flux figures require descending sorting")
    if config["sorting_variable"] != "each_contribution_independently":
        raise ValueError("radiant-flux curves must rank each contribution independently")
    if config["plot_style"] != "independently_ranked_lines":
        raise ValueError("radiant-flux contribution figures must use line-only ranked curves")
    surface_lw_plot = config["surface_longwave_classification_plot"]
    if surface_lw_plot["sort_direction"] != "descending":
        raise ValueError(
            "surface-longwave classification curves must use descending sorting")
    if float(surface_lw_plot["minimum_mean_absorbed_flux_Wm2"]) < 0:
        raise ValueError("surface-longwave classification threshold cannot be negative")
    if config["enabled"] and not config["record_primary_mechanisms"]:
        raise ValueError("primary absorbed-flux mechanisms are required when recording is enabled")
    if (config["enabled"] and config["plot_mode"] in resolved_modes
            and not config["record_material_resolved_sources"]):
        raise ValueError("source-resolved plot modes require material source recording")
    if (config["enabled"] and surface_lw_plot["enabled"]
            and not config["record_material_resolved_sources"]):
        raise ValueError(
            "surface-longwave classification plots require material source recording")
    for key in ("separate_color_mode", "combined_color_mode"):
        if config[key] not in {"ranked_emphasis", "fixed_physical_categories"}:
            raise ValueError(f"invalid {key}: {config[key]}")
    mr = config["material_resolved"]
    if int(mr["maximum_categories"]) < 4:
        raise ValueError("material_resolved.maximum_categories must be at least 4")
    if float(mr["minimum_mean_fraction_percent"]) < 0:
        raise ValueError("material resolved fraction threshold cannot be negative")
    return config


def canonical_sw_source(material_name: str) -> str:
    """Map a current TREC-Route material/object name to a SW result column."""
    name = str(material_name).lower()
    mapping = {
        "asphalt_road": "sw_reflected_asphalt_road_absorbed_Wm2",
        "asphalt_pedestrian_path": "sw_reflected_asphalt_path_absorbed_Wm2",
        "asphalt_pedestrian": "sw_reflected_asphalt_path_absorbed_Wm2",
        "concrete_sidewalk": "sw_reflected_concrete_sidewalk_absorbed_Wm2",
        "concrete_pedestrian": "sw_reflected_concrete_sidewalk_absorbed_Wm2",
        "paving_stone_path": "sw_reflected_paving_stones_absorbed_Wm2",
        "paving_stone_pedestrian": "sw_reflected_paving_stones_absorbed_Wm2",
        "generic_ground": "sw_reflected_generic_ground_absorbed_Wm2",
        "ground": "sw_reflected_generic_ground_absorbed_Wm2",
        "unpaved_path": "sw_reflected_generic_ground_absorbed_Wm2",
        "grass_or_low_vegetation": "sw_reflected_grass_absorbed_Wm2",
        "grass_lawn": "sw_reflected_grass_absorbed_Wm2",
        "grass": "sw_reflected_grass_absorbed_Wm2",
        "wall": "sw_reflected_building_wall_absorbed_Wm2",
        "roof": "sw_reflected_roof_absorbed_Wm2",
        "vegetation": "sw_reflected_vegetation_absorbed_Wm2",
        "water": "sw_reflected_water_absorbed_Wm2",
        "concrete_road": "sw_reflected_concrete_road_absorbed_Wm2",
        "asphalt_parking": "sw_reflected_asphalt_parking_absorbed_Wm2",
        "concrete_parking": "sw_reflected_concrete_parking_absorbed_Wm2",
        "gravel_parking": "sw_reflected_gravel_parking_absorbed_Wm2",
        "artificial_turf": "sw_reflected_artificial_turf_absorbed_Wm2",
        "sports_surface": "sw_reflected_sports_surface_absorbed_Wm2",
        "playground_surface": "sw_reflected_playground_surface_absorbed_Wm2",
        "bare_ground": "sw_reflected_bare_ground_absorbed_Wm2",
        "pedestrian_crossing": "sw_reflected_pedestrian_crossing_absorbed_Wm2",
        "pedestrian_plaza": "sw_reflected_pedestrian_plaza_absorbed_Wm2",
    }
    return mapping.get(name, "sw_reflected_other_absorbed_Wm2")


def canonical_lw_source(material_name: str) -> str:
    """Map a current TREC-Route material/object name to a LW result column."""
    name = str(material_name).lower()
    mapping = {
        "asphalt_road": "lw_asphalt_road_absorbed_Wm2",
        "asphalt_pedestrian_path": "lw_asphalt_path_absorbed_Wm2",
        "asphalt_pedestrian": "lw_asphalt_path_absorbed_Wm2",
        "concrete_sidewalk": "lw_concrete_sidewalk_absorbed_Wm2",
        "concrete_pedestrian": "lw_concrete_sidewalk_absorbed_Wm2",
        "paving_stone_path": "lw_paving_stones_absorbed_Wm2",
        "paving_stone_pedestrian": "lw_paving_stones_absorbed_Wm2",
        "generic_ground": "lw_generic_ground_absorbed_Wm2",
        "ground": "lw_generic_ground_absorbed_Wm2",
        "unpaved_path": "lw_generic_ground_absorbed_Wm2",
        "grass_or_low_vegetation": "lw_grass_absorbed_Wm2",
        "grass_lawn": "lw_grass_absorbed_Wm2",
        "grass": "lw_grass_absorbed_Wm2",
        "wall": "lw_building_wall_absorbed_Wm2",
        "roof": "lw_roof_absorbed_Wm2",
        "tree_canopy": "lw_tree_canopy_absorbed_Wm2",
        "tree_trunk": "lw_tree_trunk_absorbed_Wm2",
        "water": "lw_water_absorbed_Wm2",
        "concrete_road": "lw_concrete_road_absorbed_Wm2",
        "asphalt_parking": "lw_asphalt_parking_absorbed_Wm2",
        "concrete_parking": "lw_concrete_parking_absorbed_Wm2",
        "gravel_parking": "lw_gravel_parking_absorbed_Wm2",
        "artificial_turf": "lw_artificial_turf_absorbed_Wm2",
        "sports_surface": "lw_sports_surface_absorbed_Wm2",
        "playground_surface": "lw_playground_surface_absorbed_Wm2",
        "bare_ground": "lw_bare_ground_absorbed_Wm2",
        "pedestrian_crossing": "lw_pedestrian_crossing_absorbed_Wm2",
        "pedestrian_plaza": "lw_pedestrian_plaza_absorbed_Wm2",
    }
    return mapping.get(name, "lw_other_surface_absorbed_Wm2")


def mrt_from_absorbed_flux_c(total_flux_wm2: Any, person_emissivity: float,
                             sigma: float) -> np.ndarray:
    """Apply the nonlinear Stefan-Boltzmann inversion to absorbed flux."""
    total = np.asarray(total_flux_wm2, dtype=float)
    if person_emissivity <= 0 or sigma <= 0 or np.any(total < 0):
        raise ValueError("MRT inversion requires non-negative flux and positive constants")
    return np.power(total / (person_emissivity * sigma), 0.25) - 273.15


def _allclose_or_raise(actual: np.ndarray, expected: np.ndarray, name: str,
                       absolute_tolerance: float, relative_tolerance: float) -> None:
    if not np.allclose(actual, expected, atol=absolute_tolerance,
                       rtol=relative_tolerance, equal_nan=False):
        error = np.abs(np.asarray(actual) - np.asarray(expected))
        raise ValueError(
            f"absorbed radiant-flux conservation failed for {name}: "
            f"maximum absolute residual {float(np.max(error)):.6g} W m^-2")


def validate_contribution_arrays(contributions: Mapping[str, Any],
                                 *, absolute_tolerance: float = 1.0e-4,
                                 relative_tolerance: float = 1.0e-6,
                                 expected_mrt_c: Any | None = None,
                                 person_emissivity: float = 0.97,
                                 sigma: float = 5.670374419e-8,
                                 mrt_tolerance_c: float = 2.0e-4) -> dict[str, float]:
    """Validate primary, material, total, and optional MRT closure."""
    missing = [key for key in PRIMARY_COLUMNS + TOTAL_COLUMNS if key not in contributions]
    if missing:
        raise ValueError(f"radiant-flux record is missing required fields: {missing}")
    arrays = {key: np.asarray(value, dtype=float) for key, value in contributions.items()}
    shape = arrays[PRIMARY_COLUMNS[0]].shape
    for key, value in arrays.items():
        if value.shape != shape:
            raise ValueError(f"contribution {key} has shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"contribution {key} contains non-finite values")
        if np.any(value < -absolute_tolerance):
            raise ValueError(f"contribution {key} contains negative absorbed flux")
    sw = arrays[PRIMARY_COLUMNS[0]] + arrays[PRIMARY_COLUMNS[1]] + arrays[PRIMARY_COLUMNS[2]]
    lw = arrays[PRIMARY_COLUMNS[3]] + arrays[PRIMARY_COLUMNS[4]]
    total = sw + lw
    _allclose_or_raise(arrays["sw_total_absorbed_Wm2"], sw, "shortwave total",
                       absolute_tolerance, relative_tolerance)
    _allclose_or_raise(arrays["lw_total_absorbed_Wm2"], lw, "longwave total",
                       absolute_tolerance, relative_tolerance)
    _allclose_or_raise(arrays["total_absorbed_radiant_flux_Wm2"], total, "total flux",
                       absolute_tolerance, relative_tolerance)
    sw_sources = [arrays[key] for key in SW_SOURCE_COLUMNS if key in arrays]
    if sw_sources:
        _allclose_or_raise(arrays[PRIMARY_COLUMNS[2]], np.sum(sw_sources, axis=0),
                           "classified reflected shortwave",
                           absolute_tolerance, relative_tolerance)
    lw_sources = [arrays[key] for key in LW_SOURCE_COLUMNS if key in arrays]
    if lw_sources:
        _allclose_or_raise(arrays[PRIMARY_COLUMNS[4]], np.sum(lw_sources, axis=0),
                           "classified surface longwave",
                           absolute_tolerance, relative_tolerance)
    mrt_error = 0.0
    if expected_mrt_c is not None:
        derived = mrt_from_absorbed_flux_c(total, person_emissivity, sigma)
        expected = np.asarray(expected_mrt_c, dtype=float)
        mrt_error = float(np.max(np.abs(derived - expected)))
        if not np.allclose(derived, expected, atol=mrt_tolerance_c, rtol=0.0):
            raise ValueError(
                "MRT reconstructed from recorded absorbed flux differs from the "
                f"authoritative MRT by up to {mrt_error:.6g} degC")
    return {
        "maximum_sw_closure_error_Wm2": float(np.max(np.abs(arrays["sw_total_absorbed_Wm2"] - sw))),
        "maximum_lw_closure_error_Wm2": float(np.max(np.abs(arrays["lw_total_absorbed_Wm2"] - lw))),
        "maximum_total_closure_error_Wm2": float(np.max(np.abs(arrays["total_absorbed_radiant_flux_Wm2"] - total))),
        "maximum_mrt_reconstruction_error_C": mrt_error,
    }


def sorted_visualization_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable descending exposure copy without mutating route order."""
    original_index = frame["original_route_index"].to_numpy(copy=True)
    ranked = frame.sort_values("total_absorbed_radiant_flux_Wm2",
                               ascending=False, kind="mergesort").copy()
    ranked.insert(0, "sorted_receptor_rank", np.arange(1, len(ranked) + 1))
    if not np.array_equal(frame["original_route_index"].to_numpy(), original_index):
        raise RuntimeError("route order changed while preparing ranked visualization")
    return ranked.reset_index(drop=True)


def independently_ranked_columns(frame: pd.DataFrame,
                                 columns: Iterable[str], *,
                                 ascending: bool = False) -> pd.DataFrame:
    """Sort every flux series independently in the requested direction.

    The returned rows are contribution-specific ranks, not simultaneous route
    receptors. The input remains in authoritative walking order.
    """
    original_index = frame["original_route_index"].to_numpy(copy=True)
    ranked = pd.DataFrame(index=np.arange(len(frame)))
    for key in columns:
        values = np.sort(frame[key].to_numpy(dtype=float))
        ranked[key] = values if ascending else values[::-1]
    if not np.array_equal(frame["original_route_index"].to_numpy(), original_index):
        raise RuntimeError("route order changed while ranking contribution curves")
    return ranked


def sample_route_contribution_matrices(
        matrices: Mapping[str, Any], time_hours: np.ndarray,
        arrival_hours: np.ndarray, point_indices: np.ndarray,
        authoritative_mrt_c: np.ndarray, *, person_emissivity: float,
        sigma: float, validation: Mapping[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Interpolate contribution matrices without changing route-point order.

    Existing route stages interpolate MRT in temperature space. Because the
    Stefan-Boltzmann transform is nonlinear, independently interpolating flux
    components would otherwise differ slightly from that authoritative route
    MRT. A single positive closure scale is therefore applied at each route
    point. It preserves component fractions and makes the additional record
    reconstruct the exact MRT already used by UTCI/JOS-3.
    """
    time_hours = np.asarray(time_hours, dtype=float)
    arrival_hours = np.asarray(arrival_hours, dtype=float)
    point_indices = np.asarray(point_indices, dtype=int)
    authoritative_mrt_c = np.asarray(authoritative_mrt_c, dtype=float)
    if not (len(arrival_hours) == len(point_indices) == len(authoritative_mrt_c)):
        raise ValueError("route contribution sampling arrays must have equal length")
    sampled: dict[str, np.ndarray] = {}
    target_hours = np.mod(arrival_hours, 24.0)
    for key, matrix in matrices.items():
        array = np.asarray(matrix)
        if array.ndim != 2 or array.shape[0] != len(time_hours):
            raise ValueError(f"contribution matrix {key} has incompatible shape {array.shape}")
        if np.any(point_indices < 0) or np.any(point_indices >= array.shape[1]):
            raise ValueError("route point index is outside contribution matrix")
        values = np.empty(len(point_indices), dtype=float)
        for index, (hour, point_index) in enumerate(zip(target_hours, point_indices)):
            values[index] = np.interp(
                hour, time_hours, array[:, point_index], period=24.0)
        sampled[key] = values
    raw_total = sampled["total_absorbed_radiant_flux_Wm2"]
    target_total = person_emissivity * sigma * (authoritative_mrt_c + 273.15) ** 4
    if np.any(raw_total <= 0) or np.any(target_total <= 0):
        raise ValueError("route flux closure requires positive total radiant flux")
    scale = target_total / raw_total
    for key in sampled:
        sampled[key] = sampled[key] * scale
    validate_contribution_arrays(
        sampled,
        absolute_tolerance=float(validation["absolute_tolerance_Wm2"]),
        relative_tolerance=float(validation["relative_tolerance"]),
        expected_mrt_c=authoritative_mrt_c,
        person_emissivity=person_emissivity,
        sigma=sigma,
        mrt_tolerance_c=float(validation["mrt_absolute_tolerance_C"]),
    )
    return sampled, scale


def aggregate_plot_categories(frame: pd.DataFrame, mode: str,
                              config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build exactly conservative primary or material-resolved plot columns."""
    if mode == "primary":
        columns = list(PRIMARY_COLUMNS)
        plot = frame[columns].copy()
        return plot, columns, []
    if mode == "surface_longwave_resolved":
        fixed = [PRIMARY_COLUMNS[0], PRIMARY_COLUMNS[1], PRIMARY_COLUMNS[2],
                 PRIMARY_COLUMNS[3]]
        source = [key for key in LW_SOURCE_COLUMNS if key in frame.columns]
    elif mode == "material_resolved":
        fixed = [PRIMARY_COLUMNS[0], PRIMARY_COLUMNS[1], PRIMARY_COLUMNS[3]]
        source = [key for key in SW_SOURCE_COLUMNS + LW_SOURCE_COLUMNS if key in frame.columns]
    else:
        raise ValueError(f"unknown contribution plot mode {mode!r}")
    means = {key: float(frame[key].mean()) for key in source}
    mean_total = float(frame["total_absorbed_radiant_flux_Wm2"].mean())
    options = config["material_resolved"]
    threshold_fraction = float(options["minimum_mean_fraction_percent"]) / 100.0
    threshold_absolute = float(options["minimum_mean_absorbed_flux_Wm2"])
    positive_sources = [key for key in source if means[key] > 1.0e-12]
    eligible = [key for key in positive_sources if means[key] >= threshold_absolute
                and means[key] >= threshold_fraction * max(mean_total, 1.0e-12)]
    eligible.sort(key=lambda key: (-means[key], key))
    n_source_slots = max(0, int(options["maximum_categories"]) - len(fixed))
    retained = eligible[:n_source_slots]
    grouped = [key for key in positive_sources if key not in retained]
    plot = frame[fixed + retained].copy()
    columns = fixed + retained
    if grouped:
        if not options["group_remaining_as_other"]:
            retained = sorted(positive_sources, key=lambda key: (-means[key], key))
            plot = frame[fixed + retained].copy()
            columns = fixed + retained
            grouped = []
        else:
            plot["Other"] = frame[grouped].sum(axis=1)
            columns.append("Other")
    stack_total = plot[columns].sum(axis=1).to_numpy()
    authoritative = frame["total_absorbed_radiant_flux_Wm2"].to_numpy()
    if not np.allclose(stack_total, authoritative, atol=1.0e-4, rtol=1.0e-6):
        raise ValueError("material-resolved aggregation does not conserve total flux")
    return plot, columns, grouped


def order_categories_by_route_mean(plot_frame: pd.DataFrame,
                                   columns: Iterable[str]) -> list[str]:
    """Order line categories from largest to smallest route-average flux."""
    return sorted(columns, key=lambda key: (-float(plot_frame[key].mean()), key))


def colors_for_order(order: list[str], mode: str,
                     config: Mapping[str, Any]) -> list[Any]:
    """Return deterministic colors in bottom-to-top legend order."""
    if mode == "fixed_physical_categories":
        fixed = config.get("fixed_category_colors", {})
        fallback = plt.get_cmap("tab20")
        canonical_order = PRIMARY_COLUMNS + SW_SOURCE_COLUMNS + LW_SOURCE_COLUMNS + ["Other"]
        return [fixed.get(
            key, fallback((canonical_order.index(key) if key in canonical_order else 0) % 20))
            for key in order]
    if mode != "ranked_emphasis":
        raise ValueError(f"unknown color mode {mode!r}")
    palette = [plt.get_cmap("tab20")(index) for index in range(20)
               if index not in {6, 7}]
    return [config["largest_contribution_color"]] + [
        palette[(index - 1) % len(palette)] for index in range(1, len(order))
    ]


def _draw_ranked_lines(ax: Any, frame: pd.DataFrame, route_id: Any,
                       config: Mapping[str, Any], color_mode: str,
                       annotation: str | None = None) -> tuple[list[str], list[str]]:
    plot, columns, grouped = aggregate_plot_categories(frame, config["plot_mode"], config)
    order = order_categories_by_route_mean(plot, columns)
    colors = colors_for_order(order, color_mode, config)
    ranked = independently_ranked_columns(
        pd.concat([frame[["original_route_index"]], plot], axis=1), order)
    x = np.arange(1, len(frame) + 1)
    for key, color in zip(order, colors):
        ax.plot(x, ranked[key].to_numpy(dtype=float), color=color, linewidth=1.25,
                label=DISPLAY_LABELS.get(key, key))
    total = np.sort(
        frame["total_absorbed_radiant_flux_Wm2"].to_numpy(dtype=float))[::-1]
    ax.plot(x, total, color="black", linewidth=1.5, label="Total absorbed flux")
    ax.set_title(f"Route {route_id}: Independently ranked absorbed radiant-flux contributions")
    ax.set_ylabel(r"Absorbed radiant flux (W m$^{-2}$)")
    ax.set_xlim(1, max(1, len(ranked)))
    if annotation:
        ax.text(0.01, 0.98, annotation, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    return order, grouped


def plot_ranked_radiant_flux_contributions(route_frames: Mapping[Any, pd.DataFrame],
                                            output_dir: str | Path,
                                            config: Mapping[str, Any],
                                            route_annotations: Mapping[Any, str] | None = None) -> list[Path]:
    """Save separate ranked-emphasis and combined fixed-color route figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    export = config["export"]
    route_annotations = route_annotations or {}

    def save(fig: Any, stem: Path) -> None:
        fig.tight_layout()
        if export["pdf"]:
            path = stem.with_suffix(".pdf")
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)
        if export["png"]:
            path = stem.with_suffix(".png")
            fig.savefig(path, dpi=int(export["dpi"]), bbox_inches="tight")
            saved.append(path)

    if config["separate_figure_per_route"]:
        for route_id, frame in route_frames.items():
            fig, ax = plt.subplots(figsize=(10.0, 5.2))
            _draw_ranked_lines(ax, frame, route_id, config,
                               config["separate_color_mode"],
                               route_annotations.get(route_id))
            ax.set_xlabel("Contribution-specific receptor rank (each curve independently sorted high to low)")
            save(fig, output / f"route_{route_id}_ranked_radiant_flux_contributions")
            plt.close(fig)

    if config["combined_multi_panel_figure"] and route_frames:
        n = len(route_frames)
        fig, axes = plt.subplots(n, 1, figsize=(10.5, 4.0 * n),
                                 sharey=bool(config["shared_y_axis"]), squeeze=False)
        for ax, (route_id, frame) in zip(axes[:, 0], route_frames.items()):
            _draw_ranked_lines(ax, frame, route_id, config,
                               config["combined_color_mode"],
                               route_annotations.get(route_id))
        axes[-1, 0].set_xlabel(
            "Contribution-specific receptor rank (each curve independently sorted high to low)")
        save(fig, output / "all_routes_ranked_radiant_flux_contributions")
        plt.close(fig)
    return saved


def _surface_longwave_columns(frame: pd.DataFrame,
                              config: Mapping[str, Any]) -> list[str]:
    """Return every available nonzero surface-LW source in mean-descending order."""
    threshold = float(config["surface_longwave_classification_plot"]
                      ["minimum_mean_absorbed_flux_Wm2"])
    available = [key for key in LW_SOURCE_COLUMNS if key in frame.columns]
    if not available:
        raise ValueError("surface-longwave classification plot requires source fields")
    classified_total = frame[available].sum(axis=1).to_numpy(dtype=float)
    surface_total = frame["lw_surface_total_absorbed_Wm2"].to_numpy(dtype=float)
    validation = config["validation"]
    if not np.allclose(
            classified_total, surface_total,
            atol=float(validation["absolute_tolerance_Wm2"]),
            rtol=float(validation["relative_tolerance"])):
        raise ValueError(
            "classified sources do not reconstruct total surface longwave")
    shown = [key for key in available if float(frame[key].mean()) > threshold]
    return sorted(shown, key=lambda key: (-float(frame[key].mean()), key))


def _draw_surface_longwave_classification_lines(
        ax: Any, frame: pd.DataFrame, route_id: Any,
        config: Mapping[str, Any], annotation: str | None = None) -> list[str]:
    """Draw total surface LW and independently descending material-source curves."""
    columns = _surface_longwave_columns(frame, config)
    ranked = independently_ranked_columns(
        pd.concat([frame[["original_route_index"]], frame[columns]], axis=1),
        columns, ascending=False)
    x = np.arange(1, len(frame) + 1)
    colors = colors_for_order(columns, "fixed_physical_categories", config)
    for key, color in zip(columns, colors):
        ax.plot(x, ranked[key].to_numpy(dtype=float), color=color, linewidth=1.25,
                label=DISPLAY_LABELS.get(key, key))
    surface_total = np.sort(
        frame["lw_surface_total_absorbed_Wm2"].to_numpy(dtype=float))[::-1]
    ax.plot(x, surface_total, color="black", linewidth=1.7,
            label="Total surface longwave")
    ax.set_title(
        f"Route {route_id}: Surface-longwave flux by source classification")
    ax.set_ylabel(r"Absorbed surface-longwave flux (W m$^{-2}$)")
    ax.set_xlim(1, max(1, len(frame)))
    if annotation:
        ax.text(0.01, 0.98, annotation, transform=ax.transAxes, ha="left", va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0.0, fontsize=7, framealpha=0.9)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    return columns


def plot_surface_longwave_classifications(
        route_frames: Mapping[Any, pd.DataFrame], output_dir: str | Path,
        config: Mapping[str, Any],
        route_annotations: Mapping[Any, str] | None = None,
        ) -> tuple[list[Path], dict[str, list[str]]]:
    """Save separate and combined high-to-low surface-LW classification plots."""
    options = config["surface_longwave_classification_plot"]
    if not options["enabled"]:
        return [], {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    export = config["export"]
    route_annotations = route_annotations or {}
    saved: list[Path] = []
    classes_by_route: dict[str, list[str]] = {}

    def save(fig: Any, stem: Path) -> None:
        fig.tight_layout()
        if export["pdf"]:
            path = stem.with_suffix(".pdf")
            fig.savefig(path, bbox_inches="tight")
            saved.append(path)
        if export["png"]:
            path = stem.with_suffix(".png")
            fig.savefig(path, dpi=int(export["dpi"]), bbox_inches="tight")
            saved.append(path)

    if options["separate_figure_per_route"]:
        for route_id, frame in route_frames.items():
            fig, ax = plt.subplots(figsize=(10.0, 5.2))
            classes_by_route[str(route_id)] = _draw_surface_longwave_classification_lines(
                ax, frame, route_id, config, route_annotations.get(route_id))
            ax.set_xlabel(
                "Classification-specific receptor rank "
                "(each curve independently sorted high to low)")
            save(fig, output / f"route_{route_id}_surface_longwave_by_class_descending")
            plt.close(fig)

    if options["combined_multi_panel_figure"] and route_frames:
        n = len(route_frames)
        fig, axes = plt.subplots(
            n, 1, figsize=(10.5, 4.0 * n),
            sharey=bool(options["shared_y_axis"]), squeeze=False)
        for ax, (route_id, frame) in zip(axes[:, 0], route_frames.items()):
            classes_by_route[str(route_id)] = _draw_surface_longwave_classification_lines(
                ax, frame, route_id, config, route_annotations.get(route_id))
        axes[-1, 0].set_xlabel(
            "Classification-specific receptor rank "
            "(each curve independently sorted high to low)")
        save(fig, output / "all_routes_surface_longwave_by_class_descending")
        plt.close(fig)
    return saved, classes_by_route


def route_contribution_summary(route_frames: Mapping[Any, pd.DataFrame],
                               config: Mapping[str, Any]) -> pd.DataFrame:
    """Create deterministic primary and material-source route summaries."""
    rows: list[dict[str, Any]] = []
    for route_id, frame in route_frames.items():
        total_mean = float(frame["total_absorbed_radiant_flux_Wm2"].mean())
        available_sw = [key for key in SW_SOURCE_COLUMNS if key in frame.columns]
        available_lw = [key for key in LW_SOURCE_COLUMNS if key in frame.columns]
        modes = [("primary", PRIMARY_COLUMNS, set(), set(PRIMARY_COLUMNS))]
        if available_lw:
            _, retained_lw, grouped_lw = aggregate_plot_categories(
                frame, "surface_longwave_resolved", config)
            modes.append((
                "surface_longwave_resolved",
                [PRIMARY_COLUMNS[0], PRIMARY_COLUMNS[1], PRIMARY_COLUMNS[2],
                 PRIMARY_COLUMNS[3]] + available_lw,
                set(grouped_lw), set(retained_lw)))
        available_sources = available_sw + available_lw
        if available_sources:
            _, retained_material, grouped_material = aggregate_plot_categories(
                frame, "material_resolved", config)
            modes.append((
                "material_resolved",
                [PRIMARY_COLUMNS[0], PRIMARY_COLUMNS[1], PRIMARY_COLUMNS[3]]
                + available_sources,
                set(grouped_material), set(retained_material)))
        for aggregation_mode, columns, grouped_set, retained_set in modes:
            means = {key: float(frame[key].mean()) for key in columns}
            order = sorted(columns, key=lambda key: (-means[key], key))
            rank = {key: index + 1 for index, key in enumerate(order)}
            for key in columns:
                values = frame[key].to_numpy(dtype=float)
                rows.append({
                    "route_id": route_id,
                    "aggregation_mode": aggregation_mode,
                    "contribution_category": key,
                    "contribution_label": DISPLAY_LABELS.get(key, key),
                    "mean_absorbed_flux_Wm2": float(np.mean(values)),
                    "median_absorbed_flux_Wm2": float(np.median(values)),
                    "maximum_absorbed_flux_Wm2": float(np.max(values)),
                    "minimum_absorbed_flux_Wm2": float(np.min(values)),
                    "standard_deviation_Wm2": float(np.std(values)),
                    "mean_fraction_percent": (100.0 * float(np.mean(values)) /
                                               total_mean if total_mean > 0 else np.nan),
                    "plot_order_rank": rank[key],
                    # Retained for compatibility with the first contribution
                    # table schema; figures now use independent line curves.
                    "stack_rank": rank[key],
                    "grouped_into_other": key in grouped_set,
                    "retained_plot_category": key in retained_set,
                    "retained_material_plot_category": key in retained_set,
                    "number_of_points": len(frame),
                })
    return pd.DataFrame(rows).sort_values(
        ["route_id", "aggregation_mode", "stack_rank"], kind="mergesort")


def write_contribution_metadata(path: str | Path, metadata: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
