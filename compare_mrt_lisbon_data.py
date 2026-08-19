#!/usr/bin/env python3
"""Standalone validation of TREC-Route MRT against Lisbon observations.

This utility is deliberately outside ``start.sh`` and ``pipeline_ui.py``.
Lisbon validation is problem-specific: it joins the experimental mobile MRT
records in ``input/lisbonN/measurements`` to the corresponding TREC-Route
route-point results in ``run_output/lisbonN``.

Pairing is exact by ``route_id`` and acquisition ``seq``.  Projected
coordinates and recorded local clock times are checked after the join, so the
program never silently substitutes a nearest spatial or temporal sample.
Positive residual and bias mean TREC-Route predicts a higher MRT than measured.

Examples
--------
Compare every Lisbon case whose simulation output is available::

    python3 compare_mrt_lisbon_data.py

This produces separate case/route figures.  Cross-case aggregation is disabled
by default because the six mobile routes represent different sites, dates, and
exposure sequences.  It can be requested explicitly for a diagnostic export::

    python3 compare_mrt_lisbon_data.py \
        --aggregate-output-dir run_output/lisbon_mrt_validation

Require all six cases to be available::

    python3 compare_mrt_lisbon_data.py --require-all

Compare selected cases::

    python3 compare_mrt_lisbon_data.py --case lisbon1 --case lisbon3

Where four-component radiometer fields are available, the script also creates
a clearly labelled two-hemisphere absorbed-flux diagnostic.  This is more
physically informative than trying to decompose MRT into additive temperature
terms, but it is not a six-directional human-radiation measurement.

The script does not alter model results, measurement files, the generalized
pipeline, or any physical parameter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ROOT / "input"
DEFAULT_OUTPUT_ROOT = ROOT / "run_output"
MEASUREMENT_RELATIVE = Path("measurements/experimental_measurements_all_routes.csv")
PREDICTION_RELATIVE = Path("viz/route_utci/routes_points.csv")
MRT_TIMES_RELATIVE = Path("mrt_facet_out/times.csv")
CASE_OUTPUT_RELATIVE = Path("validation/mrt_lisbon")
FLUX_RELATIVE = Path("viz/route_utci/radiant_flux_contributions")

REQUIRED_MEASURED = {
    "route_id", "seq", "measured_mrt_C", "timestamp_utc_refined",
    "timestamp_local_refined", "x_proj_m", "y_proj_m", "latitude",
    "longitude", "SWin", "AirTemp", "HRel", "WS",
}
REQUIRED_PREDICTED = {
    "route_id", "seq", "x_proj_m", "y_proj_m", "cumdist_m",
    "arrival_hour", "tmrt_c", "utci_c", "ta_c", "rh_pct", "wind_ms",
}
FLUX_COLUMNS = {
    "route_id", "point_id", "original_route_index",
    "sw_total_absorbed_Wm2", "lw_sky_absorbed_Wm2",
    "lw_surface_total_absorbed_Wm2", "lw_total_absorbed_Wm2",
    "total_absorbed_radiant_flux_Wm2",
}
PERSON_SHORTWAVE_ABSORPTIVITY = 0.70
PERSON_LONGWAVE_EMISSIVITY = 0.97
TWO_HEMISPHERE_SW_EFFECTIVE_FACTOR = 0.25
TWO_HEMISPHERE_LW_VIEW_FACTOR = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TREC-Route Lisbon MRT with experimental mobile MRT")
    parser.add_argument(
        "--case", action="append", default=None,
        help="Case folder name, repeatable (default: discover input/lisbon*)")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--aggregate-output-dir", type=Path,
        default=None,
        help="Optional cross-case diagnostic output. Omit to keep every route separate.")
    parser.add_argument(
        "--require-all", action="store_true",
        help="Fail if any selected/discovered case lacks simulation results")
    parser.add_argument(
        "--coordinate-tolerance-m", type=float, default=0.01,
        help="Maximum coordinate discrepancy after route_id/seq join")
    parser.add_argument(
        "--clock-tolerance-s", type=float, default=1.0,
        help="Maximum difference between recorded and simulated local clock")
    parser.add_argument(
        "--sunlit-threshold-wm2", type=float, default=120.0,
        help="Measured incoming-shortwave threshold for day sun/shade diagnostics")
    parser.add_argument(
        "--minimum-subgroup-n", type=int, default=3,
        help="Do not report metrics for smaller subgroups")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def case_sort_key(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z_-]+)(\d+)", name)
    return (match.group(1).lower(), int(match.group(2))) if match else (name.lower(), 0)


def discover_cases(input_root: Path, selected: list[str] | None) -> list[str]:
    """Return deterministic Lisbon case names with valid manifests."""
    if selected:
        names = sorted(set(selected), key=case_sort_key)
    else:
        names = sorted(
            (path.name for path in input_root.glob("lisbon*")
             if path.is_dir() and (path / "case.json").is_file()),
            key=case_sort_key,
        )
    if not names:
        raise FileNotFoundError(f"no Lisbon cases found below {input_root}")
    for name in names:
        if not (input_root / name / "case.json").is_file():
            raise FileNotFoundError(f"selected case has no manifest: {input_root / name}")
    return names


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def finite_numeric(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{context}: required numeric values contain NaN or infinity")


def route_metadata(case_dir: Path) -> dict[int, dict]:
    """Load route names and measurement provenance from route sidecars."""
    output: dict[int, dict] = {}
    for path in sorted((case_dir / "routes").glob("route_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        route_id = int(data["route_id"])
        if route_id in output:
            raise ValueError(f"duplicate route ID {route_id} in {case_dir / 'routes'}")
        output[route_id] = data
    if not output:
        raise FileNotFoundError(f"no route metadata found in {case_dir / 'routes'}")
    return output


def simulation_date_and_timezone(times_path: Path) -> tuple[str, str]:
    """Read the authoritative simulated civil date from the MRT time table."""
    times = pd.read_csv(times_path, usecols=["time"])
    parsed = pd.to_datetime(times["time"], errors="coerce", format="mixed")
    if parsed.isna().any() or parsed.empty:
        raise ValueError(f"{times_path}: invalid or empty time column")
    dates = sorted({stamp.date().isoformat() for stamp in parsed})
    if len(dates) != 1:
        raise ValueError(f"{times_path}: expected one simulation date, found {dates}")
    timezone_name = str(parsed.iloc[0].tzinfo) if parsed.iloc[0].tzinfo else "naive"
    return dates[0], timezone_name


def decimal_local_hour(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("recorded local timestamps contain invalid values")
    return (
        parsed.dt.hour.to_numpy(float)
        + parsed.dt.minute.to_numpy(float) / 60.0
        + parsed.dt.second.to_numpy(float) / 3600.0
        + parsed.dt.microsecond.to_numpy(float) / 3.6e9
    )


def periodic_interp(source_hour: np.ndarray, values: np.ndarray,
                    target_hour: np.ndarray) -> np.ndarray:
    order = np.argsort(source_hour)
    return np.interp(target_hour, source_hour[order], values[order], period=24.0)


def add_atmospheric_forcing_columns(frame: pd.DataFrame, times_path: Path) -> None:
    """Attach case-wide atmospheric forcing at each measured arrival time."""
    times = pd.read_csv(times_path)
    required = {"time", "GHI_Wm2"}
    require_columns(times, required, times_path)
    parsed = pd.to_datetime(times["time"], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"{times_path}: invalid forcing timestamps")
    source_hour = (parsed.dt.hour.to_numpy(float)
                   + parsed.dt.minute.to_numpy(float) / 60.0
                   + parsed.dt.second.to_numpy(float) / 3600.0)
    target_hour = frame["arrival_hour_local"].to_numpy(float) % 24.0
    frame["trec_atmospheric_ghi_wm2"] = periodic_interp(
        source_hour, times["GHI_Wm2"].to_numpy(float), target_hour)
    if "cloud_fraction" in times:
        frame["trec_cloud_fraction"] = periodic_interp(
            source_hour, times["cloud_fraction"].to_numpy(float), target_hour)
    else:
        frame["trec_cloud_fraction"] = np.nan
    if "LWin_Wm2" in times:
        values = times["LWin_Wm2"].to_numpy(float)
        frame["trec_supplied_atmospheric_lwin_wm2"] = (
            periodic_interp(source_hour, values, target_hour)
            if np.isfinite(values).all() else np.nan)
    else:
        frame["trec_supplied_atmospheric_lwin_wm2"] = np.nan


def add_trec_flux_columns(frame: pd.DataFrame, result_dir: Path) -> None:
    """Join receptor-level absorbed flux without altering route-point order."""
    blocks: list[pd.DataFrame] = []
    for route_id in sorted(frame["route_id"].unique().astype(int)):
        path = result_dir / FLUX_RELATIVE / f"route_{route_id}_radiant_flux_contributions.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"absorbed-flux route table missing: {path}; enable contribution recording")
        flux = pd.read_csv(path)
        require_columns(flux, FLUX_COLUMNS, path)
        if flux.duplicated(["route_id", "point_id"]).any():
            raise ValueError(f"{path}: duplicate route_id/point_id")
        if not np.array_equal(flux["point_id"].to_numpy(int),
                              flux["original_route_index"].to_numpy(int)):
            raise ValueError(f"{path}: plotting/export order no longer matches route order")
        blocks.append(flux.loc[:, sorted(FLUX_COLUMNS)])
    flux_all = pd.concat(blocks, ignore_index=True).rename(columns={"point_id": "seq"})
    before = frame[["route_id", "seq"]].copy()
    merged = frame.merge(flux_all, on=["route_id", "seq"], how="left",
                         validate="one_to_one", sort=False)
    if merged[list(FLUX_COLUMNS - {"route_id", "point_id", "original_route_index"})].isna().any().any():
        raise ValueError("absorbed-flux join left unmatched route points")
    if not before.equals(merged[["route_id", "seq"]]):
        raise RuntimeError("absorbed-flux join changed route-point ordering")
    for column in merged.columns:
        if column not in frame.columns:
            frame[column] = merged[column].to_numpy()


def add_measured_flux_diagnostics(frame: pd.DataFrame) -> None:
    """Build approximate absorbed fluxes from the four-component radiometer.

    The shortwave factor is a globe-like effective projected-area diagnostic;
    the longwave factor assigns half the body view to each measured hemisphere.
    These terms are useful for mechanism diagnosis but are not an exact
    six-directional standing-person reconstruction.
    """
    required = ("measured_swin_wm2", "measured_swout_wm2",
                "measured_lwin_wm2", "measured_lwout_wm2")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"four-component flux diagnostic unavailable; missing {missing}")
    frame["measured_sw_absorbed_diagnostic_wm2"] = (
        PERSON_SHORTWAVE_ABSORPTIVITY * TWO_HEMISPHERE_SW_EFFECTIVE_FACTOR
        * (frame["measured_swin_wm2"] + frame["measured_swout_wm2"]))
    frame["measured_lw_sky_absorbed_proxy_wm2"] = (
        PERSON_LONGWAVE_EMISSIVITY * TWO_HEMISPHERE_LW_VIEW_FACTOR
        * frame["measured_lwin_wm2"])
    frame["measured_lw_surface_absorbed_proxy_wm2"] = (
        PERSON_LONGWAVE_EMISSIVITY * TWO_HEMISPHERE_LW_VIEW_FACTOR
        * frame["measured_lwout_wm2"])
    frame["measured_lw_absorbed_diagnostic_wm2"] = (
        frame["measured_lw_sky_absorbed_proxy_wm2"]
        + frame["measured_lw_surface_absorbed_proxy_wm2"])
    frame["measured_total_absorbed_diagnostic_wm2"] = (
        frame["measured_sw_absorbed_diagnostic_wm2"]
        + frame["measured_lw_absorbed_diagnostic_wm2"])
    sigma = 5.670374419e-8
    frame["measured_flux_equivalent_mrt_c"] = (
        (frame["measured_total_absorbed_diagnostic_wm2"]
         / (PERSON_LONGWAVE_EMISSIVITY * sigma)) ** 0.25 - 273.15)


def load_case_pairs(
    case_name: str,
    input_root: Path,
    output_root: Path,
    coordinate_tolerance_m: float,
    clock_tolerance_s: float,
    sunlit_threshold_wm2: float,
) -> tuple[pd.DataFrame, dict]:
    """Load and rigorously pair one case's observations and predictions."""
    case_dir = input_root / case_name
    result_dir = output_root / case_name
    measurement_path = case_dir / MEASUREMENT_RELATIVE
    prediction_path = result_dir / PREDICTION_RELATIVE
    times_path = result_dir / MRT_TIMES_RELATIVE
    for path in (measurement_path, prediction_path, times_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    measured = pd.read_csv(measurement_path)
    predicted = pd.read_csv(prediction_path)
    require_columns(measured, REQUIRED_MEASURED, measurement_path)
    require_columns(predicted, REQUIRED_PREDICTED, prediction_path)
    finite_numeric(
        measured,
        ["route_id", "seq", "measured_mrt_C", "x_proj_m", "y_proj_m",
         "latitude", "longitude", "SWin", "AirTemp", "HRel", "WS"],
        str(measurement_path),
    )
    finite_numeric(predicted, REQUIRED_PREDICTED, str(prediction_path))
    for label, frame in (("measurement", measured), ("prediction", predicted)):
        if frame.duplicated(["route_id", "seq"]).any():
            duplicates = frame.loc[
                frame.duplicated(["route_id", "seq"], keep=False), ["route_id", "seq"]]
            raise ValueError(f"{label} has duplicate route_id/seq rows: "
                             f"{duplicates.head().to_dict('records')}")

    observed_columns = [
        "route_id", "seq", "measured_mrt_C", "timestamp_utc_refined",
        "timestamp_local_refined", "x_proj_m", "y_proj_m", "latitude",
        "longitude", "SWin", "AirTemp", "HRel", "WS",
    ]
    optional_measured = [
        "TIMESTAMP", "RECORD", "BGTemp_C", "UTCI_approx", "DewP", "WDir",
        "BPress", "NR", "SWout", "LWin", "LWout", "SWnet", "LWnet",
        "GPS_Alt",
    ]
    observed_columns += [name for name in optional_measured if name in measured.columns]
    paired = predicted.merge(
        measured.loc[:, observed_columns], on=["route_id", "seq"], how="outer",
        suffixes=("_predicted", "_measured"), indicator=True,
        validate="one_to_one",
    )
    if not (paired["_merge"] == "both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"{case_name}: measurement/prediction pairing is incomplete: {counts}")
    paired = paired.drop(columns="_merge")

    coordinate_error = np.hypot(
        paired["x_proj_m_predicted"] - paired["x_proj_m_measured"],
        paired["y_proj_m_predicted"] - paired["y_proj_m_measured"],
    )
    max_coordinate_error = float(coordinate_error.max())
    if max_coordinate_error > coordinate_tolerance_m:
        raise ValueError(
            f"{case_name}: maximum paired coordinate error {max_coordinate_error:.6f} m "
            f"exceeds tolerance {coordinate_tolerance_m:.6f} m")

    observed_hour = decimal_local_hour(paired["timestamp_local_refined"])
    predicted_hour = pd.to_numeric(paired["arrival_hour"], errors="raise").to_numpy(float) % 24.0
    clock_delta_s = np.abs(predicted_hour - observed_hour) * 3600.0
    clock_delta_s = np.minimum(clock_delta_s, 86400.0 - clock_delta_s)
    max_clock_error = float(clock_delta_s.max())
    if max_clock_error > clock_tolerance_s:
        raise ValueError(
            f"{case_name}: maximum paired clock error {max_clock_error:.3f} s "
            f"exceeds tolerance {clock_tolerance_s:.3f} s")

    simulation_date, simulation_timezone = simulation_date_and_timezone(times_path)
    metadata = route_metadata(case_dir)
    missing_route_metadata = sorted(set(paired["route_id"].astype(int)) - set(metadata))
    if missing_route_metadata:
        raise ValueError(f"{case_name}: missing metadata for routes {missing_route_metadata}")
    paired["case_id"] = case_name
    paired["route_name"] = paired["route_id"].map(
        {key: value["name"] for key, value in metadata.items()})
    paired["period"] = paired["route_id"].map({
        key: value.get("measurement_provenance", {}).get(
            "period", "night" if "night" in value["name"].lower() else "day")
        for key, value in metadata.items()
    })
    observed_timestamp = pd.to_datetime(
        paired["timestamp_local_refined"], errors="coerce", format="mixed")
    paired["observation_date"] = observed_timestamp.map(lambda stamp: stamp.date().isoformat())
    paired["simulation_date"] = simulation_date
    paired["simulation_date_matches_observation"] = (
        paired["observation_date"] == paired["simulation_date"])
    paired["coordinate_pairing_error_m"] = coordinate_error
    paired["clock_pairing_error_s"] = clock_delta_s
    paired["trec_route_mrt_c"] = pd.to_numeric(paired["tmrt_c"], errors="raise")
    paired["measured_mrt_c"] = pd.to_numeric(paired["measured_mrt_C"], errors="raise")
    paired["residual_c"] = paired["trec_route_mrt_c"] - paired["measured_mrt_c"]
    paired["absolute_residual_c"] = paired["residual_c"].abs()
    paired["measured_swin_wm2"] = pd.to_numeric(paired["SWin"], errors="raise")
    paired["exposure_class"] = np.where(
        paired["period"].eq("night"), "night",
        np.where(paired["measured_swin_wm2"] >= sunlit_threshold_wm2,
                 "day_sunlit", "day_shaded"),
    )

    output = pd.DataFrame({
        "case_id": paired["case_id"],
        "route_id": paired["route_id"].astype(int),
        "route_name": paired["route_name"],
        "period": paired["period"],
        "seq": paired["seq"].astype(int),
        "timestamp_utc": paired["timestamp_utc_refined"],
        "timestamp_local": paired["timestamp_local_refined"],
        "observation_date": paired["observation_date"],
        "simulation_date": paired["simulation_date"],
        "simulation_date_matches_observation": paired[
            "simulation_date_matches_observation"],
        "arrival_hour_local": paired["arrival_hour"],
        "latitude": paired["latitude"],
        "longitude": paired["longitude"],
        "x_projected_m": paired["x_proj_m_measured"],
        "y_projected_m": paired["y_proj_m_measured"],
        "distance_along_route_m": paired["cumdist_m"],
        "coordinate_pairing_error_m": paired["coordinate_pairing_error_m"],
        "clock_pairing_error_s": paired["clock_pairing_error_s"],
        "measured_mrt_c": paired["measured_mrt_c"],
        "trec_route_mrt_c": paired["trec_route_mrt_c"],
        "residual_c": paired["residual_c"],
        "absolute_residual_c": paired["absolute_residual_c"],
        "measured_swin_wm2": paired["measured_swin_wm2"],
        "exposure_class": paired["exposure_class"],
        "measured_air_temperature_c": paired["AirTemp"],
        "trec_route_air_temperature_c": paired["ta_c"],
        "measured_rh_pct": paired["HRel"],
        "trec_route_rh_pct": paired["rh_pct"],
        "measured_wind_ms": paired["WS"],
        "trec_route_wind_ms": paired["wind_ms"],
        "trec_route_utci_c": paired["utci_c"],
    })
    for source, target in (
        ("BGTemp_C", "measured_black_globe_temperature_c"),
        ("SWout", "measured_swout_wm2"), ("LWin", "measured_lwin_wm2"),
        ("LWout", "measured_lwout_wm2"), ("NR", "measured_net_radiation_wm2"),
    ):
        if source in paired:
            output[target] = paired[source]
    if not np.isfinite(output[["measured_mrt_c", "trec_route_mrt_c",
                               "residual_c"]].to_numpy(float)).all():
        raise ValueError(f"{case_name}: paired MRT values are not finite")
    add_atmospheric_forcing_columns(output, times_path)
    add_trec_flux_columns(output, result_dir)
    add_measured_flux_diagnostics(output)

    audit = {
        "case_id": case_name,
        "site_name": manifest.get("site_name", case_name),
        "n_pairs": int(len(output)),
        "route_ids": sorted(output["route_id"].unique().astype(int).tolist()),
        "simulation_date": simulation_date,
        "simulation_timezone": simulation_timezone,
        "observation_dates": sorted(output["observation_date"].unique().tolist()),
        "date_aligned_pairs": int(output["simulation_date_matches_observation"].sum()),
        "date_mismatched_pairs": int((~output["simulation_date_matches_observation"]).sum()),
        "maximum_coordinate_pairing_error_m": max_coordinate_error,
        "maximum_clock_pairing_error_s": max_clock_error,
        "sunlit_threshold_wm2": sunlit_threshold_wm2,
        "residual_convention": "TREC-Route minus measured MRT",
        "measurement_file": str(measurement_path.resolve()),
        "prediction_file": str(prediction_path.resolve()),
        "mrt_times_file": str(times_path.resolve()),
        "absorbed_flux_convention": (
            "Two-hemisphere diagnostic: SW=0.70*0.25*(SWin+SWout); "
            "LW=0.97*0.5*(LWin+LWout). Not a six-directional human measurement."),
    }
    return output.sort_values(["route_id", "seq"]).reset_index(drop=True), audit


def metric_record(
    case_id: str,
    scope: str,
    group: str,
    frame: pd.DataFrame,
    minimum_n: int,
) -> dict[str, object] | None:
    """Calculate standard prediction-versus-observation metrics."""
    measured = frame["measured_mrt_c"].to_numpy(float)
    predicted = frame["trec_route_mrt_c"].to_numpy(float)
    valid = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[valid], predicted[valid]
    n = len(measured)
    if n < minimum_n:
        return None
    residual = predicted - measured
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    observed_range = float(np.ptp(measured))
    sst = float(np.sum((measured - measured.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / sst) if sst > 0 else np.nan
    if np.std(measured) > 0 and np.std(predicted) > 0:
        pearson_r = float(stats.pearsonr(measured, predicted).statistic)
        regression = stats.linregress(measured, predicted)
        slope, intercept = float(regression.slope), float(regression.intercept)
    else:
        pearson_r = slope = intercept = np.nan
    denominator = float(np.sum(
        (np.abs(predicted - measured.mean()) + np.abs(measured - measured.mean())) ** 2))
    willmott = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else np.nan
    return {
        "case_id": case_id, "scope": scope, "group": str(group), "n": n,
        "mean_measured_mrt_c": float(measured.mean()),
        "mean_trec_route_mrt_c": float(predicted.mean()),
        "mean_bias_error_c": float(residual.mean()),
        "mae_c": float(np.abs(residual).mean()),
        "rmse_c": rmse,
        "normalized_rmse_pct_observed_range": (
            100.0 * rmse / observed_range if observed_range > 0 else np.nan),
        "maximum_absolute_error_c": float(np.abs(residual).max()),
        "r2_coefficient_of_determination": r2,
        "pearson_r": pearson_r,
        "willmott_agreement_index": willmott,
        "regression_slope": slope,
        "regression_intercept_c": intercept,
    }


def calculate_metrics(frame: pd.DataFrame, case_id: str,
                      minimum_n: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(scope: str, group: str, subset: pd.DataFrame) -> None:
        record = metric_record(case_id, scope, group, subset, minimum_n)
        if record is not None:
            records.append(record)

    add("overall", "all", frame)
    for scope, column in (
        ("route", "route_name"),
        ("period", "period"),
        ("exposure_class", "exposure_class"),
        ("date_alignment", "simulation_date_matches_observation"),
    ):
        for group, subset in frame.groupby(column, sort=True, observed=True):
            label = ("matched" if bool(group) else "mismatched") \
                if scope == "date_alignment" else str(group)
            add(scope, label, subset)
    return pd.DataFrame(records)


FLUX_COMPONENTS = {
    "shortwave_total_diagnostic": (
        "measured_sw_absorbed_diagnostic_wm2", "sw_total_absorbed_Wm2"),
    "longwave_sky_proxy": (
        "measured_lw_sky_absorbed_proxy_wm2", "lw_sky_absorbed_Wm2"),
    "longwave_surface_proxy": (
        "measured_lw_surface_absorbed_proxy_wm2",
        "lw_surface_total_absorbed_Wm2"),
    "longwave_total_diagnostic": (
        "measured_lw_absorbed_diagnostic_wm2", "lw_total_absorbed_Wm2"),
    "total_absorbed_diagnostic": (
        "measured_total_absorbed_diagnostic_wm2",
        "total_absorbed_radiant_flux_Wm2"),
}


def flux_metric_record(case_id: str, scope: str, group: str, component: str,
                       frame: pd.DataFrame, minimum_n: int) -> dict | None:
    measured_column, predicted_column = FLUX_COMPONENTS[component]
    measured = frame[measured_column].to_numpy(float)
    predicted = frame[predicted_column].to_numpy(float)
    valid = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[valid], predicted[valid]
    if len(measured) < minimum_n:
        return None
    residual = predicted - measured
    correlation = (float(stats.pearsonr(measured, predicted).statistic)
                   if np.std(measured) > 0 and np.std(predicted) > 0 else np.nan)
    return {
        "case_id": case_id, "scope": scope, "group": str(group),
        "component": component, "n": int(len(measured)),
        "mean_measured_absorbed_flux_wm2": float(measured.mean()),
        "mean_trec_route_absorbed_flux_wm2": float(predicted.mean()),
        "mean_bias_error_wm2": float(residual.mean()),
        "mae_wm2": float(np.abs(residual).mean()),
        "rmse_wm2": float(np.sqrt(np.mean(residual ** 2))),
        "maximum_absolute_error_wm2": float(np.abs(residual).max()),
        "pearson_r": correlation,
    }


def calculate_flux_metrics(frame: pd.DataFrame, case_id: str,
                           minimum_n: int) -> pd.DataFrame:
    records: list[dict] = []
    groups = [("overall", "all", frame)]
    for scope, column in (("route", "route_name"),
                          ("period", "period"),
                          ("exposure_class", "exposure_class")):
        groups.extend((scope, str(group), subset) for group, subset in
                      frame.groupby(column, sort=True, observed=True))
    for scope, group, subset in groups:
        for component in FLUX_COMPONENTS:
            record = flux_metric_record(
                case_id, scope, group, component, subset, minimum_n)
            if record is not None:
                records.append(record)
    return pd.DataFrame(records)


def solar_forcing_envelope_summary(frame: pd.DataFrame,
                                   case_id: str) -> pd.DataFrame:
    """Compare route shortwave maxima with case-wide atmospheric GHI.

    Pointwise correlation is retained only as a shade-contaminated diagnostic;
    the upper-envelope ratio is the quantity relevant to cloud attenuation.
    """
    rows = []
    for (route_id, route_name), subset in frame[frame["period"].eq("day")].groupby(
            ["route_id", "route_name"], sort=True):
        observed = subset["measured_swin_wm2"].to_numpy(float)
        forcing = subset["trec_atmospheric_ghi_wm2"].to_numpy(float)
        valid = np.isfinite(observed) & np.isfinite(forcing) & (forcing > 50)
        if not valid.any():
            continue
        observed, forcing = observed[valid], forcing[valid]
        q95_observed = float(np.quantile(observed, 0.95))
        q95_forcing = float(np.quantile(forcing, 0.95))
        rows.append({
            "case_id": case_id, "route_id": int(route_id),
            "route_name": route_name, "n_daytime_samples": int(len(observed)),
            "measured_swin_max_wm2": float(observed.max()),
            "measured_swin_q95_wm2": q95_observed,
            "trec_atmospheric_ghi_max_wm2": float(forcing.max()),
            "trec_atmospheric_ghi_q95_wm2": q95_forcing,
            "q95_mobile_to_atmospheric_ratio": (
                q95_observed / q95_forcing if q95_forcing > 0 else np.nan),
            "mean_trec_cloud_fraction": float(
                subset["trec_cloud_fraction"].mean()),
            "pointwise_pearson_r_shade_contaminated": (
                float(stats.pearsonr(observed, forcing).statistic)
                if np.std(observed) > 0 and np.std(forcing) > 0 else np.nan),
            "interpretation": (
                "Use upper-envelope ratio for atmospheric/cloud diagnosis; "
                "pointwise correlation is confounded by local shade."),
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_case_timeseries(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    routes = list(frame.groupby(["route_id", "route_name"], sort=True))
    fig, axes = plt.subplots(len(routes), 1, figsize=(10, 3.7 * len(routes)),
                             squeeze=False)
    for ax, ((route_id, route_name), route) in zip(axes[:, 0], routes):
        route = route.sort_values("seq")
        timestamp = pd.to_datetime(route["timestamp_local"], format="mixed")
        elapsed_min = (timestamp - timestamp.iloc[0]).dt.total_seconds() / 60.0
        ax.plot(elapsed_min, route["measured_mrt_c"], color="black", lw=1.5,
                label="Experiment")
        ax.plot(elapsed_min, route["trec_route_mrt_c"], color="#d62728", lw=1.4,
                label="TREC-Route")
        aligned = bool(route["simulation_date_matches_observation"].all())
        suffix = "date aligned" if aligned else "simulation/observation dates differ"
        ax.set_title(f"Route {route_id} — {route_name} ({suffix})")
        ax.set_xlabel("Elapsed device time (min)")
        ax.set_ylabel("Mean radiant temperature (°C)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{frame['case_id'].iloc[0]}: measured and TREC-Route MRT", y=1.01)
    save_figure(fig, output / "mrt_timeseries_comparison", dpi)


def safe_filename(value: str) -> str:
    """Return a deterministic filesystem-safe label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.")
    return cleaned or "route"


def plot_day_mrt_along_route(frame: pd.DataFrame, output: Path, dpi: int) -> list[Path]:
    """Plot measured and TREC-Route MRT versus distance for each daytime route.

    Each route receives its own figure.  No spatial resampling or cross-route
    alignment is performed: values are plotted at the exact experimental
    acquisition points paired by route ID and sequence.
    """
    day = frame[frame["period"].eq("day")].copy()
    if day.empty:
        print(f"  WARNING: {frame['case_id'].iloc[0]} has no daytime route to plot")
        return []
    saved: list[Path] = []
    for (route_id, route_name), route in day.groupby(
            ["route_id", "route_name"], sort=True):
        route = route.sort_values("seq")
        distance = route["distance_along_route_m"].to_numpy(float)
        measured = route["measured_mrt_c"].to_numpy(float)
        predicted = route["trec_route_mrt_c"].to_numpy(float)
        residual = predicted - measured
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        bias = float(np.mean(residual))

        fig, axes = plt.subplots(
            2, 1, figsize=(10.5, 7.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
        axes[0].plot(distance, measured, color="black", lw=1.6,
                     label="Experimentally measured MRT")
        axes[0].plot(distance, predicted, color="#d62728", lw=1.45,
                     label="TREC-Route MRT")
        axes[0].set_ylabel("Mean radiant temperature (°C)")
        axes[0].set_title(
            f"{frame['case_id'].iloc[0]}, Route {int(route_id)} "
            f"({route_name}): MRT along the measured trajectory")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=9)
        axes[0].text(
            0.01, 0.02,
            f"n = {len(route)}   MBE = {bias:+.2f} °C   "
            f"MAE = {mae:.2f} °C   RMSE = {rmse:.2f} °C",
            transform=axes[0].transAxes, fontsize=8.5, va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "0.75"},
        )

        axes[1].plot(distance, residual, color="#6a3d9a", lw=1.25,
                     label="TREC-Route − measured MRT")
        axes[1].axhline(0.0, color="black", lw=0.9, ls="--")
        axes[1].set_xlabel("Distance along measured route (m)")
        axes[1].set_ylabel("Residual (°C)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)

        stem = output / (
            f"route_{int(route_id)}_{safe_filename(route_name)}_"
            "mrt_along_route_comparison")
        save_figure(fig, stem, dpi)
        saved.append(stem)
    return saved


def plot_case_scatter(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    colors = {"day": "#e68613", "night": "#2474b5"}
    for (route_id, route_name, period), route in frame.groupby(
            ["route_id", "route_name", "period"], sort=True):
        ax.scatter(route["measured_mrt_c"], route["trec_route_mrt_c"], s=16,
                   alpha=0.55, color=colors.get(period, "#666666"),
                   label=f"Route {route_id}: {route_name}")
    values = np.r_[frame["measured_mrt_c"], frame["trec_route_mrt_c"]]
    limits = [float(np.nanmin(values) - 1.0), float(np.nanmax(values) + 1.0)]
    ax.plot(limits, limits, "k--", lw=1.1, label="1:1")
    ax.set(xlim=limits, ylim=limits,
           xlabel="Experimentally measured MRT (°C)",
           ylabel="TREC-Route MRT (°C)",
           title=f"{frame['case_id'].iloc[0]}: measured versus predicted MRT")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, output / "mrt_measured_vs_trec_route_scatter", dpi)


def plot_case_residuals(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specifications = [
        ("measured_mrt_c", "Measured MRT (°C)"),
        ("measured_swin_wm2", "Measured incoming shortwave (W m$^{-2}$)"),
        ("measured_wind_ms", "Measured wind speed (m s$^{-1}$)"),
        ("distance_along_route_m", "Distance along measured route (m)"),
    ]
    for ax, (column, label) in zip(axes.flat, specifications):
        for (route_id, route_name), route in frame.groupby(
                ["route_id", "route_name"], sort=True):
            ax.scatter(route[column], route["residual_c"], s=13, alpha=0.5,
                       label=f"Route {route_id}: {route_name}")
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel(label)
        ax.set_ylabel("TREC-Route − measured MRT (°C)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{frame['case_id'].iloc[0]}: MRT residual diagnostics")
    save_figure(fig, output / "mrt_residual_diagnostics", dpi)


def plot_day_absorbed_flux_along_route(frame: pd.DataFrame, output: Path,
                                       dpi: int) -> None:
    """Plot measured diagnostic and TREC-Route fluxes along each day route."""
    specifications = [
        ("measured_total_absorbed_diagnostic_wm2",
         "total_absorbed_radiant_flux_Wm2", "Total absorbed flux"),
        ("measured_sw_absorbed_diagnostic_wm2",
         "sw_total_absorbed_Wm2", "Absorbed shortwave diagnostic"),
        ("measured_lw_absorbed_diagnostic_wm2",
         "lw_total_absorbed_Wm2", "Absorbed longwave diagnostic"),
        ("measured_lw_surface_absorbed_proxy_wm2",
         "lw_surface_total_absorbed_Wm2", "Surface-longwave proxy"),
    ]
    day = frame[frame["period"].eq("day")]
    for (route_id, route_name), route in day.groupby(
            ["route_id", "route_name"], sort=True):
        route = route.sort_values("seq")
        distance = route["distance_along_route_m"].to_numpy(float)
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for ax, (measured_column, model_column, title) in zip(axes, specifications):
            ax.plot(distance, route[measured_column], color="black", lw=1.35,
                    label="Four-component measurement diagnostic")
            ax.plot(distance, route[model_column], color="#d62728", lw=1.25,
                    label="TREC-Route")
            residual = route[model_column].to_numpy(float) \
                - route[measured_column].to_numpy(float)
            ax.set_ylabel("Absorbed flux\n(W m$^{-2}$)")
            ax.set_title(f"{title}; mean bias {residual.mean():+.1f} W m$^{{-2}}",
                         fontsize=9.5)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Distance along measured route (m)")
        fig.suptitle(
            f"{frame['case_id'].iloc[0]}, Route {int(route_id)} ({route_name}): "
            "absorbed-flux diagnostics", y=1.002)
        save_figure(
            fig, output / (f"route_{int(route_id)}_{safe_filename(route_name)}_"
                           "absorbed_flux_along_route_comparison"), dpi)


def plot_absorbed_flux_scatter(frame: pd.DataFrame, output: Path,
                               dpi: int) -> None:
    specifications = [
        ("shortwave_total_diagnostic", "Shortwave"),
        ("longwave_sky_proxy", "Sky longwave proxy"),
        ("longwave_surface_proxy", "Surface longwave proxy"),
        ("longwave_total_diagnostic", "Total longwave"),
        ("total_absorbed_diagnostic", "Total absorbed"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (component, title) in zip(axes.flat, specifications):
        measured_column, predicted_column = FLUX_COMPONENTS[component]
        for (route_id, route_name), route in frame.groupby(
                ["route_id", "route_name"], sort=True):
            ax.scatter(route[measured_column], route[predicted_column],
                       s=12, alpha=0.45, label=f"Route {route_id}: {route_name}")
        values = np.r_[frame[measured_column], frame[predicted_column]]
        limits = [float(np.nanmin(values) - 5), float(np.nanmax(values) + 5)]
        ax.plot(limits, limits, "k--", lw=0.9)
        ax.set(xlim=limits, ylim=limits, title=title,
               xlabel="Measurement diagnostic (W m$^{-2}$)",
               ylabel="TREC-Route (W m$^{-2}$)")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle(
        f"{frame['case_id'].iloc[0]}: absorbed radiant-flux comparison\n"
        "(two-hemisphere diagnostic; not a six-directional human measurement)")
    save_figure(fig, output / "absorbed_flux_measured_vs_trec_route_scatter", dpi)


def plot_flux_residual_diagnostics(frame: pd.DataFrame, output: Path,
                                   dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    components = [
        ("sw_total_absorbed_Wm2", "measured_sw_absorbed_diagnostic_wm2", "Shortwave"),
        ("lw_total_absorbed_Wm2", "measured_lw_absorbed_diagnostic_wm2", "Longwave"),
        ("total_absorbed_radiant_flux_Wm2", "measured_total_absorbed_diagnostic_wm2",
         "Total"),
    ]
    for ax, (model, measured, title) in zip(axes, components):
        residual = frame[model] - frame[measured]
        ax.scatter(frame["measured_swin_wm2"], residual, s=12, alpha=0.45)
        ax.axhline(0, color="black", lw=0.9, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Measured incoming shortwave (W m$^{-2}$)")
        ax.set_ylabel("TREC-Route − diagnostic (W m$^{-2}$)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{frame['case_id'].iloc[0]}: absorbed-flux residuals")
    save_figure(fig, output / "absorbed_flux_residual_diagnostics", dpi)


def write_case_summary(audit: dict, metrics: pd.DataFrame,
                       flux_metrics: pd.DataFrame,
                       solar_envelope: pd.DataFrame, output: Path) -> None:
    overall = metrics[(metrics["scope"] == "overall") & (metrics["group"] == "all")].iloc[0]
    route_metrics = metrics[metrics["scope"] == "route"]
    lines = [
        f"# {audit['case_id']} experimental MRT comparison", "",
        "This standalone comparison does not modify or calibrate TREC-Route.",
        "Residual and MBE are defined as **TREC-Route minus measured MRT**.", "",
        "## Pairing audit", "",
        f"- Exact route/sequence pairs: **{audit['n_pairs']}**",
        f"- Maximum coordinate discrepancy: **{audit['maximum_coordinate_pairing_error_m']:.6f} m**",
        f"- Maximum local-clock discrepancy: **{audit['maximum_clock_pairing_error_s']:.3f} s**",
        f"- Simulation date: **{audit['simulation_date']}**",
        f"- Observation dates: **{', '.join(audit['observation_dates'])}**",
        f"- Date-aligned pairs: **{audit['date_aligned_pairs']}**",
        f"- Date-mismatched pairs: **{audit['date_mismatched_pairs']}**", "",
        "## Overall statistics", "",
        "| n | MBE (°C) | MAE (°C) | RMSE (°C) | R² | Pearson r | Willmott d |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (f"| {int(overall['n'])} | {overall['mean_bias_error_c']:.3f} | "
         f"{overall['mae_c']:.3f} | {overall['rmse_c']:.3f} | "
         f"{overall['r2_coefficient_of_determination']:.3f} | "
         f"{overall['pearson_r']:.3f} | "
         f"{overall['willmott_agreement_index']:.3f} |"), "",
        "## Route statistics", "",
        "| Route | n | measured mean (°C) | TREC-Route mean (°C) | MBE (°C) | MAE (°C) | RMSE (°C) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in route_metrics.itertuples():
        lines.append(
            f"| {row.group} | {row.n} | {row.mean_measured_mrt_c:.3f} | "
            f"{row.mean_trec_route_mrt_c:.3f} | {row.mean_bias_error_c:.3f} | "
            f"{row.mae_c:.3f} | {row.rmse_c:.3f} |")
    flux_overall = flux_metrics[
        (flux_metrics["scope"] == "overall") & (flux_metrics["group"] == "all")]
    lines.extend([
        "", "## Absorbed radiant-flux diagnostics", "",
        "The measurement comparison below uses a two-hemisphere approximation:",
        "`SWabs = 0.70 × 0.25 × (SWin + SWout)` and",
        "`LWabs = 0.97 × 0.5 × (LWin + LWout)`. It diagnoses mechanism bias",
        "but is not an exact six-directional standing-person absorbed-flux measurement.", "",
        "| Component | measured mean (W m⁻²) | TREC-Route mean (W m⁻²) | MBE (W m⁻²) | RMSE (W m⁻²) | r |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in flux_overall.itertuples():
        lines.append(
            f"| {row.component} | {row.mean_measured_absorbed_flux_wm2:.2f} | "
            f"{row.mean_trec_route_absorbed_flux_wm2:.2f} | "
            f"{row.mean_bias_error_wm2:+.2f} | {row.rmse_wm2:.2f} | "
            f"{row.pearson_r:.3f} |")
    lines.extend(["", "## Solar-forcing upper-envelope diagnostic", ""])
    if solar_envelope.empty:
        lines.append("No valid daytime upper-envelope samples were available.")
    else:
        lines.extend([
            "| Route | mobile SWin q95 (W m⁻²) | atmospheric GHI q95 (W m⁻²) | ratio | cloud fraction |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in solar_envelope.itertuples():
            lines.append(
                f"| {row.route_name} | {row.measured_swin_q95_wm2:.1f} | "
                f"{row.trec_atmospheric_ghi_q95_wm2:.1f} | "
                f"{row.q95_mobile_to_atmospheric_ratio:.3f} | "
                f"{row.mean_trec_cloud_fraction:.3f} |")
    lines.extend([
        "", "## Interpretation constraint", "",
        "The pooled overall correlation combines physically distinct daytime and nighttime",
        "temperature regimes and can therefore appear strong even when within-route agreement",
        "is weak. Route-, period-, exposure-, and date-alignment metrics should be used for",
        "scientific interpretation rather than relying on the pooled correlation alone.", "",
        "The day and night mobile surveys occurred on different dates, while the current",
        "TREC-Route result contains one 24-hour radiation simulation using the daytime date.",
        "Nighttime rows retain the measured local clock and measured meteorology but are",
        "marked as date-mismatched. Their statistics are diagnostic/provisional because the",
        "preceding surface-temperature history is not an independently simulated night-date",
        "history. The tables preserve this flag for later date-specific validation runs.", "",
        "Measured `SWin` is not used as pointwise atmospheric forcing. For a case that opts",
        "into `route_upper_envelope_cloud`, only its high open-sky-like envelope constrains",
        "one session-wide cloud attenuation. Pointwise ray-traced building/tree shade remains",
        "independent. This forcing sensitivity is recorded in radiation provenance metadata.", "",
    ])
    (output / "VALIDATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def save_case_outputs(frame: pd.DataFrame, audit: dict, metrics: pd.DataFrame,
                      flux_metrics: pd.DataFrame, solar_envelope: pd.DataFrame,
                      output: Path, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "mrt_experiment_comparison_points.csv", index=False)
    metrics.to_csv(output / "mrt_experiment_validation_metrics.csv", index=False)
    frame.to_csv(output / "radiant_flux_comparison_points.csv", index=False)
    flux_metrics.to_csv(output / "radiant_flux_validation_metrics.csv", index=False)
    solar_envelope.to_csv(
        output / "solar_forcing_upper_envelope_diagnostics.csv", index=False)
    (output / "pairing_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot_case_timeseries(frame, output, dpi)
    plot_day_mrt_along_route(frame, output, dpi)
    plot_case_scatter(frame, output, dpi)
    plot_case_residuals(frame, output, dpi)
    plot_day_absorbed_flux_along_route(frame, output, dpi)
    plot_absorbed_flux_scatter(frame, output, dpi)
    plot_flux_residual_diagnostics(frame, output, dpi)
    write_case_summary(audit, metrics, flux_metrics, solar_envelope, output)


def plot_pooled_scatter(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    labels = list(frame.groupby(["case_id", "route_name"], sort=True).groups)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(labels), 1)))
    for color, ((case_id, route_name), subset) in zip(
            colors, frame.groupby(["case_id", "route_name"], sort=True)):
        ax.scatter(subset["measured_mrt_c"], subset["trec_route_mrt_c"],
                   s=12, alpha=0.45, color=color, label=f"{case_id}: {route_name}")
    values = np.r_[frame["measured_mrt_c"], frame["trec_route_mrt_c"]]
    limits = [float(np.nanmin(values) - 1), float(np.nanmax(values) + 1)]
    ax.plot(limits, limits, "k--", lw=1.1, label="1:1")
    ax.set(xlim=limits, ylim=limits,
           xlabel="Experimentally measured MRT (°C)",
           ylabel="TREC-Route MRT (°C)",
           title="Lisbon mobile experiments: measured versus TREC-Route MRT")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, output / "all_cases_mrt_scatter", dpi)


def write_run_manifest(args: argparse.Namespace, audits: list[dict], skipped: list[str],
                       output: Path) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "program": str(Path(__file__).resolve()),
        "cases_compared": [audit["case_id"] for audit in audits],
        "cases_skipped_missing_results": skipped,
        "residual_convention": "TREC-Route minus measured MRT",
        "calibration_performed": False,
        "sunlit_threshold_wm2": args.sunlit_threshold_wm2,
        "coordinate_tolerance_m": args.coordinate_tolerance_m,
        "clock_tolerance_s": args.clock_tolerance_s,
        "minimum_subgroup_n": args.minimum_subgroup_n,
    }
    (output / "comparison_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.coordinate_tolerance_m <= 0 or args.clock_tolerance_s < 0:
        raise ValueError("pairing tolerances must be positive/non-negative")
    if args.sunlit_threshold_wm2 < 0 or args.minimum_subgroup_n < 2 or args.dpi < 72:
        raise ValueError("invalid sun threshold, subgroup size, or figure DPI")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    case_names = discover_cases(input_root, args.case)
    print(f"Discovered {len(case_names)} Lisbon case(s): {case_names}")

    all_frames: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_flux_metrics: list[pd.DataFrame] = []
    all_solar_envelopes: list[pd.DataFrame] = []
    audits: list[dict] = []
    skipped: list[str] = []
    for case_name in case_names:
        prediction = output_root / case_name / PREDICTION_RELATIVE
        if not prediction.is_file():
            message = f"{case_name}: simulation result missing: {prediction}"
            if args.require_all:
                raise FileNotFoundError(message)
            print(f"SKIP: {message}")
            skipped.append(case_name)
            continue
        print(f"\n[{case_name}] pairing experiment and TREC-Route results")
        frame, audit = load_case_pairs(
            case_name, input_root, output_root,
            args.coordinate_tolerance_m, args.clock_tolerance_s,
            args.sunlit_threshold_wm2,
        )
        metrics = calculate_metrics(frame, case_name, args.minimum_subgroup_n)
        flux_metrics = calculate_flux_metrics(
            frame, case_name, args.minimum_subgroup_n)
        solar_envelope = solar_forcing_envelope_summary(frame, case_name)
        case_output = output_root / case_name / CASE_OUTPUT_RELATIVE
        save_case_outputs(
            frame, audit, metrics, flux_metrics, solar_envelope,
            case_output, args.dpi)
        overall = metrics[(metrics["scope"] == "overall") & (metrics["group"] == "all")].iloc[0]
        print(
            f"  paired={len(frame)}, MBE={overall['mean_bias_error_c']:+.3f} C, "
            f"MAE={overall['mae_c']:.3f} C, RMSE={overall['rmse_c']:.3f} C, "
            f"r={overall['pearson_r']:.3f}")
        print(
            f"  coordinate error <= {audit['maximum_coordinate_pairing_error_m']:.6f} m; "
            f"clock error <= {audit['maximum_clock_pairing_error_s']:.3f} s")
        total_flux = flux_metrics[
            (flux_metrics["scope"] == "overall")
            & (flux_metrics["group"] == "all")
            & (flux_metrics["component"] == "total_absorbed_diagnostic")].iloc[0]
        print(
            f"  absorbed-flux diagnostic: MBE={total_flux['mean_bias_error_wm2']:+.2f} "
            f"W/m2, RMSE={total_flux['rmse_wm2']:.2f} W/m2")
        if audit["date_mismatched_pairs"]:
            print(
                f"  WARNING: {audit['date_mismatched_pairs']} night-route pairs use a "
                "different observation date; flagged as provisional in outputs")
        print(f"  saved: {case_output}")
        all_frames.append(frame)
        all_metrics.append(metrics)
        all_flux_metrics.append(flux_metrics)
        all_solar_envelopes.append(solar_envelope)
        audits.append(audit)

    if not all_frames:
        raise FileNotFoundError(
            "no selected Lisbon case has route-point simulation output; run the case first")

    if args.aggregate_output_dir is not None:
        aggregate_output = args.aggregate_output_dir.resolve()
        aggregate_output.mkdir(parents=True, exist_ok=True)
        pooled = pd.concat(all_frames, ignore_index=True)
        per_case_metrics = pd.concat(all_metrics, ignore_index=True)
        pooled_metrics = calculate_metrics(
            pooled, "ALL_AVAILABLE_CASES", args.minimum_subgroup_n)
        # Explicit opt-in diagnostic only; it is not the primary route result.
        case_records = []
        for case_name, subset in pooled.groupby("case_id", sort=True):
            record = metric_record(
                "ALL_AVAILABLE_CASES", "case", case_name, subset,
                args.minimum_subgroup_n)
            if record:
                case_records.append(record)
        combined_metrics = pd.concat([
            per_case_metrics,
            pooled_metrics,
            pd.DataFrame(case_records),
        ], ignore_index=True)
        pooled.to_csv(
            aggregate_output / "all_available_cases_comparison_points.csv", index=False)
        combined_metrics.to_csv(
            aggregate_output / "all_available_cases_validation_metrics.csv", index=False)
        pd.concat(all_flux_metrics, ignore_index=True).to_csv(
            aggregate_output / "all_available_cases_flux_validation_metrics.csv",
            index=False)
        solar_all = pd.concat(all_solar_envelopes, ignore_index=True)
        solar_all.to_csv(
            aggregate_output / "all_cases_solar_forcing_upper_envelope_diagnostics.csv",
            index=False)
        correlation = np.nan
        if (len(solar_all) >= 3
                and solar_all["measured_swin_q95_wm2"].std() > 0
                and solar_all["trec_atmospheric_ghi_q95_wm2"].std() > 0):
            correlation = float(stats.pearsonr(
                solar_all["measured_swin_q95_wm2"],
                solar_all["trec_atmospheric_ghi_q95_wm2"]).statistic)
        pd.DataFrame([{
            "n_case_routes": int(len(solar_all)),
            "pearson_r_mobile_swin_q95_vs_atmospheric_ghi_q95": correlation,
            "minimum_cases_for_reported_correlation": 3,
            "interpretation": (
                "Across-case upper-envelope diagnostic only; local reflections, "
                "sensor response, and remaining shade can affect mobile maxima."),
        }]).to_csv(
            aggregate_output / "solar_forcing_upper_envelope_cross_case_summary.csv",
            index=False)
        plot_pooled_scatter(pooled, aggregate_output, args.dpi)
        write_run_manifest(args, audits, skipped, aggregate_output)
        print(f"\nOptional aggregate diagnostic outputs: {aggregate_output}")
    else:
        print("\nCross-case aggregation disabled; each daytime route is plotted separately.")
    print(f"Compared cases: {[audit['case_id'] for audit in audits]}")
    if skipped:
        print(f"Skipped cases without route results: {skipped}")
    print("No geometry change or direct validation calibration was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
