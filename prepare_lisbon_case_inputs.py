#!/usr/bin/env python3
"""Prepare runnable TREC-Route inputs for the six Lisbon validation cases.

For each ``input/lisbonN`` case this program creates two measured routes:

* route 1: ``Study_Area_N___1st_measurement`` (daytime)
* route 2: ``Study_Area_N___night_measurement`` (nighttime)

The GeoPackage clock is recorded only to the minute while several ordered
samples occur within each minute.  Sub-minute times are therefore reconstructed
uniformly within each timestamp bin using the monotonically increasing RECORD
sequence.  Original timestamp values are retained in the measurement tables,
and the reconstruction method is recorded in every route sidecar.

Measured MRT and radiation are preserved under ``measurements/`` for separate
validation post-processing. Mobile ``SWin`` is never replayed as pointwise
domain-wide GHI because it contains local shade/reflection. The generated
optional radiation configuration uses only its upper envelope to estimate one
session-wide cloud attenuation; stage 05 still calculates all local shade.

Other daytime walks (held-out validation)
-----------------------------------------
``--measurement 2nd|3rd --case-suffix _w2|_w3`` builds a NEW case
``input/lisbonN<suffix>`` whose route 1 is that daytime walk and whose route 2
is the same night walk. The scene is not rebuilt: the geometry, OSM cache,
source footprint and material configuration are COPIED from ``input/lisbonN``
(whose LiDAR extraction rectangle already covers all four walks plus the
context buffer), and the manifest is ``lisbonN``'s with only the case id, the
simulation date (the walk's own date) and the provenance changed. Nothing in
``input/lisbonN`` is written. The 1st-walk default path is unchanged.

Two source layers carry non-ISO clocks: Study_Area_2 2nd stores Excel serial
days and Study_Area_2 3rd stores a time of day without a date. Both are read as
UTC, like every other layer's ``Z`` timestamps; the undated layer needs
``--assume-date`` and is flagged as such in its provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS

from generate_route import load_routes_directory, write_folder_outputs


ROOT = Path(__file__).resolve().parent
DEFAULT_GPKG = ROOT / "validation data" / "Liseben" / "Summer_data.gpkg"
PROJECT_CRS = "EPSG:3763"
LOCAL_CRS_NAME = "origin_shifted_projected_m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create day/night measured routes and remaining Lisbon case inputs")
    parser.add_argument("--gpkg", type=Path, default=DEFAULT_GPKG)
    parser.add_argument("--input-root", type=Path, default=ROOT / "input")
    parser.add_argument("--case", default="all",
                        help="all, one case number, or comma-separated numbers")
    parser.add_argument("--skip-osm-download", action="store_true")
    parser.add_argument("--overwrite-osm", action="store_true")
    parser.add_argument("--measurement", choices=DAY_MEASUREMENTS, default="1st",
                        help="daytime walk used as route 1 (default: 1st, the "
                             "published validation set)")
    parser.add_argument("--case-suffix", default="",
                        help="write a NEW case input/lisbonN<suffix> derived from "
                             "input/lisbonN (required unless --measurement 1st)")
    parser.add_argument("--assume-date", default=None,
                        help="YYYY-MM-DD for a layer whose timestamps carry no "
                             "date (Study_Area_2 3rd); recorded as an assumption")
    args = parser.parse_args()
    if args.measurement != "1st" and not args.case_suffix:
        parser.error("--measurement other than 1st needs --case-suffix, so the "
                     "published lisbonN cases are never overwritten")
    if args.case_suffix and not args.case_suffix.replace("_", "").isalnum():
        parser.error("--case-suffix may contain only letters, digits and '_'")
    return args


DAY_MEASUREMENTS = ("1st", "2nd", "3rd")


def parse_source_timestamps(values: pd.Series, assume_date: str | None
                            ) -> tuple[pd.Series, dict]:
    """UTC timestamps from the layer's TIMESTAMP column, whatever its format.

    ISO strings (every 1st-walk and night layer) parse exactly as before.
    Excel serial days (Study_Area_2 2nd) and date-less clock times
    (Study_Area_2 3rd) are converted explicitly; all are taken as UTC.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and numeric.between(30000.0, 60000.0).all():
        stamps = (pd.Timestamp("1899-12-30", tz="UTC")
                  + pd.to_timedelta(numeric.to_numpy(float) * 86400.0, unit="s"))
        return pd.Series(stamps, index=values.index).dt.round("s"), {
            "source_timestamp_format": "excel_serial_days_utc",
            "original_timestamp_resolution_s": 1,
            "subminute_reconstruction": "not needed (source resolves seconds)"}
    text = values.astype(str).str.strip()
    if text.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?").all():
        if not assume_date:
            raise ValueError("timestamps carry no date; pass --assume-date YYYY-MM-DD")
        stamps = pd.to_datetime(assume_date + " " + text, errors="coerce", utc=True)
        return stamps, {"source_timestamp_format": "time_of_day_utc_without_date",
                        "original_timestamp_resolution_s": 1,
                        "subminute_reconstruction": "not needed (source resolves seconds)",
                        "assumed_date": assume_date, "date_is_assumed": True}
    return pd.to_datetime(values, errors="coerce", utc=True), {
        "source_timestamp_format": "iso8601"}


def case_numbers(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, 7))
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(number not in range(1, 7) for number in values):
        raise ValueError("--case must select values from 1 through 6")
    return values


def refined_device_times(frame: pd.DataFrame,
                         coarse: pd.Series | None = None) -> tuple[pd.Series, float]:
    """Create strictly increasing UTC times from minute bins and RECORD order."""
    if coarse is None:
        coarse = pd.to_datetime(frame["TIMESTAMP"], errors="coerce", utc=True)
    if coarse.isna().any() or not coarse.is_monotonic_increasing:
        raise ValueError("valid measurement timestamps must be monotonic after RECORD ordering")
    counts = coarse.value_counts(sort=False)
    typical = float(np.median(60.0 / counts.to_numpy(dtype=float)))
    refined = pd.Series(index=frame.index, dtype="datetime64[ns, UTC]")
    unique = list(pd.unique(coarse))
    for group_index, stamp in enumerate(unique):
        indices = frame.index[coarse == stamp]
        if group_index + 1 < len(unique):
            gap = (pd.Timestamp(unique[group_index + 1]) - pd.Timestamp(stamp)).total_seconds()
            spacing = gap / len(indices) if 0.0 < gap <= 120.0 else typical
        else:
            spacing = typical
        refined.loc[indices] = pd.Timestamp(stamp) + pd.to_timedelta(
            np.arange(len(indices), dtype=float) * spacing, unit="s")
    if refined.isna().any() or np.any(np.diff(refined.astype("int64")) <= 0):
        raise ValueError("sub-minute timestamp reconstruction did not increase strictly")
    return refined, typical


def read_trajectory(gpkg: Path, layer: str, origin: np.ndarray,
                    route_id: int, case_id: str,
                    assume_date: str | None = None
                    ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    source = pyogrio.read_dataframe(gpkg, layer=layer)
    required = ["RECORD", "TIMESTAMP", "AirTemp", "HRel", "WS", "Tmrt"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{layer}: missing required measurement columns {missing}")
    source["_record_numeric"] = pd.to_numeric(source["RECORD"], errors="coerce")
    source["_timestamp_parsed"], timestamp_info = parse_source_timestamps(
        source["TIMESTAMP"], assume_date)
    good = (source.geometry.notna() & ~source.geometry.is_empty
            & source["_record_numeric"].notna() & source["_timestamp_parsed"].notna())
    dropped = int((~good).sum())
    source = source.loc[good].sort_values("_record_numeric", kind="stable").reset_index(drop=True)
    if len(source) < 2:
        raise ValueError(f"{layer}: fewer than two valid ordered observations")
    projected = source.to_crs(PROJECT_CRS)
    xy_projected = np.asarray([(geometry.x, geometry.y) for geometry in projected.geometry])
    xy_local = xy_projected - origin
    segment = np.linalg.norm(np.diff(xy_local, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    refined_utc, nominal_interval = refined_device_times(
        source, source["_timestamp_parsed"])
    local_time = refined_utc.dt.tz_convert("Europe/Lisbon")
    elapsed = (refined_utc - refined_utc.iloc[0]).dt.total_seconds().to_numpy(float)
    arrival_start = (local_time.iloc[0].hour + local_time.iloc[0].minute / 60.0
                     + local_time.iloc[0].second / 3600.0
                     + local_time.iloc[0].microsecond / 3.6e9)
    arrival_hour = arrival_start + elapsed / 3600.0
    lon = source.geometry.x.to_numpy(float)
    lat = source.geometry.y.to_numpy(float)
    route = pd.DataFrame({
        "seq": np.arange(len(source), dtype=int),
        "x_local_m": xy_local[:, 0], "y_local_m": xy_local[:, 1],
        "cumdist_m": cumulative,
        "lat": lat, "lon": lon,
        "x_proj_m": xy_projected[:, 0], "y_proj_m": xy_projected[:, 1],
        "timestamp_utc": refined_utc.map(lambda value: value.isoformat()),
        "timestamp_local": local_time.map(lambda value: value.isoformat()),
        "elapsed_time_s": elapsed,
        "arrival_hour_local": arrival_hour,
    })
    period = "night" if "night_measurement" in layer else "day"
    metadata = {
        "route_id": route_id,
        "name": f"{period}_experimental",
        "length_m": float(cumulative[-1]),
        "n_points": int(len(route)),
        "ds_path_m": None,
        "crs_local": LOCAL_CRS_NAME,
        "project_crs": CRS.from_user_input(PROJECT_CRS).to_string(),
        "local_origin_x": float(origin[0]), "local_origin_y": float(origin[1]),
        "start_latlon": [float(lat[0]), float(lon[0])],
        "end_latlon": [float(lat[-1]), float(lon[-1])],
        "source": "experimental_measurement",
        "network_constrained": False,
        "pedestrian_graphml": "",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "timing_source": "recorded_device_timestamps",
        "measurement_provenance": {
            "case_id": case_id,
            "source_gpkg": str(gpkg.resolve()),
            "source_layer": layer,
            "period": period,
            "original_timestamp_resolution_s": 60,
            "subminute_reconstruction": "uniform_within_timestamp_bin_by_RECORD",
            "nominal_reconstructed_sample_interval_s": nominal_interval,
            "timezone_interpretation": "source Z timestamps are UTC; converted to Europe/Lisbon",
            "dropped_invalid_rows": dropped,
            **({} if timestamp_info["source_timestamp_format"] == "iso8601"
               else timestamp_info),
        },
        "generator_args": {"program": Path(__file__).name, "layer": layer},
    }
    measurement = source.drop(columns=["geometry", "_record_numeric", "_timestamp_parsed"]).copy()
    measurement.insert(0, "route_id", route_id)
    measurement.insert(1, "seq", np.arange(len(measurement), dtype=int))
    measurement["timestamp_utc_refined"] = route["timestamp_utc"]
    measurement["timestamp_local_refined"] = route["timestamp_local"]
    measurement["x_local_m"] = route["x_local_m"]
    measurement["y_local_m"] = route["y_local_m"]
    measurement["x_proj_m"] = route["x_proj_m"]
    measurement["y_proj_m"] = route["y_proj_m"]
    measurement["latitude"] = lat
    measurement["longitude"] = lon
    measurement["measured_mrt_C"] = pd.to_numeric(source["Tmrt"], errors="coerce")
    return route, metadata, measurement


def write_routes_and_measurements(case_dir: Path, gpkg: Path, area: int,
                                  manifest: dict, measurement: str = "1st",
                                  assume_date: str | None = None
                                  ) -> tuple[list[dict], pd.DataFrame]:
    routes_dir = case_dir / "routes"
    measurements_dir = case_dir / "measurements"
    routes_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    for path in routes_dir.glob("route_*.*"):
        path.unlink()
    for name in ("routes_index.json", "route_polylines.pkl"):
        path = routes_dir / name
        if path.exists():
            path.unlink()
    origin = np.array([
        manifest["coordinates"]["local_origin_x"],
        manifest["coordinates"]["local_origin_y"],
    ], dtype=float)
    layers = [
        f"Study_Area_{area}___{measurement}_measurement",
        f"Study_Area_{area}___night_measurement",
    ]
    measurement_frames = []
    for route_id, layer in enumerate(layers, 1):
        route, metadata, measurement = read_trajectory(
            gpkg, layer, origin, route_id, manifest["case_id"],
            assume_date=assume_date if route_id == 1 else None)
        route.to_csv(routes_dir / f"route_{route_id}.csv", index=False,
                     float_format="%.12g")
        (routes_dir / f"route_{route_id}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        measurement.to_csv(
            measurements_dir / f"route_{route_id}_{metadata['name']}_measurements.csv",
            index=False)
        measurement_frames.append(measurement)
    loaded = load_routes_directory(
        routes_dir, expected_project_crs=PROJECT_CRS,
        expected_origin=tuple(origin))
    write_folder_outputs(routes_dir, loaded)
    combined = pd.concat(measurement_frames, ignore_index=True)
    combined.to_csv(measurements_dir / "experimental_measurements_all_routes.csv", index=False)
    return loaded, combined


def write_weather(case_dir: Path, routes: list[dict], measurements: pd.DataFrame) -> dict:
    rows = []
    for route in routes:
        route_id = route["route_id"]
        route_frame = route["frame"]
        measured = measurements[measurements["route_id"] == route_id].reset_index(drop=True)
        if len(measured) != len(route_frame):
            raise ValueError("measurement/route row count mismatch")
        rows.append(pd.DataFrame({
            "hour": route_frame["arrival_hour_local"].to_numpy(float) % 24.0,
            "air_temp_C": pd.to_numeric(measured["AirTemp"], errors="coerce"),
            "rh_pct": pd.to_numeric(measured["HRel"], errors="coerce"),
            "wind_ms": pd.to_numeric(measured["WS"], errors="coerce"),
            "route_id": route_id,
            "source_timestamp_utc": route_frame["timestamp_utc"],
        }))
    weather = pd.concat(rows, ignore_index=True).sort_values("hour", kind="stable")
    required = weather[["hour", "air_temp_C", "rh_pct", "wind_ms"]].to_numpy(float)
    if not np.isfinite(required).all() or weather["hour"].duplicated().any():
        raise ValueError("processed weather forcing contains invalid or duplicate hours")
    weather_dir = case_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    weather.to_csv(weather_dir / "weather.csv", index=False)
    provenance = {
        "source": "mobile experimental measurements",
        "variables": {"air_temp_C": "AirTemp", "rh_pct": "HRel", "wind_ms": "WS"},
        "time_basis": "refined device timestamps converted to Europe/Lisbon",
        "interpolation": "periodic linear interpolation by WeatherProvider",
        "important_limitations": [
            "The two forcing windows come from different measurement dates.",
            "Conditions between the observed day and night windows are interpolated.",
            "Wind measurement height was not identified in the supplied GeoPackage; no height conversion was applied.",
            "Mobile SWin is preserved for later validation but is not used as atmospheric GHI because it includes local shading.",
        ],
    }
    (weather_dir / "forcing_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def write_configs(case_dir: Path, case_id: str) -> None:
    config_dir = case_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    osm = json.loads((ROOT / "input/MMC/config/osm_ground_materials.json").read_text())
    osm["projected_crs"] = PROJECT_CRS
    osm["cache_directory"] = "osm"
    osm["raw_cache_file"] = "lisbon_complete_osm.gpkg"
    osm["input_file"] = None
    (config_dir / "osm_ground_materials.json").write_text(
        json.dumps(osm, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(
        ROOT / "input/MMC/config/radiant_flux_contribution_results.json",
        config_dir / "radiant_flux_contribution_results.json")
    radiation = {
        "mode": "route_upper_envelope_cloud",
        "source_file": "../measurements/experimental_measurements_all_routes.csv",
        "time_column": "timestamp_local_refined",
        "shortwave_column": "SWin",
        "upper_quantile": 0.95,
        "minimum_solar_elevation_deg": 10.0,
        "minimum_samples": 5,
        "scientific_note": (
            "Mobile SWin constrains one upper-envelope cloud estimate only; "
            "it is never pointwise atmospheric forcing."),
    }
    (config_dir / "radiation_forcing.json").write_text(
        json.dumps(radiation, indent=2) + "\n", encoding="utf-8")


def update_manifest(case_dir: Path, routes: list[dict], manifest: dict) -> None:
    day = routes[0]["frame"]
    night = routes[1]["frame"]
    day_date = pd.Timestamp(day["timestamp_local"].iloc[0]).date().isoformat()
    observed_speed = routes[0]["length_m"] / (
        day["elapsed_time_s"].iloc[-1] - day["elapsed_time_s"].iloc[0])
    manifest["description"] = (
        "Lisbon experimental validation case with one daytime and one nighttime mobile trajectory")
    manifest["setup_status"] = "ready"
    manifest["workflow"] = {
        "routing_network_required": False,
        "routing_reason": (
            "Routes are authoritative experimental GPS trajectories; the complete OSM cache "
            "is used for physical ground materials, not route construction."),
    }
    manifest["files"].update({
        "routes_dir": "routes",
        "weather_csv": "weather/weather.csv",
        "osm_complete_file": "osm/lisbon_complete_osm.gpkg",
        "osm_ground_config": "config/osm_ground_materials.json",
        "radiant_flux_config": "config/radiant_flux_contribution_results.json",
        "radiation_forcing_config": "config/radiation_forcing.json",
    })
    manifest["source"]["day_measurement_layer"] = routes[0]["metadata"][
        "measurement_provenance"]["source_layer"]
    manifest["source"]["night_measurement_layer"] = routes[1]["metadata"][
        "measurement_provenance"]["source_layer"]
    manifest["source"]["measured_mrt_usage"] = "preserved for later post-processing; not used as input"
    manifest["simulation_defaults"].update({
        "date": day_date,
        "departure_hour": float(day["arrival_hour_local"].iloc[0]),
        "walking_speed_ms": float(observed_speed),
        "timing_mode": "per_route_recorded_device_timestamps",
        "day_route_id": 1,
        "night_route_id": 2,
        "day_measurement_date": day_date,
        "night_measurement_date": pd.Timestamp(night["timestamp_local"].iloc[0]).date().isoformat(),
    })
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_readme(case_dir: Path, routes: list[dict], provenance: dict) -> None:
    route_lines = []
    for route in routes:
        frame = route["frame"]
        start = pd.Timestamp(frame["timestamp_local"].iloc[0])
        end = pd.Timestamp(frame["timestamp_local"].iloc[-1])
        duration_min = float(frame["elapsed_time_s"].iloc[-1]) / 60.0
        mean_speed = float(route["length_m"]) / float(frame["elapsed_time_s"].iloc[-1])
        route_lines.append(
            f"- Route {route['route_id']} (`{route['name']}`): "
            f"{route['metadata']['measurement_provenance']['source_layer']}; "
            f"{start.isoformat()} to {end.isoformat()}, {route['length_m']:.1f} m, "
            f"{duration_min:.2f} min, observed mean {mean_speed:.3f} m/s"
        )
    lines = [
        f"# {case_dir.name} input case", "",
        "This case contains two experimental mobile trajectories from `Summer_data.gpkg`:", "",
        *route_lines,
        "", "Route timing", "------------", "",
        "Stages 08 and 09 use each route's `elapsed_time_s` and `arrival_hour_local` columns.",
        "The fallback walking-speed setting is not used for these two routes. Source timestamps",
        "are minute-resolved, so ordered samples are distributed uniformly within each minute bin.",
        "A pedestrian-network download is not required: these are already-recorded pedestrian",
        "trajectories, not routes that TREC-Route must generate. The complete cached OSM feature",
        "set remains active for ground-material classification and radiation calculations.",
        "The day and night surveys occurred on different dates. The case manifest retains both",
        "dates; the current single-date radiation run uses the daytime date. This is documented",
        "for the later validation post-processor and no measurement comparison is performed here.",
        "", "Scientific separation", "---------------------", "",
        "Measured MRT, SWout, LWin, LWout, and the other observations are retained under",
        "`measurements/` for separate validation post-processing and are not used to calibrate",
        "the scene calculation.", "",
        "The mobile SWin series is not used as domain-wide atmospheric GHI because it already",
        "contains route-local shade and reflection; doing so would double-count scene geometry.",
        "Its high-sample envelope may constrain one optional session-wide cloud attenuation;",
        "Stage 05 still uses independent pvlib irradiance and ray-traces all local shade.", "",
        "Run", "---", "", "Select this input case and the matching output case in the UI, or run:", "",
        "```bash", f"INPUT_CASE_DIR=\"input/{case_dir.name}\" \\",
        f"OUTPUT_CASE_DIR=\"run_output/{case_dir.name}\" bash start.sh 2", "```", "",
    ]
    (case_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def download_osm(case_dir: Path, overwrite: bool) -> None:
    output = case_dir / "osm" / "lisbon_complete_osm.gpkg"
    if output.is_file() and not overwrite:
        print(f"  OSM cache exists: {output}")
        return
    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    command = [
        sys.executable, str(ROOT / "download_osm_complete_features.py"),
        "--ground-mesh", str(case_dir / "geometry/ground_and_water_final.stl"),
        "--config", str(case_dir / "config/osm_ground_materials.json"),
        "--output-file", str(output),
        "--local-origin-x", str(manifest["coordinates"]["local_origin_x"]),
        "--local-origin-y", str(manifest["coordinates"]["local_origin_y"]),
    ]
    if overwrite:
        command.append("--force-download")
    subprocess.run(command, cwd=ROOT, check=True)


DERIVED_COPY_FILES = (
    "geometry/building_final.stl", "geometry/vegetation_final.stl",
    "geometry/ground_and_water_final.stl", "geometry/geometry_build.json",
    "geometry/building_final_lod1_report.json",
    "osm/lisbon_complete_osm.gpkg", "osm/lisbon_complete_osm.metadata.json",
    "osm/lisbon_complete_osm.overpass.ql",
    "source/reference_measurement_route.csv",
    "source/scene_rectangle_projected.geojson",
    "config/osm_ground_materials.json",
    "config/radiant_flux_contribution_results.json",
    "config/radiation_forcing.json",
)


def derive_case(template_dir: Path, case_dir: Path, measurement: str) -> dict:
    """Create input/lisbonN<suffix> from input/lisbonN without touching it.

    The scene (geometry, OSM cache, footprint, material configuration) is
    COPIED -- never linked, so no later write can reach the template -- and the
    manifest is the template's with only identity and provenance changed. The
    simulation date is set from the walk once its route has been read.
    """
    template_dir, case_dir = template_dir.resolve(), case_dir.resolve()
    if case_dir == template_dir:
        raise ValueError("a derived case must not be the template case")
    template = json.loads((template_dir / "case.json").read_text(encoding="utf-8"))
    for relative in DERIVED_COPY_FILES:
        source = template_dir / relative
        target = case_dir / relative
        if not source.is_file():
            if relative.startswith(("geometry/building_final.stl",
                                    "geometry/vegetation_final.stl",
                                    "geometry/ground_and_water_final.stl",
                                    "osm/lisbon_complete_osm.gpkg", "config/osm",
                                    "config/radiant")):
                raise FileNotFoundError(source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if (not target.is_file() or target.stat().st_size != source.stat().st_size
                or target.stat().st_mtime < source.stat().st_mtime):
            shutil.copy2(source, target)
    manifest = json.loads(json.dumps(template))
    manifest["case_id"] = case_dir.name
    manifest["site_name"] = f"{template['site_name']} ({measurement} daytime walk)"
    manifest["description"] = (
        f"Held-out validation case: scene of {template['case_id']} (copied, not "
        f"rebuilt) with the {measurement} daytime walk as route 1 and the same "
        "night walk as route 2.")
    manifest["source"]["derived_from_case"] = template["case_id"]
    manifest["source"]["scene_copied_from"] = str(template_dir)
    return manifest


def finish_derived_manifest(case_dir: Path, manifest: dict, routes: list[dict]) -> None:
    day = routes[0]
    night = routes[1]
    day_date = pd.Timestamp(day["frame"]["timestamp_local"].iloc[0]).date().isoformat()
    provenance = day["metadata"]["measurement_provenance"]
    manifest["simulation_defaults"]["date"] = day_date
    manifest["source"]["day_measurement_layer"] = provenance["source_layer"]
    manifest["source"]["night_measurement_layer"] = night["metadata"][
        "measurement_provenance"]["source_layer"]
    manifest["source"]["day_measurement_date"] = day_date
    manifest["source"]["night_measurement_date"] = pd.Timestamp(
        night["frame"]["timestamp_local"].iloc[0]).date().isoformat()
    manifest["source"]["day_date_is_assumed"] = bool(provenance.get("date_is_assumed", False))
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")


def main() -> int:
    args = parse_args()
    gpkg = args.gpkg.resolve()
    if not gpkg.is_file():
        raise FileNotFoundError(gpkg)
    selected = case_numbers(args.case)
    for area in selected:
        template_dir = args.input_root.resolve() / f"lisbon{area}"
        case_dir = args.input_root.resolve() / f"lisbon{area}{args.case_suffix}"
        manifest_path = template_dir / "case.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepare geometry first; missing {manifest_path}")
        derived = bool(args.case_suffix)
        print(f"\n[{case_dir.name}] preparing day/night validation inputs "
              f"({args.measurement} daytime walk"
              + (f", scene copied from {template_dir.name})" if derived else ")"))
        if derived:
            manifest = derive_case(template_dir, case_dir, args.measurement)
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        routes, measurements = write_routes_and_measurements(
            case_dir, gpkg, area, manifest, args.measurement, args.assume_date)
        provenance = write_weather(case_dir, routes, measurements)
        if derived:
            # Keep the template's (Lisbon-specific) material configuration and
            # manifest layout, so only the walk differs from the published case.
            finish_derived_manifest(case_dir, manifest, routes)
        else:
            write_configs(case_dir, manifest["case_id"])
            update_manifest(case_dir, routes, manifest)
        write_readme(case_dir, routes, provenance)
        if not args.skip_osm_download and not derived:
            download_osm(case_dir, args.overwrite_osm)
        print(f"  routes: {len(routes)}; day {routes[0]['length_m']:.1f} m, "
              f"night {routes[1]['length_m']:.1f} m")
    print("\nLisbon input preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
