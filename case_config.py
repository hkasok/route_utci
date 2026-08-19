#!/usr/bin/env python3
"""Validate a TREC-Route input case and expose its manifest to start.sh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


REQUIRED_FILES = (
    "buildings_stl", "vegetation_stl", "ground_stl", "routes_dir",
    "weather_csv", "osm_complete_file", "osm_ground_config",
    "radiant_flux_config",
)
OPTIONAL_FILES = ("radiation_forcing_config", "microclimate_config",
                  "urban_radiation_config")


def _inside_case(case_dir: Path, relative_path: str, label: str) -> Path:
    path = (case_dir / relative_path).resolve()
    try:
        path.relative_to(case_dir)
    except ValueError as exc:
        raise ValueError(
            f"case manifest path {label!r} leaves the input case: {relative_path}") from exc
    return path


def load_case(case_dir: Path) -> dict:
    """Load and validate one self-contained input case."""
    case_dir = Path(case_dir).resolve()
    manifest_path = case_dir / "case.json"
    if not case_dir.is_dir():
        raise FileNotFoundError(f"input case directory not found: {case_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"case manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError(f"{manifest_path}: unsupported or missing schema_version")
    if not str(data.get("case_id", "")).strip():
        raise ValueError(f"{manifest_path}: case_id is required")

    files = data.get("files", {})
    missing = [key for key in REQUIRED_FILES if not files.get(key)]
    if missing:
        raise ValueError(f"{manifest_path}: missing file entries {missing}")
    resolved = {key: _inside_case(case_dir, files[key], key)
                for key in REQUIRED_FILES}
    for key in OPTIONAL_FILES:
        if files.get(key):
            resolved[key] = _inside_case(case_dir, files[key], key)
    for key, path in resolved.items():
        if key == "routes_dir":
            if not path.is_dir():
                raise FileNotFoundError(f"case routes directory not found: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"case input {key} not found: {path}")

    location = data.get("location", {})
    coordinates = data.get("coordinates", {})
    defaults = data.get("simulation_defaults", {})
    workflow = data.get("workflow", {})
    routing_network_required = workflow.get("routing_network_required", True)
    if not isinstance(routing_network_required, bool):
        raise ValueError(
            f"{manifest_path}: workflow.routing_network_required must be true or false")
    bbox = location.get("osm_bbox_wgs84")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{manifest_path}: location.osm_bbox_wgs84 must have four values")
    numeric = {
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "local_origin_x": coordinates.get("local_origin_x", 0.0),
        "local_origin_y": coordinates.get("local_origin_y", 0.0),
    }
    for label, value in numeric.items():
        try:
            numeric[label] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{manifest_path}: {label} must be numeric") from exc
    if not location.get("timezone") or not coordinates.get("project_crs"):
        raise ValueError(f"{manifest_path}: timezone and project_crs are required")

    return {
        "case_dir": case_dir,
        "manifest": manifest_path,
        "case_id": str(data["case_id"]),
        "site_name": str(data.get("site_name", data["case_id"])),
        "files": resolved,
        "latitude": numeric["latitude"],
        "longitude": numeric["longitude"],
        "timezone": str(location["timezone"]),
        "bbox": [float(value) for value in bbox],
        "project_crs": str(coordinates["project_crs"]),
        "local_origin_x": numeric["local_origin_x"],
        "local_origin_y": numeric["local_origin_y"],
        "defaults": defaults,
        "routing_network_required": routing_network_required,
    }


def shell_assignments(case: dict) -> str:
    """Return safely quoted assignments consumed by the Bash launcher."""
    defaults = case["defaults"]
    values = {
        "CASE_ID_FROM_MANIFEST": case["case_id"],
        "CASE_SITE_NAME": case["site_name"],
        "CASE_MANIFEST": case["manifest"],
        "CASE_BUILDINGS_STL": case["files"]["buildings_stl"],
        "CASE_VEGETATION_STL": case["files"]["vegetation_stl"],
        "CASE_GROUND_STL": case["files"]["ground_stl"],
        "CASE_ROUTES_DIR": case["files"]["routes_dir"],
        "CASE_WEATHER_CSV": case["files"]["weather_csv"],
        "CASE_OSM_COMPLETE_FILE": case["files"]["osm_complete_file"],
        "CASE_OSM_GROUND_CONFIG": case["files"]["osm_ground_config"],
        "CASE_RADIANT_FLUX_CONFIG": case["files"]["radiant_flux_config"],
        "CASE_RADIATION_FORCING_CONFIG": case["files"].get(
            "radiation_forcing_config", ""),
        "CASE_MICROCLIMATE_CONFIG": case["files"].get(
            "microclimate_config", ""),
        "CASE_URBAN_RADIATION_CONFIG": case["files"].get(
            "urban_radiation_config", ""),
        "CASE_LAT": case["latitude"],
        "CASE_LON": case["longitude"],
        "CASE_TZ": case["timezone"],
        "CASE_OSM_BBOX": " ".join(str(value) for value in case["bbox"]),
        "CASE_PROJECT_CRS": case["project_crs"],
        "CASE_LOCAL_ORIGIN_X": case["local_origin_x"],
        "CASE_LOCAL_ORIGIN_Y": case["local_origin_y"],
        "CASE_DATE": defaults.get("date", "2025-07-06"),
        "CASE_DEPARTURE_HOUR": defaults.get("departure_hour", 13.0),
        "CASE_WALKING_SPEED_MS": defaults.get("walking_speed_ms", 1.3),
        "CASE_RH_PCT": defaults.get("relative_humidity_pct", 70.0),
        "CASE_WIND_MS": defaults.get("wind_speed_ms", 3.1),
        "CASE_CLOUD": defaults.get("cloud_cover_fraction", 0.0),
        "CASE_SUBJECT_PROFILE": defaults.get("subject_profile", ""),
        "CASE_TIMING_MODE": defaults.get("timing_mode", "distance_and_walking_speed"),
        "CASE_ROUTING_NETWORK_REQUIRED": (
            1 if case["routing_network_required"] else 0),
    }
    return "\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TREC-Route case manifest")
    parser.add_argument("case_dir")
    parser.add_argument("--shell", action="store_true",
                        help="emit safely quoted Bash assignments")
    args = parser.parse_args()
    case = load_case(Path(args.case_dir))
    if args.shell:
        print(shell_assignments(case))
    else:
        print(f"VALID case {case['case_id']}: {case['site_name']}")
        print(f"  input: {case['case_dir']}")
        for key, path in case["files"].items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
