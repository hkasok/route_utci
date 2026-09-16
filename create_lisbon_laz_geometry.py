#!/usr/bin/env python3
"""Build route-specific TREC-Route STL geometry for the Lisbon measurements.

This is a case-preparation utility, not a numbered TREC-Route pipeline stage.
It reads the six mobile measurement routes in ``Summer_data.gpkg``, constructs
one rectangular scene around each route, extracts the classified points from
the intersecting LAZ tiles, and writes the three STL inputs expected by the
general case workflow::

    input/lisbonN/geometry/building_final.stl
    input/lisbonN/geometry/vegetation_final.stl
    input/lisbonN/geometry/ground_and_water_final.stl

The original LAZ and GeoPackage files are read-only.  Geometry is translated
to a per-case local frame; the projected origin is recorded in ``case.json``
and ``geometry/geometry_build.json``.  A case is never built from incomplete
LAZ coverage unless ``--allow-incomplete-coverage`` is explicitly supplied.

The authoritative route footprint is the first daytime measurement traverse,
ordered by its ``RECORD`` field.  The extraction rectangle nevertheless covers
all four measurement traverses for that study area plus ``--context-buffer-m``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "validation data" / "Liseben"
PROJECT_CRS = "EPSG:3763"
GROUND_CLASSES = (2, 9)
VEGETATION_CLASSES = (3, 4, 5)
BUILDING_CLASSES = (6,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create six Lisbon route-case STL geometry sets from classified LAZ"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gpkg", type=Path, default=None,
                        help="measurement GeoPackage (default: SOURCE/Summer_data.gpkg)")
    parser.add_argument("--laz-dir", type=Path, default=None,
                        help="classified LAZ directory (default: SOURCE/LAZ)")
    parser.add_argument("--input-root", type=Path, default=SCRIPT_DIR / "input")
    parser.add_argument("--case", default="all",
                        help="study-area number 1..6, comma-separated numbers, or all")
    parser.add_argument("--context-buffer-m", type=float, default=100.0,
                        help="rectangular context outside all observed traverses (default: 100 m)")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--ground-point-spacing-m", type=float, default=1.5)
    parser.add_argument("--building-point-spacing-m", type=float, default=0.35)
    parser.add_argument("--vegetation-point-spacing-m", type=float, default=0.50)
    # Named explicitly so a case records which crown model built it. "field"
    # is the only model without a shape prior; "radial" is a star-shaped hull
    # about a vertical axis, which reads as a field of circles from above.
    parser.add_argument("--crown-model", choices=["field", "reconstruct", "radial", "hemisphere"],
                        default="field",
                        help="Crown geometry model passed to 02_vegetation_to_stl.py "
                             "(default: field -- no shape prior)")
    parser.add_argument("--ground-raster-res-m", type=float, default=2.0)
    parser.add_argument("--allow-incomplete-coverage", action="store_true",
                        help="build clipped geometry despite missing LAZ coverage (not recommended)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="inspect routes, rectangles, and tile coverage without reading points")
    return parser.parse_args()


def selected_case_numbers(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, 7))
    try:
        values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--case must be all or comma-separated numbers from 1 to 6") from exc
    if not values or any(value < 1 or value > 6 for value in values):
        raise ValueError("--case must select one or more study areas from 1 to 6")
    return values


def require_inputs(gpkg: Path, laz_dir: Path) -> list[Path]:
    if not gpkg.is_file():
        raise FileNotFoundError(f"measurement GeoPackage not found: {gpkg}")
    if not laz_dir.is_dir():
        raise FileNotFoundError(f"LAZ directory not found: {laz_dir}")
    laz_files = list(laz_dir.glob("*.laz")) + list(laz_dir.glob("*.las"))
    # Prefer the canonical filename over desktop-created duplicates such as
    # ``tile (1).laz`` when their header signatures are identical.
    laz_files.sort(key=lambda path: ("(" in path.stem and ")" in path.stem, path.name))
    if not laz_files:
        raise FileNotFoundError(f"no .laz or .las files found in {laz_dir}")
    return laz_files


def measurement_layers(gpkg: Path, area: int) -> list[str]:
    prefix = f"Study_Area_{area}___"
    layers = [name for name, geometry_type in pyogrio.list_layers(gpkg)
              if name.startswith(prefix) and geometry_type == "Point"]
    if len(layers) != 4:
        raise ValueError(
            f"expected four measurement layers for Study Area {area}, found {len(layers)}: {layers}"
        )
    return layers


def read_measurements(gpkg: Path, layers: Iterable[str]):
    frames = []
    for layer in layers:
        frame = pyogrio.read_dataframe(gpkg, layer=layer)
        if frame.crs is None:
            raise ValueError(f"measurement layer has no CRS: {layer}")
        frame = frame.to_crs(PROJECT_CRS)
        frame["source_layer"] = layer
        frames.append(frame)
    return frames


def ordered_reference_route(frame) -> np.ndarray:
    route = frame.copy()
    if "RECORD" in route.columns:
        order = pd.to_numeric(route["RECORD"], errors="coerce")
        if order.notna().all():
            route = route.assign(_route_order=order).sort_values("_route_order", kind="stable")
    points = np.asarray(
        [(geometry.x, geometry.y) for geometry in route.geometry
         if geometry is not None and not geometry.is_empty], dtype=float
    )
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("reference measurement layer does not contain a valid route")
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
    return points[keep]


def scene_bounds(frames, buffer_m: float) -> tuple[float, float, float, float]:
    if not math.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("--context-buffer-m must be finite and positive")
    bounds = np.asarray([frame.total_bounds for frame in frames], dtype=float)
    if bounds.shape[1] != 4 or not np.isfinite(bounds).all():
        raise ValueError("measurement layers contain invalid bounds")
    return (
        float(bounds[:, 0].min() - buffer_m),
        float(bounds[:, 1].min() - buffer_m),
        float(bounds[:, 2].max() + buffer_m),
        float(bounds[:, 3].max() + buffer_m),
    )


def inspect_laz_tiles(laz_files: Iterable[Path]) -> list[dict]:
    tiles = []
    seen_signatures: dict[tuple, Path] = {}
    expected_crs = CRS.from_user_input(PROJECT_CRS)
    for path in laz_files:
        with laspy.open(path) as reader:
            header = reader.header
            crs = header.parse_crs()
            if crs is None or not CRS.from_user_input(crs).equals(expected_crs):
                raise ValueError(f"{path}: expected {PROJECT_CRS}, found {crs}")
            bounds = (float(header.mins[0]), float(header.mins[1]),
                      float(header.maxs[0]), float(header.maxs[1]))
            signature = (*bounds, int(header.point_count))
            if signature in seen_signatures:
                print(f"Ignoring duplicate LAZ tile {path.name}; identical to "
                      f"{seen_signatures[signature].name}")
                continue
            seen_signatures[signature] = path
            tiles.append({"path": path, "bounds": bounds,
                          "point_count": int(header.point_count)})
    return tiles


def tile_coverage(scene_bbox: tuple[float, float, float, float], tiles: list[dict]) -> dict:
    rectangle = box(*scene_bbox)
    selected = [tile for tile in tiles if box(*tile["bounds"]).intersects(rectangle)]
    coverage = unary_union([box(*tile["bounds"]) for tile in selected]).intersection(rectangle)
    missing = rectangle.difference(coverage)
    ratio = coverage.area / rectangle.area if rectangle.area else 0.0
    # Adjacent official tiles differ by about 1 mm at a few header bounds.
    # Treat only these tiny seam slivers as complete, never a material gap.
    seam_tolerance_m2 = max(10.0, rectangle.area * 1e-5)
    return {
        "selected": selected,
        "coverage_ratio": float(ratio),
        "missing_area_m2": float(missing.area),
        "missing_bounds": None if missing.is_empty else [float(value) for value in missing.bounds],
        "complete": bool(missing.area <= seam_tolerance_m2),
    }


def reduce_xy_grid(points: np.ndarray, spacing: float, prefer_high: bool) -> np.ndarray:
    """Keep one deterministic point per horizontal grid cell."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("point-reduction spacing must be finite and positive")
    xmin, ymin = points[:, :2].min(axis=0)
    ix = np.floor((points[:, 0] - xmin) / spacing).astype(np.int64)
    iy = np.floor((points[:, 1] - ymin) / spacing).astype(np.int64)
    width = int(ix.max()) + 1
    key = iy * width + ix
    z_order = -points[:, 2] if prefer_high else points[:, 2]
    order = np.lexsort((z_order, key))
    sorted_key = key[order]
    first = np.r_[True, sorted_key[1:] != sorted_key[:-1]]
    return points[order[first]]


def extract_classified_points(selected_tiles: list[dict], bounds, origin_xy,
                              chunk_size: int, spacings: dict[str, float]) -> tuple[dict, dict]:
    xmin, ymin, xmax, ymax = bounds
    class_sets = {
        "ground_and_water": GROUND_CLASSES,
        "building": BUILDING_CLASSES,
        "vegetation": VEGETATION_CLASSES,
    }
    accumulated: dict[str, list[np.ndarray]] = {name: [] for name in class_sets}
    raw_counts = {name: 0 for name in class_sets}
    for tile_number, tile in enumerate(selected_tiles, 1):
        print(f"    LAZ {tile_number}/{len(selected_tiles)}: {tile['path'].name}", flush=True)
        with laspy.open(tile["path"]) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                x = np.asarray(chunk.x)
                y = np.asarray(chunk.y)
                spatial = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                if not spatial.any():
                    continue
                classification = np.asarray(chunk.classification)
                z = np.asarray(chunk.z)
                for name, classes in class_sets.items():
                    mask = spatial & np.isin(classification, classes)
                    count = int(mask.sum())
                    raw_counts[name] += count
                    if count:
                        xyz = np.column_stack((x[mask] - origin_xy[0],
                                               y[mask] - origin_xy[1], z[mask]))
                        accumulated[name].append(
                            reduce_xy_grid(xyz, spacings[name], prefer_high=name != "ground_and_water")
                        )
    reduced = {}
    for name, parts in accumulated.items():
        if not parts:
            reduced[name] = np.empty((0, 3), dtype=np.float64)
        else:
            reduced[name] = reduce_xy_grid(
                np.concatenate(parts), spacings[name], prefer_high=name != "ground_and_water"
            )
    return reduced, raw_counts


def run_checked(command: list[str], label: str) -> None:
    print(f"  {label}", flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def validate_stl_outputs(paths: Iterable[Path]) -> dict:
    """Require finite, nonempty triangle meshes and return reproducible stats."""
    statistics = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required STL was not created: {path}")
        mesh = trimesh.load_mesh(path, process=False)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError(f"STL is empty: {path}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"STL contains non-finite vertices: {path}")
        statistics[path.name] = {
            "bytes": int(path.stat().st_size),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_local_xyz_m": np.asarray(mesh.bounds, dtype=float).tolist(),
        }
    return statistics


def write_geojson(path: Path, geometry, properties: dict) -> None:
    feature = {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2),
                    encoding="utf-8")


def write_route_csv(path: Path, route_xy: np.ndarray, origin_xy) -> None:
    transformer = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    segment = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    lon, lat = transformer.transform(route_xy[:, 0], route_xy[:, 1])
    frame = pd.DataFrame({
        "seq": np.arange(len(route_xy), dtype=int),
        "x_local_m": route_xy[:, 0] - origin_xy[0],
        "y_local_m": route_xy[:, 1] - origin_xy[1],
        "cumdist_m": cumulative,
        "lat": lat,
        "lon": lon,
        "x_proj_m": route_xy[:, 0],
        "y_proj_m": route_xy[:, 1],
    })
    frame.to_csv(path, index=False)


def write_case_metadata(case_dir: Path, area: int, frames, layers, scene_bbox, origin_xy,
                        reference_route, coverage, args, status: str,
                        effective_buffer_m: float) -> None:
    transformer = Transformer.from_crs(PROJECT_CRS, "EPSG:4326", always_xy=True)
    center_x = 0.5 * (scene_bbox[0] + scene_bbox[2])
    center_y = 0.5 * (scene_bbox[1] + scene_bbox[3])
    center_lon, center_lat = transformer.transform(center_x, center_y)
    west, south = transformer.transform(scene_bbox[0], scene_bbox[1])
    east, north = transformer.transform(scene_bbox[2], scene_bbox[3])
    manifest = {
        "schema_version": 1,
        "case_id": f"lisbon{area}",
        "site_name": f"Lisbon mobile measurement route {area}",
        "description": "Geometry-preparation phase; forcing, OSM materials, and final route input follow later.",
        "setup_status": status,
        "files": {
            "buildings_stl": "geometry/building_final.stl",
            "vegetation_stl": "geometry/vegetation_final.stl",
            "ground_stl": "geometry/ground_and_water_final.stl",
            "routes_dir": "routes",
            "weather_csv": "weather/weather.csv",
            "osm_complete_file": "osm/lisbon_complete_osm.gpkg",
            "osm_ground_config": "config/osm_ground_materials.json",
            "radiant_flux_config": "config/radiant_flux_contribution_results.json",
        },
        "location": {
            "latitude": center_lat,
            "longitude": center_lon,
            "timezone": "Europe/Lisbon",
            "osm_bbox_wgs84": [west, south, east, north],
        },
        "coordinates": {
            "project_crs": PROJECT_CRS,
            "local_origin_x": origin_xy[0],
            "local_origin_y": origin_xy[1],
        },
        "source": {
            "measurement_gpkg": str((args.gpkg or args.source_dir / "Summer_data.gpkg").resolve()),
            "measurement_layers": layers,
            "reference_route_layer": layers[0],
            "laz_directory": str((args.laz_dir or args.source_dir / "LAZ").resolve()),
        },
        "simulation_defaults": {
            "date": "",
            "departure_hour": 13.0,
            "walking_speed_ms": 1.3,
            "relative_humidity_pct": 50.0,
            "wind_speed_ms": 1.0,
            "cloud_cover_fraction": 0.0,
            "subject_profile": "",
        },
    }
    # PRESERVE DOWNSTREAM SETUP WHEN REBUILDING GEOMETRY.
    #
    # This function writes a fresh geometry-preparation manifest. Rebuilding
    # geometry on a case that was already prepared would otherwise silently
    # revert everything the later setup steps wrote: the manifest above hard-
    # codes weather/weather.csv, omits radiation_forcing_config entirely, and
    # sets simulation_defaults.date to "".
    #
    # That is not hypothetical. Re-running this script with --overwrite on the
    # six prepared lisbon cases deactivated the sensor-derived solar forcing
    # (weather_from_sensors.py --case, the UI's "build weather + solar from
    # sensors (activate)") and blanked the campaign dates, and the pipeline
    # then died in stage 05 with "0001-01-01 is a nonexistent time" -- after
    # minutes of ray tracing, naming neither the case nor the field.
    #
    # Geometry owns the three STL paths, the scene location/coordinates and
    # the source block. Everything else belongs to steps that run later, so it
    # is carried across when a manifest already exists.
    manifest_path = case_dir / "case.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            geometry_owned = {"buildings_stl", "vegetation_stl", "ground_stl"}
            for key, value in existing.get("files", {}).items():
                if key not in geometry_owned:
                    manifest["files"][key] = value
            for key in ("simulation_defaults", "workflow", "description",
                        "setup_status"):
                if key in existing:
                    manifest[key] = existing[key]
            preserved = sorted(
                set(existing.get("files", {})) - geometry_owned
                - set(manifest["files"]))
            print(f"  preserved existing case setup in case.json"
                  + (f" (also kept {', '.join(preserved)})" if preserved else ""))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    build = {
        "case_id": f"lisbon{area}",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_crs": PROJECT_CRS,
        "scene_bounds_projected_m": list(scene_bbox),
        "local_origin_projected_m": list(origin_xy),
        "context_buffer_m": effective_buffer_m,
        "requested_context_buffer_m": args.context_buffer_m,
        "reference_route_points": int(len(reference_route)),
        "measurement_layers": layers,
        "laz_coverage_ratio": coverage["coverage_ratio"],
        "missing_laz_area_m2": coverage["missing_area_m2"],
        "missing_laz_bounds_projected_m": coverage["missing_bounds"],
        "selected_laz_tiles": [tile["path"].name for tile in coverage["selected"]],
        "point_spacing_m": {
            "ground_and_water": args.ground_point_spacing_m,
            "building": args.building_point_spacing_m,
            "vegetation": args.vegetation_point_spacing_m,
        },
        "ground_raster_resolution_m": args.ground_raster_res_m,
        "note": "Original measurements and point clouds were not modified.",
    }
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    build_path = geometry_dir / "geometry_build.json"
    if build_path.is_file():
        try:
            previous = json.loads(build_path.read_text(encoding="utf-8"))
            same_bounds = np.allclose(previous.get("scene_bounds_projected_m", []),
                                      build["scene_bounds_projected_m"], rtol=0.0, atol=1e-6)
            if same_bounds:
                for key in ("source_point_counts", "retained_point_counts", "stl_validation"):
                    if key in previous:
                        build[key] = previous[key]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    build_path.write_text(
        json.dumps(build, indent=2) + "\n", encoding="utf-8"
    )


def prepare_case_directories(case_dir: Path) -> None:
    for relative in ("geometry", "routes", "weather", "osm", "config", "source", "build"):
        (case_dir / relative).mkdir(parents=True, exist_ok=True)


def build_case(area: int, gpkg: Path, tiles: list[dict], args) -> bool:
    case_dir = args.input_root.resolve() / f"lisbon{area}"
    layers = measurement_layers(gpkg, area)
    frames = read_measurements(gpkg, layers)
    reference_route = ordered_reference_route(frames[0])
    effective_buffer_m = args.context_buffer_m
    bbox = scene_bounds(frames, effective_buffer_m)
    coverage = tile_coverage(bbox, tiles)
    # A requested context may extend only a metre or two beyond an otherwise
    # complete official tile set. Preserve the largest covered rectangle when
    # at least 90% of the requested context remains. Never shrink around a
    # route that is itself missing source coverage.
    if not coverage["complete"] and not args.allow_incomplete_coverage:
        route_coverage = tile_coverage(scene_bounds(frames, 0.001), tiles)
        if route_coverage["complete"]:
            candidate = effective_buffer_m - 0.5
            minimum = 0.9 * effective_buffer_m
            while candidate >= minimum:
                candidate_bbox = scene_bounds(frames, candidate)
                candidate_coverage = tile_coverage(candidate_bbox, tiles)
                if candidate_coverage["complete"]:
                    print(f"[lisbon{area}] context adjusted from {effective_buffer_m:.1f} m "
                          f"to {candidate:.1f} m to remain inside supplied LAZ coverage")
                    effective_buffer_m = candidate
                    bbox = candidate_bbox
                    coverage = candidate_coverage
                    break
                candidate -= 0.5
    origin_xy = (bbox[0], bbox[1])
    prepare_case_directories(case_dir)
    status = "geometry_ready" if coverage["complete"] else "blocked_missing_laz_coverage"
    write_case_metadata(case_dir, area, frames, layers, bbox, origin_xy,
                        reference_route, coverage, args, status, effective_buffer_m)
    write_route_csv(case_dir / "source" / "reference_measurement_route.csv",
                    reference_route, origin_xy)
    write_geojson(case_dir / "source" / "scene_rectangle_projected.geojson", box(*bbox), {
        "case_id": f"lisbon{area}", "project_crs": PROJECT_CRS,
        "context_buffer_m": args.context_buffer_m,
    })
    print(f"\n[lisbon{area}] route points={len(reference_route):,}; "
          f"scene={bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f} m; "
          f"LAZ tiles={len(coverage['selected'])}; coverage={100*coverage['coverage_ratio']:.3f}%")
    if not coverage["complete"] and not args.allow_incomplete_coverage:
        print(f"  BLOCKED: missing {coverage['missing_area_m2']:,.1f} m2 of LAZ coverage; "
              f"missing bounds={coverage['missing_bounds']}")
        return False
    if args.dry_run:
        return True
    final_paths = [case_dir / "geometry" / name for name in
                   ("building_final.stl", "vegetation_final.stl", "ground_and_water_final.stl")]
    if all(path.is_file() for path in final_paths) and not args.overwrite:
        print("  geometry exists; use --overwrite to rebuild")
        build_path = case_dir / "geometry" / "geometry_build.json"
        build = json.loads(build_path.read_text(encoding="utf-8"))
        build["stl_validation"] = validate_stl_outputs(final_paths)
        build_path.write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
        return True
    split_dir = case_dir / "build" / "classified_points"
    split_dir.mkdir(parents=True, exist_ok=True)
    points, raw_counts = extract_classified_points(
        coverage["selected"], bbox, origin_xy, args.chunk_size,
        {"ground_and_water": args.ground_point_spacing_m,
         "building": args.building_point_spacing_m,
         "vegetation": args.vegetation_point_spacing_m},
    )
    for name, values in points.items():
        if name == "ground_and_water" and len(values) < 3:
            raise ValueError(f"lisbon{area}: insufficient ground/water points after extraction")
        np.save(split_dir / f"{name}_points.npy", values)
        print(f"  {name}: {raw_counts[name]:,} source points -> {len(values):,} retained")
    python = sys.executable
    run_checked([
        python, str(SCRIPT_DIR / "02_vegetation_to_stl.py"),
        "--input", str(split_dir / "vegetation_points.npy"),
        "--output", str(case_dir / "geometry" / "vegetation_final.stl"),
        "--ground-npy", str(split_dir / "ground_and_water_points.npy"),
        "--crown-model", args.crown_model,
    ], "vegetation STL")
    run_checked([
        python, str(SCRIPT_DIR / "03_buildings_to_stl.py"),
        "--input", str(split_dir / "building_points.npy"),
        "--output", str(case_dir / "geometry" / "building_final.stl"),
        "--ground-npy", str(split_dir / "ground_and_water_points.npy"),
    ], "building STL")
    run_checked([
        python, str(SCRIPT_DIR / "04_ground_to_stl.py"),
        "--input", str(split_dir / "ground_and_water_points.npy"),
        "--output", str(case_dir / "geometry" / "ground_and_water_final.stl"),
        "--raster-res", str(args.ground_raster_res_m),
    ], "ground/water STL")
    build_path = case_dir / "geometry" / "geometry_build.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["source_point_counts"] = raw_counts
    build["retained_point_counts"] = {key: int(len(value)) for key, value in points.items()}
    build["stl_validation"] = validate_stl_outputs(final_paths)
    build_path.write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    if not args.keep_intermediate:
        for path in split_dir.glob("*.npy"):
            path.unlink()
        split_dir.rmdir()
        try:
            (case_dir / "build").rmdir()
        except OSError:
            pass
    return True


def main() -> int:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.gpkg = (args.gpkg or args.source_dir / "Summer_data.gpkg").resolve()
    args.laz_dir = (args.laz_dir or args.source_dir / "LAZ").resolve()
    laz_files = require_inputs(args.gpkg, args.laz_dir)
    cases = selected_case_numbers(args.case)
    print(f"Source GeoPackage: {args.gpkg}")
    print(f"Classified point clouds: {args.laz_dir} ({len(laz_files)} tiles)")
    print(f"Cases: {', '.join('lisbon'+str(value) for value in cases)}")
    tiles = inspect_laz_tiles(laz_files)
    outcomes = {area: build_case(area, args.gpkg, tiles, args) for area in cases}
    ready = [area for area, result in outcomes.items() if result]
    blocked = [area for area, result in outcomes.items() if not result]
    print("\nSummary")
    print("  ready:", ", ".join(f"lisbon{area}" for area in ready) or "none")
    print("  blocked:", ", ".join(f"lisbon{area}" for area in blocked) or "none")
    if blocked:
        print("  Supply LAZ tiles covering each blocked case's missing bounds, then rerun.")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
