#!/usr/bin/env python3
"""Generate, import, validate, and load TREC-Route route input files.

The route-file contract is deliberately source-agnostic. A route may come from
the existing edge-disjoint OSM algorithm, a QGIS drawing exported as GeoJSON,
GPX, or any other tool. Once converted, each route consists of:

``route_<id>.csv``
    Authoritative densified geometry with columns ``seq, x_local_m, y_local_m,
    cumdist_m, lat, lon, x_proj_m, y_proj_m``.

``route_<id>.json``
    CRS, origin, source, length, spacing, and provenance metadata.

Measured mobile trajectories may additionally include ``timestamp_utc``,
``timestamp_local``, ``elapsed_time_s``, and ``arrival_hour_local``. Stages 08
and 09 use that recorded schedule instead of reconstructing time from a chosen
walking speed. Routes without those optional columns retain the historical
speed-based behavior exactly.

``routes_index.json`` lists every valid route. ``route_polylines.pkl`` is a
derived compatibility artifact used only to hand the same geometries to the
MRT ray tracer; the CSV files remain authoritative.

Appending is the default. Existing route IDs are never clobbered unless
``--overwrite`` is explicitly requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pickle
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString
from shapely.ops import unary_union

from route_selection import (find_best_corner_pair, reconstruct_route_xy,
                             resolve_endpoints)


REQUIRED_COLUMNS = (
    "seq", "x_local_m", "y_local_m", "cumdist_m",
    "lat", "lon", "x_proj_m", "y_proj_m",
)
RECORDED_TIME_COLUMNS = (
    "timestamp_utc", "timestamp_local", "elapsed_time_s", "arrival_hour_local",
)
LOCAL_CRS_NAME = "origin_shifted_projected_m"
ROUTE_RE = re.compile(r"^route_(\d+)\.csv$")


def utc_now() -> str:
    """Return a reproducible ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def route_sort_key(path: Path) -> int:
    """Extract the numeric route ID used for deterministic folder ordering."""
    match = ROUTE_RE.match(path.name)
    if not match:
        raise ValueError(f"invalid route filename {path.name!r}; expected route_<id>.csv")
    return int(match.group(1))


def route_csv_files(routes_dir: Path) -> list[Path]:
    """Return authoritative route CSVs in numeric route-ID order."""
    return sorted(
        (path for path in routes_dir.glob("route_*.csv") if ROUTE_RE.match(path.name)),
        key=route_sort_key,
    )


def _cumdist(xy: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))


def densify_polyline(xy: np.ndarray, ds: float) -> tuple[np.ndarray, float]:
    """Densify using the same endpoint-exclusive convention as route_selection."""
    xy = np.asarray(xy, dtype=float)
    if ds <= 0 or not np.isfinite(ds):
        raise ValueError("--ds-path must be a finite positive distance in meters")
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2:
        raise ValueError("a route geometry must contain at least two XY points")
    xy = xy[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("route geometry contains NaN or infinite coordinates")
    seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(seg_lens.sum())
    if total <= 0:
        raise ValueError("route geometry has zero length")
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    svals = np.arange(0.0, total, ds)
    dense = np.empty((len(svals), 2), dtype=float)
    j = 0
    for i, distance in enumerate(svals):
        while j < len(seg_lens) - 1 and distance > cum[j + 1]:
            j += 1
        fraction = 0.0 if seg_lens[j] == 0 else (distance - cum[j]) / seg_lens[j]
        dense[i] = xy[j] + fraction * (xy[j + 1] - xy[j])
    return dense, total


def _canonical_crs(value: str) -> str:
    return CRS.from_user_input(value).to_string()


def _route_frame(xy_local: np.ndarray, origin: np.ndarray,
                 project_crs: str) -> pd.DataFrame:
    xy_proj = np.asarray(xy_local, dtype=float) + origin
    to_wgs84 = Transformer.from_crs(project_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(xy_proj[:, 0], xy_proj[:, 1])
    return pd.DataFrame({
        "seq": np.arange(len(xy_local), dtype=int),
        "x_local_m": xy_local[:, 0],
        "y_local_m": xy_local[:, 1],
        "cumdist_m": _cumdist(xy_local),
        "lat": lat,
        "lon": lon,
        "x_proj_m": xy_proj[:, 0],
        "y_proj_m": xy_proj[:, 1],
    })


def write_route(routes_dir: Path, route_id: int, xy_local: np.ndarray,
                length_m: float, ds_path_m: float, source: str,
                route_name_prefix: str, project_crs: str, origin: np.ndarray,
                generator_args: dict) -> tuple[Path, Path]:
    """Write one authoritative CSV and its metadata sidecar."""
    frame = _route_frame(np.asarray(xy_local, dtype=float), origin, project_crs)
    if len(frame) < 2:
        raise ValueError("a densified route must contain at least two points")
    csv_path = routes_dir / f"route_{route_id}.csv"
    json_path = routes_dir / f"route_{route_id}.json"
    frame.to_csv(csv_path, index=False, float_format="%.17g")
    metadata = {
        "route_id": int(route_id),
        "name": f"{route_name_prefix}_{route_id}",
        "length_m": float(length_m),
        "n_points": int(len(frame)),
        "ds_path_m": float(ds_path_m),
        "crs_local": LOCAL_CRS_NAME,
        "project_crs": _canonical_crs(project_crs),
        "local_origin_x": float(origin[0]),
        "local_origin_y": float(origin[1]),
        "start_latlon": [float(frame.iloc[0]["lat"]), float(frame.iloc[0]["lon"])],
        "end_latlon": [float(frame.iloc[-1]["lat"]), float(frame.iloc[-1]["lon"])],
        "source": source,
        "network_constrained": True,
        "pedestrian_graphml": str(generator_args["graphml"]),
        "generated_utc": utc_now(),
        "generator_args": generator_args,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return csv_path, json_path


def validate_route_file(csv_path: Path, expected_project_crs: str | None = None,
                        expected_origin: tuple[float, float] | None = None) -> dict:
    """Validate one route contract and return its parsed geometry/metadata."""
    csv_path = Path(csv_path)
    metadata_path = csv_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"missing metadata sidecar: {metadata_path}")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")
    if len(frame) < 2:
        raise ValueError(f"{csv_path}: route must contain at least two points")
    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{csv_path}: route columns contain NaN or infinite values")
    seq = numeric["seq"].to_numpy(dtype=float)
    if not np.array_equal(seq, np.arange(len(frame), dtype=float)):
        raise ValueError(f"{csv_path}: seq must be consecutive integers starting at zero")
    cumdist = numeric["cumdist_m"].to_numpy(dtype=float)
    if abs(cumdist[0]) > 1e-9 or np.any(np.diff(cumdist) < -1e-9):
        raise ValueError(f"{csv_path}: cumdist_m must start at zero and be monotonic")
    xy = numeric[["x_local_m", "y_local_m"]].to_numpy(dtype=float)
    computed = _cumdist(xy)
    if not np.allclose(cumdist, computed, rtol=1e-10, atol=1e-7):
        raise ValueError(f"{csv_path}: cumdist_m is inconsistent with route geometry")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_metadata = {
        "route_id", "name", "length_m", "n_points", "ds_path_m", "crs_local",
        "project_crs", "local_origin_x", "local_origin_y", "start_latlon",
        "end_latlon", "source", "network_constrained", "pedestrian_graphml",
        "generated_utc", "generator_args",
    }
    absent = sorted(required_metadata - set(metadata))
    if absent:
        raise ValueError(f"{metadata_path}: missing required keys {absent}")
    filename_id = route_sort_key(csv_path)
    if int(metadata["route_id"]) != filename_id:
        raise ValueError(f"{csv_path}: route_id does not match its filename")
    if int(metadata["n_points"]) != len(frame):
        raise ValueError(f"{csv_path}: n_points does not match CSV row count")
    if metadata["crs_local"] != LOCAL_CRS_NAME:
        raise ValueError(f"{csv_path}: unsupported crs_local {metadata['crs_local']!r}")
    measured_trajectory = metadata.get("source") == "experimental_measurement"
    if metadata["network_constrained"] is not True and not measured_trajectory:
        raise ValueError(
            f"{csv_path}: routes must be constrained to the OSM pedestrian network")
    if measured_trajectory and not metadata.get("measurement_provenance"):
        raise ValueError(
            f"{metadata_path}: experimental_measurement requires measurement_provenance")
    project_crs = _canonical_crs(metadata["project_crs"])
    if expected_project_crs and project_crs != _canonical_crs(expected_project_crs):
        raise ValueError(f"{csv_path}: project CRS {project_crs} does not match "
                         f"expected {_canonical_crs(expected_project_crs)}")
    origin = np.array([metadata["local_origin_x"], metadata["local_origin_y"]], dtype=float)
    if not np.isfinite(origin).all():
        raise ValueError(f"{csv_path}: local origin is not finite")
    if expected_origin is not None and not np.allclose(origin, expected_origin, atol=1e-9):
        raise ValueError(f"{csv_path}: local origin {origin.tolist()} does not match "
                         f"expected {list(expected_origin)}")
    xy_proj = numeric[["x_proj_m", "y_proj_m"]].to_numpy(dtype=float)
    if not np.allclose(xy_proj, xy + origin, rtol=0.0, atol=1e-6):
        raise ValueError(f"{csv_path}: projected coordinates do not equal local + origin")
    if float(metadata["length_m"]) + 1e-7 < cumdist[-1]:
        raise ValueError(f"{csv_path}: metadata length is shorter than its geometry")
    present_time = [column in frame.columns for column in RECORDED_TIME_COLUMNS]
    if any(present_time) and not all(present_time):
        missing_time = [column for column in RECORDED_TIME_COLUMNS
                        if column not in frame.columns]
        raise ValueError(f"{csv_path}: incomplete recorded schedule; missing {missing_time}")
    if all(present_time):
        elapsed = pd.to_numeric(frame["elapsed_time_s"], errors="coerce").to_numpy(float)
        arrival = pd.to_numeric(frame["arrival_hour_local"], errors="coerce").to_numpy(float)
        # Reconstructed device times may mix whole-second and fractional-second
        # ISO strings; pandas' strict single-format inference rejects that mix.
        utc = pd.to_datetime(frame["timestamp_utc"], errors="coerce",
                             utc=True, format="mixed")
        local = pd.to_datetime(frame["timestamp_local"], errors="coerce",
                               utc=False, format="mixed")
        if (not np.isfinite(elapsed).all() or not np.isfinite(arrival).all()
                or utc.isna().any() or local.isna().any()):
            raise ValueError(f"{csv_path}: recorded schedule contains invalid values")
        if abs(elapsed[0]) > 1e-6 or np.any(np.diff(elapsed) <= 0.0):
            raise ValueError(
                f"{csv_path}: elapsed_time_s must start at zero and increase strictly")
        expected_arrival = arrival[0] + elapsed / 3600.0
        if not np.allclose(arrival, expected_arrival, rtol=0.0, atol=1e-7):
            raise ValueError(
                f"{csv_path}: arrival_hour_local is inconsistent with elapsed_time_s")
    return {"route_id": filename_id, "name": str(metadata["name"]),
            "xy": xy, "length_m": float(metadata["length_m"]),
            "frame": frame, "metadata": metadata}


def route_arrival_schedule(route: dict, departure_hour: float,
                           walking_speed_ms: float) -> tuple[np.ndarray, str]:
    """Return unwrapped arrival hours and the timing source for one route.

    A complete recorded schedule is authoritative. Otherwise this returns the
    legacy distance/speed schedule, preserving all existing route behavior.
    """
    frame = route["frame"]
    if all(column in frame.columns for column in RECORDED_TIME_COLUMNS):
        arrival = pd.to_numeric(frame["arrival_hour_local"], errors="raise").to_numpy(float)
        source = "recorded_device_timestamps"
    else:
        if not np.isfinite(walking_speed_ms) or walking_speed_ms <= 0.0:
            raise ValueError("walking speed must be finite and positive")
        xy = np.asarray(route["xy"], dtype=float)
        distance = _cumdist(xy)
        arrival = float(departure_hour) + distance / float(walking_speed_ms) / 3600.0
        source = "distance_and_walking_speed"
    if len(arrival) < 2 or not np.isfinite(arrival).all() or np.any(np.diff(arrival) <= 0.0):
        raise ValueError(
            f"route {route['route_id']}: arrival schedule must be finite and strictly increasing")
    return arrival, source


def load_routes_directory(routes_dir: Path, expected_project_crs: str | None = None,
                          expected_origin: tuple[float, float] | None = None) -> list[dict]:
    """Load and validate every route CSV in deterministic numeric-ID order."""
    routes_dir = Path(routes_dir)
    if not routes_dir.is_dir():
        raise FileNotFoundError(f"routes directory not found: {routes_dir}")
    files = route_csv_files(routes_dir)
    if not files:
        raise FileNotFoundError(f"no route_*.csv files found in {routes_dir}")
    routes = [validate_route_file(path, expected_project_crs, expected_origin)
              for path in files]
    ids = [route["route_id"] for route in routes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate route IDs in {routes_dir}")
    reference_crs = _canonical_crs(routes[0]["metadata"]["project_crs"])
    reference_origin = np.array([
        routes[0]["metadata"]["local_origin_x"],
        routes[0]["metadata"]["local_origin_y"],
    ], dtype=float)
    for route in routes[1:]:
        metadata = route["metadata"]
        route_crs = _canonical_crs(metadata["project_crs"])
        route_origin = np.array([
            metadata["local_origin_x"], metadata["local_origin_y"],
        ], dtype=float)
        if route_crs != reference_crs:
            raise ValueError(
                f"{routes_dir}: route_{route['route_id']} uses project CRS "
                f"{route_crs}, but route_{routes[0]['route_id']} uses {reference_crs}")
        if not np.allclose(route_origin, reference_origin, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{routes_dir}: route_{route['route_id']} uses local origin "
                f"{route_origin.tolist()}, but route_{routes[0]['route_id']} uses "
                f"{reference_origin.tolist()}")
    return routes


def write_folder_outputs(routes_dir: Path, routes: list[dict]) -> None:
    """Write the folder index and derived MRT polyline compatibility file."""
    entries = []
    polylines = []
    for route in routes:
        rid = int(route["route_id"])
        entries.append({
            "route_id": rid,
            "name": route["name"],
            "csv": f"route_{rid}.csv",
            "metadata": f"route_{rid}.json",
            "length_m": float(route["length_m"]),
            "n_points": int(len(route["xy"])),
            "source": route["metadata"]["source"],
            "timing_source": route["metadata"].get(
                "timing_source", "distance_and_walking_speed"),
        })
        polylines.append(np.asarray(route["xy"], dtype=float))
    index = {
        "schema_version": 1,
        "contract": "TREC-Route source-agnostic route inputs",
        "updated_utc": utc_now(),
        "n_routes": len(entries),
        "routes": entries,
    }
    (routes_dir / "routes_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (routes_dir / "route_polylines.pkl").open("wb") as stream:
        pickle.dump({"polylines": polylines,
                     "highway_tags": [f"route_{entry['route_id']}" for entry in entries]},
                    stream)


def _load_pedestrian_graph(graphml: Path | str):
    """Load the authoritative pedestrian graph used for route constraints."""
    import osmnx as ox

    graphml = Path(graphml)
    if not graphml.is_file():
        raise FileNotFoundError(f"pedestrian GraphML not found: {graphml}")
    graph = ox.load_graphml(graphml)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError(f"pedestrian GraphML is empty: {graphml}")
    return graph


def _graph_positions(graph) -> dict:
    return {node: (float(data["x"]), float(data["y"]))
            for node, data in graph.nodes(data=True)}


def _nearest_graph_nodes(pos: dict, xy: np.ndarray) -> tuple[list, np.ndarray]:
    """Map input vertices to their nearest pedestrian-graph nodes."""
    node_ids = list(pos)
    node_xy = np.asarray([pos[node] for node in node_ids], dtype=float)
    snapped = []
    distances = []
    for point in np.asarray(xy, dtype=float):
        distance = np.hypot(node_xy[:, 0] - point[0], node_xy[:, 1] - point[1])
        index = int(np.argmin(distance))
        snapped.append(node_ids[index])
        distances.append(float(distance[index]))
    return snapped, np.asarray(distances, dtype=float)


def map_match_to_pedestrian_graph(graph, xy_local: np.ndarray, ds: float,
                                  max_snap_distance_m: float) -> tuple[np.ndarray, float]:
    """Route imported waypoints through OSM pedestrian edges and densify.

    Each source vertex is treated as a waypoint. Consecutive waypoints are
    joined by the shortest path in the existing pedestrian graph, so an
    imported straight LineString cannot become a free-space route.
    """
    pos = _graph_positions(graph)
    snapped, distances = _nearest_graph_nodes(pos, xy_local)
    if np.max(distances) > max_snap_distance_m:
        index = int(np.argmax(distances))
        raise ValueError(
            f"imported waypoint {index} is {distances[index]:.2f} m from the "
            f"pedestrian network (limit {max_snap_distance_m:.2f} m)")
    waypoint_nodes = [snapped[0]]
    waypoint_nodes.extend(node for node in snapped[1:] if node != waypoint_nodes[-1])
    if len(waypoint_nodes) < 2:
        raise ValueError("imported route snaps to fewer than two pedestrian nodes")
    simple = nx.Graph(graph)
    node_path = []
    for start, end in zip(waypoint_nodes[:-1], waypoint_nodes[1:]):
        try:
            leg = nx.shortest_path(simple, start, end, weight="length")
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"no pedestrian path connects imported waypoints at nodes "
                f"{start} and {end}") from exc
        node_path.extend(leg if not node_path else leg[1:])
    dense, length = reconstruct_route_xy(graph, node_path, ds=ds)
    print(f"[routes] OSM map match: {len(xy_local)} source waypoint(s), "
          f"{len(waypoint_nodes)} snapped node(s), max snap "
          f"{np.max(distances):.2f} m")
    return dense, length


def validate_routes_on_pedestrian_graph(routes: list[dict], graphml: Path | str,
                                        tolerance_m: float) -> None:
    """Reject route geometry that leaves the authoritative pedestrian graph."""
    if tolerance_m <= 0 or not np.isfinite(tolerance_m):
        raise ValueError("network validation tolerance must be finite and positive")
    graph = _load_pedestrian_graph(graphml)
    edge_lines = []
    for u, v, data in graph.edges(data=True):
        geometry = data.get("geometry")
        if geometry is not None:
            edge_lines.append(geometry)
        else:
            edge_lines.append(LineString([
                (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
            ]))
    pedestrian_footprint = unary_union(edge_lines).buffer(tolerance_m)
    for route in routes:
        line = LineString(route["xy"])
        outside_length = float(line.difference(pedestrian_footprint).length)
        allowed = max(tolerance_m, float(line.length) * 1e-8)
        if outside_length > allowed:
            raise ValueError(
                f"route_{route['route_id']} leaves the OSM pedestrian network: "
                f"{outside_length:.3f} m outside the {tolerance_m:.3f} m tolerance")


def _same_route_geometry(xy_a: np.ndarray, xy_b: np.ndarray,
                         tolerance_m: float = 0.01) -> bool:
    """Identify duplicate paths even if their sampling points differ slightly."""
    line_a = LineString(np.asarray(xy_a, dtype=float))
    line_b = LineString(np.asarray(xy_b, dtype=float))
    return (abs(line_a.length - line_b.length) <= tolerance_m
            and line_a.hausdorff_distance(line_b) <= tolerance_m)


def _generator_args(args: argparse.Namespace) -> dict:
    return {key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()}


def _select_edge_disjoint_routes(args: argparse.Namespace) -> list[tuple[np.ndarray, float]]:
    """Run the pre-refactor graph algorithm without changing its ordering."""
    print(f"[routes] Loading routable network: {args.graphml}")
    graph_multi = _load_pedestrian_graph(args.graphml)
    graph_simple = nx.Graph(graph_multi)
    pos = {node: (float(data["x"]), float(data["y"]))
           for node, data in graph_multi.nodes(data=True)}
    print(f"[routes] {graph_simple.number_of_nodes()} nodes, "
          f"{graph_simple.number_of_edges()} edges")
    origin = (args.local_origin_x, args.local_origin_y)
    start_node, end_node, connectivity = resolve_endpoints(
        pos, args.n_routes, args.project_crs, origin,
        start_latlon=args.start_latlon, end_latlon=args.end_latlon,
        start_xy=args.start_xy, end_xy=args.end_xy,
    )
    if start_node is None or end_node is None:
        xs = [point[0] for point in pos.values()]
        ys = [point[1] for point in pos.values()]
        start_node, end_node, connectivity = find_best_corner_pair(
            graph_simple, pos, (min(xs), min(ys)), (max(xs), max(ys)),
            args.n_routes)
        source = "edge_disjoint_auto"
    else:
        connectivity = nx.edge_connectivity(graph_simple, start_node, end_node)
        source = "manual_latlon" if args.start_latlon is not None else "manual_xy"
    node_paths = list(nx.edge_disjoint_paths(
        graph_simple, start_node, end_node))[:args.n_routes]
    edge_sets = [set(frozenset((u, v)) for u, v in zip(path[:-1], path[1:]))
                 for path in node_paths]
    overlaps = sum(len(edge_sets[i] & edge_sets[j])
                   for i in range(len(edge_sets)) for j in range(i + 1, len(edge_sets)))
    if overlaps:
        raise RuntimeError(f"edge-disjoint path verification failed: {overlaps} shared edges")
    print(f"[routes] start={start_node} end={end_node} connectivity={connectivity}; "
          f"selected={len(node_paths)}; shared_edges=0")
    args._resolved_source = source
    return [reconstruct_route_xy(graph_multi, path, ds=args.ds_path)
            for path in node_paths]


def _geojson_lines(path: Path, default_crs: str = "EPSG:4326") -> tuple[list[np.ndarray], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_crs = default_crs
    crs_info = data.get("crs") if isinstance(data, dict) else None
    if isinstance(crs_info, dict):
        source_crs = crs_info.get("properties", {}).get("name", source_crs)
    if data.get("type") == "FeatureCollection":
        geometries = [feature.get("geometry") for feature in data.get("features", [])]
    elif data.get("type") == "Feature":
        geometries = [data.get("geometry")]
    else:
        geometries = [data]
    lines = []
    for geometry in geometries:
        if not geometry:
            continue
        if geometry.get("type") == "LineString":
            lines.append(np.asarray(geometry["coordinates"], dtype=float)[:, :2])
        elif geometry.get("type") == "MultiLineString":
            lines.extend(np.asarray(coords, dtype=float)[:, :2]
                         for coords in geometry["coordinates"])
    if not lines:
        raise ValueError(f"{path} contains no LineString or MultiLineString geometry")
    return lines, source_crs


def _gpx_lines(path: Path) -> list[np.ndarray]:
    root = ET.parse(path).getroot()
    lines = []
    for track in root.findall(".//{*}trk"):
        points = [(float(point.attrib["lon"]), float(point.attrib["lat"]))
                  for point in track.findall(".//{*}trkpt")]
        if len(points) >= 2:
            lines.append(np.asarray(points, dtype=float))
    for route in root.findall(".//{*}rte"):
        points = [(float(point.attrib["lon"]), float(point.attrib["lat"]))
                  for point in route.findall(".//{*}rtept")]
        if len(points) >= 2:
            lines.append(np.asarray(points, dtype=float))
    if not lines:
        raise ValueError(f"{path} contains no GPX track or route with at least two points")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/import TREC-Route input routes")
    parser.add_argument("--graphml", default="run_output/MMC/osm_paths/pedestrian_network.graphml")
    parser.add_argument("--output-dir", default="input/MMC/routes")
    parser.add_argument("--n-routes", type=int, default=3)
    parser.add_argument("--start-latlon", nargs=2, type=float, metavar=("LAT", "LON"))
    parser.add_argument("--end-latlon", nargs=2, type=float, metavar=("LAT", "LON"))
    parser.add_argument("--start-xy", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--end-xy", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--local-origin-x", type=float, default=0.0)
    parser.add_argument("--local-origin-y", type=float, default=0.0)
    parser.add_argument("--project-crs", default="EPSG:6346")
    parser.add_argument("--ds-path", type=float, default=1.0)
    parser.add_argument("--route-name-prefix", default="route")
    parser.add_argument(
        "--max-snap-distance-m", type=float, default=50.0,
        help="Maximum imported-waypoint distance from the OSM pedestrian graph")
    parser.add_argument(
        "--network-tolerance-m", type=float, default=0.5,
        help="Tolerance used to verify that route lines follow pedestrian edges")
    imports = parser.add_mutually_exclusive_group()
    imports.add_argument("--import-geojson", type=Path)
    imports.add_argument("--import-gpx", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes_dir = Path(args.output_dir)
    if args.validate_only:
        routes = load_routes_directory(routes_dir)
        network_routes = [
            route for route in routes
            if route["metadata"].get("network_constrained") is True
        ]
        if network_routes:
            validate_routes_on_pedestrian_graph(
                network_routes, args.graphml, args.network_tolerance_m)
        experimental_routes = [
            route for route in routes
            if route["metadata"].get("source") == "experimental_measurement"
        ]
        if experimental_routes:
            print(
                f"[routes] Preserved {len(experimental_routes)} measured trajectory/trajectories "
                "at their recorded coordinates; OSM map matching is intentionally not applied."
            )
        print(f"[routes] VALID: {len(routes)} route(s) in {routes_dir}")
        for route in routes:
            print(f"  route_{route['route_id']}: {len(route['xy'])} points, "
                  f"{route['length_m']:.3f} m, source={route['metadata']['source']}")
        return

    if args.n_routes < 1:
        raise SystemExit("ERROR: --n-routes must be at least one")
    if args.ds_path <= 0 or not np.isfinite(args.ds_path):
        raise SystemExit("ERROR: --ds-path must be finite and positive")
    if args.max_snap_distance_m <= 0 or not np.isfinite(args.max_snap_distance_m):
        raise SystemExit("ERROR: --max-snap-distance-m must be finite and positive")
    routes_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in list(route_csv_files(routes_dir)) + list(routes_dir.glob("route_*.json")):
            path.unlink()
        for name in ("routes_index.json", "route_polylines.pkl"):
            path = routes_dir / name
            if path.exists():
                path.unlink()
    existing_ids = [route_sort_key(path) for path in route_csv_files(routes_dir)]
    next_id = max(existing_ids, default=0) + 1
    origin = np.array([args.local_origin_x, args.local_origin_y], dtype=float)
    generator_args = _generator_args(args)

    if args.import_geojson:
        lines, source_crs = _geojson_lines(args.import_geojson)
        transformer = Transformer.from_crs(source_crs, args.project_crs, always_xy=True)
        graph = _load_pedestrian_graph(args.graphml)
        raw_routes = []
        for line in lines:
            x_proj, y_proj = transformer.transform(line[:, 0], line[:, 1])
            dense, length = map_match_to_pedestrian_graph(
                graph, np.column_stack([x_proj, y_proj]) - origin,
                args.ds_path, args.max_snap_distance_m)
            raw_routes.append((dense, length))
        source = "imported_geojson_osm_matched"
    elif args.import_gpx:
        lines = _gpx_lines(args.import_gpx)
        transformer = Transformer.from_crs("EPSG:4326", args.project_crs, always_xy=True)
        graph = _load_pedestrian_graph(args.graphml)
        raw_routes = []
        for line in lines:
            x_proj, y_proj = transformer.transform(line[:, 0], line[:, 1])
            dense, length = map_match_to_pedestrian_graph(
                graph, np.column_stack([x_proj, y_proj]) - origin,
                args.ds_path, args.max_snap_distance_m)
            raw_routes.append((dense, length))
        source = "imported_gpx_osm_matched"
    else:
        raw_routes = _select_edge_disjoint_routes(args)
        source = args._resolved_source

    existing_routes = load_routes_directory(routes_dir) if route_csv_files(routes_dir) else []
    accepted_routes = []
    comparison_geometries = [route["xy"] for route in existing_routes]
    for xy, length in raw_routes:
        if any(_same_route_geometry(xy, existing) for existing in comparison_geometries):
            print("[routes] Skipping duplicate route geometry already present in "
                  f"{routes_dir}")
            continue
        accepted_routes.append((xy, length))
        comparison_geometries.append(xy)

    for offset, (xy, length) in enumerate(accepted_routes):
        route_id = next_id + offset
        csv_path, _ = write_route(
            routes_dir, route_id, xy, length, args.ds_path, source,
            args.route_name_prefix, args.project_crs, origin, generator_args)
        print(f"[routes] Wrote {csv_path}: {len(xy)} points, {length:.3f} m")
    routes = load_routes_directory(routes_dir)
    validate_routes_on_pedestrian_graph(
        routes, args.graphml, args.network_tolerance_m)
    write_folder_outputs(routes_dir, routes)
    print(f"[routes_result] added={len(accepted_routes)} "
          f"duplicates_skipped={len(raw_routes) - len(accepted_routes)} "
          f"total={len(routes)} "
          f"output_dir={routes_dir}")


if __name__ == "__main__":
    main()
