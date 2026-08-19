#!/usr/bin/env python3
"""Optional full-surface radiation and transient conduction stage.

This stage is intentionally confined to the optional microclimate branch.  It
creates a material-boundary-conforming terrain, combines it with buildings and
vegetation, precomputes full-facet sky visibility and route-zone mutual view
factors, then advances a five-node 1-D conduction model.  The resulting surface
temperatures and net radiative fluxes seed stage 05c's air heat sources.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
import trimesh

from material_boundary_mesh import build_material_conforming_ground
from thermal_common import DEFAULT_MATERIALS, SIGMA, sky_longwave_down, sun_vector_enu
from urban_radiation_engine import UrbanRadiationEngine


DEFAULT_CONFIG = {
    "enabled": True,
    "mesh": {"boundary_tolerance_m": 0.01, "include_vegetation": True},
    "route_zone": {"buffer_m": 35.0, "view_factor_maximum_distance_m": 40.0,
                   "view_factor_cutoff": 1.0e-5, "maximum_neighbors": 96},
    "ray_tracing": {"sky_directions": 64, "facet_batch_size": 4000,
                    "ray_batch_size": 200000, "surface_offset_m": 0.002},
    "surface_energy": {"nodes": 5, "spinup_days": 2,
                       "maximum_fourier_number": 0.30,
                       "convection_a_Wm2K": 5.7, "convection_b_Wm2K_per_ms": 3.8,
                       "building_interior_temperature_C": 24.0,
                       "minimum_temperature_K": 220.0,
                       "maximum_temperature_K": 380.0},
    "output": {"write_components": False,
               "write_paraview": True, "paraview_time_stride": None},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-facet TREC-Route urban radiation")
    parser.add_argument("--buildings-stl", required=True)
    parser.add_argument("--vegetation-stl", required=True)
    parser.add_argument("--ground-stl", required=True)
    parser.add_argument("--ground-material-dir", required=True)
    parser.add_argument("--mrt-dir", required=True, help="Stage-05 prep with times/path_xyz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="urban_radiation_config.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--no-paraview", action="store_true",
                        help="Skip the ParaView (.vtp/.pvd) surface export")
    return parser.parse_args()


def _merge(base: dict, update: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    supplied = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config = _merge(DEFAULT_CONFIG, supplied)
    if not 3 <= int(config["surface_energy"]["nodes"]) <= 5:
        raise ValueError("urban-radiation surface conduction requires 3 to 5 nodes")
    if int(config["surface_energy"].get("spinup_days", 2)) < 0:
        raise ValueError("urban-radiation spinup_days cannot be negative")
    if int(config["ray_tracing"]["sky_directions"]) < 4:
        raise ValueError("urban-radiation sky_directions must be at least four")
    return config


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest()}


def material_layer_identity(path: str | Path) -> dict:
    """Content fingerprint of the relevant GPKG layer, ignoring diagnostics."""
    import geopandas as gpd
    frame = gpd.read_file(path, layer="final_ground_materials")
    digest = hashlib.sha256()
    digest.update(str(frame.crs).encode())
    for row in frame.sort_values("assigned_material").itertuples():
        digest.update(str(row.assigned_material).encode())
        digest.update(row.geometry.wkb)
    return {"path": str(Path(path).resolve()), "layer": "final_ground_materials",
            "feature_count": int(len(frame)), "sha256": digest.hexdigest()}


def _mesh_with_arrays(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _filter_invalid_faces(mesh: trimesh.Trimesh, component: str,
                          minimum_area_m2: float = 1.0e-12
                          ) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    """Remove only unusable facets from the optional radiation-scene copy.

    Source STL files are never changed.  The returned Boolean mask lets every
    face-aligned material/object array be filtered identically.
    """
    triangles = np.asarray(mesh.triangles, dtype=float)
    finite = np.isfinite(triangles).all(axis=(1, 2))
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = finite & np.isfinite(areas) & (areas > float(minimum_area_m2))
    removed = int((~valid).sum())
    if not valid.any():
        raise ValueError(f"{component} mesh contains no valid radiation facets")
    if removed:
        print(f"  {component}: excluded {removed:,}/{len(valid):,} "
              "non-finite or zero-area facets from optional radiation mesh")
    cleaned = _mesh_with_arrays(
        np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)[valid])
    cleaned.remove_unreferenced_vertices()
    report = {
        "source_faces": int(len(valid)),
        "retained_faces": int(valid.sum()),
        "excluded_invalid_faces": removed,
        "minimum_retained_area_m2": float(cleaned.area_faces.min()),
    }
    return cleaned, valid, report


def assemble_scene(args: argparse.Namespace, config: dict, output: Path):
    conforming = output / "material_conforming_ground"
    product = conforming / "ground_material_conforming.npz"
    material_source = Path(args.ground_material_dir)
    remesh_context = {
        "ground_stl": file_identity(args.ground_stl),
        "material_polygons": material_layer_identity(
            material_source / "osm_ground_materials.gpkg"),
        "material_catalog": file_identity(material_source / "ground_material_catalog.json"),
        "face_map": file_identity(material_source / "ground_face_materials.npz"),
        "boundary_tolerance_m": float(config["mesh"]["boundary_tolerance_m"]),
    }
    context_path = conforming / "material_conforming_context.json"
    old_context = (json.loads(context_path.read_text()) if context_path.is_file() else None)
    if args.force or not product.is_file() or old_context != remesh_context:
        print("Partitioning terrain triangles at OSM material boundaries...")
        report = build_material_conforming_ground(
            args.ground_stl, args.ground_material_dir, conforming,
            tolerance_m=float(config["mesh"]["boundary_tolerance_m"]))
        print(f"  {report['source_faces']:,} -> {report['final_faces']:,} ground faces; "
              f"area error {report['relative_surface_area_error']:.3e}")
        context_path.write_text(json.dumps(remesh_context, indent=2))
    ground_data = np.load(product)
    ground = _mesh_with_arrays(ground_data["vertices"], ground_data["faces"])
    ground_ids = ground_data["material_id"].astype(int)
    catalog = json.loads((conforming / "ground_material_catalog.json").read_text())
    names = list(catalog["material_names"])
    materials = dict(catalog["materials"])

    sanitation = {}
    ground, ground_valid, sanitation["ground"] = _filter_invalid_faces(
        ground, "material-conforming ground")
    ground_ids = ground_ids[ground_valid]

    building = trimesh.load(args.buildings_stl, force="mesh", process=False)
    building, _building_valid, sanitation["buildings"] = _filter_invalid_faces(
        building, "buildings")
    building_names = np.where(np.abs(building.face_normals[:, 2]) > 0.5, "roof", "wall")
    materials.update({key: dict(DEFAULT_MATERIALS[key]) for key in ("wall", "roof")})
    meshes = [ground, building]
    face_names = [np.asarray(names, dtype=object)[ground_ids], building_names.astype(object)]
    object_class = [np.full(len(ground.faces), "ground", dtype=object),
                    np.full(len(building.faces), "building", dtype=object)]
    if config["mesh"].get("include_vegetation", True):
        vegetation = trimesh.load(args.vegetation_stl, force="mesh", process=False)
        if len(vegetation.faces):
            vegetation, _vegetation_valid, sanitation["vegetation"] = \
                _filter_invalid_faces(vegetation, "vegetation")
            meshes.append(vegetation)
            face_names.append(np.full(len(vegetation.faces), "tree_canopy", dtype=object))
            object_class.append(np.full(len(vegetation.faces), "vegetation", dtype=object))
            materials["tree_canopy"] = {
                "albedo": 0.18, "emissivity": 0.97, "k": 0.40,
                "C": 1.50e6, "depth": 0.08, "bottom_bc": "ambient"}
    scene = trimesh.util.concatenate(meshes)
    names_by_face = np.concatenate(face_names).astype(str)
    classes = np.concatenate(object_class).astype(str)
    for name in np.unique(names_by_face):
        if name not in materials:
            raise KeyError(f"no thermal/radiative properties for material {name!r}")
    return scene, names_by_face, classes, materials, sanitation


def route_zone_mask(centroids: np.ndarray, route_points: np.ndarray,
                    buffer_m: float) -> np.ndarray:
    if buffer_m <= 0:
        raise ValueError("route-zone buffer must be positive")
    tree = cKDTree(np.asarray(route_points, dtype=float))
    distance, _ = tree.query(centroids, k=1, workers=-1)
    return distance <= buffer_m


class SurfaceConduction:
    """Vectorized 3-to-5 node explicit finite-difference substrate model."""

    def __init__(self, material_names: np.ndarray, materials: dict, nodes: int,
                 initial_temperature_K: float, building_interior_K: float):
        self.nodes = int(nodes)
        self.names = np.asarray(material_names)
        self.k = np.array([materials[name].get("k", materials[name].get(
            "thermal_conductivity", 1.0)) for name in self.names], dtype=float)
        self.capacity = np.array([materials[name].get("C", materials[name].get(
            "density", 2000.0) * materials[name].get("specific_heat", 1000.0))
                                  for name in self.names], dtype=float)
        self.depth = np.array([materials[name].get("depth", 0.3)
                               for name in self.names], dtype=float)
        self.dx = self.depth / (self.nodes - 1)
        self.alpha = self.k / self.capacity
        self.state = np.full((len(self.names), self.nodes), initial_temperature_K, dtype=float)
        self.fixed_back = np.full(len(self.names), initial_temperature_K, dtype=float)
        building = np.isin(self.names, ["wall", "roof"])
        self.fixed_back[building] = building_interior_K
        self.fixed_mask = np.array([
            materials[name].get("bottom_bc", "fixed") in ("fixed", "interior")
            for name in self.names], dtype=bool)

    def advance(self, dt_s: float, net_radiation_Wm2: np.ndarray,
                air_temperature_K: float, wind_speed_ms: float,
                convection_a: float, convection_b: float,
                maximum_fourier: float) -> tuple[np.ndarray, np.ndarray, int]:
        maximum_dt = maximum_fourier * np.min(self.dx**2 / self.alpha)
        steps = max(1, int(np.ceil(dt_s / maximum_dt)))
        sub_dt = dt_s / steps
        h = convection_a + convection_b * max(0.0, wind_speed_ms)
        for _ in range(steps):
            old = self.state.copy()
            conduction = self.k * (old[:, 1] - old[:, 0]) / self.dx
            sensible = h * (old[:, 0] - air_temperature_K)
            surface_capacity = self.capacity * self.dx * 0.5
            self.state[:, 0] = old[:, 0] + sub_dt * (
                net_radiation_Wm2 - sensible + conduction) / surface_capacity
            factor = self.alpha * sub_dt / self.dx**2
            self.state[:, 1:-1] = old[:, 1:-1] + factor[:, None] * (
                old[:, :-2] - 2.0 * old[:, 1:-1] + old[:, 2:])
            self.state[:, -1] = old[:, -1] + factor * (old[:, -2] - old[:, -1])
            self.state[self.fixed_mask, -1] = self.fixed_back[self.fixed_mask]
        return self.state[:, 0], h * (self.state[:, 0] - air_temperature_K), steps


def export_paraview(out: Path, time_hours: np.ndarray, config: dict) -> Path:
    """Write the full-surface radiation results as a ParaView time series.

    Reads the completed products under ``out`` so the export always matches
    the arrays downstream stages consume.  ``urban_surface_static.vtp``
    carries the mesh with time-independent per-facet data (material,
    optical properties, route zone, sky-view factor);
    ``urban_surface_<it>.vtp`` frames carry surface temperature and the
    radiation/sensible fluxes, bound together by ``urban_radiation.pvd``.
    ``paraview_legend.json`` maps the integer material/class codes back to
    names, because VTK cell arrays cannot store strings.

    ``output.paraview_time_stride`` selects every Nth timestep; the default
    (null) exports roughly hourly frames, since each frame must embed the
    full mesh and the real problem has millions of facets.
    """
    from paraview_export import write_pvd, write_vtp

    mesh_data = np.load(out / "urban_surface_mesh.npz", allow_pickle=True)
    vertices = mesh_data["vertices"].astype(np.float32)
    faces = mesh_data["faces"]
    material_names, material_id = np.unique(
        mesh_data["material_name"].astype(str), return_inverse=True)
    class_names, class_id = np.unique(
        mesh_data["object_class"].astype(str), return_inverse=True)
    static = {"material_id": material_id.astype(np.int32),
              "object_class_id": class_id.astype(np.int32),
              "albedo": mesh_data["albedo"].astype(np.float32),
              "emissivity": mesh_data["emissivity"].astype(np.float32),
              "route_zone": mesh_data["route_zone"].astype(np.uint8)}
    svf_path = out / "facet_sky_view_factor.npy"
    if svf_path.is_file():
        static["sky_view_factor"] = np.load(svf_path).astype(np.float32)

    target = out / "paraview"
    target.mkdir(parents=True, exist_ok=True)
    (target / "paraview_legend.json").write_text(json.dumps({
        "material_id": {int(i): str(name)
                        for i, name in enumerate(material_names)},
        "object_class_id": {int(i): str(name)
                            for i, name in enumerate(class_names)},
        "flux_sign": "positive into surface",
        "time_values": "local hours of day"}, indent=2))
    write_vtp(target / "urban_surface_static.vtp", vertices, faces,
              cell_data=static)

    dynamic = {name: np.load(out / f"{name}.npy", mmap_mode="r")
               for name in ("surface_temperature_K", "net_radiative_flux_Wm2",
                            "sensible_heat_flux_Wm2", "sunlit_area_fraction")}
    for name in ("sw_direct_absorbed_Wm2", "sw_diffuse_absorbed_Wm2",
                 "longwave_net_Wm2"):
        if (out / f"{name}.npy").is_file():
            dynamic[name] = np.load(out / f"{name}.npy", mmap_mode="r")
    time_hours = np.asarray(time_hours, dtype=float)
    stride = config["output"].get("paraview_time_stride")
    if stride is None:
        spacing = (float(np.median(np.diff(time_hours)))
                   if len(time_hours) > 1 else 1.0)
        stride = int(round(1.0 / max(spacing, 1e-9)))
    stride = max(1, int(stride))
    entries = []
    for it in range(0, len(time_hours), stride):
        frame = {}
        for name, matrix in dynamic.items():
            values = np.asarray(matrix[it], dtype=np.float32)
            if name == "surface_temperature_K":
                name, values = "surface_temperature_C", values - 273.15
            frame[name] = values
        name = f"urban_surface_{it:03d}.vtp"
        write_vtp(target / name, vertices, faces, cell_data=frame)
        entries.append((float(time_hours[it]), name))
    write_pvd(target / "urban_radiation.pvd", entries)
    print(f"  ParaView export: {len(entries)} frame(s), stride {stride} "
          f"-> {target / 'urban_radiation.pvd'}")
    return target / "urban_radiation.pvd"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.no_paraview:
        config["output"]["write_paraview"] = False
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mrt = Path(args.mrt_dir)
    times_path, routes_path = mrt / "times.csv", mrt / "path_xyz.npy"
    for required in (Path(args.buildings_stl), Path(args.vegetation_stl),
                     Path(args.ground_stl), times_path, routes_path,
                     Path(args.ground_material_dir) / "osm_ground_materials.gpkg"):
        if not required.is_file():
            raise FileNotFoundError(f"urban-radiation input missing: {required}")
    times = pd.read_csv(times_path)
    required_columns = {"azimuth_deg", "elevation_deg", "DNI_Wm2", "DHI_Wm2",
                        "air_temp_C", "rh_pct", "wind_ms"}
    if missing := required_columns - set(times.columns):
        raise ValueError(f"times.csv lacks urban-radiation columns: {sorted(missing)}")
    routes = np.load(routes_path)

    print("Assembling material-conforming full-domain radiation mesh...")
    mesh, material_name, object_class, materials, sanitation = assemble_scene(
        args, config, out)
    albedo = np.array([materials[name]["albedo"] for name in material_name], dtype=float)
    emissivity = np.array([materials[name]["emissivity"] for name in material_name], dtype=float)
    rz = route_zone_mask(mesh.triangles_center, routes,
                         float(config["route_zone"]["buffer_m"]))
    print(f"  Facets: {len(mesh.faces):,}; route zone: {rz.sum():,}; "
          f"ambient zone: {(~rz).sum():,}")
    engine = UrbanRadiationEngine(
        mesh, albedo, emissivity, rz,
        surface_offset_m=float(config["ray_tracing"]["surface_offset_m"]),
        ray_batch_size=int(config["ray_tracing"]["ray_batch_size"]))
    started = time.time()
    precompute_context = {
        "buildings_stl": file_identity(args.buildings_stl),
        "vegetation_stl": file_identity(args.vegetation_stl),
        "ground_stl": file_identity(args.ground_stl),
        "ground_material_catalog": file_identity(
            Path(args.ground_material_dir) / "ground_material_catalog.json"),
        "ground_material_polygons": material_layer_identity(
            Path(args.ground_material_dir) / "osm_ground_materials.gpkg"),
        "path_xyz": file_identity(routes_path),
        "mesh_config": config["mesh"], "route_zone_config": config["route_zone"],
        "ray_tracing_config": config["ray_tracing"],
        "mesh_sanitation": sanitation,
    }
    precompute_context_path = out / "urban_radiation_precompute_context.json"
    cached_context = (json.loads(precompute_context_path.read_text())
                      if precompute_context_path.is_file() else None)
    svf_path = out / "facet_sky_view_factor.npy"
    vf_path = out / "route_zone_view_factors.npz"
    if (not args.force and cached_context == precompute_context
            and svf_path.is_file() and vf_path.is_file()):
        engine.sky_view_factor = np.load(svf_path).astype(float)
        engine.view_factors = sparse.load_npz(vf_path).tocsr()
        if (engine.sky_view_factor.shape != (len(mesh.faces),)
                or engine.view_factors.shape != (len(mesh.faces), len(mesh.faces))):
            raise ValueError("cached urban-radiation geometry has invalid dimensions")
        print("Reusing exact-match full-surface SVF and route view-factor cache.")
    else:
        print("Precomputing full-domain facet sky-view factors...")
        engine.precompute_sky_view_factor(
            int(config["ray_tracing"]["sky_directions"]),
            int(config["ray_tracing"]["facet_batch_size"]))
        print("Precomputing ray-visible route-zone view-factor CSR matrix...")
        engine.precompute_route_view_factors(
            float(config["route_zone"]["view_factor_maximum_distance_m"]),
            float(config["route_zone"]["view_factor_cutoff"]),
            int(config["route_zone"]["maximum_neighbors"]))
        engine.save_precomputation(out)
        precompute_context_path.write_text(json.dumps(precompute_context, indent=2))
    np.savez_compressed(out / "urban_surface_mesh.npz",
                        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                        centroid=np.asarray(mesh.triangles_center), normal=np.asarray(mesh.face_normals),
                        area=np.asarray(mesh.area_faces), material_name=material_name,
                        object_class=object_class, route_zone=rz,
                        albedo=albedo, emissivity=emissivity)
    if args.precompute_only:
        print(f"[urban_radiation_precompute] facets={len(mesh.faces)} "
              f"seconds={time.time()-started:.1f} output_dir={out}")
        return 0

    n_times, n_faces = len(times), len(mesh.faces)
    temperature_writer = np.lib.format.open_memmap(
        out / "surface_temperature_K.npy", mode="w+", dtype=np.float32,
        shape=(n_times, n_faces))
    net_writer = np.lib.format.open_memmap(
        out / "net_radiative_flux_Wm2.npy", mode="w+", dtype=np.float32,
        shape=(n_times, n_faces))
    sensible_writer = np.lib.format.open_memmap(
        out / "sensible_heat_flux_Wm2.npy", mode="w+", dtype=np.float32,
        shape=(n_times, n_faces))
    sunlit_writer = np.lib.format.open_memmap(
        out / "sunlit_area_fraction.npy", mode="w+", dtype=np.float32,
        shape=(n_times, n_faces))
    writers = [temperature_writer, net_writer, sensible_writer, sunlit_writer]
    component_writers = {}
    if config["output"].get("write_components", False):
        for name in ("sw_direct_absorbed_Wm2", "sw_diffuse_absorbed_Wm2",
                     "longwave_net_Wm2"):
            component_writers[name] = np.lib.format.open_memmap(
                out / f"{name}.npy", mode="w+", dtype=np.float32,
                shape=(n_times, n_faces))

    nodes = int(config["surface_energy"]["nodes"])
    surface = SurfaceConduction(
        material_name, materials, nodes,
        float(times["air_temp_C"].mean()) + 273.15,
        float(config["surface_energy"]["building_interior_temperature_C"]) + 273.15)
    timestamp = pd.to_datetime(times["time"])
    dt = float(timestamp.diff().dt.total_seconds().dropna().median())
    if not 1 <= dt <= 7200:
        raise ValueError(f"invalid full-surface radiation timestep: {dt:g} s")
    spinup_days = int(config["surface_energy"].get("spinup_days", 2))
    if spinup_days < 0:
        raise ValueError("surface-energy spinup_days cannot be negative")
    rows = []
    for cycle in range(spinup_days + 1):
        save_cycle = cycle == spinup_days
        print(f"  full-surface conduction cycle {cycle+1}/{spinup_days+1}"
              + (" (saved)" if save_cycle else " (spin-up)"))
        for it, row in times.iterrows():
            sun = sun_vector_enu(row.azimuth_deg, row.elevation_deg)
            if cycle == 0:
                sunlit = (engine.direct_sunlit_fraction(sun) if row.elevation_deg > 0
                          and row.DNI_Wm2 > 0 else np.zeros(n_faces))
                sunlit_writer[it] = sunlit.astype(np.float32)
                print(f"  full-surface radiation {it+1}/{n_times} -- solar visibility")
            else:
                sunlit = np.asarray(sunlit_writer[it], dtype=float)
            lwin = (float(row.LWin_Wm2)
                    if "LWin_Wm2" in times and np.isfinite(row.LWin_Wm2)
                    else float(sky_longwave_down(
                        row.air_temp_C, row.rh_pct,
                        float(row.cloud_fraction) if "cloud_fraction" in times else 0.0)))
            sky_temperature = (lwin / SIGMA) ** 0.25
            radiative = engine.evaluate(
                sun_vector=sun, direct_normal_Wm2=float(row.DNI_Wm2),
                diffuse_horizontal_Wm2=float(row.DHI_Wm2),
                sky_temperature_K=sky_temperature,
                surface_temperature_K=surface.state[:, 0],
                sunlit_area_fraction=sunlit)
            surface_temperature, sensible, substeps = surface.advance(
                dt, radiative.net_radiative_flux_Wm2,
                float(row.air_temp_C) + 273.15, float(row.wind_ms),
                float(config["surface_energy"]["convection_a_Wm2K"]),
                float(config["surface_energy"]["convection_b_Wm2K_per_ms"]),
                float(config["surface_energy"]["maximum_fourier_number"]))
            radiative = engine.evaluate(
                sun_vector=sun, direct_normal_Wm2=float(row.DNI_Wm2),
                diffuse_horizontal_Wm2=float(row.DHI_Wm2),
                sky_temperature_K=sky_temperature,
                surface_temperature_K=surface_temperature,
                sunlit_area_fraction=sunlit)
            minimum = float(config["surface_energy"]["minimum_temperature_K"])
            maximum = float(config["surface_energy"]["maximum_temperature_K"])
            if np.any((surface_temperature < minimum) | (surface_temperature > maximum)):
                raise RuntimeError(f"surface temperature outside physical guard at timestep {it}")
            if not save_cycle:
                continue
            temperature_writer[it] = surface_temperature.astype(np.float32)
            net_writer[it] = radiative.net_radiative_flux_Wm2.astype(np.float32)
            sensible_writer[it] = sensible.astype(np.float32)
            for name, writer in component_writers.items():
                writer[it] = getattr(radiative, name).astype(np.float32)
            rows.append({"time": row["time"], "mean_surface_temperature_C":
                         float(np.average(surface_temperature - 273.15, weights=mesh.area_faces)),
                         "minimum_surface_temperature_C": float(surface_temperature.min() - 273.15),
                         "maximum_surface_temperature_C": float(surface_temperature.max() - 273.15),
                         "mean_net_radiative_flux_Wm2": float(np.average(
                             radiative.net_radiative_flux_Wm2, weights=mesh.area_faces)),
                         "route_zone_mean_net_radiative_flux_Wm2": float(np.mean(
                             radiative.net_radiative_flux_Wm2[rz])) if rz.any() else np.nan,
                         "conduction_substeps": substeps})
    for writer in writers + list(component_writers.values()):
        writer.flush()
    pd.DataFrame(rows).to_csv(out / "urban_radiation_summary_by_time.csv", index=False)
    metadata = {
        "format_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TREC-Route full-surface hybrid urban radiation",
        "flux_sign": "positive into surface",
        "n_faces": n_faces, "n_times": n_times,
        "route_zone_faces": int(rz.sum()), "ambient_zone_faces": int((~rz).sum()),
        "view_factor_nonzeros": int(engine.view_factors.nnz),
        "surface_conduction_nodes": nodes,
        "surface_spinup_days": spinup_days,
        "mesh_sanitation": sanitation,
        "inputs": {"buildings_stl": file_identity(args.buildings_stl),
                   "vegetation_stl": file_identity(args.vegetation_stl),
                   "ground_stl": file_identity(args.ground_stl),
                   "times": file_identity(times_path), "routes": file_identity(routes_path)},
        "config": config,
        "route_geometry_modified": False,
        "standard_route_MRT_physics_modified": False,
    }
    (out / "urban_radiation_metadata.json").write_text(json.dumps(metadata, indent=2))
    if config["output"].get("write_paraview", True):
        time_hours = np.array(
            [value.hour + value.minute / 60 + value.second / 3600
             for value in timestamp], dtype=float)
        export_paraview(out, time_hours, config)
    print(f"[urban_radiation_result] facets={n_faces} times={n_times} "
          f"seconds={time.time()-started:.1f} output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
