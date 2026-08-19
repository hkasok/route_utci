#!/usr/bin/env python3
"""Fast synthetic verification for full-surface TREC-Route radiation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
import trimesh

from material_boundary_mesh import (_triangulate_piece,
                                    build_material_conforming_ground)
from urban_radiation_engine import UrbanRadiationEngine


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def street_canyon() -> trimesh.Trimesh:
    ground = trimesh.creation.box(extents=[20.0, 12.0, 0.2])
    ground.apply_translation([0.0, 0.0, -0.1])
    west = trimesh.creation.box(extents=[20.0, 1.0, 8.0])
    west.apply_translation([0.0, -5.5, 4.0])
    east = trimesh.creation.box(extents=[20.0, 1.0, 8.0])
    east.apply_translation([0.0, 5.5, 4.0])
    return trimesh.util.concatenate([ground, west, east])


def test_engine() -> None:
    mesh = street_canyon()
    n = len(mesh.faces)
    route = np.linalg.norm(mesh.triangles_center[:, :2], axis=1) < 7.0
    albedo = np.linspace(0.10, 0.45, n)
    engine = UrbanRadiationEngine(
        mesh, albedo, np.full(n, 0.95), route,
        ray_batch_size=10_000)
    svf = engine.precompute_sky_view_factor(16, 64)
    matrix = engine.precompute_route_view_factors(20.0, 1e-8, n)
    sun = np.array([0.2, -0.2, 0.96])
    sun /= np.linalg.norm(sun)
    result = engine.evaluate(
        sun_vector=sun, direct_normal_Wm2=800.0,
        diffuse_horizontal_Wm2=120.0, sky_temperature_K=285.0,
        surface_temperature_K=np.full(n, 305.0))
    check("SVF bounds", np.all((svf >= 0) & (svf <= 1)))
    check("CSR view factors", matrix.format == "csr" and matrix.shape == (n, n))
    check("route-only CSR rows", matrix[~route].nnz == 0)
    reconstructed = (result.sw_direct_absorbed_Wm2
                     + result.sw_diffuse_absorbed_Wm2
                     + result.longwave_net_Wm2)
    check("net-flux conservation", np.allclose(
        reconstructed, result.net_radiative_flux_Wm2, atol=1e-10))
    back = mesh.face_normals @ sun <= 0
    check("back-facing direct flux zero", np.all(result.sw_direct_absorbed_Wm2[back] == 0))
    check("four-point shade fractions", np.all(np.isin(
        result.sunlit_area_fraction * 4.0, np.arange(5))))
    cosine = np.maximum(mesh.face_normals @ sun, 0.0)
    check("material albedo controls absorbed direct flux", np.allclose(
        result.sw_direct_absorbed_Wm2,
        (1.0 - albedo) * cosine * 800.0 * result.sunlit_area_fraction))
    isothermal = engine.evaluate(
        sun_vector=np.array([0.0, 0.0, 1.0]), direct_normal_Wm2=0.0,
        diffuse_horizontal_Wm2=0.0, sky_temperature_K=300.0,
        surface_temperature_K=np.full(n, 300.0),
        canyon_temperature_K=300.0, sunlit_area_fraction=np.zeros(n))
    check("isothermal grey longwave enclosure", np.allclose(
        isothermal.longwave_net_Wm2, 0.0, atol=1e-8))
    check("nighttime shortwave zero", np.all(
        isothermal.sw_direct_absorbed_Wm2 + isothermal.sw_diffuse_absorbed_Wm2 == 0))


def test_material_boundary() -> None:
    concave = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    parent = np.array([[0, 0, 0], [3, 0, 0.3], [0, 3, 0.3]], dtype=float)
    concave_triangles = _triangulate_piece(concave, parent, 1e-10)
    concave_area = sum(0.5 * abs(
        (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
        - (triangle[1, 1] - triangle[0, 1]) * (triangle[2, 0] - triangle[0, 0]))
        for triangle in concave_triangles)
    check("constrained concave-piece coverage",
          abs(concave_area - concave.area) < 1e-10)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ground = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [2, 0, 0.2], [2, 2, 0.4], [0, 2, 0.2],
                               [0, 0, 1.0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3], [0, 4, 3]]), process=False)
        ground_path = root / "ground.stl"
        ground.export(ground_path)
        material_dir = root / "materials"
        material_dir.mkdir()
        frame = gpd.GeoDataFrame(
            {"assigned_material": ["generic_ground", "asphalt_road"]},
            geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)], crs="EPSG:3857")
        frame.to_file(material_dir / "osm_ground_materials.gpkg",
                      layer="final_ground_materials", driver="GPKG")
        catalog = {
            "material_names": ["generic_ground", "asphalt_road"],
            "materials": {
                "generic_ground": {"albedo": 0.18, "emissivity": 0.95},
                "asphalt_road": {"albedo": 0.12, "emissivity": 0.95}}}
        (material_dir / "ground_material_catalog.json").write_text(json.dumps(catalog))
        np.savez_compressed(material_dir / "ground_face_materials.npz",
                            material_id=np.zeros(3, dtype=np.int16))
        output = root / "out"
        report = build_material_conforming_ground(
            ground_path, material_dir, output, tolerance_m=1e-6)
        data = np.load(output / "ground_material_conforming.npz")
        check("material boundary splits terrain", report["final_faces"] > 3)
        check("vertical terrain edge face preserved",
              report["preserved_nonplanar_edge_faces"] == 1)
        check("both ground materials persist", set(data["material_id"].tolist()) == {0, 1})
        check("remesh area conservation", report["relative_surface_area_error"] < 5e-5,
              f"{report['relative_surface_area_error']:.3e}")
        vertices = data["vertices"][data["faces"]]
        crossing = ((vertices[:, :, 0].min(axis=1) < 1 - 1e-8)
                    & (vertices[:, :, 0].max(axis=1) > 1 + 1e-8))
        check("no face crosses material interface", not crossing.any())


def test_surface_model() -> None:
    spec = importlib.util.spec_from_file_location("urban_stage", "05d_urban_radiation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    materials = {"test": {"k": 1.0, "C": 2e6, "depth": 0.3,
                           "bottom_bc": "fixed"}}
    model = module.SurfaceConduction(np.array(["test"]), materials, 5, 300.0, 297.0)
    before = model.state[0, 0]
    after, sensible, steps = model.advance(
        600.0, np.array([500.0]), 300.0, 1.0, 5.7, 3.8, 0.3)
    check("five-node solar heating", after[0] > before and sensible[0] > 0)
    check("stable conduction substepping", steps >= 1 and np.isfinite(model.state).all())

    # Real vegetation exports can contain repeated-vertex triangles.  Stage 4
    # must sanitize only its derived radiation copy and keep face labels aligned.
    invalid = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]]),
        faces=np.array([[0, 1, 2], [1, 3, 3]]), process=False)
    labels = np.array(["tree_canopy", "tree_canopy"])
    cleaned, valid, report = module._filter_invalid_faces(invalid, "synthetic vegetation")
    check("invalid optional-radiation facet excluded",
          len(cleaned.faces) == 1 and report["excluded_invalid_faces"] == 1)
    check("face-aligned classifications remain aligned",
          labels[valid].tolist() == ["tree_canopy"])
    UrbanRadiationEngine(
        cleaned, np.array([0.18]), np.array([0.97]), np.array([True]))
    check("sanitized radiation mesh accepted", True)


def test_paraview_export() -> None:
    spec = importlib.util.spec_from_file_location("urban_stage", "05d_urban_radiation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from paraview_export import read_vtk_appended

    mesh = street_canyon()
    n_faces = len(mesh.faces)
    n_times = 4
    rng = np.random.default_rng(11)
    temperature_k = rng.uniform(290.0, 320.0, (n_times, n_faces)).astype(np.float32)
    net = rng.normal(50.0, 200.0, (n_times, n_faces)).astype(np.float32)
    sensible = rng.normal(20.0, 80.0, (n_times, n_faces)).astype(np.float32)
    sunlit = rng.uniform(0.0, 1.0, (n_times, n_faces)).astype(np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        material = np.where(mesh.triangles_center[:, 2] > 0.1,
                            "wall", "asphalt_road").astype(object)
        np.savez_compressed(
            out / "urban_surface_mesh.npz",
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            centroid=np.asarray(mesh.triangles_center),
            normal=np.asarray(mesh.face_normals),
            area=np.asarray(mesh.area_faces), material_name=material,
            object_class=np.full(n_faces, "ground", dtype=object),
            route_zone=np.ones(n_faces, dtype=bool),
            albedo=np.linspace(0.1, 0.4, n_faces),
            emissivity=np.full(n_faces, 0.95))
        np.save(out / "surface_temperature_K.npy", temperature_k)
        np.save(out / "net_radiative_flux_Wm2.npy", net)
        np.save(out / "sensible_heat_flux_Wm2.npy", sensible)
        np.save(out / "sunlit_area_fraction.npy", sunlit)
        np.save(out / "facet_sky_view_factor.npy",
                np.linspace(0.0, 1.0, n_faces))
        # 15-minute steps with a null stride must auto-select hourly frames.
        time_hours = 10.0 + 0.25 * np.arange(n_times)
        config = {"output": {"write_paraview": True,
                             "paraview_time_stride": None}}
        pvd = module.export_paraview(out, time_hours, config)
        check("hourly auto-stride collection", pvd.is_file()
              and pvd.read_text().count("<DataSet") == 1)
        frame = read_vtk_appended(out / "paraview" / "urban_surface_000.vtp")
        arrays = frame["arrays"]
        check("surface mesh geometry round-trip",
              np.array_equal(arrays["__points__"],
                             np.asarray(mesh.vertices, dtype=np.float32))
              and np.array_equal(arrays["connectivity"],
                                 np.asarray(mesh.faces).reshape(-1)))
        check("surface temperature exported in Celsius", np.allclose(
            arrays["surface_temperature_C"], temperature_k[0] - 273.15,
            atol=1e-4))
        check("radiation results attached per facet",
              np.array_equal(arrays["net_radiative_flux_Wm2"], net[0])
              and np.array_equal(arrays["sensible_heat_flux_Wm2"], sensible[0])
              and np.array_equal(arrays["sunlit_area_fraction"], sunlit[0]))
        static = read_vtk_appended(
            out / "paraview" / "urban_surface_static.vtp")["arrays"]
        legend = json.loads(
            (out / "paraview" / "paraview_legend.json").read_text())
        decoded = np.array([legend["material_id"][str(code)]
                            for code in static["material_id"]])
        check("material codes decode through the legend",
              (decoded == material.astype(str)).all())
        check("static optical and zone data present",
              np.allclose(static["albedo"], np.linspace(0.1, 0.4, n_faces),
                          atol=1e-7)
              and static["route_zone"].min() == 1
              and np.allclose(static["sky_view_factor"],
                              np.linspace(0.0, 1.0, n_faces), atol=1e-7))
        explicit = {"output": {"write_paraview": True,
                               "paraview_time_stride": 1}}
        pvd = module.export_paraview(out, time_hours, explicit)
        check("explicit stride exports every timestep",
              pvd.read_text().count("<DataSet") == n_times)


if __name__ == "__main__":
    test_engine()
    test_material_boundary()
    test_surface_model()
    test_paraview_export()
    print("All urban-radiation checks passed.")
