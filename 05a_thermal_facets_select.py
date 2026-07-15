"""
05a_thermal_facets_select.py -- select the ROUTE-RELEVANT surface facets
and build the longwave view matrix that links pedestrian route points to
those facets.

Why this stage exists
---------------------
The plan (agreed with the physics discussion):
  * FULL geometry is still used for shadow casting (unchanged, in 05/05b).
  * Only surfaces VISIBLE from the route -- within a capped distance --
    get a surface-temperature energy balance (05b). View factors decay
    quickly with distance, so a facade 500 m away contributes ~nothing to
    the longwave a pedestrian receives; solving its temperature is wasted
    work. The visible-set restriction is exactly what makes the
    route-based approach cheaper than a SOLWEIG-style full-domain map.

What it does
------------
From each (optionally strided) route point, a full SPHERE of rays is cast
against buildings + ground + vegetation:

  no hit, upward   -> sky        (weight accumulated per point)
  no hit, downward -> "default"  (falls back to the legacy surround term)
  hit beyond --max-distance      -> "default" (distance cull)
  vegetation hit   -> vegetation (radiates at ~air temperature in 05)
  building/ground hit within range -> that triangle becomes a THERMAL
        FACET; the (point, facet) pair gets the ray's weight in a sparse
        view matrix W.

The union of all hit triangles IS the route-relevant facet set: nothing
invisible from the route is ever solved. Facet outward normals are
oriented toward the side the rays arrived from (i.e., toward the air).

Outputs (in --output-dir)
-------------------------
  facets.npz            centroids, oriented normals, areas, class codes,
                        (mesh_id, face_id) provenance
  lw_view_matrix.npz    scipy CSR, (n_coarse_points x n_facets); row sums
                        + sky + veg + default weights = 1 exactly
  lw_point_weights.npz  w_sky, w_veg, w_default per coarse point
  point_map.npy         index of the coarse point serving each FULL-
                        resolution route point (nearest neighbor)
  selection_report.txt  human-readable sanity numbers

Verification hooks
------------------
  * Row sums checked == 1 to 1e-9 before saving (hard assert).
  * verify_thermal_pipeline.py T1/T2 exercise this matrix analytically.

Run:
    python3 05a_thermal_facets_select.py \
        --buildings-stl .../building_final.stl \
        --vegetation-stl .../vegetation_final.stl \
        --ground-stl .../ground_and_water_final.stl \
        --mrt-dir mrt_network_output/ \
        --output-dir thermal_facets_output/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh

from thermal_common import (CLASS_GROUND, CLASS_ROOF, CLASS_WALL,
                            get_intersector, make_sphere_directions,
                            nearest_hit_multi)

MESH_BUILDINGS = 0
MESH_GROUND = 1
MESH_VEGETATION = 2


def parse_args():
    p = argparse.ArgumentParser(description="Select route-visible thermal facets "
                                            "and build the LW view matrix")
    p.add_argument("--buildings-stl", required=True)
    p.add_argument("--vegetation-stl", required=True)
    p.add_argument("--ground-stl", required=True)
    p.add_argument("--mrt-dir", required=True,
                   help="Output dir of 05_mrt_network_raytrace.py "
                        "(needs path_xyz.npy)")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--point-stride", type=int, default=8,
                   help="Use every Nth route point for the LW view raytrace "
                        "(default 8 -> ~2 m at 0.25 m path sampling). The LW "
                        "environment varies smoothly, so full 0.25 m "
                        "resolution buys nothing but cost; every full-"
                        "resolution point is mapped to its nearest traced "
                        "point via point_map.npy.")
    p.add_argument("--n-lw-azimuth", type=int, default=24)
    p.add_argument("--n-lw-elevation", type=int, default=18,
                   help="Elevation bands over the FULL sphere (-90..+90 deg)")
    p.add_argument("--body-model", choices=["cylinder", "sphere"],
                   default="cylinder",
                   help="Angular weighting of the human receptor for LW "
                        "(default cylinder = standing person)")
    p.add_argument("--max-distance", type=float, default=300.0,
                   help="Distance cull for LW-relevant surfaces, meters "
                        "(default 300). Hits beyond this fall back to the "
                        "legacy surround term.")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Coarse points per ray batch")
    p.add_argument("--roof-normal-z", type=float, default=0.7,
                   help="Building faces with oriented normal z above this "
                        "are classed 'roof', otherwise 'wall'")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mrt_dir = Path(args.mrt_dir)

    print("=" * 70)
    print("Loading geometry...")
    meshes = {
        MESH_BUILDINGS: trimesh.load(args.buildings_stl, force="mesh"),
        MESH_GROUND: trimesh.load(args.ground_stl, force="mesh"),
        MESH_VEGETATION: trimesh.load(args.vegetation_stl, force="mesh"),
    }
    intersectors = [(mid, get_intersector(m)) for mid, m in meshes.items()]

    path_xyz = np.load(mrt_dir / "path_xyz.npy")
    n_full = len(path_xyz)
    coarse_idx = np.arange(0, n_full, max(1, args.point_stride))
    coarse_xyz = path_xyz[coarse_idx]
    n_coarse = len(coarse_xyz)
    print(f"  Route points: {n_full:,} full -> {n_coarse:,} traced "
          f"(stride {args.point_stride})")

    # Map every full-resolution point to its nearest traced point
    from scipy.spatial import cKDTree
    _, point_map = cKDTree(coarse_xyz).query(path_xyz)
    point_map = point_map.astype(np.int32)

    directions, weights = make_sphere_directions(
        args.n_lw_azimuth, args.n_lw_elevation, body=args.body_model)
    ndirs = len(directions)
    print(f"  LW directions per point: {ndirs} "
          f"(body model: {args.body_model})")
    assert abs(weights.sum() - 1.0) < 1e-12

    # ------------------------------------------------------------------
    # Full-sphere raytrace, accumulating the sparse view matrix
    # ------------------------------------------------------------------
    facet_key_to_col = {}          # (mesh_id, face_id) -> compact column
    facet_orient_sign = []         # +1 keep mesh normal, -1 flip (per facet)
    rows, cols, vals = [], [], []
    w_sky = np.zeros(n_coarse)
    w_veg = np.zeros(n_coarse)
    w_default = np.zeros(n_coarse)

    face_normals = {mid: np.asarray(m.face_normals) for mid, m in meshes.items()}

    t0 = time.time()
    n_batches = int(np.ceil(n_coarse / args.batch_size))
    for bi, start in enumerate(range(0, n_coarse, args.batch_size)):
        end = min(start + args.batch_size, n_coarse)
        pts = coarse_xyz[start:end]
        m = len(pts)
        origins = np.repeat(pts, ndirs, axis=0)
        dirs = np.tile(directions, (m, 1))
        wts = np.tile(weights, m)
        pt_of_ray = np.repeat(np.arange(start, end), ndirs)

        hit_mesh, hit_face, _ = nearest_hit_multi(
            intersectors, origins, dirs, args.max_distance)

        no_hit = hit_mesh < 0
        up = dirs[:, 2] > 0.0
        np.add.at(w_sky, pt_of_ray[no_hit & up], wts[no_hit & up])
        np.add.at(w_default, pt_of_ray[no_hit & ~up], wts[no_hit & ~up])
        veg = hit_mesh == MESH_VEGETATION
        np.add.at(w_veg, pt_of_ray[veg], wts[veg])

        solid = (hit_mesh == MESH_BUILDINGS) | (hit_mesh == MESH_GROUND)
        s_idx = np.where(solid)[0]
        for ri in s_idx:
            key = (int(hit_mesh[ri]), int(hit_face[ri]))
            col = facet_key_to_col.get(key)
            if col is None:
                col = len(facet_key_to_col)
                facet_key_to_col[key] = col
                # Orient outward normal toward the ray origin side: the ray
                # travels along dirs[ri] and hits the face, so the outward
                # (air-side) normal must oppose the ray direction.
                nrm = face_normals[key[0]][key[1]]
                facet_orient_sign.append(
                    -1.0 if float(np.dot(nrm, dirs[ri])) > 0.0 else 1.0)
            rows.append(pt_of_ray[ri])
            cols.append(col)
            vals.append(wts[ri])

        if (bi + 1) % max(1, n_batches // 20) == 0 or bi == n_batches - 1:
            el = time.time() - t0
            print(f"  batch {bi + 1}/{n_batches} -- {el:.0f}s elapsed, "
                  f"{len(facet_key_to_col):,} facets so far")

    n_facets = len(facet_key_to_col)
    W = sp.coo_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n_coarse, n_facets)).tocsr()
    W.sum_duplicates()

    # ------------------------------------------------------------------
    # HARD VERIFICATION: weights must partition unity at every point
    # ------------------------------------------------------------------
    total = np.asarray(W.sum(axis=1)).ravel() + w_sky + w_veg + w_default
    err = np.abs(total - 1.0).max()
    print(f"\n  Weight-partition check: max |sum - 1| = {err:.2e}")
    assert err < 1e-9, "View weights do not partition unity -- bug upstream"

    # ------------------------------------------------------------------
    # Facet metadata (centroid, oriented outward normal, area, class)
    # ------------------------------------------------------------------
    keys = sorted(facet_key_to_col, key=facet_key_to_col.get)
    mesh_id = np.array([k[0] for k in keys], dtype=np.int32)
    face_id = np.array([k[1] for k in keys], dtype=np.int64)
    sign = np.asarray(facet_orient_sign)

    centroids = np.empty((n_facets, 3))
    normals = np.empty((n_facets, 3))
    areas = np.empty(n_facets)
    for mid, mesh in meshes.items():
        sel = mesh_id == mid
        if not sel.any():
            continue
        f = face_id[sel]
        centroids[sel] = mesh.triangles_center[f]
        normals[sel] = np.asarray(mesh.face_normals)[f] * sign[sel, None]
        areas[sel] = mesh.area_faces[f]

    classes = np.where(mesh_id == MESH_GROUND, CLASS_GROUND,
                       np.where(normals[:, 2] > args.roof_normal_z,
                                CLASS_ROOF, CLASS_WALL)).astype(np.int8)

    np.savez(out_dir / "facets.npz", mesh_id=mesh_id, face_id=face_id,
             centroid=centroids, normal=normals, area=areas, cls=classes)
    sp.save_npz(out_dir / "lw_view_matrix.npz", W)
    np.savez(out_dir / "lw_point_weights.npz", w_sky=w_sky, w_veg=w_veg,
             w_default=w_default)
    np.save(out_dir / "point_map.npy", point_map)
    np.save(out_dir / "coarse_index.npy", coarse_idx)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    n_bldg_faces = len(meshes[MESH_BUILDINGS].faces)
    n_grnd_faces = len(meshes[MESH_GROUND].faces)
    report = (
        f"Route-visible facet selection\n"
        f"  traced points        : {n_coarse:,} (of {n_full:,} route points)\n"
        f"  thermal facets kept  : {n_facets:,}\n"
        f"    building faces     : {(mesh_id == MESH_BUILDINGS).sum():,} "
        f"of {n_bldg_faces:,} in mesh "
        f"({100 * (mesh_id == MESH_BUILDINGS).sum() / max(1, n_bldg_faces):.1f}%)\n"
        f"    ground faces       : {(mesh_id == MESH_GROUND).sum():,} "
        f"of {n_grnd_faces:,} in mesh "
        f"({100 * (mesh_id == MESH_GROUND).sum() / max(1, n_grnd_faces):.1f}%)\n"
        f"    walls/roofs/ground : {(classes == CLASS_WALL).sum():,} / "
        f"{(classes == CLASS_ROOF).sum():,} / "
        f"{(classes == CLASS_GROUND).sum():,}\n"
        f"  mean weights         : sky {w_sky.mean():.3f}, veg {w_veg.mean():.3f}, "
        f"surfaces {np.asarray(W.sum(axis=1)).mean():.3f}, "
        f"default {w_default.mean():.3f}\n"
        f"  max |weight sum - 1| : {err:.2e}\n")
    (out_dir / "selection_report.txt").write_text(report)
    print("\n" + report)
    print(f"[facet_selection] n_facets={n_facets} output_dir={out_dir}")


if __name__ == "__main__":
    main()
