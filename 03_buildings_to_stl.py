"""
03_buildings_to_stl.py -- cluster building points into individual buildings,
extract each one's real 2D footprint (including concave shapes like L/U/H
floor plans -- NOT a convex hull, which would wrongly fill in courtyards
and notches), simplify it to a low vertex count, and extrude to a closed
LoD1-style solid (flat roof at the cluster's representative height).

Pipeline per building cluster:
    1. Rasterize the cluster's XY points to a binary occupancy grid
    2. Extract the outline contour (marching-squares style) -- this
       naturally follows concave shapes, unlike a convex hull
    3. Simplify the contour polygon (Douglas-Peucker) to a handful of
       vertices instead of one per grid cell
    4. Triangulate the simplified polygon (earcut) for the roof and floor
    5. Extrude walls between floor (local ground estimate) and roof height
    6. Combine floor + walls + roof into one closed, watertight solid

Run:
    python3 03_buildings_to_stl.py \
        --input split/building_points.npy \
        --output buildings.stl \
        --ground-npy split/ground_and_water_points.npy \
        --raster-res 0.5 --eps 3.0 --min-samples 30
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from grid_cluster import grid_cluster_2d
from skimage.measure import find_contours, approximate_polygon
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.spatial import cKDTree
import mapbox_earcut as earcut


def parse_args():
    p = argparse.ArgumentParser(description="Building points -> per-building LoD1 solid STL")
    p.add_argument("--input", required=True, help="Path to building_points.npy (Nx3 array)")
    p.add_argument("--output", required=True, help="Output STL path")
    p.add_argument("--ground-npy", required=True,
                    help="Path to ground_and_water_points.npy, used to estimate each "
                         "building's local base elevation")
    p.add_argument("--raster-res", type=float, default=0.5,
                    help="Footprint rasterization cell size, meters (default: 0.5)")
    p.add_argument("--cell-size", type=float, default=0.5,
                    help="Clustering grid cell size, meters -- points in the same or "
                         "connected grid cells become one building. Effective merge "
                         "distance is roughly cell_size * (1 + 2*connect_radius) "
                         "(default: 0.5)")
    p.add_argument("--connect-radius", type=int, default=1,
                    help="Grid cells of dilation before connecting components -- bridges "
                         "small gaps between sparse points (default: 1)")
    p.add_argument("--min-cluster-points", type=int, default=50,
                    help="Skip clusters with fewer than this many points (default: 50)")
    p.add_argument("--simplify-tolerance", type=float, default=0.75,
                    help="Douglas-Peucker polygon simplification tolerance, meters "
                         "(default: 0.75 -- larger = fewer footprint vertices)")
    p.add_argument("--roof-percentile", type=float, default=90.0,
                    help="Percentile of cluster Z used as roof height -- using a high "
                         "percentile rather than max() avoids a single noisy point spiking "
                         "the roof height (default: 90.0)")
    return p.parse_args()


def local_ground_elevation(building_xy_center, ground_pts, kdtree, k=50):
    """Estimate local ground elevation near a building by averaging the k
    nearest ground points -- more robust than a single nearest point."""
    dists, idx = kdtree.query(building_xy_center, k=min(k, len(ground_pts)))
    return float(np.mean(ground_pts[idx, 2]))


def extract_footprint_polygon(cluster_pts, raster_res, simplify_tolerance):
    """
    Rasterize a building cluster's XY footprint and extract its outline as a
    simplified polygon, preserving concave shapes (L/U/H floor plans).
    Returns an (M, 2) array of polygon vertices in world XY, or None if no
    valid contour could be extracted.
    """
    x, y = cluster_pts[:, 0], cluster_pts[:, 1]
    xmin, ymin = x.min(), y.min()

    nx = int(np.ceil((x.max() - xmin) / raster_res)) + 3
    ny = int(np.ceil((y.max() - ymin) / raster_res)) + 3

    grid = np.zeros((ny, nx), dtype=bool)
    col = np.clip(((x - xmin) / raster_res).astype(int) + 1, 0, nx - 1)
    row = np.clip(((y - ymin) / raster_res).astype(int) + 1, 0, ny - 1)
    grid[row, col] = True

    # Close small gaps from point sparsity, fill fully-enclosed interior holes
    grid = binary_closing(grid, structure=np.ones((3, 3)), iterations=2)
    grid = binary_fill_holes(grid)

    contours = find_contours(grid.astype(float), level=0.5)
    if not contours:
        return None

    # Use the longest contour (outer boundary); footprint area from grid
    # cell count is enough to skip more precise multi-contour handling here.
    contour = max(contours, key=len)
    simplified = approximate_polygon(contour, tolerance=simplify_tolerance / raster_res)

    # contour coords are (row, col) -> convert back to world (x, y)
    poly_x = simplified[:, 1] * raster_res + xmin - raster_res
    poly_y = simplified[:, 0] * raster_res + ymin - raster_res
    poly = np.column_stack([poly_x, poly_y])

    # Drop the duplicated closing vertex if present (first == last)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    if len(poly) < 3:
        return None
    return poly


def extrude_footprint(poly_xy, z_base, z_top):
    """
    Build a closed watertight solid from a (possibly concave, possibly
    non-convex) simple polygon: floor + walls + roof.
    """
    n = len(poly_xy)

    # Triangulate floor/roof polygon with earcut (handles concave polygons).
    verts2d = poly_xy.astype(np.float64)
    rings = np.array([n], dtype=np.uint32)
    tri_idx = earcut.triangulate_float64(verts2d, rings)
    tri_idx = np.asarray(tri_idx).reshape(-1, 3)

    if len(tri_idx) == 0:
        return None

    floor_verts = np.column_stack([poly_xy, np.full(n, z_base)])
    roof_verts = np.column_stack([poly_xy, np.full(n, z_top)])

    all_verts = np.vstack([floor_verts, roof_verts])
    faces = []

    # Floor (normal down -- reverse winding) and roof (normal up)
    for tri in tri_idx:
        faces.append([tri[0], tri[2], tri[1]])          # floor, flipped
        faces.append([tri[0] + n, tri[1] + n, tri[2] + n])  # roof

    # Walls: one quad (2 triangles) per polygon edge
    for i in range(n):
        j = (i + 1) % n
        f0, f1 = i, j
        r0, r1 = i + n, j + n
        faces.append([f0, f1, r1])
        faces.append([f0, r1, r0])

    mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(faces), process=True)
    return mesh


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.load(in_path)
    ground_pts = np.load(args.ground_npy)
    print(f"[bld] Loaded {len(pts):,} building points, {len(ground_pts):,} ground points")

    if len(pts) == 0:
        print("[bld] No building points -- writing empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[bld_result] n_buildings=0 total_faces=0 output={out_path}")
        return

    ground_kdtree = cKDTree(ground_pts[:, :2])

    xy = pts[:, :2]
    print(f"[bld] Clustering with grid-based connected components "
          f"(cell_size={args.cell_size}, connect_radius={args.connect_radius}) ...")
    labels, n_clusters = grid_cluster_2d(xy, cell_size=args.cell_size,
                                          connect_radius_cells=args.connect_radius)
    unique_labels = sorted(set(labels))
    print(f"[bld] Found {len(unique_labels)} candidate building clusters")

    building_meshes = []
    n_kept = 0
    n_skipped = 0

    for label in unique_labels:
        cluster_pts = pts[labels == label]
        if len(cluster_pts) < args.min_cluster_points:
            n_skipped += 1
            continue

        poly = extract_footprint_polygon(cluster_pts, args.raster_res, args.simplify_tolerance)
        if poly is None:
            print(f"[bld] WARNING: no valid footprint for a cluster of "
                  f"{len(cluster_pts)} points -- skipped")
            n_skipped += 1
            continue

        centroid_xy = cluster_pts[:, :2].mean(axis=0)
        z_base = local_ground_elevation(centroid_xy, ground_pts, ground_kdtree)
        z_top = float(np.percentile(cluster_pts[:, 2], args.roof_percentile))

        if z_top <= z_base:
            print(f"[bld] WARNING: roof height <= ground for a cluster -- skipped "
                  f"(z_base={z_base:.2f}, z_top={z_top:.2f})")
            n_skipped += 1
            continue

        mesh = extrude_footprint(poly, z_base, z_top)
        if mesh is None:
            print(f"[bld] WARNING: triangulation failed for a cluster -- skipped")
            n_skipped += 1
            continue

        building_meshes.append(mesh)
        n_kept += 1
        print(f"[bld]   building {n_kept}: {len(cluster_pts)} pts -> "
              f"{len(poly)} footprint vertices, {len(mesh.faces)} faces, "
              f"height {z_top - z_base:.1f} m")

    print(f"[bld] Buildings kept: {n_kept}, skipped: {n_skipped}")

    if not building_meshes:
        print("[bld] No valid buildings produced -- writing empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[bld_result] n_buildings=0 total_faces=0 output={out_path}")
        return

    combined = trimesh.util.concatenate(building_meshes)
    combined.export(str(out_path))

    total_faces = len(combined.faces)
    avg_faces = total_faces / n_kept
    print(f"[bld] Total faces: {total_faces} across {n_kept} buildings "
          f"({avg_faces:.1f} faces/building average)")
    print(f"[bld] Wrote: {out_path}")
    print(f"[bld_result] n_buildings={n_kept} n_skipped={n_skipped} "
          f"total_faces={total_faces} avg_faces_per_building={avg_faces:.1f} output={out_path}")


if __name__ == "__main__":
    main()
