"""
04_ground_to_stl.py -- rasterize classified ground+water points into a
regular-grid DEM and triangulate it into a mesh. Deliberately dense/simple
at this stage (one quad->2 triangles per grid cell) -- the existing
01_blender_planar_decimate.py + 01a clustering pre-pass + 02 QECD pipeline
(already built and tested) is what actually collapses this down to a
minimal triangle count afterward, so this stage focuses purely on
correctly reconstructing the terrain surface from scattered points.

Gaps (grid cells with no LIDAR ground return -- e.g. directly under a
building) are filled by nearest-neighbor interpolation from surrounding
ground points, so the terrain continues smoothly under buildings/trees
rather than leaving holes.

Run:
    python3 04_ground_to_stl.py \
        --input split/ground_and_water_points.npy \
        --output ground_and_water.stl \
        --raster-res 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import griddata


def parse_args():
    p = argparse.ArgumentParser(description="Ground/water points -> DEM mesh STL")
    p.add_argument("--input", required=True, help="Path to ground_and_water_points.npy")
    p.add_argument("--output", required=True, help="Output STL path")
    p.add_argument("--raster-res", type=float, default=0.5,
                    help="DEM grid cell size, meters (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.load(in_path)
    print(f"[ground] Loaded {len(pts):,} ground/water points")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    nx = int(np.ceil((xmax - xmin) / args.raster_res)) + 1
    ny = int(np.ceil((ymax - ymin) / args.raster_res)) + 1
    print(f"[ground] Grid: {nx} x {ny} cells at {args.raster_res} m resolution")

    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # Nearest-neighbor fill for any grid cell without a nearby point (e.g.
    # under a building footprint) -- keeps the terrain continuous.
    GZ = griddata((x, y), z, (GX, GY), method="linear")
    nan_mask = np.isnan(GZ)
    if nan_mask.any():
        GZ_nn = griddata((x, y), z, (GX, GY), method="nearest")
        GZ[nan_mask] = GZ_nn[nan_mask]
        print(f"[ground] Filled {nan_mask.sum()} grid cells with nearest-neighbor "
              f"interpolation (no direct ground return, e.g. under buildings)")

    # Triangulate the grid (2 triangles per cell)
    verts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v00 = i * nx + j
            v10 = i * nx + (j + 1)
            v01 = (i + 1) * nx + j
            v11 = (i + 1) * nx + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    mesh.export(str(out_path))

    print(f"[ground] Faces: {len(mesh.faces)}")
    print(f"[ground] Wrote: {out_path}")
    print(f"[ground_result] n_points={len(pts)} grid={nx}x{ny} "
          f"total_faces={len(mesh.faces)} output={out_path}")


if __name__ == "__main__":
    main()
