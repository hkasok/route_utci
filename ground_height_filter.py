"""
ground_height_filter.py -- filter vegetation points to only those
meaningfully above the local ground surface, using a fast rasterized
lookup rather than per-point nearest-neighbor queries (which would be too
slow at multi-million-point scale).

WHY THIS IS NEEDED: our vegetation classification (01_split_classes.py)
merges low/medium/high vegetation (ASPRS classes 3/4/5) into one bucket,
because at the classification-splitting stage there's no reliable way to
know which points are "grass under a tree" vs. "the tree itself" -- both
are just "vegetation". But XY-based clustering (grid_cluster.py) then pulls
ground-hugging grass/shrub points sharing a tree's footprint into the SAME
cluster as the tree's canopy, and the convex hull spans the full vertical
range -- producing a solid pillar from ground to canopy top instead of a
floating crown. Confirmed directly: a synthetic floating canopy (z=2-8m)
plus separate grass points (z=0-0.3m) sharing its XY footprint produced a
combined hull spanning z=0-8m.

METHOD: rasterize the ground+water point cloud into a coarse height grid
(same technique as 04_ground_to_stl.py), then look up each vegetation
point's local ground elevation via O(1) grid indexing (fast enough for
10M+ points) rather than a per-point spatial query. Points within
`min_height_above_ground` of their local ground cell are dropped before
clustering.
"""

import numpy as np
from scipy.interpolate import griddata


def build_ground_height_grid(ground_pts, cell_size=2.0):
    """
    Rasterize ground+water points into a coarse height grid. Coarser than
    the full ground DEM (default 2m vs 0.5m) is fine and faster here --
    this is only used to estimate "how far above ground is this vegetation
    point", not to reconstruct terrain detail.
    """
    x, y, z = ground_pts[:, 0], ground_pts[:, 1], ground_pts[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    nx = int(np.ceil((xmax - xmin) / cell_size)) + 1
    ny = int(np.ceil((ymax - ymin) / cell_size)) + 1

    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(grid_x, grid_y)

    GZ = griddata((x, y), z, (GX, GY), method="linear")
    nan_mask = np.isnan(GZ)
    if nan_mask.any():
        GZ_nn = griddata((x, y), z, (GX, GY), method="nearest")
        GZ[nan_mask] = GZ_nn[nan_mask]

    return GZ, xmin, ymin, cell_size, nx, ny


def height_above_ground(pts, ground_height_grid):
    """
    Look up each point's height above its local ground cell via fast O(1)
    grid indexing (no per-point spatial search -- safe at 10M+ point scale).
    """
    GZ, xmin, ymin, cell_size, nx, ny = ground_height_grid
    col = np.clip(((pts[:, 0] - xmin) / cell_size).astype(np.int64), 0, nx - 1)
    row = np.clip(((pts[:, 1] - ymin) / cell_size).astype(np.int64), 0, ny - 1)
    local_ground_z = GZ[row, col]
    return pts[:, 2] - local_ground_z


def filter_above_ground(pts, ground_pts, min_height_above_ground=1.5, ground_cell_size=2.0):
    """
    Drop points within `min_height_above_ground` of the local ground
    surface. Returns (filtered_pts, n_dropped).
    """
    grid = build_ground_height_grid(ground_pts, cell_size=ground_cell_size)
    hag = height_above_ground(pts, grid)
    keep_mask = hag >= min_height_above_ground
    return pts[keep_mask], int((~keep_mask).sum())
