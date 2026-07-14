"""
tree_crown_segmentation.py -- split a cluster of vegetation points that may
contain MULTIPLE touching/overlapping tree canopies into individual trees.

This is needed because simple point-connectivity clustering (grid_cluster.py)
cannot distinguish "one big tree" from "five trees whose canopies touch" --
if the points are spatially connected, they become one cluster, which for a
convex hull means filling in every gap between separate trees as if it were
solid canopy. On a real campus with trees planted in rows or clusters, this
merges large numbers of distinct trees into single blobby volumes.

METHOD (standard in LIDAR forestry / remote sensing -- "individual tree
crown delineation via canopy height model watershed segmentation"):

  1. Rasterize the cluster's points into a Canopy Height Model (CHM): the
     MAXIMUM point height in each XY grid cell (i.e. the canopy "skin").
  2. Smooth the CHM slightly to suppress spurious tiny local peaks from
     individual noisy returns.
  3. Find local maxima in the smoothed CHM ("tree tops") with a minimum
     separation distance -- even when two canopies touch and merge at their
     base/edges, their APEXES are almost always still distinct height
     peaks, which is what makes this approach work where simple point
     connectivity doesn't.
  4. Run marker-controlled watershed on the (inverted) CHM using the tree
     tops as seeds -- this "floods" outward from each treetop and splits
     the merged canopy at the natural saddle points (valleys) BETWEEN
     treetops, which is exactly where one tree's crown ends and another's
     begins.
  5. Assign every input point to whichever watershed region its XY falls
     into.

This only needs to run on clusters that are plausibly more than one tree
(see should_segment() below) -- small clusters are already almost
certainly a single tree and segmenting them risks fragmenting one real
canopy into pieces.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def rasterize_chm(pts, cell_size):
    """
    Build a Canopy Height Model: max Z per XY grid cell. Returns the CHM
    array, plus (xmin, ymin, cell_size) needed to map grid <-> world coords,
    and a per-point (row, col) index array.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xmin, ymin = x.min(), y.min()

    col = ((x - xmin) / cell_size).astype(np.int64)
    row = ((y - ymin) / cell_size).astype(np.int64)
    nx, ny = int(col.max()) + 1, int(row.max()) + 1

    # IMPORTANT: must initialize with -inf, not NaN, for np.maximum.at to
    # work -- np.maximum(nan, x) propagates NaN forever (IEEE754 semantics),
    # so a NaN-initialized array never actually gets updated.
    chm = np.full((ny, nx), -np.inf, dtype=np.float64)
    np.maximum.at(chm, (row, col), z)

    # Now convert genuinely-empty cells (never touched, still -inf) to NaN.
    chm[np.isneginf(chm)] = np.nan

    return chm, xmin, ymin, row, col


def should_segment(pts, cell_size, min_points_for_segmentation, min_extent_for_segmentation):
    """
    Heuristic gate: only attempt crown segmentation on clusters that are
    plausibly more than one tree, to avoid wastefully (and riskily)
    fragmenting genuinely single small trees.
    """
    if len(pts) < min_points_for_segmentation:
        return False
    extent = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    return max(extent) > min_extent_for_segmentation


def merge_small_fragments(pts, labels, min_fragment_points):
    """
    Safety net: any crown with fewer than min_fragment_points is almost
    certainly a spurious fragment (a small bump on a neighboring canopy's
    surface, not a real distinct tree) rather than a genuine small tree.
    Merge it into whichever OTHER crown has the nearest centroid, rather
    than leaving it as an isolated sliver hull.

    This runs regardless of how well-tuned min_tree_distance/smooth_sigma
    are -- it's a structural backstop against the "field of pebbles"
    failure mode, not a substitute for reasonable parameters.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return labels

    counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
    small = [lbl for lbl, c in counts.items() if c < min_fragment_points]
    large = [lbl for lbl in unique_labels if lbl not in small]

    if not small or not large:
        return labels  # nothing to merge, or everything is small (leave as-is)

    centroids = {lbl: pts[labels == lbl, :2].mean(axis=0) for lbl in unique_labels}
    large_centroids = np.array([centroids[lbl] for lbl in large])

    new_labels = labels.copy()
    for lbl in small:
        d = np.linalg.norm(large_centroids - centroids[lbl], axis=1)
        nearest_large = large[int(np.argmin(d))]
        new_labels[labels == lbl] = nearest_large

    return new_labels


def segment_tree_crowns(pts, chm_res=0.25, smooth_sigma=2.0, min_tree_distance_m=3.0,
                         min_height_above_local_ground=1.0, min_fragment_points=None):
    """
    Split a (possibly multi-tree) point cluster into individual tree crowns.

    Parameters
    ----------
    pts : (N, 3) array
        Vegetation points for this cluster.
    chm_res : float
        Canopy height model grid resolution, meters.
    smooth_sigma : float
        Gaussian smoothing sigma (in grid cells) applied to the CHM before
        peak detection, to suppress noise-driven spurious treetops.
    min_tree_distance_m : float
        Minimum allowed distance between detected treetops, meters --
        should be roughly your minimum realistic trunk spacing.
    min_height_above_local_ground : float
        CHM cells lower than (max_height - this) by a large margin are
        still included in watershed (this only affects peak filtering,
        not segmentation extent).
    min_fragment_points : int or None
        If given, any resulting crown with fewer than this many points is
        merged into its nearest (by centroid) larger neighbor rather than
        kept as a separate tiny sliver hull -- a structural backstop
        against over-segmentation regardless of parameter tuning.

    Returns
    -------
    tree_labels : (N,) int array
        Per-point tree id (0-indexed), one id per detected crown.
    n_trees : int
        Number of distinct crowns found.
    """
    chm, xmin, ymin, row, col = rasterize_chm(pts, chm_res)

    # Fill NaN gaps (empty cells) with the local min for smoothing/peak
    # purposes only -- doesn't affect which points get which label later.
    valid = ~np.isnan(chm)
    if valid.sum() == 0:
        return np.zeros(len(pts), dtype=int), 1

    fill_value = np.nanmin(chm)
    chm_filled = np.where(valid, chm, fill_value)

    chm_smooth = ndimage.gaussian_filter(chm_filled, sigma=smooth_sigma)

    min_distance_cells = max(1, int(round(min_tree_distance_m / chm_res)))
    coords = peak_local_max(
        chm_smooth,
        min_distance=min_distance_cells,
        exclude_border=False,
    )

    if len(coords) == 0:
        # No clear peaks found -- treat the whole cluster as one tree.
        return np.zeros(len(pts), dtype=int), 1

    markers = np.zeros(chm_smooth.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1

    # Watershed on the INVERTED CHM: floods "downhill" from each treetop
    # marker, naturally splitting at saddle points between adjacent peaks.
    labels_grid = watershed(-chm_smooth, markers=markers, mask=valid)

    # Any point whose cell didn't get a watershed label (rare edge case,
    # e.g. isolated valid cell with no path to a marker) falls back to
    # nearest-marker assignment via a distance transform.
    unlabeled = valid & (labels_grid == 0)
    if unlabeled.any():
        nearest_label_idx = ndimage.distance_transform_edt(
            markers == 0, return_distances=False, return_indices=True
        )
        labels_grid[unlabeled] = markers[tuple(idx[unlabeled] for idx in nearest_label_idx)]

    point_labels = labels_grid[row, col] - 1  # 0-indexed
    point_labels = np.clip(point_labels, 0, None)  # safety: no -1s leak through

    if min_fragment_points is not None:
        point_labels = merge_small_fragments(pts, point_labels, min_fragment_points)

    n_trees = len(np.unique(point_labels))
    return point_labels, n_trees
