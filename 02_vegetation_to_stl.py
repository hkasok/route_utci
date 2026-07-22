"""
02_vegetation_to_stl.py -- cluster vegetation points into individual trees and
represent each crown as a flat-side-down hemisphere (a dome). LiDAR
vegetation returns are noisy (stray points, mixed classification, multi-path
returns), so instead of a convex hull that wraps every point exactly -- or a
full sphere whose lower half wastefully fills the trunk/pedestrian zone -- we
fit a robust dome to each crown's CORE mass:

  * flat circular base sits at a low percentile of the crown's height (so it
    starts above pedestrian head height and never clips below ground), and
  * a half-sphere bulges upward, with radius set from a high percentile of
    the horizontal point spread (ignoring the outermost stray branch tips)
    and capped at a realistic maximum crown radius.

Outlier points (single leaves, thin protruding branches, low hanging shoots)
are deliberately left outside the dome rather than inflating it. Neighboring
domes are allowed to overlap freely. This trades exact crown-shape fidelity
for a noise-tolerant, physically-plausible proxy that's cheap to compute and
cheap to raytrace. See fit_hemisphere() for the full fitting rules.

Pipeline:
  0. Height-above-ground filter (ground_height_filter.py): drops points too
     close to the local ground surface BEFORE clustering. Necessary because
     our vegetation classification merges low/medium/high vegetation into
     one bucket, so ground-hugging grass/shrub points sharing a tree's XY
     footprint would otherwise get pulled into the same cluster as the
     canopy above -- producing a convex hull that's a solid pillar from
     ground to canopy top instead of a floating crown. Confirmed and fixed
     directly: a synthetic floating canopy (z=2-8m) plus grass at its base
     (z=0-0.3m) produced a combined hull spanning z=0-8m before this
     filter, and correctly floats 2-8m after it.
  1. Grid-based connected components (grid_cluster.py) -- fast, memory-safe
     macro clustering.
  2. Canopy-height-model watershed segmentation (tree_crown_segmentation.py)
     -- splits macro-clusters large enough to plausibly be multiple trees
     into individual crowns.

Run:
    python3 02_vegetation_to_stl.py \
        --input split/vegetation_points.npy \
        --output vegetation.stl \
        --ground-npy split/ground_and_water_points.npy \
        --min-height-above-ground 1.5 \
        --cell-size 0.5 --connect-radius 1 --min-hull-points 10 \
        --crown-horiz-quantile 0.9 --max-crown-radius 10.0
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from grid_cluster import grid_cluster_2d
from tree_crown_segmentation import segment_tree_crowns, should_segment
from ground_height_filter import filter_above_ground


def fit_hemisphere(points, horiz_quantile=0.9, base_z_quantile=0.15,
                   min_radius=1.0, max_radius=10.0):
    """Fit a flat-side-down hemisphere (dome) to one tree crown's points.

    A crown is modeled as a dome: a horizontal circular flat base with a
    half-sphere bulging upward from it. The fit is deliberately robust to
    stray LiDAR returns (single leaves, thin protruding branches, low
    hanging shoots) -- it represents the CORE crown mass rather than the
    convex extent of every point:

      * Center (cx, cy): per-axis median of the horizontal positions --
        resistant to a few outlying points pulling the crown sideways.
      * Radius R: the `horiz_quantile` (default 90th percentile) of each
        point's horizontal distance from the center. The outermost ~10% of
        points -- stray branch tips -- fall OUTSIDE the dome instead of
        inflating it. R is clamped to [min_radius, max_radius], where
        max_radius is a realistic upper bound on crown radius so a noisy or
        merged cluster can't produce an absurdly large dome.
      * Base elevation (base_z): the `base_z_quantile` (default 15th
        percentile) of the points' z. Using a low percentile rather than
        the minimum ignores a few low hanging points/branches, and because
        near-ground vegetation was already dropped by the height-above-
        ground filter upstream, this base sits well above the pedestrian
        zone -- the dome rarely occupies the lowest ~2 m and never clips
        below ground.

    The dome's flat side is at z = base_z and it rises to base_z + R, so its
    height equals its horizontal radius. Returns (cx, cy, base_z, R).
    """
    cx, cy = np.median(points[:, 0]), np.median(points[:, 1])
    horiz_dist = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    radius = float(np.quantile(horiz_dist, horiz_quantile))
    radius = min(max(radius, min_radius), max_radius)
    base_z = float(np.quantile(points[:, 2], base_z_quantile))
    return float(cx), float(cy), base_z, radius


def build_hemisphere_mesh(cx, cy, base_z, radius, subdivisions=2):
    """Watertight flat-side-down hemisphere mesh at (cx, cy, base_z), dome up.

    Built directly as a UV-parametrized dome (top pole -> equator) plus a flat
    circular base cap, so there's no dependency on shapely/slice_plane. The
    flat base lies at z = base_z and the dome rises to z = base_z + radius.
    `subdivisions` controls tessellation density (n_lon = 4 * 2**subdivisions).
    """
    n_lon = max(4, 4 * (2 ** int(subdivisions)))     # sectors around the axis
    n_lat = max(2, n_lon // 2)                        # rings pole -> equator

    verts = [[0.0, 0.0, radius]]                      # 0: top pole
    for i in range(1, n_lat + 1):                     # rings; i=n_lat is equator
        lat = (np.pi / 2.0) * (i / n_lat)
        z = radius * np.cos(lat)
        rr = radius * np.sin(lat)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([rr * np.cos(phi), rr * np.sin(phi), z])
    base_center = len(verts)
    verts.append([0.0, 0.0, 0.0])                     # base center (flat side)

    def ring(i, j):                                   # vertex index in ring i (1..n_lat)
        return 1 + (i - 1) * n_lon + (j % n_lon)

    faces = []
    for j in range(n_lon):                            # top cap fan (pole -> ring 1)
        faces.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, n_lat):                         # dome bands
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    for j in range(n_lon):                            # flat base cap (downward)
        faces.append([base_center, ring(n_lat, j + 1), ring(n_lat, j)])

    dome = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=True)
    dome.apply_translation([cx, cy, base_z])
    return dome


def parse_args():
    p = argparse.ArgumentParser(description="Vegetation points -> per-crown hemisphere (dome) STL")
    p.add_argument("--input", required=True, help="Path to vegetation_points.npy (Nx3 array)")
    p.add_argument("--output", required=True, help="Output STL path")

    p.add_argument("--ground-npy", default=None,
                    help="Path to ground_and_water_points.npy. If given, points too close "
                         "to the local ground surface are dropped before clustering (fixes "
                         "canopy-to-ground pillar artifacts from merged low/high vegetation "
                         "classes). Strongly recommended -- omit only if you don't have a "
                         "ground reference available.")
    p.add_argument("--min-height-above-ground", type=float, default=1.5,
                    help="Drop vegetation points within this height (meters) of the local "
                         "ground surface (default: 1.5)")
    p.add_argument("--ground-cell-size", type=float, default=2.0,
                    help="Ground height lookup grid resolution, meters -- coarser than the "
                         "full ground DEM is fine here (default: 2.0)")

    p.add_argument("--cell-size", type=float, default=0.5,
                    help="Macro clustering grid cell size, meters (default: 0.5)")
    p.add_argument("--connect-radius", type=int, default=1,
                    help="Macro clustering dilation radius, grid cells (default: 1)")
    p.add_argument("--min-hull-points", type=int, default=10,
                    help="Skip final tree clusters with fewer than this many points "
                         "(default: 10)")
    p.add_argument("--crown-horiz-quantile", type=float, default=0.9,
                    help="Crown dome radius = this quantile of points' horizontal distance "
                         "from the crown center. The outermost (1 - q) fraction of points -- "
                         "stray branch tips/leaves -- fall OUTSIDE the dome instead of "
                         "inflating it. Lower = tighter, more outlier-resistant (default: 0.9)")
    p.add_argument("--crown-base-z-quantile", type=float, default=0.15,
                    help="Crown dome flat base sits at this quantile of the crown points' "
                         "height. A low (but non-zero) percentile ignores a few low hanging "
                         "points while keeping the base above the pedestrian zone and above "
                         "ground (default: 0.15)")
    p.add_argument("--max-crown-radius", type=float, default=10.0,
                    help="Realistic upper bound on crown dome radius, meters (~20 m diameter). "
                         "Caps domes from noisy or under-segmented clusters. Also bounds dome "
                         "HEIGHT, since a hemisphere's height equals its radius (default: 10.0)")
    p.add_argument("--min-crown-radius", type=float, default=1.0,
                    help="Floor on crown dome radius, meters -- avoids degenerate slivers for "
                         "very tight clusters (default: 1.0)")
    p.add_argument("--crown-subdivisions", type=int, default=2,
                    help="Icosphere subdivision level for each crown dome -- higher is "
                         "smoother but more faces (default: 2)")

    p.add_argument("--crown-chm-res", type=float, default=0.25,
                    help="Canopy height model resolution for crown segmentation, meters "
                         "(default: 0.25)")
    p.add_argument("--crown-smooth-sigma", type=float, default=2.0,
                    help="CHM smoothing sigma, grid cells -- higher suppresses more of the "
                         "natural bumpiness within a single canopy/hedge row that would "
                         "otherwise be mistaken for separate treetops (default: 2.0)")
    p.add_argument("--crown-min-tree-distance", type=float, default=3.0,
                    help="Minimum distance between detected treetops, meters -- should "
                         "roughly match your site's minimum realistic trunk spacing (most "
                         "campus/urban trees: 4m+). Too low causes 'field of pebbles' "
                         "over-segmentation of natural canopy texture into many tiny "
                         "spurious fragments; too high risks merging genuinely distinct "
                         "nearby trees. (default: 3.0)")
    p.add_argument("--crown-min-fragment-points", type=int, default=50,
                    help="Structural backstop: any detected crown with fewer than this many "
                         "points gets merged into its nearest larger neighbor rather than "
                         "kept as an isolated sliver hull. Set to 0 to disable (default: 50)")
    p.add_argument("--crown-min-points", type=int, default=300,
                    help="Only attempt crown segmentation on macro-clusters with at least "
                         "this many points (default: 300)")
    p.add_argument("--crown-min-extent", type=float, default=8.0,
                    help="Only attempt crown segmentation on macro-clusters whose XY extent "
                         "exceeds this, meters -- a real single tree crown rarely exceeds "
                         "this diameter, so bigger almost certainly means multiple trees "
                         "(default: 8.0)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.load(in_path)
    print(f"[veg] Loaded {len(pts):,} vegetation points")

    if len(pts) == 0:
        print("[veg] No vegetation points -- writing an empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
        return

    if args.ground_npy is not None:
        ground_pts = np.load(args.ground_npy)
        print(f"[veg] Applying height-above-ground filter "
              f"(min_height={args.min_height_above_ground} m) ...")
        pts, n_dropped = filter_above_ground(
            pts, ground_pts,
            min_height_above_ground=args.min_height_above_ground,
            ground_cell_size=args.ground_cell_size,
        )
        print(f"[veg] Dropped {n_dropped:,} near-ground points, {len(pts):,} remain")
        if len(pts) == 0:
            print("[veg] No points remain after ground filtering -- writing empty STL.")
            trimesh.Trimesh().export(str(out_path))
            print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
            return
    else:
        print("[veg] WARNING: no --ground-npy given -- skipping height-above-ground "
              "filtering. Canopy-to-ground pillar artifacts are possible if low "
              "vegetation points share a tree's footprint.")

    xy = pts[:, :2]
    print(f"[veg] Macro-clustering with grid-based connected components "
          f"(cell_size={args.cell_size}, connect_radius={args.connect_radius}) ...")
    macro_labels, n_macro = grid_cluster_2d(xy, cell_size=args.cell_size,
                                             connect_radius_cells=args.connect_radius)
    unique_macro = np.unique(macro_labels)
    print(f"[veg] Found {len(unique_macro)} macro-clusters")

    tree_meshes = []
    n_trees_kept = 0
    n_trees_skipped = 0
    n_macro_segmented = 0
    n_macro_passthrough = 0

    for macro_id in unique_macro:
        cluster_pts = pts[macro_labels == macro_id]

        if should_segment(cluster_pts, args.cell_size,
                           args.crown_min_points, args.crown_min_extent):
            crown_labels, n_crowns = segment_tree_crowns(
                cluster_pts,
                chm_res=args.crown_chm_res,
                smooth_sigma=args.crown_smooth_sigma,
                min_tree_distance_m=args.crown_min_tree_distance,
                min_fragment_points=args.crown_min_fragment_points if args.crown_min_fragment_points > 0 else None,
            )
            n_macro_segmented += 1
            sub_clusters = [cluster_pts[crown_labels == c] for c in np.unique(crown_labels)]
        else:
            n_macro_passthrough += 1
            sub_clusters = [cluster_pts]

        for tree_pts in sub_clusters:
            if len(tree_pts) < args.min_hull_points:
                n_trees_skipped += 1
                continue
            try:
                cx, cy, base_z, radius = fit_hemisphere(
                    tree_pts,
                    horiz_quantile=args.crown_horiz_quantile,
                    base_z_quantile=args.crown_base_z_quantile,
                    min_radius=args.min_crown_radius,
                    max_radius=args.max_crown_radius,
                )
                dome = build_hemisphere_mesh(
                    cx, cy, base_z, radius,
                    subdivisions=args.crown_subdivisions,
                )
            except Exception as e:
                print(f"[veg] WARNING: hemisphere fit failed for a cluster of {len(tree_pts)} points: {e}")
                n_trees_skipped += 1
                continue
            if dome is None or len(dome.faces) == 0:
                n_trees_skipped += 1
                continue
            tree_meshes.append(dome)
            n_trees_kept += 1

    print(f"[veg] Macro-clusters segmented into multiple trees: {n_macro_segmented}")
    print(f"[veg] Macro-clusters treated as a single tree: {n_macro_passthrough}")
    print(f"[veg] Trees kept: {n_trees_kept}, skipped (too few points): {n_trees_skipped}")

    if not tree_meshes:
        print("[veg] No valid crown domes produced -- writing empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
        return

    combined = trimesh.util.concatenate(tree_meshes)
    combined.export(str(out_path))

    total_faces = len(combined.faces)
    avg_faces = total_faces / n_trees_kept
    print(f"[veg] Total faces: {total_faces} across {n_trees_kept} trees "
          f"({avg_faces:.1f} faces/tree average)")
    print(f"[veg] Wrote: {out_path}")
    print(f"[veg_result] n_trees={n_trees_kept} n_skipped={n_trees_skipped} "
          f"n_macro_segmented={n_macro_segmented} n_macro_passthrough={n_macro_passthrough} "
          f"total_faces={total_faces} avg_faces_per_tree={avg_faces:.1f} output={out_path}")


if __name__ == "__main__":
    main()
