"""
02_vegetation_to_stl.py -- cluster vegetation points into individual trees and
represent each as a robust bounding sphere. LiDAR vegetation returns are
noisy (stray points, mixed classification, multi-path returns), so instead
of a convex hull that wraps every point exactly -- including outliers -- we
fit a sphere centered on the cluster's median position with a radius sized
to contain a configurable fraction of the points (default 90%), and let the
rest fall outside. This trades exact crown-shape fidelity for a
noise-tolerant proxy that's cheap to compute and cheap to raytrace.

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
        --sphere-contain-fraction 0.9
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from grid_cluster import grid_cluster_2d
from tree_crown_segmentation import segment_tree_crowns, should_segment
from ground_height_filter import filter_above_ground


def fit_sphere(points, contain_fraction=0.9, min_radius=0.5):
    """Robust bounding sphere for one tree's points.

    Center is the per-axis median (resistant to stray outlier points), and
    the radius is set to the `contain_fraction` quantile of distances from
    that center, so the sphere contains most -- not necessarily all -- of
    the cluster. Points further out (noise, multi-path returns) are simply
    left outside instead of dragging the whole shape out to reach them.
    """
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    radius = max(float(np.quantile(dists, contain_fraction)), min_radius)
    return center, radius


def parse_args():
    p = argparse.ArgumentParser(description="Vegetation points -> per-tree bounding-sphere STL")
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
    p.add_argument("--sphere-contain-fraction", type=float, default=0.9,
                    help="Fraction of each tree's points the fitted bounding sphere must "
                         "contain (radius = this quantile of distances from the cluster's "
                         "median center). Lower values are more resistant to noisy/outlier "
                         "LiDAR returns but shrink the sphere further from the full point "
                         "spread (default: 0.9)")
    p.add_argument("--sphere-min-radius", type=float, default=0.5,
                    help="Floor on fitted sphere radius, meters -- avoids degenerate "
                         "slivers for very tight clusters (default: 0.5)")
    p.add_argument("--sphere-subdivisions", type=int, default=2,
                    help="Icosphere subdivision level for each tree mesh -- higher is "
                         "smoother but more faces (0=20 faces, 1=80, 2=320, 3=1280) "
                         "(default: 2)")

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
                center, radius = fit_sphere(
                    tree_pts,
                    contain_fraction=args.sphere_contain_fraction,
                    min_radius=args.sphere_min_radius,
                )
                sphere = trimesh.creation.icosphere(
                    subdivisions=args.sphere_subdivisions, radius=radius
                )
                sphere.apply_translation(center)
            except Exception as e:
                print(f"[veg] WARNING: sphere fit failed for a cluster of {len(tree_pts)} points: {e}")
                n_trees_skipped += 1
                continue
            tree_meshes.append(sphere)
            n_trees_kept += 1

    print(f"[veg] Macro-clusters segmented into multiple trees: {n_macro_segmented}")
    print(f"[veg] Macro-clusters treated as a single tree: {n_macro_passthrough}")
    print(f"[veg] Trees kept: {n_trees_kept}, skipped (too few points): {n_trees_skipped}")

    if not tree_meshes:
        print("[veg] No valid tree spheres produced -- writing empty placeholder STL.")
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
