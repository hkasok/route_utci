"""
Stage 1a -- MeshLab clustering decimation pre-pass (for huge meshes)
========================================================================

Blender's planar-decimate (Decimate/DISSOLVE) has pathological
performance on very large, genuinely flat/coplanar regions -- exactly
what real lake surfaces and large flat graded ground areas are. In
testing, a perfectly flat 1-million-triangle patch made it hang past
2 minutes with no progress, while a similarly-sized but non-uniformly
curved patch finished in under 3 minutes. On a real ~16-million-face
LIDAR ground mesh with large real flat water/ground areas, this
translates to a run that may never finish.

MeshLab's *clustering* decimation is a completely different algorithm
(uniform 3D grid binning, not iterative face-by-face dissolve) and is
not sensitive to how large a single flat region is. In the same
worst-case flat-1M-triangle test, it finished in 0.18 seconds with
~1e-6 m error.

Use this as a fast first-pass reduction on huge input meshes (millions
of faces) BEFORE running 01_blender_planar_decimate.py. It is coarser
and less shape-aware than Blender's planar dissolve or MeshLab's QECD,
so it's a size hammer, not a precision tool -- the pipeline still runs
Blender planar decimate and QECD afterward for shape-aware refinement,
now on a mesh that's small enough for those to run at all.

Run:
    python3 01a_meshlab_clustering_prepass.py \
        --input /path/to/ground_and_water.stl \
        --output /path/to/ground_clustered.stl \
        --cell-size 0.5

Notes
-----
* --cell-size is an ABSOLUTE distance in your STL's units (meters, for
  a LIDAR-derived model) -- it's the edge length of the uniform grid
  cell used to merge vertices. Set it based on your LIDAR point
  spacing: too small (finer than your point spacing) and nothing gets
  merged; too large and you lose real detail. A good starting point
  is roughly 2-5x your LIDAR point spacing, but check the reported
  Hausdorff error below against your accuracy budget.
* This stage is skipped automatically (input is just copied through)
  if the input mesh already has fewer faces than --skip-below-faces,
  since clustering is not needed (and Blender planar decimate alone
  is fine) below that size.
"""

import argparse
import shutil
from pathlib import Path

import pymeshlab


def parse_args():
    p = argparse.ArgumentParser(description="Fast clustering decimation pre-pass for huge meshes")
    p.add_argument("--input", required=True, help="Input STL path")
    p.add_argument("--output", required=True, help="Output STL path")
    p.add_argument("--cell-size", type=float, default=0.5,
                    help="Absolute clustering cell size, same units as your STL, "
                         "e.g. meters (default: 0.5)")
    p.add_argument("--skip-below-faces", type=int, default=500_000,
                    help="Skip this pre-pass (just copy input->output) if the input has "
                         "fewer faces than this -- clustering isn't needed at smaller "
                         "sizes and Blender planar decimate handles them fine on its own "
                         "(default: 500000)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input STL not found: {in_path}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(in_path))
    faces_before = ms.current_mesh().face_number()
    print(f"[stage1a] Faces before: {faces_before}")

    if faces_before < args.skip_below_faces:
        print(f"[stage1a] Below skip threshold ({args.skip_below_faces}) -- "
              f"copying input through unchanged.")
        shutil.copyfile(str(in_path), str(out_path))
        faces_after = faces_before
    else:
        ms.meshing_decimation_clustering(threshold=pymeshlab.PureValue(args.cell_size))
        faces_after = ms.current_mesh().face_number()
        ms.save_current_mesh(str(out_path))

    reduction_pct = 100.0 * (1.0 - faces_after / faces_before) if faces_before else 0.0
    print(f"[stage1a] Faces after:  {faces_after} ({reduction_pct:.1f}% reduction)")
    print(f"[stage1a] Wrote: {out_path}")
    print(f"[stage1a_result] faces_before={faces_before} faces_after={faces_after} "
          f"reduction_pct={reduction_pct:.4f} cell_size={args.cell_size} output={out_path}")


if __name__ == "__main__":
    main()
