#!/usr/bin/env python3
"""
main.py -- classified LAZ -> minimal-triangle-count ground / building /
vegetation STLs for shading & simple ground/urban-canopy CFD modeling.

Not aimed at watertight multi-region CFD meshing (no boolean subtraction,
objects are allowed to overlap the ground slightly) -- optimized purely
for: (a) correct occlusion for ray-traced shading, (b) minimum triangle
count for speed, (c) fast, robust processing that won't hang on real
LIDAR-scale data.

Pipeline:
    0. Split classified LAZ into ground+water / building / vegetation
       point sets (by ASPRS classification code).
    1. Vegetation: DBSCAN-cluster into individual trees, convex hull each
       (watertight, hugs real canopy shape, no trunk).
    2. Buildings: DBSCAN-cluster into individual buildings, extract each
       one's real footprint (concave-aware, NOT a convex hull -- L/U/H
       shaped buildings are common on campuses), extrude to a closed
       LoD1 solid.
    3. Ground+water: rasterize to a DEM mesh, then reduce via clustering
       decimation (handles huge/near-planar real terrain -- Blender's
       dissolve alone has proven pathological performance here) followed
       by a timeout-guarded Blender planar-decimate pass.

Run:
    python3 main.py --input classified.laz --output-dir out/
"""

import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DECIMATION_SCRIPT_DIR = SCRIPT_DIR  # expects 01_blender_planar_decimate.py and
                                     # 01a_meshlab_clustering_prepass.py to be
                                     # copied alongside this file -- see README
                                     # note at the bottom of this file.


def parse_args():
    p = argparse.ArgumentParser(description="Classified LAZ -> minimal-triangle shading STLs")
    p.add_argument("--input", required=True, help="Classified LAZ/LAS input file")
    p.add_argument("--output-dir", required=True, help="Output directory")

    p.add_argument("--veg-cell-size", type=float, default=0.5,
                    help="Vegetation clustering grid cell size, meters (default: 0.5)")
    p.add_argument("--veg-connect-radius", type=int, default=1,
                    help="Vegetation clustering dilation radius, grid cells (default: 1)")
    p.add_argument("--veg-min-hull-points", type=int, default=10)

    p.add_argument("--bld-cell-size", type=float, default=0.5,
                    help="Building clustering grid cell size, meters (default: 0.5)")
    p.add_argument("--bld-connect-radius", type=int, default=1,
                    help="Building clustering dilation radius, grid cells (default: 1)")
    p.add_argument("--bld-min-cluster-points", type=int, default=50)
    p.add_argument("--bld-raster-res", type=float, default=0.5,
                    help="Building footprint rasterization resolution, meters (default: 0.5)")
    p.add_argument("--bld-simplify-tolerance", type=float, default=0.75,
                    help="Building footprint polygon simplification tolerance, meters "
                         "(default: 0.75)")
    p.add_argument("--bld-roof-percentile", type=float, default=90.0)

    p.add_argument("--ground-raster-res", type=float, default=0.5,
                    help="Ground DEM rasterization resolution, meters (default: 0.5)")
    p.add_argument("--ground-cluster-cell-size", type=float, default=2.0,
                    help="Ground clustering pre-pass cell size, meters -- should be a few "
                         "times the raster resolution to actually reduce anything "
                         "(default: 2.0)")
    p.add_argument("--ground-angle-limit-deg", type=float, default=5.0,
                    help="Ground planar-decimate angle tolerance, degrees (default: 5.0)")
    p.add_argument("--ground-planar-timeout-sec", type=int, default=300,
                    help="Max seconds for the ground planar-decimate step before it's "
                         "skipped and the clustering-pre-pass output is used as-is "
                         "(default: 300)")

    p.add_argument("--blender-exe", default="blender")
    return p.parse_args()


class StageTimeout(RuntimeError):
    pass


class PipelineError(RuntimeError):
    pass


def run_logged(cmd, log_file, step_label, timeout_sec=None):
    """Run a subprocess with a hard timeout enforced via a background reader
    thread (works even if the subprocess produces zero output while stuck --
    see the earlier pipeline's postmortem on Blender's planar-decimate hang)."""
    print(f"\n---- {step_label} ----")
    lines = []
    with open(log_file, "a") as lf:
        lf.write(f"\n---- {step_label} ----\n$ {' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def _reader():
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
                lf.flush()
                lines.append(line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            t.join(timeout=5)
            msg = f"{step_label} TIMED OUT after {timeout_sec}s and was killed."
            print(f"\n[TIMEOUT] {msg}")
            raise StageTimeout(msg)
        t.join(timeout=5)
        if proc.returncode != 0:
            raise PipelineError(f"{step_label} failed (exit {proc.returncode}). See {log_file}.")
    return lines


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    split_dir = out_dir / "00_split"
    raw_dir = out_dir / "01_raw"
    final_dir = out_dir / "02_final"
    for d in (split_dir, raw_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "pipeline_log.txt"
    log_file.write_text("")

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        sys.exit(1)

    t0 = time.time()

    # ---- Stage 0: split classes ----
    run_logged(
        [sys.executable, str(SCRIPT_DIR / "01_split_classes.py"),
         "--input", str(in_path), "--output-dir", str(split_dir)],
        log_file, "Stage 0: split classified LAZ"
    )

    # ---- Stage 1: vegetation (convex hull per tree) ----
    veg_out = final_dir / "vegetation_final.stl"
    run_logged(
        [sys.executable, str(SCRIPT_DIR / "02_vegetation_to_stl.py"),
         "--input", str(split_dir / "vegetation_points.npy"),
         "--output", str(veg_out),
         "--ground-npy", str(split_dir / "ground_and_water_points.npy"),
         "--cell-size", str(args.veg_cell_size),
         "--connect-radius", str(args.veg_connect_radius),
         "--min-hull-points", str(args.veg_min_hull_points)],
        log_file, "Stage 1: vegetation -> per-tree convex hulls"
    )

    # ---- Stage 2: buildings (footprint extrusion) ----
    bld_out = final_dir / "building_final.stl"
    run_logged(
        [sys.executable, str(SCRIPT_DIR / "03_buildings_to_stl.py"),
         "--input", str(split_dir / "building_points.npy"),
         "--output", str(bld_out),
         "--ground-npy", str(split_dir / "ground_and_water_points.npy"),
         "--raster-res", str(args.bld_raster_res),
         "--cell-size", str(args.bld_cell_size),
         "--connect-radius", str(args.bld_connect_radius),
         "--min-cluster-points", str(args.bld_min_cluster_points),
         "--simplify-tolerance", str(args.bld_simplify_tolerance),
         "--roof-percentile", str(args.bld_roof_percentile)],
        log_file, "Stage 2: buildings -> footprint extrusion"
    )

    # ---- Stage 3: ground (raster -> clustering pre-pass -> timeout-guarded planar decimate) ----
    ground_raw = raw_dir / "ground_and_water_raster.stl"
    run_logged(
        [sys.executable, str(SCRIPT_DIR / "04_ground_to_stl.py"),
         "--input", str(split_dir / "ground_and_water_points.npy"),
         "--output", str(ground_raw),
         "--raster-res", str(args.ground_raster_res)],
        log_file, "Stage 3a: ground -> DEM raster mesh"
    )

    ground_clustered = raw_dir / "ground_and_water_clustered.stl"
    run_logged(
        [sys.executable, str(DECIMATION_SCRIPT_DIR / "01a_meshlab_clustering_prepass.py"),
         "--input", str(ground_raw),
         "--output", str(ground_clustered),
         "--cell-size", str(args.ground_cluster_cell_size),
         "--skip-below-faces", "10000"],
        log_file, "Stage 3b: ground -> clustering pre-pass"
    )

    ground_out = final_dir / "ground_and_water_final.stl"
    try:
        run_logged(
            [args.blender_exe, "--background", "--factory-startup", "--python",
             str(DECIMATION_SCRIPT_DIR / "01_blender_planar_decimate.py"), "--",
             "--input", str(ground_clustered),
             "--output", str(ground_out),
             "--angle-limit-deg", str(args.ground_angle_limit_deg)],
            log_file, "Stage 3c: ground -> planar decimate",
            timeout_sec=args.ground_planar_timeout_sec,
        )
    except StageTimeout:
        warn = ("Ground planar-decimate timed out -- using the clustering pre-pass "
                "output as the final ground mesh instead (still a large reduction from "
                "the raw DEM raster, just without the extra lossless flat-region merge).")
        print(warn)
        with open(log_file, "a") as lf:
            lf.write(warn + "\n")
        shutil.copyfile(str(ground_clustered), str(ground_out))

    t1 = time.time()

    print(f"\nDone in {t1 - t0:.1f}s. Final STLs:")
    print(f"  {veg_out}")
    print(f"  {bld_out}")
    print(f"  {ground_out}")


if __name__ == "__main__":
    # Force line-buffered stdout regardless of how this script is invoked --
    # when stdout is redirected to a file (e.g. `> log.txt`), Python defaults
    # to large block buffering, so print() output can sit invisible in a
    # buffer for a long time instead of showing up live in the log. This
    # makes `tail -f` actually show progress as it happens.
    import sys as _sys
    try:
        _sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # older Python without reconfigure(); -u flag is the fallback
    main()

# NOTE: this script calls 01_blender_planar_decimate.py and
# 01a_meshlab_clustering_prepass.py from the earlier decimation pipeline.
# Copy those two files into this same directory (or adjust
# DECIMATION_SCRIPT_DIR above to point at wherever you kept them).
