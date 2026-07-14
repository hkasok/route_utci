"""
convert_laz_to_local_meters.py
================================

Converts a LAZ/LAS tile from Florida State Plane East (NAD83, US Survey
Feet -- EPSG:2236, the standard CRS for USGS LPC Florida tiles) into a
LOCAL, meters-based coordinate system suitable for CFD/OpenFOAM.

This does two things, both necessary:

  1. Unit conversion: feet -> meters, using the exact US Survey Foot
     factor (1200/3937 m per ft = 0.3048006096... m), NOT the
     international foot (0.3048 m exactly) -- Florida State Plane uses
     US Survey Feet, and the two differ by about 2 ppm, which sounds
     tiny but adds up to real error over a 1km domain.

  2. Re-centering to a local origin: raw State Plane coordinates sit
     around X=860,000 / Y=515,000. Feeding coordinates that large
     directly into mesh generation causes real floating-point
     precision problems (tiny cell sizes at large absolute coordinates
     lose precision). This shifts everything so your area of interest
     is near (0, 0, 0) instead.

By default the origin is set to the min corner of the point cloud's
bounding box (in the new meters system) -- i.e. the output cloud's
bounding box will start at (0, 0, <local min Z>). Use --origin-x/--origin-y
if you'd rather set a specific reference point (e.g. a known campus
corner) as the new (0, 0).

Run:
    python3 convert_laz_to_local_meters.py \
        --input USGS_LPC_FL_MiamiDade_D23_LID2024_319039_0901.laz \
        --output tile_local_meters.laz

    # Or with an explicit origin (in the ORIGINAL feet CRS, EPSG:2236):
    python3 convert_laz_to_local_meters.py \
        --input tile.laz --output tile_local_meters.laz \
        --origin-x 860000 --origin-y 515000

The output file keeps the SAME classification codes (ground/building/
vegetation/etc.) unchanged -- only X/Y/Z and their units are touched.
"""

import argparse
from pathlib import Path

import laspy
import numpy as np

# US Survey Foot -> meter, exact rational factor (NOT the international
# foot, 0.3048 m exactly -- Florida State Plane / EPSG:2236 uses US
# Survey Feet).
US_SURVEY_FOOT_TO_METER = 1200.0 / 3937.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a Florida State Plane East (ftUS) LAZ/LAS tile to local meters"
    )
    p.add_argument("--input", required=True, help="Input LAZ/LAS path (feet, e.g. EPSG:2236)")
    p.add_argument("--output", required=True, help="Output LAZ/LAS path (local meters)")
    p.add_argument("--origin-x", type=float, default=None,
                    help="Origin X in the ORIGINAL feet CRS (default: min X of the input cloud)")
    p.add_argument("--origin-y", type=float, default=None,
                    help="Origin Y in the ORIGINAL feet CRS (default: min Y of the input cloud)")
    p.add_argument("--origin-z", type=float, default=None,
                    help="Origin Z in the ORIGINAL feet CRS (default: min Z of the input cloud, "
                         "i.e. lowest ground point becomes Z=0)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input LAZ/LAS not found: {in_path}")

    print(f"[convert] Reading: {in_path}")
    las = laspy.read(str(in_path))
    n = len(las.points)
    print(f"[convert] Points: {n:,}")

    x_ft = np.asarray(las.x, dtype=np.float64)
    y_ft = np.asarray(las.y, dtype=np.float64)
    z_ft = np.asarray(las.z, dtype=np.float64)

    print(f"[convert] Input bounds (feet): "
          f"X=[{x_ft.min():.1f}, {x_ft.max():.1f}] "
          f"Y=[{y_ft.min():.1f}, {y_ft.max():.1f}] "
          f"Z=[{z_ft.min():.1f}, {z_ft.max():.1f}]")

    origin_x_ft = args.origin_x if args.origin_x is not None else x_ft.min()
    origin_y_ft = args.origin_y if args.origin_y is not None else y_ft.min()
    origin_z_ft = args.origin_z if args.origin_z is not None else z_ft.min()

    print(f"[convert] Origin (feet, original CRS): "
          f"X={origin_x_ft:.2f} Y={origin_y_ft:.2f} Z={origin_z_ft:.2f}")

    # Shift first (in feet, to keep the subtraction well-conditioned), then convert units.
    x_m = (x_ft - origin_x_ft) * US_SURVEY_FOOT_TO_METER
    y_m = (y_ft - origin_y_ft) * US_SURVEY_FOOT_TO_METER
    z_m = (z_ft - origin_z_ft) * US_SURVEY_FOOT_TO_METER

    print(f"[convert] Output bounds (local meters): "
          f"X=[{x_m.min():.3f}, {x_m.max():.3f}] "
          f"Y=[{y_m.min():.3f}, {y_m.max():.3f}] "
          f"Z=[{z_m.min():.3f}, {z_m.max():.3f}]")

    # Build a fresh header with appropriate scale/offset for the new (much smaller)
    # coordinate range -- reuses the same point format so all extra dimensions
    # (classification, intensity, return number, etc.) survive untouched.
    new_header = laspy.LasHeader(
        point_format=las.header.point_format, version=las.header.version
    )
    # A finer scale is fine (and appropriate) now that coordinates are small;
    # 1mm resolution is more than enough for this use case.
    new_header.scales = [0.001, 0.001, 0.001]
    new_header.offsets = [0.0, 0.0, 0.0]

    new_las = laspy.LasData(new_header)
    new_las.x = x_m
    new_las.y = y_m
    new_las.z = z_m

    # Copy over all other point dimensions unchanged (classification, intensity,
    # return_number, etc.) -- anything present in the source point format.
    skip = {"X", "Y", "Z", "x", "y", "z"}
    for dim_name in las.point_format.dimension_names:
        if dim_name in skip:
            continue
        try:
            setattr(new_las, dim_name, np.asarray(las[dim_name]))
        except Exception as e:
            print(f"[convert] WARNING: could not copy dimension '{dim_name}': {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_las.write(str(out_path))
    print(f"[convert] Wrote: {out_path}")

    print(f"[convert_result] points={n} "
          f"origin_ft=({origin_x_ft:.2f},{origin_y_ft:.2f},{origin_z_ft:.2f}) "
          f"output_bounds_m=X[{x_m.min():.3f},{x_m.max():.3f}]"
          f"_Y[{y_m.min():.3f},{y_m.max():.3f}]_Z[{z_m.min():.3f},{z_m.max():.3f}] "
          f"output={out_path}")


if __name__ == "__main__":
    main()
