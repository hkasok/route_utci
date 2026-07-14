"""
crop_laz.py -- cut a small spatial window out of a large LAZ file, so you
can test the full pipeline (timing, cluster counts, output quality) in
seconds/minutes on a small area before committing to the full ~1.5km tile.

Two ways to specify the crop region:
  1. --center-x/--center-y + --half-size: a square window centered on a
     point (e.g. a specific building you want to test against)
  2. --xmin/--xmax/--ymin/--ymax: an explicit bounding box

If you don't know good coordinates yet, run with --list-bounds first to
print the full tile's extent, then pick a window inside it.

Run:
    # Just check the tile's overall extent first:
    python3 crop_laz.py --input tile.laz --list-bounds

    # Crop a 200m x 200m window centered on a point:
    python3 crop_laz.py --input tile.laz --output crop_test.laz \
        --center-x 574200 --center-y 2849100 --half-size 100

    # Or an explicit bounding box:
    python3 crop_laz.py --input tile.laz --output crop_test.laz \
        --xmin 574100 --xmax 574300 --ymin 2849000 --ymax 2849200
"""

import argparse
from pathlib import Path

import laspy
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Crop a small test window out of a large LAZ file")
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="Output cropped LAZ path (not needed with --list-bounds)")
    p.add_argument("--list-bounds", action="store_true",
                    help="Just print the input file's X/Y/Z bounds and point count, then exit")

    p.add_argument("--center-x", type=float, default=None)
    p.add_argument("--center-y", type=float, default=None)
    p.add_argument("--half-size", type=float, default=100.0,
                    help="Half-width of the square crop window, meters (default: 100 -> 200x200m box)")

    p.add_argument("--xmin", type=float, default=None)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"[crop] Reading header: {in_path}")
    with laspy.open(str(in_path)) as f:
        header = f.header
        n_total = header.point_count
        mins, maxs = header.mins, header.maxs

    print(f"[crop] Total points: {n_total:,}")
    print(f"[crop] X range: {mins[0]:.2f} - {maxs[0]:.2f}  (span {maxs[0]-mins[0]:.1f} m)")
    print(f"[crop] Y range: {mins[1]:.2f} - {maxs[1]:.2f}  (span {maxs[1]-mins[1]:.1f} m)")
    print(f"[crop] Z range: {mins[2]:.2f} - {maxs[2]:.2f}")

    if args.list_bounds:
        return

    if args.xmin is not None:
        xmin, xmax, ymin, ymax = args.xmin, args.xmax, args.ymin, args.ymax
    elif args.center_x is not None:
        xmin = args.center_x - args.half_size
        xmax = args.center_x + args.half_size
        ymin = args.center_y - args.half_size
        ymax = args.center_y + args.half_size
    else:
        raise ValueError("Provide either --center-x/--center-y/--half-size or "
                          "--xmin/--xmax/--ymin/--ymax (or use --list-bounds to check "
                          "the file's extent first)")

    if args.output is None:
        raise ValueError("--output is required unless using --list-bounds")

    print(f"[crop] Crop window: X=[{xmin:.2f}, {xmax:.2f}] Y=[{ymin:.2f}, {ymax:.2f}]")

    print("[crop] Reading full file (this is the slow part for a big tile) ...")
    las = laspy.read(str(in_path))
    x = np.asarray(las.x)
    y = np.asarray(las.y)

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    n_kept = int(mask.sum())
    print(f"[crop] Points in window: {n_kept:,} / {n_total:,} ({100*n_kept/n_total:.2f}%)")

    if n_kept == 0:
        print("[crop] WARNING: zero points in this window -- check your coordinates are in "
              "the same CRS/units as the file (see the bounds printed above).")
        return

    cropped = las[mask]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.write(str(out_path))
    print(f"[crop] Wrote: {out_path}")


if __name__ == "__main__":
    main()
