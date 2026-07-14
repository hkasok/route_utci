"""
01_split_classes.py -- split a classified LAZ into ground+water / building /
vegetation point sets.

ASPRS classification codes used:
    2  = Ground
    9  = Water
    3,4,5 = Low/Medium/High vegetation
    6  = Building

Run:
    python3 01_split_classes.py --input classified.laz --output-dir split/
"""

import argparse
from pathlib import Path

import laspy
import numpy as np

GROUND_CLASSES = {2, 9}          # ground + water combined (one output)
VEGETATION_CLASSES = {3, 4, 5}   # low/medium/high vegetation combined
BUILDING_CLASSES = {6}


def parse_args():
    p = argparse.ArgumentParser(description="Split classified LAZ into ground/building/vegetation")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split] Reading: {in_path}")
    las = laspy.read(str(in_path))
    classification = np.asarray(las.classification)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])

    groups = {
        "ground_and_water": GROUND_CLASSES,
        "building": BUILDING_CLASSES,
        "vegetation": VEGETATION_CLASSES,
    }

    for name, class_set in groups.items():
        mask = np.isin(classification, list(class_set))
        pts = xyz[mask]
        out_path = out_dir / f"{name}_points.npy"
        np.save(out_path, pts)
        print(f"[split] {name}: {len(pts):,} points -> {out_path}")

    print("[split] Done.")


if __name__ == "__main__":
    main()
