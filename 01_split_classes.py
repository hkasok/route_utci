"""
01_split_classes.py -- split a classified LAZ into ground+water / building /
vegetation point sets, in metres.

ASPRS classification codes used:
    2  = Ground
    9  = Water
    3,4,5 = Low/Medium/High vegetation
    6  = Building

UNITS. Every downstream stage assumes metres on all three axes. US LiDAR is
often delivered with mixed units: the 2021 Miami-Dade tiles, for example, are
NAD83(2011) / UTM 17N (metres) horizontally but NAVD88 height in US survey
feet vertically. The tile's CRS is therefore read here and each axis is
converted to metres from its own unit (US survey foot 1200/3937 m,
international foot 0.3048 m). A tile without a usable CRS is refused unless
the caller states the units with --xy-unit/--z-unit, because a silent
assumption is how a whole scene once ended up 3.28 times too tall.

Run:
    python3 01_split_classes.py --input classified.laz --output-dir split/
"""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

GROUND_CLASSES = {2, 9}          # ground + water combined (one output)
VEGETATION_CLASSES = {3, 4, 5}   # low/medium/high vegetation combined
BUILDING_CLASSES = {6}

UNIT_TO_METRE = {
    "metre": 1.0, "meter": 1.0, "m": 1.0,
    "us survey foot": 1200.0 / 3937.0, "us_survey_foot": 1200.0 / 3937.0,
    "ftus": 1200.0 / 3937.0,
    "foot": 0.3048, "ft": 0.3048, "international foot": 0.3048,
}


def factor_for(unit_name: str) -> float:
    key = unit_name.strip().lower()
    if key not in UNIT_TO_METRE:
        raise SystemExit(f"[split] unknown axis unit {unit_name!r}; pass --xy-unit/--z-unit")
    return UNIT_TO_METRE[key]


def axis_units(las_header):
    """(horizontal unit, vertical unit or None) from the tile's CRS."""
    crs = las_header.parse_crs()
    if crs is None:
        return None, None
    horiz = vert = None
    for axis in crs.axis_info:
        direction = (axis.direction or "").lower()
        if direction in ("east", "north") and horiz is None:
            horiz = axis.unit_name
        elif direction == "up":
            vert = axis.unit_name
    return horiz, vert


def parse_args():
    p = argparse.ArgumentParser(description="Split classified LAZ into ground/building/vegetation")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--xy-unit", default=None,
                   help="override the horizontal unit (metre, us survey foot, foot)")
    p.add_argument("--z-unit", default=None,
                   help="override the vertical unit (metre, us survey foot, foot); "
                        "needed when the CRS carries no vertical axis and heights are not metres")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split] Reading: {in_path}")
    las = laspy.read(str(in_path))
    horiz, vert = axis_units(las.header)
    xy_unit = args.xy_unit or horiz
    if xy_unit is None:
        raise SystemExit("[split] the tile has no CRS; pass --xy-unit and --z-unit explicitly")
    if args.z_unit:
        z_unit = args.z_unit
    elif vert is not None:
        z_unit = vert
    else:
        # A 2-D CRS says nothing about heights; by convention they share the
        # horizontal unit (true for metric European tiles and for all-feet
        # State Plane tiles). State it, so it is visible in the log.
        z_unit = xy_unit
        print(f"[split] CRS has no vertical axis; taking heights in the horizontal unit ({xy_unit})")
    fxy, fz = factor_for(xy_unit), factor_for(z_unit)
    print(f"[split] Units: horizontal {xy_unit} (x{fxy:.10f} to m), vertical {z_unit} (x{fz:.10f} to m)")

    classification = np.asarray(las.classification)
    xyz = np.column_stack([np.asarray(las.x) * fxy, np.asarray(las.y) * fxy,
                           np.asarray(las.z) * fz])

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

    (out_dir / "split_units.json").write_text(json.dumps({
        "input": str(in_path), "horizontal_unit": xy_unit, "vertical_unit": z_unit,
        "horizontal_to_metre": fxy, "vertical_to_metre": fz}, indent=1))
    print("[split] Done.")


if __name__ == "__main__":
    main()
