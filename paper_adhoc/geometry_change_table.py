#!/usr/bin/env python3
"""Old (pre-11-Oct-2026) versus rebuilt geometry, per scene: roof fit against
the building returns, rays from open air reaching the underside of a
floating floor, courtyard area kept, crown volume and canopy cover.
Writes paper_adhoc/geometry_change_table.json."""
import json, subprocess, sys
from pathlib import Path
import numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
R = Path(__file__).resolve().parents[1]
TOOLS = R / "geometry_rebuild_20261011" / "tools"
out = {}
cases = [f"lisbon{n}" for n in range(1, 7)] + ["MMC"]
for c in cases:
    new = R / "input" / c / "geometry"
    old = R / "input" / c / "geometry.bak_pregeom_20261011"
    pts = (R / "geometry_rebuild_20261011" / ("mmc/00_split" if c == "MMC" else f"input_new/{c}/build/classified_points")
           / "building_points.npy")
    row = {}
    for tag, g in (("old", old), ("new", new)):
        chk = json.loads(subprocess.run([sys.executable, str(TOOLS / "check_buildings.py"),
                                         str(g / "building_final.stl"), str(pts)],
                                        capture_output=True, text=True).stdout)
        leak = subprocess.run([sys.executable, str(TOOLS / "leak_test.py"), str(g / "building_final.stl"),
                               str(g / "ground_and_water_final.stl")], capture_output=True, text=True).stdout
        under = float(leak.split("underside of a floor: ")[1].split("(")[1].split(")")[0])
        veg = trimesh.load(g / "vegetation_final.stl", force="mesh", process=False)
        row[tag] = {"roof_minus_return_m": chk["roof_minus_return_m"],
                    "building_faces": chk["faces"],
                    "floor_underside_hit_fraction": under,
                    "crown_volume_m3": float(abs(veg.volume)),
                    "crown_faces": int(len(veg.faces))}
    rep = new / "building_final_lod1_report.json"
    if rep.is_file():
        r = json.loads(rep.read_text())
        row["new"].update(courtyard_area_kept_m2=r["courtyard_area_kept_m2"], n_parts=r["n_parts"])
    out[c] = row
    print(c, json.dumps({t: {k: (v if not isinstance(v, dict) else v["mae"]) for k, v in row[t].items()}
                         for t in row}), flush=True)
(R / "paper_adhoc" / "geometry_change_table.json").write_text(json.dumps(out, indent=1))
