#!/usr/bin/env python3
"""paper_solweig_inputs.py -- rasterise a TREC-Route scene for SOLWEIG.

STANDALONE. SOLWEIG works on 2.5-D rasters; TREC-Route on triangulated
meshes. For a like-for-like comparison of the two radiation models the rasters
are cast from the SAME meshes TREC-Route uses (input/<case>/geometry), with
vertical rays at every pixel centre:

    dem   ground mesh, first hit from above
    dsm   max(ground, building) -- buildings first hit from above
    cdsm  vegetation top above ground (relative), 0 where no crown
    tdsm  vegetation base above ground (relative): first hit of an upward ray
          from the ground into the crown, so SOLWEIG's trunk zone follows the
          reconstructed crown base instead of a fixed trunk ratio
    land_cover  UMEP IDs from TREC-Route's own ground-material map
          (0 paved, 1 asphalt, 2 buildings, 5 grass, 6 bare soil, 7 water)

The rasters are written as .npy in local coordinates (row 0 = north edge), with
a JSON sidecar recording the origin, pixel size and the material mapping.

Usage: python3 paper_solweig_inputs.py --case lisbon1 [--pixel 1.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent

# TREC-Route ground material -> UMEP land-cover ID
LC_PAVED, LC_ASPHALT, LC_BUILDING, LC_GRASS, LC_BARE, LC_WATER = 0, 1, 2, 5, 6, 7


def material_to_landcover(name: str) -> int:
    n = name.lower()
    if "water" in n:
        return LC_WATER
    if "asphalt" in n:
        return LC_ASPHALT
    if "grass" in n or "turf" in n or "lawn" in n:
        return LC_GRASS
    if "bare" in n or "gravel" in n or "soil" in n or "dirt" in n or "sand" in n:
        return LC_BARE
    return LC_PAVED   # calcada, paving stone, concrete, plaza, crossing, sports, generic ground


def intersector(mesh):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    return RayMeshIntersector(mesh)


def first_hit_z(inter, origins, direction, batch=200000):
    """z of the first hit along `direction` (nan where none) and the triangle id."""
    z = np.full(len(origins), np.nan)
    tri = np.full(len(origins), -1, dtype=np.int64)
    d = np.tile(direction, (1, 1))
    for s in range(0, len(origins), batch):
        o = origins[s:s + batch]
        loc, ir, it = inter.intersects_location(o, np.repeat(d, len(o), axis=0), multiple_hits=False)
        z[s + ir] = loc[:, 2]
        tri[s + ir] = it
    return z, tri


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--pixel", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "run_output" / args.case / "solweig" / "inputs"
    out.mkdir(parents=True, exist_ok=True)

    geo = ROOT / "input" / args.case / "geometry"
    ground = trimesh.load(geo / "ground_and_water_final.stl", force="mesh")
    build = trimesh.load(geo / "building_final.stl", force="mesh")
    veg = trimesh.load(geo / "vegetation_final.stl", force="mesh")
    gi, bi, vi = intersector(ground), intersector(build), intersector(veg)

    (x0, y0, _), (x1, y1, _) = ground.bounds
    px = args.pixel
    xs = np.arange(x0 + px / 2, x1, px)
    ys = np.arange(y1 - px / 2, y0, -px)          # row 0 = north
    X, Y = np.meshgrid(xs, ys)
    shape = X.shape
    top = max(ground.bounds[1, 2], build.bounds[1, 2], veg.bounds[1, 2]) + 50.0
    o_top = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, top)])
    down = np.array([[0.0, 0.0, -1.0]])

    print(f"{args.case}: {shape[1]} x {shape[0]} pixels at {px} m")
    dem, gtri = first_hit_z(gi, o_top, down)
    if np.isnan(dem).any():                        # tiny gaps at the mesh rim
        from scipy.interpolate import NearestNDInterpolator
        ok = ~np.isnan(dem)
        dem[~ok] = NearestNDInterpolator(o_top[ok, :2], dem[ok])(o_top[~ok, :2])
    bz, _ = first_hit_z(bi, o_top, down)
    vz, _ = first_hit_z(vi, o_top, down)
    o_up = np.column_stack([X.ravel(), Y.ravel(), dem + 0.05])
    vb, _ = first_hit_z(vi, o_up, np.array([[0.0, 0.0, 1.0]]))

    is_bld = np.isfinite(bz) & (bz > dem + 0.5)
    dsm = np.where(is_bld, bz, dem)
    has_veg = np.isfinite(vz) & (vz > dem + 0.5) & ~is_bld
    cdsm = np.where(has_veg, vz - dem, 0.0)
    tdsm = np.where(has_veg & np.isfinite(vb) & (vb < vz), np.clip(vb - dem, 0.0, None), 0.0)
    tdsm = np.minimum(tdsm, cdsm)

    cat = json.loads((ROOT / "run_output" / args.case / "osm_ground_materials"
                      / "ground_material_catalog.json").read_text())
    names = cat["material_names"]
    mid = np.load(ROOT / "run_output" / args.case / "osm_ground_materials"
                  / "ground_face_materials.npz")["material_id"]
    lut = np.array([material_to_landcover(n) for n in names], dtype=np.int16)
    lc = np.full(X.size, LC_PAVED, dtype=np.int16)
    hit = gtri >= 0
    lc[hit] = lut[mid[gtri[hit]]]
    lc[is_bld] = LC_BUILDING

    for name, arr in (("dem", dem), ("dsm", dsm), ("cdsm", cdsm), ("tdsm", tdsm), ("land_cover", lc)):
        np.save(out / f"{name}.npy", arr.reshape(shape).astype(np.float32 if name != "land_cover" else np.int16))
    meta = dict(case=args.case, pixel_size=px, x_min=float(xs[0] - px / 2), y_max=float(ys[0] + px / 2),
                shape=list(shape), coordinates="local (case.json local_origin)",
                landcover_map={n: int(material_to_landcover(n)) for n in names},
                fractions=dict(building=float(is_bld.mean()), vegetation=float(has_veg.mean()),
                               trunk_zone_mean_m=float(tdsm[has_veg].mean()) if has_veg.any() else 0.0,
                               landcover={int(k): float((lc == k).mean()) for k in np.unique(lc)}))
    (out / "raster_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["fractions"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
