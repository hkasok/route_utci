"""
surface_selected.py -- STANDALONE diagnostic (NOT part of the normal pipeline).

Splits each input STL into the surfaces the workflow actually SELECTS for
longwave computation and the surfaces it does not, so you can load both halves
in a mesh viewer and see exactly what the route "sees".

WHAT "SELECTED" MEANS HERE
--------------------------
It is the same test 05a_thermal_facets_select.py applies: from every (strided)
route point a FULL SPHERE of rays is cast against buildings + ground +
vegetation, and a triangle is SELECTED if it is the nearest hit of some ray
within --max-distance. So "selected" = visible from the route AND inside the
longwave range cap. Anything invisible from the route, or beyond the distance
cull, is unselected and never gets a surface temperature in 05b.

This script reuses the SAME functions as 05a (make_sphere_directions,
nearest_hit_multi, get_intersector) and the SAME default parameters, so the
split is faithful rather than an approximation. Because only the ray
DIRECTIONS decide which triangles are hit, the --body-model weighting (which
scales view factors, not visibility) does not change the split.

ONE ASYMMETRY WORTH KNOWING -- VEGETATION
-----------------------------------------
In the real workflow buildings and ground triangles become "thermal facets"
that get a 1D energy balance in 05b. Vegetation NEVER does: rays that hit
canopy are accumulated into a separate w_veg weight and the canopy is assumed
to radiate at air temperature. So under 05a's own definition the vegetation
"selected facet" set is empty.
That would be useless to look at, so for the VEGETATION mesh this script
reports the triangles that are hit by the longwave rays within range -- i.e.
the canopy that actually contributes to the pedestrian's longwave environment.
The report labels which rule was applied to each mesh.

OUTPUTS (in --output-dir, default ./surface_selected)
----------------------------------------------------
  <stem>_selected.stl     triangles selected for longwave computation
  <stem>_unselected.stl   the remaining triangles
  selection_split_report.txt   face/area counts and percentages per mesh
  selected_faces_<stem>.npy    the selected face indices (for scripting)

Coordinates are preserved exactly (no transform), so the two halves overlay
the original STL and each other.

Run (defaults mirror the pipeline's 05a settings):
    python3 surface_selected.py \
        --buildings-stl out_full/02_final/building_final.stl \
        --vegetation-stl out_full/02_final/vegetation_final.stl \
        --ground-stl out_full/02_final/ground_and_water_final.stl \
        --mrt-dir run_output/mrt_facet_out

Optional cross-check against a real 05a run (proves this reproduces it):
        --facets-npz run_output/thermal_out/facets.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh

from thermal_common import (get_intersector, make_sphere_directions,
                            nearest_hit_multi)

# Mesh ids must match 05a_thermal_facets_select.py
MESH_BUILDINGS = 0
MESH_GROUND = 1
MESH_VEGETATION = 2

MESH_LABEL = {MESH_BUILDINGS: "buildings", MESH_GROUND: "ground",
              MESH_VEGETATION: "vegetation"}
# Which rule defines "selected" for each mesh (see module docstring).
MESH_RULE = {
    MESH_BUILDINGS: "thermal facet (gets a 05b surface-temperature solve)",
    MESH_GROUND: "thermal facet (gets a 05b surface-temperature solve)",
    MESH_VEGETATION: "LW-visible canopy (contributes w_veg at air temperature; "
                     "never gets an energy balance)",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Split each input STL into route-visible (selected for "
                    "longwave computation) and unselected surfaces")
    p.add_argument("--buildings-stl", required=True)
    p.add_argument("--vegetation-stl", required=True)
    p.add_argument("--ground-stl", required=True)
    p.add_argument("--mrt-dir", required=True,
                   help="Output dir of 05_mrt_network_raytrace.py (needs "
                        "path_xyz.npy -- the route points to look from)")
    p.add_argument("--output-dir", default="surface_selected",
                   help="Output folder (default: ./surface_selected)")

    # Defaults deliberately identical to 05a so the split matches the workflow.
    p.add_argument("--point-stride", type=int, default=8,
                   help="Use every Nth route point (05a default: 8)")
    p.add_argument("--n-lw-azimuth", type=int, default=24,
                   help="Azimuth samples (05a default: 24)")
    p.add_argument("--n-lw-elevation", type=int, default=18,
                   help="Elevation bands over the full sphere (05a default: 18)")
    p.add_argument("--max-distance", type=float, default=300.0,
                   help="Longwave range cap, meters (05a default: 300). "
                        "Triangles farther than this are UNSELECTED.")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Route points per ray batch (05a default: 500)")

    p.add_argument("--facets-npz", default=None,
                   help="Optional facets.npz from a real 05a run. If given, the "
                        "buildings/ground selection computed here is compared "
                        "against it and any disagreement is reported.")
    p.add_argument("--no-unselected", action="store_true",
                   help="Write only the *_selected.stl halves (skip the "
                        "usually much larger unselected meshes)")
    return p.parse_args()


def selected_faces_by_mesh(meshes, coarse_xyz, args):
    """Return {mesh_id: sorted unique array of selected face indices}.

    Identical ray test to 05a: nearest hit per ray across all three meshes,
    culled at --max-distance. Vectorized with np.unique instead of 05a's
    per-ray Python loop (we only need the face SET, not the view matrix).
    """
    intersectors = [(mid, get_intersector(m)) for mid, m in meshes.items()]
    directions, _weights = make_sphere_directions(
        args.n_lw_azimuth, args.n_lw_elevation, body="cylinder")
    ndirs = len(directions)
    print(f"  LW directions per point: {ndirs} (full sphere)")

    hits = {mid: [] for mid in meshes}
    n_coarse = len(coarse_xyz)
    n_batches = int(np.ceil(n_coarse / args.batch_size))
    t0 = time.time()
    for bi, start in enumerate(range(0, n_coarse, args.batch_size)):
        end = min(start + args.batch_size, n_coarse)
        pts = coarse_xyz[start:end]
        m = len(pts)
        origins = np.repeat(pts, ndirs, axis=0)
        dirs = np.tile(directions, (m, 1))

        hit_mesh, hit_face, _ = nearest_hit_multi(
            intersectors, origins, dirs, args.max_distance)

        for mid in meshes:
            sel = hit_mesh == mid
            if sel.any():
                hits[mid].append(np.unique(hit_face[sel]))

        if (bi + 1) % max(1, n_batches // 10) == 0 or bi == n_batches - 1:
            uniq = sum(len(np.unique(np.concatenate(v))) if v else 0
                       for v in hits.values())
            print(f"  batch {bi + 1}/{n_batches} -- {time.time() - t0:.0f}s "
                  f"elapsed, {uniq:,} unique faces hit so far")

    out = {}
    for mid in meshes:
        out[mid] = (np.unique(np.concatenate(hits[mid])).astype(np.int64)
                    if hits[mid] else np.empty(0, dtype=np.int64))
    return out


def write_half(mesh, face_idx, path):
    """Write the sub-mesh of `face_idx` to `path`. Returns (n_faces, area)."""
    face_idx = np.asarray(face_idx, dtype=np.int64)
    if len(face_idx) == 0:
        # Keep the file slot so the output set is predictable, but empty.
        trimesh.Trimesh().export(str(path))
        return 0, 0.0
    sub = mesh.submesh([face_idx], append=True)
    sub.export(str(path))
    return len(face_idx), float(mesh.area_faces[face_idx].sum())


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_paths = {
        MESH_BUILDINGS: Path(args.buildings_stl),
        MESH_GROUND: Path(args.ground_stl),
        MESH_VEGETATION: Path(args.vegetation_stl),
    }

    print("=" * 70)
    print("Loading geometry...")
    meshes = {}
    for mid, path in stl_paths.items():
        if not path.is_file():
            raise SystemExit(f"ERROR: input STL not found: {path}")
        meshes[mid] = trimesh.load(str(path), force="mesh")
        print(f"  {MESH_LABEL[mid]:<10} {len(meshes[mid].faces):>9,} faces  "
              f"({path.name})")

    mrt_dir = Path(args.mrt_dir)
    pxyz = mrt_dir / "path_xyz.npy"
    if not pxyz.is_file():
        raise SystemExit(
            f"ERROR: {pxyz} not found. Run the pipeline through step 3 first "
            f"(./start.sh 3) so the route points exist, or point --mrt-dir at "
            f"an existing MRT output directory.")
    path_xyz = np.load(pxyz)
    coarse_idx = np.arange(0, len(path_xyz), max(1, args.point_stride))
    coarse_xyz = path_xyz[coarse_idx]
    print(f"\n  Route points: {len(path_xyz):,} full -> {len(coarse_xyz):,} "
          f"traced (stride {args.point_stride})")
    print(f"  Longwave range cap: {args.max_distance:g} m")

    print("\nRay-casting the route's full-sphere view (same test as 05a)...")
    selected = selected_faces_by_mesh(meshes, coarse_xyz, args)

    # ------------------------------------------------------------------
    # Optional cross-check against a real 05a run
    # ------------------------------------------------------------------
    crosscheck_lines = []
    if args.facets_npz:
        fz = np.load(args.facets_npz)
        for mid in (MESH_BUILDINGS, MESH_GROUND):
            ref = np.unique(fz["face_id"][fz["mesh_id"] == mid]).astype(np.int64)
            got = selected[mid]
            only_ref = np.setdiff1d(ref, got)
            only_got = np.setdiff1d(got, ref)
            status = ("EXACT MATCH" if not len(only_ref) and not len(only_got)
                      else f"DIFFERS (+{len(only_got)} here, "
                           f"-{len(only_ref)} vs 05a)")
            crosscheck_lines.append(
                f"  {MESH_LABEL[mid]:<10} 05a {len(ref):,} vs here "
                f"{len(got):,} -> {status}")
        print("\nCross-check vs 05a facets.npz:")
        print("\n".join(crosscheck_lines))
        if any("DIFFERS" in l for l in crosscheck_lines):
            print("  NOTE: a difference means the ray parameters here do not "
                  "match that 05a run (check --point-stride / --n-lw-* / "
                  "--max-distance against its config.json).")

    # ------------------------------------------------------------------
    # Split and export
    # ------------------------------------------------------------------
    print(f"\nWriting split meshes to {out_dir}/ ...")
    rows = []
    for mid, mesh in meshes.items():
        stem = stl_paths[mid].stem
        n_all = len(mesh.faces)
        sel = selected[mid]
        unsel = np.setdiff1d(np.arange(n_all, dtype=np.int64), sel)

        sel_path = out_dir / f"{stem}_selected.stl"
        n_sel, a_sel = write_half(mesh, sel, sel_path)
        print(f"  {sel_path.name:<44} {n_sel:>9,} faces")

        n_unsel, a_unsel = 0, 0.0
        if not args.no_unselected:
            unsel_path = out_dir / f"{stem}_unselected.stl"
            n_unsel, a_unsel = write_half(mesh, unsel, unsel_path)
            print(f"  {unsel_path.name:<44} {n_unsel:>9,} faces")
        else:
            n_unsel = len(unsel)
            a_unsel = float(mesh.area_faces[unsel].sum()) if len(unsel) else 0.0

        np.save(out_dir / f"selected_faces_{stem}.npy", sel)
        a_all = float(mesh.area_faces.sum())
        rows.append(dict(mesh=MESH_LABEL[mid], stem=stem, n_all=n_all,
                         n_sel=n_sel, n_unsel=n_unsel, a_all=a_all,
                         a_sel=a_sel, a_unsel=a_unsel, rule=MESH_RULE[mid]))
        # A split must conserve every triangle -- assert it rather than trust it.
        assert n_sel + n_unsel == n_all, (
            f"{stem}: split lost triangles ({n_sel}+{n_unsel} != {n_all})")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = ["Route-visible surface split (standalone diagnostic)",
             "",
             f"  route points     : {len(path_xyz):,} full -> "
             f"{len(coarse_xyz):,} traced (stride {args.point_stride})",
             f"  LW range cap     : {args.max_distance:g} m",
             f"  ray directions   : {args.n_lw_azimuth} az x "
             f"{args.n_lw_elevation} el (full sphere)",
             ""]
    for r in rows:
        pf = 100.0 * r["n_sel"] / max(1, r["n_all"])
        pa = 100.0 * r["a_sel"] / max(1e-12, r["a_all"])
        lines += [
            f"  {r['mesh'].upper()}  ({r['stem']})",
            f"    rule for 'selected': {r['rule']}",
            f"    faces  selected {r['n_sel']:>9,} / {r['n_all']:>9,} "
            f"({pf:5.1f}%)   unselected {r['n_unsel']:>9,}",
            f"    area   selected {r['a_sel']:>12,.0f} m2 of "
            f"{r['a_all']:>12,.0f} m2 ({pa:5.1f}%)",
            ""]
    if crosscheck_lines:
        lines += ["  Cross-check vs 05a facets.npz:"] + crosscheck_lines + [""]
    lines += ["  Files: <stem>_selected.stl / <stem>_unselected.stl",
              "         selected_faces_<stem>.npy (face indices)",
              "  Coordinates are unchanged, so both halves overlay the original.",
              ""]
    report = "\n".join(lines)
    (out_dir / "selection_split_report.txt").write_text(report)
    print("\n" + report)
    print(f"[surface_selected] output_dir={out_dir} "
          + " ".join(f"{r['mesh']}_selected={r['n_sel']}" for r in rows))


if __name__ == "__main__":
    main()
