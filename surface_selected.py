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

ONE ROUTE AT A TIME (default route 2)
-------------------------------------
Rays are cast from ONE route only -- --route-id, default 2 -- so "selected" is
what THAT route sees and everything else (including surfaces the other routes
see) lands in the unselected half. Route membership comes from
path_segment_id.npy written by stage 05: it holds the polyline index of each
route point, and route_polylines.pkl stores one polyline per route in route
order, so route_id = segment_id + 1.
Pass --all-routes to use every route at once (which is what the real 05a run
does, and the only mode where --facets-npz can match exactly).

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
  <stem>_route<N>_selected.stl     triangles selected for longwave computation
  <stem>_route<N>_unselected.stl   the remaining triangles
  <ground-stem>_route<N>_selected_material_<material>.stl
                                   selected ground split by configured material
  selection_split_report.txt       face/area counts and percentages per mesh
  ground_material_selection_report.csv
                                   selected face/area totals by ground material
  selected_faces_<stem>_route<N>.npy  the selected face indices (for scripting)

The route tag is in the filenames so running a different --route-id cannot
silently overwrite a previous route's split (it is "all" for --all-routes).

Coordinates are preserved exactly (no transform), so the two halves overlay
the original STL and each other.

Run (no arguments needed after a normal pipeline run -- all inputs default to
the pipeline's standard locations, viewpoints default to route 2):
    python3 surface_selected.py

    # a different route, or all of them
    python3 surface_selected.py --route-id 1
    python3 surface_selected.py --all-routes

    # non-standard locations
    python3 surface_selected.py \
        --buildings-stl path/to/building_final.stl \
        --vegetation-stl path/to/vegetation_final.stl \
        --ground-stl path/to/ground_and_water_final.stl \
        --mrt-dir path/to/mrt_out

Optional cross-check against a real 05a run (only meaningful with
--all-routes, since 05a traces every route):
        --all-routes --facets-npz run_output/thermal_out/facets.npz
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from thermal_common import (get_intersector, make_sphere_directions,
                            nearest_hit_multi)
from osm_ground_materials import (GROUND_FACE_MATERIAL_MAP,
                                  GROUND_MATERIAL_CATALOG)

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
    # Defaults are the pipeline's standard locations (same as start.sh), so a
    # bare `python3 surface_selected.py` works after a normal run.
    p.add_argument("--buildings-stl", default="out_full/02_final/building_final.stl",
                   help="(default: out_full/02_final/building_final.stl)")
    p.add_argument("--vegetation-stl", default="out_full/02_final/vegetation_final.stl",
                   help="(default: out_full/02_final/vegetation_final.stl)")
    p.add_argument("--ground-stl", default="out_full/02_final/ground_and_water_final.stl",
                   help="(default: out_full/02_final/ground_and_water_final.stl)")
    p.add_argument("--mrt-dir", default="run_output/mrt_facet_out",
                   help="Output dir of 05_mrt_network_raytrace.py (needs "
                        "path_xyz.npy + path_segment_id.npy -- the route points "
                        "to look from). Default: run_output/mrt_facet_out")
    p.add_argument("--output-dir", default="surface_selected",
                   help="Output folder (default: ./surface_selected)")
    p.add_argument(
        "--ground-material-dir", default="run_output/osm_ground_materials",
        help="Folder containing ground_face_materials.npz and "
             "ground_material_catalog.json. These classify selected ground "
             "faces into separate material STLs (default: "
             "run_output/osm_ground_materials)")
    p.add_argument(
        "--no-ground-material-stls", action="store_true",
        help="Skip the additional selected-ground STL split by material")

    p.add_argument("--route-id", type=int, default=2,
                   help="Cast rays from THIS route only, so 'selected' is what "
                        "this one route sees and everything else (including what "
                        "the other routes see) is unselected. 1-based, matching "
                        "the route ids reported by stages 08/09 (default: 2)")
    p.add_argument("--all-routes", action="store_true",
                   help="Use every route instead of a single one -- this is what "
                        "the real 05a run does, and the only mode in which "
                        "--facets-npz can match exactly.")

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


def load_ground_material_map(material_dir, ground_mesh):
    """Load and validate the production material ID assigned to every ground face."""
    material_dir = Path(material_dir)
    map_path = material_dir / GROUND_FACE_MATERIAL_MAP
    catalog_path = material_dir / GROUND_MATERIAL_CATALOG
    missing = [str(path) for path in (map_path, catalog_path) if not path.is_file()]
    if missing:
        raise SystemExit(
            "ERROR: ground-material export requires the production material "
            f"outputs; missing: {', '.join(missing)}. Run the OSM ground-material "
            "stage first or pass --ground-material-dir.")

    with catalog_path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    material_names = catalog.get("material_names")
    if (not isinstance(material_names, list) or not material_names
            or len(set(material_names)) != len(material_names)
            or not all(isinstance(name, str) and name
                       and name.replace("_", "").isalnum()
                       for name in material_names)):
        raise SystemExit(
            f"ERROR: invalid material_names in {catalog_path}")

    with np.load(map_path) as archive:
        if "material_id" not in archive:
            raise SystemExit(f"ERROR: {map_path} does not contain material_id")
        material_id = np.asarray(archive["material_id"])
    if material_id.ndim != 1 or len(material_id) != len(ground_mesh.faces):
        raise SystemExit(
            f"ERROR: {map_path} has {len(material_id):,} material IDs but the "
            f"ground STL has {len(ground_mesh.faces):,} faces. The files are "
            "from different geometry runs.")
    if not np.issubdtype(material_id.dtype, np.integer):
        if not np.all(np.equal(material_id, np.floor(material_id))):
            raise SystemExit(f"ERROR: {map_path} contains non-integer material IDs")
        material_id = material_id.astype(np.int64)
    else:
        material_id = material_id.astype(np.int64, copy=False)
    if np.any(material_id < 0) or np.any(material_id >= len(material_names)):
        bad = np.unique(material_id[
            (material_id < 0) | (material_id >= len(material_names))])
        raise SystemExit(
            f"ERROR: {map_path} contains out-of-range material IDs {bad.tolist()} "
            f"for {len(material_names)} catalog materials")

    signature = catalog.get("ground_mesh_signature", {})
    expected_faces = signature.get("n_faces")
    if expected_faces is not None and int(expected_faces) != len(ground_mesh.faces):
        raise SystemExit(
            "ERROR: ground material catalog face count does not match ground STL")
    expected_bounds = np.asarray(signature.get("bounds", []), dtype=float)
    if (expected_bounds.shape == (2, 3)
            and not np.allclose(expected_bounds, ground_mesh.bounds,
                                atol=1.0e-5, rtol=0.0)):
        raise SystemExit(
            "ERROR: ground material catalog bounds do not match ground STL; "
            "refusing to assign materials to the wrong geometry")
    return material_id, material_names, map_path, catalog_path


def write_selected_ground_material_stls(mesh, selected_faces, material_id,
                                        material_names, stem, route_tag, out_dir):
    """Intersect selected ground faces with each material and export one STL each."""
    selected_mask = np.zeros(len(mesh.faces), dtype=bool)
    selected_mask[np.asarray(selected_faces, dtype=np.int64)] = True
    rows = []
    exported_faces = []
    for mid, material_name in enumerate(material_names):
        face_idx = np.flatnonzero(selected_mask & (material_id == mid)).astype(np.int64)
        path = out_dir / (
            f"{stem}_{route_tag}_selected_material_{material_name}.stl")
        n_faces, area_m2 = write_half(mesh, face_idx, path)
        np.save(
            out_dir / (
                f"selected_faces_{stem}_{route_tag}_material_{material_name}.npy"),
            face_idx)
        rows.append({
            "material_id": mid,
            "material_name": material_name,
            "selected_face_count": n_faces,
            "selected_area_m2": area_m2,
            "stl_file": path.name,
        })
        if len(face_idx):
            exported_faces.append(face_idx)
        print(f"  {path.name:<72} {n_faces:>9,} faces")

    combined = (np.sort(np.concatenate(exported_faces)) if exported_faces
                else np.empty(0, dtype=np.int64))
    expected = np.sort(np.asarray(selected_faces, dtype=np.int64))
    if not np.array_equal(combined, expected):
        raise RuntimeError(
            "ground material STL split did not conserve the selected ground faces")
    return rows


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

    ground_material_data = None
    if not args.no_ground_material_stls:
        ground_material_data = load_ground_material_map(
            args.ground_material_dir, meshes[MESH_GROUND])
        material_id, material_names, map_path, catalog_path = ground_material_data
        print(f"  Ground materials: {len(material_names)} classes from {map_path}")
        print(f"                    catalog {catalog_path}")

    mrt_dir = Path(args.mrt_dir)
    pxyz = mrt_dir / "path_xyz.npy"
    if not pxyz.is_file():
        raise SystemExit(
            f"ERROR: {pxyz} not found. Run the pipeline through step 3 first "
            f"(./start.sh 3) so the route points exist, or point --mrt-dir at "
            f"an existing MRT output directory.")
    path_xyz = np.load(pxyz)
    n_all_pts = len(path_xyz)

    # ------------------------------------------------------------------
    # Restrict the viewpoints to ONE route unless --all-routes.
    # path_segment_id.npy holds the polyline index per route point, and
    # route_polylines.pkl stores one polyline per route in route order, so
    # route_id = segment_id + 1.
    # ------------------------------------------------------------------
    if args.all_routes:
        route_tag = "all"
        route_desc = "ALL routes"
        print(f"\n  Viewpoints: ALL routes ({n_all_pts:,} points)")
    else:
        seg_path = mrt_dir / "path_segment_id.npy"
        if not seg_path.is_file():
            raise SystemExit(
                f"ERROR: {seg_path} not found, so route membership is unknown. "
                f"Re-run stage 05 (./start.sh 3) to write it, or pass "
                f"--all-routes to use every route point.")
        seg = np.load(seg_path)
        if len(seg) != n_all_pts:
            raise SystemExit(
                f"ERROR: path_segment_id.npy has {len(seg):,} entries but "
                f"path_xyz.npy has {n_all_pts:,}. They are from different runs "
                f"-- re-run stage 05 so both match.")
        route_ids = seg + 1                      # 1-based, as stages 08/09 report
        available = np.unique(route_ids)
        counts = {int(r): int((route_ids == r).sum()) for r in available}
        print(f"\n  Route points per route: "
              + ", ".join(f"route {r}: {c:,}" for r, c in counts.items()))
        if args.route_id not in counts:
            raise SystemExit(
                f"ERROR: --route-id {args.route_id} not present. This MRT run "
                f"contains route(s) {sorted(counts)}. Pick one of those, or use "
                f"--all-routes.")
        keep = route_ids == args.route_id
        path_xyz = path_xyz[keep]
        route_tag = f"route{args.route_id}"
        route_desc = f"route {args.route_id} ONLY"
        print(f"  Viewpoints: {route_desc} -- {len(path_xyz):,} of "
              f"{n_all_pts:,} route points")

    coarse_idx = np.arange(0, len(path_xyz), max(1, args.point_stride))
    coarse_xyz = path_xyz[coarse_idx]
    print(f"  Traced: {len(coarse_xyz):,} points (stride {args.point_stride})")
    print(f"  Longwave range cap: {args.max_distance:g} m")

    print(f"\nRay-casting the full-sphere view from {route_desc} "
          f"(same test as 05a)...")
    selected = selected_faces_by_mesh(meshes, coarse_xyz, args)

    # ------------------------------------------------------------------
    # Optional cross-check against a real 05a run
    # ------------------------------------------------------------------
    crosscheck_lines = []
    if args.facets_npz:
        if not args.all_routes:
            print(f"\n  NOTE: --facets-npz compares against a 05a run that traced "
                  f"ALL routes, but this split used {route_desc}. Fewer selected "
                  f"faces here is EXPECTED, not an error. Use --all-routes for a "
                  f"like-for-like check.")
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
        if any("DIFFERS" in l for l in crosscheck_lines) and args.all_routes:
            print("  NOTE: a difference means the ray parameters here do not "
                  "match that 05a run (check --point-stride / --n-lw-* / "
                  "--max-distance against its config.json).")

    # ------------------------------------------------------------------
    # Split and export
    # ------------------------------------------------------------------
    print(f"\nWriting split meshes to {out_dir}/ ...")
    rows = []
    material_rows = []
    for mid, mesh in meshes.items():
        stem = stl_paths[mid].stem
        n_all = len(mesh.faces)
        sel = selected[mid]
        unsel = np.setdiff1d(np.arange(n_all, dtype=np.int64), sel)

        sel_path = out_dir / f"{stem}_{route_tag}_selected.stl"
        n_sel, a_sel = write_half(mesh, sel, sel_path)
        print(f"  {sel_path.name:<52} {n_sel:>9,} faces")

        n_unsel, a_unsel = 0, 0.0
        if not args.no_unselected:
            unsel_path = out_dir / f"{stem}_{route_tag}_unselected.stl"
            n_unsel, a_unsel = write_half(mesh, unsel, unsel_path)
            print(f"  {unsel_path.name:<52} {n_unsel:>9,} faces")
        else:
            n_unsel = len(unsel)
            a_unsel = float(mesh.area_faces[unsel].sum()) if len(unsel) else 0.0

        np.save(out_dir / f"selected_faces_{stem}_{route_tag}.npy", sel)
        if mid == MESH_GROUND and ground_material_data is not None:
            print("  Selected ground faces by material:")
            material_rows = write_selected_ground_material_stls(
                mesh, sel, material_id, material_names, stem, route_tag, out_dir)
        a_all = float(mesh.area_faces.sum())
        rows.append(dict(mesh=MESH_LABEL[mid], stem=stem, n_all=n_all,
                         n_sel=n_sel, n_unsel=n_unsel, a_all=a_all,
                         a_sel=a_sel, a_unsel=a_unsel, rule=MESH_RULE[mid]))
        # A split must conserve every triangle -- assert it rather than trust it.
        assert n_sel + n_unsel == n_all, (
            f"{stem}: split lost triangles ({n_sel}+{n_unsel} != {n_all})")

    if material_rows:
        material_report_path = out_dir / "ground_material_selection_report.csv"
        with material_report_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(material_rows[0]))
            writer.writeheader()
            writer.writerows(material_rows)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = ["Route-visible surface split (standalone diagnostic)",
             "",
             f"  VIEWPOINTS       : {route_desc}",
             f"  route points     : {len(path_xyz):,} used (of {n_all_pts:,} "
             f"in this MRT run) -> {len(coarse_xyz):,} traced "
             f"(stride {args.point_stride})",
             f"  LW range cap     : {args.max_distance:g} m",
             f"  ray directions   : {args.n_lw_azimuth} az x "
             f"{args.n_lw_elevation} el (full sphere)",
             ""]
    if not args.all_routes:
        lines += [f"  NOTE: 'selected' is what route {args.route_id} sees. Surfaces "
                  f"visible only from the",
                  f"        other routes are in the UNSELECTED half by design.",
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
    if material_rows:
        lines += ["  SELECTED GROUND BY MATERIAL"]
        for item in material_rows:
            lines += [
                f"    {item['material_name']:<28} "
                f"{item['selected_face_count']:>9,} faces  "
                f"{item['selected_area_m2']:>12,.1f} m2"]
        lines += [
            "",
            "    Every selected ground face belongs to exactly one material STL.",
            "    Zero-face configured materials are exported as empty placeholders.",
            "    Details: ground_material_selection_report.csv",
            ""]
    if crosscheck_lines:
        lines += ["  Cross-check vs 05a facets.npz:"] + crosscheck_lines + [""]
    lines += [f"  Files: <stem>_{route_tag}_selected.stl / "
              f"<stem>_{route_tag}_unselected.stl",
              f"         selected_faces_<stem>_{route_tag}.npy (face indices)",
              f"         <ground-stem>_{route_tag}_selected_material_<material>.stl",
              f"         selected_faces_<ground-stem>_{route_tag}_material_<material>.npy",
              "  Coordinates are unchanged, so both halves overlay the original.",
              ""]
    report = "\n".join(lines)
    (out_dir / "selection_split_report.txt").write_text(report)
    print("\n" + report)
    print(f"[surface_selected] output_dir={out_dir} viewpoints={route_tag} "
          + " ".join(f"{r['mesh']}_selected={r['n_sel']}" for r in rows))


if __name__ == "__main__":
    main()
