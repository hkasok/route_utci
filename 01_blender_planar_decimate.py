"""
Stage 1 -- Blender headless planar decimation
================================================

Collapses coplanar triangles (flat roofs, ground, walls -- the classic
LIDAR-grid over-tessellation) using Blender's Decimate modifier in
"Planar" mode (API name: decimate_type='DISSOLVE'). This does NOT
approximate curved geometry the way quadric edge-collapse does -- it
only merges faces that are already (near-)coplanar within
`angle_limit`, so it should not distort geometry.

Run headless:

    blender --background --factory-startup --python 01_blender_planar_decimate.py -- \
        --input /path/to/input.stl \
        --output /path/to/output_dir/01_planar_decimated.stl \
        --angle-limit-deg 5.0 \
        [--delimit-normal] [--no-delimit-normal]

Notes
-----
* `--factory-startup` avoids the user's local Blender preferences/addons
  affecting a batch/automated run -- important for reproducibility.
* angle_limit is the maximum angle (degrees) between adjacent face
  normals for them to be considered "the same plane" and dissolved.
  Larger = more aggressive (merges gently-curved surfaces too, which
  can matter for e.g. slightly curved LIDAR-derived roofs); smaller =
  more conservative (only truly flat regions collapse).
* `delimit={'NORMAL'}` (the default here) stops dissolving across a
  sharp edge even if the angle_limit would otherwise allow it -- this
  is what keeps building corners crisp.
"""

import sys
import argparse
from pathlib import Path

import bpy


def parse_args():
    # Blender passes its own args first, then "--", then ours.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Planar decimation (Blender Decimate/DISSOLVE)")
    p.add_argument("--input", required=True, help="Input STL path")
    p.add_argument("--output", required=True, help="Output STL path (will be created)")
    p.add_argument("--angle-limit-deg", type=float, default=5.0,
                    help="Max angle (deg) between adjacent face normals to treat as coplanar")
    p.add_argument("--no-delimit-normal", action="store_true",
                    help="Disable the NORMAL delimiter (more aggressive, may round sharp edges)")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _op_exists(module, op_name):
    """
    bpy.ops attribute access is lazy: hasattr() returns True even for
    operators that don't exist (it only fails at call time). dir()
    reflects what's actually registered, so use that instead.
    """
    return op_name in dir(module)


def import_stl(path):
    """Import STL using whichever operator this Blender version provides."""
    path = str(path)
    if _op_exists(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    elif _op_exists(bpy.ops, "import_mesh") and _op_exists(bpy.ops.import_mesh, "stl"):
        bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise RuntimeError("No STL import operator found in this Blender version.")


def export_stl(path):
    """Export STL using whichever operator this Blender version provides."""
    path = str(path)
    # Select everything so export operators that respect selection still work.
    bpy.ops.object.select_all(action="SELECT")
    if _op_exists(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    elif _op_exists(bpy.ops, "export_mesh") and _op_exists(bpy.ops.export_mesh, "stl"):
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)
    else:
        raise RuntimeError("No STL export operator found in this Blender version.")


def total_face_count():
    total = 0
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            total += len(obj.data.polygons)
    return total


def join_all_meshes():
    """
    Merge every imported mesh object into a single object so the
    Decimate modifier (and later stages) operate on one consistent
    mesh. STL has no object hierarchy, so this is safe/expected.
    """
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No mesh objects found after import.")

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]

    if len(mesh_objs) > 1:
        bpy.ops.object.join()

    return bpy.context.view_layer.objects.active


def apply_planar_decimate(obj, angle_limit_deg, use_delimit_normal):
    # Ensure single-user mesh data -- STL import / join can leave the mesh
    # data-block multi-user, which blocks modifier_apply.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.make_single_user(object=True, obdata=True)

    dm = obj.modifiers.new(name="PlanarDecimate", type="DECIMATE")
    dm.decimate_type = "DISSOLVE"
    dm.angle_limit = angle_limit_deg * 3.141592653589793 / 180.0
    dm.delimit = {"NORMAL"} if use_delimit_normal else set()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=dm.name)

    # "Planar" dissolve merges coplanar faces into n-gons (not triangles).
    # STL only supports triangles, so the exporter will re-triangulate
    # those n-gons on write -- meaning the *actual* triangle count in the
    # output file is higher than obj.data.polygons right after dissolve.
    # Triangulate explicitly now so face counts we report/log match what
    # actually lands in the STL.
    tm = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=tm.name)


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input STL not found: {in_path}")

    print(f"[stage1] Loading: {in_path}")
    clear_scene()
    import_stl(in_path)

    faces_before = total_face_count()
    print(f"[stage1] Faces before planar decimate: {faces_before}")

    obj = join_all_meshes()

    apply_planar_decimate(
        obj,
        angle_limit_deg=args.angle_limit_deg,
        use_delimit_normal=not args.no_delimit_normal,
    )

    faces_after = len(obj.data.polygons)
    reduction_pct = 100.0 * (1.0 - faces_after / faces_before) if faces_before else 0.0
    print(f"[stage1] Faces after planar decimate:  {faces_after} "
          f"({reduction_pct:.1f}% reduction)")

    export_stl(out_path)
    print(f"[stage1] Wrote: {out_path}")

    # Machine-readable summary line for the orchestrating shell script to parse
    print(f"[stage1_result] faces_before={faces_before} faces_after={faces_after} "
          f"reduction_pct={reduction_pct:.4f} output={out_path}")


if __name__ == "__main__":
    main()
