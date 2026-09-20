"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds a small material'd box, runs it through assets/lod.py's generate_lods (so the scene has
three named _LOD0/_LOD1/_LOD2 objects with the baked normal maps LOD1/2 always carry), exports
the group with assets/export.py's export_asset, and reports the written glb/textures/asset.json
as JSON, so the pytest suite gets automated coverage of bpy-only code it cannot import directly.
Invoked as:
    blender --background --factory-startup --python export_selftest.py -- <repo_root> <out_dir> <result.json>
"""

import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy


def _read_args() -> tuple[Path, Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, out_dir, result_path = argv[separator + 1:separator + 4]
    return Path(repo_root), Path(out_dir), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _detailed_box(name: str, material: "bpy.types.Material") -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _uv_unwrap(obj: "bpy.types.Object") -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def _box_at(name: str, location, material: "bpy.types.Material") -> "bpy.types.Object":
    from assets import hardsurface as hs

    obj = hs.bevelled_box(name, size=(1.0, 1.0, 1.0), bevel_width=0.0, location=location)
    obj.data.materials.append(material)
    return obj


def _no_drift_across_repeated_export_calls(export_module, out_dir) -> dict:
    """A live MCP session rebuilds individual parts of a multi-object asset and re-exports
    between fixes. export_asset must never permanently move an object: rebuild anchor fresh, then
    export it alone, then add a second object built at different, asymmetric coordinates and
    export both together - anchor's obj.location must come back to exactly where it started
    both times, or a later patch rebuilding just one part would drift out of alignment with parts
    left untouched, which is exactly what happened to the windmill's tower sections."""
    material = bpy.data.materials.new("drift_test_material")
    material.use_nodes = True

    anchor = _box_at("drift_anchor", (2.0, 0.0, 0.5), material)
    location_before_first = tuple(anchor.location)
    export_module.export_asset([anchor], name="drift-test", kind="prop")
    location_after_first = tuple(anchor.location)

    companion = _box_at("drift_companion", (-5.0, 9.0, 0.5), material)
    export_module.export_asset([anchor, companion], name="drift-test", kind="prop")
    location_after_second = tuple(anchor.location)

    return {
        "location_before_first": location_before_first,
        "location_after_first": location_after_first,
        "location_after_second": location_after_second,
        "no_drift": location_before_first == location_after_first == location_after_second,
    }


def main() -> None:
    repo_root, out_dir, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from terrain import config

        config.ASSET_OUT_DIR = out_dir

        from assets import export, lod

        material = bpy.data.materials.new("scratch_atlas")
        material.use_nodes = True
        lod0 = _detailed_box("hero", material)
        _uv_unwrap(lod0)

        lod_set = lod.generate_lods(lod0, ratios=(1.0, 0.5, 0.15), normal_map_size=128)

        name = "export-selftest"
        result = export.export_asset(lod_set.objects, name=name, kind="prop")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        counts = result.triangle_counts
        ok = (
            result.glb_path.is_file()
            and result.manifest_path.is_file()
            and list(counts) == ["LOD0", "LOD1", "LOD2"]
            and counts["LOD0"] > counts["LOD1"] > counts["LOD2"] > 0
            and len(result.materials) == 3
            and len(result.textures) == 2
            and all((out_dir / name / "textures" / t).is_file() for t in result.textures)
            and all(result.bbox_max[i] >= result.bbox_min[i] for i in range(3))
            and manifest["kind"] == "prop"
            and manifest["contact_sheet"] == export.CONTACT_SHEET_NAME
        )

        drift_check = _no_drift_across_repeated_export_calls(export, out_dir)
        ok = ok and drift_check["no_drift"]

        payload = {"ok": ok, "result": result.as_dict(), "manifest": manifest, "drift_check": drift_check}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
