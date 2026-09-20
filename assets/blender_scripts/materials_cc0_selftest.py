"""Runs inside Blender's own interpreter via `blender --background --python`.

Confirms assets/materials.py's get_material()/assign_material() can resolve a CC0 material name
(from library/asset_materials.json) and a procedural material name interchangeably, that the CC0
material's node tree actually wires in the pulled image textures, and that an unknown name still
raises. Invoked as:
    blender --background --factory-startup --python materials_cc0_selftest.py -- <repo_root> <result.json>
"""

import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, result_path = argv[separator + 1:separator + 3]
    return Path(repo_root), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _multi_face_object() -> "bpy.types.Object":
    mesh = bpy.data.meshes.new("materials_cc0_selftest_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("materials_cc0_selftest_obj", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _run_checks(mats) -> dict:
    checks: dict = {}
    index = mats.load_cc0_index()
    checks["cc0_names"] = sorted(index)
    assert index, "library/asset_materials.json has no materials; run the pull first"

    obj = _multi_face_object()
    cc0_name = sorted(index)[0]
    assignments = {cc0_name: [0, 1], "concrete": [2, 3]}
    slot_indices = {name: mats.assign_material(obj, faces, name) for name, faces in assignments.items()}
    checks["slots_distinct"] = len(set(slot_indices.values())) == len(slot_indices)

    cc0_material = mats.get_material(cc0_name)
    image_nodes = [n for n in cc0_material.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage"]
    checks["cc0_image_node_count"] = len(image_nodes)
    checks["cc0_images_loaded"] = all(n.image is not None for n in image_nodes)

    bsdf = next(n for n in cc0_material.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    checks["base_color_from_image"] = bsdf.inputs["Base Color"].links[0].from_node.bl_idname == "ShaderNodeTexImage"
    checks["normal_from_normal_map"] = bsdf.inputs["Normal"].links[0].from_node.bl_idname == "ShaderNodeNormalMap"

    try:
        mats.get_material("not_a_real_material")
        checks["unknown_name_raises"] = False
    except ValueError:
        checks["unknown_name_raises"] = True

    return checks


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from assets import materials as mats

        checks = _run_checks(mats)
        ok = (
            checks["slots_distinct"]
            and checks["cc0_image_node_count"] == 3
            and checks["cc0_images_loaded"]
            and checks["base_color_from_image"]
            and checks["normal_from_normal_map"]
            and checks["unknown_name_raises"]
        )
        payload = {"ok": ok, "checks": checks}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
