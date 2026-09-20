"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds a small multi-face object, assigns at least 3 of the 6 procedural materials to
different face groups, and confirms each face's material slot is set correctly and each
material's node tree has no image/AO node baked into its color output. Invoked as:
    blender --background --factory-startup --python materials_selftest.py -- <repo_root> <result.json>
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


def _has_baked_color_nodes(material: "bpy.types.Material") -> bool:
    forbidden = {"ShaderNodeTexImage", "ShaderNodeAmbientOcclusion"}
    return any(node.bl_idname in forbidden for node in material.node_tree.nodes)


def _multi_face_object() -> "bpy.types.Object":
    mesh = bpy.data.meshes.new("materials_selftest_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("materials_selftest_obj", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _run_checks(mats) -> dict:
    checks: dict = {}
    checks["registered_materials"] = sorted(mats.MATERIAL_BUILDERS.keys())

    obj = _multi_face_object()
    assert len(obj.data.polygons) == 6, f"expected a 6-face cube, got {len(obj.data.polygons)}"

    assignments = {
        "concrete": [0, 1],
        "brick": [2, 3],
        "metal": [4],
        "glass": [5],
    }
    slot_indices: dict[str, int] = {}
    for name, face_indices in assignments.items():
        slot_indices[name] = mats.assign_material(obj, face_indices, name)

    face_report = []
    for polygon in obj.data.polygons:
        slot = polygon.material_index
        material_name = obj.data.materials[slot].name if slot < len(obj.data.materials) else None
        face_report.append({"index": polygon.index, "material_index": slot, "material_name": material_name})
    checks["faces"] = face_report

    expected_material_by_face = {}
    for name, face_indices in assignments.items():
        for index in face_indices:
            expected_material_by_face[index] = name
    checks["faces_match_expected"] = all(
        face["material_name"] == expected_material_by_face[face["index"]] for face in face_report
    )

    checks["slot_count"] = len(obj.data.materials)
    checks["slots_distinct"] = len(set(slot_indices.values())) == len(slot_indices)

    all_materials = {name: mats.get_material(name) for name in mats.MATERIAL_BUILDERS}
    checks["no_baked_color"] = {
        name: not _has_baked_color_nodes(material) for name, material in all_materials.items()
    }
    checks["use_nodes"] = {name: material.use_nodes for name, material in all_materials.items()}

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
            checks["faces_match_expected"]
            and checks["slots_distinct"]
            and all(checks["no_baked_color"].values())
            and all(checks["use_nodes"].values())
            and len(checks["registered_materials"]) == 7
        )
        payload = {"ok": ok, "checks": checks}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
