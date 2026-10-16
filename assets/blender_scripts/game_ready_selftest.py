"""Runs inside Blender's own interpreter via `blender --background --python`.

Exercises assets/game_ready.py's game_ready_pass on a small multi-object, multi-material scene
and reports the resulting geometry/origin/UV/material state as JSON, so the pytest suite gets
automated coverage of bpy-only code it cannot import directly. Invoked as:
    blender --background --factory-startup --python game_ready_selftest.py -- <repo_root> <result.json>
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


def _box(name: str, size: tuple, location: tuple, material: "bpy.types.Material") -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    sx, sy, sz = size
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy
        v.co.z = (v.co.z + 0.5) * sz
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    obj.scale = (1.5, 1.5, 1.5)
    obj.rotation_euler = (0.0, 0.0, 0.3)
    return obj


def _mesh_report(obj: "bpy.types.Object") -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    loose = sum(1 for v in bm.verts if not v.link_faces)
    bm.free()
    return {"non_manifold_edges": non_manifold, "loose_verts": loose, "location": list(obj.location)}


def _collapsed_faces(obj: "bpy.types.Object") -> int:
    layer = obj.data.uv_layers.active
    collapsed = 0
    for polygon in obj.data.polygons:
        uvs = [layer.data[index].uv for index in polygon.loop_indices]
        area = 0.0
        for i in range(len(uvs)):
            a, b = uvs[i], uvs[(i + 1) % len(uvs)]
            area += a.x * b.y - b.x * a.y
        if abs(area) < 1e-12:
            collapsed += 1
    return collapsed


def _moving_part_case(gr) -> dict:
    """A static stub and a moving <part>_ANIM plug, each half sunk into a body: the stub's buried
    faces lose their texels, the plug keeps its pivot and every face, since it uncovers them when
    it moves."""
    from assets import motion

    material = bpy.data.materials.new("scratch_moving")
    body = _box("body", (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), material)
    stub = _box("stub", (0.4, 0.4, 0.4), (-0.5, 0.0, 1.7), material)
    plug = _box("plug", (0.4, 0.4, 0.4), (0.5, 0.0, 1.7), material)
    for obj in (body, stub, plug):
        obj.scale = (1.0, 1.0, 1.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    motion.pivot_part(plug, (0.5, 0.0, 1.7), "Z")
    gr.game_ready_pass([body, stub, plug], atlas_size=512)
    return {
        "plug_location": [round(value, 6) for value in plug.location],
        "stub_collapsed": _collapsed_faces(stub),
        "plug_collapsed": _collapsed_faces(plug),
    }


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from assets import game_ready as gr

        material_a = bpy.data.materials.new("scratch_a")
        material_b = bpy.data.materials.new("scratch_b")
        box_a = _box("box_a", (2.0, 1.0, 3.0), (5.0, 2.0, 1.0), material_a)
        box_b = _box("box_b", (1.0, 1.0, 1.0), (5.0, 2.0, 4.0), material_b)
        objects = [box_a, box_b]

        report = gr.game_ready_pass(objects, atlas_size=1024)

        mesh_checks = {obj.name: _mesh_report(obj) for obj in objects}
        materials_used = {slot.material.name for obj in objects for slot in obj.material_slots if slot.material}
        uv_bounds = []
        for obj in objects:
            layer = obj.data.uv_layers.active
            for loop in layer.data:
                uv_bounds.append((loop.uv.x, loop.uv.y))
        uv_min = [min(u for u, _ in uv_bounds), min(v for _, v in uv_bounds)]
        uv_max = [max(u for u, _ in uv_bounds), max(v for _, v in uv_bounds)]

        ok = (
            report.triangle_count > 0
            and materials_used == {report.material_name}
            and all(check["non_manifold_edges"] == 0 and check["loose_verts"] == 0 for check in mesh_checks.values())
            and all(list(obj.location) == [0.0, 0.0, 0.0] for obj in objects)
            and uv_min[0] >= -1e-6 and uv_min[1] >= -1e-6
            and uv_max[0] <= 1.0 + 1e-6 and uv_max[1] <= 1.0 + 1e-6
        )
        moving = _moving_part_case(gr)
        ok = (
            ok
            and moving["plug_location"] == [0.5, 0.0, 1.7]
            and moving["stub_collapsed"] > 0
            and moving["plug_collapsed"] == 0
        )
        payload = {
            "ok": ok,
            "moving": moving,
            "report": report.as_dict(),
            "mesh_checks": mesh_checks,
            "materials_used": sorted(materials_used),
            "uv_min": uv_min,
            "uv_max": uv_max,
        }
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
