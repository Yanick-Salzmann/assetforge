"""Runs inside Blender's own interpreter via `blender --background --python`.

Assembles a small shed from every assets/hardsurface.py helper, assigns it real hard-surface
materials (assets/materials.py), runs it through assets/game_ready.py's UV/atlas steps and
assets/atlas_bake.py's bake, then exports it with assets/export.py:export_asset - a visual smoke
test of what the hard-surface pipeline can currently build end to end (geometry, materials,
textures), independent of any live MCP session. Invoked as:
    blender --background --factory-startup --python demo_showcase_build.py -- <repo_root> <glb_path> <result_json>

glb_path is expected in the out/assets/<name>/<name>.glb shape assets/export.py:export_asset
itself writes to - config.ASSET_OUT_DIR is derived back out of it (glb_path.parent.parent) so this
script writes to wherever the caller's own config.ASSET_OUT_DIR pointed, tests included.
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
    repo_root, glb_path, result_path = argv[separator + 1:separator + 4]
    return Path(repo_root), Path(glb_path), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _grid_plane(name: str, size_x: float, size_y: float, location) -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=3, y_segments=3, size=1.0)
    for v in bm.verts:
        v.co.x *= size_x * 0.5
        v.co.y *= size_y * 0.5
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj


def _build_shed(hs, materials) -> list:
    walls = hs.bevelled_box("shed_walls", size=(3.0, 2.4, 2.4), bevel_width=0.02, bevel_segments=2)

    door_frame_obj = hs.door_frame(
        walls, centre=(0.0, 0.0, 0.0), size=(0.9, 2.0), wall_thickness=0.7, wall_normal=(0.0, -1.0, 0.0)
    )

    walls = hs.window_recess(
        walls, centre=(-1.0, 0.0, 1.35), size=(0.6, 0.7), depth=0.05, wall_thickness=0.7, wall_normal=(0.0, -1.0, 0.0)
    )
    walls = hs.window_recess(
        walls, centre=(1.0, 0.0, 1.35), size=(0.6, 0.7), depth=0.05, wall_thickness=0.7, wall_normal=(0.0, -1.0, 0.0)
    )

    hs.greeble(walls, count=10, depth_range=(0.02, 0.05), seed=7)

    roof = _grid_plane("shed_roof", 3.0, 2.4, (0.0, 0.0, 2.4))
    hs.roof_edge(roof, overhang=0.3, thickness=0.04)

    trim = hs.cornice(
        "shed_trim", length=3.0, profile_size=(0.06, 0.05), location=(0.0, -1.25, 2.35)
    )

    posts_base = hs.bevelled_box("fence_post", size=(0.08, 0.08, 1.0), bevel_width=0.01, bevel_segments=1, location=(2.4, -1.6, 0.0))
    hs.array(posts_base, count=4, offset=(0.5, 0.0, 0.0))

    step_half = hs.bevelled_box("step", size=(0.5, 0.35, 0.15), bevel_width=0.01, bevel_segments=1, location=(0.3, -1.4, 0.0))
    hs.mirror(step_half, axis="X")

    glass_pane = hs.bevelled_box(
        "window_glass", size=(0.6, 0.02, 0.7), bevel_width=0.0, bevel_segments=0, location=(-1.0, 0.0, 1.35)
    )

    materials.assign_material(walls, None, "plaster")
    materials.assign_material(roof, None, "roofing")
    materials.assign_material(door_frame_obj, None, "metal")
    materials.assign_material(trim, None, "metal")
    materials.assign_material(posts_base, None, "metal")
    materials.assign_material(step_half, None, "concrete")
    materials.assign_material(glass_pane, None, "glass")

    return [walls, door_frame_obj, roof, trim, posts_base, step_half, glass_pane]


def main() -> None:
    repo_root, glb_path, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from terrain import config

        config.ASSET_OUT_DIR = glb_path.parent.parent

        from assets import atlas_bake, export, game_ready
        from assets import hardsurface as hs
        from assets import materials

        objects = _build_shed(hs, materials)

        for obj in objects:
            hs.apply_transforms(obj)
            hs.recalc_normals(obj)

        game_ready.uv_unwrap_and_pack(objects)
        bake_result = atlas_bake.bake_atlas(objects, atlas_size=game_ready.ATLAS_SIZE)
        material = game_ready.consolidate_material(objects, bake_result=bake_result)
        game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)

        triangle_count = 0
        for obj in objects:
            obj.data.calc_loop_triangles()
            triangle_count += len(obj.data.loop_triangles)

        name = glb_path.stem
        result = export.export_asset(objects, name=name, kind="prop")

        payload = {
            "ok": True,
            "triangle_count": triangle_count,
            "object_names": [obj.name for obj in objects],
            "textures": result.textures,
            "materials": result.materials,
        }
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
