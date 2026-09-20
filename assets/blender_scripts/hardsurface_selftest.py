"""Runs inside Blender's own interpreter via `blender --background --python`.

Exercises every assets/hardsurface.py helper and reports closed/manifold/loose-vertex geometry
checks as JSON, so the pytest suite gets automated coverage of bpy-only code it cannot import
directly. Invoked as:
    blender --background --factory-startup --python hardsurface_selftest.py -- <repo_root> <result.json>
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


def _mesh_report(obj: "bpy.types.Object") -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    loose = sum(1 for v in bm.verts if not v.link_faces)
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    faces = len(bm.faces)
    bm.free()
    return {"non_manifold_edges": non_manifold, "loose_verts": loose, "degenerate_faces": degenerate, "faces": faces}


def _run_checks(hs) -> dict:
    checks: dict[str, dict] = {}

    box = hs.bevelled_box("box", size=(2.0, 1.0, 3.0), bevel_width=0.05, bevel_segments=2)
    checks["bevelled_box"] = _mesh_report(box)

    cutter = hs.bevelled_box("cutter", size=(0.5, 2.0, 0.5), bevel_width=0.0, location=(0.0, 0.0, 1.0))
    cut = hs.panel_cut(box, cutter)
    checks["panel_cut"] = _mesh_report(cut)

    arr_base = hs.bevelled_box("arr_base", size=(0.3, 0.3, 1.0), bevel_width=0.0)
    hs.array(arr_base, count=4, offset=(0.5, 0.0, 0.0))
    checks["array"] = _mesh_report(arr_base)

    half = hs.bevelled_box("half", size=(1.0, 1.0, 1.0), bevel_width=0.0, location=(0.5, 0.0, 0.0))
    bm = bmesh.new()
    bm.from_mesh(half.data)
    bm.faces.ensure_lookup_table()
    cap = [f for f in bm.faces if all(abs(v.co.x) < 1e-6 for v in f.verts)]
    bmesh.ops.delete(bm, geom=cap, context="FACES_ONLY")
    bm.to_mesh(half.data)
    bm.free()
    half.data.update()
    hs.mirror(half, axis="X")
    checks["mirror"] = _mesh_report(half)

    plane_mesh = bpy.data.meshes.new("plane_mesh")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    bm.to_mesh(plane_mesh)
    bm.free()
    plane = bpy.data.objects.new("plane", plane_mesh)
    bpy.context.collection.objects.link(plane)
    hs.solidify(plane, thickness=0.05)
    checks["solidify"] = _mesh_report(plane)

    greeble_box = hs.bevelled_box("greeble_box", size=(2.0, 2.0, 2.0), bevel_width=0.0)
    hs.greeble(greeble_box, count=6, depth_range=(0.05, 0.1), seed=1)
    checks["greeble"] = _mesh_report(greeble_box)

    roof_mesh = bpy.data.meshes.new("roof_mesh")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=3, y_segments=3, size=1.0)
    bm.to_mesh(roof_mesh)
    bm.free()
    roof = bpy.data.objects.new("roof", roof_mesh)
    bpy.context.collection.objects.link(roof)
    hs.roof_edge(roof, overhang=0.3, thickness=0.03)
    checks["roof_edge"] = _mesh_report(roof)

    gable = hs.gable_roof("gable", span=2.0, length=3.0, ridge_height=0.8, overhang=0.3, thickness=0.03, location=(8.0, 0.0, 0.0))
    checks["gable_roof"] = _mesh_report(gable)

    strip = hs.cornice("cornice_strip", 2.0, (0.08, 0.06), location=(0.0, 0.0, 3.0))
    checks["cornice"] = _mesh_report(strip)

    import mathutils

    hinge = hs.bevelled_box("hinge", size=(0.1, 0.02, 0.15), bevel_width=0.0)
    surface_normal = mathutils.Vector((0.0, -1.0, 0.0))
    hs.attach_to_surface(hinge, point=(1.0, 0.0, 1.0), normal=surface_normal)
    checks["attach_to_surface"] = _mesh_report(hinge)
    surface_point = mathutils.Vector((1.0, 0.0, 1.0))
    hinge_verts = [hinge.matrix_world @ v.co for v in hinge.data.vertices]
    base_verts = sorted(hinge_verts, key=lambda v: (v - surface_point).dot(surface_normal))[:4]
    base_centre = sum(base_verts, mathutils.Vector()) / len(base_verts)
    checks["attach_to_surface"]["clears_standoff"] = (
        abs((base_centre - surface_point).length - hs.STANDOFF_M) < 1e-6
    )

    wall_win = hs.bevelled_box("wall_win", size=(3.0, 0.3, 3.0), bevel_width=0.0)
    hs.window_recess(wall_win, centre=(0.0, -0.15, 1.5), size=(0.8, 1.0), depth=0.05, wall_thickness=0.3)
    checks["window_recess"] = _mesh_report(wall_win)

    wall_door = hs.bevelled_box("wall_door", size=(3.0, 0.3, 3.0), bevel_width=0.0)
    door_centre = (0.0, -0.15, 0.0)
    door_size = (0.9, 2.1)
    door = hs.door_frame(wall_door, centre=door_centre, size=door_size, wall_thickness=0.3)
    checks["door_frame_wall"] = _mesh_report(wall_door)
    checks["door_frame_frame"] = _mesh_report(door)
    opening_bottom = door_centre[2] - door_size[1] * 0.5
    opening_top = door_centre[2] + door_size[1] * 0.5
    trim_corners = [door.matrix_world @ mathutils.Vector(c) for c in door.bound_box]
    checks["door_frame_frame"]["brackets_opening"] = (
        min(c.z for c in trim_corners) <= opening_bottom + 1e-6
        and max(c.z for c in trim_corners) >= opening_top - 1e-6
    )

    frame_parts = hs.ladder_frame("ladder", length=3.0, width=0.6, rung_count=5, location=(4.0, 0.0, 0.0))
    for part in frame_parts:
        checks[part.name] = _mesh_report(part)

    frustum_obj = hs.frustum("frustum", r0=0.5, r1=0.2, length=2.0, location=(-3.0, 0.0, 0.0))
    checks["frustum"] = _mesh_report(frustum_obj)

    pyramid_obj = hs.frustum("pyramid", r0=0.5, r1=0.0, length=1.0, segments=4, location=(-3.0, 3.0, 0.0))
    checks["frustum_apex"] = _mesh_report(pyramid_obj)

    point_a = (1.0, 2.0, 0.5)
    point_b = (1.0, 2.0 - 3.0, 0.5)
    shaft = hs.cylinder_between("shaft", point_a, point_b, r0=0.15)
    checks["cylinder_between"] = _mesh_report(shaft)
    shaft_verts = [shaft.matrix_world @ v.co for v in shaft.data.vertices]
    far_end = min(shaft_verts, key=lambda v: (v - mathutils.Vector(point_b)).length)
    checks["cylinder_between"]["reaches_target"] = (far_end - mathutils.Vector(point_b)).length < 0.2

    stair_treads = hs.stairs("stairs", tread_count=3, total_rise=0.6, total_run=0.9, width=1.0, location=(5.0, 0.0, 0.0))
    heights = []
    for tread in stair_treads:
        checks[tread.name] = _mesh_report(tread)
        corners = [tread.matrix_world @ mathutils.Vector(c) for c in tread.bound_box]
        heights.append((min(c.y for c in corners), max(c.z for c in corners)))
    heights.sort(key=lambda pair: -pair[0])
    checks["stairs_monotonic"] = {
        "non_manifold_edges": 0,
        "loose_verts": 0,
        "degenerate_faces": 0,
        "ok": all(heights[i][1] > heights[i + 1][1] for i in range(len(heights) - 1)),
    }

    return checks


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from assets import hardsurface as hs

        checks = _run_checks(hs)
        ok = all(
            report["non_manifold_edges"] == 0 and report["loose_verts"] == 0 and report["degenerate_faces"] == 0
            for report in checks.values()
        ) and checks["cylinder_between"]["reaches_target"] and checks["stairs_monotonic"]["ok"] and checks["door_frame_frame"]["brackets_opening"] and checks["attach_to_surface"]["clears_standoff"]
        payload = {"ok": ok, "checks": checks}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
