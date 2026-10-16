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

    import math

    brace_a = mathutils.Vector((10.0, 0.0, 0.0))
    brace_b = mathutils.Vector((11.0, 1.0, 2.0))
    brace = hs.box_between("brace", brace_a, brace_b, width=0.1, height=0.05, bevel_width=0.005)
    checks["box_between"] = _mesh_report(brace)
    brace_verts = [brace.matrix_world @ v.co for v in brace.data.vertices]
    checks["box_between"]["spans_endpoints"] = (
        min((v - brace_a).length for v in brace_verts) < 0.1 and min((v - brace_b).length for v in brace_verts) < 0.1
    )
    flat = hs.box_between("flat_brace", (12.0, 0.0, 0.0), (12.0, 0.0, 2.0), width=0.3, height=0.02, up=(1.0, 0.0, 0.0))
    checks["box_between_up"] = _mesh_report(flat)
    flat_corners = [flat.matrix_world @ mathutils.Vector(c) for c in flat.bound_box]
    checks["box_between_up"]["width_follows_up"] = (
        max(c.x for c in flat_corners) - min(c.x for c in flat_corners) > 0.29
    )

    barrel = hs.lathe("barrel", [(0.25, 0.0), (0.3, 0.45), (0.25, 0.9)], segments=12, location=(14.0, 0.0, 0.0))
    checks["lathe"] = _mesh_report(barrel)
    finial = hs.lathe("finial", [(0.0, 0.0), (0.1, 0.1), (0.05, 0.3), (0.0, 0.4)], segments=8, location=(15.0, 0.0, 0.0))
    checks["lathe_apex"] = _mesh_report(finial)

    hood = hs.loft(
        "hood",
        [
            [(-0.4, y, 1.0), (0.4, y, 1.0), (0.4 - 0.1 * k, y, 1.6), (-0.4 + 0.1 * k, y, 1.6)]
            for k, y in enumerate((0.0, 1.0, 2.0))
        ],
    )
    checks["loft"] = _mesh_report(hood)

    rail = hs.prism("rail", [(0.0, 0.0), (0.3, 0.0), (0.3, 0.1), (0.1, 0.25), (0.0, 0.25)], -1.0, 1.0, axis="X", bevel_width=0.01)
    checks["prism"] = _mesh_report(rail)

    fender = hs.annular_band("fender", (0.0, 0.9), 0.85, 0.9, 1.0, 1.4, angle_start=0.0, angle_end=math.pi, axis="X", segments=12)
    checks["annular_band"] = _mesh_report(fender)
    sleeve = hs.annular_band("sleeve", (20.0, 0.0), 0.1, 0.12, 0.0, 0.3, axis="Z", segments=12)
    checks["annular_band_full"] = _mesh_report(sleeve)

    ring = hs.torus("ring", (22.0, 0.0, 0.5), 0.05, 0.018, axis="X")
    checks["torus"] = _mesh_report(ring)

    tyre_profile = [(0.53, -0.22), (0.62, -0.27), (0.76, -0.275), (0.82, -0.25), (0.84, -0.2), (0.84, 0.0), (0.84, 0.2), (0.82, 0.25), (0.76, 0.275), (0.62, 0.27), (0.53, 0.22)]
    tyre = hs.lugged_tyre("tyre", (25.0, 0.0, 0.9), tyre_profile, segments=24, lug_height=0.05, lug_skew=0.9)
    checks["lugged_tyre"] = _mesh_report(tyre)

    slope = hs.roof_slope("slope", (-2.0, 0.0, 3.0), (-2.0, 4.0, 3.0), (0.0, 4.0, 4.0), (0.0, 0.0, 4.0), thickness=0.1)
    checks["roof_slope"] = _mesh_report(slope)
    slope_corners = [slope.matrix_world @ mathutils.Vector(c) for c in slope.bound_box]
    checks["roof_slope"]["top_on_plane"] = abs(max(c.z for c in slope_corners) - 4.0) < 1e-4

    gable_wall = hs.triangular_gable("gable_wall", span=4.0, base_z=3.0, apex_z=4.5, thickness=0.3, y=6.0)
    checks["triangular_gable"] = _mesh_report(gable_wall)

    lumpy = hs.bevelled_box("lumpy", size=(2.0, 0.5, 1.5), bevel_width=0.05, location=(30.0, 0.0, 0.0))
    before = [v.co.copy() for v in lumpy.data.vertices]
    hs.wobble(lumpy, 0.02, seed=7, wavelength=0.8, keep_below_z=0.01)
    hs.smooth_by_angle(lumpy)
    checks["wobble"] = _mesh_report(lumpy)
    moved = [(v.co - b).length for v, b in zip(lumpy.data.vertices, before)]
    checks["wobble"]["bounded"] = max(moved) <= 0.02 * 3 ** 0.5 + 1e-6 and max(moved) > 0.0
    checks["wobble"]["footing_flush"] = all(abs(v.co.z - b.z) < 1e-9 for v, b in zip(lumpy.data.vertices, before) if b.z < 0.01)

    return checks


def _world_bbox(obj) -> list[float]:
    import mathutils

    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    return [min(c[i] for c in corners) for i in range(3)] + [max(c[i] for c in corners) for i in range(3)]


def _animated_fixture(hs, out_dir: Path) -> dict:
    """A crank-slider and a gear pair on a base plate, pivoted and animated through assets.motion
    and exported through assets.export: the clips must reach the glb, the rod's far end must ride
    the piston's wrist pin, and re-running a helper must replace its clip rather than stack one."""
    import mathutils

    from assets import export, motion
    from terrain import config

    config.ASSET_OUT_DIR = out_dir
    scene = bpy.context.scene
    scene.render.fps = 24
    material = bpy.data.materials.new("motion_atlas")
    material.use_nodes = True
    radius = 0.3
    rod_length = 0.9
    centre = (49.0, -0.15, 1.0)

    plate = hs.bevelled_box("motion_plate", size=(3.0, 1.0, 0.2), bevel_width=0.0, location=(50.0, 0.0, 0.0))
    crank = hs.box_between("crank", centre, (centre[0] + radius, centre[1], centre[2]), 0.08, 0.08)
    crank_before = _world_bbox(crank)
    motion.pivot_part(crank, centre, axis="Y")
    pin = (centre[0] + radius, -0.25, centre[2])
    rod = hs.box_between("rod", pin, (pin[0] + rod_length, pin[1], pin[2]), 0.06, 0.06)
    motion.pivot_part(rod, pin, axis="Y")
    wrist = (centre[0] + radius + rod_length, -0.35, centre[2])
    piston = hs.bevelled_box("piston", size=(0.25, 0.25, 0.25), bevel_width=0.0, location=(wrist[0], wrist[1], wrist[2] - 0.125))
    motion.pivot_part(piston, wrist, axis="Y")
    gears = []
    for index, (x, gear_radius) in enumerate(((51.2, 0.2), (51.7, 0.3))):
        gear = hs.cylinder_between(f"gear{index}", (x, -0.05, 1.0), (x, 0.05, 1.0), gear_radius, segments=12)
        gears.append(motion.pivot_part(gear, (x, 0.0, 1.0), axis="Y"))
    parts = [plate, crank, rod, piston, *gears]
    for part in parts:
        part.data.materials.append(material)

    motion.animate_crank(crank, rod, piston, radius, rod_length, 1.0)
    actions_before = len(bpy.data.actions)
    crank_entry = motion.animate_crank(crank, rod, piston, radius, rod_length, 1.0)
    gear_entry = motion.animate_gear_train(gears, [12, 18], 1.0)

    rod_tip_errors = []
    for frame in (1, 4, 7, 13, 19):
        scene.frame_set(frame)
        tip = rod.matrix_world @ mathutils.Vector((rod_length, 0.0, 0.0))
        offset = tip - piston.matrix_world.translation
        rod_tip_errors.append(max(abs(offset.x), abs(offset.z)))
    scene.frame_set(1)

    mechanisms = [motion.crank_linkage(crank, rod, piston, crank_entry), motion.gear_mesh(gears, [12, 18], gear_entry)]
    result = export.export_asset(parts, name="hardsurface-motion", kind="prop", animations=[crank_entry, gear_entry], mechanisms=mechanisms)
    shipped = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    document = export.glb_document(result.glb_path)
    names = [node.get("name") for node in document.get("nodes", [])]
    clips = {
        animation["name"]: sorted({names[channel["target"]["node"]] for channel in animation["channels"]})
        for animation in document.get("animations", [])
    }
    checks = {
        "pivot_keeps_geometry": all(abs(a - b) < 1e-5 for a, b in zip(crank_before, _world_bbox(crank))),
        "anim_names": [part.name for part in parts[1:]] == ["crank_ANIM", "rod_ANIM", "piston_ANIM", "gear0_ANIM", "gear1_ANIM"],
        "clip_replaced": len(bpy.data.actions) == actions_before + 1 and len(crank.animation_data.nla_tracks) == 1,
        "rod_rides_wrist_pin": max(rod_tip_errors) < 1e-4,
        "gear_loop_s": gear_entry["duration_s"] == 3.0,
        "mechanisms_shipped": shipped.get("mechanisms") == mechanisms,
        "glb_clips": clips == {"working_crank": ["crank_ANIM", "piston_ANIM", "rod_ANIM"], "working_gears": ["gear0_ANIM", "gear1_ANIM"]},
    }
    return {"checks": checks, "rod_tip_errors": rod_tip_errors, "clips": clips, "ok": all(checks.values())}


def _wind_flag_fixture(hs, out_dir: Path) -> dict:
    """A flag on a pole, skinned to a 6-bone chain running out from the pole and waved by
    assets.motion.animate_wind: the free edge must swing sideways while the pole edge stays put,
    and the clip must export as one skinned clip over the chain."""
    from assets import export, motion
    from terrain import config

    config.ASSET_OUT_DIR = out_dir
    scene = bpy.context.scene
    scene.render.fps = 24
    material = bpy.data.materials.new("flag_atlas")
    material.use_nodes = True
    bones = [f"flag{index}" for index in range(6)]
    length = 1.2
    height = 0.8
    pole_x = 60.0
    top = 2.4

    pole = hs.cylinder_between("flag_pole", (pole_x, 0.0, 0.0), (pole_x, 0.0, top + 0.1), 0.03, segments=8)
    cloth_mesh = bpy.data.meshes.new("flag_cloth_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vert in bm.verts:
        vert.co.x = (vert.co.x + 0.5) * length + 0.035
        vert.co.y *= 0.01
        vert.co.z = (vert.co.z - 0.5) * height
    edges = [edge for edge in bm.edges if abs(edge.verts[0].co.x - edge.verts[1].co.x) > 1e-6]
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=23, use_grid_fill=True)
    bm.to_mesh(cloth_mesh)
    bm.free()
    cloth = bpy.data.objects.new("flag_cloth", cloth_mesh)
    bpy.context.collection.objects.link(cloth)

    rig = bpy.data.objects.new("flag_rig", bpy.data.armatures.new("flag_rig_data"))
    rig.location = (pole_x, 0.0, top)
    bpy.context.collection.objects.link(rig)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    segment = length / len(bones)
    parent = None
    for index, bone_name in enumerate(bones):
        bone = rig.data.edit_bones.new(bone_name)
        bone.head = (0.035 + index * segment, 0.0, 0.0)
        bone.tail = (0.035 + (index + 1) * segment, 0.0, 0.0)
        bone.parent = parent
        bone.use_connect = parent is not None
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    for index, bone_name in enumerate(bones):
        group = cloth.vertex_groups.new(name=bone_name)
        centre = 0.035 + (index + 0.5) * segment
        for vert in cloth.data.vertices:
            weight = max(0.0, 1.0 - abs(vert.co.x - centre) / segment)
            if weight > 0.0:
                group.add([vert.index], weight, "REPLACE")
    cloth.parent = rig
    cloth.modifiers.new("Armature", "ARMATURE").object = rig
    for part in (pole, cloth):
        part.data.materials.append(material)

    entry = motion.animate_wind(rig, bones, 2.5, seed=5, strength=1.0)
    root_drift = 0.0
    tip_sideways = 0.0
    tip_vertical = 0.0
    for frame in range(1, 62, 4):
        scene.frame_set(frame)
        root = rig.matrix_world @ rig.pose.bones[bones[0]].head
        tip = rig.matrix_world @ rig.pose.bones[bones[-1]].tail
        root_drift = max(root_drift, (root - rig.matrix_world @ rig.data.bones[bones[0]].head_local).length)
        tip_sideways = max(tip_sideways, abs(tip.y))
        tip_vertical = max(tip_vertical, abs(tip.z - top))
    scene.frame_set(1)

    result = export.export_asset([pole, rig, cloth], name="hardsurface-flag", kind="prop", animations=[entry])
    document = export.glb_document(result.glb_path)
    checks = {
        "entry": entry["kind"] == "skinned" and entry["state"] == "idle" and entry["nodes"] == bones,
        "root_pinned": root_drift < 1e-5,
        "tip_swings_sideways": tip_sideways > 0.1,
        "tip_stays_level": tip_vertical < tip_sideways,
        "glb_clip": [animation["name"] for animation in document.get("animations", [])] == ["idle_wind"],
    }
    return {
        "checks": checks,
        "tip_sideways": tip_sideways,
        "tip_vertical": tip_vertical,
        "ok": all(checks.values()),
    }


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
        ) and checks["cylinder_between"]["reaches_target"] and checks["stairs_monotonic"]["ok"] and checks["door_frame_frame"]["brackets_opening"] and checks["attach_to_surface"]["clears_standoff"] and checks["box_between"]["spans_endpoints"] and checks["box_between_up"]["width_follows_up"] and checks["roof_slope"]["top_on_plane"] and checks["wobble"]["bounded"] and checks["wobble"]["footing_flush"]
        motion_check = _animated_fixture(hs, result_path.parent / "assets_out")
        wind_check = _wind_flag_fixture(hs, result_path.parent / "assets_out")
        ok = ok and motion_check["ok"] and wind_check["ok"]
        payload = {"ok": ok, "checks": checks, "motion_check": motion_check, "wind_check": wind_check}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
