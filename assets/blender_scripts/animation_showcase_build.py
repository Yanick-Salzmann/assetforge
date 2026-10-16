"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds a test scene that plays every assets/animation.py primitive on a simple mechanism - gear
train, crank-slider, cam trip hammer, pendulum, hinged door, oscillating plunger, spinning
millstone - keyframed one sample per frame, and saves it as a .blend to open and press play.
Invoked as:
    blender --background --factory-startup --python animation_showcase_build.py -- <repo_root> <out.blend> [<render_dir>]
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

FPS = 24
DURATION_S = 4.0
GEAR_TEETH = [12, 24, 16]
GEAR_MODULE_M = 0.06
DEPTH_M = 0.1
BAY_SPACING_M = 2.6


def _read_args() -> tuple[Path, Path, Path | None]:
    argv = sys.argv[sys.argv.index("--") + 1:]
    render_dir = Path(argv[2]) if len(argv) > 2 else None
    return Path(argv[0]), Path(argv[1]), render_dir


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _material(name: str, colour: tuple[float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*colour, 1.0)
    material.diffuse_color = (*colour, 1.0)
    return material


def _add_box(bm: bmesh.types.BMesh, size: tuple[float, float, float], centre: tuple[float, float, float], rotation_y: float = 0.0) -> None:
    matrix = Matrix.Rotation(rotation_y, 4, "Y") @ Matrix.Translation(Vector(centre)) @ Matrix.Diagonal((*size, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=matrix)


def _add_disc(bm: bmesh.types.BMesh, radius: float, depth: float, axis: str, segments: int = 32) -> None:
    rotation = Matrix.Rotation(math.pi / 2, 4, "X") if axis == "Y" else Matrix.Identity(4)
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth, matrix=rotation)


def _object(name: str, bm: bmesh.types.BMesh, material, location: tuple[float, float, float], parent=None) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    obj.location = location
    obj.rotation_mode = "XYZ"
    obj.parent = parent
    bpy.context.collection.objects.link(obj)
    return obj


def _label(text: str, location: tuple[float, float, float]) -> None:
    curve = bpy.data.curves.new(f"label_{text}", "FONT")
    curve.body = text
    curve.size = 0.22
    curve.align_x = "CENTER"
    label = bpy.data.objects.new(f"label_{text}", curve)
    label.location = location
    label.rotation_euler = (math.pi / 2, 0.0, 0.0)
    bpy.context.collection.objects.link(label)


def _keyframe(obj: bpy.types.Object, data_path: str, index: int, values) -> None:
    for frame, value in enumerate(values, start=1):
        getattr(obj, data_path)[index] = value
        obj.keyframe_insert(data_path, index=index, frame=frame)
    action = obj.animation_data.action
    curves = action.fcurves if hasattr(action, "fcurves") else [
        curve
        for layer in action.layers
        for strip in layer.strips
        for channelbag in strip.channelbags
        for curve in channelbag.fcurves
    ]
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def _gear(name: str, teeth: int, material, location, tooth_phase: float) -> bpy.types.Object:
    radius = GEAR_MODULE_M * teeth / 2
    bm = bmesh.new()
    _add_disc(bm, radius - GEAR_MODULE_M * 0.6, DEPTH_M, "Y", segments=max(24, teeth * 2))
    tooth = GEAR_MODULE_M * 1.6
    for index in range(teeth):
        angle = tooth_phase + math.tau * index / teeth
        centre = (radius * math.cos(angle), 0.0, radius * math.sin(angle))
        matrix = Matrix.Translation(Vector(centre)) @ Matrix.Rotation(-angle, 4, "Y") @ Matrix.Diagonal((tooth, DEPTH_M, GEAR_MODULE_M * 1.2, 1.0))
        bmesh.ops.create_cube(bm, size=1.0, matrix=matrix)
    _add_box(bm, (radius * 1.2, DEPTH_M * 1.2, GEAR_MODULE_M), (0.0, 0.0, 0.0))
    return _object(name, bm, material, location)


def _gear_train(anim, origin_x: float, materials) -> None:
    tracks = anim.gear_train(GEAR_TEETH, 1.0, DURATION_S, FPS)
    radii = [GEAR_MODULE_M * teeth / 2 for teeth in GEAR_TEETH]
    x = origin_x - sum(radii) + radii[0]
    for index, (teeth, track) in enumerate(zip(GEAR_TEETH, tracks)):
        phase = 0.0 if index % 2 == 0 else math.pi / teeth
        gear = _gear(f"Gear{index}_ANIM", teeth, materials[index % len(materials)], (x, 0.0, 1.0), phase)
        _keyframe(gear, "rotation_euler", 1, [-value for value in track.values])
        if index + 1 < len(radii):
            x += radii[index] + radii[index + 1]
    _label("gear_train 12:24:16", (origin_x, 0.0, 2.0))


def _crank_slider(anim, origin_x: float, materials) -> None:
    radius = 0.3
    rod_length = 0.9
    motion = anim.crank_slider(radius, rod_length, 1.0, DURATION_S, FPS)
    pivot = (origin_x - 0.6, 0.0, 1.0)
    bm = bmesh.new()
    _add_disc(bm, radius * 1.15, DEPTH_M * 0.5, "Y")
    _add_box(bm, (radius, DEPTH_M, 0.08), (radius / 2, -DEPTH_M * 0.5, 0.0))
    crank = _object("Crank_ANIM", bm, materials[0], pivot)
    _keyframe(crank, "rotation_euler", 1, [-value for value in motion.crank.values])
    bm = bmesh.new()
    _add_box(bm, (rod_length, DEPTH_M * 0.6, 0.05), (rod_length / 2, -DEPTH_M, 0.0))
    rod = _object("Rod_ANIM", bm, materials[1], pivot)
    pins = [(pivot[0] + radius * math.cos(theta), pivot[2] + radius * math.sin(theta)) for theta in motion.crank.values]
    _keyframe(rod, "location", 0, [pin[0] for pin in pins])
    _keyframe(rod, "location", 2, [pin[1] for pin in pins])
    _keyframe(rod, "rotation_euler", 1, [-value for value in motion.rod.values])
    bm = bmesh.new()
    _add_box(bm, (0.25, 0.25, 0.25), (0.0, 0.0, 0.0))
    piston = _object("Piston_ANIM", bm, materials[2], pivot)
    _keyframe(piston, "location", 0, [pivot[0] + value for value in motion.slider.values])
    _label("crank_slider", (origin_x, 0.0, 2.0))


def _trip_hammer(anim, origin_x: float, materials) -> None:
    lift = anim.cam_follower(0.35, 2.0, DURATION_S, FPS, lobes=2, rise_fraction=0.8)
    cam_spin = anim.spin(2.0, DURATION_S, FPS)
    pivot = (origin_x - 0.7, 0.0, 0.9)
    bm = bmesh.new()
    _add_box(bm, (1.3, DEPTH_M, 0.08), (0.65, 0.0, 0.0))
    _add_box(bm, (0.22, 0.22, 0.3), (1.3, 0.0, -0.1))
    hammer = _object("Hammer_ANIM", bm, materials[0], pivot)
    _keyframe(hammer, "rotation_euler", 1, [-value for value in lift.values])
    bm = bmesh.new()
    _add_disc(bm, 0.15, DEPTH_M, "Y")
    _add_box(bm, (0.08, DEPTH_M, 0.42), (0.0, 0.0, 0.0))
    cam = _object("Cam_ANIM", bm, materials[1], (origin_x - 0.15, DEPTH_M * 1.5, 0.75))
    _keyframe(cam, "rotation_euler", 1, [-value for value in cam_spin.values])
    bm = bmesh.new()
    _add_box(bm, (0.4, 0.4, 0.5), (0.0, 0.0, 0.0))
    _object("Anvil", bm, materials[2], (origin_x + 0.6, 0.0, 0.45))
    _label("cam_follower 2 lobes", (origin_x, 0.0, 2.0))


def _pendulum(anim, origin_x: float, materials) -> None:
    length = 1.0
    swing = anim.pendulum(0.5, round(anim.pendulum_period(length) * 2) / 2, DURATION_S, FPS)
    bm = bmesh.new()
    _add_box(bm, (0.04, 0.04, length), (0.0, 0.0, -length / 2))
    bob = Matrix.Translation(Vector((0.0, 0.0, -length)))
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.12, matrix=bob)
    pendulum = _object("Pendulum_ANIM", bm, materials[0], (origin_x, 0.0, 1.8))
    _keyframe(pendulum, "rotation_euler", 1, swing.values)
    _label("pendulum", (origin_x, 0.0, 2.0))


def _hinge_door(anim, origin_x: float, materials) -> None:
    swing = anim.hinge_swing(math.radians(100), 4.0, DURATION_S, FPS, hold_fraction=0.15)
    bm = bmesh.new()
    _add_box(bm, (0.9, 0.06, 1.6), (0.45, 0.0, 0.8))
    door = _object("Door_ANIM", bm, materials[0], (origin_x - 0.45, 0.0, 0.0))
    _keyframe(door, "rotation_euler", 2, swing.values)
    _label("hinge_swing", (origin_x, 0.0, 2.0))


def _plunger_and_millstone(anim, origin_x: float, materials) -> None:
    bob = anim.oscillate(0.25, 1.0, DURATION_S, FPS, offset=0.9)
    bm = bmesh.new()
    _add_box(bm, (0.3, 0.3, 0.6), (0.0, 0.0, 0.0))
    plunger = _object("Plunger_ANIM", bm, materials[0], (origin_x - 0.5, 0.0, 0.9))
    _keyframe(plunger, "location", 2, bob.values)
    turn = anim.spin(4.0, DURATION_S, FPS, clockwise=True)
    bm = bmesh.new()
    _add_disc(bm, 0.45, 0.2, "Z")
    _add_box(bm, (0.1, 0.5, 0.1), (0.0, 0.25, 0.15))
    stone = _object("Millstone_ANIM", bm, materials[1], (origin_x + 0.5, 0.0, 0.1))
    _keyframe(stone, "rotation_euler", 2, turn.values)
    _label("oscillate + spin", (origin_x, 0.0, 2.0))


def _stage(count: int) -> None:
    width = BAY_SPACING_M * count
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(width / 2 - BAY_SPACING_M / 2, 1.0, 0.0))
    floor = bpy.context.active_object
    floor.scale = (width + 2.0, 6.0, 1.0)
    floor.data.materials.append(_material("floor", (0.35, 0.35, 0.33)))
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (width / 2 - BAY_SPACING_M / 2, -15.0, 3.5)
    camera.rotation_euler = (math.radians(81), 0.0, 0.0)
    camera_data.lens = 28
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    bpy.context.collection.objects.link(sun)


def _render_frames(render_dir: Path, frames: list[int]) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 560
    render_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(render_dir / f"frame_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)


def main() -> None:
    repo_root, blend_path, render_dir = _read_args()
    sys.path.insert(0, str(repo_root))
    from assets import animation as anim

    _clear_scene()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = anim.frame_count(DURATION_S, FPS)
    palette = [
        _material("brass", (0.8, 0.6, 0.2)),
        _material("iron", (0.3, 0.32, 0.38)),
        _material("wood", (0.55, 0.35, 0.2)),
        _material("red", (0.7, 0.15, 0.12)),
    ]
    builders = [_gear_train, _crank_slider, _trip_hammer, _pendulum, _hinge_door, _plunger_and_millstone]
    for index, build in enumerate(builders):
        build(anim, index * BAY_SPACING_M, palette[index % 2:] + palette[:index % 2])
    _stage(len(builders))
    if render_dir is not None:
        _render_frames(render_dir, [1, 13, 25, 37])
    scene.frame_set(1)
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


main()
