"""Runs inside Blender's own interpreter via `blender --background --python`.

Not part of the assetforge package import graph - Blender's bundled Python has no access to
our venv (torch/numpy/pillow), so this script only ever touches bpy, mathutils and the stdlib.
Invoked as: blender --background --factory-startup --python contact_sheet_render.py -- <args.json>
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

VIEW_DIRECTIONS = {
    "front": Vector((0.0, 1.0, 0.0)),
    "back": Vector((0.0, -1.0, 0.0)),
    "right": Vector((1.0, 0.0, 0.0)),
    "left": Vector((-1.0, 0.0, 0.0)),
    "top": Vector((0.0, 0.0, 1.0)),
    "bottom": Vector((0.0, 0.0, -1.0)),
}


def _read_args() -> dict:
    argv = sys.argv
    separator = argv.index("--")
    return json.loads(Path(argv[separator + 1]).read_text(encoding="utf-8"))


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _set_cycles_device(requested: str) -> dict:
    """Force Cycles onto the named compute device explicitly - never inherit user prefs.

    Left to user preferences, a headless run silently falls back to CPU and a render that
    should take seconds on the RTX 4070 instead takes minutes.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = requested
    prefs.get_devices()
    enabled = []
    for device in prefs.devices:
        device.use = device.type == requested
        if device.use:
            enabled.append(device.name)
    if enabled:
        scene.cycles.device = "GPU"
        return {"device": requested, "gpus": enabled}
    scene.cycles.device = "CPU"
    print(f"WARNING: no {requested} device found, falling back to CPU explicitly", file=sys.stderr)
    return {"device": "CPU", "gpus": []}


def _build_world() -> None:
    """Flat, even ambient light - no single dominant shadow direction to bias any one view."""
    world = bpy.data.worlds.new("Neutral")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.6, 0.62, 0.66, 1.0)
    background.inputs["Strength"].default_value = 1.2
    bpy.context.scene.world = world


def _import_asset(glb_path: str) -> list:
    bpy.ops.import_scene.gltf(filepath=glb_path)
    bpy.context.view_layer.update()
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def _world_bbox(objects: list) -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            minimum.x, minimum.y, minimum.z = min(minimum.x, point.x), min(minimum.y, point.y), min(minimum.z, point.z)
            maximum.x, maximum.y, maximum.z = max(maximum.x, point.x), max(maximum.y, point.y), max(maximum.z, point.z)
    return minimum, maximum


def _scale_reference(bbox_min: Vector, bbox_max: Vector, height_m: float, clearance: float) -> list:
    """A plain 1.8 m humanoid capsule standing beside the asset, per the CLAUDE.md scale check."""
    radius = min(0.22, height_m * 0.12)
    body_height = max(height_m - 2.0 * radius, 0.1)
    x = bbox_max.x + clearance + radius
    y = (bbox_min.y + bbox_max.y) / 2.0
    z = bbox_min.z

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=body_height, location=(x, y, z + body_height / 2.0))
    body = bpy.context.active_object
    body.name = "ScaleReferenceBody"

    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(x, y, z + body_height + radius))
    head = bpy.context.active_object
    head.name = "ScaleReferenceHead"

    material = bpy.data.materials.new("ScaleReferenceMaterial")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.45, 0.6, 1.0)
    body.data.materials.append(material)
    head.data.materials.append(material)
    bpy.context.view_layer.update()
    return [body, head]


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    if direction.length < 1e-9:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _clear_lights() -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def _add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> None:
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = energy
    light_data.size = max(size, 0.05)
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)


def _build_light_rig(camera_location: Vector, target: Vector, radius: float) -> None:
    """A soft key/fill pair carried with the camera, so every view is lit the same way."""
    _clear_lights()
    forward = target - camera_location
    if forward.length < 1e-6:
        forward = Vector((0.0, -1.0, 0.0))
    forward = forward.normalized()
    up_hint = Vector((0.0, 0.0, 1.0))
    right = forward.cross(up_hint)
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    right = right.normalized()
    up = right.cross(forward).normalized()
    spread = max(radius, 0.5)
    key_pos = camera_location + right * spread * 0.7 + up * spread * 0.9
    fill_pos = camera_location - right * spread * 1.0 + up * spread * 0.3
    energy = max(spread, 1.0) ** 2 * 250.0
    _add_area_light("KeyLight", key_pos, target, energy=energy, size=spread * 1.2)
    _add_area_light("FillLight", fill_pos, target, energy=energy * 0.4, size=spread * 1.8)


def _render(scene, resolution: list, samples: int, filepath: str) -> None:
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.cycles.samples = samples
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = _read_args()
    _clear_scene()

    device_info = _set_cycles_device(args.get("device", "OPTIX"))
    _build_world()

    asset_objects = _import_asset(args["glb_path"])
    if not asset_objects:
        raise RuntimeError(f"no mesh objects found in {args['glb_path']}")
    bbox_min, bbox_max = _world_bbox(asset_objects)

    human_height_m = float(args.get("human_height_m", 1.8))
    margin = float(args.get("margin", 1.15))
    clearance = max(0.3, (bbox_max.x - bbox_min.x) * 0.15)
    reference_objects = _scale_reference(bbox_min, bbox_max, human_height_m, clearance)

    combined_min, combined_max = _world_bbox(asset_objects + reference_objects)
    centre = (combined_min + combined_max) / 2.0
    extent = combined_max - combined_min
    diag = max(extent.x, extent.y, extent.z, 0.1)
    radius = max(extent.length / 2.0, 0.1)

    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("ContactCam")
    camera = bpy.data.objects.new("ContactCam", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.clip_start = 0.01
    camera_data.clip_end = diag * 6.0 + 20.0

    resolution = args["resolution"]
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    samples = int(args.get("samples", 64))
    outputs = args["outputs"]
    ortho_distance = diag * 2.0 + 5.0

    for view in args["views"]:
        if view in VIEW_DIRECTIONS:
            direction = VIEW_DIRECTIONS[view]
            camera_data.type = "ORTHO"
            if view in ("front", "back"):
                camera_data.ortho_scale = max(extent.x, extent.z) * margin
            elif view in ("left", "right"):
                camera_data.ortho_scale = max(extent.y, extent.z) * margin
            else:
                camera_data.ortho_scale = max(extent.x, extent.y) * margin
            camera.location = centre + direction * ortho_distance
            _look_at(camera, centre)
        elif view == "three_quarter":
            camera_data.type = "PERSP"
            camera_data.lens = 35.0
            distance = (radius / math.tan(camera_data.angle / 2.0)) * margin * 1.4
            azimuth = math.radians(45.0)
            altitude = math.radians(35.0)
            offset = Vector(
                (math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude))
            ) * distance
            camera.location = centre + offset
            _look_at(camera, centre)
        elif view == "worm_eye":
            camera_data.type = "PERSP"
            camera_data.lens = 24.0
            distance = (radius / math.tan(camera_data.angle / 2.0)) * margin * 1.6
            azimuth = math.radians(20.0)
            altitude = math.radians(-25.0)
            offset = Vector(
                (math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude))
            ) * distance
            camera.location = centre + offset
            camera.location.z = max(camera.location.z, combined_min.z + 0.05)
            _look_at(camera, centre)
        else:
            raise RuntimeError(f"unknown view {view!r}")

        _build_light_rig(camera.location, centre, radius)
        _render(scene, resolution, samples, outputs[view])

    result = {
        "device": device_info,
        "bbox_min": [bbox_min.x, bbox_min.y, bbox_min.z],
        "bbox_max": [bbox_max.x, bbox_max.y, bbox_max.z],
    }
    Path(args["result_json"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
