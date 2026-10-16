"""Runs inside Blender's own interpreter via `blender --background --python`.

Imports an exported animated glb, plays one clip and renders it from a single 3/4 camera at
evenly spaced frames, framed on the union of every sampled pose so moving parts never leave the
shot. Reuses the contact sheet's device, world and light rig. Invoked as:
    blender --background --factory-startup --python animation_sheet_render.py -- <args.json>
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contact_sheet_render as sheet


def _read_args() -> dict:
    argv = sys.argv
    return json.loads(Path(argv[argv.index("--") + 1]).read_text(encoding="utf-8"))


def _bone_shapes() -> set:
    return {
        bone.custom_shape.name
        for obj in bpy.data.objects
        if obj.type == "ARMATURE"
        for bone in obj.pose.bones
        if bone.custom_shape is not None
    }


def _asset_meshes() -> list:
    shapes = _bone_shapes()
    return [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name not in shapes and not obj.hide_render]


def _play(clip: str) -> bpy.types.Action:
    action = bpy.data.actions.get(clip)
    if action is None:
        raise RuntimeError(f"clip {clip!r} not in the glb; it has {[a.name for a in bpy.data.actions]}")
    for obj in bpy.data.objects:
        data = obj.animation_data
        if data is None:
            continue
        data.action = None
        for track in data.nla_tracks:
            track.mute = True
            for strip in track.strips:
                if strip.action == action:
                    data.action = action
                    data.action_slot = strip.action_slot
    for shape in _bone_shapes():
        bpy.data.objects[shape].hide_render = True
    return action


def _sample_frames(start: float, frame_count: int, count: int) -> list:
    return [start + round(index * frame_count / count) for index in range(count)]


def _union_bbox(meshes: list, frames: list) -> tuple:
    scene = bpy.context.scene
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    for frame in frames:
        scene.frame_set(int(frame))
        low, high = sheet._world_bbox(meshes)
        minimum = Vector(map(min, minimum, low))
        maximum = Vector(map(max, maximum, high))
    return minimum, maximum


def main() -> None:
    args = _read_args()
    sheet._clear_scene()
    scene = bpy.context.scene
    scene.render.fps = int(args["fps"])
    scene.render.fps_base = 1.0
    device_info = sheet._set_cycles_device(args.get("device", "OPTIX"))
    sheet._build_world()
    sheet._import_asset(args["glb_path"])
    action = _play(args["clip"])
    meshes = _asset_meshes()
    if not meshes:
        raise RuntimeError(f"no mesh objects found in {args['glb_path']}")

    frames = _sample_frames(action.frame_range[0], int(args["frame_count"]), len(args["outputs"]))
    bbox_min, bbox_max = _union_bbox(meshes, frames)
    centre = (bbox_min + bbox_max) / 2.0
    radius = max((bbox_max - bbox_min).length / 2.0, 0.1)

    camera_data = bpy.data.cameras.new("FilmstripCam")
    camera = bpy.data.objects.new("FilmstripCam", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "PERSP"
    camera_data.lens = 35.0
    camera_data.clip_start = 0.01
    camera_data.clip_end = radius * 20.0 + 20.0
    margin = float(args.get("margin", 1.15))
    distance = radius / math.tan(camera_data.angle / 2.0) * margin
    azimuth = math.radians(float(args.get("azimuth_deg", -60.0)))
    altitude = math.radians(float(args.get("altitude_deg", 25.0)))
    camera.location = centre + Vector(
        (math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude))
    ) * distance
    sheet._look_at(camera, centre)
    sheet._build_light_rig(camera.location, centre, radius)

    rendered = []
    for frame, output in zip(frames, args["outputs"]):
        scene.frame_set(int(frame))
        sheet._render(scene, args["resolution"], int(args.get("samples", 32)), output)
        rendered.append(int(frame - action.frame_range[0]))

    result = {
        "device": device_info,
        "frames": rendered,
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
    }
    Path(args["result_json"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
