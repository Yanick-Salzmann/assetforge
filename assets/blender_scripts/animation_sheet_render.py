"""Runs inside Blender's own interpreter via `blender --background --python`.

Imports an exported animated glb, plays one clip and renders it from a single 3/4 camera at
evenly spaced frames, framed on the union of every sampled pose so moving parts never leave the
shot. With focus "moving" the camera frames only the meshes that move during the clip, clips away
whatever stands between it and them and lets no static mesh cast a shadow, so a small moving part
inside a building is still seen. Reuses the contact sheet's device, world and light rig. Invoked as:
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

MOVE_THRESHOLD_M = 0.001
FOCUS_CONTEXT = 1.6
FOCUS_MIN_RADIUS_M = 0.6
CLUSTER_GAP_M = 1.5
NEAR_CLIP_FACTOR = 1.2
VIEW_AZIMUTHS_DEG = tuple(-60.0 + 45.0 * step for step in range(8))
VIEW_ALTITUDES_DEG = (25.0, 50.0, 75.0)
CLUTTER_GRID = 12
BLOCKED_TARGET_WEIGHT = 50
LIGHT_IRRADIANCE = {"KeyLight": 39.0, "FillLight": 15.6}


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


def _evaluated_corners(obj) -> list:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]


def _union_bbox(meshes: list, frames: list) -> tuple:
    scene = bpy.context.scene
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    for frame in frames:
        scene.frame_set(int(frame))
        for obj in meshes:
            for point in _evaluated_corners(obj):
                minimum = Vector(map(min, minimum, point))
                maximum = Vector(map(max, maximum, point))
    return minimum, maximum


def _moving_meshes(meshes: list, frames: list) -> list:
    scene = bpy.context.scene
    scene.frame_set(int(frames[0]))
    rest = {obj.name: _evaluated_corners(obj) for obj in meshes}
    moving = set()
    for frame in frames[1:]:
        scene.frame_set(int(frame))
        for obj in meshes:
            if obj.name in moving:
                continue
            if any((a - b).length > MOVE_THRESHOLD_M for a, b in zip(rest[obj.name], _evaluated_corners(obj))):
                moving.add(obj.name)
    return [obj for obj in meshes if obj.name in moving]


def _bbox_centre(obj) -> Vector:
    corners = _evaluated_corners(obj)
    return sum(corners, Vector()) / len(corners)


def _first_cluster(moving: list) -> list:
    """The moving meshes chained within CLUSTER_GAP_M of the first one, so a clip that drives
    the same part in several far-apart places frames one place up close."""
    centres = {obj.name: _bbox_centre(obj) for obj in moving}
    cluster = [moving[0]]
    remaining = moving[1:]
    grown = True
    while grown:
        grown = False
        for obj in list(remaining):
            if any((centres[obj.name] - centres[member.name]).length <= CLUSTER_GAP_M for member in cluster):
                cluster.append(obj)
                remaining.remove(obj)
                grown = True
    return cluster


def _direction(azimuth_deg: float, altitude_deg: float) -> Vector:
    azimuth = math.radians(azimuth_deg)
    altitude = math.radians(altitude_deg)
    return Vector((math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude)))


def _sight_targets(framed: list) -> list:
    targets = []
    for obj in framed:
        corners = _evaluated_corners(obj)
        middle = sum(corners, Vector()) / len(corners)
        targets.append(middle)
        targets.extend(middle.lerp(corner, 0.5) for corner in corners)
    return targets


def _blocked_sightlines(eye: Vector, targets: list, framed_names: set) -> int:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    blocked = 0
    for target in targets:
        ray = target - eye
        hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, eye, ray.normalized(), distance=ray.length * 0.98)
        if hit and obj.name not in framed_names:
            blocked += 1
    return blocked


def _foreground_clutter(eye: Vector, centre: Vector, near: float, half_fov: float, framed_names: set) -> int:
    """Rays across the frame that stop on static geometry no farther from the near clip plane
    than the framed parts' own centre: walls beside or cut open in front of the parts."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    forward = (centre - eye).normalized()
    right = forward.cross(Vector((0.0, 0.0, 1.0)))
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    right = right.normalized()
    up = right.cross(forward).normalized()
    spread = math.tan(half_fov)
    clutter = 0
    for row in range(CLUTTER_GRID):
        for column in range(CLUTTER_GRID):
            u = (column + 0.5) / CLUTTER_GRID * 2.0 - 1.0
            v = (row + 0.5) / CLUTTER_GRID * 2.0 - 1.0
            ray = (forward + right * u * spread + up * v * spread).normalized()
            hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, eye, ray, distance=near)
            if hit and obj.name not in framed_names:
                clutter += 1
    return clutter


def _clearest_view(centre: Vector, near: float, half_fov: float, framed: list, preferred: tuple) -> tuple:
    """The view whose sightlines from the near clip plane to the framed parts cross the least
    static geometry, then whose frame holds the least foreground wall; the preferred 3/4 view
    wins every tie."""
    targets = _sight_targets(framed)
    names = {obj.name for obj in framed}
    views = [preferred] + [(azimuth, altitude) for altitude in VIEW_ALTITUDES_DEG for azimuth in VIEW_AZIMUTHS_DEG]

    def score(view: tuple) -> int:
        eye = centre + _direction(*view) * near
        blocked = _blocked_sightlines(eye, targets, names)
        return blocked * BLOCKED_TARGET_WEIGHT + _foreground_clutter(eye, centre, near, half_fov, names)

    return min(views, key=score)


def _match_light_distance(target: Vector) -> None:
    """Rig lights carried up close keep the contact sheet's irradiance at the target."""
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.name in LIGHT_IRRADIANCE:
            obj.data.energy = LIGHT_IRRADIANCE[obj.name] * (obj.location - target).length_squared


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
    moving = _moving_meshes(meshes, frames)
    focused = args.get("focus", "moving") == "moving" and bool(moving)
    framed = _first_cluster(moving) if focused else meshes
    bbox_min, bbox_max = _union_bbox(framed, frames)
    centre = (bbox_min + bbox_max) / 2.0
    radius = max((bbox_max - bbox_min).length / 2.0, 0.1)
    part_radius = radius
    if focused:
        radius = max(radius * FOCUS_CONTEXT, FOCUS_MIN_RADIUS_M)

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
    view = (float(args.get("azimuth_deg", -60.0)), float(args.get("altitude_deg", 25.0)))
    near = part_radius * NEAR_CLIP_FACTOR
    if focused:
        scene.frame_set(int(frames[0]))
        view = _clearest_view(centre, near, camera_data.angle / 2.0 / margin, framed, view)
    direction = _direction(*view)
    camera.location = centre + direction * distance
    sheet._look_at(camera, centre)
    if focused:
        camera_data.clip_start = max(0.01, distance - near)
        sheet._build_light_rig(centre + direction * near, centre, radius)
        _match_light_distance(centre)
    else:
        sheet._build_light_rig(camera.location, centre, radius)

    rendered = []
    for frame, output in zip(frames, args["outputs"]):
        scene.frame_set(int(frame))
        sheet._render(scene, args["resolution"], int(args.get("samples", 32)), output)
        rendered.append(int(frame - action.frame_range[0]))

    result = {
        "device": device_info,
        "frames": rendered,
        "focus": "moving" if focused else "asset",
        "moving_meshes": [obj.name for obj in moving],
        "framed_meshes": [obj.name for obj in framed],
        "view_deg": list(view),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
    }
    Path(args["result_json"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
