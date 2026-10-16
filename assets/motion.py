"""bpy helpers that pivot parts and write assets.animation tracks as exportable clips.

Runs inside Blender's own interpreter - only bpy, mathutils, the stdlib and the stdlib-only
assets.animation / assets.animation_manifest. A live MCP session imports this the way it imports
assets.hardsurface, so the calls it records in recipe.py rebuild the same clips headless.

pivot_part moves a part's origin onto its pivot without moving its geometry and names it
<part>_ANIM. Every animate_* helper samples a primitive one key per frame from frame 1 with linear
interpolation, writes it into one action named <state>_<clip> with a slot per animated part, pushes
that action onto an NLA track of the same name, and returns the animations[] entry to hand to
assets.export.export_asset, and records it on the scene (recorded_clips). Calling a helper again
with the same clip name replaces the clip.
crank_linkage and gear_mesh return the matching mechanisms[] entries (assets.mechanism_manifest)
for export_asset(..., mechanisms=[...]), so validate_asset checks the parts stay coupled.

Rotations are about a principal axis of the part's parent space, so parts rest with identity
rotation. The crank-slider works in the plane spanned by PLANE_AXES[axis] = (u, v), right-handed
about the axis: the crank turns from u towards v, the slider runs along u, and at rest the crank
pin sits at r * u with the rod lying along +u from it.
"""

from __future__ import annotations

import json
import math
from typing import Sequence

import bpy
from mathutils import Matrix, Vector

from assets import animation
from assets import animation_manifest
from assets import mechanism_manifest

FIRST_FRAME = 1
AXES = ("X", "Y", "Z")
AXIS_PROPERTY = "anim_axis"
PLANE_AXES = {
    "X": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "Y": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "Z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}
CHANNELS = {"location": "location", "rotation": "rotation_euler"}
CLIPS_PROPERTY = "assetforge_animations"
MECHANISMS_PROPERTY = "assetforge_mechanisms"


class MotionRigError(ValueError):
    pass


def _axis(obj: bpy.types.Object, axis: str | None) -> str:
    chosen = axis or obj.get(AXIS_PROPERTY)
    if chosen not in AXES:
        raise MotionRigError(f"{obj.name}: axis must be one of {AXES}; pass it or set it with pivot_part")
    return chosen


def pivot_part(obj: bpy.types.Object, pivot: Sequence[float], axis: str = "Z", part: str | None = None) -> bpy.types.Object:
    """Move obj's origin to the world-space pivot, leaving its geometry where it is, record its
    rotation axis and rename it <part>_ANIM (part defaults to its current name)."""
    if axis not in AXES:
        raise MotionRigError(f"axis must be one of {AXES}, got {axis!r}")
    if obj.type != "MESH":
        raise MotionRigError(f"{obj.name}: only mesh parts can be pivoted")
    base = part or obj.name.removesuffix(animation_manifest.ANIM_SUFFIX)
    name = animation_manifest.anim_node_name(base)
    bpy.context.view_layer.update()
    local_pivot = obj.matrix_world.inverted() @ Vector(pivot)
    obj.data.transform(Matrix.Translation(-local_pivot))
    obj.data.update()
    obj.matrix_world = obj.matrix_world @ Matrix.Translation(local_pivot)
    obj.rotation_mode = "XYZ"
    obj.name = name
    obj[AXIS_PROPERTY] = axis
    bpy.context.view_layer.update()
    if obj.name != name:
        raise MotionRigError(f"an object named {name!r} already exists")
    return obj


def _scene_fps(fps: int | None) -> int:
    scene = bpy.context.scene
    scene_fps = scene.render.fps
    if scene.render.fps_base != 1.0:
        raise MotionRigError("scene fps_base must be 1 for exact-loop clips")
    if fps is not None and fps != scene_fps:
        raise MotionRigError(f"clip fps {fps} differs from scene fps {scene_fps}; the exporter samples at the scene rate")
    return scene_fps


def recorded_clips() -> list[dict]:
    """The animations[] entries of every clip written into this scene, in the order written - kept
    on the scene so a saved .blend carries them to the headless game_ready_build."""
    raw = bpy.context.scene.get(CLIPS_PROPERTY)
    return list(json.loads(raw).values()) if raw else []


def _store_clips(clips: dict) -> None:
    bpy.context.scene[CLIPS_PROPERTY] = json.dumps(clips)


def _record_clip(entry: dict) -> None:
    clips = {clip["name"]: clip for clip in recorded_clips()}
    clips[entry["name"]] = entry
    _store_clips(clips)


def _forget_clip(name: str) -> None:
    clips = {clip["name"]: clip for clip in recorded_clips() if clip["name"] != name}
    _store_clips(clips)


def recorded_mechanisms() -> list[dict]:
    """The mechanisms[] entries crank_linkage and gear_mesh returned in this scene whose clips are
    still recorded, in the order made."""
    raw = bpy.context.scene.get(MECHANISMS_PROPERTY)
    clips = {clip["name"] for clip in recorded_clips()}
    return [entry for entry in json.loads(raw) if entry["clip"] in clips] if raw else []


def _record_mechanism(entry: dict) -> dict:
    raw = bpy.context.scene.get(MECHANISMS_PROPERTY)
    entries = [other for other in json.loads(raw) if other != entry] if raw else []
    entries.append(entry)
    bpy.context.scene[MECHANISMS_PROPERTY] = json.dumps(entries)
    return entry


def _remove_clip(name: str) -> None:
    action = bpy.data.actions.get(name)
    for obj in bpy.data.objects:
        data = obj.animation_data
        if data is None:
            continue
        for track in list(data.nla_tracks):
            if track.name == name:
                data.nla_tracks.remove(track)
        if action is not None and data.action == action:
            data.action = None
    if action is not None:
        bpy.data.actions.remove(action)
    _forget_clip(name)


def write_clip(
    channels: Sequence[tuple[bpy.types.Object, str, int, Sequence[float]]],
    state: str,
    clip: str,
    fps: int,
    loop: bool = True,
    bones: Sequence[str] = (),
) -> dict:
    """One action <state>_<clip> keying every (part, data_path, index, values) channel from
    FIRST_FRAME, a slot per part, pushed to an NLA track per part; returns its animations[] entry.
    With bones the clip is skinned: the parts are armatures keyed on those pose bones."""
    if not channels:
        raise MotionRigError("a clip needs at least one channel")
    lengths = {len(values) for _, _, _, values in channels}
    if len(lengths) != 1 or min(lengths) < 2:
        raise MotionRigError("every channel of a clip needs the same number of samples, at least 2")
    for obj, _, _, _ in channels:
        if bones and obj.type != "ARMATURE":
            raise MotionRigError(f"{obj.name}: skinned clips key armatures only")
        if not bones and not animation_manifest.ANIM_NODE_NAME.match(obj.name):
            raise MotionRigError(f"{obj.name}: animated parts are named <part>{animation_manifest.ANIM_SUFFIX}; use pivot_part")
    samples = lengths.pop()
    name = animation_manifest.clip_name(state, clip)
    _remove_clip(name)

    action = bpy.data.actions.new(name)
    layer = action.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    slots: dict[str, bpy.types.ActionSlot] = {}
    for obj, data_path, index, values in channels:
        slot = slots.get(obj.name)
        if slot is None:
            slot = action.slots.new(id_type="OBJECT", name=obj.name)
            slots[obj.name] = slot
        bag = strip.channelbag(slot, ensure=True)
        curve = bag.fcurves.new(data_path, index=index)
        curve.keyframe_points.add(samples)
        coordinates = []
        for frame, value in enumerate(values, start=FIRST_FRAME):
            coordinates.extend((float(frame), float(value)))
        curve.keyframe_points.foreach_set("co", coordinates)
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
        curve.update()

    nodes = []
    for obj_name, slot in slots.items():
        obj = bpy.data.objects[obj_name]
        data = obj.animation_data or obj.animation_data_create()
        track = data.nla_tracks.new()
        track.name = name
        nla_strip = track.strips.new(name, FIRST_FRAME, action)
        nla_strip.name = name
        nla_strip.action_slot = slot
        nodes.append(obj_name)

    frames = samples - 1
    kind = "skinned" if bones else "rigid"
    entry = animation_manifest.manifest_entry(state, clip, kind, frames / fps, fps, list(bones) or nodes, loop=loop)
    _record_clip(entry)
    return entry


def _index(axis: str) -> int:
    return AXES.index(axis)


def animate_spin(
    obj: bpy.types.Object,
    period_s: float,
    state: str = "idle",
    clip: str = "spin",
    duration_s: float | None = None,
    fps: int | None = None,
    axis: str | None = None,
    clockwise: bool = False,
) -> dict:
    rate = _scene_fps(fps)
    chosen = _axis(obj, axis)
    track = animation.spin(period_s, duration_s or period_s, rate, clockwise=clockwise)
    return write_clip([(obj, "rotation_euler", _index(chosen), track.values)], state, clip, rate)


def animate_oscillate(
    obj: bpy.types.Object,
    amplitude: float,
    period_s: float,
    state: str = "idle",
    clip: str = "oscillate",
    channel: str = "location",
    duration_s: float | None = None,
    fps: int | None = None,
    axis: str | None = None,
    phase: float = 0.0,
) -> dict:
    """Sine about the rest pose: metres along axis for channel "location", radians about it for
    channel "rotation"."""
    if channel not in CHANNELS:
        raise MotionRigError(f"channel must be one of {tuple(CHANNELS)}, got {channel!r}")
    rate = _scene_fps(fps)
    index = _index(_axis(obj, axis))
    data_path = CHANNELS[channel]
    rest = getattr(obj, data_path)[index]
    track = animation.oscillate(amplitude, period_s, duration_s or period_s, rate, phase=phase, offset=rest)
    return write_clip([(obj, data_path, index, track.values)], state, clip, rate)


def animate_hinge(
    obj: bpy.types.Object,
    open_angle: float,
    period_s: float,
    state: str = "working",
    clip: str = "hinge",
    duration_s: float | None = None,
    fps: int | None = None,
    axis: str | None = None,
    hold_fraction: float = 0.25,
) -> dict:
    rate = _scene_fps(fps)
    index = _index(_axis(obj, axis))
    track = animation.hinge_swing(open_angle, period_s, duration_s or period_s, rate, hold_fraction=hold_fraction)
    return write_clip([(obj, "rotation_euler", index, track.values)], state, clip, rate)


def animate_crank(
    crank: bpy.types.Object,
    rod: bpy.types.Object,
    slider: bpy.types.Object,
    crank_radius_m: float,
    rod_length_m: float,
    period_s: float,
    state: str = "working",
    clip: str = "crank",
    duration_s: float | None = None,
    fps: int | None = None,
    axis: str | None = None,
    phase: float = 0.0,
) -> dict:
    """Drive a crank, its connecting rod and the slider it pushes from the crank's pivot. All
    three share a parent; the rod's origin is its crank-pin end and the slider's its wrist pin."""
    if not (crank.parent == rod.parent == slider.parent):
        raise MotionRigError("crank, rod and slider must share a parent")
    rate = _scene_fps(fps)
    chosen = _axis(crank, axis)
    index = _index(chosen)
    u, v = (Vector(direction) for direction in PLANE_AXES[chosen])
    centre = crank.location.copy()
    motion = animation.crank_slider(crank_radius_m, rod_length_m, period_s, duration_s or period_s, rate, phase=phase)
    pins = [centre + crank_radius_m * (u * math.cos(theta) + v * math.sin(theta)) for theta in motion.crank.values]
    slides = [centre + u * offset for offset in motion.slider.values]
    channels = [(crank, "rotation_euler", index, motion.crank.values), (rod, "rotation_euler", index, motion.rod.values)]
    for component in range(3):
        if any(abs(direction[component]) > 0.0 for direction in (u, v)):
            channels.append((rod, "location", component, [pin[component] for pin in pins]))
        if abs(u[component]) > 0.0:
            channels.append((slider, "location", component, [slide[component] for slide in slides]))
    return write_clip(channels, state, clip, rate)


def animate_gear_train(
    gears: Sequence[bpy.types.Object],
    teeth: Sequence[int],
    driver_period_s: float,
    state: str = "working",
    clip: str = "gears",
    duration_s: float | None = None,
    fps: int | None = None,
    axis: str | None = None,
    clockwise: bool = False,
) -> dict:
    """Meshing gears in a chain, the first driving: neighbours counter-rotate at their tooth
    ratio. The clip defaults to the shortest exact loop of every gear."""
    if len(gears) != len(teeth):
        raise MotionRigError("one tooth count per gear")
    rate = _scene_fps(fps)
    duration = duration_s or animation.gear_train_loop_duration(teeth, driver_period_s, rate)
    tracks = animation.gear_train(teeth, driver_period_s, duration, rate, clockwise=clockwise)
    channels = [(gear, "rotation_euler", _index(_axis(gear, axis)), track.values) for gear, track in zip(gears, tracks)]
    return write_clip(channels, state, clip, rate)



def animate_wind(
    rig: bpy.types.Object,
    bones: Sequence[str],
    duration_s: float,
    seed: int,
    strength: float = 1.0,
    clip: str = "wind",
    fps: int | None = None,
    bend_axis: str = "Z",
    twist_axis: str = "Y",
    gust_period_s: float | None = None,
    max_bend: float = 0.8,
    flutter: float = 0.35,
) -> dict:
    """An idle wind clip on a bone chain listed from the attached edge outwards: each pose bone
    bends about its local bend_axis with the travelling wave and twists about twist_axis (Y runs
    along a bone) with the flutter, from assets.animation.wind_wave."""
    if rig.type != "ARMATURE":
        raise MotionRigError(f"{rig.name}: wind needs an armature")
    if bend_axis not in AXES or twist_axis not in AXES or bend_axis == twist_axis:
        raise MotionRigError(f"bend_axis and twist_axis must be two different axes of {AXES}")
    missing = [bone for bone in bones if bone not in rig.pose.bones]
    if missing:
        raise MotionRigError(f"{rig.name} has no bones {missing}")
    rate = _scene_fps(fps)
    chain = animation.wind_wave(
        len(bones), duration_s, rate, seed, strength=strength, gust_period_s=gust_period_s, max_bend=max_bend, flutter=flutter
    )
    channels = []
    for bone, bend, twist in zip(bones, chain.bend, chain.twist):
        rig.pose.bones[bone].rotation_mode = "XYZ"
        data_path = f'pose.bones["{bone}"].rotation_euler'
        channels.append((rig, data_path, _index(bend_axis), bend.values))
        channels.append((rig, data_path, _index(twist_axis), twist.values))
    return write_clip(channels, "idle", clip, rate, bones=bones)


def crank_linkage(crank: bpy.types.Object, rod: bpy.types.Object, slider: bpy.types.Object, entry: dict) -> dict:
    """The rod's crank-pin end rides the crank and the slider's wrist pin rides the rod."""
    return _record_mechanism(mechanism_manifest.linkage_entry(entry["name"], [(rod.name, crank.name), (slider.name, rod.name)]))


def gear_mesh(gears: Sequence[bpy.types.Object], teeth: Sequence[int], entry: dict) -> dict:
    return _record_mechanism(mechanism_manifest.gears_entry(entry["name"], [gear.name for gear in gears], list(teeth)))
