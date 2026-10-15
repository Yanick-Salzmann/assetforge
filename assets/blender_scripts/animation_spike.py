"""Runs inside Blender's own interpreter via `blender --background --python`.

af-6jc.1 spike: builds a static base with a spinning child node (rigid object TRS animation) and a
skinned 4-bone strip, gives each its own action on an NLA track, exports the lot with one shared
material under every glTF animation mode, and reports what each glb actually contains.
Invoked as:
    blender --background --factory-startup --python animation_spike.py -- <out_dir> <result.json>
"""

import json
import math
import struct
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

FPS = 24
SPIN_FRAMES = 48
WAVE_FRAMES = 36
STRIP_HEIGHT_M = 2.0
STRIP_BONES = 4
MODES = ("ACTIONS", "NLA_TRACKS", "ACTIVE_ACTIONS")


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    out_dir, result_path = argv[separator + 1:separator + 3]
    return Path(out_dir), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def _shared_material() -> bpy.types.Material:
    material = bpy.data.materials.new("spike_mat")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.5, 0.4, 1.0)
    return material


def _box(name: str, size: tuple[float, float, float], offset: tuple[float, float, float], material) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vert in bm.verts:
        vert.co.x = vert.co.x * size[0] + offset[0]
        vert.co.y = vert.co.y * size[1] + offset[1]
        vert.co.z = vert.co.z * size[2] + offset[2]
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def _push_to_nla(id_owner, action: bpy.types.Action, track_name: str) -> None:
    animation_data = id_owner.animation_data_create()
    track = animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(action.name, int(action.frame_range[0]), action)
    strip.name = action.name
    animation_data.action = None


def _spinning_rotor(material) -> list[bpy.types.Object]:
    base = _box("Base", (1.0, 1.0, 0.4), (0.0, 0.0, 0.2), material)
    rotor = _box("Rotor_ANIM", (1.6, 0.15, 0.08), (0.0, 0.0, 0.0), material)
    rotor.location = (0.0, 0.0, 0.5)
    rotor.parent = base
    rotor.rotation_mode = "XYZ"
    rotor.animation_data_create()
    rotor.rotation_euler = (0.0, 0.0, 0.0)
    rotor.keyframe_insert("rotation_euler", index=2, frame=1)
    rotor.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    rotor.keyframe_insert("rotation_euler", index=2, frame=1 + SPIN_FRAMES)
    action = rotor.animation_data.action
    action.name = "rotor_spin"
    for fcurve in _fcurves(action, rotor):
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"
    _push_to_nla(rotor, action, "rotor_spin")
    return [base, rotor]


def _fcurves(action: bpy.types.Action, owner) -> list:
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                curves.extend(channelbag.fcurves)
    return curves


def _strip_mesh(material) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("Strip_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vert in bm.verts:
        vert.co.x *= 0.3
        vert.co.y *= 0.05
        vert.co.z = (vert.co.z + 0.5) * STRIP_HEIGHT_M
    edges = [edge for edge in bm.edges if abs(edge.verts[0].co.z - edge.verts[1].co.z) > 1e-6]
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=STRIP_BONES * 4 - 1, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Strip", mesh)
    obj.data.materials.append(material)
    obj.location = (2.0, 0.0, 0.0)
    bpy.context.collection.objects.link(obj)
    return obj


def _skinned_strip(material) -> list[bpy.types.Object]:
    strip = _strip_mesh(material)
    armature_data = bpy.data.armatures.new("StripRig_data")
    rig = bpy.data.objects.new("StripRig", armature_data)
    rig.location = strip.location
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    segment = STRIP_HEIGHT_M / STRIP_BONES
    parent = None
    for index in range(STRIP_BONES):
        bone = armature_data.edit_bones.new(f"seg{index}")
        bone.head = (0.0, 0.0, index * segment)
        bone.tail = (0.0, 0.0, (index + 1) * segment)
        if parent is not None:
            bone.parent = parent
            bone.use_connect = True
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    for index in range(STRIP_BONES):
        group = strip.vertex_groups.new(name=f"seg{index}")
        for vert in strip.data.vertices:
            centre = (index + 0.5) * segment
            weight = max(0.0, 1.0 - abs(vert.co.z - centre) / segment)
            if weight > 0.0:
                group.add([vert.index], weight, "REPLACE")
    strip.parent = rig
    strip.location = (0.0, 0.0, 0.0)
    modifier = strip.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    rig.animation_data_create()
    for frame, sign in ((1, 1.0), (1 + WAVE_FRAMES // 2, -1.0), (1 + WAVE_FRAMES, 1.0)):
        for index in range(STRIP_BONES):
            pose_bone = rig.pose.bones[f"seg{index}"]
            pose_bone.rotation_mode = "XYZ"
            pose_bone.rotation_euler = (sign * 0.25, 0.0, 0.0)
            pose_bone.keyframe_insert("rotation_euler", frame=frame)
    action = rig.animation_data.action
    action.name = "strip_wave"
    _push_to_nla(rig, action, "strip_wave")
    return [strip, rig]


def _glb_json(path: Path) -> dict:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length].decode("utf-8"))


def _summarise(path: Path) -> dict:
    document = _glb_json(path)
    nodes = document.get("nodes", [])
    accessors = document.get("accessors", [])
    animations = []
    for animation in document.get("animations", []):
        channels = []
        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            times = accessors[sampler["input"]]
            channels.append(
                {
                    "node": nodes[channel["target"]["node"]].get("name"),
                    "path": channel["target"]["path"],
                    "interpolation": sampler.get("interpolation", "LINEAR"),
                    "keys": times["count"],
                    "t_min": times.get("min"),
                    "t_max": times.get("max"),
                }
            )
        animations.append({"name": animation.get("name"), "channels": channels})
    return {
        "bytes": path.stat().st_size,
        "nodes": [
            {
                "name": node.get("name"),
                "children": [nodes[child].get("name") for child in node.get("children", [])],
                "mesh": "mesh" in node,
                "skin": "skin" in node,
                "translation": node.get("translation"),
            }
            for node in nodes
        ],
        "materials": [material.get("name") for material in document.get("materials", [])],
        "skins": [
            {"joints": [nodes[joint].get("name") for joint in skin["joints"]]}
            for skin in document.get("skins", [])
        ],
        "animations": animations,
    }


def _export(objects: list[bpy.types.Object], path: Path, mode: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_animations=True,
        export_animation_mode=mode,
        export_frame_range=False,
        export_force_sampling=True,
        export_anim_slide_to_zero=True,
        export_optimize_animation_size=True,
        export_skins=True,
        export_def_bones=False,
    )


def main() -> None:
    out_dir, result_path = _read_args()
    result: dict = {"ok": False, "blender": bpy.app.version_string}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_scene()
        scene = bpy.context.scene
        scene.render.fps = FPS
        scene.frame_start = 1
        scene.frame_end = 1 + max(SPIN_FRAMES, WAVE_FRAMES)
        material = _shared_material()
        objects = _spinning_rotor(material) + _skinned_strip(material)
        result["modes"] = {}
        for mode in MODES:
            path = out_dir / f"animation_spike_{mode.lower()}.glb"
            _export(objects, path, mode)
            result["modes"][mode] = _summarise(path)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


main()
