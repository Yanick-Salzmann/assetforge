"""Runs inside Blender's own interpreter via `blender --background <blend> --python`.

The long-running half of an asset's step 8 (game_ready + atlas bake + LODs + export) as a
headless job, so it never runs inside a live MCP call: the bake passes block Blender's only
thread for minutes, longer than the blender-mcp bridge's transport timeout. The live session
saves its scene to a .blend and hands that file to this script; the agent launches it as a
background shell job and is notified when it exits. Invoked as:
    blender --background <blend_path> --factory-startup --python game_ready_build.py -- <repo_root> <name> <kind> <result_json>

Every mesh object named `<name>_*` in the .blend is one part of the asset. Parts named
`<name>_*_ANIM` (motion.pivot_part) are its moving parts: they keep their pivot and stay separate
nodes through the bake and every LOD, and the clips and mechanisms motion recorded on the scene
(motion.recorded_clips, motion.recorded_mechanisms) are exported with them, as are the FX_<type>
sockets fx.add_fx_socket recorded (fx.recorded_fx) and the kit props props.add_prop placed (props.recorded_props); a socket parented to a static part is
unparented in place before the static parts are joined. Meshes deformed by an armature (assets.cloth
panels) are skinned parts: they stay out of the join, keep their bind pose, get an unreduced copy
per LOD, and their armatures and the skinned clips over those armatures' bones go out with them. Export is refused
unless the asset's approval gate is open and the approval names every one of those clips. Progress goes to stdout as `[game_ready] <stage>` lines;
the outcome goes to result_json as {"ok": true, ...} or {"ok": false, "error": <traceback>}.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import bpy


def _read_args() -> tuple[Path, str, str, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, name, kind, result_path = argv[separator + 1:separator + 5]
    return Path(repo_root), name, kind, Path(result_path)


def _stage(label: str, started: float) -> None:
    print(f"[game_ready] {label} ({time.monotonic() - started:.0f}s)", flush=True)


def _parts(name: str) -> list:
    prefix = f"{name}_"
    return [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name.startswith(prefix)]


def _rigs(skinned: list) -> list:
    rigs = []
    for part in skinned:
        for modifier in part.modifiers:
            if modifier.type == "ARMATURE" and modifier.object is not None and modifier.object not in rigs:
                rigs.append(modifier.object)
    return rigs


def main() -> None:
    repo_root, name, kind, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("ASSETFORGE_WORKSPACE", str(repo_root))
    started = time.monotonic()

    payload: dict = {"ok": False}
    try:
        from assets import atlas_bake, export, fx, game_ready, gate, lod, motion, props
        from assets import hardsurface as hs

        if kind not in export.VALID_KINDS:
            raise ValueError(f"kind {kind!r} must be one of {export.VALID_KINDS}")

        objects = _parts(name)
        if not objects:
            raise ValueError(f"no mesh objects named {name}_* in {bpy.data.filepath}")
        moving = [obj for obj in objects if game_ready.is_moving_part(obj)]
        moving_names = {obj.name for obj in moving}
        skinned = [obj for obj in objects if game_ready.is_skinned_part(obj) and not game_ready.is_moving_part(obj)]
        rigs = _rigs(skinned)
        for rig in rigs:
            rig.data.pose_position = "REST"
        bone_names = {bone.name for rig in rigs for bone in rig.data.bones}
        animations = [
            entry
            for entry in motion.recorded_clips()
            if set(entry["nodes"]) <= (moving_names if entry["kind"] == "rigid" else bone_names)
        ]
        clip_names = {entry["name"] for entry in animations}
        mechanisms = [entry for entry in motion.recorded_mechanisms() if entry["clip"] in clip_names]
        gate.open_gate(name).require_approved(clips=[entry["name"] for entry in animations])
        fx_entries = fx.recorded_fx()
        sockets = [bpy.data.objects[entry["node"]] for entry in fx_entries]
        placements = props.recorded_props()
        _stage(f"{len(objects)} parts, {len(skinned)} skinned, clips {[entry['name'] for entry in animations]}, fx {[entry['node'] for entry in fx_entries]}", started)

        static = [obj for obj in objects if not game_ready.is_animated_part(obj)]
        if not static:
            raise ValueError(f"{name} has no static parts to carry the moving ones")
        for socket in sockets:
            if socket.parent is not None and socket.parent.name not in moving_names and socket.parent not in rigs:
                world = socket.matrix_world.copy()
                socket.parent = None
                socket.matrix_world = world
        for obj in objects:
            if obj in static:
                hs.apply_transforms(obj)
            hs.recalc_normals(obj)
        layout = game_ready.uv_unwrap_and_pack(objects)
        _stage(f"uv unwrap and pack {layout.as_dict()}", started)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in static:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = static[0]
        bpy.ops.object.join()
        lod0_object = bpy.context.view_layer.objects.active
        lod0_object.name = name
        objects = [lod0_object, *moving, *skinned]
        _stage(f"joined, {len(moving)} moving part(s), {len(skinned)} skinned", started)

        atlas_bake.prepare_bake()
        atlas_bake.bake_albedo(objects, atlas_size=game_ready.ATLAS_SIZE, layout=layout)
        _stage("albedo baked", started)
        atlas_bake.bake_normal(objects, atlas_size=game_ready.ATLAS_SIZE, layout=layout)
        _stage("normal baked", started)
        atlas_bake.bake_orm(objects, atlas_size=game_ready.ATLAS_SIZE, layout=layout)
        _stage("orm baked", started)
        bake_result = atlas_bake.collect_bake_result()
        atlas_bake.purge_stale_images()

        material = game_ready.consolidate_material(objects, bake_result=bake_result)
        game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)
        _stage("atlas material", started)

        lod_set = lod.generate_lods(lod0_object, moving=moving, skinned=skinned)
        for rig in rigs:
            rig.data.pose_position = "POSE"
        _stage("lods", started)

        result = export.export_asset(
            [*lod_set.objects, *rigs, *sockets],
            name=name,
            kind=kind,
            animations=animations,
            mechanisms=mechanisms,
            fx=fx_entries,
            props=placements,
        )
        _stage("exported", started)
        payload = {"ok": True, "seconds": round(time.monotonic() - started), "atlas": layout.as_dict(), **result.as_dict()}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[game_ready] {'done' if payload['ok'] else 'failed'}", flush=True)


if __name__ == "__main__":
    main()
