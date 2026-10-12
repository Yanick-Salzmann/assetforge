"""Runs inside Blender's own interpreter via `blender --background <blend> --python`.

The long-running half of an asset's step 8 (game_ready + atlas bake + LODs + export) as a
headless job, so it never runs inside a live MCP call: the bake passes block Blender's only
thread for minutes, longer than the blender-mcp bridge's transport timeout. The live session
saves its scene to a .blend and hands that file to this script; the agent launches it as a
background shell job and is notified when it exits. Invoked as:
    blender --background <blend_path> --factory-startup --python game_ready_build.py -- <repo_root> <name> <kind> <result_json>

Every mesh object named `<name>_*` in the .blend is one part of the asset. Export is refused
unless the asset's approval gate is open. Progress goes to stdout as `[game_ready] <stage>` lines;
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


def main() -> None:
    repo_root, name, kind, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("ASSETFORGE_WORKSPACE", str(repo_root))
    started = time.monotonic()

    payload: dict = {"ok": False}
    try:
        from assets import atlas_bake, export, game_ready, gate, lod
        from assets import hardsurface as hs

        gate.open_gate(name).require_approved()
        if kind not in export.VALID_KINDS:
            raise ValueError(f"kind {kind!r} must be one of {export.VALID_KINDS}")

        objects = _parts(name)
        if not objects:
            raise ValueError(f"no mesh objects named {name}_* in {bpy.data.filepath}")
        _stage(f"{len(objects)} parts", started)

        for obj in objects:
            hs.apply_transforms(obj)
            hs.recalc_normals(obj)
        game_ready.uv_unwrap_and_pack(objects)
        _stage("uv unwrap and pack", started)

        atlas_bake.prepare_bake()
        atlas_bake.bake_albedo(objects, atlas_size=game_ready.ATLAS_SIZE)
        _stage("albedo baked", started)
        atlas_bake.bake_normal(objects, atlas_size=game_ready.ATLAS_SIZE)
        _stage("normal baked", started)
        atlas_bake.bake_orm(objects, atlas_size=game_ready.ATLAS_SIZE)
        _stage("orm baked", started)
        bake_result = atlas_bake.collect_bake_result()
        atlas_bake.purge_stale_images()

        material = game_ready.consolidate_material(objects, bake_result=bake_result)
        game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)
        _stage("atlas material", started)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
        lod0_object = bpy.context.view_layer.objects.active
        lod0_object.name = name
        lod_set = lod.generate_lods(lod0_object)
        _stage("lods", started)

        result = export.export_asset(lod_set.objects, name=name, kind=kind)
        _stage("exported", started)
        payload = {"ok": True, "seconds": round(time.monotonic() - started), **result.as_dict()}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[game_ready] {'done' if payload['ok'] else 'failed'}", flush=True)


if __name__ == "__main__":
    main()
