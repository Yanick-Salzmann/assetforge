"""Runs inside Blender's own interpreter via `blender --background <blend> --python`.

The long-running half of asset-prop's steps 4-5 (one shared atlas bake across a whole prop set,
then one export per prop) as a headless job, so it never runs inside a live MCP call: the bake
passes block Blender's only thread for minutes, longer than the blender-mcp bridge's transport
timeout. The live session saves its scene to a .blend and hands that file to this script; the
agent launches it as a background shell job and is notified when it exits. Invoked as:
    blender --background <blend_path> --factory-startup --python prop_set_build.py -- <repo_root> <result_json>

Every mesh object tagged with assets.prop_set.tag is one part of the prop it names; untagged
objects are ignored. Props export with kind "prop" and no LODs. Progress goes to stdout as
`[prop_set] <stage>` lines; the outcome goes to result_json as {"ok": true, "props": {...}} or
{"ok": false, "error": <traceback>}.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import bpy


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, result_path = argv[separator + 1:separator + 3]
    return Path(repo_root), Path(result_path)


def _stage(label: str, started: float) -> None:
    print(f"[prop_set] {label} ({time.monotonic() - started:.0f}s)", flush=True)


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("ASSETFORGE_WORKSPACE", str(repo_root))
    started = time.monotonic()

    payload: dict = {"ok": False}
    try:
        from assets import atlas_bake, export, game_ready, prop_set
        from assets import hardsurface as hs

        groups = prop_set.group(obj for obj in bpy.data.objects if obj.type == "MESH")
        if not groups:
            raise ValueError(f"no mesh objects tagged with {prop_set.PROP_PROPERTY!r} in {bpy.data.filepath}")
        objects = [obj for parts in groups.values() for obj in parts]
        _stage(f"{len(groups)} props, {len(objects)} parts", started)

        for obj in objects:
            hs.apply_transforms(obj)
            hs.recalc_normals(obj)
        layout = game_ready.uv_unwrap_and_pack(objects)
        _stage("uv unwrap and pack", started)

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

        props = {}
        for prop, parts in groups.items():
            props[prop] = export.export_asset(parts, name=prop, kind="prop").as_dict()
            _stage(f"exported {prop}", started)
        payload = {"ok": True, "seconds": round(time.monotonic() - started), "props": props}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[prop_set] {'done' if payload['ok'] else 'failed'}", flush=True)


if __name__ == "__main__":
    main()
