"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds every assets/cloth.py type (flag, pennant, banner, awning), waves it with its idle wind
clip and exports it skinned as out/assets/cloth-<kind>/, reporting as JSON that the attached edge
stays put, the free edge swings through the cloth's thickness rather than along it, the cloth
clears its rod and the glb carries one skin and the one clip. tests/test_cloth.py then runs
validate_asset on each export. Invoked as:
    blender --background --factory-startup --python cloth_selftest.py -- <repo_root> <result.json>
"""

import json
import sys
import traceback
from pathlib import Path

import bpy

DURATION_S = 2.5
FPS = 24


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, result_path = argv[separator + 1:separator + 3]
    return Path(repo_root), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials):
        for block in list(collection):
            collection.remove(block)


def _check_kind(kind: str) -> dict:
    from assets import cloth, export

    _clear_scene()
    scene = bpy.context.scene
    scene.render.fps = FPS
    material = bpy.data.materials.new(f"cloth_{kind}_atlas")
    material.use_nodes = True
    cloth_rig = getattr(cloth, kind)(f"cloth_{kind}", material=material)
    entry = cloth.animate_wind(cloth_rig, DURATION_S, seed=7)

    rig = cloth_rig.rig
    first = rig.data.bones[cloth_rig.bones[0]]
    last = rig.data.bones[cloth_rig.bones[-1]]
    rest_root = rig.matrix_world @ first.head_local
    rest_tip = rig.matrix_world @ last.tail_local
    normal = (rig.matrix_world.to_3x3() @ first.x_axis).normalized()
    along = (rest_tip - rest_root).normalized()
    root_drift = 0.0
    through = 0.0
    lengthwise = 0.0
    for frame in range(1, round(DURATION_S * FPS) + 1, 3):
        scene.frame_set(frame)
        root = rig.matrix_world @ rig.pose.bones[cloth_rig.bones[0]].head
        tip = rig.matrix_world @ rig.pose.bones[cloth_rig.bones[-1]].tail
        root_drift = max(root_drift, (root - rest_root).length)
        through = max(through, abs((tip - rest_tip).dot(normal)))
        lengthwise = max(lengthwise, abs((tip - rest_tip).dot(along)))
    scene.frame_set(1)

    rod = cloth_rig.rods[0]
    rod_points = [rod.matrix_world @ vertex.co for vertex in rod.data.vertices]
    cloth_points = [cloth_rig.cloth.matrix_world @ vertex.co for vertex in cloth_rig.cloth.data.vertices]
    rod_gap = min((point - rod_point).length for point in cloth_points for rod_point in rod_points)

    result = export.export_asset(cloth_rig.objects, name=f"cloth-{kind}", kind="prop", animations=[entry])
    document = export.glb_document(result.glb_path)
    clips = [animation["name"] for animation in document.get("animations", [])]
    skinned = [node for node in document.get("nodes", []) if "skin" in node]
    checks = {
        "entry": entry["kind"] == "skinned" and entry["state"] == "idle" and entry["nodes"] == cloth_rig.bones,
        "root_pinned": root_drift < 1e-5,
        "swings_through_thickness": through > 0.02,
        "swings_more_than_it_stretches": lengthwise < through,
        "clears_rod": rod_gap > 0.0015,
        "glb_clip": clips == [entry["name"]],
        "one_skin": len(document.get("skins", [])) == 1 and len(skinned) == 1,
    }
    return {
        "checks": checks,
        "through_m": through,
        "lengthwise_m": lengthwise,
        "rod_gap_m": rod_gap,
        "tris": len(cloth_rig.cloth.data.polygons) * 2,
        "ok": all(checks.values()),
    }


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))

    payload: dict = {"ok": False}
    try:
        from assets import cloth_shapes
        from terrain import config

        config.ASSET_OUT_DIR = result_path.parent / "assets_out"
        kinds = {kind: _check_kind(kind) for kind in cloth_shapes.BUILDERS}
        payload = {"ok": all(report["ok"] for report in kinds.values()), "kinds": kinds}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
