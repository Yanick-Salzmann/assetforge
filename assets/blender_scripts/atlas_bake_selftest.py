"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds a small two-material test cube (flat, known Base Color/Roughness/Metallic per material, no
noise nodes, so the expected bake output is known exactly), runs it through
assets/game_ready.py:uv_unwrap_and_pack, assets/atlas_bake.py:bake_atlas and
assets/game_ready.py:consolidate_material (wiring the bake result in), then exports it with
assets/export.py:export_asset - so the pytest suite gets automated coverage of bpy-only code it
cannot import directly. Samples the baked albedo/orm atlas at each material's face centroid and
compares against that material's known values, which is the "flat-lit render of the baked albedo
alone against known material base colors" af-4ir.1.2's acceptance criteria asks for: the DIFFUSE
bake with pass_filter={'COLOR'} already is that flat-lit, lighting-free capture. Invoked as:
    blender --background --factory-startup --python atlas_bake_selftest.py -- <repo_root> <out_dir> <result.json>
"""

import array
import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

ATLAS_SIZE = 512
TOLERANCE = 0.05


def _read_args() -> tuple[Path, Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, out_dir, result_path = argv[separator + 1:separator + 4]
    return Path(repo_root), Path(out_dir), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


def _flat_material(name: str, base_color: tuple, roughness: float, metallic: float) -> "bpy.types.Material":
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def _cube() -> "bpy.types.Object":
    mesh = bpy.data.meshes.new("atlas_bake_selftest_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("atlas_bake_selftest_obj", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _face_centroid_uv(obj: "bpy.types.Object", face_index: int) -> tuple[float, float]:
    polygon = obj.data.polygons[face_index]
    uv_layer = obj.data.uv_layers.active.data
    loops = range(polygon.loop_start, polygon.loop_start + polygon.loop_total)
    us = [uv_layer[li].uv[0] for li in loops]
    vs = [uv_layer[li].uv[1] for li in loops]
    return sum(us) / len(us), sum(vs) / len(vs)


def _read_pixels(image: "bpy.types.Image") -> "array.array":
    width, height = image.size
    buf = array.array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(buf)
    return buf


def _sample(pixels: "array.array", size: int, u: float, v: float) -> tuple[float, float, float, float]:
    x = min(int(u * size), size - 1)
    y = min(int(v * size), size - 1)
    idx = (y * size + x) * 4
    return pixels[idx], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]


def _srgb_encode(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def main() -> None:
    repo_root, out_dir, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from terrain import config

        config.ASSET_OUT_DIR = out_dir

        from assets import atlas_bake, export, game_ready

        red = _flat_material("bake_test_red", (0.8, 0.1, 0.1, 1.0), roughness=0.2, metallic=0.0)
        blue = _flat_material("bake_test_blue", (0.1, 0.2, 0.85, 1.0), roughness=0.7, metallic=1.0)

        obj = _cube()
        obj.data.materials.append(red)
        obj.data.materials.append(blue)
        for polygon in obj.data.polygons:
            polygon.material_index = 0 if polygon.index < 3 else 1

        objects = [obj]
        game_ready.uv_unwrap_and_pack(objects)

        # uv_unwrap_and_pack triangulates in place, so face indices/counts above no longer apply -
        # regroup by each triangle's (still-correct) material_index now that packing is done.
        face_slot = {polygon.index: polygon.material_index for polygon in obj.data.polygons}

        # Bake and wire twice, reusing the same atlas_material name both times - this is what a
        # live session does on a second patch/re-bake pass. wire_atlas_textures must replace its
        # own prior nodes rather than accumulate them, or export_asset would pick up every stale
        # image alongside the current three.
        stale_bake_result = atlas_bake.bake_atlas(objects, atlas_size=ATLAS_SIZE, samples=8)
        game_ready.consolidate_material(objects, bake_result=stale_bake_result)

        bake_result = atlas_bake.bake_atlas(objects, atlas_size=ATLAS_SIZE, samples=8)
        material = game_ready.consolidate_material(objects, bake_result=bake_result)

        image_nodes_after_rewire = [n for n in material.node_tree.nodes if n.type == "TEX_IMAGE"]
        rewire_is_idempotent = (
            len(image_nodes_after_rewire) == 3
            and {n.image.name for n in image_nodes_after_rewire}
            == {bake_result.albedo.name, bake_result.normal.name, bake_result.orm.name}
        )

        result = export.export_asset(objects, name="atlas-bake-selftest", kind="prop")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        expected = {
            0: {"albedo": (0.8, 0.1, 0.1), "roughness": 0.2, "metallic": 0.0},
            1: {"albedo": (0.1, 0.2, 0.85), "roughness": 0.7, "metallic": 1.0},
        }
        albedo_pixels = _read_pixels(bake_result.albedo)
        orm_pixels = _read_pixels(bake_result.orm)

        sample_checks = []
        for face_index, slot in face_slot.items():
            u, v = _face_centroid_uv(obj, face_index)
            r, g, b, _ = _sample(albedo_pixels, ATLAS_SIZE, u, v)
            occlusion, roughness, metallic, _ = _sample(orm_pixels, ATLAS_SIZE, u, v)
            exp = expected[slot]
            expected_albedo_encoded = tuple(_srgb_encode(c) for c in exp["albedo"])
            albedo_close = all(abs(actual - want) < TOLERANCE for actual, want in zip((r, g, b), expected_albedo_encoded))
            roughness_close = abs(roughness - exp["roughness"]) < TOLERANCE
            metallic_close = abs(metallic - exp["metallic"]) < TOLERANCE
            sample_checks.append(
                {
                    "face_index": face_index,
                    "slot": slot,
                    "albedo": [r, g, b],
                    "expected_albedo": list(expected_albedo_encoded),
                    "albedo_close": albedo_close,
                    "occlusion": occlusion,
                    "roughness": roughness,
                    "expected_roughness": exp["roughness"],
                    "roughness_close": roughness_close,
                    "metallic": metallic,
                    "expected_metallic": exp["metallic"],
                    "metallic_close": metallic_close,
                }
            )

        textures = set(manifest["textures"])
        expected_textures = {
            f"{bake_result.albedo.name}.png",
            f"{bake_result.normal.name}.png",
            f"{bake_result.orm.name}.png",
        }

        power_of_two = all(
            image.size[0] == ATLAS_SIZE and image.size[1] == ATLAS_SIZE
            for image in (bake_result.albedo, bake_result.normal, bake_result.orm)
        )

        ok = (
            result.glb_path.is_file()
            and result.manifest_path.is_file()
            and textures == expected_textures
            and power_of_two
            and manifest["materials"] == [material.name]
            and rewire_is_idempotent
            and all(check["albedo_close"] and check["roughness_close"] and check["metallic_close"] for check in sample_checks)
        )
        payload = {
            "ok": ok,
            "manifest": manifest,
            "sample_checks": sample_checks,
            "rewire_is_idempotent": rewire_is_idempotent,
            "bake_result": bake_result.as_dict(),
        }
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
