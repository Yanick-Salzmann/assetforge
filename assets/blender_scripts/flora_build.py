from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ASSETFORGE_WORKSPACE", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bpy
import numpy as np

from assets import export, flora, foliage_atlas, impostor, impostor_render, silhouette
from assets.recipe import RecipeSession
from terrain import vegetation
from terrain.config import ASSET_OUT_DIR, LIBRARY_DIR

SEED = 7
POP_MASK_PX = 480
LOD_PAIRS = ((0, 1), (1, 2), (2, 0))
PREFIX = "flora_"
ATLAS_NAME = "flora_atlas"
RECIPE = (
    "import runpy\n"
    "from pathlib import Path\n\n"
    f"runpy.run_path(str(Path(r\"{REPO.as_posix()}\") / \"assets/blender_scripts/flora_build.py\"), run_name=\"__main__\")"
)


def bark_strip() -> np.ndarray | None:
    index = json.loads((LIBRARY_DIR / "asset_materials.json").read_text(encoding="utf-8"))["materials"]
    entry = index.get("bark_cc0")
    if entry is None:
        return None
    path = LIBRARY_DIR / entry["maps"]["albedo"]
    if not path.is_file():
        return None
    size = foliage_atlas.BARK_WIDTH
    image = bpy.data.images.load(str(path))
    image.scale(size, size)
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return foliage_atlas.tile_bark(pixels.reshape(size, size, 4))


def atlas_image(pixels: np.ndarray) -> bpy.types.Image:
    size = foliage_atlas.ATLAS_SIZE
    image = bpy.data.images.new(ATLAS_NAME, size, size, alpha=True)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    return image


def flora_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new("flora")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    cutout = tree.nodes.new("ShaderNodeMath")
    cutout.operation = "ROUND"
    tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(texture.outputs["Alpha"], cutout.inputs[0])
    tree.links.new(cutout.outputs["Value"], bsdf.inputs["Alpha"])
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    material.use_backface_culling = False
    material.surface_render_method = "DITHERED"
    return material


def to_object(name: str, data: flora.MeshData, material: bpy.types.Material) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(data.vertices, [], data.faces)
    layer = mesh.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", [value for face in data.uvs for corner in face for value in corner])
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.normals_split_custom_set([corner for face in data.normals for corner in face])
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def silhouette_pops(meshes: list[flora.MeshData], alpha: np.ndarray, pixels: tuple[int, ...]) -> list[float]:
    """For each switch, the worst blurred_pop over silhouette.VIEWS between the two LODs it swaps,
    at the size the bounding sphere has on screen there. The impostor is rendered from LOD0, so
    the last switch compares LOD2 with LOD0."""
    frame = silhouette.Frame.around(meshes[0])
    pops = []
    for (a, b), px in zip(LOD_PAIRS, pixels):
        size = px * max(1, round(POP_MASK_PX / px))
        worst = max(
            silhouette.blurred_pop(
                silhouette.coverage(meshes[a], alpha, direction, size, frame),
                silhouette.coverage(meshes[b], alpha, direction, size, frame),
                px,
            )
            for direction in silhouette.VIEWS.values()
        )
        pops.append(round(worst, 3))
    return pops


def base_centre(meshes: list[flora.MeshData]) -> np.ndarray:
    points = np.concatenate([np.asarray(mesh.vertices) for mesh in meshes])
    low = points.min(axis=0)
    high = points.max(axis=0)
    return np.array(((low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, low[2]))


def record_lods(name: str, meshes: list[flora.MeshData], sphere: impostor_render.Sphere, has_impostor: bool, alpha: np.ndarray) -> None:
    transitions = flora.LODS if has_impostor else flora.LODS - 1
    pixels = vegetation.switch_pixels(transitions)
    path = ASSET_OUT_DIR / name / "asset.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["lod"] = {
        "screen_fractions": list(vegetation.LOD_SCREEN_FRACTIONS[:transitions]),
        "switch_distances_m": [round(d, 2) for d in vegetation.switch_distances(2.0 * sphere.radius, transitions)],
        "switch_px": list(pixels),
        "silhouette_pop": silhouette_pops(meshes, alpha, pixels),
        "sheet": "lod_sheet.png",
    }
    if has_impostor:
        centre = np.asarray(sphere.centre) - base_centre(meshes)
        manifest["impostor"] = {
            "albedo": f"textures/{impostor_render.ALBEDO_NAME}",
            "normal": f"textures/{impostor_render.NORMAL_NAME}",
            "layout": impostor.LAYOUT,
            "frames": impostor.FRAMES,
            "frame_px": impostor.FRAME_PX,
            "centre": [round(value, 4) for value in impostor.blender_to_gltf(tuple(centre))],
            "radius": round(sphere.radius, 4),
        }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    pixels = foliage_atlas.build(SEED, bark_strip())
    alpha = np.ascontiguousarray(pixels[..., 3])
    image = atlas_image(pixels)
    material = flora_material(image)
    only = set(os.environ.get("FLORA_MODELS", "").split(",")) - {""}
    for model, (kind, build) in flora.MODELS.items():
        if only and model not in only:
            continue
        name = f"{PREFIX}{model}"
        meshes = [build(lod=lod) for lod in range(flora.LODS)]
        objects = [to_object(f"{name}_LOD{lod}", mesh, material) for lod, mesh in enumerate(meshes)]
        export.export_asset(objects, name=name, kind="vegetation")
        out_dir = ASSET_OUT_DIR / name
        has_impostor = kind in vegetation.IMPOSTOR_KINDS
        if has_impostor:
            sphere = impostor_render.render_atlas(objects[0], image, out_dir / "textures")
        else:
            sphere = impostor_render.bounding_sphere(objects[0])
        albedo = out_dir / "textures" / impostor_render.ALBEDO_NAME if has_impostor else None
        impostor_render.render_lod_sheet(objects, image, albedo, sphere, out_dir / "lod_sheet.png")
        record_lods(name, meshes, sphere, has_impostor, alpha)
        RecipeSession(name=name, snippets=[RECIPE]).write()


if __name__ == "__main__":
    main()
