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

from assets import export, flora, foliage_atlas
from assets.recipe import RecipeSession
from terrain.config import LIBRARY_DIR

SEED = 7
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


def atlas_image() -> bpy.types.Image:
    pixels = foliage_atlas.build(SEED, bark_strip())
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


def main() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    material = flora_material(atlas_image())
    for model, (_, build) in flora.MODELS.items():
        name = f"{PREFIX}{model}"
        obj = to_object(name, build(), material)
        export.export_asset([obj], name=name, kind="vegetation")
        RecipeSession(name=name, snippets=[RECIPE]).write()


if __name__ == "__main__":
    main()
