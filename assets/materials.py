"""Hard-surface materials: procedural (concrete, plaster, brick, metal, glass, roofing) and CC0.

Runs inside Blender's own interpreter, never the uv venv - only bpy and the stdlib (the CC0 path
reads library/asset_materials.json directly with json/pathlib rather than importing
library.materials, which pulls in requests through library.polyhaven and isn't guaranteed
present in Blender's own interpreter). Procedural materials are shader nodes only
(noise/voronoi/brick textures feeding a Principled BSDF) - no image textures, no baked AO, no
network pull. CC0 materials are photographed PolyHaven tiling sets pulled ahead of time by
library/polyhaven.py; get_material()/assign_material() try the procedural registry first, then
the CC0 index, so a caller names either kind the same way. The engine lights the asset, per
CLAUDE.md's asset conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

import bpy

Color = tuple[float, float, float, float]

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "library"
CC0_INDEX_FILE = LIBRARY_DIR / "asset_materials.json"


def _bsdf_material(name: str) -> tuple[bpy.types.Material, bpy.types.NodeTree, bpy.types.ShaderNodeBsdfPrincipled]:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300.0, 0.0)
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material, tree, bsdf


def _noise(tree: bpy.types.NodeTree, scale: float, detail: float = 4.0, location: tuple[float, float] = (-600.0, 0.0)) -> bpy.types.ShaderNodeTexNoise:
    node = tree.nodes.new("ShaderNodeTexNoise")
    node.inputs["Scale"].default_value = scale
    node.inputs["Detail"].default_value = detail
    node.location = location
    return node


def _ramp(
    tree: bpy.types.NodeTree,
    factor_socket: bpy.types.NodeSocket,
    low: Color,
    high: Color,
    location: tuple[float, float] = (-350.0, 0.0),
) -> bpy.types.ShaderNodeValToRGB:
    node = tree.nodes.new("ShaderNodeValToRGB")
    node.location = location
    node.color_ramp.elements[0].color = low
    node.color_ramp.elements[1].color = high
    tree.links.new(factor_socket, node.inputs["Fac"])
    return node


def concrete_material(name: str = "concrete") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=6.0, detail=4.0, location=(-600.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.24, 0.24, 0.22, 1.0), (0.5, 0.49, 0.46, 1.0), location=(-350.0, 150.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    grain = _noise(tree, scale=45.0, detail=6.0, location=(-600.0, -150.0))
    grain_ramp = _ramp(tree, grain.outputs["Fac"], (0.55, 0.55, 0.55, 1.0), (0.88, 0.88, 0.88, 1.0), location=(-350.0, -150.0))
    tree.links.new(grain_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def plaster_material(name: str = "plaster") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=3.0, detail=2.0)
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.72, 0.7, 0.65, 1.0), (0.84, 0.82, 0.78, 1.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def brick_material(name: str = "brick") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=20.0, detail=4.0, location=(-900.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.28, 0.09, 0.06, 1.0), (0.45, 0.17, 0.12, 1.0), location=(-650.0, 150.0))

    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.location = (-350.0, 0.0)
    brick.inputs["Mortar"].default_value = (0.62, 0.6, 0.57, 1.0)
    brick.inputs["Scale"].default_value = 4.0
    brick.inputs["Mortar Size"].default_value = 0.02
    tree.links.new(tint_ramp.outputs["Color"], brick.inputs["Color1"])
    tree.links.new(tint_ramp.outputs["Color"], brick.inputs["Color2"])
    tree.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])

    roughness_ramp = _ramp(tree, brick.outputs["Factor"], (0.6, 0.6, 0.6, 1.0), (0.9, 0.9, 0.9, 1.0), location=(-100.0, -200.0))
    tree.links.new(roughness_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def metal_material(name: str = "metal") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=8.0, detail=3.0, location=(-600.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.52, 0.53, 0.55, 1.0), (0.62, 0.63, 0.65, 1.0), location=(-350.0, 150.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    brushing = _noise(tree, scale=60.0, detail=8.0, location=(-600.0, -150.0))
    brushing_ramp = _ramp(tree, brushing.outputs["Fac"], (0.12, 0.12, 0.12, 1.0), (0.4, 0.4, 0.4, 1.0), location=(-350.0, -150.0))
    tree.links.new(brushing_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 1.0
    return material


def wood_material(name: str = "wood") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=3.0, detail=2.0, location=(-900.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.24, 0.13, 0.06, 1.0), (0.42, 0.26, 0.12, 1.0), location=(-650.0, 150.0))

    grain = tree.nodes.new("ShaderNodeTexWave")
    grain.location = (-650.0, -150.0)
    grain.wave_type = "BANDS"
    grain.bands_direction = "Z"
    grain.inputs["Scale"].default_value = 12.0
    grain.inputs["Distortion"].default_value = 3.0
    grain.inputs["Detail"].default_value = 3.0
    grain_ramp = _ramp(tree, grain.outputs["Fac"], (0.8, 0.8, 0.8, 1.0), (1.0, 1.0, 1.0, 1.0), location=(-350.0, -150.0))

    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.location = (-100.0, 150.0)
    tree.links.new(tint_ramp.outputs["Color"], mix.inputs["A"])
    tree.links.new(grain_ramp.outputs["Color"], mix.inputs["B"])
    tree.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    roughness_ramp = _ramp(tree, grain.outputs["Fac"], (0.55, 0.55, 0.55, 1.0), (0.75, 0.75, 0.75, 1.0), location=(-100.0, -350.0))
    tree.links.new(roughness_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def glass_material(name: str = "glass") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=2.0, detail=2.0)
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.82, 0.88, 0.9, 1.0), (0.87, 0.92, 0.94, 1.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    return material


def roofing_material(name: str = "roofing") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=15.0, detail=4.0, location=(-900.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.1, 0.08, 0.07, 1.0), (0.22, 0.17, 0.14, 1.0), location=(-650.0, 150.0))

    shingles = tree.nodes.new("ShaderNodeTexBrick")
    shingles.location = (-350.0, 0.0)
    shingles.inputs["Mortar"].default_value = (0.05, 0.05, 0.05, 1.0)
    shingles.inputs["Scale"].default_value = 10.0
    shingles.inputs["Mortar Size"].default_value = 0.03
    shingles.inputs["Brick Width"].default_value = 0.3
    shingles.inputs["Row Height"].default_value = 0.15
    tree.links.new(tint_ramp.outputs["Color"], shingles.inputs["Color1"])
    tree.links.new(tint_ramp.outputs["Color"], shingles.inputs["Color2"])
    tree.links.new(shingles.outputs["Color"], bsdf.inputs["Base Color"])

    roughness_ramp = _ramp(tree, shingles.outputs["Factor"], (0.7, 0.7, 0.7, 1.0), (0.92, 0.92, 0.92, 1.0), location=(-100.0, -200.0))
    tree.links.new(roughness_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


MATERIAL_BUILDERS: dict[str, Callable[[str], bpy.types.Material]] = {
    "concrete": concrete_material,
    "plaster": plaster_material,
    "brick": brick_material,
    "metal": metal_material,
    "glass": glass_material,
    "roofing": roofing_material,
    "wood": wood_material,
}


def load_cc0_index(path: Path = CC0_INDEX_FILE) -> dict[str, dict]:
    """Read the plain-JSON asset material index (stdlib-only; no library.materials import, which
    pulls in requests through library.polyhaven and isn't guaranteed present in Blender's own
    interpreter)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("materials", {})


def _cc0_image(tree: bpy.types.NodeTree, path: Path, non_color: bool, location: tuple[float, float]) -> bpy.types.ShaderNodeTexImage:
    node = tree.nodes.new("ShaderNodeTexImage")
    node.location = location
    node.image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    return node


def cc0_material(name: str, entry: dict, library_dir: Path = LIBRARY_DIR) -> bpy.types.Material:
    """Build a tiling image-texture material from a library.materials-style index entry.

    Only albedo/normal/roughness are wired in - no AO node, per CLAUDE.md's rule against baking
    AO darkening into an asset's own albedo once af-4ir.1.2's atlas bake runs on top of this.
    """
    material, tree, bsdf = _bsdf_material(name)

    coord = tree.nodes.new("ShaderNodeTexCoord")
    coord.location = (-900.0, 0.0)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.location = (-700.0, 0.0)
    scale = 1.0 / float(entry["tiling_m"])
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])

    maps = entry["maps"]
    albedo = _cc0_image(tree, library_dir / maps["albedo"], non_color=False, location=(-400.0, 250.0))
    tree.links.new(mapping.outputs["Vector"], albedo.inputs["Vector"])
    tree.links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])

    roughness = _cc0_image(tree, library_dir / maps["roughness"], non_color=True, location=(-400.0, -50.0))
    tree.links.new(mapping.outputs["Vector"], roughness.inputs["Vector"])
    tree.links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])

    normal = _cc0_image(tree, library_dir / maps["normal"], non_color=True, location=(-400.0, -350.0))
    tree.links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.location = (-150.0, -350.0)
    tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    return material


def get_material(name: str, cc0_index: Mapping[str, dict] | None = None) -> bpy.types.Material:
    """Return the named material from bpy.data if it already exists, else build it - from the
    procedural registry first, falling back to the CC0 index so callers can name either kind
    without caring which they get."""
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    builder = MATERIAL_BUILDERS.get(name)
    if builder is not None:
        return builder(name)
    index = load_cc0_index() if cc0_index is None else cc0_index
    entry = index.get(name)
    if entry is not None:
        return cc0_material(name, entry)
    raise ValueError(f"no procedural or CC0 material builder registered for {name!r}")


def assign_material(
    obj: bpy.types.Object,
    face_indices: Iterable[int] | None,
    name: str,
    cc0_index: Mapping[str, dict] | None = None,
) -> int:
    """Assign the named material (procedural or CC0) to face_indices (all faces if None), adding a slot on obj if needed."""
    material = get_material(name, cc0_index)
    if material.name not in obj.data.materials:
        obj.data.materials.append(material)
    slot_index = obj.data.materials.find(material.name)
    polygons = obj.data.polygons
    target = range(len(polygons)) if face_indices is None else face_indices
    for index in target:
        polygons[index].material_index = slot_index
    return slot_index
