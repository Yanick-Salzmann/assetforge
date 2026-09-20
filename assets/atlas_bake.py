"""Composable bpy helpers that bake the shared atlas UV layout (assets/game_ready.py:
uv_unwrap_and_pack) into albedo/normal/orm textures.

Runs inside Blender's own interpreter, never the uv venv - only bpy, mathutils, numpy (bundled
with Blender's own Python) and the stdlib. A live MCP session imports this the same way it imports
assets.game_ready. The headless self-test in assets/blender_scripts/atlas_bake_selftest.py does
the same via subprocess.

Call bake_atlas on the object group right after uv_unwrap_and_pack, while every object still
carries its own per-face material (concrete/brick/metal/... from assets/materials.py) - it reads
those materials to render into the shared atlas. assets/game_ready.py:consolidate_material then
replaces those per-face materials with one atlas_material and, given this module's bake result,
wires the three images into its Principled BSDF.

Albedo and orm's Metallic channel are both baked through an EMIT proxy rather than Cycles' own
DIFFUSE/pass_filter={'COLOR'} bake: a material with Metallic=1 (assets/materials.py's
metal_material, for one) has no diffuse closure at all, so a DIFFUSE-type bake of its Base Color
silently comes back black regardless of the actual colour - confirmed on a two-material test cube
where the fully-metallic face's DIFFUSE bake read (0, 0, 0) while its Metallic=0 neighbour baked
correctly. Rerouting the input through an Emission shader (in place of the material's real surface
output, restored after) bakes the raw Base Color/Metallic value regardless of how the BSDF would
otherwise route it. Normal, AO and Roughness use their own direct Cycles bake types - only Metallic
lacks one, and Base Color needs the same treatment for the reason above. orm.png packs
occlusion/roughness/metallic into R/G/B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import bpy
import numpy as np

ALBEDO_NAME = "albedo"
NORMAL_NAME = "normal"
ORM_NAME = "orm"
DEFAULT_ATLAS_SIZE = 2048
DEFAULT_BAKE_SAMPLES = 8
DEFAULT_BAKE_DEVICE = "OPTIX"
DEFAULT_MARGIN = 4


@dataclass
class AtlasBakeResult:
    albedo: bpy.types.Image
    normal: bpy.types.Image
    orm: bpy.types.Image

    def as_dict(self) -> dict:
        return {"albedo": self.albedo.name, "normal": self.normal.name, "orm": self.orm.name}


def _select_only(objects: Sequence[bpy.types.Object]) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def _configure_cycles_device(device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.get_devices()
    enabled = [d for d in prefs.devices if d.type == device]
    for d in prefs.devices:
        d.use = d in enabled
    scene.cycles.device = "GPU" if enabled else "CPU"


def _principled_bsdf(material: bpy.types.Material) -> bpy.types.ShaderNode:
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    raise ValueError(f"material {material.name!r} has no Principled BSDF node")


def _unique_materials(objects: Sequence[bpy.types.Object]) -> list[bpy.types.Material]:
    materials: list[bpy.types.Material] = []
    seen: set[str] = set()
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is not None and material.name not in seen:
                seen.add(material.name)
                materials.append(material)
    return materials


def _attach_bake_targets(materials: Sequence[bpy.types.Material], image: bpy.types.Image) -> list[tuple[bpy.types.Material, bpy.types.ShaderNode]]:
    created = []
    for material in materials:
        tree = material.node_tree
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        for other in tree.nodes:
            other.select = False
        node.select = True
        tree.nodes.active = node
        created.append((material, node))
    return created


def _remove_bake_targets(created: Sequence[tuple[bpy.types.Material, bpy.types.ShaderNode]]) -> None:
    for material, node in created:
        material.node_tree.nodes.remove(node)


def _bake_direct(
    objects: Sequence[bpy.types.Object],
    materials: Sequence[bpy.types.Material],
    image: bpy.types.Image,
    bake_type: str,
    pass_filter: set[str] | None = None,
) -> None:
    _select_only(objects)
    targets = _attach_bake_targets(materials, image)
    try:
        kwargs = {"type": bake_type}
        if pass_filter is not None:
            kwargs["pass_filter"] = pass_filter
        bpy.ops.object.bake(**kwargs)
    finally:
        _remove_bake_targets(targets)


def _rewire_input_to_emission(material: bpy.types.Material, input_name: str):
    """Temporarily replace material's surface output with an Emission shader fed by whatever
    currently drives bsdf.inputs[input_name] (a link, or its scalar/color default_value), so an
    EMIT bake captures that raw input regardless of how the Principled BSDF would otherwise route
    it - e.g. Base Color on a Metallic=1 material contributes nothing to a DIFFUSE bake, since a
    fully metallic surface has no diffuse closure to sample, but still needs to end up in
    albedo.png. Returns a restore() callback that undoes the rewiring and removes any nodes it
    added."""
    tree = material.node_tree
    bsdf = _principled_bsdf(material)
    source_input = bsdf.inputs[input_name]
    output_node = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
    surface_input = output_node.inputs["Surface"]
    original_from = surface_input.links[0].from_socket if surface_input.is_linked else None

    created_nodes = []
    emission = tree.nodes.new("ShaderNodeEmission")
    created_nodes.append(emission)
    if source_input.is_linked:
        source = source_input.links[0].from_socket
    else:
        value_node = tree.nodes.new("ShaderNodeRGB" if source_input.type == "RGBA" else "ShaderNodeValue")
        value_node.outputs[0].default_value = source_input.default_value
        created_nodes.append(value_node)
        source = value_node.outputs[0]
    tree.links.new(source, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], surface_input)

    def restore() -> None:
        if original_from is not None:
            tree.links.new(original_from, surface_input)
        for node in created_nodes:
            tree.nodes.remove(node)

    return restore


def _bake_input_via_emission(objects: Sequence[bpy.types.Object], materials: Sequence[bpy.types.Material], image: bpy.types.Image, input_name: str) -> None:
    _select_only(objects)
    targets = _attach_bake_targets(materials, image)
    restores = [_rewire_input_to_emission(material, input_name) for material in materials]
    try:
        bpy.ops.object.bake(type="EMIT")
    finally:
        for restore in restores:
            restore()
        _remove_bake_targets(targets)


def _pack_orm(ao_image: bpy.types.Image, roughness_image: bpy.types.Image, metallic_image: bpy.types.Image, atlas_size: int) -> bpy.types.Image:
    pixel_count = atlas_size * atlas_size * 4
    ao = np.empty(pixel_count, dtype=np.float32)
    ao_image.pixels.foreach_get(ao)
    roughness = np.empty(pixel_count, dtype=np.float32)
    roughness_image.pixels.foreach_get(roughness)
    metallic = np.empty(pixel_count, dtype=np.float32)
    metallic_image.pixels.foreach_get(metallic)

    ao = ao.reshape(-1, 4)
    roughness = roughness.reshape(-1, 4)
    metallic = metallic.reshape(-1, 4)

    packed = np.empty_like(ao)
    packed[:, 0] = ao[:, 0]
    packed[:, 1] = roughness[:, 0]
    packed[:, 2] = metallic[:, 0]
    packed[:, 3] = 1.0

    orm = bpy.data.images.new(ORM_NAME, width=atlas_size, height=atlas_size, alpha=False)
    orm.colorspace_settings.name = "Non-Color"
    orm.pixels.foreach_set(packed.ravel())
    orm.update()
    return orm


def bake_atlas(
    objects: Sequence[bpy.types.Object],
    atlas_size: int = DEFAULT_ATLAS_SIZE,
    samples: int = DEFAULT_BAKE_SAMPLES,
    device: str = DEFAULT_BAKE_DEVICE,
    margin: int = DEFAULT_MARGIN,
) -> AtlasBakeResult:
    """Bake objects' current per-face materials into one shared albedo/normal/orm atlas, reading
    the UV layout uv_unwrap_and_pack already packed. Call this before consolidate_material, which
    still needs the objects' original materials to wire this result's images into the new atlas
    material."""
    objects = list(objects)
    if not objects:
        raise ValueError("bake_atlas requires at least one object")
    materials = _unique_materials(objects)
    if not materials:
        raise ValueError("bake_atlas requires objects with at least one material assigned")

    _configure_cycles_device(device)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = margin

    albedo = bpy.data.images.new(ALBEDO_NAME, width=atlas_size, height=atlas_size, alpha=False)
    _bake_input_via_emission(objects, materials, albedo, "Base Color")

    normal = bpy.data.images.new(NORMAL_NAME, width=atlas_size, height=atlas_size, alpha=False)
    normal.colorspace_settings.name = "Non-Color"
    _bake_direct(objects, materials, normal, bake_type="NORMAL")

    ao_scratch = bpy.data.images.new("orm_ao_scratch", width=atlas_size, height=atlas_size, alpha=False)
    ao_scratch.colorspace_settings.name = "Non-Color"
    _bake_direct(objects, materials, ao_scratch, bake_type="AO")

    roughness_scratch = bpy.data.images.new("orm_roughness_scratch", width=atlas_size, height=atlas_size, alpha=False)
    roughness_scratch.colorspace_settings.name = "Non-Color"
    _bake_direct(objects, materials, roughness_scratch, bake_type="ROUGHNESS")

    metallic_scratch = bpy.data.images.new("orm_metallic_scratch", width=atlas_size, height=atlas_size, alpha=False)
    metallic_scratch.colorspace_settings.name = "Non-Color"
    _bake_input_via_emission(objects, materials, metallic_scratch, "Metallic")

    orm = _pack_orm(ao_scratch, roughness_scratch, metallic_scratch, atlas_size)
    for scratch in (ao_scratch, roughness_scratch, metallic_scratch):
        bpy.data.images.remove(scratch)

    return AtlasBakeResult(albedo=albedo, normal=normal, orm=orm)


def wire_atlas_textures(material: bpy.types.Material, bake_result: AtlasBakeResult) -> None:
    """Wire a bake_atlas result's albedo/normal/orm images into material's Principled BSDF: Base
    Color, a tangent-space Normal Map node, and a Separate Color node splitting orm.png's
    Green/Blue channels into Roughness/Metallic. orm.png's Red (occlusion) channel is baked but
    left unwired here - Principled BSDF has no AO input, and CLAUDE.md forbids baking AO darkening
    into an asset's own albedo; the engine reads occlusion from the shared orm texture directly.

    Do not re-set normal/orm's image.colorspace_settings.name here even defensively - bake_atlas
    already set it to "Non-Color", and re-assigning it to the SAME value on an already-baked
    generated image silently zeroes its RGB pixel data (confirmed: nonzero pixel count dropped
    from a full bake to exactly width*height, i.e. only the alpha channel survived).

    Idempotent: consolidate_material reuses the same atlas_material data-block by name across
    repeated calls (a live session re-baking after a patch, or this function running twice on the
    same material), so any TEX_IMAGE/NormalMap/SeparateColor nodes a PRIOR call left wired here are
    removed first - otherwise they sit orphaned in the tree and export_asset's _export_textures
    would write every stale image alongside the current three."""
    if not material.use_nodes:
        material.use_nodes = True
    tree = material.node_tree
    bsdf = _principled_bsdf(material)

    for node in list(tree.nodes):
        if node.bl_idname in ("ShaderNodeTexImage", "ShaderNodeNormalMap", "ShaderNodeSeparateColor"):
            tree.nodes.remove(node)

    albedo_node = tree.nodes.new("ShaderNodeTexImage")
    albedo_node.image = bake_result.albedo
    albedo_node.location = (-600.0, 300.0)
    tree.links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])

    normal_tex_node = tree.nodes.new("ShaderNodeTexImage")
    normal_tex_node.image = bake_result.normal
    normal_tex_node.location = (-600.0, 0.0)
    normal_map_node = tree.nodes.new("ShaderNodeNormalMap")
    normal_map_node.location = (-300.0, 0.0)
    tree.links.new(normal_tex_node.outputs["Color"], normal_map_node.inputs["Color"])
    tree.links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

    orm_node = tree.nodes.new("ShaderNodeTexImage")
    orm_node.image = bake_result.orm
    orm_node.location = (-600.0, -300.0)
    separate_node = tree.nodes.new("ShaderNodeSeparateColor")
    separate_node.location = (-300.0, -300.0)
    tree.links.new(orm_node.outputs["Color"], separate_node.inputs["Color"])
    tree.links.new(separate_node.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(separate_node.outputs["Blue"], bsdf.inputs["Metallic"])
