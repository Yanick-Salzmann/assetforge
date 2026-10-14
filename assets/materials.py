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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import bpy

from terrain.config import LIBRARY_DIR

Color = tuple[float, float, float, float]

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


PAINT_GREEN: Color = (0.06, 0.2, 0.05, 1.0)
PAINT_CHARCOAL: Color = (0.035, 0.035, 0.038, 1.0)
PAINT_CREAM: Color = (0.62, 0.55, 0.34, 1.0)
LENS_AMBER: Color = (0.85, 0.35, 0.02, 1.0)
LENS_RED: Color = (0.6, 0.02, 0.02, 1.0)


def paint_material(name: str = "paint", colour: Color = PAINT_GREEN) -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=4.0, detail=3.0, location=(-600.0, 150.0))
    shade = tuple(channel * 0.88 for channel in colour[:3]) + (1.0,)
    tint_ramp = _ramp(tree, tint.outputs["Fac"], shade, colour, location=(-350.0, 150.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    wear = _noise(tree, scale=30.0, detail=6.0, location=(-600.0, -150.0))
    wear_ramp = _ramp(tree, wear.outputs["Fac"], (0.28, 0.28, 0.28, 1.0), (0.45, 0.45, 0.45, 1.0), location=(-350.0, -150.0))
    tree.links.new(wear_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def rubber_material(name: str = "rubber") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=12.0, detail=4.0, location=(-600.0, 150.0))
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.018, 0.018, 0.018, 1.0), (0.035, 0.034, 0.032, 1.0), location=(-350.0, 150.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    grain = _noise(tree, scale=80.0, detail=6.0, location=(-600.0, -150.0))
    grain_ramp = _ramp(tree, grain.outputs["Fac"], (0.78, 0.78, 0.78, 1.0), (0.92, 0.92, 0.92, 1.0), location=(-350.0, -150.0))
    tree.links.new(grain_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def tinted_glass_material(name: str = "tinted_glass") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _noise(tree, scale=2.0, detail=2.0)
    tint_ramp = _ramp(tree, tint.outputs["Fac"], (0.04, 0.055, 0.06, 1.0), (0.06, 0.075, 0.08, 1.0))
    tree.links.new(tint_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def _wall_coords(tree: bpy.types.NodeTree, location: tuple[float, float] = (-1200.0, 0.0)) -> bpy.types.NodeSocket:
    coord = tree.nodes.new("ShaderNodeTexCoord")
    coord.location = location
    split = tree.nodes.new("ShaderNodeSeparateXYZ")
    split.location = (location[0] + 200.0, location[1])
    tree.links.new(coord.outputs["Object"], split.inputs["Vector"])
    along = tree.nodes.new("ShaderNodeMath")
    along.operation = "ADD"
    along.location = (location[0] + 400.0, location[1] + 100.0)
    tree.links.new(split.outputs["X"], along.inputs[0])
    tree.links.new(split.outputs["Y"], along.inputs[1])
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.location = (location[0] + 600.0, location[1])
    tree.links.new(along.outputs["Value"], combine.inputs["X"])
    tree.links.new(split.outputs["Z"], combine.inputs["Y"])
    return combine.outputs["Vector"]


@dataclass(frozen=True)
class Weathering:
    roughness: tuple[float, float]
    edge_colour: Color
    edge_strength: float = 0.6
    edge_radius_m: float = 0.05
    grime_colour: Color = (0.13, 0.1, 0.07, 1.0)
    grime_strength: float = 0.55
    grime_height_m: float = 0.7
    dust_colour: Color = (0.42, 0.35, 0.25, 1.0)
    dust_strength: float = 0.45
    soot_colour: Color | None = None
    soot_coverage: float = 0.0
    flake_coverage: float = 0.0
    flake_scale: float = 1.4
    streak_strength: float = 0.12
    macro_range: tuple[float, float] = (0.86, 1.08)
    bump_scale: float = 6.0
    bump_strength: float = 0.25


def _position(tree: bpy.types.NodeTree) -> bpy.types.NodeSocket:
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-1800.0, 0.0)
    return geometry.outputs["Position"]


def _world_noise(
    tree: bpy.types.NodeTree,
    position: bpy.types.NodeSocket,
    scale: float,
    detail: float = 4.0,
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bpy.types.NodeSocket:
    node = _noise(tree, scale=scale, detail=detail, location=(-1400.0, 0.0))
    if stretch == (1.0, 1.0, 1.0):
        tree.links.new(position, node.inputs["Vector"])
        return node.outputs["Fac"]
    squash = tree.nodes.new("ShaderNodeVectorMath")
    squash.operation = "MULTIPLY"
    squash.inputs[1].default_value = stretch
    tree.links.new(position, squash.inputs[0])
    tree.links.new(squash.outputs["Vector"], node.inputs["Vector"])
    return node.outputs["Fac"]


def _map_range(
    tree: bpy.types.NodeTree,
    value: bpy.types.NodeSocket,
    from_min: float,
    from_max: float,
    to_min: float = 0.0,
    to_max: float = 1.0,
) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeMapRange")
    node.clamp = True
    node.inputs["From Min"].default_value = from_min
    node.inputs["From Max"].default_value = from_max
    node.inputs["To Min"].default_value = to_min
    node.inputs["To Max"].default_value = to_max
    tree.links.new(value, node.inputs["Value"])
    return node.outputs["Result"]


def _math(tree: bpy.types.NodeTree, operation: str, a: bpy.types.NodeSocket, b: bpy.types.NodeSocket | float) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.use_clamp = True
    tree.links.new(a, node.inputs[0])
    if isinstance(b, float):
        node.inputs[1].default_value = b
    else:
        tree.links.new(b, node.inputs[1])
    return node.outputs["Value"]


def _multiply_colour(tree: bpy.types.NodeTree, base: bpy.types.NodeSocket, factor: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = "MULTIPLY"
    node.inputs["Factor"].default_value = 1.0
    tree.links.new(base, node.inputs["A"])
    tree.links.new(factor, node.inputs["B"])
    return node.outputs["Result"]


def _mix_colour(tree: bpy.types.NodeTree, base: bpy.types.NodeSocket, colour: Color, factor: bpy.types.NodeSocket) -> bpy.types.NodeSocket:
    node = tree.nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    tree.links.new(factor, node.inputs["Factor"])
    tree.links.new(base, node.inputs["A"])
    node.inputs["B"].default_value = colour
    return node.outputs["Result"]


def _edge_mask(tree: bpy.types.NodeTree, radius_m: float) -> bpy.types.NodeSocket:
    bevel = tree.nodes.new("ShaderNodeBevel")
    bevel.samples = 16
    bevel.inputs["Radius"].default_value = radius_m
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    tree.links.new(bevel.outputs["Normal"], dot.inputs[0])
    tree.links.new(geometry.outputs["Normal"], dot.inputs[1])
    return _map_range(tree, dot.outputs["Value"], 0.97, 0.75)


def _weather(
    tree: bpy.types.NodeTree,
    bsdf: bpy.types.ShaderNodeBsdfPrincipled,
    colour: bpy.types.NodeSocket,
    weathering: Weathering,
    recess: bpy.types.NodeSocket | None = None,
) -> None:
    position = _position(tree)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(position, height.inputs["Vector"])
    facing = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(geometry.outputs["Normal"], facing.inputs["Vector"])

    macro = _map_range(tree, _world_noise(tree, position, scale=0.35, detail=3.0), 0.3, 0.7, *weathering.macro_range)
    macro_colour = tree.nodes.new("ShaderNodeCombineColor")
    for channel in ("Red", "Green", "Blue"):
        tree.links.new(macro, macro_colour.inputs[channel])
    colour = _multiply_colour(tree, colour, macro_colour.outputs["Color"])

    if weathering.soot_colour is not None:
        clouds = _world_noise(tree, position, scale=2.2, detail=3.0)
        threshold = 0.5 + 0.2 * (1.0 - weathering.soot_coverage)
        soot = _map_range(tree, clouds, threshold, threshold + 0.12, 0.0, 0.85)
        colour = _mix_colour(tree, colour, weathering.soot_colour, soot)

    weather_streaks = _world_noise(tree, position, scale=2.0, detail=4.0, stretch=(5.0, 5.0, 0.35))
    colour = _mix_colour(tree, colour, weathering.grime_colour, _map_range(tree, weather_streaks, 0.45, 0.75, 0.0, weathering.streak_strength))

    if weathering.flake_coverage > 0.0:
        blotches = _world_noise(tree, position, scale=weathering.flake_scale, detail=6.0)
        threshold = 0.72 - 0.3 * weathering.flake_coverage
        flakes = _map_range(tree, blotches, threshold, threshold + 0.04)
        colour = _mix_colour(tree, colour, weathering.edge_colour, flakes)

    chips = _map_range(tree, _world_noise(tree, position, scale=9.0, detail=6.0), 0.35, 0.6)
    edge = _math(tree, "MULTIPLY", _edge_mask(tree, weathering.edge_radius_m), chips)
    colour = _mix_colour(tree, colour, weathering.edge_colour, _math(tree, "MULTIPLY", edge, weathering.edge_strength))

    rise = _map_range(tree, height.outputs["Z"], 0.0, weathering.grime_height_m, 1.0, 0.0)
    streaks = _map_range(tree, _world_noise(tree, position, scale=3.0, detail=5.0, stretch=(2.5, 2.5, 0.6)), 0.3, 0.7, 0.4, 1.0)
    grime = _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", rise, streaks), weathering.grime_strength)
    colour = _mix_colour(tree, colour, weathering.grime_colour, grime)

    upward = _map_range(tree, facing.outputs["Z"], 0.55, 0.95)
    drifts = _map_range(tree, _world_noise(tree, position, scale=1.6, detail=4.0), 0.35, 0.65, 0.3, 1.0)
    dust = _math(tree, "MULTIPLY", _math(tree, "MULTIPLY", upward, drifts), weathering.dust_strength)
    colour = _mix_colour(tree, colour, weathering.dust_colour, dust)
    tree.links.new(colour, bsdf.inputs["Base Color"])

    low, high = weathering.roughness
    roughness = _map_range(tree, _world_noise(tree, position, scale=4.0, detail=5.0), 0.3, 0.7, low, high)
    roughness = _math(tree, "ADD", roughness, _math(tree, "MULTIPLY", dust, 0.3))
    roughness = _math(tree, "SUBTRACT", roughness, _math(tree, "MULTIPLY", edge, 0.12))
    tree.links.new(roughness, bsdf.inputs["Roughness"])

    relief = _math(tree, "SUBTRACT", _world_noise(tree, position, scale=weathering.bump_scale, detail=6.0), _math(tree, "MULTIPLY", edge, 0.6))
    if recess is not None:
        relief = _math(tree, "SUBTRACT", relief, recess)
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = weathering.bump_strength
    bump.inputs["Distance"].default_value = 0.02
    tree.links.new(relief, bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.0


def _tint(tree: bpy.types.NodeTree, low: Color, high: Color, scale: float, detail: float = 5.0) -> bpy.types.NodeSocket:
    mottle = _world_noise(tree, _position(tree), scale=scale, detail=detail)
    return _ramp(tree, mottle, low, high).outputs["Color"]


def _weathered_material(name: str, low: Color, high: Color, weathering: Weathering, scale: float = 4.0) -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    _weather(tree, bsdf, _tint(tree, low, high, scale), weathering)
    return material


MUD: Color = (0.3, 0.22, 0.14, 1.0)

MUDBRICK_WEATHERING = Weathering(roughness=(0.86, 0.97), edge_colour=(0.46, 0.38, 0.27, 1.0), edge_strength=0.5, bump_scale=5.0, bump_strength=0.4)
MUD_PLASTER_WEATHERING = Weathering(roughness=(0.84, 0.96), edge_colour=MUD, edge_strength=0.55, flake_coverage=0.2, bump_scale=3.0, bump_strength=0.3)
WHITEWASH_WEATHERING = Weathering(
    roughness=(0.8, 0.94),
    edge_colour=(0.4, 0.33, 0.23, 1.0),
    edge_strength=0.7,
    flake_coverage=0.35,
    streak_strength=0.12,
    grime_strength=0.65,
    grime_height_m=0.9,
    bump_scale=3.0,
    bump_strength=0.25,
)
LIMESTONE_WEATHERING = Weathering(roughness=(0.72, 0.9), edge_colour=(0.78, 0.74, 0.64, 1.0), edge_strength=0.4, bump_scale=8.0, bump_strength=0.35)
TERRACOTTA_WEATHERING = Weathering(
    roughness=(0.8, 0.94),
    edge_colour=(0.6, 0.42, 0.28, 1.0),
    edge_strength=0.55,
    edge_radius_m=0.03,
    soot_colour=(0.07, 0.05, 0.04, 1.0),
    soot_coverage=0.25,
    dust_strength=0.35,
    bump_scale=10.0,
    bump_strength=0.15,
)
RAW_CLAY_WEATHERING = Weathering(
    roughness=(0.55, 0.85),
    edge_colour=(0.42, 0.35, 0.27, 1.0),
    edge_strength=0.35,
    edge_radius_m=0.03,
    grime_strength=0.3,
    dust_strength=0.3,
    macro_range=(0.8, 1.1),
    bump_scale=7.0,
    bump_strength=0.35,
)
KILN_WEATHERING = Weathering(
    roughness=(0.86, 0.97),
    edge_colour=(0.55, 0.4, 0.28, 1.0),
    edge_strength=0.5,
    soot_colour=(0.05, 0.04, 0.035, 1.0),
    soot_coverage=0.55,
    bump_scale=4.0,
    bump_strength=0.45,
)
REED_WEATHERING = Weathering(roughness=(0.78, 0.92), edge_colour=(0.68, 0.6, 0.42, 1.0), edge_strength=0.4, dust_strength=0.35, bump_scale=12.0, bump_strength=0.2)
AGED_WOOD_WEATHERING = Weathering(
    roughness=(0.72, 0.9),
    edge_colour=(0.5, 0.45, 0.37, 1.0),
    edge_strength=0.55,
    edge_radius_m=0.03,
    bump_scale=10.0,
    bump_strength=0.25,
)
EGYPT_PAINT_WEATHERING = Weathering(
    roughness=(0.78, 0.92),
    edge_colour=(0.56, 0.52, 0.44, 1.0),
    edge_strength=0.85,
    flake_coverage=0.2,
    flake_scale=4.0,
    bump_scale=3.0,
    bump_strength=0.2,
)


def mudbrick_material(name: str = "mudbrick") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _tint(tree, (0.25, 0.18, 0.11, 1.0), (0.36, 0.27, 0.17, 1.0), scale=9.0, detail=4.0)
    alternate = _multiply_colour(tree, tint, _tint(tree, (0.82, 0.8, 0.78, 1.0), (1.0, 1.0, 1.0, 1.0), scale=2.0))

    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.location = (-350.0, 0.0)
    brick.offset = 0.5
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Brick Width"].default_value = 0.36
    brick.inputs["Row Height"].default_value = 0.12
    brick.inputs["Mortar Size"].default_value = 0.012
    brick.inputs["Bias"].default_value = 0.0
    brick.inputs["Mortar"].default_value = (0.4, 0.33, 0.24, 1.0)
    tree.links.new(_wall_coords(tree), brick.inputs["Vector"])
    tree.links.new(tint, brick.inputs["Color1"])
    tree.links.new(alternate, brick.inputs["Color2"])
    _weather(tree, bsdf, brick.outputs["Color"], MUDBRICK_WEATHERING)
    return material


def mud_plaster_material(name: str = "mud_plaster") -> bpy.types.Material:
    return _weathered_material(name, (0.34, 0.26, 0.17, 1.0), (0.44, 0.35, 0.24, 1.0), MUD_PLASTER_WEATHERING, scale=1.5)


def whitewash_material(name: str = "whitewash") -> bpy.types.Material:
    return _weathered_material(name, (0.52, 0.48, 0.4, 1.0), (0.64, 0.6, 0.51, 1.0), WHITEWASH_WEATHERING, scale=1.2)


def limestone_material(name: str = "limestone") -> bpy.types.Material:
    return _weathered_material(name, (0.5, 0.45, 0.35, 1.0), (0.64, 0.59, 0.48, 1.0), LIMESTONE_WEATHERING, scale=3.0)


def terracotta_material(name: str = "terracotta") -> bpy.types.Material:
    return _weathered_material(name, (0.36, 0.15, 0.07, 1.0), (0.5, 0.24, 0.12, 1.0), TERRACOTTA_WEATHERING, scale=5.0)


def raw_clay_material(name: str = "raw_clay") -> bpy.types.Material:
    return _weathered_material(name, (0.2, 0.16, 0.12, 1.0), (0.31, 0.25, 0.19, 1.0), RAW_CLAY_WEATHERING, scale=4.0)


def kiln_clay_material(name: str = "kiln_clay") -> bpy.types.Material:
    return _weathered_material(name, (0.27, 0.15, 0.09, 1.0), (0.42, 0.27, 0.16, 1.0), KILN_WEATHERING, scale=2.5)


def _banded(tree: bpy.types.NodeTree, tint: bpy.types.NodeSocket, direction: str, scale: float, distortion: float, floor: float) -> bpy.types.NodeSocket:
    bands = tree.nodes.new("ShaderNodeTexWave")
    bands.wave_type = "BANDS"
    bands.bands_direction = direction
    bands.inputs["Scale"].default_value = scale
    bands.inputs["Distortion"].default_value = distortion
    bands.inputs["Detail"].default_value = 3.0
    bands_ramp = _ramp(tree, bands.outputs["Fac"], (floor, floor, floor, 1.0), (1.0, 1.0, 1.0, 1.0))
    return _multiply_colour(tree, tint, bands_ramp.outputs["Color"])


def reed_material(name: str = "reed") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _tint(tree, (0.38, 0.3, 0.16, 1.0), (0.52, 0.43, 0.24, 1.0), scale=4.0, detail=2.0)
    _weather(tree, bsdf, _banded(tree, tint, "X", 40.0, 1.5, 0.72), REED_WEATHERING)
    return material


def aged_wood_material(name: str = "aged_wood") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    position = _position(tree)
    grain = _world_noise(tree, position, scale=4.0, detail=6.0, stretch=(6.0, 6.0, 0.5))
    tint = _ramp(tree, grain, (0.18, 0.14, 0.1, 1.0), (0.36, 0.29, 0.21, 1.0)).outputs["Color"]
    _weather(tree, bsdf, tint, AGED_WOOD_WEATHERING)
    return material


def egyptian_paint_material(name: str, colour: Color) -> bpy.types.Material:
    shade = tuple(channel * 0.8 for channel in colour[:3]) + (1.0,)
    return _weathered_material(name, shade, colour, EGYPT_PAINT_WEATHERING, scale=2.0)


@dataclass(frozen=True)
class Stain:
    centre: tuple[float, float]
    radius_m: float
    colour: Color
    strength: float = 0.7


PACKED_EARTH_WEATHERING = Weathering(
    roughness=(0.82, 0.96),
    edge_colour=(0.42, 0.35, 0.25, 1.0),
    edge_strength=0.4,
    grime_strength=0.0,
    dust_strength=0.15,
    streak_strength=0.0,
    macro_range=(0.8, 1.1),
    bump_scale=6.0,
    bump_strength=0.35,
)


def _voronoi(tree: bpy.types.NodeTree, position: bpy.types.NodeSocket, scale: float, feature: str = "F1") -> bpy.types.ShaderNodeTexVoronoi:
    node = tree.nodes.new("ShaderNodeTexVoronoi")
    node.feature = feature
    node.inputs["Scale"].default_value = scale
    tree.links.new(position, node.inputs["Vector"])
    return node


def _stain_mask(tree: bpy.types.NodeTree, position: bpy.types.NodeSocket, stain: Stain) -> bpy.types.NodeSocket:
    offset = tree.nodes.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    offset.inputs[1].default_value = (stain.centre[0], stain.centre[1], 0.0)
    tree.links.new(position, offset.inputs[0])
    flatten = tree.nodes.new("ShaderNodeVectorMath")
    flatten.operation = "MULTIPLY"
    flatten.inputs[1].default_value = (1.0, 1.0, 0.0)
    tree.links.new(offset.outputs["Vector"], flatten.inputs[0])
    distance = tree.nodes.new("ShaderNodeVectorMath")
    distance.operation = "LENGTH"
    tree.links.new(flatten.outputs["Vector"], distance.inputs[0])
    ragged = _map_range(tree, _world_noise(tree, position, scale=1.8, detail=5.0), 0.3, 0.7, -0.35, 0.35)
    reach = _math(tree, "ADD", _map_range(tree, distance.outputs["Value"], 0.0, stain.radius_m, 1.0, 0.0), ragged)
    return _math(tree, "MULTIPLY", reach, stain.strength)


def packed_earth_material(name: str = "packed_earth", stains: Sequence[Stain] = ()) -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    position = _position(tree)
    broad = _world_noise(tree, position, scale=0.25, detail=3.0)
    mottle = _world_noise(tree, position, scale=1.5, detail=6.0)
    blend = _math(tree, "ADD", _math(tree, "MULTIPLY", broad, 0.6), _math(tree, "MULTIPLY", mottle, 0.4))
    colour = _ramp(tree, _map_range(tree, blend, 0.35, 0.65), (0.25, 0.19, 0.13, 1.0), (0.42, 0.34, 0.23, 1.0)).outputs["Color"]

    patches = _voronoi(tree, position, scale=1.1)
    patch_value = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(patches.outputs["Color"], patch_value.inputs["Color"])
    patch_shade = _map_range(tree, patch_value.outputs["Red"], 0.0, 1.0, 0.88, 1.06)
    patch_colour = tree.nodes.new("ShaderNodeCombineColor")
    for channel in ("Red", "Green", "Blue"):
        tree.links.new(patch_shade, patch_colour.inputs[channel])
    colour = _multiply_colour(tree, colour, patch_colour.outputs["Color"])

    for stain in stains:
        colour = _mix_colour(tree, colour, stain.colour, _stain_mask(tree, position, stain))

    wobble = _noise(tree, scale=1.2, detail=4.0)
    tree.links.new(position, wobble.inputs["Vector"])
    warp = tree.nodes.new("ShaderNodeVectorMath")
    warp.operation = "MULTIPLY_ADD"
    warp.inputs[1].default_value = (0.35, 0.35, 0.35)
    tree.links.new(wobble.outputs["Color"], warp.inputs[0])
    tree.links.new(position, warp.inputs[2])
    crack_cells = _voronoi(tree, warp.outputs["Vector"], scale=1.6, feature="DISTANCE_TO_EDGE")
    crack_lines = _map_range(tree, crack_cells.outputs["Distance"], 0.03, 0.0)
    crack_zones = _map_range(tree, _world_noise(tree, position, scale=0.45, detail=3.0), 0.56, 0.64)
    cracks = _math(tree, "MULTIPLY", crack_lines, crack_zones)
    colour = _mix_colour(tree, colour, (0.12, 0.09, 0.06, 1.0), _math(tree, "MULTIPLY", cracks, 0.8))

    grit = _voronoi(tree, position, scale=14.0)
    grit_pick = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(grit.outputs["Color"], grit_pick.inputs["Color"])
    stones = _math(tree, "MULTIPLY", _map_range(tree, grit.outputs["Distance"], 0.32, 0.22), _map_range(tree, grit_pick.outputs["Green"], 0.84, 0.88))
    light_stone = _map_range(tree, grit_pick.outputs["Blue"], 0.45, 0.55)
    dark_stone = _map_range(tree, grit_pick.outputs["Blue"], 0.55, 0.45)
    colour = _mix_colour(tree, colour, (0.17, 0.13, 0.09, 1.0), _math(tree, "MULTIPLY", stones, dark_stone))
    colour = _mix_colour(tree, colour, (0.47, 0.41, 0.31, 1.0), _math(tree, "MULTIPLY", stones, _math(tree, "MULTIPLY", light_stone, 0.7)))

    _weather(tree, bsdf, colour, PACKED_EARTH_WEATHERING, recess=_math(tree, "SUBTRACT", _math(tree, "MULTIPLY", cracks, 0.8), _math(tree, "MULTIPLY", stones, 0.4)))
    return material


COPPER_WEATHERING = Weathering(
    roughness=(0.38, 0.62),
    edge_colour=(0.78, 0.45, 0.27, 1.0),
    edge_strength=0.7,
    edge_radius_m=0.02,
    grime_strength=0.35,
    dust_strength=0.25,
    soot_colour=(0.08, 0.05, 0.04, 1.0),
    soot_coverage=0.2,
    bump_scale=14.0,
    bump_strength=0.15,
)
CHARCOAL_WEATHERING = Weathering(
    roughness=(0.7, 0.9),
    edge_colour=(0.16, 0.15, 0.14, 1.0),
    edge_strength=0.5,
    edge_radius_m=0.02,
    grime_strength=0.0,
    dust_colour=(0.3, 0.27, 0.23, 1.0),
    dust_strength=0.2,
    streak_strength=0.0,
    macro_range=(0.75, 1.15),
    bump_scale=18.0,
    bump_strength=0.6,
)
SLAG_WEATHERING = Weathering(
    roughness=(0.35, 0.75),
    edge_colour=(0.24, 0.2, 0.18, 1.0),
    edge_strength=0.45,
    edge_radius_m=0.02,
    grime_strength=0.0,
    dust_strength=0.3,
    streak_strength=0.0,
    bump_scale=12.0,
    bump_strength=0.55,
)
MALACHITE_WEATHERING = Weathering(
    roughness=(0.62, 0.86),
    edge_colour=(0.3, 0.52, 0.38, 1.0),
    edge_strength=0.45,
    edge_radius_m=0.02,
    grime_strength=0.2,
    dust_strength=0.35,
    streak_strength=0.0,
    bump_scale=16.0,
    bump_strength=0.5,
)
LEATHER_WEATHERING = Weathering(
    roughness=(0.55, 0.8),
    edge_colour=(0.42, 0.3, 0.2, 1.0),
    edge_strength=0.55,
    edge_radius_m=0.03,
    soot_colour=(0.08, 0.06, 0.05, 1.0),
    soot_coverage=0.2,
    bump_scale=9.0,
    bump_strength=0.3,
)
COPPER: Color = (0.56, 0.28, 0.15, 1.0)
VERDIGRIS: Color = (0.2, 0.42, 0.34, 1.0)


def aged_copper_material(name: str = "aged_copper") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    position = _position(tree)
    tint = _tint(tree, (0.38, 0.17, 0.09, 1.0), COPPER, scale=6.0)
    patina = _map_range(tree, _world_noise(tree, position, scale=3.5, detail=6.0), 0.58, 0.68, 0.0, 0.85)
    _weather(tree, bsdf, _mix_colour(tree, tint, VERDIGRIS, patina), COPPER_WEATHERING)
    tree.links.new(_map_range(tree, patina, 0.0, 0.85, 0.9, 0.1), bsdf.inputs["Metallic"])
    return material


def charcoal_material(name: str = "charcoal") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    lumps = _voronoi(tree, _position(tree), scale=22.0, feature="DISTANCE_TO_EDGE")
    seams = _ramp(tree, _map_range(tree, lumps.outputs["Distance"], 0.0, 0.08), (0.02, 0.02, 0.02, 1.0), (0.09, 0.085, 0.08, 1.0)).outputs["Color"]
    _weather(tree, bsdf, _multiply_colour(tree, seams, _tint(tree, (0.8, 0.8, 0.8, 1.0), (1.0, 1.0, 1.0, 1.0), scale=8.0)), CHARCOAL_WEATHERING)
    return material


def slag_material(name: str = "slag") -> bpy.types.Material:
    return _weathered_material(name, (0.06, 0.05, 0.05, 1.0), (0.2, 0.13, 0.1, 1.0), SLAG_WEATHERING, scale=9.0)


def malachite_material(name: str = "malachite") -> bpy.types.Material:
    material, tree, bsdf = _bsdf_material(name)
    tint = _tint(tree, (0.05, 0.22, 0.14, 1.0), (0.16, 0.42, 0.28, 1.0), scale=12.0)
    rock = _map_range(tree, _world_noise(tree, _position(tree), scale=5.0, detail=5.0), 0.5, 0.62)
    _weather(tree, bsdf, _mix_colour(tree, tint, (0.36, 0.3, 0.22, 1.0), rock), MALACHITE_WEATHERING)
    return material


def leather_material(name: str = "leather") -> bpy.types.Material:
    return _weathered_material(name, (0.2, 0.12, 0.07, 1.0), (0.34, 0.21, 0.12, 1.0), LEATHER_WEATHERING, scale=3.0)


EGYPTIAN_BLUE: Color = (0.08, 0.2, 0.3, 1.0)
OCHRE_RED: Color = (0.38, 0.13, 0.06, 1.0)


MATERIAL_BUILDERS: dict[str, Callable[[str], bpy.types.Material]] = {
    "mudbrick": mudbrick_material,
    "mud_plaster": mud_plaster_material,
    "limestone": limestone_material,
    "terracotta": terracotta_material,
    "raw_clay": raw_clay_material,
    "kiln_clay": kiln_clay_material,
    "reed": reed_material,
    "whitewash": whitewash_material,
    "aged_wood": aged_wood_material,
    "packed_earth": packed_earth_material,
    "aged_copper": aged_copper_material,
    "charcoal": charcoal_material,
    "slag": slag_material,
    "malachite": malachite_material,
    "leather": leather_material,
    "paint_egyptian_blue": lambda name: egyptian_paint_material(name, EGYPTIAN_BLUE),
    "paint_ochre_red": lambda name: egyptian_paint_material(name, OCHRE_RED),
    "concrete": concrete_material,
    "plaster": plaster_material,
    "brick": brick_material,
    "metal": metal_material,
    "glass": glass_material,
    "roofing": roofing_material,
    "wood": wood_material,
    "paint": paint_material,
    "paint_charcoal": lambda name: paint_material(name, PAINT_CHARCOAL),
    "paint_cream": lambda name: paint_material(name, PAINT_CREAM),
    "lens_amber": lambda name: paint_material(name, LENS_AMBER),
    "lens_red": lambda name: paint_material(name, LENS_RED),
    "rubber": rubber_material,
    "tinted_glass": tinted_glass_material,
}


def load_cc0_index(path: Path = CC0_INDEX_FILE) -> dict[str, dict]:
    """Read the plain-JSON asset material index (stdlib-only; no library.materials import, which
    pulls in requests through library.polyhaven and isn't guaranteed present in Blender's own
    interpreter)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("materials", {})


BOX_PROJECTION_BLEND = 0.2


def _cc0_image(tree: bpy.types.NodeTree, path: Path, non_color: bool, location: tuple[float, float]) -> bpy.types.ShaderNodeTexImage:
    node = tree.nodes.new("ShaderNodeTexImage")
    node.location = location
    node.image = bpy.data.images.load(str(path), check_existing=True)
    node.projection = "BOX"
    node.projection_blend = BOX_PROJECTION_BLEND
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
    tree.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

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
