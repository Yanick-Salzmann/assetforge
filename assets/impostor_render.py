"""Render a vegetation prop's hemi-octahedral impostor atlas and its LOD review sheet in Cycles.

Runs inside Blender. The whole atlas is one render: every frame is a linked copy of LOD0 rotated
by assets.impostor.frame_rotation and moved to its cell, seen from above by one orthographic
camera. Two passes share the copies through a view-layer material override - unlit albedo with
the alpha cutout (Standard view transform, so the texture's sRGB bytes come back unchanged) and
the shading normal in glTF object axes encoded as n * 0.5 + 0.5 (Raw view transform). The colour
of both is bled into the transparent texels so mips and bilinear taps never darken an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bpy
import numpy as np
from mathutils import Matrix, Vector

from assets import impostor
from assets.foliage_atlas import bleed

SAMPLES = 16
TRANSPARENT_BOUNCES = 128
SHEET_PX = 512
SHEET_VIEW: tuple[float, float, float] = (0.6, -0.6, 0.53)
SHEET_BACKGROUND = (0.42, 0.45, 0.5, 1.0)
ALBEDO_NAME = "impostor_albedo.png"
NORMAL_NAME = "impostor_normal.png"


@dataclass(frozen=True)
class Sphere:
    centre: Vector
    radius: float


def bounding_sphere(obj: bpy.types.Object) -> Sphere:
    points = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)
    centre = (points.min(axis=0) + points.max(axis=0)) * 0.5
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    return Sphere(Vector(centre), radius)


def _scene(name: str, size_px: tuple[int, int], transparent: bool) -> bpy.types.Scene:
    scene = bpy.data.scenes.new(name)
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    gpus = [device for device in prefs.devices if device.type == "OPTIX"]
    for device in prefs.devices:
        device.use = device in gpus
    scene.cycles.device = "GPU" if gpus else "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.transparent_max_bounces = TRANSPARENT_BOUNCES
    scene.render.resolution_x, scene.render.resolution_y = size_px
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    return scene


def _camera(scene: bpy.types.Scene, rotation: Matrix, location: Vector, ortho_scale: float) -> None:
    data = bpy.data.cameras.new(f"{scene.name}_camera")
    data.type = "ORTHO"
    data.ortho_scale = ortho_scale
    data.clip_start = 0.01
    data.clip_end = 10000.0
    camera = bpy.data.objects.new(f"{scene.name}_camera", data)
    camera.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()
    scene.collection.objects.link(camera)
    scene.camera = camera


def _cutout(tree: bpy.types.NodeTree, image: bpy.types.Image, colour_socket) -> None:
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    cutout = tree.nodes.new("ShaderNodeMath")
    cutout.operation = "ROUND"
    emission = tree.nodes.new("ShaderNodeEmission")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    mix = tree.nodes.new("ShaderNodeMixShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(texture.outputs["Alpha"], cutout.inputs[0])
    tree.links.new(cutout.outputs["Value"], mix.inputs["Fac"])
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    tree.links.new(colour_socket(texture), emission.inputs["Color"])


def unlit_material(name: str, image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    _cutout(material.node_tree, image, lambda texture: texture.outputs["Color"])
    return material


def normal_material(name: str, image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    def encoded(_texture):
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        local = tree.nodes.new("ShaderNodeVectorTransform")
        local.vector_type = "NORMAL"
        local.convert_from = "WORLD"
        local.convert_to = "OBJECT"
        split = tree.nodes.new("ShaderNodeSeparateXYZ")
        negate = tree.nodes.new("ShaderNodeMath")
        negate.operation = "MULTIPLY"
        negate.inputs[1].default_value = -1.0
        gltf = tree.nodes.new("ShaderNodeCombineXYZ")
        scale = tree.nodes.new("ShaderNodeVectorMath")
        scale.operation = "MULTIPLY_ADD"
        scale.inputs[1].default_value = (0.5, 0.5, 0.5)
        scale.inputs[2].default_value = (0.5, 0.5, 0.5)
        tree.links.new(geometry.outputs["Normal"], local.inputs["Vector"])
        tree.links.new(local.outputs["Vector"], split.inputs["Vector"])
        tree.links.new(split.outputs["X"], gltf.inputs["X"])
        tree.links.new(split.outputs["Z"], gltf.inputs["Y"])
        tree.links.new(split.outputs["Y"], negate.inputs[0])
        tree.links.new(negate.outputs["Value"], gltf.inputs["Z"])
        tree.links.new(gltf.outputs["Vector"], scale.inputs[0])
        return scale.outputs["Vector"]

    _cutout(tree, image, encoded)
    return material


def _render(scene: bpy.types.Scene, path: Path, view_transform: str) -> None:
    scene.view_settings.view_transform = view_transform
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=scene.name)


def _bleed_file(path: Path) -> None:
    image = bpy.data.images.load(str(path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bled = bleed(pixels.reshape(height, width, 4))
    image.pixels.foreach_set(bled.ravel())
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def _remove_scene(scene: bpy.types.Scene) -> None:
    for obj in list(scene.collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.scenes.remove(scene)


def render_atlas(
    lod0: bpy.types.Object,
    image: bpy.types.Image,
    textures_dir: Path,
    frames: int = impostor.FRAMES,
    frame_px: int = impostor.FRAME_PX,
) -> Sphere:
    """Write textures_dir/impostor_albedo.png and impostor_normal.png for lod0 (identity transform,
    Blender axes) and return the bounding sphere every frame is framed on."""
    sphere = bounding_sphere(lod0)
    cell = 2.0 * sphere.radius
    half = frames * sphere.radius
    scene = _scene(f"{lod0.name}_impostor", (frames * frame_px, frames * frame_px), transparent=True)
    for column in range(frames):
        for row in range(frames):
            direction = impostor.gltf_to_blender(impostor.frame_direction(column, row, frames))
            rotation = Matrix(impostor.frame_rotation(direction).tolist())
            copy = bpy.data.objects.new(f"{lod0.name}_frame_{column}_{row}", lod0.data)
            target = Vector(((column + 0.5) * cell - half, (row + 0.5) * cell - half, 0.0))
            copy.matrix_world = Matrix.Translation(target) @ rotation.to_4x4() @ Matrix.Translation(-sphere.centre)
            scene.collection.objects.link(copy)
    _camera(scene, Matrix.Identity(3), Vector((0.0, 0.0, 2.0 * sphere.radius + 1.0)), frames * cell)
    textures_dir.mkdir(parents=True, exist_ok=True)
    layer = scene.view_layers[0]
    layer.material_override = unlit_material(f"{lod0.name}_impostor_albedo", image)
    _render(scene, textures_dir / ALBEDO_NAME, "Standard")
    layer.material_override = normal_material(f"{lod0.name}_impostor_normal", image)
    _render(scene, textures_dir / NORMAL_NAME, "Raw")
    _remove_scene(scene)
    for name in (ALBEDO_NAME, NORMAL_NAME):
        _bleed_file(textures_dir / name)
    return sphere


def _impostor_quad(name: str, sphere: Sphere, direction: Vector, frames: int, image: bpy.types.Image) -> bpy.types.Object:
    """A camera-facing quad over sphere showing the atlas frame nearest direction (Blender axes),
    exactly as a viewer would draw the impostor."""
    u, v = impostor.encode(impostor.blender_to_gltf(tuple(direction)))
    column = min(int((u + 1.0) * 0.5 * frames), frames - 1)
    row = min(int((v + 1.0) * 0.5 * frames), frames - 1)
    right, up = impostor.view_basis(tuple(direction))
    r = sphere.radius
    corners = [sphere.centre + Vector(right) * sx * r + Vector(up) * sy * r for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
    layer = mesh.uv_layers.new(name="UVMap")
    u0 = column / frames
    v0 = row / frames
    step = 1.0 / frames
    layer.data.foreach_set("uv", [u0, v0, u0 + step, v0, u0 + step, v0 + step, u0, v0 + step])
    mesh.materials.append(unlit_material(f"{name}_material", image))
    return bpy.data.objects.new(name, mesh)


def render_lod_sheet(
    lods: Sequence[bpy.types.Object],
    atlas: bpy.types.Image,
    impostor_albedo: Path | None,
    sphere: Sphere,
    path: Path,
    frames: int = impostor.FRAMES,
) -> Path:
    """One unlit three-quarter row, side by side at one scale: each LOD, then the impostor quad
    when there is one. Unlit on both sides, so only silhouette and coverage can differ."""
    direction = Vector(SHEET_VIEW).normalized()
    right, up = (Vector(axis) for axis in impostor.view_basis(tuple(direction)))
    columns = len(lods) + (1 if impostor_albedo is not None else 0)
    spacing = 2.1 * sphere.radius
    scene = _scene(f"{lods[0].name}_lod_sheet", (columns * SHEET_PX, SHEET_PX), transparent=False)
    world = bpy.data.worlds.new(f"{scene.name}_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = SHEET_BACKGROUND
    scene.world = world
    shared = unlit_material(f"{scene.name}_unlit", atlas)
    offsets = [right * spacing * (index - (columns - 1) * 0.5) for index in range(columns)]
    for lod, offset in zip(lods, offsets):
        copy = bpy.data.objects.new(f"{lod.name}_sheet", lod.data)
        copy.matrix_world = Matrix.Translation(offset)
        copy.material_slots[0].link = "OBJECT"
        copy.material_slots[0].material = shared
        scene.collection.objects.link(copy)
    if impostor_albedo is not None:
        image = bpy.data.images.load(str(impostor_albedo))
        quad = _impostor_quad(f"{lods[0].name}_impostor_quad", sphere, direction, frames, image)
        quad.matrix_world = Matrix.Translation(offsets[-1])
        scene.collection.objects.link(quad)
    look = Matrix((right, up, direction)).transposed()
    _camera(scene, look, sphere.centre + direction * 4.0 * sphere.radius, columns * spacing)
    _render(scene, path, "Standard")
    _remove_scene(scene)
    return path
