"""Runs inside Blender's own interpreter via `blender --background --python`.

Not part of the assetforge package import graph - Blender's bundled Python has no access to
our venv (torch/numpy/pillow), so this script only ever touches bpy, mathutils and the stdlib.
Invoked as: blender --background --factory-startup --python beauty_render.py -- <args.json>
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

CHANNEL_LABEL = {"r": "Red", "g": "Green", "b": "Blue"}


def _read_args() -> dict:
    argv = sys.argv
    separator = argv.index("--")
    return json.loads(Path(argv[separator + 1]).read_text(encoding="utf-8"))


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _load_image(path: str, non_color: bool = True) -> bpy.types.Image:
    image = bpy.data.images.load(path, check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    return image


def _set_cycles_device(requested: str) -> dict:
    """Force Cycles onto the named compute device explicitly - never inherit user prefs.

    Left to user preferences, a headless run silently falls back to CPU and a render that
    should take seconds on the RTX 4070 instead takes minutes.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = requested
    prefs.get_devices()
    enabled = []
    for device in prefs.devices:
        device.use = device.type == requested
        if device.use:
            enabled.append(device.name)
    if enabled:
        scene.cycles.device = "GPU"
        return {"device": requested, "gpus": enabled}
    scene.cycles.device = "CPU"
    print(f"WARNING: no {requested} device found, falling back to CPU explicitly", file=sys.stderr)
    return {"device": "CPU", "gpus": []}


def _build_world() -> None:
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.55, 0.68, 0.85, 1.0)
    background.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def _build_sun(sun_args: dict) -> None:
    light = bpy.data.lights.new("Sun", type="SUN")
    light.energy = float(sun_args.get("energy", 4.0))
    light.angle = math.radians(1.0)
    obj = bpy.data.objects.new("Sun", light)
    bpy.context.collection.objects.link(obj)
    altitude = math.radians(float(sun_args.get("altitude_deg", 45.0)))
    azimuth = math.radians(float(sun_args.get("azimuth_deg", 315.0)))
    obj.rotation_euler = (math.pi / 2.0 - altitude, 0.0, azimuth)


def _build_terrain(args: dict) -> bpy.types.Object:
    size = float(args["world_size_m"])
    grid_resolution = int(args["grid_resolution"])
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=grid_resolution,
        y_subdivisions=grid_resolution,
        size=size,
        calc_uvs=True,
        location=(size / 2.0, size / 2.0, 0.0),
    )
    terrain = bpy.context.active_object
    terrain.name = "Terrain"

    height_texture = bpy.data.textures.new("HeightDisplace", type="IMAGE")
    height_texture.image = _load_image(args["height_png"])
    height_texture.extension = "EXTEND"

    displace = terrain.modifiers.new("Displace", "DISPLACE")
    displace.texture = height_texture
    displace.texture_coords = "UV"
    displace.direction = "Z"
    displace.mid_level = 0.0
    displace.strength = float(args["height_range_m"])
    return terrain


def _splat_weight_socket(tree, splat_nodes: dict, texture: int, channel: str):
    image_node, separate_node = splat_nodes[str(texture)]
    if channel == "a":
        return image_node.outputs["Alpha"]
    return separate_node.outputs[CHANNEL_LABEL[channel]]


def _build_material(args: dict) -> bpy.types.Material:
    material = bpy.data.materials.new("TerrainSplat")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tex_coord = tree.nodes.new("ShaderNodeTexCoord")

    splat_nodes = {}
    for key, path in args["splat_textures"].items():
        image_node = tree.nodes.new("ShaderNodeTexImage")
        image_node.image = _load_image(path)
        image_node.interpolation = "Linear"
        tree.links.new(tex_coord.outputs["UV"], image_node.inputs["Vector"])
        separate_node = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(image_node.outputs["Color"], separate_node.inputs["Color"])
        splat_nodes[key] = (image_node, separate_node)

    world_size = float(args["world_size_m"])
    materials_meta = args["materials"]

    shaders: list[tuple] = []
    for layer in args["layers"]:
        meta = materials_meta[layer["material"]]
        scale = world_size / float(meta["tiling_m"])

        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (scale, scale, scale)
        tree.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

        albedo = tree.nodes.new("ShaderNodeTexImage")
        albedo.image = _load_image(meta["albedo"], non_color=False)
        tree.links.new(mapping.outputs["Vector"], albedo.inputs["Vector"])

        normal_tex = tree.nodes.new("ShaderNodeTexImage")
        normal_tex.image = _load_image(meta["normal"])
        tree.links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])

        roughness_tex = tree.nodes.new("ShaderNodeTexImage")
        roughness_tex.image = _load_image(meta["roughness"])
        tree.links.new(mapping.outputs["Vector"], roughness_tex.inputs["Vector"])

        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        tree.links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])
        tree.links.new(roughness_tex.outputs["Color"], bsdf.inputs["Roughness"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

        weight_socket = _splat_weight_socket(tree, splat_nodes, layer["texture"], layer["channel"])
        shaders.append((bsdf.outputs["BSDF"], weight_socket))

    blended_shader, running_sum = shaders[0]
    for shader_socket, weight_socket in shaders[1:]:
        total = tree.nodes.new("ShaderNodeMath")
        total.operation = "ADD"
        tree.links.new(running_sum, total.inputs[0])
        tree.links.new(weight_socket, total.inputs[1])

        safe_total = tree.nodes.new("ShaderNodeMath")
        safe_total.operation = "MAXIMUM"
        safe_total.inputs[1].default_value = 1e-6
        tree.links.new(total.outputs["Value"], safe_total.inputs[0])

        factor = tree.nodes.new("ShaderNodeMath")
        factor.operation = "DIVIDE"
        tree.links.new(weight_socket, factor.inputs[0])
        tree.links.new(safe_total.outputs["Value"], factor.inputs[1])

        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(factor.outputs["Value"], mix.inputs["Fac"])
        tree.links.new(blended_shader, mix.inputs[1])
        tree.links.new(shader_socket, mix.inputs[2])

        blended_shader = mix.outputs["Shader"]
        running_sum = total.outputs["Value"]

    tree.links.new(blended_shader, output.inputs["Surface"])
    return material


def _ground_height(depsgraph, x: float, y: float) -> float:
    origin = Vector((x, y, 1.0e6))
    direction = Vector((0.0, 0.0, -1.0))
    hit, location, _normal, _index, _obj, _matrix = bpy.context.scene.ray_cast(
        depsgraph, origin, direction
    )
    return location.z if hit else 0.0


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _render(scene, resolution_x: int, resolution_y: int, samples: int, filepath: str) -> None:
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.cycles.samples = samples
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = _read_args()
    _clear_scene()

    device_info = _set_cycles_device(args.get("render", {}).get("device", "OPTIX"))
    _build_world()
    _build_sun(args.get("sun", {}))
    terrain = _build_terrain(args)
    terrain.data.materials.append(_build_material(args))

    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("BeautyCam")
    camera = bpy.data.objects.new("BeautyCam", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_size = float(args["world_size_m"])
    camera_args = args["camera"]
    centre_frac = camera_args.get("centre_frac", [0.5, 0.5])
    centre_x = float(centre_frac[0]) * world_size
    centre_y = float(centre_frac[1]) * world_size
    centre_z = _ground_height(depsgraph, centre_x, centre_y)
    target = Vector((centre_x, centre_y, centre_z))
    patch = float(camera_args.get("patch_size_m", 200.0))

    render_args = args.get("render", {})
    samples = int(render_args.get("samples", 128))
    resolution_x = int(render_args.get("resolution_x", 960))
    resolution_y = int(render_args.get("resolution_y", 540))
    outputs = args["output"]

    three_quarter = camera_args.get("three_quarter", {})
    height_range = float(args["height_range_m"])
    distance = max(patch * float(three_quarter.get("distance_factor", 1.6)), height_range * 1.2)
    azimuth = math.radians(float(three_quarter.get("azimuth_deg", 45.0)))
    altitude = math.radians(float(three_quarter.get("altitude_deg", 35.0)))
    offset = Vector(
        (math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude))
    ) * distance
    camera.location = target + offset
    _look_at(camera, target)
    _render(scene, resolution_x, resolution_y, samples, outputs["three_quarter"])

    ground = camera_args.get("ground", {})
    look_azimuth = math.radians(float(ground.get("look_azimuth_deg", three_quarter.get("azimuth_deg", 45.0))))
    back_distance = float(ground.get("back_distance_m", max(10.0, patch * 0.25)))
    ground_x = centre_x - math.cos(look_azimuth) * back_distance
    ground_y = centre_y - math.sin(look_azimuth) * back_distance
    ground_z = _ground_height(depsgraph, ground_x, ground_y)
    eye_height = float(ground.get("eye_height_m", 1.8))
    camera.location = Vector((ground_x, ground_y, ground_z + eye_height))
    _look_at(camera, Vector((centre_x, centre_y, centre_z + eye_height * 0.5)))
    _render(scene, resolution_x, resolution_y, samples, outputs["ground"])

    result = {"device": device_info, "centre_world_m": [centre_x, centre_y, centre_z]}
    Path(args["result_json"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
