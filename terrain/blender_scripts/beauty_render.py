"""Runs inside Blender's own interpreter via `blender --background --python`.

Not part of the assetforge package import graph - Blender's bundled Python has no access to
our venv (torch/numpy/pillow), so this script only ever touches bpy, mathutils and the stdlib.
Invoked as: blender --background --factory-startup --python beauty_render.py -- <args.json>
"""

import array
import json
import math
import random
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector

TRANSPARENT_BOUNCES = 64
REDUCED_LOD = re.compile(r"_LOD[1-9]\d*(\.\d+)?$")
REDUCED_BUILDING_LOD = re.compile(r"_LOD[1-9]\d*(_ANIM)?(\.\d+)?$")
CHANNEL_LABEL = {"r": "Red", "g": "Green", "b": "Blue"}
RECORD_FIELDS = 7
PLUME_RINGS = 24
PLUME_SEGMENTS = 16
PLUME_NOISE_SCALE = 1.2
GRAVITY_M_S2 = 9.81
SPARK_CONE_RAD = 0.6


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


def _build_world(sky_args: dict) -> None:
    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    tree = world.node_tree
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = True
    sky.sun_elevation = float(sky_args["sun_elevation_rad"])
    sky.sun_rotation = float(sky_args["sun_rotation_rad"])
    sky.air_density = float(sky_args["air_density"])
    sky.aerosol_density = float(sky_args["aerosol_density"])
    background = tree.nodes["Background"]
    tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def _build_moon(moon_args: dict) -> None:
    """A full-moon sun lamp, present only when the moon is the key light."""
    energy = float(moon_args.get("energy", 0.0))
    if energy <= 0.0:
        return
    light = bpy.data.lights.new("Moon", type="SUN")
    light.energy = energy
    light.color = tuple(moon_args["colour"])
    light.angle = math.radians(0.5)
    obj = bpy.data.objects.new("Moon", light)
    bpy.context.collection.objects.link(obj)
    toward_light = -Vector(moon_args["direction"])
    obj.rotation_euler = toward_light.to_track_quat("-Z", "Y").to_euler()


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


def _variant_collection(entry: dict) -> bpy.types.Collection:
    """The meshes one scatter variant instances: its GLB, or a kind-sized stand-in when absent."""
    collection = bpy.data.collections.new(f"Scatter_{entry['name']}")
    bpy.context.scene.collection.children.link(collection)
    before = set(bpy.data.objects)
    if entry.get("glb"):
        bpy.ops.import_scene.gltf(filepath=entry["glb"])
    else:
        size = float(entry["size_m"])
        if entry["kind"] in ("conifer", "cactus", "grass", "flower"):
            bpy.ops.mesh.primitive_cone_add(
                vertices=8, radius1=size * 0.25, depth=size, location=(0.0, 0.0, size / 2.0)
            )
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=1, radius=size / 2.0, location=(0.0, 0.0, size / 2.0)
            )
    for obj in set(bpy.data.objects) - before:
        if REDUCED_LOD.search(obj.name):
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    bpy.context.view_layer.layer_collection.children[collection.name].exclude = True
    return collection


def _instancer_group(collection: bpy.types.Collection) -> bpy.types.NodeTree:
    group = bpy.data.node_groups.new(f"Instancer_{collection.name}", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    collection_info = nodes.new("GeometryNodeCollectionInfo")
    collection_info.transform_space = "ORIGINAL"
    collection_info.inputs["Collection"].default_value = collection
    rotation = nodes.new("GeometryNodeInputNamedAttribute")
    rotation.data_type = "FLOAT_VECTOR"
    rotation.inputs["Name"].default_value = "instance_rotation"
    euler = nodes.new("FunctionNodeEulerToRotation")
    scale = nodes.new("GeometryNodeInputNamedAttribute")
    scale.data_type = "FLOAT"
    scale.inputs["Name"].default_value = "instance_scale"
    instancer = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(group_in.outputs[0], instancer.inputs["Points"])
    links.new(collection_info.outputs[0], instancer.inputs["Instance"])
    links.new(rotation.outputs["Attribute"], euler.inputs["Euler"])
    links.new(euler.outputs["Rotation"], instancer.inputs["Rotation"])
    links.new(scale.outputs["Attribute"], instancer.inputs["Scale"])
    links.new(instancer.outputs["Instances"], group_out.inputs[0])
    return group


def _build_scatter(entries: list) -> None:
    """One point cloud per species variant, instancing that variant's meshes by geometry nodes."""
    if not entries:
        return
    groups = {}
    for entry in entries:
        key = entry.get("glb") or f"primitive:{entry['kind']}"
        if key not in groups:
            groups[key] = _instancer_group(_variant_collection(entry))
        values = array.array("f")
        values.frombytes(Path(entry["records"]).read_bytes())
        count = len(values) // RECORD_FIELDS
        mesh = bpy.data.meshes.new(f"Points_{entry['name']}")
        mesh.vertices.add(count)
        mesh.vertices.foreach_set("co", [values[RECORD_FIELDS * i + axis] for i in range(count) for axis in range(3)])
        rotation = mesh.attributes.new("instance_rotation", "FLOAT_VECTOR", "POINT")
        rotation.data.foreach_set(
            "vector", [values[RECORD_FIELDS * i + 3 + axis] for i in range(count) for axis in range(3)]
        )
        scale = mesh.attributes.new("instance_scale", "FLOAT", "POINT")
        scale.data.foreach_set("value", values[6::RECORD_FIELDS])
        obj = bpy.data.objects.new(f"Scatter_{entry['name']}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        modifier = obj.modifiers.new("Instancer", "NODES")
        modifier.node_group = groups[key]


def _reduced(obj: bpy.types.Object) -> bool:
    node = obj
    while node is not None:
        if REDUCED_BUILDING_LOD.search(node.name):
            return True
        node = node.parent
    return False


def _socket(objects, node: str):
    for obj in objects:
        if obj.name == node or obj.name.startswith(f"{node}."):
            return obj
    return None


def _plume_path(fx: dict, up: Vector) -> list:
    """Where a particle born at the socket has got to at each step of its mean life: rising at the
    profile's speed along the emit axis while its horizontal velocity relaxes toward the wind."""
    life = float(fx["life_s"])
    carry = float(fx["carry"])
    downwind = Vector((fx["downwind"][0], fx["downwind"][1], 0.0))
    start = float(fx["radius_start_m"])
    end = float(fx["radius_end_m"])
    path = []
    for ring in range(PLUME_RINGS + 1):
        fraction = ring / PLUME_RINGS
        age = life * fraction
        drift = 0.0
        if carry > 0.0:
            drift = float(fx["wind_m_s"]) * (age - (1.0 - math.exp(-carry * age)) / carry)
        point = up * float(fx["rise_m_s"]) * age + downwind * drift
        path.append((point, max(start + (end - start) * fraction, 0.01)))
    return path


def _tube(name: str, path: list) -> bpy.types.Mesh:
    """A closed tube of rings around the path, capped at both ends, for a volume shader."""
    vertices = []
    faces = []
    for index, (point, radius) in enumerate(path):
        ahead = path[min(index + 1, len(path) - 1)][0] - path[max(index - 1, 0)][0]
        tangent = ahead.normalized() if ahead.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        reference = Vector((1.0, 0.0, 0.0)) if abs(tangent.x) < 0.9 else Vector((0.0, 1.0, 0.0))
        side = tangent.cross(reference).normalized()
        other = tangent.cross(side)
        for segment in range(PLUME_SEGMENTS):
            angle = 2.0 * math.pi * segment / PLUME_SEGMENTS
            vertices.append(tuple(point + (side * math.cos(angle) + other * math.sin(angle)) * radius))
    for ring in range(len(path) - 1):
        for segment in range(PLUME_SEGMENTS):
            a = ring * PLUME_SEGMENTS + segment
            b = ring * PLUME_SEGMENTS + (segment + 1) % PLUME_SEGMENTS
            faces.append((a, b, b + PLUME_SEGMENTS, a + PLUME_SEGMENTS))
    last = (len(path) - 1) * PLUME_SEGMENTS
    faces.append(tuple(reversed(range(PLUME_SEGMENTS))))
    faces.append(tuple(last + segment for segment in range(PLUME_SEGMENTS)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def _plume_material(fx: dict, chord: Vector) -> bpy.types.Material:
    """Principled Volume fading out along the plume and broken up by noise: smoke and steam carry
    density in the emitter colour, fire carries emission only."""
    material = bpy.data.materials.new(f"FX_{fx['node']}")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    links = tree.links
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    coords = tree.nodes.new("ShaderNodeTexCoord")
    along = tree.nodes.new("ShaderNodeVectorMath")
    along.operation = "DOT_PRODUCT"
    along.inputs[1].default_value = chord / max(chord.length_squared, 1e-6)
    links.new(coords.outputs["Object"], along.inputs[0])
    fade = tree.nodes.new("ShaderNodeMapRange")
    fade.inputs["To Min"].default_value = 1.0
    fade.inputs["To Max"].default_value = 0.0
    links.new(along.outputs["Value"], fade.inputs["Value"])
    noise = tree.nodes.new("ShaderNodeTexNoise")
    reach = max(float(fx["radius_end_m"]), float(fx["radius_start_m"]), 0.05)
    noise.inputs["Scale"].default_value = PLUME_NOISE_SCALE / reach
    noise.inputs["Detail"].default_value = 4.0
    links.new(coords.outputs["Object"], noise.inputs["Vector"])
    puffs = tree.nodes.new("ShaderNodeMapRange")
    puffs.inputs["From Min"].default_value = 0.35
    puffs.inputs["From Max"].default_value = 0.7
    links.new(noise.outputs["Fac"], puffs.inputs["Value"])
    amount = tree.nodes.new("ShaderNodeMath")
    amount.operation = "MULTIPLY"
    links.new(fade.outputs["Result"], amount.inputs[0])
    links.new(puffs.outputs["Result"], amount.inputs[1])
    scaled = tree.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    links.new(amount.outputs["Value"], scaled.inputs[0])
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    colour = tuple(fx["colour"]) + (1.0,)
    if float(fx["emission"]) > 0.0:
        scaled.inputs[1].default_value = float(fx["emission"])
        volume.inputs["Density"].default_value = 0.0
        volume.inputs["Emission Color"].default_value = colour
        links.new(scaled.outputs["Value"], volume.inputs["Emission Strength"])
    else:
        scaled.inputs[1].default_value = float(fx["density"])
        volume.inputs["Color"].default_value = colour
        links.new(scaled.outputs["Value"], volume.inputs["Density"])
    links.new(volume.outputs["Volume"], output.inputs["Volume"])
    return material


def _emissive(name: str, colour: list, strength: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = tuple(colour) + (1.0,)
    emission.inputs["Strength"].default_value = strength
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _sparks(fx: dict, origin: Vector, basis) -> bpy.types.Object:
    """A frozen burst: octahedral specks at random ages along ballistic paths in the emit cone."""
    rng = random.Random(fx["node"])
    speck = float(fx["size_m"]) * 0.5
    vertices = []
    faces = []
    for _ in range(int(fx["count"])):
        tilt = rng.random() * SPARK_CONE_RAD
        heading = rng.random() * 2.0 * math.pi
        speed = float(fx["rise_m_s"]) * (0.7 + 0.6 * rng.random())
        direction = Vector((math.sin(tilt) * math.cos(heading), math.sin(tilt) * math.sin(heading), math.cos(tilt)))
        age = rng.random() * float(fx["life_s"])
        centre = (basis @ direction) * speed * age + Vector((0.0, 0.0, -0.5 * GRAVITY_M_S2 * age * age))
        base = len(vertices)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                offset = Vector((0.0, 0.0, 0.0))
                offset[axis] = sign * speck
                vertices.append(tuple(centre + offset))
        faces.extend(
            [
                (base + 0, base + 2, base + 4),
                (base + 2, base + 1, base + 4),
                (base + 1, base + 3, base + 4),
                (base + 3, base + 0, base + 4),
                (base + 2, base + 0, base + 5),
                (base + 1, base + 2, base + 5),
                (base + 3, base + 1, base + 5),
                (base + 0, base + 3, base + 5),
            ]
        )
    mesh = bpy.data.meshes.new(f"FX_{fx['node']}_sparks")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(_emissive(mesh.name, fx["colour"], float(fx["emission"])))
    obj = bpy.data.objects.new(mesh.name, mesh)
    obj.location = origin
    return obj


def _build_fx(entries: list, imported) -> list:
    """Each running fx socket of one building as static geometry and lights in world space."""
    built = []
    for fx in entries:
        socket = _socket(imported, fx["node"])
        if socket is None:
            continue
        origin = socket.matrix_world.to_translation()
        basis = socket.matrix_world.to_3x3().normalized()
        up = (basis @ Vector((0.0, 0.0, 1.0))).normalized()
        if "light_w" in fx:
            light = bpy.data.lights.new(f"FX_{fx['node']}_light", type="POINT")
            light.energy = float(fx["light_w"])
            light.color = tuple(fx["colour"])
            light.shadow_soft_size = float(fx["size_m"])
            lamp = bpy.data.objects.new(light.name, light)
            lamp.location = origin + up * float(fx["size_m"])
            bpy.context.scene.collection.objects.link(lamp)
        if fx["type"] == "sparks":
            bpy.context.scene.collection.objects.link(_sparks(fx, origin, basis))
        elif "life_s" in fx:
            path = _plume_path(fx, up)
            mesh = _tube(f"FX_{fx['node']}_plume", path)
            mesh.materials.append(_plume_material(fx, path[-1][0]))
            plume = bpy.data.objects.new(mesh.name, mesh)
            plume.location = origin
            bpy.context.scene.collection.objects.link(plume)
        built.append(fx["node"])
    return built


def _build_buildings(entries: list) -> list:
    """Each building's LOD0 at its site: the glb's top-level nodes parented under a root empty at
    the site location turned by its yaw, as the viewer stands it."""
    placed = []
    for entry in entries:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=entry["glb"])
        imported = set(bpy.data.objects) - before
        for obj in [obj for obj in imported if _reduced(obj)]:
            imported.discard(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
        root = bpy.data.objects.new(f"Building_{entry['name']}", None)
        bpy.context.scene.collection.objects.link(root)
        root.location = Vector(entry["location"])
        root.rotation_euler = (0.0, 0.0, float(entry["yaw_rad"]))
        for obj in imported:
            if obj.parent is None:
                obj.parent = root
        bpy.context.view_layer.update()
        fx = _build_fx(entry.get("fx", []), imported)
        placed.append({"name": entry["name"], "fx": fx})
    return placed


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


def _render(scene, resolution_x: int, resolution_y: int, samples: int, filepath: str, file_format: str) -> None:
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = "RGB"
    scene.cycles.samples = samples
    scene.cycles.transparent_max_bounces = TRANSPARENT_BOUNCES
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def _luminances(path: str) -> list[float]:
    image = bpy.data.images.load(path)
    pixels = list(image.pixels)
    bpy.data.images.remove(image)
    return sorted(
        0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2]
        for index in range(0, len(pixels), 4)
    )


def _adapt_exposure(scene, exposure_args: dict, resolution_x: int, resolution_y: int, probe_path: str) -> float:
    """Meter a small, quick render of this view and expose its log-average luminance to the key,
    held back where that would push the brightest few percent past the highlight white.

    The physical sky spans several orders of magnitude between noon and moonlight, so a fixed
    exposure would leave every render but one time of day blown out or black.
    """
    scene.view_settings.exposure = 0.0
    divisor = int(exposure_args["probe_divisor"])
    _render(
        scene,
        max(resolution_x // divisor, 16),
        max(resolution_y // divisor, 16),
        int(exposure_args["probe_samples"]),
        probe_path,
        "OPEN_EXR",
    )
    luminances = _luminances(probe_path)
    average = math.exp(sum(math.log(1.0e-6 + value) for value in luminances) / max(len(luminances), 1))
    highlight_index = min(int(len(luminances) * float(exposure_args["highlight_percentile"])), len(luminances) - 1)
    highlight = max(luminances[highlight_index], 1.0e-6)
    stops = min(
        math.log2(float(exposure_args["key"]) / average),
        math.log2(float(exposure_args["highlight_white"]) / highlight),
    )
    stops = min(max(stops, float(exposure_args["min_stops"])), float(exposure_args["max_stops"]))
    scene.view_settings.exposure = stops
    return stops


def _render_view(
    scene,
    exposure_args: dict,
    resolution_x: int,
    resolution_y: int,
    samples: int,
    filepath: str,
    probe_path: str,
) -> float:
    stops = _adapt_exposure(scene, exposure_args, resolution_x, resolution_y, probe_path)
    _render(scene, resolution_x, resolution_y, samples, filepath, "PNG")
    return stops


def main() -> None:
    args = _read_args()
    _clear_scene()

    device_info = _set_cycles_device(args.get("render", {}).get("device", "OPTIX"))
    lighting = args["lighting"]
    _build_world(lighting["sky"])
    _build_moon(lighting["moon"])
    terrain = _build_terrain(args)
    terrain.data.materials.append(_build_material(args))

    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("BeautyCam")
    camera_data.clip_start = 0.1
    camera_data.clip_end = 4.0 * (float(args["world_size_m"]) + float(args["height_range_m"]))
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
    exposure_args = lighting["exposure"]
    probe_path = str(Path(args["result_json"]).with_name("exposure_probe.exr"))
    exposure_stops = {}

    three_quarter = camera_args.get("three_quarter", {})
    height_range = float(args["height_range_m"])
    distance = max(patch * float(three_quarter.get("distance_factor", 1.6)), height_range * 1.2)
    azimuth = math.radians(float(three_quarter.get("azimuth_deg", 45.0)))
    altitude = math.radians(float(three_quarter.get("altitude_deg", 35.0)))
    offset = Vector(
        (math.cos(azimuth) * math.cos(altitude), math.sin(azimuth) * math.cos(altitude), math.sin(altitude))
    ) * distance

    ground = camera_args.get("ground", {})
    look_azimuth = math.radians(float(ground.get("look_azimuth_deg", three_quarter.get("azimuth_deg", 45.0))))
    back_distance = float(ground.get("back_distance_m", max(10.0, patch * 0.25)))
    ground_x = centre_x - math.cos(look_azimuth) * back_distance
    ground_y = centre_y - math.sin(look_azimuth) * back_distance
    ground_z = _ground_height(depsgraph, ground_x, ground_y)
    eye_height = float(ground.get("eye_height_m", 1.8))

    _build_scatter(args.get("scatter", []))
    buildings = _build_buildings(args.get("buildings", []))

    camera.location = target + offset
    _look_at(camera, target)
    exposure_stops["three_quarter"] = _render_view(
        scene, exposure_args, resolution_x, resolution_y, samples, outputs["three_quarter"], probe_path
    )

    camera.location = Vector((ground_x, ground_y, ground_z + eye_height))
    _look_at(camera, Vector((centre_x, centre_y, centre_z + eye_height * 0.5)))
    exposure_stops["ground"] = _render_view(
        scene, exposure_args, resolution_x, resolution_y, samples, outputs["ground"], probe_path
    )

    result = {
        "device": device_info,
        "centre_world_m": [centre_x, centre_y, centre_z],
        "exposure_stops": exposure_stops,
        "buildings": buildings,
    }
    Path(args["result_json"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
