"""Composable bpy helpers that turn a game-ready LOD0 mesh into a shipped LOD0/1/2 set.

Runs inside Blender's own interpreter, never the uv venv - only bpy, bmesh and the stdlib. A
live MCP session imports this the same way it imports assets.game_ready and assets.hardsurface.
The headless self-test in assets/blender_scripts/lod_selftest.py does the same via subprocess.

Call generate_lods on the object game_ready_pass already ran on - it builds two reduced LODs
by decimating each loose part on its own under a shape-deviation cap, gives each LOD its own
packed UVs, and bakes an albedo/ORM/normal set from LOD0 onto it, so each reduced LOD ships with
its own material and textures while LOD0 keeps the shared atlas.

Earlier versions kept the LOD0 atlas on every LOD and re-fit the decimated faces' UVs into
LOD0's island rectangles (af-3b4, af-xvx). That cannot be made correct: a collapsed triangle
matches no LOD0 island exactly, so its UVs land on texels of other faces, window holes or empty
atlas space, which shows as black patches once the walls survive decimation (af-4ir.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

from assets.game_ready import triangulate, uv_unwrap_and_pack

LOD_RATIOS: tuple[float, float, float] = (1.0, 0.5, 0.15)
DEFAULT_TEXTURE_SIZE = 1024
DEFAULT_CAGE_EXTRUSION_RATIO = 0.02
DEFAULT_BAKE_SAMPLES = 4
DEFAULT_BAKE_DEVICE = "OPTIX"
MIN_PROTECTED_PART_DIAGONAL_RATIO = 0.05
MIN_CLOSED_SHAPE_TRIANGLES = 12
LOD_DEVIATION_RATIOS: tuple[float, float, float] = (0.0, 0.008, 0.02)
DEVIATION_RELAXATION_STEPS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


@dataclass
class LodTextures:
    albedo: str
    orm: str
    normal: str

    def as_dict(self) -> dict:
        return {"albedo": self.albedo, "orm": self.orm, "normal": self.normal}


@dataclass
class LodSet:
    objects: list[bpy.types.Object]
    triangle_counts: list[int]
    textures: list[LodTextures | None]

    def as_dict(self) -> dict:
        return {
            "object_names": [obj.name for obj in self.objects],
            "triangle_counts": self.triangle_counts,
            "textures": [entry.as_dict() if entry is not None else None for entry in self.textures],
        }


def _select_only(objects: list[bpy.types.Object]) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[-1]


def _triangle_count(obj: bpy.types.Object) -> int:
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def _connected_parts(mesh: bpy.types.Mesh) -> list[list[int]]:
    """Vertex indices grouped by loose part, via union-find over the mesh's own edges."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    parent = list(range(len(bm.verts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for edge in bm.edges:
        root_a, root_b = find(edge.verts[0].index), find(edge.verts[1].index)
        if root_a != root_b:
            parent[root_a] = root_b

    groups: dict[int, list[int]] = {}
    for vert in bm.verts:
        groups.setdefault(find(vert.index), []).append(vert.index)
    bm.free()
    return list(groups.values())


def _part_diagonal(mesh: bpy.types.Mesh, indices: list[int]) -> float:
    xs = [mesh.vertices[i].co.x for i in indices]
    ys = [mesh.vertices[i].co.y for i in indices]
    zs = [mesh.vertices[i].co.z for i in indices]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2) ** 0.5


def _mesh_triangle_count(mesh: bpy.types.Mesh) -> int:
    return sum(len(polygon.vertices) - 2 for polygon in mesh.polygons)


def _mesh_diagonal(mesh: bpy.types.Mesh) -> float:
    return _part_diagonal(mesh, list(range(len(mesh.vertices)))) if len(mesh.vertices) else 0.0


def _part_meshes(obj: bpy.types.Object) -> list[bpy.types.Mesh]:
    """One standalone mesh per loose part of obj, in obj's own coordinates. Decimating parts one
    at a time is what lets each part be held to its own shape: a single Decimate over the joined
    mesh spends its whole budget wherever collapses are cheapest, which on a building means the
    big flat walls (af-4ir.8), and can bridge vertices of unrelated parts."""
    mesh = obj.data
    parts = _connected_parts(mesh)
    part_of_vert: dict[int, int] = {}
    local_index: dict[int, int] = {}
    for part_index, indices in enumerate(parts):
        for position, vert_index in enumerate(indices):
            part_of_vert[vert_index] = part_index
            local_index[vert_index] = position
    faces: list[list[tuple[int, ...]]] = [[] for _ in parts]
    for polygon in mesh.polygons:
        faces[part_of_vert[polygon.vertices[0]]].append(tuple(local_index[v] for v in polygon.vertices))
    meshes = []
    for part_index, indices in enumerate(parts):
        if not faces[part_index]:
            continue
        part_mesh = bpy.data.meshes.new(f"{obj.name}_part_{part_index}")
        part_mesh.from_pydata([tuple(mesh.vertices[v].co) for v in indices], [], faces[part_index])
        part_mesh.update()
        meshes.append(part_mesh)
    return meshes


def _bvh(mesh: bpy.types.Mesh) -> BVHTree:
    return BVHTree.FromPolygons([v.co for v in mesh.vertices], [tuple(p.vertices) for p in mesh.polygons])


def _sample_points(mesh: bpy.types.Mesh) -> list:
    return [v.co for v in mesh.vertices] + [p.center for p in mesh.polygons]


def _deviation(original: bpy.types.Mesh, reduced: bpy.types.Mesh) -> float:
    """Symmetric worst-case distance between two meshes, sampled at vertices and face centres in
    both directions. Reduced-to-original catches faces that left the surface; original-to-reduced
    catches surface the reduction dropped - a tower corner collapsed across the wall puts its
    face centres metres inside the original even though every remaining vertex still sits on it."""
    original_tree = _bvh(original)
    reduced_tree = _bvh(reduced)
    worst = 0.0
    for points, tree in ((_sample_points(reduced), original_tree), (_sample_points(original), reduced_tree)):
        for point in points:
            distance = tree.find_nearest(point)[3]
            if distance is None:
                return float("inf")
            worst = max(worst, distance)
    return worst


def _collapsed(mesh: bpy.types.Mesh, ratio: float) -> bpy.types.Mesh:
    scratch = bpy.data.objects.new("lod_decimate_scratch", mesh)
    bpy.context.collection.objects.link(scratch)
    try:
        modifier = scratch.modifiers.new(name="lod_decimate", type="DECIMATE")
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return bpy.data.meshes.new_from_object(scratch.evaluated_get(depsgraph), depsgraph=depsgraph)
    finally:
        bpy.data.objects.remove(scratch, do_unlink=True)


def _reduce_part(
    original: bpy.types.Mesh,
    previous: bpy.types.Mesh,
    ratio: float,
    overall_diagonal: float,
    tolerance: float,
) -> bpy.types.Mesh:
    """Reduce one loose part toward ratio * its LOD0 triangle count, starting from its previous
    LOD, accepting the first candidate whose deviation from the LOD0 part stays within tolerance.

    Parts that are small next to the whole asset (hinges, finials, quoins) or that the ratio would
    push below a closed box (12 tris - long thin beams, barge boards) keep their previous shape;
    collapsing them punches holes (af-6cu, af-dds). For everything else the ratio is relaxed in
    steps toward no reduction until the shape holds, so a part that cannot lose triangles without
    folding in (a boolean-cut tower wall at LOD2) simply keeps more of them."""
    original_triangles = _mesh_triangle_count(original)
    small = _mesh_diagonal(original) < overall_diagonal * MIN_PROTECTED_PART_DIAGONAL_RATIO
    if small or original_triangles * ratio < MIN_CLOSED_SHAPE_TRIANGLES:
        return previous.copy()
    previous_triangles = _mesh_triangle_count(previous)
    relative = original_triangles * ratio / max(previous_triangles, 1)
    if relative >= 1.0:
        return previous.copy()
    for step in DEVIATION_RELAXATION_STEPS:
        candidate = _collapsed(previous, relative + (1.0 - relative) * step)
        if candidate.polygons and _deviation(original, candidate) <= tolerance:
            return candidate
        bpy.data.meshes.remove(candidate)
    return previous.copy()


def _joined_object(name: str, meshes: list[bpy.types.Mesh], material: bpy.types.Material) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for part in meshes:
        offset = len(vertices)
        vertices.extend(tuple(v.co) for v in part.vertices)
        faces.extend(tuple(offset + i for i in polygon.vertices) for polygon in part.polygons)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _bound_box_diagonal(obj: bpy.types.Object) -> float:
    xs = [corner[0] for corner in obj.bound_box]
    ys = [corner[1] for corner in obj.bound_box]
    zs = [corner[2] for corner in obj.bound_box]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2) ** 0.5


def _principled_bsdf(material: bpy.types.Material) -> bpy.types.ShaderNode:
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    raise ValueError(f"material {material.name!r} has no Principled BSDF node")


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


def _active_bake_target(material: bpy.types.Material, image: bpy.types.Image) -> bpy.types.ShaderNode:
    tree = material.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    for other in tree.nodes:
        other.select = False
    node.select = True
    tree.nodes.active = node
    return node


def _socket_feeding(input_socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket | None:
    return input_socket.links[0].from_socket if input_socket.is_linked else None


def _orm_source(material: bpy.types.Material) -> bpy.types.NodeSocket | None:
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeSeparateColor":
            return _socket_feeding(node.inputs[0])
    return None


def _emit_from(material: bpy.types.Material, source: bpy.types.NodeSocket | None, fallback: tuple[float, float, float, float]):
    """Route source (or a constant fallback colour) straight to material's surface output as an
    Emission shader, so an EMIT bake copies the raw texel value instead of a lit result. Returns a
    restore() callback that puts the original surface link back and removes the added nodes."""
    tree = material.node_tree
    output_node = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")
    surface_input = output_node.inputs["Surface"]
    original_from = surface_input.links[0].from_socket if surface_input.is_linked else None
    created = []
    emission = tree.nodes.new("ShaderNodeEmission")
    created.append(emission)
    if source is None:
        constant = tree.nodes.new("ShaderNodeRGB")
        constant.outputs[0].default_value = fallback
        created.append(constant)
        source = constant.outputs[0]
    tree.links.new(source, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], surface_input)

    def restore() -> None:
        if original_from is not None:
            tree.links.new(original_from, surface_input)
        for node in created:
            tree.nodes.remove(node)

    return restore


def _transfer_bake(lod0: bpy.types.Object, lod_target: bpy.types.Object, image: bpy.types.Image, bake_type: str) -> None:
    material = lod_target.data.materials[0]
    target_node = _active_bake_target(material, image)
    try:
        _select_only([lod0, lod_target])
        outcome = bpy.ops.object.bake(type=bake_type)
        if outcome != {"FINISHED"}:
            raise RuntimeError(f"LOD {bake_type} bake onto {lod_target.name!r} returned {sorted(outcome)}")
    finally:
        material.node_tree.nodes.remove(target_node)


def _transfer_emission(
    lod0: bpy.types.Object,
    lod_target: bpy.types.Object,
    image: bpy.types.Image,
    source: bpy.types.NodeSocket | None,
    fallback: tuple[float, float, float, float],
) -> None:
    restore = _emit_from(lod0.data.materials[0], source, fallback)
    try:
        _transfer_bake(lod0, lod_target, image, "EMIT")
    finally:
        restore()


def _lod_material(lod_target: bpy.types.Object, albedo: bpy.types.Image, orm: bpy.types.Image, normal: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new(f"{lod_target.name}_material")
    material.use_nodes = True
    tree = material.node_tree
    bsdf = _principled_bsdf(material)

    albedo_node = tree.nodes.new("ShaderNodeTexImage")
    albedo_node.image = albedo
    tree.links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])

    orm_node = tree.nodes.new("ShaderNodeTexImage")
    orm_node.image = orm
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(orm_node.outputs["Color"], separate.inputs["Color"])
    tree.links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])

    normal_node = tree.nodes.new("ShaderNodeTexImage")
    normal_node.image = normal
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    tree.links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    lod_target.data.materials.clear()
    lod_target.data.materials.append(material)
    for polygon in lod_target.data.polygons:
        polygon.material_index = 0
    return material


def bake_lod_textures(
    lod0: bpy.types.Object,
    lod_target: bpy.types.Object,
    image_size: int = DEFAULT_TEXTURE_SIZE,
    samples: int = DEFAULT_BAKE_SAMPLES,
    device: str = DEFAULT_BAKE_DEVICE,
) -> LodTextures:
    """Give lod_target a fresh UV layout and its own albedo/ORM/normal set, each baked from LOD0's
    surface (selected-to-active), wired into a new material of its own.

    Reusing LOD0's atlas on a decimated mesh cannot work in general: a collapsed triangle no
    longer matches any LOD0 island, so whatever UV it is given samples texels that belong to
    another face, a window hole or empty atlas space - black patches on the walls (af-4ir.9).
    Baking from LOD0 onto the LOD's own packed UVs samples whatever LOD0 surface lies under every
    texel, so a texel can only ever hold colour that exists on the asset. Albedo and ORM are EMIT
    bakes of LOD0's raw texture values, so no lighting or AO darkening lands in them."""
    triangulate(lod_target)
    uv_unwrap_and_pack([lod_target])
    lod0_material = lod0.data.materials[0]
    lod0_bsdf = _principled_bsdf(lod0_material)

    albedo = bpy.data.images.new(f"{lod_target.name}_albedo", width=image_size, height=image_size, alpha=False)
    orm = bpy.data.images.new(f"{lod_target.name}_orm", width=image_size, height=image_size, alpha=False)
    orm.colorspace_settings.name = "Non-Color"
    normal = bpy.data.images.new(f"{lod_target.name}_normal", width=image_size, height=image_size, alpha=False)
    normal.colorspace_settings.name = "Non-Color"
    _lod_material(lod_target, albedo, orm, normal)

    _configure_cycles_device(device)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = _bound_box_diagonal(lod0) * DEFAULT_CAGE_EXTRUSION_RATIO
    scene.render.bake.margin = 4
    try:
        base_color = tuple(lod0_bsdf.inputs["Base Color"].default_value)
        _transfer_emission(lod0, lod_target, albedo, _socket_feeding(lod0_bsdf.inputs["Base Color"]), base_color)
        roughness = lod0_bsdf.inputs["Roughness"].default_value
        metallic = lod0_bsdf.inputs["Metallic"].default_value
        _transfer_emission(lod0, lod_target, orm, _orm_source(lod0_material), (1.0, roughness, metallic, 1.0))
        _transfer_bake(lod0, lod_target, normal, "NORMAL")
    finally:
        scene.render.bake.use_selected_to_active = False
    return LodTextures(albedo=albedo.name, orm=orm.name, normal=normal.name)


def generate_lods(
    lod0: bpy.types.Object,
    ratios: tuple[float, float, float] = LOD_RATIOS,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    deviation_ratios: tuple[float, float, float] = LOD_DEVIATION_RATIOS,
) -> LodSet:
    """Build LOD1 and LOD2 from lod0 one loose part at a time, then bake an albedo/ORM/normal set
    from lod0 onto each one's own fresh UVs (texture_size for LOD1, half that for LOD2). lod0
    itself becomes the LOD0 entry, renamed with a _LOD0 suffix, and keeps the shared atlas.

    ratios are triangle targets relative to LOD0; deviation_ratios cap how far, as a fraction of
    lod0's bbox diagonal, any reduced part may move from its LOD0 shape. A part that cannot reach
    its target within the cap keeps more triangles - shape wins over the ratio, because a folded
    wall is visible at any distance (church tower at LOD2). Each part's LOD2 is reduced from its
    LOD1, never above it, so counts can only fall (af-k4g); deviation and textures are always
    measured and baked against LOD0 so error is not compounded."""
    if len(ratios) != 3 or ratios[0] != 1.0:
        raise ValueError("ratios must be a 3-tuple with ratios[0] == 1.0 (LOD0 is full resolution)")
    if not lod0.data.materials:
        raise ValueError("lod0 must already have a material assigned (run game_ready_pass first)")

    base_name = lod0.name
    lod0.name = f"{base_name}_LOD0"
    overall_diagonal = _bound_box_diagonal(lod0)
    originals = _part_meshes(lod0)

    objects = [lod0]
    textures: list[LodTextures | None] = [None]
    scratch_meshes = list(originals)
    previous = originals
    for index, ratio in enumerate(ratios[1:], start=1):
        tolerance = overall_diagonal * deviation_ratios[index]
        reduced = [
            _reduce_part(original, prior, ratio, overall_diagonal, tolerance)
            for original, prior in zip(originals, previous)
        ]
        scratch_meshes.extend(reduced)
        lod = _joined_object(f"{base_name}_LOD{index}", reduced, lod0.data.materials[0])
        lod.matrix_world = lod0.matrix_world.copy()
        textures.append(bake_lod_textures(lod0, lod, image_size=max(texture_size >> (index - 1), 64)))
        objects.append(lod)
        previous = reduced

    for mesh in scratch_meshes:
        bpy.data.meshes.remove(mesh)

    return LodSet(
        objects=objects,
        triangle_counts=[_triangle_count(obj) for obj in objects],
        textures=textures,
    )
