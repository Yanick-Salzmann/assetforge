"""Composable bpy helpers that turn a game-ready LOD0 mesh into a shipped LOD0/1/2 set.

Runs inside Blender's own interpreter, never the uv venv - only bpy, bmesh and the stdlib. A
live MCP session imports this the same way it imports assets.game_ready and assets.hardsurface.
The headless self-test in assets/blender_scripts/lod_selftest.py does the same via subprocess.

Call generate_lods on the object game_ready_pass already ran on - it decimates two reduced
copies, gives each its own packed UVs, and bakes an albedo/ORM/normal set from LOD0 onto it, so
each reduced LOD ships with its own material and textures while LOD0 keeps the shared atlas.

Earlier versions kept the LOD0 atlas on every LOD and re-fit the decimated faces' UVs into
LOD0's island rectangles (af-3b4, af-xvx). That cannot be made correct: a collapsed triangle
matches no LOD0 island exactly, so its UVs land on texels of other faces, window holes or empty
atlas space, which shows as black patches once the walls survive decimation (af-4ir.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import bmesh
import bpy

from assets.game_ready import triangulate, uv_unwrap_and_pack

LOD_RATIOS: tuple[float, float, float] = (1.0, 0.5, 0.15)
DEFAULT_TEXTURE_SIZE = 1024
DEFAULT_CAGE_EXTRUSION_RATIO = 0.02
DEFAULT_BAKE_SAMPLES = 4
DEFAULT_BAKE_DEVICE = "OPTIX"
MIN_PROTECTED_PART_DIAGONAL_RATIO = 0.05
MIN_CLOSED_SHAPE_TRIANGLES = 12


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


def _duplicate(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    mesh = obj.data.copy()
    duplicate = obj.copy()
    duplicate.data = mesh
    duplicate.name = name
    bpy.context.collection.objects.link(duplicate)
    return duplicate


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


def _part_triangle_count(mesh: bpy.types.Mesh, indices: list[int]) -> int:
    index_set = set(indices)
    mesh.calc_loop_triangles()
    return sum(1 for tri in mesh.loop_triangles if tri.vertices[0] in index_set)


def _part_diagonal(mesh: bpy.types.Mesh, indices: list[int]) -> float:
    xs = [mesh.vertices[i].co.x for i in indices]
    ys = [mesh.vertices[i].co.y for i in indices]
    zs = [mesh.vertices[i].co.z for i in indices]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2) ** 0.5


def _protect_small_parts(
    obj: bpy.types.Object,
    ratio: float,
    min_diagonal_ratio: float = MIN_PROTECTED_PART_DIAGONAL_RATIO,
    min_closed_shape_triangles: int = MIN_CLOSED_SHAPE_TRIANGLES,
) -> tuple[str | None, int]:
    """Vertex group naming every vertex safe to hand the Decimate modifier, plus the triangle
    count of the parts left out of it.

    A single flat COLLAPSE ratio applied to a whole joined multi-part mesh punches real holes
    through parts that are small or thin relative to the object as a whole - hinges, finials,
    glass panes - because the same ratio that removes negligible tris from the big parts removes
    all of the few tris those small parts have. Loose parts whose own bbox diagonal falls under
    min_diagonal_ratio of the object's full diagonal are left out of the returned group entirely,
    which is what keeps the Decimate modifier's vertex_group input from touching them at all.

    The diagonal ratio alone misses long/thin parts (full-span timber beams, barge boards) that
    are not small relative to the whole asset but were already built with few source triangles -
    ratio * their own triangle count can fall below what a closed manifold shape needs (12 tris
    for a simple box) well before the diagonal check would ever flag them. Any part whose own
    triangle count is already under min_closed_shape_triangles, or would be decimated under it by
    this ratio, is protected too.

    Returns (None, 0) when every part clears both thresholds, so the modifier runs unconstrained
    exactly as it did before this existed.
    """
    mesh = obj.data
    parts = _connected_parts(mesh)
    if len(parts) <= 1:
        return None, 0
    overall_diagonal = _bound_box_diagonal(obj)
    diagonal_threshold = overall_diagonal * min_diagonal_ratio
    protected: set[int] = set()
    protected_triangles = 0
    for indices in parts:
        part_triangles = _part_triangle_count(mesh, indices)
        if _part_diagonal(mesh, indices) < diagonal_threshold or part_triangles * ratio < min_closed_shape_triangles:
            protected.update(indices)
            protected_triangles += part_triangles
    if not protected:
        return None, 0
    eligible = obj.vertex_groups.new(name="lod_decimate_eligible")
    eligible.add(
        [vert.index for vert in mesh.vertices if vert.index not in protected], 1.0, "REPLACE"
    )
    return eligible.name, protected_triangles


def _whole_mesh_ratio(ratio: float, total_triangles: int, protected_triangles: int) -> float:
    """Decimate's ratio targets the whole mesh even when a vertex group pins part of it, so the
    pinned triangles' share of the cut lands on the eligible parts instead. When small protected
    parts (quoins, trim, hardware) hold most of a building's triangles, a raw 0.5 asks the walls
    and roofs to give up more triangles than they have and they collapse to nothing. Scaling the
    target so only the eligible triangles shrink by ratio keeps every part present."""
    if total_triangles <= 0:
        return ratio
    eligible_triangles = total_triangles - protected_triangles
    return (eligible_triangles * ratio + protected_triangles) / total_triangles


def _decimate(obj: bpy.types.Object, ratio: float) -> None:
    modifier = obj.modifiers.new(name="lod_decimate", type="DECIMATE")
    modifier.use_collapse_triangulate = True
    group_name, protected_triangles = _protect_small_parts(obj, ratio)
    modifier.ratio = _whole_mesh_ratio(ratio, _triangle_count(obj), protected_triangles)
    if group_name is not None:
        modifier.vertex_group = group_name
    _select_only([obj])
    bpy.ops.object.modifier_apply(modifier=modifier.name)


DEGENERATE_ISLAND_DIAGONAL_RATIO = 0.05


def _remove_decimate_debris(obj: bpy.types.Object) -> int:
    """Delete any post-Decimate connected component that is a single free-floating triangle
    spanning a large fraction of the object's own bounds - a rare COLLAPSE_TRIANGULATE artifact
    at the boundary between _protect_small_parts's protected and unprotected vertex groups, where
    the modifier bridges two unrelated collapsed vertices into one degenerate face instead of
    touching a real part. Confirmed on medieval_tavern: three orphan triangles, each its own
    3-vertex connected component with a bbox diagonal of several metres (spanning unrelated parts
    of the building), reported as 9 boundary edges (holes) by validate_asset's watertight check -
    real geometry never produces a lone triangle disconnected from everything else. A genuine tiny
    part (a bolt head, a sliver of trim) has vertices close together, so a bbox-diagonal threshold
    tells the two apart without needing to know which case produced a given 3-vertex island.
    Returns the number of triangles removed."""
    mesh = obj.data
    parts = _connected_parts(mesh)
    if len(parts) <= 1:
        return 0
    threshold = _bound_box_diagonal(obj) * DEGENERATE_ISLAND_DIAGONAL_RATIO
    debris_verts: set[int] = {
        index
        for indices in parts
        if len(indices) == 3 and _part_diagonal(mesh, indices) >= threshold
        for index in indices
    }
    if not debris_verts:
        return 0
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    doomed = [bm.verts[i] for i in debris_verts]
    removed = sum(len(v.link_faces) for v in doomed)
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return removed


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
) -> LodSet:
    """Decimate lod0 into two further LODs at ratios[1] and ratios[2], and bake an albedo/ORM/
    normal set from lod0 onto each one's own fresh UVs (texture_size for LOD1, half that for
    LOD2). lod0 itself becomes the LOD0 entry, renamed with a _LOD0 suffix, and keeps the atlas.

    Each LOD is decimated from the previous one at the relative ratio between them, not from
    lod0 directly. _protect_small_parts keeps any part that a given ratio would crush below a
    closed shape at full resolution, and a lower ratio protects more parts - decimating LOD2
    straight from lod0 left every mid-size part at its lod0 count and shipped medieval_tavern
    with LOD2 above LOD1 (af-k4g). Chaining means a part protected at LOD2 still carries its
    LOD1 reduction, so counts can only fall. Textures always bake from lod0, never from the
    previous LOD, so detail is resampled once."""
    if len(ratios) != 3 or ratios[0] != 1.0:
        raise ValueError("ratios must be a 3-tuple with ratios[0] == 1.0 (LOD0 is full resolution)")
    if not lod0.data.materials:
        raise ValueError("lod0 must already have a material assigned (run game_ready_pass first)")

    base_name = lod0.name
    lod0.name = f"{base_name}_LOD0"

    objects = [lod0]
    textures: list[LodTextures | None] = [None]
    previous = lod0
    for index, ratio in enumerate(ratios[1:], start=1):
        lod = _duplicate(previous, f"{base_name}_LOD{index}")
        _decimate(lod, ratio / ratios[index - 1])
        _remove_decimate_debris(lod)
        textures.append(bake_lod_textures(lod0, lod, image_size=max(texture_size >> (index - 1), 64)))
        objects.append(lod)
        previous = lod

    return LodSet(
        objects=objects,
        triangle_counts=[_triangle_count(obj) for obj in objects],
        textures=textures,
    )
