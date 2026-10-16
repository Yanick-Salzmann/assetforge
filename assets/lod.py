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

from dataclasses import dataclass, field
from typing import Sequence

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from assets import animation_manifest
from assets.atlas_bake import dilate_gutters
from assets.game_ready import is_moving_part, triangulate, uv_unwrap_and_pack

LOD_RATIOS: tuple[float, float, float] = (1.0, 0.5, 0.15)
DEFAULT_TEXTURE_SIZE = 1024
DEFAULT_CAGE_EXTRUSION_RATIO = 0.02
DEFAULT_BAKE_SAMPLES = 4
DEFAULT_BAKE_DEVICE = "OPTIX"
BAKE_MARGIN_PX = 4
MIN_CLOSED_SHAPE_TRIANGLES = 12
LOD_DEVIATION_RATIOS: tuple[float, float, float] = (0.0, 0.008, 0.02)
PART_DEVIATION_RATIOS: tuple[float, float, float] = (0.0, 0.04, 0.1)
DEVIATION_RELAXATION_STEPS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
MAX_CULLED_AREA_FRACTION = 0.1


@dataclass
class LodTextures:
    albedo: str
    orm: str
    normal: str

    def as_dict(self) -> dict:
        return {"albedo": self.albedo, "orm": self.orm, "normal": self.normal}


@dataclass
class LodSet:
    """objects holds every object of every LOD, LOD0 first; levels groups them per LOD (the static
    body first, then its animated parts), and triangle_counts and textures are per LOD."""

    objects: list[bpy.types.Object]
    triangle_counts: list[int]
    textures: list[LodTextures | None]
    levels: list[list[bpy.types.Object]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "object_names": [obj.name for obj in self.objects],
            "triangle_counts": self.triangle_counts,
            "textures": [entry.as_dict() if entry is not None else None for entry in self.textures],
            "levels": [[obj.name for obj in level] for level in self.levels],
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


def _mesh_area(mesh: bpy.types.Mesh) -> float:
    return sum(polygon.area for polygon in mesh.polygons)


def _culled_parts(diagonals: list[float], areas: list[float], tolerance: float) -> set[int]:
    """Parts small enough to drop from a LOD outright: every point of a part whose whole extent
    fits inside the LOD's deviation tolerance is already within that error of where it sits, so
    removing it is no worse than the shape change the LOD accepts anyway. Bolts, rivets and
    levers are what keep a many-part asset's LOD2 near its LOD0 count (af-cae), because each is
    too small to decimate. Smallest first, and never more than MAX_CULLED_AREA_FRACTION of the
    asset's surface, so an asset built mostly from small parts (a chain, a pile) keeps its body.
    A larger tolerance only ever extends the set, so a part culled at LOD1 stays culled at LOD2."""
    budget = sum(areas) * MAX_CULLED_AREA_FRACTION
    candidates = sorted(
        (index for index, diagonal in enumerate(diagonals) if diagonal <= tolerance),
        key=lambda index: (diagonals[index], index),
    )
    culled: set[int] = set()
    spent = 0.0
    for index in candidates:
        if spent + areas[index] > budget:
            break
        spent += areas[index]
        culled.add(index)
    return culled


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
    tolerance: float,
    part_deviation_ratio: float,
) -> bpy.types.Mesh:
    """Reduce one loose part toward ratio * its LOD0 triangle count, starting from its previous
    LOD, accepting the first candidate whose deviation from the LOD0 part stays within both the
    asset-wide tolerance and part_deviation_ratio of the part's own diagonal, and that opens no
    new boundary edges.

    The part-relative cap is what lets small parts (pots, finials, hinges) be reduced at all
    without losing their silhouette: the asset-wide tolerance alone is larger than a whole pot,
    so it would accept a pot collapsed to a tetrahedron. The target never drops below a closed
    box (12 tris - long thin beams, barge boards), and a candidate that punches a hole is refused
    (af-6cu, af-dds). The ratio is relaxed in steps toward no reduction until the shape holds, so
    a part that cannot lose triangles without folding in (a boolean-cut tower wall at LOD2) simply
    keeps more of them; many-part buildings (af-x36) shed most of their LOD2 count here."""
    target = max(_mesh_triangle_count(original) * ratio, MIN_CLOSED_SHAPE_TRIANGLES)
    previous_triangles = _mesh_triangle_count(previous)
    relative = target / max(previous_triangles, 1)
    if relative >= 1.0:
        return previous.copy()
    part_tolerance = min(tolerance, _mesh_diagonal(original) * part_deviation_ratio)
    previous_boundary = _boundary_edge_count(previous)
    for step in DEVIATION_RELAXATION_STEPS:
        candidate = _collapsed(previous, relative + (1.0 - relative) * step)
        if (
            candidate.polygons
            and _boundary_edge_count(candidate) <= previous_boundary
            and _deviation(original, candidate) <= part_tolerance
        ):
            return candidate
        bpy.data.meshes.remove(candidate)
    return previous.copy()


def _boundary_edge_count(mesh: bpy.types.Mesh) -> int:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    count = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    bm.free()
    return count


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


def _group_diagonal(objects: Sequence[bpy.types.Object]) -> float:
    if len(objects) == 1:
        return _bound_box_diagonal(objects[0])
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return (high - low).length


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


def _transfer_bake(
    sources: Sequence[bpy.types.Object],
    targets: Sequence[bpy.types.Object],
    image: bpy.types.Image,
    bake_type: str,
) -> None:
    """Bake from every source onto each target in turn into one shared image. Every target after
    the first keeps what is already in the image, so the targets' islands accumulate."""
    material = targets[0].data.materials[0]
    target_node = _active_bake_target(material, image)
    bake = bpy.context.scene.render.bake
    try:
        for index, target in enumerate(targets):
            bake.use_clear = index == 0
            _select_only([*sources, target])
            outcome = bpy.ops.object.bake(type=bake_type)
            if outcome != {"FINISHED"}:
                raise RuntimeError(f"LOD {bake_type} bake onto {target.name!r} returned {sorted(outcome)}")
    finally:
        bake.use_clear = True
        material.node_tree.nodes.remove(target_node)


def _transfer_emission(
    sources: Sequence[bpy.types.Object],
    targets: Sequence[bpy.types.Object],
    image: bpy.types.Image,
    source: bpy.types.NodeSocket | None,
    fallback: tuple[float, float, float, float],
) -> None:
    restore = _emit_from(sources[0].data.materials[0], source, fallback)
    try:
        _transfer_bake(sources, targets, image, "EMIT")
    finally:
        restore()


def _lod_material(targets: Sequence[bpy.types.Object], albedo: bpy.types.Image, orm: bpy.types.Image, normal: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new(f"{targets[0].name}_material")
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

    for target in targets:
        target.data.materials.clear()
        target.data.materials.append(material)
        for polygon in target.data.polygons:
            polygon.material_index = 0
    return material


def bake_lod_textures(
    lod0: bpy.types.Object | Sequence[bpy.types.Object],
    lod_target: bpy.types.Object | Sequence[bpy.types.Object],
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
    bakes of LOD0's raw texture values, so no lighting or AO darkening lands in them.

    Either side may be a group: an animated asset's LOD is its static body plus a copy of every
    moving part. The targets are packed into one layout and share one material and texture set,
    each baked from every LOD0 object in turn; the Cycles margin is off then, since it would paint
    over islands already baked for another target, and dilate_gutters fills the gutters instead."""
    sources = [lod0] if isinstance(lod0, bpy.types.Object) else list(lod0)
    targets = [lod_target] if isinstance(lod_target, bpy.types.Object) else list(lod_target)
    lod0_material = sources[0].data.materials[0]
    foreign = [obj.name for obj in sources if obj.data.materials[0] != lod0_material]
    if foreign:
        raise ValueError(f"LOD0 objects {foreign} do not share {lod0_material.name!r}; run game_ready_pass on them together")
    for target in targets:
        triangulate(target)
    layout = uv_unwrap_and_pack(targets, atlas_size=image_size)
    lod0_bsdf = _principled_bsdf(lod0_material)

    stem = targets[0].name
    albedo = bpy.data.images.new(f"{stem}_albedo", width=image_size, height=image_size, alpha=False)
    orm = bpy.data.images.new(f"{stem}_orm", width=image_size, height=image_size, alpha=False)
    orm.colorspace_settings.name = "Non-Color"
    normal = bpy.data.images.new(f"{stem}_normal", width=image_size, height=image_size, alpha=False)
    normal.colorspace_settings.name = "Non-Color"
    _lod_material(targets, albedo, orm, normal)

    _configure_cycles_device(device)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = _group_diagonal(sources) * DEFAULT_CAGE_EXTRUSION_RATIO
    scene.render.bake.margin = BAKE_MARGIN_PX if len(targets) == 1 else 0
    scene.render.bake.margin_type = "EXTEND"
    try:
        base_color = tuple(lod0_bsdf.inputs["Base Color"].default_value)
        _transfer_emission(sources, targets, albedo, _socket_feeding(lod0_bsdf.inputs["Base Color"]), base_color)
        roughness = lod0_bsdf.inputs["Roughness"].default_value
        metallic = lod0_bsdf.inputs["Metallic"].default_value
        _transfer_emission(sources, targets, orm, _orm_source(lod0_material), (1.0, roughness, metallic, 1.0))
        _transfer_bake(sources, targets, normal, "NORMAL")
    finally:
        scene.render.bake.use_selected_to_active = False
    for image in (albedo, orm, normal):
        dilate_gutters(image, layout)
    return LodTextures(albedo=albedo.name, orm=orm.name, normal=normal.name)


@dataclass
class _PartReduction:
    """One LOD0 object split into loose parts and reduced level by level; meshes[level] holds the
    parts kept at that level (level 0 is the originals)."""

    meshes: list[list[bpy.types.Mesh]]
    scratch: list[bpy.types.Mesh]


def _reduce_object(
    obj: bpy.types.Object,
    ratios: tuple[float, float, float],
    overall_diagonal: float,
    deviation_ratios: tuple[float, float, float],
) -> _PartReduction:
    originals = _part_meshes(obj)
    diagonals = [_mesh_diagonal(mesh) for mesh in originals]
    areas = [_mesh_area(mesh) for mesh in originals]
    levels = [list(originals)]
    scratch = list(originals)
    previous: list[bpy.types.Mesh | None] = list(originals)
    for index, ratio in enumerate(ratios[1:], start=1):
        tolerance = overall_diagonal * deviation_ratios[index]
        culled = _culled_parts(diagonals, areas, tolerance)
        reduced = [
            None
            if prior is None or part_index in culled
            else _reduce_part(original, prior, ratio, tolerance, PART_DEVIATION_RATIOS[index])
            for part_index, (original, prior) in enumerate(zip(originals, previous))
        ]
        kept = [mesh for mesh in reduced if mesh is not None]
        scratch.extend(kept)
        levels.append(kept)
        previous = reduced
    return _PartReduction(meshes=levels, scratch=scratch)


def _share_animation(part: bpy.types.Object, copy: bpy.types.Object) -> None:
    """Drive copy with part's own clips: the same actions and slots, on NLA tracks of the same
    names, so the glTF exporter writes both nodes' channels into each clip."""
    source = part.animation_data
    if source is None:
        return
    data = copy.animation_data_create()
    if source.action is not None:
        data.action = source.action
        data.action_slot = source.action_slot
    for track in source.nla_tracks:
        mirrored = data.nla_tracks.new()
        mirrored.name = track.name
        mirrored.mute = track.mute
        for strip in track.strips:
            added = mirrored.strips.new(strip.name, int(strip.frame_start), strip.action)
            added.name = strip.name
            added.action_slot = strip.action_slot


def _mirror_part(part: bpy.types.Object, copy: bpy.types.Object, counterparts: dict) -> None:
    """Give copy part's parent (or that parent's own copy at this LOD), rest transform, custom
    properties and clips. A copy must not hang under a LOD0 node, which the viewer hides
    together with its children when it shows another LOD."""
    if part.parent is not None:
        copy.parent = counterparts.get(part.parent, part.parent)
        copy.parent_type = part.parent_type
        copy.matrix_parent_inverse = part.matrix_parent_inverse.copy()
    copy.rotation_mode = part.rotation_mode
    copy.location = part.location.copy()
    copy.rotation_euler = part.rotation_euler.copy()
    copy.rotation_quaternion = part.rotation_quaternion.copy()
    copy.scale = part.scale.copy()
    for key in part.keys():
        copy[key] = part[key]
    _share_animation(part, copy)


def generate_lods(
    lod0: bpy.types.Object,
    ratios: tuple[float, float, float] = LOD_RATIOS,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    deviation_ratios: tuple[float, float, float] = LOD_DEVIATION_RATIOS,
    moving: Sequence[bpy.types.Object] = (),
) -> LodSet:
    """Build LOD1 and LOD2 from lod0 one loose part at a time, then bake an albedo/ORM/normal set
    from lod0 onto each one's own fresh UVs (texture_size for LOD1, half that for LOD2). lod0
    itself becomes the LOD0 entry, renamed with a _LOD0 suffix, and keeps the shared atlas.

    ratios are triangle targets relative to LOD0; deviation_ratios cap how far, as a fraction of
    lod0's bbox diagonal, any reduced part may move from its LOD0 shape. A part that cannot reach
    its target within the cap keeps more triangles - shape wins over the ratio, because a folded
    wall is visible at any distance (church tower at LOD2). Each part's LOD2 is reduced from its
    LOD1, never above it, so counts can only fall (af-k4g); deviation and textures are always
    measured and baked against LOD0 so error is not compounded. Parts no bigger than a LOD's
    deviation cap are dropped from it entirely (_culled_parts).

    moving lists an animated asset's rigid <part>_ANIM objects, sharing lod0's atlas material.
    Each is reduced on its own, in its own pivot frame and against the whole asset's diagonal,
    into a <part>_LOD<n>_ANIM copy with the same parent (or its copy), rest transform and clips,
    so every LOD plays the same animation. Each LOD's static body and copies share one material
    and texture set, baked from every LOD0 object."""
    if len(ratios) != 3 or ratios[0] != 1.0:
        raise ValueError("ratios must be a 3-tuple with ratios[0] == 1.0 (LOD0 is full resolution)")
    if not lod0.data.materials:
        raise ValueError("lod0 must already have a material assigned (run game_ready_pass first)")
    moving = list(moving)
    unnamed = [part.name for part in moving if not is_moving_part(part)]
    if unnamed:
        raise ValueError(f"moving parts must be named <part>{animation_manifest.ANIM_SUFFIX}: {unnamed}")

    base_name = lod0.name
    lod0.name = f"{base_name}_LOD0"
    sources = [lod0, *moving]
    overall_diagonal = _group_diagonal(sources)
    reductions = [_reduce_object(obj, ratios, overall_diagonal, deviation_ratios) for obj in sources]

    levels = [sources]
    textures: list[LodTextures | None] = [None]
    for index in range(1, len(ratios)):
        body = _joined_object(f"{base_name}_LOD{index}", reductions[0].meshes[index], lod0.data.materials[0])
        body.matrix_world = lod0.matrix_world.copy()
        counterparts = {lod0: body}
        for part, reduction in zip(moving, reductions[1:]):
            name = animation_manifest.lod_copy_name(part.name, index)
            counterparts[part] = _joined_object(name, reduction.meshes[index], lod0.data.materials[0])
        for part in moving:
            _mirror_part(part, counterparts[part], counterparts)
        bpy.context.view_layer.update()
        level = [body, *(counterparts[part] for part in moving)]
        textures.append(bake_lod_textures(sources, level, image_size=max(texture_size >> (index - 1), 64)))
        levels.append(level)

    for reduction in reductions:
        for mesh in reduction.scratch:
            bpy.data.meshes.remove(mesh)

    return LodSet(
        objects=[obj for level in levels for obj in level],
        triangle_counts=[sum(_triangle_count(obj) for obj in level) for level in levels],
        textures=textures,
        levels=levels,
    )
