"""Composable bpy helpers that turn a game-ready LOD0 mesh into a shipped LOD0/1/2 set.

Runs inside Blender's own interpreter, never the uv venv - only bpy, bmesh and the stdlib. A
live MCP session imports this the same way it imports assets.game_ready and assets.hardsurface.
The headless self-test in assets/blender_scripts/lod_selftest.py does the same via subprocess.

Call generate_lods on the object game_ready_pass already ran on - it decimates two reduced
copies and bakes a normal map from LOD0 onto each one, so silhouette loss at distance is masked
by shading instead of showing as faceting. The reduced LODs get their own copy of the shared
atlas material - same nodes, only the Normal input's image differs - so the non-normal channels
(albedo, ORM) stay the one shared texture set across all three LODs.

Each reduced LOD's UVs are re-fit per island against LOD0's own already-packed atlas rectangles
(_refit_uvs_to_lod0_islands) rather than left as Decimate's raw vertex-interpolated output: an
edge collapse near an island boundary can interpolate a triangle's UV corners past its island's
packed bounds into a neighbouring island's region, which validate_asset's UV overlap check
correctly flags (af-3b4). Re-fitting keeps every decimated face's UV inside the exact atlas
rectangle its island already owns on LOD0, which both eliminates the cross-island bleed and keeps
the reduced LOD's UVs landing on roughly the same atlas texels LOD0's own UVs do - the shared
albedo/ORM atlas stays approximately correct on the reduced LODs without needing its own bake.
"""

from __future__ import annotations

from dataclasses import dataclass

import bmesh
import bpy

from assets.game_ready import TRIANGLE_UV_SHRINK, triangulate
from assets.uv_islands import axis_key, face_islands, project_point

LOD_RATIOS: tuple[float, float, float] = (1.0, 0.5, 0.15)
DEFAULT_NORMAL_MAP_SIZE = 1024
DEFAULT_CAGE_EXTRUSION_RATIO = 0.02
DEFAULT_BAKE_SAMPLES = 4
DEFAULT_BAKE_DEVICE = "OPTIX"
MIN_PROTECTED_PART_DIAGONAL_RATIO = 0.05
MIN_CLOSED_SHAPE_TRIANGLES = 12


@dataclass
class LodSet:
    objects: list[bpy.types.Object]
    triangle_counts: list[int]
    normal_maps: list[str | None]

    def as_dict(self) -> dict:
        return {
            "object_names": [obj.name for obj in self.objects],
            "triangle_counts": self.triangle_counts,
            "normal_maps": self.normal_maps,
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
) -> str | None:
    """Vertex group naming every vertex safe to hand the Decimate modifier.

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

    Returns None when every part clears both thresholds, so the modifier runs unconstrained
    exactly as it did before this existed.
    """
    mesh = obj.data
    parts = _connected_parts(mesh)
    if len(parts) <= 1:
        return None
    overall_diagonal = _bound_box_diagonal(obj)
    diagonal_threshold = overall_diagonal * min_diagonal_ratio
    protected = {
        index
        for indices in parts
        if _part_diagonal(mesh, indices) < diagonal_threshold
        or _part_triangle_count(mesh, indices) * ratio < min_closed_shape_triangles
        for index in indices
    }
    if not protected:
        return None
    eligible = obj.vertex_groups.new(name="lod_decimate_eligible")
    eligible.add(
        [vert.index for vert in mesh.vertices if vert.index not in protected], 1.0, "REPLACE"
    )
    return eligible.name


def _decimate(obj: bpy.types.Object, ratio: float) -> None:
    modifier = obj.modifiers.new(name="lod_decimate", type="DECIMATE")
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    group_name = _protect_small_parts(obj, ratio)
    if group_name is not None:
        modifier.vertex_group = group_name
    _select_only([obj])
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def _part_centers(mesh: bpy.types.Mesh, parts: list[list[int]]) -> list[tuple[float, float, float]]:
    centers = []
    for indices in parts:
        xs = [mesh.vertices[i].co.x for i in indices]
        ys = [mesh.vertices[i].co.y for i in indices]
        zs = [mesh.vertices[i].co.z for i in indices]
        centers.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0))
    return centers


def _lod0_island_rects(lod0: bpy.types.Object) -> tuple[list[dict], list[tuple[float, float, float]]]:
    """Read lod0's own already-packed, non-overlapping UVs and recover each island's exact atlas
    rectangle (min/size in 0-1 UV space), so a decimated LOD's faces can later be refit into the
    same rectangle instead of trusting Decimate's raw UV interpolation across island boundaries
    (af-3b4). Also returns lod0's own loose-part centres (see _refit_uvs_to_lod0_islands for why a
    decimated face is matched to its rect by loose part first, not by island centroid alone)."""
    parts = _connected_parts(lod0.data)
    part_of_vert = {index: part_index for part_index, indices in enumerate(parts) for index in indices}

    bm = bmesh.new()
    bm.from_mesh(lod0.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.verify()

    rects = []
    for group in face_islands(bm):
        axis, sign = axis_key(group[0].normal)
        uv_points = [tuple(loop[uv_layer].uv) for face in group for loop in face.loops]
        uv_min = (min(p[0] for p in uv_points), min(p[1] for p in uv_points))
        uv_max = (max(p[0] for p in uv_points), max(p[1] for p in uv_points))
        rects.append(
            {
                "axis": axis,
                "sign": sign,
                "part_index": part_of_vert[group[0].verts[0].index],
                "center_3d": tuple(sum(c) / len(c) for c in zip(*[v.co for face in group for v in face.verts])),
                "uv_min": uv_min,
                "uv_size": (max(uv_max[0] - uv_min[0], 1e-6), max(uv_max[1] - uv_min[1], 1e-6)),
            }
        )
    bm.free()
    return rects, _part_centers(lod0.data, parts)


def _refit_uvs_to_lod0_islands(
    obj: bpy.types.Object, rects: list[dict], lod0_part_centers: list[tuple[float, float, float]]
) -> None:
    """Re-derive obj's UVs entirely from its own current (post-decimate) geometry, mapping each
    face into the exact atlas rectangle its matching lod0 island already owns, rather than keeping
    whatever UV Decimate's vertex-interpolation left behind.

    A face is matched to a candidate rect in two stages: first by loose part (obj's own loose parts
    are matched to lod0's by nearest part centre - decimation never bridges two parts that weren't
    already connected, so a part's centre barely moves), then, among that part's own rects, by
    (axis, sign) and nearest island centre. Matching a face straight to its nearest island centre
    with no part stage first (tried during development) misroutes a face near the edge of one large
    island into a small, unrelated island that merely happens to sit physically closer to that one
    face than the large island's own averaged-out centre does - confirmed with a beam running
    through a box's centre and hinges sitting flush on the box's corners, both easily closer to a
    stray corner face than the box's own whole-wall island centre is.

    Every face routed to the same rect is then linearly fit - independently per axis, translate-
    then-scale - from its own bucket's local bbox into that rect's uv_min/uv_size span. Because the
    fit always maps a bucket's own bbox onto [uv_min, uv_min + uv_size], no bucket's UVs can ever
    leave the rectangle its island already owned on lod0 - and those rectangles were already proven
    non-overlapping there - so this cannot introduce the cross-island UV bleed Decimate's raw
    interpolation does, regardless of how much a bucket's own footprint grew or shrank relative to
    lod0's."""
    triangulate(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.verify()

    parts = _connected_parts(obj.data)
    part_centers = _part_centers(obj.data, parts)
    part_match = [
        min(
            range(len(lod0_part_centers)),
            key=lambda k: sum((lod0_part_centers[k][d] - center[d]) ** 2 for d in range(3)),
        )
        for center in part_centers
    ]
    matched_lod0_part_of_vert = {
        index: part_match[part_index] for part_index, indices in enumerate(parts) for index in indices
    }

    buckets: dict[int, list[bmesh.types.BMFace]] = {}
    for face in bm.faces:
        axis, sign = axis_key(face.normal)
        lod0_part = matched_lod0_part_of_vert[face.verts[0].index]
        candidates = [
            i for i, r in enumerate(rects) if r["axis"] == axis and r["sign"] == sign and r["part_index"] == lod0_part
        ]
        if not candidates:
            candidates = [i for i, r in enumerate(rects) if r["axis"] == axis and r["sign"] == sign]
        if candidates:
            centroid = tuple(sum(c) / len(c) for c in zip(*[v.co for v in face.verts]))
            chosen = min(
                candidates,
                key=lambda i: sum((rects[i]["center_3d"][k] - centroid[k]) ** 2 for k in range(3)),
            )
        else:
            chosen = 0
        buckets.setdefault(chosen, []).append(face)

    for index, faces in buckets.items():
        rect = rects[index]
        axis = rect["axis"]
        points_by_face = {face.index: [project_point(v.co, axis) for v in face.verts] for face in faces}
        xs = [p[0] for pts in points_by_face.values() for p in pts]
        ys = [p[1] for pts in points_by_face.values() for p in pts]
        bbox_min_x, bbox_min_y = min(xs), min(ys)
        bbox_width = max(max(xs) - bbox_min_x, 1e-6)
        bbox_height = max(max(ys) - bbox_min_y, 1e-6)
        uv_min_x, uv_min_y = rect["uv_min"]
        uv_width, uv_height = rect["uv_size"]

        for face in faces:
            corners = [
                (
                    uv_min_x + (px - bbox_min_x) / bbox_width * uv_width,
                    uv_min_y + (py - bbox_min_y) / bbox_height * uv_height,
                )
                for px, py in points_by_face[face.index]
            ]
            centroid_x = sum(u for u, _ in corners) / len(corners)
            centroid_y = sum(v for _, v in corners) / len(corners)
            for loop, (u, v) in zip(face.loops, corners):
                loop[uv_layer].uv = (
                    centroid_x + (u - centroid_x) * (1.0 - TRIANGLE_UV_SHRINK),
                    centroid_y + (v - centroid_y) * (1.0 - TRIANGLE_UV_SHRINK),
                )

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


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


def bake_normal_map(
    lod0: bpy.types.Object,
    lod_target: bpy.types.Object,
    base_material: bpy.types.Material,
    image_size: int = DEFAULT_NORMAL_MAP_SIZE,
    samples: int = DEFAULT_BAKE_SAMPLES,
    device: str = DEFAULT_BAKE_DEVICE,
) -> bpy.types.Image:
    """Bake LOD0's surface normal onto lod_target's own UVs (selected-to-active), and give
    lod_target a copy of base_material with only that image wired into the Normal input."""
    lod_material = base_material.copy()
    lod_material.name = f"{lod_target.name}_material"
    lod_target.data.materials.clear()
    lod_target.data.materials.append(lod_material)
    for polygon in lod_target.data.polygons:
        polygon.material_index = 0

    image = bpy.data.images.new(f"{lod_target.name}_normal", width=image_size, height=image_size, alpha=False)
    image.colorspace_settings.name = "Non-Color"

    node_tree = lod_material.node_tree
    image_node = node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = image
    for node in node_tree.nodes:
        node.select = False
    image_node.select = True
    node_tree.nodes.active = image_node

    normal_map_node = node_tree.nodes.new("ShaderNodeNormalMap")
    node_tree.links.new(image_node.outputs["Color"], normal_map_node.inputs["Color"])
    node_tree.links.new(normal_map_node.outputs["Normal"], _principled_bsdf(lod_material).inputs["Normal"])

    _configure_cycles_device(device)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = _bound_box_diagonal(lod0) * DEFAULT_CAGE_EXTRUSION_RATIO
    scene.render.bake.margin = 4

    _select_only([lod0, lod_target])
    bpy.ops.object.bake(type="NORMAL")

    return image


def generate_lods(
    lod0: bpy.types.Object,
    ratios: tuple[float, float, float] = LOD_RATIOS,
    normal_map_size: int = DEFAULT_NORMAL_MAP_SIZE,
) -> LodSet:
    """Decimate lod0 into two further LODs at ratios[1] and ratios[2], and bake a normal map
    from lod0 onto each. lod0 itself becomes the LOD0 entry, renamed with a _LOD0 suffix."""
    if len(ratios) != 3 or ratios[0] != 1.0:
        raise ValueError("ratios must be a 3-tuple with ratios[0] == 1.0 (LOD0 is full resolution)")
    if not lod0.data.materials:
        raise ValueError("lod0 must already have a material assigned (run game_ready_pass first)")

    base_name = lod0.name
    lod0.name = f"{base_name}_LOD0"
    base_material = lod0.data.materials[0]
    lod0_rects, lod0_part_centers = _lod0_island_rects(lod0)

    objects = [lod0]
    normal_maps: list[str | None] = [None]
    for index, ratio in enumerate(ratios[1:], start=1):
        lod = _duplicate(lod0, f"{base_name}_LOD{index}")
        _decimate(lod, ratio)
        _refit_uvs_to_lod0_islands(lod, lod0_rects, lod0_part_centers)
        image = bake_normal_map(lod0, lod, base_material, image_size=normal_map_size)
        objects.append(lod)
        normal_maps.append(image.name)

    return LodSet(
        objects=objects,
        triangle_counts=[_triangle_count(obj) for obj in objects],
        normal_maps=normal_maps,
    )
