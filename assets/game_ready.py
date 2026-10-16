"""Composable bpy helpers that make a modelled object (or object group) ready for export.

Runs inside Blender's own interpreter, never the uv venv - only bpy, bmesh, mathutils and the
stdlib. A live MCP session imports this the same way it imports assets.hardsurface: by inserting
the repo root into sys.path. The headless self-test in
assets/blender_scripts/game_ready_selftest.py does the same via subprocess.

Order matters in game_ready_pass: transforms and normals are baked first, since origin_set and
the UV/material passes below read the post-transform mesh and would otherwise pack or centre
against stale geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from assets import animation_manifest, atlas_bake, atlas_pack, atlas_raster, uv_strips
from assets.hardsurface import apply_transforms, recalc_normals
from assets.uv_islands import axis_key, face_islands, project_point, shapes_overlap

ATLAS_SIZE = 2048
ATLAS_MATERIAL_NAME = "atlas_material"
TRIANGLE_UV_SHRINK = 0.005
BLOCK_PX = 4
GUTTER_PX_PER_ATLAS_PX = 1.0 / 128.0
PLANE_PROJECTION_MIN_TILT = math.radians(3.0)
PLANE_PROJECTION_MAX_FACES = 600
BASE_FACE_NORMAL_Z = -0.99
BASE_FACE_TOLERANCE = 1e-3
HIDDEN_SAMPLE_PULL = 0.05
INSIDE_RAY_DIRECTIONS = (Vector((0.3123, 0.4711, 0.8249)).normalized(), Vector((-0.6427, 0.2311, -0.7305)).normalized())
INSIDE_RAY_MAX_HITS = 64
INSIDE_RAY_STEP = 1e-5
_AXES = (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)))
_AXIS_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def gutter_px(atlas_px: int) -> int:
    """Pixels between any two islands' content on an atlas_px atlas: 16 at 2048, never below
    2 * BLOCK_PX, always a multiple of it - so each island keeps half the gutter of its own colour
    through BC7 blocks and the first mip levels."""
    step = 2 * BLOCK_PX
    return max(step, int(atlas_px * GUTTER_PX_PER_ATLAS_PX) // step * step)


@dataclass
class AtlasLayout:
    """The packed UV layout uv_unwrap_and_pack wrote, kept so the bake can fill each island's
    gutter from that island alone (atlas_bake.dilate_gutters)."""

    packing: atlas_pack.PixelPacking
    owners: list[int]
    triangles: list[tuple[tuple[float, float], ...]]
    hidden_faces: int
    plane_projected_islands: int
    strip_islands: int = 0
    _labels: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _fills: dict[int, tuple] = field(default_factory=dict, repr=False)

    def labels(self, size: int) -> np.ndarray:
        """Island owning each pixel centre of a size x size image of this layout (-1 for none)."""
        if size not in self._labels:
            scale = size / self.packing.atlas_px
            triangles = np.asarray(self.triangles, dtype=np.float64).reshape(-1, 3, 2) * scale
            self._labels[size] = atlas_raster.rasterize(triangles, np.asarray(self.owners, dtype=np.int32), size, size)
        return self._labels[size]

    def fill(self, size: int) -> tuple:
        """atlas_raster.fill_map for a size x size image, computed once and shared by every bake."""
        if size not in self._fills:
            self._fills[size] = atlas_raster.fill_map(self.labels(size), self.packing.footprints)
        return self._fills[size]

    def as_dict(self) -> dict:
        return {
            "atlas_px": self.packing.atlas_px,
            "gutter_px": self.packing.gutter_px,
            "block_px": self.packing.block_px,
            "density_px_per_m": round(self.packing.density, 3),
            "islands": len(self.packing.footprints),
            "hidden_faces": self.hidden_faces,
            "plane_projected_islands": self.plane_projected_islands,
            "strip_islands": self.strip_islands,
        }


@dataclass
class GameReadyReport:
    object_names: list[str]
    triangle_count: int
    material_name: str
    resized_images: list[str]

    def as_dict(self) -> dict:
        return {
            "object_names": self.object_names,
            "triangle_count": self.triangle_count,
            "material_name": self.material_name,
            "resized_images": self.resized_images,
        }


def _select_only(objects: Sequence[bpy.types.Object]) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def is_moving_part(obj: bpy.types.Object) -> bool:
    """A rigid animated part (<part>_ANIM): its origin is its pivot, so its location is never
    applied into the mesh, and faces it hides at rest may show once it moves."""
    return bool(animation_manifest.ANIM_NODE_NAME.match(obj.name))


def triangulate(obj: bpy.types.Object) -> bpy.types.Object:
    """Convert every ngon/quad to triangles so later UV packing sees the same primitives the
    exporter and validator do - packing ngons approximates their shape and leaves more islands
    overlapping than packing their actual triangles does."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def _inside(tree: BVHTree, point: Vector) -> bool:
    for direction in INSIDE_RAY_DIRECTIONS:
        hits = 0
        origin = point
        for _ in range(INSIDE_RAY_MAX_HITS):
            location, _normal, _index, _distance = tree.ray_cast(origin, direction)
            if location is None:
                break
            hits += 1
            origin = location + direction * INSIDE_RAY_STEP
        if hits % 2 == 0:
            return False
    return True


def hidden_faces(meshes: Sequence[bmesh.types.BMesh], moving: Sequence[bool] | None = None) -> list[set[int]]:
    """Per mesh, the indices of faces no camera can see: downward faces lying on the asset's base
    plane, and faces whose centre and (slightly pulled-in) corners all sit inside another part's
    closed volume - a pot's foot sunk into the plinth, a beam end buried in a wall. Meshes must be
    in a shared coordinate frame (transforms applied) and every part closed, which validate_asset
    already requires. These faces keep their geometry - the mesh stays watertight - but get no
    atlas texels (af-zla). A mesh flagged in moving (an animated part) neither loses faces nor
    hides another's: what it covers at rest it uncovers in motion."""
    if not meshes:
        return []
    moving = list(moving) if moving is not None else [False] * len(meshes)
    base_z = min(v.co.z for bm in meshes for v in bm.verts)
    trees = [BVHTree.FromBMesh(bm) for bm in meshes]
    boxes = []
    for bm in meshes:
        xs = [v.co.x for v in bm.verts]
        ys = [v.co.y for v in bm.verts]
        zs = [v.co.z for v in bm.verts]
        boxes.append((Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))))

    def within(point: Vector, box) -> bool:
        low, high = box
        return all(low[k] <= point[k] <= high[k] for k in range(3))

    hidden: list[set[int]] = []
    for index, bm in enumerate(meshes):
        found: set[int] = set()
        if moving[index]:
            hidden.append(found)
            continue
        for face in bm.faces:
            corners = [v.co for v in face.verts]
            if face.normal.z < BASE_FACE_NORMAL_Z and all(co.z <= base_z + BASE_FACE_TOLERANCE for co in corners):
                found.add(face.index)
                continue
            centre = face.calc_center_median()
            samples = [centre] + [co.lerp(centre, HIDDEN_SAMPLE_PULL) for co in corners]
            for other, (tree, box) in enumerate(zip(trees, boxes)):
                if other == index or moving[other] or not all(within(sample, box) for sample in samples):
                    continue
                if all(_inside(tree, sample) for sample in samples):
                    found.add(face.index)
                    break
        hidden.append(found)
    return hidden


def _signed_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total * 0.5


def _plane_projection(group: Sequence[bmesh.types.BMFace], axis: int, sign: int, axis_points: dict) -> dict | None:
    """Project a curved or slanted island onto the plane across its area-weighted normal instead of
    the axis plane, so its texels are not stretched by up to 1/cos(54.7 deg) = 1.73 relative to the
    rest of the atlas (af-alp.3). The basis is the axis plane's own basis rotated onto that normal,
    keeping the island's handedness. None when the tilt is negligible, the island too large to
    check, or the new projection would flip or overlap any face - the axis projection stands then."""
    normal = Vector()
    for face in group:
        normal += face.normal * face.calc_area()
    if normal.length < 1e-12 or len(group) > PLANE_PROJECTION_MAX_FACES:
        return None
    normal.normalize()
    axis_direction = _AXES[axis] * sign
    if normal.angle(axis_direction) < PLANE_PROJECTION_MIN_TILT:
        return None
    rotation = axis_direction.rotation_difference(normal)
    first, second = _AXIS_PLANE[axis]
    u_axis = rotation @ _AXES[first]
    v_axis = rotation @ _AXES[second]
    points = {face.index: [(v.co.dot(u_axis), v.co.dot(v_axis)) for v in face.verts] for face in group}
    for face in group:
        before = _signed_area(axis_points[face.index])
        after = _signed_area(points[face.index])
        if abs(after) < 1e-12 or (before > 0.0) != (after > 0.0):
            return None
    shapes = [points[face.index] for face in group]
    bounds = [(min(p[0] for p in s), min(p[1] for p in s), max(p[0] for p in s), max(p[1] for p in s)) for s in shapes]
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = bounds[i], bounds[j]
            if a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]:
                continue
            if shapes_overlap(shapes[i], shapes[j]):
                return None
    return points


def _pack_with_fallback(sizes: list[tuple[float, float]], atlas_size: int) -> atlas_pack.PixelPacking:
    """pack_pixels at gutter_px(atlas_size), halving the gutter (and the block grid with it) only
    when the islands cannot fit at all - a tiny LOD texture with hundreds of islands still gets a
    layout, and validate_asset's gutter check reports the narrower gutter instead of the build
    failing outright."""
    gutter = gutter_px(atlas_size)
    while True:
        block = max(1, min(BLOCK_PX, gutter // 2))
        try:
            return atlas_pack.pack_pixels(sizes, atlas_size, gutter, block)
        except ValueError:
            if gutter == 0:
                raise
            gutter = gutter // 2 if gutter > 2 else 0


def _strip_islands(bm: bmesh.types.BMesh, excluded: set[int]) -> list[dict]:
    positions = np.array([tuple(v.co) for v in bm.verts], dtype=np.float64).reshape(-1, 3)
    triangles = np.array([[v.index for v in face.verts] for face in bm.faces], dtype=np.int64).reshape(-1, 3)
    rows = [face.index for face in bm.faces if face.index not in excluded]
    islands = []
    for strip in uv_strips.strip_islands(positions, triangles, rows):
        islands.append({"faces": [bm.faces[row] for row in strip.triangles], "points": dict(zip(strip.triangles, strip.points))})
    return islands


def _hidden_in_world(meshes: Sequence[tuple[bpy.types.Object, bmesh.types.BMesh]]) -> list[set[int]]:
    world = []
    for obj, bm in meshes:
        copy = bm.copy()
        copy.transform(obj.matrix_world)
        world.append(copy)
    try:
        return hidden_faces(world, [is_moving_part(obj) for obj, _ in meshes])
    finally:
        for copy in world:
            copy.free()


def uv_unwrap_and_pack(
    objects: Sequence[bpy.types.Object],
    atlas_size: int = ATLAS_SIZE,
    cull_hidden: bool = True,
) -> AtlasLayout:
    """Axis-project every object's faces into islands (grouped by contiguous dominant-normal-axis
    region) and pack every island from every object into one shared 0-1 atlas with a deterministic
    shelf packer in atlas pixels (atlas_pack.pack_pixels): one uniform texel density, and a
    gutter_px(atlas_size) gutter between islands on a BC7-block grid, so neither the bake margin
    nor any of the first mip levels can carry one island's colour into another (af-alp). Slanted
    and curved islands are projected onto their own mean plane where that is safe
    (_plane_projection), and faces nobody can see (hidden_faces) are collapsed to one point instead
    of spending texels. The side of every revolved part (lathe, frustum, cylinder_between) is
    unwrapped first as a few continuous strips (assets/uv_strips.py, af-zch.1) instead of one
    island per normal direction; caps and everything else take the axis path. Returns the AtlasLayout the bake needs to fill each gutter from its own
    island.

    Does not use bpy.ops.uv.smart_project or bpy.ops.uv.pack_islands: on hard-surface meshes with
    many small bevel/greeble faces, smart_project collapses most islands to zero UV area (verified
    on a bare bevelled cube, no greeble needed) and pack_islands fails to keep ~90+ islands from
    overlapping regardless of margin/margin_method/angle_limit - see af-4ir.3. Axis projection
    cannot degenerate a non-degenerate 3D face to zero area, and the shelf packer's placement
    invariant (each rect strictly right of or below every earlier one in its row) makes overlap
    impossible by construction rather than a packing-quality outcome.

    Each triangle is also shrunk toward its own centroid by TRIANGLE_UV_SHRINK: the EXACT boolean
    solver (panel_cut) leaves near-degenerate sliver triangles at some cut seams whose projected
    UV corners can differ from a same-island neighbour's by less than floating-point noise, which
    the packer's per-island bounding-box separation does not reach since both slivers sit inside
    the same island. The shrink gives every triangle real clearance from its neighbours instead.
    """
    islands: list[dict] = []
    meshes: list[tuple[bpy.types.Object, bmesh.types.BMesh]] = []

    for obj in objects:
        triangulate(obj)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.faces.ensure_lookup_table()
        meshes.append((obj, bm))

    hidden = _hidden_in_world(meshes) if cull_hidden else [set() for _ in meshes]
    plane_projected = 0
    strip_count = 0
    for (obj, bm), excluded in zip(meshes, hidden):
        strips = _strip_islands(bm, excluded)
        strip_count += len(strips)
        stripped = {face.index for strip in strips for face in strip["faces"]}
        groups = [(strip["faces"], strip["points"]) for strip in strips]
        for group in face_islands(bm, excluded | stripped):
            axis, sign = axis_key(group[0].normal)
            points = {face.index: [project_point(v.co, axis) for v in face.verts] for face in group}
            tilted = _plane_projection(group, axis, sign, points)
            if tilted is not None:
                points = tilted
                plane_projected += 1
            groups.append((group, points))
        for group, points in groups:
            xs = [p[0] for pts in points.values() for p in pts]
            ys = [p[1] for pts in points.values() for p in pts]
            min_x, min_y = min(xs), min(ys)
            islands.append(
                {
                    "bm": bm,
                    "faces": group,
                    "points": points,
                    "min": (min_x, min_y),
                    "width": max(max(xs) - min_x, 1e-6),
                    "height": max(max(ys) - min_y, 1e-6),
                }
            )

    hidden_count = sum(len(faces) for faces in hidden)
    sizes = [(isl["width"], isl["height"]) for isl in islands]
    if hidden_count:
        sizes.append((0.0, 0.0))
    packing = _pack_with_fallback(sizes, atlas_size)
    density = packing.density

    owners: list[int] = []
    triangles: list[tuple[tuple[float, float], ...]] = []
    for index, isl in enumerate(islands):
        uv_layer = isl["bm"].loops.layers.uv.verify()
        min_x, min_y = isl["min"]
        origin_x, origin_y = packing.content_origins[index]
        for face in isl["faces"]:
            corners = [(origin_x + (px - min_x) * density, origin_y + (py - min_y) * density) for px, py in isl["points"][face.index]]
            centroid_x = sum(x for x, _ in corners) / len(corners)
            centroid_y = sum(y for _, y in corners) / len(corners)
            shrunk = [
                (centroid_x + (x - centroid_x) * (1.0 - TRIANGLE_UV_SHRINK), centroid_y + (y - centroid_y) * (1.0 - TRIANGLE_UV_SHRINK))
                for x, y in corners
            ]
            for loop, (x, y) in zip(face.loops, shrunk):
                loop[uv_layer].uv = (x / atlas_size, y / atlas_size)
            owners.append(index)
            triangles.append(tuple(shrunk))

    if hidden_count:
        hidden_x, hidden_y = packing.content_origins[-1]
        hidden_uv = ((hidden_x + packing.block_px * 0.5) / atlas_size, (hidden_y + packing.block_px * 0.5) / atlas_size)
        for (obj, bm), excluded in zip(meshes, hidden):
            uv_layer = bm.loops.layers.uv.verify()
            for face_index in excluded:
                for loop in bm.faces[face_index].loops:
                    loop[uv_layer].uv = hidden_uv

    for obj, bm in meshes:
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()

    return AtlasLayout(
        packing=packing,
        owners=owners,
        triangles=triangles,
        hidden_faces=hidden_count,
        plane_projected_islands=plane_projected,
        strip_islands=strip_count,
    )


def consolidate_material(
    objects: Sequence[bpy.types.Object],
    material_name: str = ATLAS_MATERIAL_NAME,
    bake_result: atlas_bake.AtlasBakeResult | None = None,
) -> bpy.types.Material:
    """Replace every material slot on every object with one shared atlas material. If bake_result
    is given (from atlas_bake.bake_atlas, called on objects' original per-face materials before
    this wipes them), wire its albedo/normal/orm images into the new atlas material."""
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
        material.use_nodes = True
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
    if bake_result is not None:
        atlas_bake.wire_atlas_textures(material, bake_result)
    return material


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def enforce_power_of_two_textures(material: bpy.types.Material, max_size: int = ATLAS_SIZE) -> list[str]:
    """Resize any image texture on material to the nearest power of two, capped at max_size."""
    resized: list[str] = []
    if material.node_tree is None:
        return resized
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image = node.image
        width, height = image.size[0], image.size[1]
        target_width = min(_next_power_of_two(width), max_size)
        target_height = min(_next_power_of_two(height), max_size)
        if (width, height) != (target_width, target_height):
            image.scale(target_width, target_height)
            resized.append(image.name)
    return resized


def game_ready_pass(objects: Sequence[bpy.types.Object], atlas_size: int = ATLAS_SIZE, material_name: str = ATLAS_MATERIAL_NAME) -> GameReadyReport:
    """Apply transforms/normals, pack one UV atlas and one material, and enforce power-of-two
    texture sizes. Runs before export; produces the UV layout the texture bake writes into.

    Deliberately does not touch object position: an earlier version centred the whole group's
    origin here, permanently baking a translation into every object's obj.location. Live MCP
    sessions call this (and export) repeatedly while patching individual parts of a multi-object
    asset - each patch rebuilds some objects fresh at raw design coordinates while leaving others
    untouched, so a translation baked in at one call and not the next left the two groups in
    different coordinate frames permanently, no later call could reconcile it, and the drift
    compounded with every patch. assets/export.py:export_asset now computes the same base-centre
    point but only holds the shift for the instant it writes the glb, then reverts it - so the
    live scene's object positions never accumulate anything across repeated calls.

    Animated parts (<part>_ANIM, from motion.pivot_part) keep their transform: the location is
    their pivot and the rotation is keyed by their clips, so applying it would bake whatever frame
    the scene sits on into the mesh. They already rest with identity rotation and scale."""
    objects = list(objects)
    if not objects:
        raise ValueError("game_ready_pass requires at least one object")

    for obj in objects:
        if not is_moving_part(obj):
            apply_transforms(obj)
        recalc_normals(obj)

    uv_unwrap_and_pack(objects)
    material = consolidate_material(objects, material_name)
    resized_images = enforce_power_of_two_textures(material, atlas_size)

    triangle_count = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        triangle_count += len(obj.data.loop_triangles)
    return GameReadyReport(
        object_names=[obj.name for obj in objects],
        triangle_count=triangle_count,
        material_name=material.name,
        resized_images=resized_images,
    )
