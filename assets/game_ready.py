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

from dataclasses import dataclass
from typing import Sequence

import bmesh
import bpy

from assets import atlas_bake
from assets.hardsurface import apply_transforms, recalc_normals
from assets.uv_islands import axis_key, face_islands, project_point

ATLAS_SIZE = 2048
ATLAS_MATERIAL_NAME = "atlas_material"
TRIANGLE_UV_SHRINK = 0.005


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


def _shelf_pack(sizes: list[tuple[float, float]], canvas_width: float, margin: float) -> tuple[list[tuple[float, float]], float, float]:
    """Place (width, height) rects left-to-right, wrapping to a new row below when a row would
    exceed canvas_width - the tallest-first order keeps rows dense. Every rect is placed strictly
    right of, or strictly below, every earlier rect in its row/column, so no two rects ever
    overlap regardless of how good the canvas_width guess turns out to be."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][1], reverse=True)
    offsets = [(0.0, 0.0)] * len(sizes)
    cursor_x = margin
    cursor_y = margin
    row_height = 0.0
    max_x = margin
    for i in order:
        width, height = sizes[i]
        if cursor_x > margin and cursor_x + width + margin > canvas_width:
            cursor_x = margin
            cursor_y += row_height + margin
            row_height = 0.0
        offsets[i] = (cursor_x, cursor_y)
        max_x = max(max_x, cursor_x + width + margin)
        cursor_x += width + margin
        row_height = max(row_height, height)
    total_height = cursor_y + row_height + margin
    return offsets, max_x, total_height


def uv_unwrap_and_pack(objects: Sequence[bpy.types.Object], margin: float = 0.01) -> None:
    """Axis-project every object's faces into islands (grouped by contiguous dominant-normal-axis
    region) and pack every island from every object into one shared 0-1 atlas with a deterministic
    shelf packer.

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
    meshes_by_object: dict[int, tuple[bpy.types.Object, bmesh.types.BMesh]] = {}

    for obj in objects:
        triangulate(obj)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        meshes_by_object[id(bm)] = (obj, bm)

        for group in face_islands(bm):
            axis, _sign = axis_key(group[0].normal)
            points = {face.index: [project_point(v.co, axis) for v in face.verts] for face in group}
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

    sizes = [(isl["width"], isl["height"]) for isl in islands]
    total_area = sum(w * h for w, h in sizes)
    canvas_width = max(total_area**0.5 * 1.3, max((w for w, _ in sizes), default=0.0) + 2.0 * margin, margin * 4.0)
    offsets, max_x, total_height = _shelf_pack(sizes, canvas_width, margin)
    norm = max(max_x, total_height, 1e-6)

    for isl, (off_x, off_y) in zip(islands, offsets):
        uv_layer = isl["bm"].loops.layers.uv.verify()
        min_x, min_y = isl["min"]
        for face in isl["faces"]:
            corners = [((px - min_x + off_x) / norm, (py - min_y + off_y) / norm) for px, py in isl["points"][face.index]]
            centroid_x = sum(u for u, _ in corners) / len(corners)
            centroid_y = sum(v for _, v in corners) / len(corners)
            for loop, (u, v) in zip(face.loops, corners):
                loop[uv_layer].uv = (
                    centroid_x + (u - centroid_x) * (1.0 - TRIANGLE_UV_SHRINK),
                    centroid_y + (v - centroid_y) * (1.0 - TRIANGLE_UV_SHRINK),
                )

    for obj, bm in meshes_by_object.values():
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()


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
    live scene's object positions never accumulate anything across repeated calls."""
    objects = list(objects)
    if not objects:
        raise ValueError("game_ready_pass requires at least one object")

    for obj in objects:
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
