"""Composable bpy helpers that export a game-ready object group to a shipped .glb + asset.json.

Runs inside Blender's own interpreter, never the uv venv - only bpy, mathutils and the stdlib. A
live MCP session imports this the same way it imports assets.game_ready and assets.lod. The
headless self-test in assets/blender_scripts/export_selftest.py does the same via subprocess.

Call export_asset on the object group game_ready_pass (and, if used, lod.generate_lods) already
ran on: it writes out/assets/<name>/<name>.glb (the glTF exporter converts Blender's Z-up to
Y-up on write), out/assets/<name>/textures/ (every unique image any assigned material's Principled
BSDF references), and out/assets/<name>/asset.json (tri counts per LOD, world bbox in metres,
material and texture lists, budget category, contact sheet path).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bpy
from mathutils import Vector

from assets.recipe import validate_name
from terrain import config

MANIFEST_NAME = "asset.json"
TEXTURES_DIRNAME = "textures"
CONTACT_SHEET_NAME = "contact_sheet.png"
VALID_KINDS = ("hero_building", "prop", "vehicle")

_LOD_SUFFIX = re.compile(r"_LOD(\d+)$")


class ExportError(ValueError):
    """Raised when an asset cannot be exported: no objects, an unknown budget kind, or a bad name."""


@dataclass
class ExportResult:
    glb_path: Path
    manifest_path: Path
    textures: list[str]
    materials: list[str]
    triangle_counts: dict[str, int]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "glb_path": str(self.glb_path),
            "manifest_path": str(self.manifest_path),
            "textures": self.textures,
            "materials": self.materials,
            "triangle_counts": self.triangle_counts,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
        }


def _select_only(objects: Sequence[bpy.types.Object]) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def _lod_group(object_name: str) -> str:
    match = _LOD_SUFFIX.search(object_name)
    return f"LOD{match.group(1)}" if match else "LOD0"


def _triangle_counts(objects: Sequence[bpy.types.Object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        obj.data.calc_loop_triangles()
        group = _lod_group(obj.name)
        counts[group] = counts.get(group, 0) + len(obj.data.loop_triangles)
    return dict(sorted(counts.items(), key=lambda item: int(item[0][len("LOD"):])))


def _world_bbox(objects: Sequence[bpy.types.Object]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    xs, ys, zs = (c.x for c in corners), (c.y for c in corners), (c.z for c in corners)
    xs, ys, zs = list(xs), list(ys), list(zs)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _unique_materials(objects: Sequence[bpy.types.Object]) -> list[bpy.types.Material]:
    materials: list[bpy.types.Material] = []
    seen: set[str] = set()
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is not None and material.name not in seen:
                seen.add(material.name)
                materials.append(material)
    return materials


def _export_textures(materials: Sequence[bpy.types.Material], textures_dir: Path) -> list[str]:
    textures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    seen: set[str] = set()
    for material in materials:
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            image = node.image
            if image.name in seen:
                continue
            seen.add(image.name)
            filename = f"{image.name}.png"
            image.filepath_raw = str(textures_dir / filename)
            image.file_format = "PNG"
            image.save()
            written.append(filename)
    return sorted(written)


def export_glb(objects: Sequence[bpy.types.Object], glb_path: Path) -> Path:
    """Export objects to glb_path. The glTF exporter converts Blender's Z-up to Y-up on write."""
    objects = list(objects)
    if not objects:
        raise ExportError("export requires at least one object")
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    _select_only(objects)
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )
    return glb_path


def _base_centre_shift(objects: Sequence[bpy.types.Object]) -> Vector:
    """The translation that would put the combined bbox's base-centre at world origin - the
    "origin at base centre" export convention, computed fresh from wherever objects actually are
    right now rather than assumed from an earlier bake."""
    bbox_min, bbox_max = _world_bbox(objects)
    return Vector(((bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5, bbox_min[2]))


def export_asset(objects: Sequence[bpy.types.Object], name: str, kind: str) -> ExportResult:
    """Export objects to out/assets/<name>/<name>.glb, write their textures to textures/, and
    record tri counts per LOD, world bbox, materials and texture list in asset.json.

    Objects are translated to the base-centre convention only for the instant the glb is written,
    then translated back - a live MCP session calls this repeatedly while patching individual
    parts of a multi-object asset, and permanently baking the shift into obj.location (as an
    earlier version did) leaves objects rebuilt after one call in a different coordinate frame
    from objects that were already shifted by an earlier call, a drift no later call can undo.
    Reverting after every call keeps every object at its original, consistent design coordinates
    no matter how many times the asset gets re-exported."""
    objects = list(objects)
    if not objects:
        raise ExportError("export requires at least one object")
    if kind not in VALID_KINDS:
        raise ExportError(f"kind {kind!r} must be one of {VALID_KINDS}")
    validate_name(name)

    shift = _base_centre_shift(objects)
    for obj in objects:
        obj.location = obj.location - shift
    bpy.context.view_layer.update()
    try:
        out_dir = config.ASSET_OUT_DIR / name
        glb_path = export_glb(objects, out_dir / f"{name}.glb")

        materials = _unique_materials(objects)
        textures = _export_textures(materials, out_dir / TEXTURES_DIRNAME)
        triangle_counts = _triangle_counts(objects)
        bbox_min, bbox_max = _world_bbox(objects)
        material_names = [material.name for material in materials]

        manifest = {
            "name": name,
            "kind": kind,
            "glb": glb_path.name,
            "textures": textures,
            "materials": material_names,
            "triangle_counts": triangle_counts,
            "bbox_min": list(bbox_min),
            "bbox_max": list(bbox_max),
            "contact_sheet": CONTACT_SHEET_NAME,
        }
        manifest_path = out_dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        for obj in objects:
            obj.location = obj.location + shift
        bpy.context.view_layer.update()

    return ExportResult(
        glb_path=glb_path,
        manifest_path=manifest_path,
        textures=textures,
        materials=material_names,
        triangle_counts=triangle_counts,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
