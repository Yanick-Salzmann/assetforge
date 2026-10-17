"""bpy helper that places a kit prop on an asset: an empty named `PROP_<label>` at the prop's
transform, instancing the kit glb's LOD0 so the prop shows in the viewport and contact sheets,
which export_asset records in asset.json props[] instead of joining into the glb.

Runs inside Blender's own interpreter. Every placement is recorded on the scene by empty name
(recorded_props reads the empties' current transforms), so a saved .blend carries them to the
headless game_ready_build and a prop moved by hand in a live session exports where it now stands.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

import bpy

from assets import prop_manifest
from terrain import config

NODE_PREFIX = "PROP_"
PROPS_PROPERTY = "assetforge_props"
KIT_COLLECTION_PREFIX = "KIT_"
REDUCED_LOD = re.compile(r"_LOD[1-9]\d*(\.\d+)?$")


class PropError(ValueError):
    pass


def is_prop_node(obj: bpy.types.Object) -> bool:
    return obj.name.startswith(NODE_PREFIX)


def _recorded() -> dict[str, str]:
    raw = bpy.context.scene.get(PROPS_PROPERTY)
    return json.loads(raw) if raw else {}


def recorded_props() -> list[dict]:
    """Every prop add_prop placed in this scene whose empty still exists, in the order placed:
    {mesh, location, rotation, scale} in Blender world coordinates, read from the empty now."""
    placements = []
    for name, mesh in _recorded().items():
        empty = bpy.data.objects.get(name)
        if empty is None:
            continue
        location, rotation, scale = empty.matrix_world.decompose()
        placements.append(
            {
                "mesh": mesh,
                "location": list(location),
                "rotation": list(rotation.to_euler("XYZ")),
                "scale": max(scale),
            }
        )
    return placements


def _under_reduced_lod(obj: bpy.types.Object) -> bool:
    node = obj
    while node is not None:
        if REDUCED_LOD.search(node.name):
            return True
        node = node.parent
    return False


def _kit_collection(mesh: str) -> bpy.types.Collection:
    name = f"{KIT_COLLECTION_PREFIX}{mesh}"
    existing = bpy.data.collections.get(name)
    if existing is not None:
        return existing
    from library import kits

    index = kits.build()
    if mesh not in index:
        raise PropError(f"unknown kit prop {mesh!r}")
    path = index[mesh].path(config.LIBRARY_DIR)
    if not path.is_file():
        raise PropError(f"kit prop {mesh!r} is not on disk at {path}; pull the kit library")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = set(bpy.data.objects) - before
    reduced = {obj for obj in imported if _under_reduced_lod(obj)}
    collection = bpy.data.collections.new(name)
    for obj in imported - reduced:
        for owner in list(obj.users_collection):
            owner.objects.unlink(obj)
        collection.objects.link(obj)
    for obj in reduced:
        bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def add_prop(
    mesh: str,
    location: Sequence[float],
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    label: str = "",
) -> tuple[bpy.types.Object, dict]:
    """Place kit prop mesh (`<pack>/<model>`) at world location, XYZ Euler rotation (radians) and
    uniform scale; return the empty and its placement. Re-using a label replaces that prop."""
    prop_manifest.prop_entry(mesh, location, rotation, scale)
    name = f"{NODE_PREFIX}{label or mesh.split('/')[-1]}"
    if label and name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    empty = bpy.data.objects.new(name, None)
    empty.instance_type = "COLLECTION"
    empty.instance_collection = _kit_collection(mesh)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = location
    empty.rotation_euler = rotation
    empty.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    recorded = _recorded()
    recorded[empty.name] = mesh
    bpy.context.scene[PROPS_PROPERTY] = json.dumps(recorded)
    placement = {"mesh": mesh, "location": list(location), "rotation": list(rotation), "scale": float(scale)}
    return empty, placement
