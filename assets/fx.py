"""bpy helper that places an FX socket: an empty named `FX_<type>[_<label>]` at the emitter's
transform, which export_asset ships as a glb node beside its asset.json fx[] entry.

Runs inside Blender's own interpreter. The emit direction is the empty's local +Z, which the glTF
exporter turns into +Y. A socket parented to a `<part>_ANIM` node rides that part's animation.
Every socket's fx[] entry is also recorded on the scene (recorded_fx), so a saved .blend carries it
to the headless game_ready_build.
"""

from __future__ import annotations

import json
from typing import Sequence

import bpy

from assets import fx_manifest

SOCKET_DISPLAY_SIZE_M = 0.25
FX_PROPERTY = "assetforge_fx"


def recorded_fx() -> list[dict]:
    """The fx[] entries of every socket add_fx_socket made in this scene whose empty still exists,
    in the order made."""
    raw = bpy.context.scene.get(FX_PROPERTY)
    entries = json.loads(raw) if raw else []
    return [entry for entry in entries if entry["node"] in bpy.data.objects]


def _record_fx(entry: dict) -> None:
    raw = bpy.context.scene.get(FX_PROPERTY)
    entries = [other for other in json.loads(raw) if other["node"] != entry["node"]] if raw else []
    entries.append(entry)
    bpy.context.scene[FX_PROPERTY] = json.dumps(entries)


def add_fx_socket(
    fx_type: str,
    location: Sequence[float],
    state: str,
    rate: float,
    size: float,
    colour: Sequence[float],
    label: str = "",
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    parent: bpy.types.Object | None = None,
) -> tuple[bpy.types.Object, dict]:
    """Create the socket empty at world location/rotation (Euler XYZ radians) and return it with
    its fx[] entry. With parent, the socket keeps its world transform and follows the parent."""
    name = fx_manifest.fx_node_name(fx_type, label)
    entry = fx_manifest.fx_entry(name, fx_type, state, rate, size, colour)
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    socket = bpy.data.objects.new(name, None)
    socket.empty_display_type = "SINGLE_ARROW"
    socket.empty_display_size = SOCKET_DISPLAY_SIZE_M
    bpy.context.scene.collection.objects.link(socket)
    socket.location = location
    socket.rotation_euler = rotation
    if parent is not None:
        bpy.context.view_layer.update()
        world = socket.matrix_world.copy()
        socket.parent = parent
        socket.matrix_parent_inverse = parent.matrix_world.inverted()
        socket.matrix_world = world
    bpy.context.view_layer.update()
    _record_fx(entry)
    return socket, entry
