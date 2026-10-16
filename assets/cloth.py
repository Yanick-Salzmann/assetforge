"""bpy builders for skinned cloth: flags, pennants, banners and awnings on their rods.

Runs inside Blender's own interpreter. Geometry, skin weights and layouts come from the pure
assets.cloth_shapes; this module turns a ClothLayout into a closed cloth mesh `<name>_cloth`
skinned by an Armature modifier to `<name>_rig`, whose bone chain `<name>_bone<i>` runs from the
attached edge outwards, plus a closed rod `<name>_rod<i>` per attachment. animate_wind keys the
idle wind clip on the chain with the layout's per-kind bend amplitude.

Pass every ClothRig.objects to assets.export.export_asset together with the clip entries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import bpy

from assets import cloth_shapes
from assets import hardsurface
from assets import motion

if cloth_shapes.STANDOFF_M != hardsurface.STANDOFF_M:
    raise RuntimeError("cloth_shapes.STANDOFF_M must equal hardsurface.STANDOFF_M")


@dataclass(frozen=True)
class ClothRig:
    name: str
    kind: str
    cloth: bpy.types.Object
    rig: bpy.types.Object
    rods: list[bpy.types.Object]
    bones: list[str]
    max_bend: float

    @property
    def objects(self) -> list[bpy.types.Object]:
        return [*self.rods, self.rig, self.cloth]


def _cloth_mesh(name: str, panel: cloth_shapes.Panel) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_cloth_mesh")
    mesh.from_pydata(panel.vertices, [], panel.faces)
    layer = mesh.uv_layers.new(name="UVMap")
    for polygon, corners in zip(mesh.polygons, panel.uvs):
        for loop_index, uv in zip(polygon.loop_indices, corners):
            layer.data[loop_index].uv = uv
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_cloth", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _rig(name: str, panel: cloth_shapes.Panel) -> tuple[bpy.types.Object, list[str]]:
    rig = bpy.data.objects.new(f"{name}_rig", bpy.data.armatures.new(f"{name}_rig_data"))
    bpy.context.collection.objects.link(rig)
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    names = []
    parent = None
    for index, (head, tail) in enumerate(zip(panel.bone_heads, panel.bone_tails)):
        bone = rig.data.edit_bones.new(f"{name}_bone{index}")
        bone.head = head
        bone.tail = tail
        bone.align_roll(panel.cross_axis)
        bone.parent = parent
        bone.use_connect = parent is not None
        parent = bone
        names.append(bone.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig, names


def _skin(cloth: bpy.types.Object, rig: bpy.types.Object, bones: list[str], panel: cloth_shapes.Panel) -> None:
    groups = [cloth.vertex_groups.new(name=bone) for bone in bones]
    for vertex_index, weights in enumerate(panel.weights):
        for bone_index, weight in weights:
            groups[bone_index].add([vertex_index], weight, "REPLACE")
    cloth.parent = rig
    cloth.modifiers.new("Armature", "ARMATURE").object = rig


def build(name: str, layout: cloth_shapes.ClothLayout, material: bpy.types.Material | None = None) -> ClothRig:
    rods = [
        hardsurface.cylinder_between(f"{name}_rod{index}", rod.start, rod.end, rod.radius, segments=8)
        for index, rod in enumerate(layout.rods)
    ]
    cloth = _cloth_mesh(name, layout.panel)
    rig, bones = _rig(name, layout.panel)
    _skin(cloth, rig, bones, layout.panel)
    if material is not None:
        for part in (*rods, cloth):
            part.data.materials.clear()
            part.data.materials.append(material)
    return ClothRig(name, layout.kind, cloth, rig, rods, bones, layout.max_bend)


def flag(name: str, material: bpy.types.Material | None = None, **params) -> ClothRig:
    return build(name, cloth_shapes.flag(**params), material)


def pennant(name: str, material: bpy.types.Material | None = None, **params) -> ClothRig:
    return build(name, cloth_shapes.pennant(**params), material)


def banner(name: str, material: bpy.types.Material | None = None, **params) -> ClothRig:
    return build(name, cloth_shapes.banner(**params), material)


def awning(name: str, material: bpy.types.Material | None = None, **params) -> ClothRig:
    return build(name, cloth_shapes.awning(**params), material)


def animate_wind(
    cloth_rig: ClothRig,
    duration_s: float,
    seed: int,
    strength: float = 1.0,
    clip: str | None = None,
    fps: int | None = None,
    gust_period_s: float | None = None,
) -> dict:
    """The idle wind clip `idle_<clip>` (clip defaults to `wind_<name>`) over the cloth's chain."""
    return motion.animate_wind(
        cloth_rig.rig,
        cloth_rig.bones,
        duration_s,
        seed,
        strength=strength,
        clip=clip or "wind_" + re.sub(r"[^a-z0-9_]", "_", cloth_rig.name.lower()),
        fps=fps,
        gust_period_s=gust_period_s,
        max_bend=cloth_rig.max_bend,
    )
