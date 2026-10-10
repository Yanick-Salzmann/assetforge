from __future__ import annotations

import math
import random
from typing import Sequence

import bmesh
import bpy
from mathutils import Vector, noise

from assets.hardsurface import _link, apply_modifier

SIDE = 0
CAP = 1
WELD_M = 1e-3


def _decimate(obj: bpy.types.Object, target_tris: int) -> bpy.types.Object:
    tris = sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)
    if tris <= target_tris:
        return obj
    modifier = obj.modifiers.new("decimate", "DECIMATE")
    modifier.ratio = target_tris / tris
    modifier.use_collapse_triangulate = True
    return apply_modifier(obj, modifier)


def rock(
    name: str,
    seed: int,
    size: Sequence[float],
    facets: int = 9,
    facet_depth: float = 0.18,
    roughness: float = 0.12,
    base_cut: float = 0.3,
    subdivisions: int = 4,
    target_tris: int = 450,
) -> bpy.types.Object:
    """A chiselled boulder: a noisy sphere cut by random planes, flattened on a base and decimated.

    size is the full extent in metres before the base cut; base_cut is the fraction of the height
    sliced off the bottom so the rock sits on the ground. Origin at base centre.
    """
    rng = random.Random(seed)
    offset = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    planes = []
    for _ in range(facets):
        normal = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1) * 0.6 + 0.3)).normalized()
        planes.append((normal, 1.0 - facet_depth * rng.uniform(0.4, 1.0)))

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    half = Vector(size) * 0.5
    for vertex in bm.verts:
        direction = vertex.co.normalized()
        bump = noise.fractal(direction * 1.6 + offset, 0.5, 2.0, 4)
        point = direction * (1.0 + roughness * bump)
        for normal, distance in planes:
            reach = point.dot(normal)
            if reach > distance:
                point -= normal * (reach - distance)
        vertex.co = Vector((point.x * half.x, point.y * half.y, point.z * half.z))
    floor = -half.z + base_cut * 2.0 * half.z
    for vertex in bm.verts:
        if vertex.co.z < floor:
            vertex.co.z = floor + (vertex.co.z - floor) * 0.02
    lowest = min(vertex.co.z for vertex in bm.verts)
    for vertex in bm.verts:
        vertex.co.z -= lowest
    bm.to_mesh(mesh)
    bm.free()
    obj = _link(mesh, name)
    return _decimate(obj, target_tris)


def _frame(tangent: Vector, previous: Vector | None) -> Vector:
    reference = previous if previous is not None else (Vector((1, 0, 0)) if abs(tangent.x) < 0.9 else Vector((0, 1, 0)))
    side = reference - tangent * reference.dot(tangent)
    return side.normalized()


def tube(
    name: str,
    path: Sequence[Sequence[float]],
    radii: Sequence[float],
    sides: int = 16,
    ribs: int = 0,
    rib_depth: float = 0.0,
    jitter: float = 0.0,
    seed: int = 0,
    top: str = "flat",
    dome_rings: int = 3,
) -> bpy.types.Object:
    """A closed tube swept along a polyline: flat bottom cap, flat or domed top.

    ribs grooves run along the length at rib_depth (fraction of the radius); jitter roughens every
    ring radially. Side faces carry material index SIDE, flat caps CAP.
    """
    rng = random.Random(seed)
    points = [Vector(point) for point in path]
    bm = bmesh.new()
    rings = []
    side = None
    for index, point in enumerate(points):
        ahead = points[min(index + 1, len(points) - 1)]
        behind = points[max(index - 1, 0)]
        tangent = (ahead - behind).normalized()
        side = _frame(tangent, side)
        other = tangent.cross(side)
        ring = []
        for step in range(sides):
            angle = 2.0 * math.pi * step / sides
            groove = 0.0
            if ribs:
                groove = rib_depth * (0.5 - 0.5 * math.cos(angle * ribs))
            radius = radii[index] * (1.0 - groove) * (1.0 + jitter * rng.uniform(-1.0, 1.0))
            ring.append(bm.verts.new(point + (side * math.cos(angle) + other * math.sin(angle)) * radius))
        rings.append(ring)

    if top == "dome":
        tip = points[-1]
        tangent = (points[-1] - points[-2]).normalized()
        base = rings[-1]
        for level in range(1, dome_rings + 1):
            fraction = level / (dome_rings + 1)
            lift = math.sin(fraction * math.pi / 2.0) * radii[-1]
            shrink = math.cos(fraction * math.pi / 2.0)
            rings.append([bm.verts.new(tip + (vertex.co - tip) * shrink + tangent * lift) for vertex in base])
        apex = bm.verts.new(tip + tangent * radii[-1])

    for lower, upper in zip(rings, rings[1:]):
        for step in range(sides):
            face = bm.faces.new((lower[step], lower[(step + 1) % sides], upper[(step + 1) % sides], upper[step]))
            face.material_index = SIDE

    bottom = bm.faces.new(list(reversed(rings[0])))
    bottom.material_index = CAP
    if top == "dome":
        for step in range(sides):
            face = bm.faces.new((rings[-1][step], rings[-1][(step + 1) % sides], apex))
            face.material_index = SIDE
    else:
        cap = bm.faces.new(rings[-1])
        cap.material_index = CAP

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _link(mesh, name)


def union(target: bpy.types.Object, others: Sequence[bpy.types.Object]) -> bpy.types.Object:
    """Exact boolean union of closed solids into target; the others are deleted."""
    for other in others:
        modifier = target.modifiers.new("union", "BOOLEAN")
        modifier.operation = "UNION"
        modifier.solver = "EXACT"
        modifier.object = other
        apply_modifier(target, modifier)
        bpy.data.objects.remove(other, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD_M)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=WELD_M)
    bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 4])
    bm.to_mesh(target.data)
    bm.free()
    return target


def triangles(obj: bpy.types.Object) -> int:
    return sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)
