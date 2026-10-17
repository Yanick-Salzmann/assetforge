"""Egyptian pottery kit geometry (af-sel.2): hand-thrown vessels, a log pile and a sherd cluster,
each one closed object with its origin at its base centre, built for the `pottery` pack by
assets/blender_scripts/pottery_set_build.py.

Vessels are lathe profiles made imperfect the way a wheel-thrown pot is: the whole body leans, the
section is squashed oval about a random axis, and the radius wobbles in two or three low lobes that
drift with height. The base ring stays on z = 0 so every pot stands on its foot.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import bmesh
import bpy
from mathutils import Matrix, Vector

from assets import hardsurface as hs

SIDE = 0
CAP = 1
JAR_SMALL = [(0.0, 0.0), (0.08, 0.0), (0.14, 0.08), (0.15, 0.18), (0.11, 0.28), (0.08, 0.32), (0.1, 0.36), (0.07, 0.36), (0.0, 0.3)]
THROWN = [(0.0, 0.0), (0.1, 0.0), (0.13, 0.1), (0.11, 0.22), (0.09, 0.3), (0.07, 0.3), (0.0, 0.25)]
STORAGE = [(0.0, 0.0), (0.13, 0.0), (0.22, 0.07), (0.25, 0.2), (0.22, 0.32), (0.16, 0.38), (0.17, 0.42), (0.13, 0.42), (0.0, 0.36)]
BOWL = [(0.0, 0.0), (0.08, 0.0), (0.17, 0.08), (0.2, 0.16), (0.17, 0.16), (0.0, 0.05)]
AMPHORA = [(0.0, 0.0), (0.04, 0.02), (0.12, 0.12), (0.24, 0.45), (0.22, 0.65), (0.11, 0.8), (0.08, 0.88), (0.11, 0.95), (0.08, 0.95), (0.0, 0.85)]
WATER_JAR = [(0.0, 0.0), (0.12, 0.0), (0.25, 0.18), (0.28, 0.4), (0.2, 0.6), (0.14, 0.66), (0.17, 0.75), (0.13, 0.75), (0.0, 0.65)]
PROFILES = {
    "jar_small": JAR_SMALL,
    "thrown": THROWN,
    "storage": STORAGE,
    "bowl": BOWL,
    "amphora": AMPHORA,
    "water_jar": WATER_JAR,
}


def _finish(obj: bpy.types.Object) -> bpy.types.Object:
    hs.apply_transforms(obj)
    hs.recalc_normals(obj)
    return hs.smooth_by_angle(obj)


def _smooth_profile(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Insert a Catmull-Rom midpoint into every span of a lathe profile, so the silhouette reads
    round and LOD reduction has rings to remove."""
    smoothed = [points[0]]
    for index in range(len(points) - 1):
        p0 = Vector(points[max(index - 1, 0)])
        p1 = Vector(points[index])
        p2 = Vector(points[index + 1])
        p3 = Vector(points[min(index + 2, len(points) - 1)])
        middle = (-p0 + 9.0 * p1 + 9.0 * p2 - p3) / 16.0
        if 0 < index < len(points) - 2:
            smoothed.append((max(middle.x, 0.005), middle.y))
        smoothed.append(points[index + 1])
    return smoothed


def vessel(
    name: str,
    profile: str,
    seed: int,
    segments: int = 24,
    lean_deg: float = 3.0,
    oval: float = 0.05,
    wobble: float = 0.025,
    scale: float = 1.0,
) -> bpy.types.Object:
    """A lathe vessel from PROFILES, leaned lean_deg towards a random heading, squashed oval (the
    fractional radius difference between its long and short axes) and wobbled by wobble of its
    radius in low lobes."""
    rng = random.Random(seed)
    points = _smooth_profile([(radius * scale, z * scale) for radius, z in PROFILES[profile]])
    obj = hs.lathe(name, points, segments=segments)
    height = max(z for _, z in points)
    heading = rng.uniform(0.0, math.tau)
    oval_axis = rng.uniform(0.0, math.pi)
    lobes = rng.choice((2, 3))
    phase = rng.uniform(0.0, math.tau)
    drift = rng.uniform(-1.2, 1.2)
    lean = math.tan(math.radians(lean_deg))
    long_axis = Vector((math.cos(oval_axis), math.sin(oval_axis), 0.0))
    for vertex in obj.data.vertices:
        co = vertex.co
        radial = Vector((co.x, co.y, 0.0))
        if radial.length < 1e-6:
            continue
        angle = math.atan2(co.y, co.x)
        t = co.z / height
        along = radial.dot(long_axis)
        across = radial - long_axis * along
        radial = long_axis * along * (1.0 + 0.5 * oval) + across * (1.0 - 0.5 * oval)
        radial *= 1.0 + wobble * math.sin(lobes * angle + phase + drift * t) * min(1.0, 4.0 * t)
        co.x = radial.x
        co.y = radial.y
    for vertex in obj.data.vertices:
        vertex.co.x += math.cos(heading) * lean * vertex.co.z
        vertex.co.y += math.sin(heading) * lean * vertex.co.z
    obj.data.update()
    return _finish(obj)


def _log(name: str, start: Vector, end: Vector, radius: float, rng: random.Random) -> bpy.types.Object:
    obj = hs.cylinder_between(name, tuple(start), tuple(end), radius, radius * rng.uniform(0.85, 1.0), segments=12)
    hs.apply_transforms(obj)
    axis = (end - start).normalized()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    along = [e for e in bm.edges if abs((e.verts[1].co - e.verts[0].co).normalized().dot(axis)) > 0.9]
    bmesh.ops.subdivide_edges(bm, edges=along, cuts=2, use_grid_fill=True)
    for vertex in bm.verts:
        t = (vertex.co - start).dot(axis)
        if 0.01 < t < (end - start).length - 0.01:
            radial = vertex.co - (start + axis * t)
            vertex.co += radial * rng.uniform(-0.06, 0.06)
    bm.to_mesh(obj.data)
    bm.free()
    for polygon in obj.data.polygons:
        polygon.material_index = CAP if abs(polygon.normal.dot(axis)) > 0.9 else SIDE
    return obj


def log_pile(name: str, seed: int, length: float = 1.1, radius: float = 0.075, rows: Sequence[int] = (3, 2, 1)) -> bpy.types.Object:
    """Split logs stacked in a pyramid, each row resting in the grooves of the one below and
    sunk 1 cm into it, ends staggered and every log turned a few degrees off parallel."""
    rng = random.Random(seed)
    logs = []
    spacing = 2.0 * radius * 0.98
    z = radius * 0.9
    for row, count in enumerate(rows):
        for index in range(count):
            x = (index - 0.5 * (count - 1)) * spacing
            yaw = math.radians(rng.uniform(-6.0, 6.0))
            shift = rng.uniform(-0.08, 0.08)
            half = 0.5 * length * rng.uniform(0.85, 1.0)
            direction = Vector((math.sin(yaw), math.cos(yaw), 0.0))
            centre = Vector((x, shift, z))
            logs.append(_log(f"{name}_{row}_{index}", centre - direction * half, centre + direction * half, radius * rng.uniform(0.85, 1.05), rng))
        z += spacing * math.sqrt(3.0) * 0.5 - 0.01
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for log in logs:
        log.select_set(True)
    bpy.context.view_layer.objects.active = logs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return _finish(obj)


def _sherd(bm: bmesh.types.BMesh, rng: random.Random, centre: Vector, size: float, thickness: float) -> None:
    sides = rng.randint(4, 6)
    curvature = rng.uniform(1.5, 3.0) * rng.choice((-1.0, 1.0))
    outline = []
    for k in range(sides):
        angle = math.tau * k / sides + rng.uniform(-0.3, 0.3)
        reach = size * rng.uniform(0.55, 1.0)
        outline.append((math.cos(angle) * reach, math.sin(angle) * reach))
    tilt = Matrix.Rotation(math.radians(rng.uniform(4.0, 14.0)), 3, Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0)).normalized())
    yaw = Matrix.Rotation(rng.uniform(0.0, math.tau), 3, "Z")
    turn = yaw @ tilt

    def place(x: float, y: float, z: float) -> Vector:
        bend = 0.5 * curvature * (x * x + y * y)
        return turn @ Vector((x, y, z + bend))

    inner = [(0.5 * x, 0.5 * y) for x, y in outline]
    layers = {}
    for level, z in (("top", thickness), ("bottom", 0.0)):
        layers[level] = (place(0.0, 0.0, z), [place(x, y, z) for x, y in inner], [place(x, y, z) for x, y in outline])
    floor = min(point.z for centre_point, ring, rim in layers.values() for point in [centre_point, *ring, *rim])
    offset = centre - Vector((0.0, 0.0, floor))
    verts = {}
    for level, (centre_point, ring, rim) in layers.items():
        verts[level] = (bm.verts.new(centre_point + offset), [bm.verts.new(p + offset) for p in ring], [bm.verts.new(p + offset) for p in rim])
    for level, flip in (("top", False), ("bottom", True)):
        hub, ring, rim = verts[level]
        for k in range(sides):
            n = (k + 1) % sides
            for face in ((hub, ring[k], ring[n]), (ring[k], rim[k], rim[n], ring[n])):
                bm.faces.new(tuple(reversed(face)) if flip else face)
    top_rim = verts["top"][2]
    bottom_rim = verts["bottom"][2]
    for k in range(sides):
        n = (k + 1) % sides
        bm.faces.new((bottom_rim[k], bottom_rim[n], top_rim[n], top_rim[k]))


def sherd_cluster(name: str, seed: int, count: int = 9, spread: float = 0.45, size: float = 0.07) -> bpy.types.Object:
    """Broken pot fragments strewn around a point: thin curved slabs with irregular 4-6 sided
    outlines, tilted, kept apart so no two touch, every one resting on the ground plane."""
    rng = random.Random(seed)
    bm = bmesh.new()
    placed: list[Vector] = []
    tries = 0
    while len(placed) < count and tries < 500:
        tries += 1
        r = spread * math.sqrt(rng.random())
        angle = rng.uniform(0.0, math.tau)
        centre = Vector((r * math.cos(angle), r * math.sin(angle), 0.0))
        piece = size * rng.uniform(0.6, 1.3)
        if any((centre - other).length < 2.4 * size for other in placed):
            continue
        placed.append(centre)
        _sherd(bm, rng, centre, piece, rng.uniform(0.007, 0.011))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = hs._link(mesh, name)
    return _finish(obj)
