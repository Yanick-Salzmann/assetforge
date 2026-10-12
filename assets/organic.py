from __future__ import annotations

import math
import random
from typing import Sequence

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

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
    """Exact boolean union of closed solids into target; the others are deleted. Face material
    indices survive the union even before any material is assigned."""
    original = list(target.data.materials)
    slots = 1 + max(polygon.material_index for obj in (target, *others) for polygon in obj.data.polygons)
    for obj in (target, *others):
        while len(obj.data.materials) < slots:
            obj.data.materials.append(_placeholder_material(len(obj.data.materials)))
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
    _repair_non_manifold(bm)
    bm.to_mesh(target.data)
    bm.free()
    indices = [polygon.material_index for polygon in target.data.polygons]
    target.data.materials.clear()
    for material in original:
        target.data.materials.append(material)
    target.data.polygons.foreach_set("material_index", indices)
    target.data.update()
    return target


def _repair_non_manifold(bm: bmesh.types.BMesh, passes: int = 4) -> None:
    """Cut out the faces around any non-manifold edge the exact boolean left where two solids
    touch tangentially, then cap the holes - a small local patch instead of a broken solid."""
    for _ in range(passes):
        bad = [edge for edge in bm.edges if not edge.is_manifold]
        if not bad:
            return
        faces = {face for edge in bad for vertex in edge.verts for face in vertex.link_faces}
        material = max(faces, key=lambda face: face.calc_area()).material_index
        bmesh.ops.delete(bm, geom=list(faces), context="FACES")
        bmesh.ops.delete(bm, geom=[vertex for vertex in bm.verts if not vertex.link_faces], context="VERTS")
        boundary = [edge for edge in bm.edges if edge.is_boundary]
        filled = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
        for face in filled["faces"]:
            face.material_index = material
        bmesh.ops.triangulate(bm, faces=filled["faces"])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)


def _placeholder_material(index: int) -> bpy.types.Material:
    name = f"organic_slot_{index}"
    return bpy.data.materials.get(name) or bpy.data.materials.new(name)


def triangles(obj: bpy.types.Object) -> int:
    return sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)


def _transform(obj: bpy.types.Object, matrix: Matrix) -> bpy.types.Object:
    obj.data.transform(matrix)
    obj.data.update()
    return obj


def _uniform_material(obj: bpy.types.Object, index: int) -> bpy.types.Object:
    for polygon in obj.data.polygons:
        polygon.material_index = index
    return obj


def settle(obj: bpy.types.Object) -> bpy.types.Object:
    """Shift obj's mesh so its lowest vertex sits at z = 0."""
    lowest = min(vertex.co.z for vertex in obj.data.vertices)
    return _transform(obj, Matrix.Translation((0.0, 0.0, -lowest)))


def ground_cut(obj: bpy.types.Object, material_index: int = SIDE) -> bpy.types.Object:
    """Slice a closed solid flat at z = 0, drop everything below and cap the cut with material_index.

    Roots, stems and trunks are modelled running into the ground and trimmed here, so the asset
    keeps a closed flat base that sits on the terrain instead of floating ends."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geometry = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geometry, plane_co=(0.0, 0.0, 0.0), plane_no=(0.0, 0.0, 1.0), clear_inner=True, dist=1e-5)
    for vertex in bm.verts:
        if abs(vertex.co.z) < 1e-5:
            vertex.co.z = 0.0
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    if boundary:
        filled = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
        for face in filled["faces"]:
            face.material_index = material_index
        bmesh.ops.triangulate(bm, faces=filled["faces"])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def _finish(obj: bpy.types.Object, target_tris: int) -> bpy.types.Object:
    ground_cut(obj)
    _decimate(obj, target_tris)
    return settle(obj)


def _triangulate_ngons(obj: bpy.types.Object) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 4])
    bm.to_mesh(obj.data)
    bm.free()


def stump(
    name: str,
    seed: int,
    radius: float = 0.3,
    height: float = 0.5,
    buttresses: int = 5,
    flare: float = 1.5,
    jag: float = 0.0,
    sides: int = 18,
    target_tris: int = 1600,
) -> bpy.types.Object:
    """A tree stump with flared buttress roots diving into the ground.

    jag > 0 breaks the top into splinters up to jag metres tall instead of a flat saw cut. Bark
    faces carry material index SIDE, the cut top CAP. Origin at base centre.
    """
    rng = random.Random(seed)
    levels = 7
    path = [(0.0, 0.0, -0.15 + (height + 0.15) * t / (levels - 1)) for t in range(levels)]
    radii = [radius * (1.0 + (flare - 1.0) * math.exp(-4.0 * max(z, 0.0) / height)) for _, _, z in path]
    trunk = tube(name, path, radii, sides=sides, jitter=0.05, seed=seed)
    if jag > 0.0:
        top = max(vertex.co.z for vertex in trunk.data.vertices)
        phase = rng.uniform(0.0, math.pi)
        for vertex in trunk.data.vertices:
            if vertex.co.z > top - 1e-4:
                angle = math.atan2(vertex.co.y, vertex.co.x)
                vertex.co.z += jag * abs(math.sin(angle * 2.5 + phase)) * rng.uniform(0.3, 1.0)
        _triangulate_ngons(trunk)
    roots = []
    for index in range(buttresses):
        angle = 2.0 * math.pi * (index + rng.uniform(-0.25, 0.25)) / buttresses
        reach = radius * rng.uniform(1.6, 2.4)
        lift = height * rng.uniform(0.35, 0.55)
        direction = Vector((math.cos(angle), math.sin(angle), 0.0))
        root_path = []
        for step in range(5):
            t = step / 4.0
            point = direction * (radius * 0.4 + reach * t)
            point.z = lift * (1.0 - t) ** 1.6 - 0.12 * t
            root_path.append(point)
        thickness = radius * rng.uniform(0.32, 0.42)
        root_radii = [thickness * (1.0 - 0.7 * step / 4.0) for step in range(5)]
        root = tube(f"{name}_root{index}", root_path, root_radii, sides=10, jitter=0.06, seed=seed + index + 1, top="dome", dome_rings=1)
        roots.append(_uniform_material(root, SIDE))
    union(trunk, roots)
    return _finish(trunk, target_tris)


def root_cluster(
    name: str,
    seed: int,
    roots: int = 5,
    reach: float = 1.2,
    thickness: float = 0.09,
    arch: float = 0.18,
    target_tris: int = 1500,
) -> bpy.types.Object:
    """Gnarled exposed roots arching out of the ground from a low central knot.

    Every face carries material index SIDE. Origin at base centre.
    """
    rng = random.Random(seed)
    offset = Vector((rng.uniform(-50, 50), rng.uniform(-50, 50), 0.0))
    knot = tube(
        name,
        [(0.0, 0.0, -0.1), (0.0, 0.0, 0.05), (0.02, 0.0, thickness * 2.0), (0.03, 0.01, thickness * 3.2)],
        [thickness * 3.6, thickness * 3.2, thickness * 2.7, thickness * 2.3],
        sides=12,
        jitter=0.1,
        seed=seed,
        top="dome",
        dome_rings=2,
    )
    _uniform_material(knot, SIDE)
    parts = []
    steps = 7
    for index in range(roots):
        angle = 2.0 * math.pi * (index + rng.uniform(-0.3, 0.3)) / roots
        length = reach * rng.uniform(0.6, 1.0)
        height = arch * rng.uniform(0.6, 1.2)
        path = []
        for step in range(steps):
            t = step / (steps - 1)
            wander = angle + 0.5 * noise.noise(offset + Vector((index * 3.1, t * 2.0, 0.0)))
            radial = length * t
            dive = thickness * 2.5 * max(t - 0.8, 0.0) / 0.2
            sink = thickness * 1.2 * max(noise.noise(offset + Vector((index * 5.7, t * 3.5, 1.0))), 0.0)
            point = Vector((math.cos(wander) * radial, math.sin(wander) * radial, 0.0))
            point.z = thickness * 1.2 * (1.0 - t) + height * math.sin(math.pi * min(t * 1.25, 1.0)) - dive - sink
            path.append(point)
        radii = [thickness * rng.uniform(0.85, 1.15) * (1.7 - 1.4 * (step / (steps - 1)) ** 0.7) for step in range(steps)]
        root = tube(f"{name}_root{index}", path, radii, sides=9, jitter=0.08, seed=seed + index + 1, top="dome", dome_rings=1)
        parts.append(_uniform_material(root, SIDE))
    union(knot, parts)
    return _finish(knot, target_tris)


def revolve(
    name: str,
    profile: Sequence[tuple[float, float, int]],
    segments: int = 16,
    seed: int = 0,
    wobble: float = 0.0,
) -> bpy.types.Object:
    """A closed surface of revolution from (radius, z, material_index) points listed bottom to
    top. A zero-radius first or last point collapses to an apex; the band from point i to i + 1
    takes point i + 1's material index, a flat end cap its own point's. wobble roughens every
    ring radially with smooth noise."""
    offset = Vector((seed * 7.31 % 97.0, seed * 3.17 % 89.0, 0.0))
    bm = bmesh.new()
    rings = []
    for radius, z, _ in profile:
        if radius <= 1e-6:
            rings.append([bm.verts.new((0.0, 0.0, z))])
            continue
        ring = []
        for step in range(segments):
            angle = 2.0 * math.pi * step / segments
            sample = offset + Vector((math.cos(angle) * 1.3, math.sin(angle) * 1.3, z * 2.0))
            scale = 1.0 + wobble * noise.noise(sample)
            ring.append(bm.verts.new((radius * scale * math.cos(angle), radius * scale * math.sin(angle), z)))
        rings.append(ring)
    for index in range(len(rings) - 1):
        lower = rings[index]
        upper = rings[index + 1]
        material = profile[index + 1][2]
        for step in range(segments):
            following = (step + 1) % segments
            if len(lower) == 1:
                corners = (lower[0], upper[following], upper[step])
            elif len(upper) == 1:
                corners = (lower[step], lower[following], upper[0])
            else:
                corners = (lower[step], lower[following], upper[following], upper[step])
            face = bm.faces.new(corners)
            face.material_index = material
    if len(rings[0]) > 1:
        bottom = bm.faces.new(list(reversed(rings[0])))
        bottom.material_index = profile[0][2]
    if len(rings[-1]) > 1:
        top = bm.faces.new(rings[-1])
        top.material_index = profile[-1][2]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _link(mesh, name)


def _mushroom_body(
    name: str,
    seed: int,
    stem_height: float,
    stem_radius: float,
    cap_radius: float,
    cap_height: float,
    segments: int,
) -> bpy.types.Object:
    h = stem_height
    r = stem_radius
    c = cap_height
    big = cap_radius
    profile = [
        (r * 1.2, -0.08 * h, SIDE),
        (r * 1.15, 0.1 * h, SIDE),
        (r * 1.0, 0.5 * h, SIDE),
        (r * 0.92, h, SIDE),
        (big * 0.75, h + 0.12 * c, SIDE),
        (big, h + 0.22 * c, CAP),
        (big * 0.97, h + 0.42 * c, CAP),
        (big * 0.82, h + 0.7 * c, CAP),
        (big * 0.5, h + 0.92 * c, CAP),
        (0.0, h + c, CAP),
    ]
    return revolve(name, profile, segments=segments, seed=seed, wobble=0.06)


def mushroom(
    name: str,
    seed: int,
    stem_height: float = 0.14,
    stem_radius: float = 0.03,
    cap_radius: float = 0.09,
    cap_height: float = 0.07,
    segments: int = 20,
    target_tris: int = 1200,
) -> bpy.types.Object:
    """A single mushroom: bulbous stem, gilled underside and domed cap.

    Stem and gills carry material index SIDE, the cap top CAP. Origin at base centre.
    """
    body = _mushroom_body(name, seed, stem_height, stem_radius, cap_radius, cap_height, segments)
    return _finish(body, target_tris)


def mushroom_cluster(
    name: str,
    seed: int,
    count: int = 5,
    spread: float = 0.12,
    size: float = 1.0,
    lean_deg: float = 22.0,
    segments: int = 14,
    target_tris: int = 1800,
) -> bpy.types.Object:
    """A clump of mushrooms of mixed sizes leaning out from a shared base.

    size scales every member; spread is the base radius of the clump in metres. Material indices
    as in mushroom. Origin at base centre.
    """
    rng = random.Random(seed)
    members = []
    for index in range(count):
        scale = size * (1.0 if index == 0 else rng.uniform(0.45, 0.85))
        body = _mushroom_body(
            name if index == 0 else f"{name}_m{index}",
            seed + index,
            0.14 * scale * rng.uniform(0.85, 1.2),
            0.026 * scale,
            0.075 * scale * rng.uniform(0.85, 1.15),
            0.06 * scale,
            segments,
        )
        angle = 2.0 * math.pi * index / count + rng.uniform(-0.4, 0.4)
        distance = 0.0 if index == 0 else spread * rng.uniform(0.5, 1.0)
        lean = math.radians(0.0 if index == 0 else lean_deg * rng.uniform(0.6, 1.2))
        axis = Vector((-math.sin(angle), math.cos(angle), 0.0))
        base = Vector((math.cos(angle) * distance, math.sin(angle) * distance, 0.0))
        spin = Matrix.Rotation(rng.uniform(0.0, 2.0 * math.pi), 4, "Z")
        _transform(body, Matrix.Translation(base) @ Matrix.Rotation(lean, 4, axis) @ spin)
        members.append(body)
    union(members[0], members[1:])
    return _finish(members[0], target_tris)


def branching_coral(
    name: str,
    seed: int,
    height: float = 0.6,
    radius: float = 0.045,
    depth: int = 2,
    forks: tuple[int, int] = (2, 3),
    spread_deg: float = 38.0,
    sides: int = 10,
    target_tris: int = 1900,
) -> bpy.types.Object:
    """A staghorn-style coral: a trunk forking depth times into round-tipped branches.

    Every face carries material index SIDE. Origin at base centre.
    """
    rng = random.Random(seed)
    branches = []

    def grow(start: Vector, direction: Vector, length: float, thickness: float, level: int) -> None:
        points = [start]
        headings = [direction]
        heading = direction.copy()
        for _ in range(4):
            heading = (heading + Vector((rng.gauss(0, 0.15), rng.gauss(0, 0.15), 0.1))).normalized()
            points.append(points[-1] + heading * (length / 4.0))
            headings.append(heading)
        radii = [thickness * (1.0 - 0.35 * step / 4.0) for step in range(5)]
        branches.append((points, radii))
        if level >= depth:
            return
        for _ in range(rng.randint(*forks)):
            fork = rng.choice((2, 3))
            along = headings[fork]
            axis = Matrix.Rotation(rng.uniform(0.0, 2.0 * math.pi), 3, along) @ _frame(along, None)
            turned = Matrix.Rotation(math.radians(rng.uniform(18.0, spread_deg)), 3, axis) @ along
            grow(points[fork], turned.normalized(), length * 0.7, radii[fork] * 0.78, level + 1)

    grow(Vector((0.0, 0.0, -radius)), Vector((0.0, 0.0, 1.0)), height * 0.45, radius, 0)
    tubes = [
        tube(name if index == 0 else f"{name}_b{index}", points, radii, sides=sides, seed=seed + index, top="dome", dome_rings=2)
        for index, (points, radii) in enumerate(branches)
    ]
    for obj in tubes:
        _uniform_material(obj, SIDE)
    union(tubes[0], tubes[1:])
    return _finish(tubes[0], target_tris)


def dome_coral(
    name: str,
    seed: int,
    radius: float = 0.4,
    height: float = 0.32,
    lobes: float = 0.16,
    subdivisions: int = 4,
    target_tris: int = 1600,
) -> bpy.types.Object:
    """A brain or boulder coral: a lobed dome sliced flat on the sea floor.

    The grooves live in the material, not the mesh. Every face carries material index SIDE.
    Origin at base centre.
    """
    rng = random.Random(seed)
    offset = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=1.0)
    for vertex in bm.verts:
        direction = vertex.co.normalized()
        bump = noise.fractal(direction * 1.4 + offset, 0.5, 2.0, 3)
        point = direction * (1.0 + lobes * bump)
        vertex.co = Vector((point.x * radius, point.y * radius, point.z * height - 0.25 * height))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _finish(_link(mesh, name), target_tris)
