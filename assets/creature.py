from __future__ import annotations

import random

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

from assets.creature_spec import Bone, Species
from assets.hardsurface import _link, apply_modifier
from assets.organic import WELD_M, _repair_non_manifold, ground_cut, settle

SMOOTH_BANDS = 5
SMOOTH_MIN_VOXELS = 1.5
SMOOTH_FULL_RADIUS_M = 0.05


def _capsule(bm: bmesh.types.BMesh, bone: Bone, segments: int) -> None:
    head = Vector(bone.head)
    tail = Vector(bone.tail)
    axis = tail - head
    rotation = axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
    middle = (head + tail) * 0.5
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=segments,
        radius1=bone.radius[0],
        radius2=bone.radius[1],
        depth=axis.length,
        matrix=Matrix.Translation(middle) @ rotation,
    )
    for point, radius in ((head, bone.radius[0]), (tail, bone.radius[1])):
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=segments,
            v_segments=max(6, segments // 2),
            radius=radius,
            matrix=Matrix.Translation(point),
        )


def skeleton_mesh(name: str, species: Species, segments: int = 16) -> bpy.types.Object:
    """Every bone as an overlapping tapered capsule in one mesh; not yet a single surface."""
    bm = bmesh.new()
    for bone in species.bones:
        _capsule(bm, bone, segments)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return _link(mesh, name)


def _remesh(obj: bpy.types.Object, voxel_m: float) -> bpy.types.Object:
    modifier = obj.modifiers.new("remesh", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = voxel_m
    modifier.adaptivity = 0.0
    modifier.use_smooth_shade = False
    return apply_modifier(obj, modifier)


def _smooth(bm: bmesh.types.BMesh, species: Species) -> None:
    """Fillet the capsule joints, but only as hard as the local bone is thick: a horn tip or
    tine a few voxels wide would shrink away under the same smoothing as the torso."""
    bands: list[list[bmesh.types.BMVert]] = [[] for _ in range(SMOOTH_BANDS)]
    for vertex in bm.verts:
        point = tuple(vertex.co)
        nearest = min(species.bones, key=lambda bone: bone.surface_distance(point))
        t, _ = nearest.closest(point)
        radius = nearest.radius[0] + (nearest.radius[1] - nearest.radius[0]) * t
        weight = (radius - SMOOTH_MIN_VOXELS * species.voxel_m) / SMOOTH_FULL_RADIUS_M
        band = min(SMOOTH_BANDS - 1, int(max(0.0, weight) * SMOOTH_BANDS))
        bands[band].append(vertex)
    for _ in range(species.smooth):
        for band, verts in enumerate(bands):
            if band and verts:
                factor = 0.5 * band / (SMOOTH_BANDS - 1)
                bmesh.ops.smooth_vert(bm, verts=verts, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)


def _displace(bm: bmesh.types.BMesh, species: Species) -> None:
    if species.noise_m <= 0.0:
        return
    rng = random.Random(species.seed)
    offset = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    bm.normal_update()
    moves = []
    for vertex in bm.verts:
        sample = vertex.co / species.noise_scale_m + offset
        moves.append(vertex.normal * species.noise_m * noise.fractal(sample, 0.6, 2.0, 3))
    for vertex, move in zip(bm.verts, moves):
        vertex.co += move


def _surface(obj: bpy.types.Object, species: Species) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    _smooth(bm, species)
    _displace(bm, species)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _keep_largest_shell(obj: bpy.types.Object) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    unvisited = set(bm.faces)
    shells = []
    while unvisited:
        seed = unvisited.pop()
        shell = {seed}
        stack = [seed]
        while stack:
            face = stack.pop()
            for edge in face.edges:
                for other in edge.link_faces:
                    if other in unvisited:
                        unvisited.remove(other)
                        shell.add(other)
                        stack.append(other)
        shells.append(shell)
    if len(shells) > 1:
        largest = max(shells, key=len)
        doomed = [face for shell in shells if shell is not largest for face in shell]
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bmesh.ops.delete(bm, geom=[vertex for vertex in bm.verts if not vertex.link_faces], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def _decimate(obj: bpy.types.Object, target_tris: int) -> bpy.types.Object:
    """Collapse to target_tris mirrored about X, so left and right stay the same creature."""
    tris = sum(len(polygon.vertices) - 2 for polygon in obj.data.polygons)
    if tris <= target_tris:
        return obj
    modifier = obj.modifiers.new("decimate", "DECIMATE")
    modifier.ratio = target_tris / tris
    modifier.use_collapse_triangulate = True
    modifier.use_symmetry = True
    modifier.symmetry_axis = "X"
    obj = apply_modifier(obj, modifier)
    obj.data.validate(clean_customdata=False)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD_M)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=WELD_M)
    bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 3])
    _repair_non_manifold(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def paint_regions(obj: bpy.types.Object, species: Species) -> list[str]:
    """Give each face the material index of the region of the bone whose capsule surface lies
    nearest its centre; downward hide faces become underside where the palette has one. Returns
    the region per material index."""
    regions = species.regions()
    slot = {region: index for index, region in enumerate(regions)}
    indices = []
    for polygon in obj.data.polygons:
        region = species.region_at(tuple(polygon.center), polygon.normal.z)
        indices.append(slot[region])
    obj.data.polygons.foreach_set("material_index", indices)
    obj.data.update()
    return regions


def creature(name: str, species: Species, segments: int = 16) -> bpy.types.Object:
    """Skin species' skeleton into one closed body: capsules, voxel remesh (one manifold surface
    however the capsules overlap), smoothing that fillets the joints, normal-direction fractal
    noise, a sink below z = 0 and ground_cut for flat soles, region material indices
    (species.regions() order) painted on the dense surface so small accents survive, and a
    mirrored decimation to target_tris. Origin at base centre, +Y forward."""
    obj = skeleton_mesh(name, species, segments)
    _remesh(obj, species.voxel_m)
    _keep_largest_shell(obj)
    _surface(obj, species)
    settle(obj)
    obj.data.transform(Matrix.Translation((0.0, 0.0, -species.sink_m)))
    ground_cut(obj)
    paint_regions(obj, species)
    obj = _decimate(obj, species.target_tris)
    settle(obj)
    _centre(obj)
    return obj


def _centre(obj: bpy.types.Object) -> None:
    xs = [vertex.co.x for vertex in obj.data.vertices]
    ys = [vertex.co.y for vertex in obj.data.vertices]
    shift = Vector((-(min(xs) + max(xs)) * 0.5, -(min(ys) + max(ys)) * 0.5, 0.0))
    if shift.length > 1e-6:
        obj.data.transform(Matrix.Translation(shift))
        obj.data.update()

