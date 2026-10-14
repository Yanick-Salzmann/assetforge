"""Composable bpy/bmesh helpers for hard-surface modelling.

Runs inside Blender's own interpreter, never the uv venv - only bpy, bmesh, mathutils and the
stdlib. A live MCP session imports this by inserting the repo root into sys.path; the headless
self-test in assets/blender_scripts/hardsurface_selftest.py does the same via subprocess.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import bmesh
import bpy
from mathutils import Matrix, Vector

Size2 = tuple[float, float]
Size3 = tuple[float, float, float]

STANDOFF_M = 0.002


def _link(mesh: bpy.types.Mesh, name: str, location: Sequence[float] = (0.0, 0.0, 0.0)) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj


def _select_only(obj: bpy.types.Object) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transforms(obj: bpy.types.Object, location: bool = True, rotation: bool = True, scale: bool = True) -> bpy.types.Object:
    """Bake object-level location/rotation/scale into the mesh, per the origin-at-base-centre convention."""
    _select_only(obj)
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)
    return obj


def recalc_normals(obj: bpy.types.Object) -> bpy.types.Object:
    """Recompute consistent outward face normals, e.g. after a boolean leaves them mixed."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> bpy.types.Object:
    _select_only(obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def bevelled_box(
    name: str,
    size: Size3 = (1.0, 1.0, 1.0),
    bevel_width: float = 0.02,
    bevel_segments: int = 2,
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """A box with local origin at base centre (Z-up), chamfered along every edge."""
    sx, sy, sz = size
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy
        v.co.z = (v.co.z + 0.5) * sz
    if bevel_width > 0.0 and bevel_segments > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_width, segments=bevel_segments, affect="EDGES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = _link(mesh, name, location)
    return apply_transforms(obj)


def frustum(
    name: str,
    r0: float,
    r1: float,
    length: float,
    segments: int = 8,
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """An N-gon frustum (a cylinder when r0 == r1, a cone/pyramid when r1 == 0) running from the
    origin along local +Z to length, base radius r0 and top radius r1. Capped on any end whose
    radius is non-zero. A tower section, mast, post or turret base - anything cylindrical - is
    this shape, optionally reoriented afterward (see cylinder_between for connecting two arbitrary
    points).

    A zero (or near-zero) radius end collapses to a single shared apex vertex rather than
    `segments` distinct vertices stacked at the same point - the latter looks identical in the
    viewport but leaves the side faces degenerate (zero area) and the apex edges non-manifold
    (shared by more than two triangles once positions are merged), which only shows up downstream
    in validate_asset, not here."""
    apex_eps = 1e-6
    bottom_is_point = r0 <= apex_eps
    top_is_point = r1 <= apex_eps

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bottom_apex = bm.verts.new((0.0, 0.0, 0.0)) if bottom_is_point else None
    top_apex = bm.verts.new((0.0, 0.0, length)) if top_is_point else None

    bottom, top = [], []
    base_angle = math.pi / segments
    step = 2.0 * math.pi / segments
    for k in range(segments):
        angle = base_angle + k * step
        c, s = math.cos(angle), math.sin(angle)
        bottom.append(bottom_apex if bottom_is_point else bm.verts.new((r0 * c, r0 * s, 0.0)))
        top.append(top_apex if top_is_point else bm.verts.new((r1 * c, r1 * s, length)))
    bm.verts.ensure_lookup_table()
    for k in range(segments):
        k2 = (k + 1) % segments
        face_verts = []
        for v in (bottom[k], bottom[k2], top[k2], top[k]):
            if v not in face_verts:
                face_verts.append(v)
        if len(face_verts) >= 3:
            bm.faces.new(face_verts)
    if not bottom_is_point:
        bm.faces.new(bottom[::-1])
    if not top_is_point:
        bm.faces.new(top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = _link(mesh, name, location)
    return apply_transforms(obj)


def cylinder_between(
    name: str,
    point_a: Sequence[float],
    point_b: Sequence[float],
    r0: float,
    r1: float | None = None,
    segments: int = 8,
) -> bpy.types.Object:
    """A frustum spanning point_a to point_b, oriented via to_track_quat from the two endpoints
    rather than a hand-picked rotation Euler - guessing the Euler sign for a connecting shaft
    (windshaft, mast, strut) is exactly the mistake that leaves it pointing back into the parent
    structure instead of out to the point it was meant to reach, and to_track_quat cannot get the
    direction backwards the way a manually-chosen rotation can."""
    a = Vector(point_a)
    b = Vector(point_b)
    direction = b - a
    length = direction.length
    obj = frustum(name, r0, r0 if r1 is None else r1, length, segments=segments)
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    apply_transforms(obj, location=False, rotation=True, scale=False)
    obj.location = a
    return apply_transforms(obj)


def attach_to_surface(
    obj: bpy.types.Object,
    point: Sequence[float],
    normal: Sequence[float],
    clearance: float = STANDOFF_M,
) -> bpy.types.Object:
    """Place a small hardware part (hinge, handle, bracket, sign, trim) - authored with its own
    local origin at its base centre, like bevelled_box and frustum - flush against a host
    surface at point, oriented so its local +Z follows normal outward from the host.

    Flush-mounted parts are conventionally modelled as their own independent closed solid and
    only need to look attached at render distance, but game_ready_pass's final bpy.ops.object.
    join() concatenates every part's mesh data into one primitive, and validate_asset's manifold
    check re-welds vertices by rounded position (WELD_PRECISION, 1e-5 m) across that whole
    primitive before judging edge adjacency. A part whose base sits exactly on, or a few cm
    embedded in, the host surface produces vertices that weld onto the host's own, turning a
    harmless touching-solids seam into edges shared by more than two triangles - reported as
    genuine non-manifold geometry even though nothing is actually broken (af-fte). clearance
    must clear that 1e-5 m rounding by a wide margin - the STANDOFF_M default (2 mm) is far
    below anything visible at building/prop scale but far above the weld threshold.
    """
    normal_v = Vector(normal).normalized()
    obj.location = Vector(point) + normal_v * clearance
    obj.rotation_euler = normal_v.to_track_quat("Z", "Y").to_euler()
    return apply_transforms(obj)


def panel_cut(obj: bpy.types.Object, cutter: bpy.types.Object, operation: str = "DIFFERENCE") -> bpy.types.Object:
    """Boolean-cut obj with cutter (a window, door or panel-line volume), then discard the cutter."""
    modifier = obj.modifiers.new(name="panel_cut", type="BOOLEAN")
    modifier.operation = operation
    modifier.object = cutter
    modifier.solver = "EXACT"
    apply_modifier(obj, modifier)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return recalc_normals(obj)


def array(obj: bpy.types.Object, count: int, offset: Size3, apply: bool = True) -> bpy.types.Object:
    """Repeat obj count times along a constant per-copy offset."""
    modifier = obj.modifiers.new(name="array", type="ARRAY")
    modifier.count = count
    modifier.use_relative_offset = False
    modifier.use_constant_offset = True
    modifier.constant_offset_displace = offset
    if apply:
        apply_modifier(obj, modifier)
    return obj


def mirror(obj: bpy.types.Object, axis: str = "X", merge: bool = True, apply: bool = True) -> bpy.types.Object:
    """Mirror obj across its local origin plane on the given axis, welding the seam by default.

    obj must be open (no cap face) on the side touching the mirror plane - mirroring a fully
    closed solid whose face lies exactly on that plane produces two overlapping caps and a
    non-manifold seam instead of one continuous shell.
    """
    axis = axis.upper()
    modifier = obj.modifiers.new(name="mirror", type="MIRROR")
    modifier.use_axis = (axis == "X", axis == "Y", axis == "Z")
    modifier.use_bisect_axis = (axis == "X", axis == "Y", axis == "Z")
    modifier.use_mirror_merge = merge
    if apply:
        apply_modifier(obj, modifier)
    return obj


def solidify(obj: bpy.types.Object, thickness: float, apply: bool = True) -> bpy.types.Object:
    """Thicken a shell (e.g. a panel or roof plane) into closed, manifold geometry."""
    modifier = obj.modifiers.new(name="solidify", type="SOLIDIFY")
    modifier.thickness = thickness
    modifier.offset = -1.0
    if apply:
        apply_modifier(obj, modifier)
    return recalc_normals(obj)


def greeble(
    obj: bpy.types.Object,
    face_indices: Iterable[int] | None = None,
    count: int = 8,
    depth_range: tuple[float, float] = (0.01, 0.05),
    inset_ratio: float = 0.25,
    seed: int = 0,
) -> bpy.types.Object:
    """Punch a handful of small inset-and-extrude greebles into a face selection for hard-surface detail."""
    rng = random.Random(seed)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    if face_indices is not None:
        candidate_faces = [bm.faces[i] for i in face_indices if 0 <= i < len(bm.faces)]
    else:
        candidate_faces = list(bm.faces)
    rng.shuffle(candidate_faces)
    for face in candidate_faces[:count]:
        if not face.is_valid:
            continue
        thickness = face.calc_perimeter() * inset_ratio * 0.1
        bmesh.ops.inset_individual(bm, faces=[face], thickness=thickness, use_even_offset=True)
        normal = face.normal.copy()
        extrude_result = bmesh.ops.extrude_discrete_faces(bm, faces=[face])
        depth = rng.uniform(*depth_range)
        for vert in extrude_result["faces"][0].verts:
            vert.co += normal * depth
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def _rotation_from_normal(normal: Sequence[float]):
    """Orient a cutter box so its local Y (thickness/through-cut axis) points along normal,
    keeping local Z (height) pointing world-up - swapping these leaves the through-cut axis
    vertical and the height axis horizontal, under-cutting the wall's actual thickness."""
    return Vector(normal).to_track_quat("Y", "Z").to_euler()


def _cutter_box(name: str, size: Size3, centre: Sequence[float], normal: Sequence[float]) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    sx, sy, sz = size
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy
        v.co.z *= sz
    bm.to_mesh(mesh)
    bm.free()
    obj = _link(mesh, name, centre)
    obj.rotation_euler = _rotation_from_normal(normal)
    apply_transforms(obj, location=False, rotation=True, scale=False)
    return obj


def window_recess(
    obj: bpy.types.Object,
    centre: Sequence[float],
    size: Size2,
    depth: float,
    wall_thickness: float,
    wall_normal: Sequence[float] = (0.0, -1.0, 0.0),
) -> bpy.types.Object:
    """Cut a window opening through the wall and leave a recessed sill, centred at a local point."""
    width, height = size
    through = _cutter_box(
        "window_through", (width, wall_thickness * 4.0, height), centre, wall_normal
    )
    obj = panel_cut(obj, through)
    sill_centre = Vector(centre) - Vector(wall_normal).normalized() * (wall_thickness * 0.5 - depth * 0.5)
    recess = _cutter_box(
        "window_recess", (width * 1.3, depth, height * 1.3), sill_centre, wall_normal
    )
    return panel_cut(obj, recess)


def door_frame(
    obj: bpy.types.Object,
    centre: Sequence[float],
    size: Size2,
    wall_thickness: float,
    frame_width: float = 0.05,
    wall_normal: Sequence[float] = (0.0, -1.0, 0.0),
) -> bpy.types.Object:
    """Cut a door opening and leave a raised frame lip around it.

    Both the outer trim and the inner cutter that hollows it must be centred on the same point as
    the opening itself (in X and Z) - centring one of them on a point offset from the opening, as
    an earlier version of this function did for the outer box's Z, leaves the trim covering only
    part of the doorway with the rest unframed."""
    width, height = size
    centre_v = Vector(centre)
    opening = _cutter_box(
        "door_opening", (width, wall_thickness * 4.0, height), centre, wall_normal
    )
    obj = panel_cut(obj, opening)
    outer_height = height + frame_width * 2.0
    outer = bevelled_box(
        "door_frame",
        (width + frame_width * 2.0, wall_thickness * 0.5, outer_height),
        bevel_width=frame_width * 0.3,
        bevel_segments=1,
        location=centre_v - Vector((0.0, wall_thickness * 0.25, outer_height * 0.5)),
    )
    inner = bevelled_box(
        "door_frame_inner",
        (width, wall_thickness * 0.6, height),
        bevel_width=0.0,
        location=centre_v - Vector((0.0, wall_thickness * 0.3, height * 0.5)),
    )
    return panel_cut(outer, inner)


def roof_edge(
    plane: bpy.types.Object,
    overhang: float,
    thickness: float,
) -> bpy.types.Object:
    """Extend a roof plane's border by overhang and give it real thickness via solidify."""
    bm = bmesh.new()
    bm.from_mesh(plane.data)
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        original_verts = {v for e in boundary_edges for v in e.verts}
        centre = sum((v.co for v in original_verts), Vector()) / len(original_verts)
        result = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
        new_verts = [g for g in result["geom"] if isinstance(g, bmesh.types.BMVert)]
        for new_vert in new_verts:
            old_vert = next(
                (e.other_vert(new_vert) for e in new_vert.link_edges if e.other_vert(new_vert) in original_verts),
                None,
            )
            if old_vert is None:
                continue
            direction = old_vert.co - centre
            direction.z = 0.0
            if direction.length > 1e-9:
                new_vert.co = old_vert.co + direction.normalized() * overhang
    bm.to_mesh(plane.data)
    bm.free()
    plane.data.update()
    return solidify(plane, thickness)


def gable_roof(
    name: str,
    span: float,
    length: float,
    ridge_height: float,
    overhang: float,
    thickness: float,
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Two roof slopes meeting at a ridge, built as one connected shell before solidify.

    Two independently-solidified planes joined afterward each rim-cap their own edge at the
    ridge, leaving it shared by more than two triangles once the join welds the seam (af-rs6).
    Building both slopes as a single bmesh first makes the ridge an ordinary interior edge -
    shared by exactly the two slope faces - so roof_edge's boundary walk (and the solidify it
    calls) only rim-caps the true perimeter (eaves and gable ends), never the ridge."""
    half_span = span * 0.5
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    eave_left_near = bm.verts.new((-half_span, 0.0, 0.0))
    eave_left_far = bm.verts.new((-half_span, length, 0.0))
    ridge_near = bm.verts.new((0.0, 0.0, ridge_height))
    ridge_far = bm.verts.new((0.0, length, ridge_height))
    eave_right_near = bm.verts.new((half_span, 0.0, 0.0))
    eave_right_far = bm.verts.new((half_span, length, 0.0))
    bm.faces.new((eave_left_near, eave_left_far, ridge_far, ridge_near))
    bm.faces.new((ridge_near, ridge_far, eave_right_far, eave_right_near))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = _link(mesh, name, location)
    apply_transforms(obj)
    return roof_edge(obj, overhang, thickness)


def ladder_frame(
    name: str,
    length: float,
    width: float,
    bar_size: float = 0.08,
    rung_count: int = 6,
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[bpy.types.Object]:
    """Two parallel rails spanning the full local-Z length, base at the origin, plus rung_count
    perpendicular rungs bridging them. Unlike a single centre spine with spaced-out cross slats,
    the rails give the shape a continuous silhouette edge along its whole length, so a long thin
    assembly (a sail frame, a truss, a fence run) still reads as one coherent shape instead of
    scattered dashes when its long axis points away from the camera."""
    half_width = width * 0.5
    bevel_width = bar_size * 0.15
    rail_a = bevelled_box(f"{name}_rail_a", size=(bar_size, bar_size, length), bevel_width=bevel_width, bevel_segments=1, location=(-half_width, 0.0, 0.0))
    rail_b = bevelled_box(f"{name}_rail_b", size=(bar_size, bar_size, length), bevel_width=bevel_width, bevel_segments=1, location=(half_width, 0.0, 0.0))
    objects = [rail_a, rail_b]
    for i in range(rung_count):
        t = length * (i + 0.5) / rung_count
        rung = bevelled_box(f"{name}_rung_{i}", size=(width, bar_size, bar_size), bevel_width=bevel_width, bevel_segments=1, location=(0.0, 0.0, t))
        objects.append(rung)
    if any(location):
        for obj in objects:
            obj.location = location
            apply_transforms(obj)
    return objects


def stairs(
    name: str,
    tread_count: int,
    total_rise: float,
    total_run: float,
    width: float,
    location: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, -1.0, 0.0),
) -> list[bpy.types.Object]:
    """tread_count solid risers ascending from location (ground level, nearest the target) out
    along direction, each one farther away and shorter than the last. Height and distance both
    come from the same loop index, so a tread can't end up farther away yet taller than the one
    in front of it - the exact mistake that put the windmill's tallest step behind its shortest
    one instead of the other way around."""
    d = Vector(direction).normalized()
    rise_per_tread = total_rise / tread_count
    run_per_tread = total_run / tread_count
    origin = Vector(location)
    objects = []
    for i in range(tread_count):
        height = total_rise - rise_per_tread * i
        near_offset = run_per_tread * i
        centre = origin + d * (near_offset + run_per_tread * 0.5)
        tread = bevelled_box(
            f"{name}_tread_{i}",
            size=(width, run_per_tread, height),
            bevel_width=min(run_per_tread, height) * 0.15,
            bevel_segments=1,
            location=(centre.x, centre.y, origin.z),
        )
        objects.append(tread)
    return objects


def cornice(
    wall: bpy.types.Object,
    length: float,
    profile_size: Size2,
    location: Sequence[float] = (0.0, 0.0, 0.0),
    segments: int = 1,
    axis: str = "X",
) -> bpy.types.Object:
    """A bevelled trim strip run along a wall's top edge; returns the new cornice object."""
    width, depth = profile_size
    long_size = (length, depth, width) if axis.upper() == "X" else (depth, length, width)
    strip = bevelled_box(
        "cornice",
        long_size,
        bevel_width=min(width, depth) * 0.2,
        bevel_segments=segments,
        location=location,
    )
    return strip


def _finish(bm: bmesh.types.BMesh, name: str) -> bpy.types.Object:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return apply_transforms(_link(mesh, name))


def _axis_point(axis: str, axial: float, u: float, v: float) -> tuple[float, float, float]:
    """Map an axial coordinate and an in-plane (u, v) pair to world XYZ: axis X takes (u, v) as
    (y, z), Y as (x, z), Z as (x, y)."""
    axis = axis.upper()
    if axis == "X":
        return (axial, u, v)
    if axis == "Y":
        return (u, axial, v)
    if axis == "Z":
        return (u, v, axial)
    raise ValueError(f"axis {axis!r} must be X, Y or Z")


def _axis_coordinates(axis: str, point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = point
    axis = axis.upper()
    if axis == "X":
        return (x, y, z)
    if axis == "Y":
        return (y, x, z)
    return (z, x, y)


def _bridge_rings(bm: bmesh.types.BMesh, rings: list[list], closed: bool = True) -> None:
    count = len(rings[0])
    for lower, upper in zip(rings, rings[1:]):
        for k in range(count if closed else count - 1):
            k2 = (k + 1) % count
            face_verts = []
            for v in (lower[k], lower[k2], upper[k2], upper[k]):
                if v not in face_verts:
                    face_verts.append(v)
            bm.faces.new(face_verts)


def box_between(
    name: str,
    point_a: Sequence[float],
    point_b: Sequence[float],
    width: float,
    height: float,
    bevel_width: float = 0.0,
    up: Sequence[float] | None = None,
) -> bpy.types.Object:
    """A width x height bar running from point_a to point_b - a beam, strut, brace, link arm or
    tie rod. Like cylinder_between it is oriented from the two endpoints, never a guessed Euler.
    The width runs along up projected off the bar axis when given (e.g. a wall tangent, so a
    brace lies flat against a curved tower), otherwise along whatever to_track_quat picks."""
    a = Vector(point_a)
    b = Vector(point_b)
    direction = b - a
    segments = 1 if bevel_width > 0.0 else 0
    obj = bevelled_box(name, size=(width, height, direction.length), bevel_width=bevel_width, bevel_segments=segments)
    if up is None:
        obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    else:
        z_axis = direction.normalized()
        y_axis = z_axis.cross(Vector(up)).normalized()
        x_axis = y_axis.cross(z_axis).normalized()
        obj.rotation_euler = Matrix((x_axis, y_axis, z_axis)).transposed().to_euler()
    apply_transforms(obj, location=False, rotation=True, scale=False)
    obj.location = a
    return apply_transforms(obj)


def lathe(
    name: str,
    profile: Sequence[Size2],
    segments: int = 16,
    location: Sequence[float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Revolve a (radius, z) profile, listed bottom to top, around local Z - a barrel, bottle,
    baluster, vase, finial, bell or turned leg. Both ends are capped; an end whose radius is zero
    collapses to one apex vertex, as in frustum. Interior radii must be positive."""
    apex_eps = 1e-6
    last = len(profile) - 1
    bm = bmesh.new()
    rings: list[list] = []
    for index, (radius, z) in enumerate(profile):
        if radius <= apex_eps and 0 < index < last:
            raise ValueError("only the first and last profile points may have zero radius")
        if radius <= apex_eps:
            rings.append([bm.verts.new((0.0, 0.0, z))] * segments)
            continue
        ring = []
        for k in range(segments):
            angle = math.pi / segments + k * 2.0 * math.pi / segments
            ring.append(bm.verts.new((radius * math.cos(angle), radius * math.sin(angle), z)))
        rings.append(ring)
    _bridge_rings(bm, rings)
    if profile[0][0] > apex_eps:
        bm.faces.new(rings[0][::-1])
    if profile[-1][0] > apex_eps:
        bm.faces.new(rings[-1])
    obj = _finish(bm, name)
    obj.location = location
    return apply_transforms(obj)


def loft(name: str, sections: Sequence[Sequence[Sequence[float]]]) -> bpy.types.Object:
    """Skin a closed solid through two or more closed cross-section loops of world-space points,
    capped at the first and last - a hood, hull, fuselage, nacelle or tapered housing. Every
    section needs the same point count, listed in the same rotational order and starting at the
    corresponding corner, or the skin twists."""
    if len(sections) < 2:
        raise ValueError("loft needs at least two sections")
    if len({len(section) for section in sections}) != 1:
        raise ValueError("every loft section needs the same point count")
    bm = bmesh.new()
    rings = [[bm.verts.new(tuple(point)) for point in section] for section in sections]
    _bridge_rings(bm, rings)
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    return _finish(bm, name)


def prism(
    name: str,
    profile: Sequence[Size2],
    start: float,
    end: float,
    axis: str = "X",
    bevel_width: float = 0.0,
) -> bpy.types.Object:
    """Extrude a closed 2D profile from start to end along a world axis - a chassis rail, fender
    side, wedge, bracket plate or any shape easier drawn as an outline than built from boxes. The
    profile lies in the plane across the axis, as (y, z) for X, (x, z) for Y and (x, y) for Z,
    and must not self-intersect; its winding does not matter."""
    bm = bmesh.new()
    a = [bm.verts.new(_axis_point(axis, start, u, v)) for u, v in profile]
    b = [bm.verts.new(_axis_point(axis, end, u, v)) for u, v in profile]
    _bridge_rings(bm, [a, b])
    bm.faces.new(a[::-1])
    bm.faces.new(b)
    if bevel_width > 0.0:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_width, segments=2, affect="EDGES", clamp_overlap=True)
    return _finish(bm, name)


def annular_band(
    name: str,
    centre: Size2,
    inner_radius: float,
    outer_radius: float,
    start: float,
    end: float,
    angle_start: float = 0.0,
    angle_end: float = 2.0 * math.pi,
    axis: str = "X",
    segments: int = 16,
) -> bpy.types.Object:
    """A thick curved band: the ring between inner_radius and outer_radius around centre (in the
    plane across axis, the same (u, v) convention as prism), swept from angle_start to angle_end
    in radians and extruded from start to end along axis - a mudguard, wheel arch, heat shield,
    curved hatch, collar, pipe sleeve or flange. A full 2*pi sweep is a closed tube."""
    full = abs(angle_end - angle_start) >= 2.0 * math.pi - 1e-9
    count = segments if full else segments + 1
    angles = [angle_start + (angle_end - angle_start) * k / segments for k in range(count)]
    cu, cv = centre
    bm = bmesh.new()

    def ring(radius: float, axial: float) -> list:
        return [bm.verts.new(_axis_point(axis, axial, cu + radius * math.cos(t), cv + radius * math.sin(t))) for t in angles]

    outer_a = ring(outer_radius, start)
    outer_b = ring(outer_radius, end)
    inner_a = ring(inner_radius, start)
    inner_b = ring(inner_radius, end)
    _bridge_rings(bm, [outer_a, outer_b, inner_b, inner_a, outer_a], closed=full)
    if not full:
        bm.faces.new((outer_a[0], outer_b[0], inner_b[0], inner_a[0]))
        bm.faces.new((outer_a[-1], inner_a[-1], inner_b[-1], outer_b[-1]))
    return _finish(bm, name)


def torus(
    name: str,
    centre: Sequence[float],
    major_radius: float,
    minor_radius: float,
    axis: str = "Z",
    major_segments: int = 12,
    minor_segments: int = 6,
) -> bpy.types.Object:
    """A ring around axis through centre - a hitch eye, lifting ring, door knocker, hose clamp,
    porthole rim or rope coil."""
    bm = bmesh.new()
    rings = []
    for i in range(major_segments):
        phi = 2.0 * math.pi * i / major_segments
        ring = []
        for j in range(minor_segments):
            theta = 2.0 * math.pi * j / minor_segments
            radial = major_radius + minor_radius * math.cos(theta)
            ring.append(bm.verts.new(_axis_point(axis, minor_radius * math.sin(theta), radial * math.cos(phi), radial * math.sin(phi))))
        rings.append(ring)
    _bridge_rings(bm, rings + [rings[0]])
    obj = _finish(bm, name)
    obj.location = centre
    return apply_transforms(obj)


def lugged_tyre(
    name: str,
    centre: Sequence[float],
    profile: Sequence[Size2],
    segments: int = 32,
    lug_height: float = 0.0,
    lug_skew: float = 0.0,
    lug_rows: tuple[int, int] | None = None,
    axis: str = "X",
) -> bpy.types.Object:
    """A tyre revolved around axis through centre from a closed (radius, axial offset) section
    loop, with lug_height tread lugs extruded from alternating section faces. lug_rows names the
    two section edges the lugs alternate between (default the two straddling the middle of the
    loop); lug_skew twists each lug by its axial distance, so a non-zero skew gives a chevron."""
    count = len(profile)
    rows = lug_rows if lug_rows is not None else (count // 2 - 1, count // 2)
    bm = bmesh.new()
    rings = []
    for k in range(segments):
        theta = 2.0 * math.pi * k / segments
        rings.append([bm.verts.new(_axis_point(axis, offset, radius * math.cos(theta), radius * math.sin(theta))) for radius, offset in profile])
    tread = {}
    for k in range(segments):
        k2 = (k + 1) % segments
        for i in range(count):
            i2 = (i + 1) % count
            tread[(k, i)] = bm.faces.new((rings[k][i], rings[k][i2], rings[k2][i2], rings[k2][i]))
    if lug_height > 0.0:
        lug_faces = [tread[(k, rows[k % 2])] for k in range(segments)]
        extruded = bmesh.ops.extrude_discrete_faces(bm, faces=lug_faces)
        for face in extruded["faces"]:
            for vert in face.verts:
                axial, u, v = _axis_coordinates(axis, vert.co)
                radius = math.hypot(u, v) + lug_height
                theta = math.atan2(v, u) + lug_skew * abs(axial)
                vert.co = Vector(_axis_point(axis, axial, radius * math.cos(theta), radius * math.sin(theta)))
    obj = _finish(bm, name)
    obj.location = centre
    return apply_transforms(obj)


def roof_slope(
    name: str,
    eave_start: Sequence[float],
    eave_end: Sequence[float],
    ridge_end: Sequence[float],
    ridge_start: Sequence[float],
    thickness: float,
) -> bpy.types.Object:
    """One roof plane through four corners (eave_start to eave_end along the eave, ridge_start
    above eave_start), solidified downward by thickness so its top face is exactly the plane
    given - for roofs gable_roof's single symmetric shell does not fit (hips, lean-tos, unequal
    pitches). Two slopes meeting at a ridge stay two solids: overlap them slightly at the ridge."""
    corners = [Vector(eave_start), Vector(eave_end), Vector(ridge_end), Vector(ridge_start)]
    normal = (corners[1] - corners[0]).cross(corners[3] - corners[0])
    if normal.z < 0.0:
        corners = [corners[0], corners[3], corners[2], corners[1]]
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(corner) for corner in corners])
    bm.normal_update()
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = apply_transforms(_link(mesh, name))
    return solidify(obj, thickness)


def triangular_gable(
    name: str,
    span: float,
    base_z: float,
    apex_z: float,
    thickness: float,
    centre_x: float = 0.0,
    y: float = 0.0,
) -> bpy.types.Object:
    """The triangular wall infill of a gable end: span wide across X centred on centre_x, from
    base_z (the eave line) up to apex_z, thickness deep along Y centred on y."""
    half_span = span * 0.5
    profile = [(centre_x - half_span, base_z), (centre_x + half_span, base_z), (centre_x, apex_z)]
    return prism(name, profile, y - thickness * 0.5, y + thickness * 0.5, axis="Y")


def smooth_by_angle(obj: bpy.types.Object, angle: float = math.radians(50.0)) -> bpy.types.Object:
    """Shade smooth with every edge sharper than angle (or on an open border) kept hard, so curved
    lathe/cylinder parts read round while box corners stay crisp."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for face in bm.faces:
        face.smooth = True
    for edge in bm.edges:
        edge.smooth = edge.is_manifold and edge.calc_face_angle(math.pi) < angle
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def wobble(obj: bpy.types.Object, amplitude: float, seed: int, wavelength: float = 1.0, keep_below_z: float | None = None) -> bpy.types.Object:
    """Displace every vertex by a smooth seeded field of at most amplitude per axis - hand-made
    mudbrick, plastered walls and thrown pots instead of CAD-straight solids. The field is a sum of
    sines over world position with period around wavelength, so coincident vertices of one mesh
    move together and the mesh stays closed. Vertices below keep_below_z do not move (a footing
    that must stay flush with the ground)."""
    rng = random.Random(seed)
    terms = []
    for _ in range(3):
        for _ in range(3):
            k = Vector((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))).normalized() * (2.0 * math.pi / (wavelength * rng.uniform(0.7, 1.4)))
            terms.append((k, rng.uniform(0.0, 2.0 * math.pi)))
    mesh = obj.data
    world = obj.matrix_world
    inverse = world.inverted()
    for vertex in mesh.vertices:
        position = world @ vertex.co
        if keep_below_z is not None and position.z < keep_below_z:
            continue
        offset = Vector((0.0, 0.0, 0.0))
        for axis in range(3):
            total = 0.0
            for k, phase in terms[axis * 3:axis * 3 + 3]:
                total += math.sin(k.dot(position) + phase)
            offset[axis] = amplitude * total / 3.0
        vertex.co = inverse @ (position + offset)
    mesh.update()
    return obj
