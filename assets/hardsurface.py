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
from mathutils import Vector

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
