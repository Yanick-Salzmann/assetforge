"""Axis-projection UV island helpers shared by assets/game_ready.py (initial atlas packing) and
assets/lod.py (re-fitting a decimated LOD's faces back into that same packing without leaving
Decimate's raw vertex-interpolated UVs free to cross into a neighbouring island - af-3b4).

Runs inside Blender's own interpreter - only bmesh and mathutils.
"""

from __future__ import annotations

import bmesh
from mathutils import Vector

_AXIS_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def axis_key(normal: Vector) -> tuple[int, int]:
    """(dominant axis, sign of that component). Two faces only ever belong to the same island
    when both match: axis alone is not enough, since opposite-facing faces (e.g. the box's +X and
    -X sides) can otherwise share a dominant axis and, if a chain of same-axis faces connects them,
    get merged into one island whose flattened projection folds back on itself and self-overlaps -
    the actual cause of af-4ir.3's reported overlaps (confirmed: every overlapping triangle pair
    traced back to two faces inside the same axis-only island, not two different islands)."""
    ax, ay, az = abs(normal.x), abs(normal.y), abs(normal.z)
    if az >= ax and az >= ay:
        return 2, (1 if normal.z >= 0 else -1)
    if ay >= ax:
        return 1, (1 if normal.y >= 0 else -1)
    return 0, (1 if normal.x >= 0 else -1)


def project_point(co: Vector, axis: int) -> tuple[float, float]:
    ia, ib = _AXIS_PLANE[axis]
    return co[ia], co[ib]


def _edge_axes(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    n = len(points)
    axes = []
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        axes.append((-(y2 - y1), x2 - x1))
    return axes


def shapes_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Separating-axis test for two convex 2D polygons (every face here is a triangle by the
    time islands are built).

    Uses <=, not <, when checking whether an axis separates the two shapes: two triangles that
    merely touch along a shared edge or vertex (interval max(a) exactly equals interval min(b))
    must count as separated, not overlapping - the strict < this used before treated every such
    touch as an overlap, which is every edge-adjacent triangle pair in a triangulated mesh, so
    face_islands never merged so much as two triangles of the same flat face into one island
    (confirmed: a plain triangulated cube produced 12 single-triangle islands, and a 588-triangle
    subdivided box produced 588). That starved assets/lod.py's post-decimate UV re-fit of any
    island big enough to absorb a decimated region without different islands' target faces
    colliding in the same tiny cell (af-3b4). The exact-duplicate-coplanar-triangle case <=
    still must catch (af-4ir.3, see face_islands) has strictly positive overlap area, not just a
    touching boundary, so it remains correctly flagged under <=."""
    for ax, ay in _edge_axes(a) + _edge_axes(b):
        if ax == 0.0 and ay == 0.0:
            continue
        a_dots = [px * ax + py * ay for px, py in a]
        b_dots = [px * ax + py * ay for px, py in b]
        if max(a_dots) <= min(b_dots) or max(b_dots) <= min(a_dots):
            return False
    return True


def face_islands(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMFace]]:
    """Group faces into islands: faces connected by a shared edge whose dominant normal axis and
    sign match, so a flat multi-face region (a wall, a bevel strip) becomes one island while
    perpendicular faces at a corner, or faces on the opposite side of the mesh, split into
    separate ones.

    A candidate face is only merged if its projected shape does not overlap any shape already in
    the island: the EXACT boolean solver (panel_cut) can leave two exactly-coplanar, same-facing
    triangles at a cut seam that geometrically overlap each other - axis+sign matching alone
    cannot tell them apart from a legitimate flat region, so the merge itself is guarded instead.
    A face rejected here is picked up as its own island's seed later.
    """
    key_of = {face.index: axis_key(face.normal) for face in bm.faces}
    visited: set[int] = set()
    islands: list[list[bmesh.types.BMFace]] = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        visited.add(seed.index)
        axis, sign = key_of[seed.index]
        group = [seed]
        shapes = [[project_point(v.co, axis) for v in seed.verts]]
        stack = [seed]
        while stack:
            current = stack.pop()
            for edge in current.edges:
                for other in edge.link_faces:
                    if other.index in visited or key_of[other.index] != (axis, sign):
                        continue
                    other_shape = [project_point(v.co, axis) for v in other.verts]
                    if any(shapes_overlap(other_shape, shape) for shape in shapes):
                        continue
                    visited.add(other.index)
                    group.append(other)
                    shapes.append(other_shape)
                    stack.append(other)
        islands.append(group)
    return islands
