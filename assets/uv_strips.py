"""Cylindrical strip unwrap for revolved parts - lathe pots, frustum posts, logs, kilns (af-zch.1).

Axis projection slices a round part into 8-24 islands, one per dominant normal direction. This
finds each closed part whose side wraps a full turn around an axis and unwraps that side as
continuous strips instead: u is the angle around the axis times the band's radius, v the distance
along the band's profile curve. A band is a connected run of side faces whose radius stays within
BAND_MAX_RADIUS_SPREAD, so a straight post is one island and a bellied pot two or three. Caps and
anything this rejects stay on the axis-projection path.

Pure numpy, no bpy: assets/game_ready.py feeds it triangles and applies what it returns.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SIDE_MAX_AXIAL = 0.8
SIDE_MAX_TANGENTIAL = 0.25
SIDE_MAX_CORNER_ANGLE = math.radians(35.0)
COVERAGE_BINS = 12
MIN_SIDE_AREA_FRACTION = 0.4
BAND_MAX_SLOPE_DELTA = math.radians(60.0)
BAND_MAX_RADIUS_SPREAD = 1.5
PROFILE_SOLVE_ITERATIONS = 400
PROFILE_SOLVE_TOLERANCE = 1e-12
BAND_MIN_AREA_RATIO = 0.6
BAND_MAX_AREA_RATIO = 1.6
APEX_RADIUS_FRACTION = 1e-6
MIN_BAND_TRIANGLES = 8
OVERLAP_TOLERANCE = 1e-9
CONTINUITY_TOLERANCE = 1e-9


@dataclass
class Strip:
    """One unwrapped band: points[k] holds the three (u, v) corners in metres of triangles[k], a
    row index into the triangles passed to strip_islands."""

    triangles: list[int]
    points: list[tuple[tuple[float, float], ...]]
    radius: float


def _edge_axes(points) -> list[tuple[float, float]]:
    n = len(points)
    axes = []
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        axes.append((-(y2 - y1), x2 - x1))
    return axes


def shapes_overlap(a, b) -> bool:
    """Separating-axis test for two convex 2D polygons. Touching along an edge or at a vertex
    counts as separated, within OVERLAP_TOLERANCE of the projected extent so rounding on a shared
    edge does not read as overlap."""
    for ax, ay in _edge_axes(a) + _edge_axes(b):
        if ax == 0.0 and ay == 0.0:
            continue
        a_dots = [px * ax + py * ay for px, py in a]
        b_dots = [px * ax + py * ay for px, py in b]
        slack = OVERLAP_TOLERANCE * max(abs(d) for d in a_dots + b_dots)
        if max(a_dots) <= min(b_dots) + slack or max(b_dots) <= min(a_dots) + slack:
            return False
    return True


def any_overlap(shapes: list) -> bool:
    """True when any two of shapes overlap, by a sweep over their sorted u extents."""
    bounds = [(min(p[0] for p in s), max(p[0] for p in s), min(p[1] for p in s), max(p[1] for p in s)) for s in shapes]
    order = sorted(range(len(shapes)), key=lambda i: bounds[i][0])
    for position, i in enumerate(order):
        _, max_u, min_v, max_v = bounds[i]
        for j in order[position + 1:]:
            other = bounds[j]
            if other[0] >= max_u:
                break
            if other[3] <= min_v or max_v <= other[2]:
                continue
            if shapes_overlap(shapes[i], shapes[j]):
                return True
    return False


def _components(triangles: np.ndarray, rows: list[int]) -> list[list[int]]:
    parent = {row: row for row in rows}

    def find(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    owner: dict[tuple[int, int], int] = {}
    for row in rows:
        a, b, c = (int(i) for i in triangles[row])
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge), max(edge))
            if key in owner:
                parent[find(row)] = find(owner[key])
            else:
                owner[key] = row
    groups: dict[int, list[int]] = {}
    for row in rows:
        groups.setdefault(find(row), []).append(row)
    return list(groups.values())


def _wrap(angle: np.ndarray | float):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


@dataclass
class _Frame:
    axis: np.ndarray
    centre: np.ndarray
    first: np.ndarray
    second: np.ndarray

    def cylindrical(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = points - self.centre
        return offset @ self.axis, np.hypot(offset @ self.first, offset @ self.second), np.arctan2(offset @ self.second, offset @ self.first)


def _side_faces(frame: _Frame, corners: np.ndarray, centroids: np.ndarray, normals: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, rho, theta = frame.cylindrical(centroids)
    _, _, corner_theta = frame.cylindrical(corners.reshape(-1, 3))
    corner_theta = corner_theta.reshape(-1, 3)
    corner_rho = frame.cylindrical(corners.reshape(-1, 3))[1].reshape(-1, 3)
    radial = np.cos(theta)[:, None] * frame.first + np.sin(theta)[:, None] * frame.second
    tangent = np.cross(frame.axis, radial)
    axial = np.abs(normals @ frame.axis)
    tangential = np.abs(np.einsum("ij,ij->i", normals, tangent))
    offsets = np.abs(_wrap(corner_theta - theta[:, None]))
    offsets = np.where(corner_rho < scale * APEX_RADIUS_FRACTION, 0.0, offsets)
    side = (rho > scale * APEX_RADIUS_FRACTION) & (axial <= SIDE_MAX_AXIAL) & (tangential <= SIDE_MAX_TANGENTIAL) & (offsets.max(axis=1) <= SIDE_MAX_CORNER_ANGLE)
    return side, theta, radial


def _covers_full_turn(theta: np.ndarray) -> bool:
    bins = np.floor((theta + math.pi) / (2.0 * math.pi) * COVERAGE_BINS).astype(int) % COVERAGE_BINS
    return len(np.unique(bins)) == COVERAGE_BINS


def _candidate_axes(centroids: np.ndarray, areas: np.ndarray) -> list[np.ndarray]:
    weights = areas / areas.sum()
    mean = weights @ centroids
    offsets = centroids - mean
    covariance = (offsets * weights[:, None]).T @ offsets
    _, vectors = np.linalg.eigh(covariance)
    return [np.eye(3)[k] for k in range(3)] + [vectors[:, k] / np.linalg.norm(vectors[:, k]) for k in range(3)]


def _bands(side_rows: list[int], triangles: np.ndarray, slope: np.ndarray, rho: np.ndarray) -> list[list[int]]:
    neighbours: dict[tuple[int, int], list[int]] = {}
    for row in side_rows:
        a, b, c = (int(i) for i in triangles[row])
        for edge in ((a, b), (b, c), (c, a)):
            neighbours.setdefault((min(edge), max(edge)), []).append(row)
    adjacency: dict[int, list[int]] = {row: [] for row in side_rows}
    for rows in neighbours.values():
        for row in rows:
            adjacency[row].extend(other for other in rows if other != row)
    visited: set[int] = set()
    bands = []
    for seed in side_rows:
        if seed in visited:
            continue
        visited.add(seed)
        band = [seed]
        low = high = rho[seed]
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for other in adjacency[current]:
                if other in visited:
                    continue
                if abs(_wrap(slope[other] - slope[seed])) > BAND_MAX_SLOPE_DELTA:
                    continue
                if max(high, rho[other]) > BAND_MAX_RADIUS_SPREAD * min(low, rho[other]):
                    continue
                low = min(low, rho[other])
                high = max(high, rho[other])
                visited.add(other)
                band.append(other)
                queue.append(other)
        bands.append(band)
    return bands


def _profile_lengths(band_triangles: np.ndarray, h: np.ndarray, rho: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Distance along the profile for every vertex of band_triangles (rows of local vertex
    indices), solved in least squares so each edge's change matches its length along the profile
    tangent of the faces either side - the arc length of the band's profile curve, which a single
    projection only matches where the profile is straight."""
    count = len(h)
    edges: dict[tuple[int, int], list[np.ndarray]] = {}
    for triangle, tangent in zip(band_triangles, tangents):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(triangle[a]), int(triangle[b])
            edges.setdefault((min(i, j), max(i, j)), []).append(tangent)
    starts = np.array([i for i, _ in edges], dtype=np.int64)
    ends = np.array([j for _, j in edges], dtype=np.int64)
    mean = np.array([np.mean(t, axis=0) for t in edges.values()])
    mean /= np.maximum(np.linalg.norm(mean, axis=1), 1e-30)[:, None]
    target = (h[ends] - h[starts]) * mean[:, 0] + (rho[ends] - rho[starts]) * mean[:, 1]

    neighbours: dict[int, list[tuple[int, float]]] = {}
    for i, j, d in zip(starts.tolist(), ends.tolist(), target.tolist()):
        neighbours.setdefault(i, []).append((j, d))
        neighbours.setdefault(j, []).append((i, -d))
    values = np.zeros(count)
    seen = np.zeros(count, dtype=bool)
    seen[int(band_triangles[0][0])] = True
    queue = deque([int(band_triangles[0][0])])
    while queue:
        current = queue.popleft()
        for other, d in neighbours[current]:
            if not seen[other]:
                seen[other] = True
                values[other] = values[current] + d
                queue.append(other)

    def laplacian(x: np.ndarray) -> np.ndarray:
        diff = x[ends] - x[starts]
        out = np.zeros(count)
        np.add.at(out, ends, diff)
        np.add.at(out, starts, -diff)
        return out

    rhs = np.zeros(count)
    np.add.at(rhs, ends, target)
    np.add.at(rhs, starts, -target)
    residual = rhs - laplacian(values)
    direction = residual.copy()
    error = float(residual @ residual)
    for _ in range(PROFILE_SOLVE_ITERATIONS):
        if error < PROFILE_SOLVE_TOLERANCE:
            break
        step = laplacian(direction)
        alpha = error / float(direction @ step)
        values += alpha * direction
        residual -= alpha * step
        updated = float(residual @ residual)
        direction = residual + (updated / error) * direction
        error = updated
    return values


def _seam(theta: np.ndarray) -> float:
    ordered = np.sort(theta)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 2.0 * math.pi]))
    widest = int(np.argmax(gaps))
    return float(ordered[widest] + gaps[widest] * 0.5)


def _continuous(local: np.ndarray, u: np.ndarray, period: float) -> bool:
    """True when every vertex lands on one u for all its triangles, up to whole turns across the
    seam. A cone's apex sits on the axis, has no angle of its own and takes each face's angle
    instead, so its fan unwraps as a sawtooth of one-triangle islands touching at the apex row
    (af-5my) - such a band is left to axis projection."""
    first = np.full(int(local.max()) + 1, np.nan)
    flat_local = local.ravel()
    flat_u = u.ravel()
    order = np.arange(len(flat_local))[::-1]
    first[flat_local[order]] = flat_u[order]
    offset = flat_u - first[flat_local]
    residual = offset - period * np.round(offset / period)
    return bool(np.all(np.abs(residual) <= CONTINUITY_TOLERANCE * period))


def _unwrap_band(frame: _Frame, band: list[int], band_vertices: np.ndarray, corners: np.ndarray, normals: np.ndarray, areas: np.ndarray, theta: np.ndarray, radial: np.ndarray, rho: np.ndarray, scale: float) -> Strip | None:
    rows = np.asarray(band)
    weights = areas[rows]
    radius = float(weights @ rho[rows] / weights.sum())
    if radius <= 0.0:
        return None
    normal_h = normals[rows] @ frame.axis
    normal_rho = np.einsum("ij,ij->i", normals[rows], radial[rows])
    tangents = np.stack([normal_rho, -normal_h], axis=1)

    seam = _seam(theta[rows])
    face_theta = seam + (theta[rows] - seam) % (2.0 * math.pi)
    h, corner_rho, corner_theta = frame.cylindrical(corners[rows].reshape(-1, 3))
    h = h.reshape(-1, 3)
    corner_rho = corner_rho.reshape(-1, 3)
    corner_theta = corner_theta.reshape(-1, 3)
    unwrapped = seam + (corner_theta - seam) % (2.0 * math.pi)
    unwrapped = unwrapped - 2.0 * math.pi * np.round((unwrapped - face_theta[:, None]) / (2.0 * math.pi))
    unwrapped = np.where(corner_rho < scale * APEX_RADIUS_FRACTION, face_theta[:, None], unwrapped)
    u = unwrapped * radius
    vertices, local = np.unique(band_vertices, return_inverse=True)
    local = local.reshape(-1, 3)
    vertex_h = np.zeros(len(vertices))
    vertex_rho = np.zeros(len(vertices))
    vertex_h[local.ravel()] = h.ravel()
    vertex_rho[local.ravel()] = corner_rho.ravel()
    v = _profile_lengths(local, vertex_h, vertex_rho, tangents)[local]
    if not _continuous(local, u, 2.0 * math.pi * radius):
        return None

    signed = 0.5 * ((u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0]))
    if np.all(signed < 0.0):
        u = -u
        signed = -signed
    if not np.all(signed > 0.0):
        return None
    ratio = signed / areas[rows]
    if ratio.min() < BAND_MIN_AREA_RATIO or ratio.max() > BAND_MAX_AREA_RATIO:
        return None
    shapes = [tuple((float(u[k, j]), float(v[k, j])) for j in range(3)) for k in range(len(rows))]
    if any_overlap(shapes):
        return None
    return Strip(triangles=[int(r) for r in rows], points=shapes, radius=radius)


def _band_edges(triangles: np.ndarray, band: list[int]) -> set[tuple[int, int]]:
    edges = set()
    for row in band:
        a, b, c = (int(i) for i in triangles[row])
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add((min(i, j), max(i, j)))
    return edges


def _component_strips(rows: list[int], positions: np.ndarray, triangles: np.ndarray, normals: np.ndarray, areas: np.ndarray) -> list[Strip]:
    index = np.asarray(rows)
    component_areas = areas[index]
    total = float(component_areas.sum())
    if total <= 0.0:
        return []
    corners = positions[triangles[index]]
    centroids = corners.mean(axis=1)
    centre = (component_areas / total) @ centroids
    scale = float(np.linalg.norm(corners.reshape(-1, 3).max(axis=0) - corners.reshape(-1, 3).min(axis=0)))
    if scale <= 0.0:
        return []

    best = None
    for axis in _candidate_axes(centroids, component_areas):
        first, second = _basis(axis)
        frame = _Frame(axis=axis, centre=centre, first=first, second=second)
        side, theta, radial = _side_faces(frame, corners, centroids, normals[index], scale)
        if not side.any() or not _covers_full_turn(theta[side]):
            continue
        fraction = float(component_areas[side].sum()) / total
        if fraction >= MIN_SIDE_AREA_FRACTION and (best is None or fraction > best[0]):
            best = (fraction, frame, side, theta, radial)
    if best is None:
        return []

    _, frame, side, theta, radial = best
    local_normals = normals[index]
    _, rho, _ = frame.cylindrical(centroids)
    slope = np.arctan2(local_normals @ frame.axis, np.einsum("ij,ij->i", local_normals, radial))
    side_rows = [k for k in range(len(rows)) if side[k]]
    local_triangles = triangles[index]

    def unwrap(band: list[int]) -> Strip | None:
        return _unwrap_band(frame, band, local_triangles[band], corners, local_normals, component_areas, theta, radial, rho, scale)

    bands = _bands(side_rows, local_triangles, slope, rho)
    unwrapped = [unwrap(band) for band in bands]
    for small in sorted(range(len(bands)), key=lambda k: len(bands[k])):
        if not bands[small] or len(bands[small]) >= MIN_BAND_TRIANGLES:
            continue
        edges = _band_edges(local_triangles, bands[small])
        neighbours = [k for k in range(len(bands)) if k != small and bands[k] and edges & _band_edges(local_triangles, bands[k])]
        for k in sorted(neighbours, key=lambda k: -len(bands[k])):
            merged = unwrap(bands[k] + bands[small])
            if merged is not None:
                bands[k] = bands[k] + bands[small]
                unwrapped[k] = merged
                bands[small] = []
                unwrapped[small] = None
                break
    strips = []
    for strip in unwrapped:
        if strip is not None:
            strip.triangles = [rows[k] for k in strip.triangles]
            strips.append(strip)
    return strips


def strip_islands(positions: np.ndarray, triangles: np.ndarray, rows: list[int] | None = None) -> list[Strip]:
    """Strips for every revolved closed part among triangles[rows] (all rows by default):
    positions is float[N, 3], triangles int[M, 3] wound outward. Each part is an edge-connected
    component; it qualifies when, about one of the world axes or its own principal axes through
    its area centroid, the faces whose normals point away from the axis (no more than
    SIDE_MAX_TANGENTIAL sideways, each spanning at most SIDE_MAX_CORNER_ANGLE around it - so
    eight segments or more, never a box) cover a full turn and at least MIN_SIDE_AREA_FRACTION
    of the part's area. A band is dropped, and its faces left to axis projection, when any of its
    triangles flips, overlaps another or has its area scaled outside
    [BAND_MIN_AREA_RATIO, BAND_MAX_AREA_RATIO]."""
    positions = np.asarray(positions, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if rows is None:
        rows = list(range(len(triangles)))
    if not rows:
        return []
    corners = positions[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    areas = 0.5 * lengths
    normals = cross / np.maximum(lengths, 1e-30)[:, None]
    strips = []
    for component in _components(triangles, [r for r in rows if areas[r] > 0.0]):
        strips.extend(_component_strips(component, positions, triangles, normals, areas))
    return strips
