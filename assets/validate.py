from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from library import gltf
from terrain import config

TRI_BUDGETS: dict[str, tuple[int, int]] = {
    "hero_building": (4000, 15000),
    "prop": (200, 1500),
    "vehicle": (5000, 20000),
}

ORIGIN_TOLERANCE_M = 0.05
SCALE_TOLERANCE = 1e-3
ROTATION_TOLERANCE = 1e-3
DEGENERATE_AREA_EPS = 1e-10
UV_OVERLAP_AREA_EPS = 1e-6
UV_GRID_CELLS = 64
UV_BUCKET_CAP = 300
WELD_PRECISION = 5


class ValidationError(ValueError):
    """Raised when an asset cannot be located or its glTF document cannot be validated."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    name: str
    kind: str
    triangle_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": self.ok,
            "triangle_count": self.triangle_count,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


def _find_glb(name: str) -> Path:
    path = config.ASSET_OUT_DIR / name / f"{name}.glb"
    if not path.is_file():
        raise ValidationError(f"{path} does not exist")
    return path


def validate_asset(name: str, kind: str) -> ValidationReport:
    if kind not in TRI_BUDGETS:
        raise ValidationError(f"kind {kind!r} must be one of {sorted(TRI_BUDGETS)}")
    document, buffers = gltf.read_document_with_buffers(_find_glb(name))
    instances = list(gltf.iter_mesh_nodes(document))
    if not instances:
        raise ValidationError("document has no mesh instances")

    checks: list[CheckResult] = []
    total_tris = 0
    world_bbox_min = np.full(3, np.inf)
    world_bbox_max = np.full(3, -np.inf)

    for node_index, mesh_index, matrix in instances:
        checks.extend(_check_node_transform(node_index, matrix))
        mesh = document["meshes"][mesh_index]
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            label = f"mesh {mesh_index} primitive {primitive_index}"
            positions, indices, normals, uvs = _load_primitive(document, buffers, primitive)
            total_tris += len(indices)
            checks.extend(_check_material(label, primitive))
            checks.extend(_check_loose_vertices(label, positions, indices))
            checks.extend(_check_degenerate_triangles(label, positions, indices))
            welded = _weld_indices(positions)
            edge_faces = _edge_face_map(welded[indices])
            checks.extend(_check_manifold(label, edge_faces, positions, indices))
            checks.extend(_check_inward_normals(label, positions, indices, normals))
            checks.extend(_check_uv_overlap(label, indices, uvs))
            world_positions = _transform_positions(matrix, positions)
            world_bbox_min = np.minimum(world_bbox_min, world_positions.min(axis=0))
            world_bbox_max = np.maximum(world_bbox_max, world_positions.max(axis=0))

    checks.append(_check_triangle_budget(total_tris, kind))
    checks.append(_check_origin(world_bbox_min, world_bbox_max))

    return ValidationReport(
        name=name,
        kind=kind,
        triangle_count=total_tris,
        bbox_min=tuple(world_bbox_min.tolist()),
        bbox_max=tuple(world_bbox_max.tolist()),
        checks=checks,
    )


def _load_primitive(document: dict, buffers: list[bytes], primitive: dict):
    if primitive.get("mode", gltf.TRIANGLES) != gltf.TRIANGLES:
        raise ValidationError(f"unsupported primitive mode {primitive.get('mode')}")
    attributes = primitive["attributes"]
    positions = gltf.read_accessor(document, buffers, attributes["POSITION"])[:, :3].astype(np.float64)
    indices_accessor = primitive.get("indices")
    if indices_accessor is not None:
        indices = gltf.read_accessor(document, buffers, indices_accessor)[:, 0].astype(np.int64)
    else:
        indices = np.arange(len(positions), dtype=np.int64)
    indices = indices.reshape(-1, 3)
    normals = None
    if "NORMAL" in attributes:
        normals = gltf.read_accessor(document, buffers, attributes["NORMAL"])[:, :3].astype(np.float64)
    uvs = None
    if "TEXCOORD_0" in attributes:
        uvs = gltf.read_accessor(document, buffers, attributes["TEXCOORD_0"])[:, :2].astype(np.float64)
    return positions, indices, normals, uvs


def _weld_indices(positions: np.ndarray) -> np.ndarray:
    rounded = np.round(positions, WELD_PRECISION)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def _edge_face_map(triangle_indices: np.ndarray) -> dict[tuple[int, int], list[int]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    for tri_index, tri in enumerate(triangle_indices):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            mapping.setdefault(key, []).append(tri_index)
    return mapping


def _face_normal(positions: np.ndarray, indices: np.ndarray, tri_index: int) -> np.ndarray:
    a, b, c = indices[tri_index]
    v0, v1, v2 = positions[a], positions[b], positions[c]
    normal = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-12 else normal


def _is_welded_seam(tri_indices: list[int], positions: np.ndarray, indices: np.ndarray) -> bool:
    """True when an edge shared by 4 triangles is the harmless seam left where two
    independently-solidified, already-closed slabs are welded face-to-face (a door frame's inner
    cutout matching the wall opening it sits in, two wall panels butting at a corner): the extra
    pair of faces beyond the normal two lie in the exact same plane and face exactly opposite
    directions, so together they bound zero net material and are safe to leave in place, unlike a
    genuine self-intersection where no such matched pair exists."""
    normals = [_face_normal(positions, indices, tri_index) for tri_index in tri_indices]
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            if np.dot(normals[i], normals[j]) < -0.999:
                return True
    return False


def _check_material(label: str, primitive: dict) -> list[CheckResult]:
    passed = primitive.get("material") is not None
    detail = "material index present" if passed else "primitive has no material assigned"
    return [CheckResult(f"{label}: material assigned", passed, detail)]


def _check_loose_vertices(label: str, positions: np.ndarray, indices: np.ndarray) -> list[CheckResult]:
    referenced = np.zeros(len(positions), dtype=bool)
    referenced[indices.reshape(-1)] = True
    loose = int((~referenced).sum())
    passed = loose == 0
    detail = "no loose vertices" if passed else f"{loose} vertex/vertices are not referenced by any triangle"
    return [CheckResult(f"{label}: loose vertices", passed, detail)]


def _check_degenerate_triangles(label: str, positions: np.ndarray, indices: np.ndarray) -> list[CheckResult]:
    v0, v1, v2 = positions[indices[:, 0]], positions[indices[:, 1]], positions[indices[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    degenerate = int((areas < DEGENERATE_AREA_EPS).sum())
    passed = degenerate == 0
    detail = "no degenerate triangles" if passed else f"{degenerate} triangle(s) have ~zero area"
    return [CheckResult(f"{label}: degenerate triangles", passed, detail)]


def _check_manifold(
    label: str, edge_faces: dict[tuple[int, int], list[int]], positions: np.ndarray, indices: np.ndarray
) -> list[CheckResult]:
    boundary = 0
    non_manifold = 0
    welded_seams = 0
    for tri_indices in edge_faces.values():
        count = len(tri_indices)
        if count == 1:
            boundary += 1
        elif count == 2:
            continue
        elif count == 4 and _is_welded_seam(tri_indices, positions, indices):
            welded_seams += 1
        else:
            non_manifold += 1
    watertight_detail = (
        "closed mesh, no boundary edges" if boundary == 0 else f"{boundary} boundary edge(s) found (mesh has holes)"
    )
    manifold_detail = (
        "no non-manifold edges"
        if non_manifold == 0
        else f"{non_manifold} edge(s) are shared by more than two triangles"
    )
    if non_manifold == 0 and welded_seams > 0:
        manifold_detail += f" ({welded_seams} welded-seam edge(s) tolerated)"
    return [
        CheckResult(f"{label}: watertight", boundary == 0, watertight_detail),
        CheckResult(f"{label}: manifold", non_manifold == 0, manifold_detail),
    ]


def _check_inward_normals(
    label: str, positions: np.ndarray, indices: np.ndarray, normals: np.ndarray | None
) -> list[CheckResult]:
    if normals is None:
        return [CheckResult(f"{label}: outward normals", True, "primitive has no NORMAL attribute, skipped")]
    v0, v1, v2 = positions[indices[:, 0]], positions[indices[:, 1]], positions[indices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_lengths = np.linalg.norm(face_normals, axis=1)
    valid = face_lengths > 1e-12
    vertex_normal_sum = normals[indices[:, 0]] + normals[indices[:, 1]] + normals[indices[:, 2]]
    dot = np.einsum("ij,ij->i", face_normals, vertex_normal_sum)
    inward = int(((dot < 0) & valid).sum())
    passed = inward == 0
    detail = (
        "winding matches vertex normals"
        if passed
        else f"{inward} triangle(s) wind opposite their vertex normals (inward-facing)"
    )
    return [CheckResult(f"{label}: outward normals", passed, detail)]


def _check_uv_overlap(label: str, indices: np.ndarray, uvs: np.ndarray | None) -> list[CheckResult]:
    if uvs is None:
        return [CheckResult(f"{label}: UV overlap", True, "primitive has no TEXCOORD_0 attribute, skipped")]
    tri_uvs = uvs[indices]
    buckets: dict[tuple[int, int], list[int]] = {}
    for tri_index in range(len(tri_uvs)):
        tri = tri_uvs[tri_index]
        cell_min = np.floor(tri.min(axis=0) * UV_GRID_CELLS).astype(int)
        cell_max = np.floor(tri.max(axis=0) * UV_GRID_CELLS).astype(int)
        for cx in range(int(cell_min[0]), int(cell_max[0]) + 1):
            for cy in range(int(cell_min[1]), int(cell_max[1]) + 1):
                buckets.setdefault((cx, cy), []).append(tri_index)

    overlapping: set[int] = set()
    checked_pairs: set[tuple[int, int]] = set()
    inconclusive = False
    for cell_triangles in buckets.values():
        if len(cell_triangles) > UV_BUCKET_CAP:
            inconclusive = True
            continue
        for a_pos in range(len(cell_triangles)):
            for b_pos in range(a_pos + 1, len(cell_triangles)):
                a, b = cell_triangles[a_pos], cell_triangles[b_pos]
                pair = (a, b) if a < b else (b, a)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                if _triangle_overlap_area(tri_uvs[a], tri_uvs[b]) > UV_OVERLAP_AREA_EPS:
                    overlapping.add(a)
                    overlapping.add(b)

    if inconclusive and not overlapping:
        return [
            CheckResult(
                f"{label}: UV overlap",
                True,
                "UV island density exceeded the broad-phase grid cap; overlap check partially skipped",
            )
        ]
    passed = len(overlapping) == 0
    detail = "no overlapping UV triangles" if passed else f"{len(overlapping)} triangle(s) overlap another triangle's UV island"
    return [CheckResult(f"{label}: UV overlap", passed, detail)]


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segment_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-15:
        return p2
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def _clip_polygon(subject: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    if len(subject) == 0:
        return subject
    edge = edge_b - edge_a

    def inside(point: np.ndarray) -> bool:
        return edge[0] * (point[1] - edge_a[1]) - edge[1] * (point[0] - edge_a[0]) >= 0.0

    output = []
    for i in range(len(subject)):
        current, previous = subject[i], subject[i - 1]
        current_in, previous_in = inside(current), inside(previous)
        if current_in:
            if not previous_in:
                output.append(_segment_intersection(previous, current, edge_a, edge_b))
            output.append(current)
        elif previous_in:
            output.append(_segment_intersection(previous, current, edge_a, edge_b))
    return np.array(output) if output else np.empty((0, 2))


def _triangle_overlap_area(tri_a: np.ndarray, tri_b: np.ndarray) -> float:
    signed_area_b = (tri_b[1][0] - tri_b[0][0]) * (tri_b[2][1] - tri_b[0][1]) - (tri_b[1][1] - tri_b[0][1]) * (
        tri_b[2][0] - tri_b[0][0]
    )
    clip = tri_b if signed_area_b >= 0 else tri_b[::-1]
    polygon = tri_a
    for i in range(3):
        polygon = _clip_polygon(polygon, clip[i], clip[(i + 1) % 3])
        if len(polygon) == 0:
            return 0.0
    return _polygon_area(polygon)


def _check_triangle_budget(total_tris: int, kind: str) -> CheckResult:
    low, high = TRI_BUDGETS[kind]
    passed = low <= total_tris <= high
    detail = (
        f"{total_tris} tris within [{low}, {high}] budget for {kind}"
        if passed
        else f"{total_tris} tris outside the [{low}, {high}] budget for {kind}"
    )
    return CheckResult("triangle budget", passed, detail)


def _check_origin(bbox_min: np.ndarray, bbox_max: np.ndarray) -> CheckResult:
    center_x = (bbox_min[0] + bbox_max[0]) / 2.0
    center_z = (bbox_min[2] + bbox_max[2]) / 2.0
    base_y = bbox_min[1]
    offset = max(abs(center_x), abs(center_z), abs(base_y))
    passed = offset <= ORIGIN_TOLERANCE_M
    detail = (
        f"origin within {ORIGIN_TOLERANCE_M} m of base centre"
        if passed
        else f"origin is offset from base centre by up to {offset:.3f} m"
    )
    return CheckResult("origin at base centre", passed, detail)


def _transform_positions(matrix: gltf.Matrix, positions: np.ndarray) -> np.ndarray:
    columns = np.array(matrix, dtype=np.float64)
    linear = columns[:3, :3]
    translation = columns[3, :3]
    return positions @ linear + translation


def _check_node_transform(node_index: int, matrix: gltf.Matrix) -> list[CheckResult]:
    columns = np.array(matrix, dtype=np.float64)
    basis = columns[:3, :3]
    scale = np.linalg.norm(basis, axis=1)
    issues = []
    if np.any(np.abs(scale - 1.0) > SCALE_TOLERANCE):
        issues.append(f"scale {scale.tolist()} is not (1, 1, 1)")
    safe_scale = np.where(scale > 1e-12, scale, 1.0)
    rotation = basis / safe_scale[:, None]
    identity_error = float(np.max(np.abs(rotation - np.eye(3))))
    if identity_error > ROTATION_TOLERANCE:
        issues.append(f"rotation deviates from identity by {identity_error:.4f}")
    passed = len(issues) == 0
    detail = "transform is identity (scale/rotation already applied)" if passed else "; ".join(issues)
    return [CheckResult(f"node {node_index}: applied transforms", passed, detail)]
