from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from assets import (
    animation_checks,
    animation_manifest,
    atlas_raster,
    fx_manifest,
    mechanism_checks,
    mechanism_manifest,
    prop_manifest,
    team_colour,
)
from library import gltf
from terrain import config

TRI_BUDGETS: dict[str, tuple[int, int]] = {
    "hero_building": (4000, 40000),
    "prop": (200, 1500),
    "vehicle": (5000, 20000),
    "hero_vehicle": (15000, 60000),
    "vegetation": (20, 600),
    "unit": (300, 4000),
    "creature": (500, 8000),
    "organic_prop": (100, 2000),
}
OPEN_SURFACE_KINDS = ("vegetation",)
LOD_CHAIN_KINDS = ("unit", "creature", "organic_prop", "vegetation")
IMPOSTOR_KINDS = ("vegetation",)
IMPOSTOR_MAPS = ("albedo", "normal")
REQUIRED_LODS = 3
LOD_STEP_MAX_RATIO = 0.75

ORIGIN_TOLERANCE_M = 0.05
FX_BBOX_TOLERANCE_M = 0.05
SCALE_TOLERANCE = 1e-3
ROTATION_TOLERANCE = 1e-3
DEGENERATE_AREA_EPS = 1e-10
UV_OVERLAP_AREA_EPS = 1e-6
UV_GRID_CELLS = 64
UV_BUCKET_CAP = 300
WELD_PRECISION = 5
ATLAS_GUTTER_MIN_PX = 8
ATLAS_GUTTER_REFERENCE_PX = 2048
ATLAS_BLEED_TOLERANCE = 0.2
ATLAS_BLEED_MAX_FRACTION = 0.02
UV_ISLAND_JOIN_TOLERANCE = 0.02
COLLAPSED_UV_AREA_EPS = 1e-12
TEXEL_DENSITY_MAX_RATIO = 2.0
TEXEL_DENSITY_PERCENTILES = (5.0, 95.0)


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
    lod_triangle_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict:
        failures = [check for check in self.checks if not check.passed]
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": bool(self.ok),
            "triangle_count": int(self.triangle_count),
            "lod_triangle_counts": {lod: int(count) for lod, count in self.lod_triangle_counts.items()},
            "bbox_min": [float(v) for v in self.bbox_min],
            "bbox_max": [float(v) for v in self.bbox_max],
            "passed_checks": len(self.checks) - len(failures),
            "failed_checks": len(failures),
            "failures": [{"name": c.name, "detail": str(c.detail)} for c in failures],
        }


def _find_glb(name: str) -> Path:
    path = config.ASSET_OUT_DIR / name / f"{name}.glb"
    if not path.is_file():
        raise ValidationError(f"{path} does not exist")
    return path


def validate_asset(name: str, kind: str) -> ValidationReport:
    if kind not in TRI_BUDGETS:
        raise ValidationError(f"kind {kind!r} must be one of {sorted(TRI_BUDGETS)}")
    glb_path = _find_glb(name)
    document, buffers = gltf.read_document_with_buffers(glb_path)
    instances = list(gltf.iter_mesh_nodes(document))
    if not instances:
        raise ValidationError("document has no mesh instances")

    checks: list[CheckResult] = []
    lod_tris: dict[int, int] = {}
    missing_uvs = 0
    world_bbox_min = np.full(3, np.inf)
    world_bbox_max = np.full(3, -np.inf)
    rigid_parts: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for node_index, mesh_index, matrix in instances:
        node = document["nodes"][node_index]
        if "skin" in node:
            matrix = _as_gltf_matrix(animation_checks.skin_bind_matrix(document, buffers, node["skin"]))
        checks.extend(_check_node_transform(node_index, matrix))
        node_name = node.get("name", "")
        lod = _lod_index(node_name)
        mesh = document["meshes"][mesh_index]
        part_positions = []
        part_indices = []
        part_vertices = 0
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            label = f"{node_name or f'node {node_index}'} mesh {mesh_index} primitive {primitive_index}"
            positions, indices, normals, uvs = _load_primitive(document, buffers, primitive)
            lod_tris[lod] = lod_tris.get(lod, 0) + len(indices)
            part_positions.append(positions)
            part_indices.append(indices + part_vertices)
            part_vertices += len(positions)
            if uvs is None:
                missing_uvs += 1
            checks.extend(_check_material(label, primitive))
            checks.extend(_check_loose_vertices(label, positions, indices))
            checks.extend(_check_degenerate_triangles(label, positions, indices))
            checks.extend(_check_inward_normals(label, positions, indices, normals))
            if kind not in OPEN_SURFACE_KINDS:
                welded = _weld_indices(positions)
                edge_faces = _edge_face_map(welded[indices])
                checks.extend(_check_manifold(label, edge_faces, positions, indices))
                checks.extend(_check_uv_overlap(label, indices, uvs))
                base_colour = _base_colour_image(document, buffers, primitive, glb_path.parent)
                checks.extend(_check_atlas_gutters(label, positions, indices, uvs, base_colour))
                checks.extend(_check_texel_density(label, positions, indices, uvs))
            world_positions = _transform_positions(matrix, positions)
            world_bbox_min = np.minimum(world_bbox_min, world_positions.min(axis=0))
            world_bbox_max = np.maximum(world_bbox_max, world_positions.max(axis=0))
        if lod == 0 and "skin" not in node and kind not in OPEN_SURFACE_KINDS:
            rigid_parts[node_index] = (np.vstack(part_positions), np.vstack(part_indices))

    lod0_tris = lod_tris.get(0, 0)
    checks.append(_check_triangle_budget(lod0_tris, kind))
    checks.append(_check_origin(world_bbox_min, world_bbox_max))
    if kind in LOD_CHAIN_KINDS:
        checks.append(_check_lod_chain(lod_tris))
    if kind in team_colour.KINDS:
        checks.append(_check_team_mask(name, kind, missing_uvs))
    manifest_check, animations = _check_animation_manifest(name)
    checks.append(manifest_check)
    if manifest_check.passed:
        checks.extend(_check_animations(document, buffers, animations, kind, rigid_parts))
        checks.extend(_check_mechanisms(name, document, buffers, animations))
    checks.append(_check_fx(name, document, world_bbox_min, world_bbox_max))
    checks.append(_check_props(name, world_bbox_min, world_bbox_max))
    checks.append(_check_impostor(name, kind))

    return ValidationReport(
        name=name,
        kind=kind,
        triangle_count=lod0_tris,
        bbox_min=tuple(world_bbox_min.tolist()),
        bbox_max=tuple(world_bbox_max.tolist()),
        checks=checks,
        lod_triangle_counts={f"LOD{lod}": lod_tris[lod] for lod in sorted(lod_tris)},
    )


def _lod_index(node_name: str) -> int:
    return animation_manifest.lod_index(node_name)


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


def _base_colour_image(document: dict, buffers: list[bytes], primitive: dict, glb_dir: Path) -> np.ndarray | None:
    """The primitive's TEXCOORD_0 base colour texture as a (rows, columns, 3) float array in 0-1,
    row 0 at the top - the orientation glTF UVs address directly. None when there is none."""
    material_index = primitive.get("material")
    if material_index is None:
        return None
    info = document["materials"][material_index].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if info is None or info.get("texCoord", 0) != 0:
        return None
    image = document["images"][document["textures"][info["index"]]["source"]]
    if "bufferView" in image:
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        data = buffers[view["buffer"]][start:start + view["byteLength"]]
    elif "uri" in image and not image["uri"].startswith("data:"):
        data = (glb_dir / image["uri"]).read_bytes()
    else:
        return None
    with Image.open(io.BytesIO(data)) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def _uv_triangle_areas(tri_uvs: np.ndarray) -> np.ndarray:
    edge_a = tri_uvs[:, 1] - tri_uvs[:, 0]
    edge_b = tri_uvs[:, 2] - tri_uvs[:, 0]
    return 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])


def _uv_islands(positions: np.ndarray, indices: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Island id per triangle, -1 for triangles collapsed to zero UV area (hidden faces). Two
    triangles share an island when they share a welded 3D vertex whose UVs match within
    UV_ISLAND_JOIN_TOLERANCE of the larger triangle's UV extent - glTF splits vertices at sharp
    normals and the packer shrinks every triangle toward its centroid, so neither shared indices
    nor exact UV equality identify an island. A shared vertex rather than a shared edge, because a
    decimated LOD leaves triangles that meet their neighbours only at a vertex (a T-junction) yet
    sit in the same packed island; a UV seam still splits, since its two sides differ in UV."""
    tri_uvs = uvs[indices]
    live = _uv_triangle_areas(tri_uvs) > COLLAPSED_UV_AREA_EPS
    extent = (tri_uvs.max(axis=1) - tri_uvs.min(axis=1)).max(axis=1)
    welded = _weld_indices(positions)[indices]
    parent = list(range(len(indices)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    corners: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tri in np.flatnonzero(live):
        for corner in range(3):
            corners.setdefault(int(welded[tri, corner]), []).append((int(tri), tri_uvs[tri, corner]))
    for sharing in corners.values():
        for i in range(len(sharing)):
            for j in range(i + 1, len(sharing)):
                tri_i, uv_i = sharing[i]
                tri_j, uv_j = sharing[j]
                if find(tri_i) == find(tri_j):
                    continue
                tolerance = UV_ISLAND_JOIN_TOLERANCE * max(extent[tri_i], extent[tri_j]) + 1e-7
                if np.abs(uv_i - uv_j).max() <= tolerance:
                    parent[find(tri_i)] = find(tri_j)

    roots = np.array([find(t) if live[t] else -1 for t in range(len(indices))], dtype=np.int64)
    ids = np.full(len(indices), -1, dtype=np.int32)
    if live.any():
        _, relabelled = np.unique(roots[live], return_inverse=True)
        ids[live] = relabelled
    return ids


def _check_atlas_gutters(
    label: str, positions: np.ndarray, indices: np.ndarray, uvs: np.ndarray | None, image: np.ndarray | None
) -> list[CheckResult]:
    """Every UV island must keep a gutter of its own colour: no two islands' covered texels closer
    than ATLAS_GUTTER_MIN_PX (scaled to the texture size), and the texels within half that of an
    island must match the island's nearest covered texel. Catches the af-alp bleed - islands ~1 px
    apart whose bake margins and mips carried each other's colours."""
    name = f"{label}: atlas gutters"
    if uvs is None or image is None:
        return [CheckResult(name, True, "no TEXCOORD_0 base colour texture, skipped")]
    height, width = image.shape[:2]
    islands = _uv_islands(positions, indices, uvs)
    live = islands >= 0
    if not live.any():
        return [CheckResult(name, True, "no textured faces, skipped")]
    triangles = uvs[indices][live] * np.array([width, height], dtype=np.float64)
    labels = atlas_raster.rasterize(triangles, islands[live], height, width)
    reach = max(1, round(ATLAS_GUTTER_MIN_PX * width / ATLAS_GUTTER_REFERENCE_PX / 2))
    grown, source, touching = atlas_raster.grow_sources(labels, reach)
    ring = (grown >= 0) & (labels < 0) & (source >= 0)
    bleeding = 0.0
    if ring.any():
        flat = image.reshape(-1, image.shape[-1])
        differences = np.abs(image[ring] - flat[source[ring]]).max(axis=1)
        bleeding = float((differences > ATLAS_BLEED_TOLERANCE).mean())
    problems = []
    if touching:
        problems.append(f"{len(touching)} island pair(s) closer than {2 * reach} px")
    if bleeding > ATLAS_BLEED_MAX_FRACTION:
        problems.append(f"{bleeding:.1%} of gutter texels within {reach} px of an island carry another colour")
    detail = "; ".join(problems) if problems else f"{int(islands.max()) + 1} islands at least {2 * reach} px apart, gutters clean"
    return [CheckResult(name, not problems, detail)]


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, cumulative[-1] * percentile / 100.0))
    return float(values[order][min(position, len(values) - 1)])


def _check_texel_density(label: str, positions: np.ndarray, indices: np.ndarray, uvs: np.ndarray | None) -> list[CheckResult]:
    """Texels per metre must be near uniform across the surface: the area-weighted 95th over 5th
    percentile of per-triangle density at most TEXEL_DENSITY_MAX_RATIO. Collapsed (hidden) faces are
    left out - they deliberately own no texels."""
    name = f"{label}: texel density"
    if uvs is None:
        return [CheckResult(name, True, "no TEXCOORD_0, skipped")]
    corners = positions[indices]
    world = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    uv_area = _uv_triangle_areas(uvs[indices])
    live = (uv_area > COLLAPSED_UV_AREA_EPS) & (world > DEGENERATE_AREA_EPS)
    if not live.any():
        return [CheckResult(name, True, "no textured faces, skipped")]
    density = np.sqrt(uv_area[live] / world[live])
    low = _weighted_percentile(density, world[live], TEXEL_DENSITY_PERCENTILES[0])
    high = _weighted_percentile(density, world[live], TEXEL_DENSITY_PERCENTILES[1])
    ratio = high / max(low, 1e-12)
    passed = ratio <= TEXEL_DENSITY_MAX_RATIO
    return [CheckResult(name, passed, f"p95/p5 texel density ratio {ratio:.2f} (max {TEXEL_DENSITY_MAX_RATIO})")]


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
    if abs(signed_area_b) <= 2.0 * COLLAPSED_UV_AREA_EPS or _polygon_area(tri_a) <= COLLAPSED_UV_AREA_EPS:
        return 0.0
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


def _check_lod_chain(lod_tris: dict[int, int]) -> CheckResult:
    missing = [f"LOD{lod}" for lod in range(REQUIRED_LODS) if lod_tris.get(lod, 0) == 0]
    if missing:
        required = f"LOD0-LOD{REQUIRED_LODS - 1} are required"
        return CheckResult("LOD chain", False, f"missing {', '.join(missing)}; {required}")
    weak = [
        f"LOD{lod} {lod_tris[lod]} > {LOD_STEP_MAX_RATIO} x LOD{lod - 1} {lod_tris[lod - 1]}"
        for lod in range(1, REQUIRED_LODS)
        if lod_tris[lod] > LOD_STEP_MAX_RATIO * lod_tris[lod - 1]
    ]
    if weak:
        return CheckResult("LOD chain", False, "; ".join(weak))
    counts = ", ".join(f"LOD{lod} {lod_tris[lod]}" for lod in range(REQUIRED_LODS))
    detail = f"{counts}, each step at most {LOD_STEP_MAX_RATIO} x the previous"
    return CheckResult("LOD chain", True, detail)


def _check_team_mask(name: str, kind: str, missing_uvs: int) -> CheckResult:
    label = "team colour mask"
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    entry = None
    if manifest_path.is_file():
        entry = json.loads(manifest_path.read_text(encoding="utf-8")).get(team_colour.MANIFEST_KEY)
    if entry is None:
        if kind in team_colour.REQUIRED_KINDS:
            detail = f"asset.json has no {team_colour.MANIFEST_KEY} entry; {kind} needs a team mask"
            return CheckResult(label, False, detail)
        return CheckResult(label, True, "no team mask, renders untinted")
    if missing_uvs:
        detail = f"{missing_uvs} primitive(s) have no TEXCOORD_0 to sample the mask with"
        return CheckResult(label, False, detail)
    mask_path = config.ASSET_OUT_DIR / name / entry["mask"]
    if not mask_path.is_file():
        return CheckResult(label, False, f"{mask_path} does not exist")
    with Image.open(mask_path) as image:
        width, height = image.size
        mask = np.asarray(image.convert("RGB"))[:, :, 0]
    if not (_is_power_of_two(width) and _is_power_of_two(height)):
        return CheckResult(label, False, f"mask is {width}x{height}, not power-of-two")
    coverage = float((mask >= team_colour.COVERAGE_THRESHOLD).mean())
    if coverage < team_colour.MIN_COVERAGE:
        detail = f"mask covers {coverage:.4%} of the atlas; nothing would take the team colour"
        return CheckResult(label, False, detail)
    return CheckResult(label, True, f"{width}x{height} mask covers {coverage:.1%} of the atlas")


def _check_animation_manifest(name: str) -> tuple[CheckResult, list[dict]]:
    label = "animations manifest"
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    if not manifest_path.is_file():
        return CheckResult(label, True, "no asset.json, static"), []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if animation_manifest.MANIFEST_KEY not in manifest:
        return CheckResult(label, True, "static, no clips"), []
    animations = manifest[animation_manifest.MANIFEST_KEY]
    errors = animation_manifest.manifest_errors(animations)
    if errors:
        return CheckResult(label, False, "; ".join(errors)), []
    clips = ", ".join(entry["name"] for entry in animations)
    return CheckResult(label, True, f"clips {clips or '(none)'}"), animations


def _check_animations(
    document: dict,
    buffers: list[bytes],
    animations: list[dict],
    kind: str,
    rigid_parts: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[CheckResult]:
    if not animations and not document.get("animations"):
        return []
    clips = animation_checks.read_clips(document, buffers)
    findings = animation_checks.clip_checks(document, animations, clips)
    findings += animation_checks.rest_checks(document, animations)
    findings += animation_checks.seam_checks(document, animations, clips)
    findings += animation_checks.budget_checks(document, animations, clips, kind)
    findings += animation_checks.sweep_checks(document, animations, clips, rigid_parts)
    return [CheckResult(*finding) for finding in findings]


def _check_mechanisms(name: str, document: dict, buffers: list[bytes], animations: list[dict]) -> list[CheckResult]:
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mechanism_manifest.MANIFEST_KEY not in manifest:
        return []
    mechanisms = manifest[mechanism_manifest.MANIFEST_KEY]
    errors = mechanism_manifest.manifest_errors(mechanisms, animations)
    if errors:
        return [CheckResult("mechanisms", False, "; ".join(errors))]
    clips = animation_checks.read_clips(document, buffers)
    findings = mechanism_checks.mechanism_checks(document, animations, mechanisms, clips)
    return [CheckResult(*finding) for finding in findings]


def _check_fx(name: str, document: dict, bbox_min: np.ndarray, bbox_max: np.ndarray) -> CheckResult:
    label = "fx sockets"
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    fx = []
    if manifest_path.is_file():
        fx = json.loads(manifest_path.read_text(encoding="utf-8")).get(fx_manifest.MANIFEST_KEY, [])
    errors = fx_manifest.manifest_errors(fx)
    if errors:
        return CheckResult(label, False, "; ".join(errors))
    nodes = document.get("nodes", [])
    worlds = animation_checks.world_matrices(document)
    sockets = {nodes[index].get("name", ""): index for index in worlds if fx_manifest.is_fx_node(nodes[index].get("name", ""))}
    low = bbox_min - FX_BBOX_TOLERANCE_M
    high = bbox_max + FX_BBOX_TOLERANCE_M
    for entry in fx:
        index = sockets.get(entry["node"])
        if index is None:
            errors.append(f"{entry['node']}: no such node in the glb")
            continue
        if "mesh" in nodes[index]:
            errors.append(f"{entry['node']}: socket carries a mesh; it must be an empty")
        position = worlds[index][:3, 3]
        if np.any(position < low) or np.any(position > high):
            errors.append(f"{entry['node']}: socket at {np.round(position, 3).tolist()} is outside the asset bbox")
    orphans = sorted(set(sockets) - {entry["node"] for entry in fx})
    if orphans:
        errors.append(f"socket node(s) {', '.join(orphans)} have no fx entry")
    if errors:
        return CheckResult(label, False, "; ".join(errors))
    if not fx:
        return CheckResult(label, True, "no fx")
    return CheckResult(label, True, f"{len(fx)} socket(s) inside the bbox: {', '.join(entry['node'] for entry in fx)}")


def _known_kit_props() -> frozenset[str]:
    from library import kits

    try:
        return frozenset(kits.build().names())
    except (kits.KitError, OSError, ValueError):
        return frozenset()


def _check_props(name: str, bbox_min: np.ndarray, bbox_max: np.ndarray) -> CheckResult:
    label = "kit props"
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    props = []
    if manifest_path.is_file():
        props = json.loads(manifest_path.read_text(encoding="utf-8")).get(prop_manifest.MANIFEST_KEY, [])
    if not props:
        return CheckResult(label, True, "no props")
    errors = prop_manifest.manifest_errors(props, _known_kit_props())
    if not errors:
        errors = prop_manifest.outside_errors(props, bbox_min.tolist(), bbox_max.tolist())
    if errors:
        return CheckResult(label, False, "; ".join(errors))
    meshes = sorted({entry["mesh"] for entry in props})
    return CheckResult(label, True, f"{len(props)} placement(s) of {len(meshes)} kit prop(s): {', '.join(meshes)}")


def _check_impostor(name: str, kind: str) -> CheckResult:
    """An asset.json impostor (assets/impostor.py) is a far LOD for scatter vegetation only:
    its albedo and normal atlases exist, are square powers of two and hold frames x frames
    frames of frame_px each."""
    label = "impostor"
    manifest_path = config.ASSET_OUT_DIR / name / "asset.json"
    entry = None
    if manifest_path.is_file():
        entry = json.loads(manifest_path.read_text(encoding="utf-8")).get("impostor")
    if entry is None:
        return CheckResult(label, True, "no impostor")
    if kind not in IMPOSTOR_KINDS:
        return CheckResult(label, False, f"kind {kind!r} may not ship an impostor; only {IMPOSTOR_KINDS} do")
    try:
        expected = int(entry["frames"]) * int(entry["frame_px"])
        if float(entry["radius"]) <= 0.0 or len(entry["centre"]) != 3:
            return CheckResult(label, False, "impostor needs a positive radius and a 3-component centre")
        maps = {key: config.ASSET_OUT_DIR / name / entry[key] for key in IMPOSTOR_MAPS}
    except (KeyError, TypeError, ValueError) as error:
        return CheckResult(label, False, f"malformed impostor entry: {error}")
    errors = []
    for key, path in maps.items():
        if not path.is_file():
            errors.append(f"{key} {path.name} missing")
            continue
        with Image.open(path) as image:
            size = image.size
        if size != (expected, expected) or not _is_power_of_two(expected):
            errors.append(f"{key} is {size[0]}x{size[1]}, expected a power-of-two {expected}x{expected}")
    if errors:
        return CheckResult(label, False, "; ".join(errors))
    return CheckResult(label, True, f"{entry['frames']}x{entry['frames']} frames of {entry['frame_px']} px")


def _as_gltf_matrix(matrix: np.ndarray) -> gltf.Matrix:
    return tuple(tuple(float(value) for value in column) for column in matrix.T)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


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
