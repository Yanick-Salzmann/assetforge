from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
TRIANGLES = 4

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

Matrix = tuple[tuple[float, float, float, float], ...]

IDENTITY: Matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class GltfError(ValueError):
    """Raised when a glTF/GLB document cannot be parsed or has unsupported geometry."""


def read_document(path: Path) -> dict:
    if path.suffix == ".glb":
        return _read_glb(path.read_bytes())
    if path.suffix == ".gltf":
        return json.loads(path.read_text(encoding="utf-8"))
    raise GltfError(f"{path} is not a .glb or .gltf file")


def _read_glb(data: bytes) -> dict:
    document, _binary = _read_glb_chunks(data)
    return document


def _read_glb_chunks(data: bytes) -> tuple[dict, bytes | None]:
    if len(data) < 20:
        raise GltfError("file is too short to be a GLB")
    magic, _version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GltfError("not a GLB file (bad magic)")
    offset = 12
    document: dict | None = None
    binary: bytes | None = None
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk_data = data[offset + 8:offset + 8 + chunk_length]
        if chunk_type == CHUNK_JSON:
            document = json.loads(chunk_data)
        elif chunk_type == CHUNK_BIN:
            binary = chunk_data
        offset += 8 + chunk_length
    if document is None:
        raise GltfError("GLB does not contain a JSON chunk")
    return document, binary


def read_document_with_buffers(path: Path) -> tuple[dict, list[bytes]]:
    if path.suffix == ".glb":
        document, binary = _read_glb_chunks(path.read_bytes())
    elif path.suffix == ".gltf":
        document = json.loads(path.read_text(encoding="utf-8"))
        binary = None
    else:
        raise GltfError(f"{path} is not a .glb or .gltf file")
    return document, _load_buffers(path, document, binary)


def _load_buffers(path: Path, document: dict, embedded_binary: bytes | None) -> list[bytes]:
    buffers = []
    for index, buffer in enumerate(document.get("buffers", [])):
        uri = buffer.get("uri")
        if uri is None:
            if embedded_binary is None:
                raise GltfError(f"buffer {index} has no uri and the GLB has no binary chunk")
            buffers.append(embedded_binary[:buffer.get("byteLength", len(embedded_binary))])
        elif uri.startswith("data:"):
            _header, _sep, encoded = uri.partition(",")
            buffers.append(base64.b64decode(encoded))
        else:
            buffers.append((path.parent / uri).read_bytes())
    return buffers


def read_accessor(document: dict, buffers: list[bytes], accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    component_type = accessor["componentType"]
    dtype = COMPONENT_DTYPES.get(component_type)
    if dtype is None:
        raise GltfError(f"unsupported accessor componentType {component_type}")
    n_components = TYPE_COMPONENTS.get(accessor["type"])
    if n_components is None:
        raise GltfError(f"unsupported accessor type {accessor['type']}")
    count = accessor["count"]
    component_size = np.dtype(dtype).itemsize
    buffer_view_index = accessor.get("bufferView")
    if buffer_view_index is None:
        return np.zeros((count, n_components), dtype=np.float32)
    buffer_view = document["bufferViews"][buffer_view_index]
    buffer = buffers[buffer_view["buffer"]]
    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    element_size = n_components * component_size
    stride = buffer_view.get("byteStride") or element_size
    raw = np.frombuffer(buffer, dtype=np.uint8, count=stride * count, offset=byte_offset)
    if stride == element_size:
        values = raw.view(dtype).reshape(count, n_components)
    else:
        rows = raw.reshape(count, stride)[:, :element_size]
        values = np.ascontiguousarray(rows).view(dtype).reshape(count, n_components)
    if accessor.get("normalized") and dtype != np.float32:
        info = np.iinfo(dtype)
        divisor = info.max if info.min == 0 else info.max
        return (values.astype(np.float32) / divisor)
    return values.astype(np.float32) if dtype != np.float32 else values


def _rotation_columns(x: float, y: float, z: float, w: float) -> tuple[tuple[float, float, float], ...]:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
        (2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
        (2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)),
    )


def _node_matrix(node: dict) -> Matrix:
    if "matrix" in node:
        m = node["matrix"]
        return tuple(tuple(m[i * 4:i * 4 + 4]) for i in range(4))
    tx, ty, tz = node.get("translation", (0.0, 0.0, 0.0))
    rotation = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    sx, sy, sz = node.get("scale", (1.0, 1.0, 1.0))
    c0, c1, c2 = _rotation_columns(*rotation)
    return (
        (c0[0] * sx, c0[1] * sx, c0[2] * sx, 0.0),
        (c1[0] * sy, c1[1] * sy, c1[2] * sy, 0.0),
        (c2[0] * sz, c2[1] * sz, c2[2] * sz, 0.0),
        (tx, ty, tz, 1.0),
    )


def _multiply(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a[k][row] * b[j][k] for k in range(4)) for row in range(4))
        for j in range(4)
    )


def _transform_point(matrix: Matrix, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    result = [0.0, 0.0, 0.0]
    for k, component in enumerate((x, y, z, 1.0)):
        column = matrix[k]
        for row in range(3):
            result[row] += column[row] * component
    return (result[0], result[1], result[2])


def _corners(minimum: list[float], maximum: list[float]):
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                yield (x, y, z)


def iter_mesh_nodes(document: dict) -> Iterator[tuple[int, int, Matrix]]:
    """Yield (node_index, mesh_index, world_matrix) for every mesh instance in the scene."""
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not scenes:
        raise GltfError("document has no scenes")
    roots = scenes[document.get("scene", 0)]["nodes"]

    def visit(node_index: int, parent: Matrix) -> Iterator[tuple[int, int, Matrix]]:
        node = nodes[node_index]
        world = _multiply(parent, _node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            yield node_index, mesh_index, world
        for child in node.get("children", ()):
            yield from visit(child, world)

    for root in roots:
        yield from visit(root, IDENTITY)


def stats_from_document(document: dict) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not scenes:
        raise GltfError("document has no scenes")
    roots = scenes[document.get("scene", 0)]["nodes"]

    triangles = 0
    bbox_min = [math.inf, math.inf, math.inf]
    bbox_max = [-math.inf, -math.inf, -math.inf]

    def visit(node_index: int, parent: Matrix) -> None:
        nonlocal triangles
        node = nodes[node_index]
        world = _multiply(parent, _node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            for primitive in meshes[mesh_index]["primitives"]:
                mode = primitive.get("mode", TRIANGLES)
                if mode != TRIANGLES:
                    raise GltfError(f"unsupported primitive mode {mode}")
                position = accessors[primitive["attributes"]["POSITION"]]
                indices_index = primitive.get("indices")
                count = accessors[indices_index]["count"] if indices_index is not None else position["count"]
                triangles += count // 3
                for corner in _corners(position["min"], position["max"]):
                    x, y, z = _transform_point(world, corner)
                    bbox_min[0], bbox_min[1], bbox_min[2] = min(bbox_min[0], x), min(bbox_min[1], y), min(bbox_min[2], z)
                    bbox_max[0], bbox_max[1], bbox_max[2] = max(bbox_max[0], x), max(bbox_max[1], y), max(bbox_max[2], z)
        for child in node.get("children", ()):
            visit(child, world)

    for root in roots:
        visit(root, IDENTITY)

    if triangles == 0:
        raise GltfError("document has no triangles")
    return triangles, (bbox_min[0], bbox_min[1], bbox_min[2]), (bbox_max[0], bbox_max[1], bbox_max[2])


def mesh_stats(path: Path) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
    return stats_from_document(read_document(path))
