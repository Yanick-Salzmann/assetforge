from __future__ import annotations

import json
import struct

import pytest

from library import gltf


def _document(nodes, meshes, accessors, scene_nodes=(0,)):
    return {
        "scene": 0,
        "scenes": [{"nodes": list(scene_nodes)}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
    }


def _position_accessor(count, minimum, maximum):
    return {"count": count, "type": "VEC3", "componentType": 5126, "min": list(minimum), "max": list(maximum)}


def test_indexed_triangle_count_and_bbox():
    document = _document(
        nodes=[{"mesh": 0}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        accessors=[
            _position_accessor(4, (-1.0, 0.0, -1.0), (1.0, 2.0, 1.0)),
            {"count": 6, "type": "SCALAR", "componentType": 5123},
        ],
    )
    tris, bbox_min, bbox_max = gltf.stats_from_document(document)
    assert tris == 2
    assert bbox_min == (-1.0, 0.0, -1.0)
    assert bbox_max == (1.0, 2.0, 1.0)


def test_unindexed_triangle_count_falls_back_to_vertex_count():
    document = _document(
        nodes=[{"mesh": 0}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    )
    tris, _, _ = gltf.stats_from_document(document)
    assert tris == 1


def test_node_translation_shifts_the_bbox():
    document = _document(
        nodes=[{"mesh": 0, "translation": [5.0, 0.0, 0.0]}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    )
    _, bbox_min, bbox_max = gltf.stats_from_document(document)
    assert bbox_min == (5.0, 0.0, 0.0)
    assert bbox_max == (6.0, 1.0, 1.0)


def test_node_scale_stretches_the_bbox():
    document = _document(
        nodes=[{"mesh": 0, "scale": [2.0, 1.0, 1.0]}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[_position_accessor(3, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))],
    )
    _, bbox_min, bbox_max = gltf.stats_from_document(document)
    assert bbox_min == (-2.0, -1.0, -1.0)
    assert bbox_max == (2.0, 1.0, 1.0)


def test_child_nodes_inherit_the_parent_transform():
    document = _document(
        nodes=[
            {"children": [1], "translation": [1.0, 0.0, 0.0]},
            {"mesh": 0},
        ],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}}]}],
        accessors=[_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    )
    _, bbox_min, bbox_max = gltf.stats_from_document(document)
    assert bbox_min == (1.0, 0.0, 0.0)
    assert bbox_max == (2.0, 1.0, 1.0)


def test_multiple_meshes_accumulate_triangles_and_bbox():
    document = _document(
        nodes=[{"mesh": 0}, {"mesh": 1, "translation": [10.0, 0.0, 0.0]}],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}}]},
            {"primitives": [{"attributes": {"POSITION": 1}}]},
        ],
        accessors=[
            _position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            _position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ],
        scene_nodes=(0, 1),
    )
    tris, bbox_min, bbox_max = gltf.stats_from_document(document)
    assert tris == 2
    assert bbox_min == (0.0, 0.0, 0.0)
    assert bbox_max == (11.0, 1.0, 1.0)


def test_unsupported_primitive_mode_is_rejected():
    document = _document(
        nodes=[{"mesh": 0}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]}],
        accessors=[_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    )
    with pytest.raises(gltf.GltfError):
        gltf.stats_from_document(document)


def test_no_triangles_is_an_error():
    document = _document(nodes=[{}], meshes=[], accessors=[])
    with pytest.raises(gltf.GltfError):
        gltf.stats_from_document(document)


def _minimal_glb_bytes() -> bytes:
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    }
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((-len(payload)) % 4)
    header = struct.pack("<III", gltf.GLB_MAGIC, 2, 12 + 8 + len(payload))
    chunk = struct.pack("<II", len(payload), gltf.CHUNK_JSON)
    return header + chunk + payload


def test_read_document_parses_a_glb_file(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(_minimal_glb_bytes())
    tris, bbox_min, bbox_max = gltf.mesh_stats(path)
    assert tris == 1
    assert bbox_min == (0.0, 0.0, 0.0)
    assert bbox_max == (1.0, 1.0, 1.0)


def test_read_document_parses_a_plain_gltf_file(tmp_path):
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [_position_accessor(3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))],
    }
    path = tmp_path / "model.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")
    tris, _, _ = gltf.mesh_stats(path)
    assert tris == 1


def test_read_document_rejects_bad_magic(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(b"NOPE" + b"\x00" * 16)
    with pytest.raises(gltf.GltfError):
        gltf.read_document(path)


def test_read_document_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "model.obj"
    path.write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(gltf.GltfError):
        gltf.read_document(path)


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _glb_with_buffer(document: dict, binary: bytes) -> bytes:
    json_chunk = _pad4(json.dumps(document).encode("utf-8"), fill=b" ")
    bin_chunk = _pad4(binary)
    header = struct.pack("<III", gltf.GLB_MAGIC, 2, 12 + 8 + len(json_chunk) + 8 + len(bin_chunk))
    return (
        header
        + struct.pack("<II", len(json_chunk), gltf.CHUNK_JSON)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), gltf.CHUNK_BIN)
        + bin_chunk
    )


def test_read_accessor_decodes_positions_and_indices(tmp_path):
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    binary = _pad4(positions) + indices
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"count": 3, "type": "VEC3", "componentType": 5126, "bufferView": 0, "min": [0, 0, 0], "max": [1, 1, 0]},
            {"count": 3, "type": "SCALAR", "componentType": 5123, "bufferView": 1},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(_pad4(positions)), "byteLength": len(indices)},
        ],
        "buffers": [{"byteLength": len(_pad4(positions)) + len(indices)}],
    }
    path = tmp_path / "model.glb"
    path.write_bytes(_glb_with_buffer(document, binary))

    loaded, buffers = gltf.read_document_with_buffers(path)
    position_values = gltf.read_accessor(loaded, buffers, 0)
    index_values = gltf.read_accessor(loaded, buffers, 1)

    assert position_values.shape == (3, 3)
    assert position_values.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert index_values[:, 0].tolist() == [0, 1, 2]


def test_iter_mesh_nodes_yields_world_matrices():
    document = _document(
        nodes=[
            {"children": [1], "translation": [1.0, 0.0, 0.0]},
            {"mesh": 0},
            {"mesh": 1, "scale": [2.0, 2.0, 2.0]},
        ],
        meshes=[{"primitives": []}, {"primitives": []}],
        accessors=[],
        scene_nodes=(0, 2),
    )
    instances = list(gltf.iter_mesh_nodes(document))
    assert [(node, mesh) for node, mesh, _matrix in instances] == [(1, 0), (2, 1)]
    _node, _mesh, world = instances[0]
    assert world[3][:3] == (1.0, 0.0, 0.0)
