from __future__ import annotations

import json
import struct

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from assets import blender as blender_discovery
from assets import contact_sheet
from library import gltf
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    yield


def test_render_rejects_a_missing_asset():
    with pytest.raises(contact_sheet.ContactSheetError, match="does not exist"):
        contact_sheet.render("no-such-asset")


def test_render_rejects_a_tiny_resolution(tmp_path):
    (config.ASSET_OUT_DIR / "prop-crate").mkdir(parents=True)
    (config.ASSET_OUT_DIR / "prop-crate" / "prop-crate.glb").write_bytes(b"\x00")
    with pytest.raises(contact_sheet.ContactSheetError, match="resolution"):
        contact_sheet.render("prop-crate", resolution=(4, 4))


def test_render_rejects_a_non_positive_human_height(tmp_path):
    (config.ASSET_OUT_DIR / "prop-crate").mkdir(parents=True)
    (config.ASSET_OUT_DIR / "prop-crate" / "prop-crate.glb").write_bytes(b"\x00")
    with pytest.raises(contact_sheet.ContactSheetError, match="human_height_m"):
        contact_sheet.render("prop-crate", human_height_m=0.0)


def test_render_rejects_a_margin_below_one(tmp_path):
    (config.ASSET_OUT_DIR / "prop-crate").mkdir(parents=True)
    (config.ASSET_OUT_DIR / "prop-crate" / "prop-crate.glb").write_bytes(b"\x00")
    with pytest.raises(contact_sheet.ContactSheetError, match="margin"):
        contact_sheet.render("prop-crate", margin=0.5)


def test_tile_size_preserves_aspect_ratio():
    assert contact_sheet._tile_size((160, 90), 320) == (320, 180)
    assert contact_sheet._tile_size((90, 160), 320) == (180, 320)


def test_compose_sheet_lays_out_every_view_in_a_four_column_grid(tmp_path):
    colors = {
        "front": (255, 0, 0),
        "back": (0, 255, 0),
        "left": (0, 0, 255),
        "right": (255, 255, 0),
        "top": (255, 0, 255),
        "bottom": (0, 255, 255),
        "three_quarter": (128, 128, 128),
        "worm_eye": (255, 255, 255),
    }
    paths = {}
    for view, color in colors.items():
        path = tmp_path / f"{view}.png"
        Image.new("RGB", (32, 32), color).save(path)
        paths[view] = path

    sheet = contact_sheet._compose_sheet(paths, tile=32)

    columns = 4
    cell_w = 32 + contact_sheet.TILE_PADDING
    cell_h = 32 + contact_sheet.LABEL_HEIGHT + contact_sheet.TILE_PADDING
    for index, view in enumerate(contact_sheet.VIEWS):
        x = contact_sheet.TILE_PADDING + (index % columns) * cell_w + 16
        y = contact_sheet.TILE_PADDING + (index // columns) * cell_h + 16
        assert sheet.getpixel((x, y)) == colors[view]


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _write_tetrahedron_glb(path) -> None:
    apex = np.array([0.0, 0.0, 1.0])
    base = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    positions = np.vstack([base, apex])
    centroid = positions.mean(axis=0)
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]

    def outward_face(face):
        v0, v1, v2 = positions[list(face)]
        normal = np.cross(v1 - v0, v2 - v0)
        to_face = v0 - centroid
        return face if np.dot(normal, to_face) >= 0 else (face[0], face[2], face[1])

    indices = np.array([outward_face(face) for face in faces])
    normals = positions - centroid
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    position_bytes = struct.pack(f"<{positions.size}f", *positions.astype(np.float32).flatten())
    normal_bytes = struct.pack(f"<{normals.size}f", *normals.astype(np.float32).flatten())
    index_bytes = struct.pack(f"<{indices.size}H", *indices.astype(np.uint16).flatten())

    binary = _pad4(position_bytes) + _pad4(normal_bytes) + index_bytes
    position_offset = 0
    normal_offset = len(_pad4(position_bytes))
    index_offset = normal_offset + len(_pad4(normal_bytes))

    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [{"name": "placeholder"}],
        "accessors": [
            {
                "count": len(positions),
                "type": "VEC3",
                "componentType": 5126,
                "bufferView": 0,
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"count": len(normals), "type": "VEC3", "componentType": 5126, "bufferView": 1},
            {"count": indices.size, "type": "SCALAR", "componentType": 5123, "bufferView": 2},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes)},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes)},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }

    json_chunk = _pad4(json.dumps(document).encode("utf-8"), fill=b" ")
    bin_chunk = _pad4(binary)
    header = struct.pack("<III", gltf.GLB_MAGIC, 2, 12 + 8 + len(json_chunk) + 8 + len(bin_chunk))
    data = (
        header
        + struct.pack("<II", len(json_chunk), gltf.CHUNK_JSON)
        + json_chunk
        + struct.pack("<II", len(bin_chunk), gltf.CHUNK_BIN)
        + bin_chunk
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_render_produces_every_view_and_a_composed_sheet_end_to_end(tmp_path):
    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "test-prop" / "test-prop.glb")

    result = contact_sheet.render(
        "test-prop",
        resolution=(160, 90),
        samples=8,
        timeout_s=180.0,
    )

    assert result.sheet.is_file()
    assert set(result.views) == set(contact_sheet.VIEWS)
    for path in result.views.values():
        assert path.is_file()
    with Image.open(result.sheet) as sheet:
        assert sheet.width > 160
        assert sheet.height > 90
    assert result.bbox_max[2] > result.bbox_min[2]
