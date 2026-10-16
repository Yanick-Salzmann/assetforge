from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from assets import fx_manifest, team_colour, validate
from library import gltf
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    yield


def test_check_loose_vertices_flags_unreferenced_vertex():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=np.float64)
    indices = np.array([[0, 1, 2]])
    result = validate._check_loose_vertices("m", positions, indices)[0]
    assert not result.passed


def test_check_loose_vertices_passes_when_all_referenced():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2]])
    result = validate._check_loose_vertices("m", positions, indices)[0]
    assert result.passed


def test_check_degenerate_triangles_flags_zero_area():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2]])
    result = validate._check_degenerate_triangles("m", positions, indices)[0]
    assert not result.passed


def test_check_degenerate_triangles_passes_for_real_area():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2]])
    result = validate._check_degenerate_triangles("m", positions, indices)[0]
    assert result.passed


def _tetrahedron():
    apex = np.array([0.0, 1.0, 0.0])
    base = np.array([[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.0, 0.0, 0.5]])
    positions = np.vstack([base, apex])
    centroid = positions.mean(axis=0)
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]

    def outward_face(face):
        v0, v1, v2 = positions[list(face)]
        normal = np.cross(v1 - v0, v2 - v0)
        to_face = v0 - centroid
        return face if np.dot(normal, to_face) >= 0 else (face[0], face[2], face[1])

    indices = np.array([outward_face(face) for face in faces])
    return positions, indices, centroid


def _manifold_checks(positions, indices):
    welded = validate._weld_indices(positions)
    edge_faces = validate._edge_face_map(welded[indices])
    return validate._check_manifold("m", edge_faces, positions, indices)


def test_tetrahedron_is_watertight_and_manifold():
    positions, indices, _centroid = _tetrahedron()
    watertight, manifold = _manifold_checks(positions, indices)
    assert watertight.passed
    assert manifold.passed


def test_open_triangle_is_not_watertight():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2]])
    watertight, _manifold = _manifold_checks(positions, indices)
    assert not watertight.passed


def test_non_manifold_edge_shared_by_three_triangles_is_flagged():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [1, -1, 0]], dtype=np.float64)
    indices = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    _watertight, manifold = _manifold_checks(positions, indices)
    assert not manifold.passed


def test_welded_seam_edge_with_a_coplanar_antiparallel_pair_is_tolerated():
    # Edge (0,1) shared by 4 triangles fanned around it at 0/90/180/240 degrees. The 0/180 pair is
    # coplanar and antiparallel - exactly the redundant rim-cap pair a welded seam (e.g. a door
    # frame's inner cutout matching the wall opening it sits in) leaves behind.
    positions = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [-0.5, -0.866, 0],
        ],
        dtype=np.float64,
    )
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]])
    _watertight, manifold = _manifold_checks(positions, indices)
    assert manifold.passed
    assert "tolerated" in manifold.detail


def test_valence_four_edge_without_a_matching_pair_is_still_flagged():
    # Same shared edge, but the 4 triangles are fanned at 0/70/150/220 degrees - no two are
    # anywhere near antiparallel, so this is genuinely broken geometry, not a welded seam.
    positions = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0.342, 0.940, 0],
            [-0.866, 0.5, 0],
            [-0.766, -0.643, 0],
        ],
        dtype=np.float64,
    )
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]])
    _watertight, manifold = _manifold_checks(positions, indices)
    assert not manifold.passed


def _angle_point(degrees: float, radius: float = 1.0) -> list[float]:
    radians = np.deg2rad(degrees)
    return [radius * np.cos(radians), radius * np.sin(radians), 0.0]


def _touching_hardware_pair(offset=(0.0, 0.0, 0.0)):
    # Two independently-modelled watertight solids (e.g. a hinge and the wall it's flush-mounted
    # against) that each contribute one wedge of two faces around a shared edge - the same shape
    # as af-rs6's still-broken roof-ridge case, deliberately at angles (10/80 vs 150/250 degrees)
    # with no antiparallel pair, so af-c5z's welded-seam tolerance does not apply and the edge is
    # judged purely on whether the two solids' edge vertices actually weld together.
    a_v0, a_v1 = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    b_v0 = list(np.array([0.0, 0.0, 0.0]) + np.array(offset))
    b_v1 = list(np.array([0.0, 0.0, 1.0]) + np.array(offset))
    positions = np.array(
        [a_v0, a_v1, _angle_point(10.0), _angle_point(80.0), b_v0, b_v1, _angle_point(150.0), _angle_point(250.0)]
    )
    indices = np.array([[0, 1, 2], [0, 1, 3], [4, 5, 6], [4, 5, 7]])
    return positions, indices


def test_two_flush_touching_solids_are_falsely_flagged_non_manifold():
    positions, indices = _touching_hardware_pair()
    _watertight, manifold = _manifold_checks(positions, indices)
    assert not manifold.passed
    assert "shared by more than two" in manifold.detail


def test_the_same_two_solids_pass_once_offset_past_weld_precision():
    # A standoff (assets.hardsurface.attach_to_surface's default STANDOFF_M is 2mm) comfortably
    # clears WELD_PRECISION's 1e-5 m rounding, so the two solids' shared-edge vertices no longer
    # weld together and each solid's own edges are correctly counted as manifold.
    positions, indices = _touching_hardware_pair(offset=(0.0, 0.002, 0.0))
    _watertight, manifold = _manifold_checks(positions, indices)
    assert manifold.passed


def test_outward_normals_pass_for_correctly_wound_tetrahedron():
    positions, indices, centroid = _tetrahedron()
    normals = positions - centroid
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    result = validate._check_inward_normals("m", positions, indices, normals)[0]
    assert result.passed


def test_inward_normals_are_flagged_when_a_face_is_flipped():
    positions, indices, centroid = _tetrahedron()
    normals = positions - centroid
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    flipped = indices.copy()
    flipped[0] = flipped[0][::-1]
    result = validate._check_inward_normals("m", positions, flipped, normals)[0]
    assert not result.passed


def test_uv_overlap_passes_for_side_by_side_triangles():
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    indices = np.array([[0, 1, 2], [3, 4, 5]])
    result = validate._check_uv_overlap("m", indices, uvs)[0]
    assert result.passed


def test_uv_overlap_flags_overlapping_triangles():
    uvs = np.array([[0, 0], [1, 0], [0, 1], [0.25, 0.25], [1.25, 0.25], [0.25, 1.25]], dtype=np.float64)
    indices = np.array([[0, 1, 2], [3, 4, 5]])
    result = validate._check_uv_overlap("m", indices, uvs)[0]
    assert not result.passed


def test_triangle_overlap_area_of_identical_triangles_is_its_own_area():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert validate._triangle_overlap_area(tri, tri) == pytest.approx(0.5)


def test_triangle_overlap_area_of_disjoint_triangles_is_zero():
    tri_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri_b = tri_a + np.array([10.0, 10.0])
    assert validate._triangle_overlap_area(tri_a, tri_b) == 0.0


def test_collapsed_uv_triangle_overlaps_nothing():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    point = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    assert validate._triangle_overlap_area(tri, point) == 0.0
    assert validate._triangle_overlap_area(point, tri) == 0.0


def test_check_triangle_budget():
    low, high = validate.TRI_BUDGETS["prop"]
    assert validate._check_triangle_budget(low, "prop").passed
    assert validate._check_triangle_budget(high, "prop").passed
    assert not validate._check_triangle_budget(low - 1, "prop").passed
    assert not validate._check_triangle_budget(high + 1, "prop").passed


def test_hero_vehicle_budget_extends_past_the_vehicle_ceiling():
    _, vehicle_high = validate.TRI_BUDGETS["vehicle"]
    assert validate._check_triangle_budget(vehicle_high + 1, "hero_vehicle").passed
    assert not validate._check_triangle_budget(vehicle_high + 1, "vehicle").passed


def test_check_origin_passes_when_base_centred():
    bbox_min = np.array([-0.5, 0.0, -0.5])
    bbox_max = np.array([0.5, 2.0, 0.5])
    assert validate._check_origin(bbox_min, bbox_max).passed


def test_check_origin_fails_when_offset():
    bbox_min = np.array([1.0, 0.0, -0.5])
    bbox_max = np.array([2.0, 2.0, 0.5])
    assert not validate._check_origin(bbox_min, bbox_max).passed


def test_check_node_transform_passes_for_identity():
    result = validate._check_node_transform(0, gltf.IDENTITY)[0]
    assert result.passed


def test_check_node_transform_fails_for_scaled_node():
    scaled = ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    result = validate._check_node_transform(0, scaled)[0]
    assert not result.passed


def test_transform_positions_applies_translation():
    matrix = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (5.0, 0.0, 0.0, 1.0))
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    world = validate._transform_positions(matrix, positions)
    assert world.tolist() == [[5.0, 0.0, 0.0], [6.0, 1.0, 1.0]]


def test_check_material_flags_missing_material():
    result = validate._check_material("m", {})[0]
    assert not result.passed
    result = validate._check_material("m", {"material": 0})[0]
    assert result.passed


def test_validate_asset_rejects_unknown_kind():
    with pytest.raises(validate.ValidationError):
        validate.validate_asset("does-not-exist", "spaceship")


def test_validate_asset_rejects_missing_file():
    with pytest.raises(validate.ValidationError):
        validate.validate_asset("does-not-exist", "prop")


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _write_tetrahedron_glb(path, node_names: tuple[str, ...] = ("",), extra_nodes: tuple[dict, ...] = ()) -> None:
    positions, indices, centroid = _tetrahedron()
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
        "scene": 0,
        "scenes": [{"nodes": list(range(len(node_names) + len(extra_nodes)))}],
        "nodes": [{"mesh": 0, "name": node_name} for node_name in node_names] + list(extra_nodes),
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


def test_validate_asset_end_to_end_on_a_closed_tetrahedron(monkeypatch, tmp_path):
    monkeypatch.setitem(validate.TRI_BUDGETS, "prop", (1, 10))
    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "test-prop" / "test-prop.glb")

    report = validate.validate_asset("test-prop", "prop")

    assert report.triangle_count == 4
    assert report.ok, [c for c in report.checks if not c.passed]
    names = {c.name for c in report.checks}
    assert any("watertight" in n for n in names)
    assert any("manifold" in n for n in names)


def test_report_as_dict_is_plain_json_even_with_numpy_values():
    report = validate.ValidationReport(
        name="m",
        kind="prop",
        triangle_count=np.int64(12),
        bbox_min=(np.float64(0.0), 0.0, 0.0),
        bbox_max=(1.0, 1.0, np.float32(1.0)),
        checks=[validate.CheckResult("c", np.bool_(True), "ok")],
    )
    payload = report.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert type(payload["passed_checks"]) is int
    assert type(payload["ok"]) is bool


def test_report_as_dict_lists_only_failing_checks():
    report = validate.ValidationReport(
        name="m",
        kind="prop",
        triangle_count=12,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        checks=[
            validate.CheckResult("good", True, "fine"),
            validate.CheckResult("bad", False, "broken"),
            validate.CheckResult("also good", True, "fine"),
        ],
    )
    payload = report.as_dict()
    assert payload["passed_checks"] == 2
    assert payload["failed_checks"] == 1
    assert payload["failures"] == [{"name": "bad", "detail": "broken"}]
    assert "checks" not in payload


def test_triangle_budget_applies_to_lod0_only(monkeypatch):
    monkeypatch.setitem(validate.TRI_BUDGETS, "prop", (1, 5))
    _write_tetrahedron_glb(
        config.ASSET_OUT_DIR / "test-lods" / "test-lods.glb",
        node_names=("test-lods_LOD0", "test-lods_LOD1", "test-lods_LOD2"),
    )

    report = validate.validate_asset("test-lods", "prop")

    assert report.triangle_count == 4
    assert report.lod_triangle_counts == {"LOD0": 4, "LOD1": 4, "LOD2": 4}
    assert report.ok, [c for c in report.checks if not c.passed]


def test_vegetation_skips_closed_mesh_and_uv_overlap_checks(monkeypatch):
    monkeypatch.setitem(validate.TRI_BUDGETS, "vegetation", (1, 10))
    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "test-plant" / "test-plant.glb")

    report = validate.validate_asset("test-plant", "vegetation")

    assert report.ok, [c for c in report.checks if not c.passed]
    names = {c.name for c in report.checks}
    assert not any("watertight" in n or "manifold" in n or "UV overlap" in n for n in names)
    assert any("outward normals" in n for n in names)


def test_rts_kinds_have_budgets_and_export_accepts_them():
    for kind in ("unit", "creature", "organic_prop"):
        low, high = validate.TRI_BUDGETS[kind]
        assert 0 < low < high
    assert validate.TRI_BUDGETS["unit"][1] < validate.TRI_BUDGETS["vehicle"][0]
    assert set(validate.LOD_CHAIN_KINDS) == {"unit", "creature", "organic_prop"}
    assert set(team_colour.KINDS) <= set(validate.LOD_CHAIN_KINDS)


def test_lod_chain_requires_three_lods():
    result = validate._check_lod_chain({0: 1000, 1: 500})
    assert not result.passed
    assert "LOD2" in result.detail


def test_lod_chain_requires_each_step_to_reduce():
    result = validate._check_lod_chain({0: 1000, 1: 900, 2: 300})
    assert not result.passed
    assert "LOD1" in result.detail


def test_lod_chain_passes_for_a_real_reduction():
    assert validate._check_lod_chain({0: 1000, 1: 500, 2: 200}).passed


def test_lod_chain_is_enforced_for_rts_kinds_only(monkeypatch):
    monkeypatch.setitem(validate.TRI_BUDGETS, "organic_prop", (1, 10))
    monkeypatch.setitem(validate.TRI_BUDGETS, "prop", (1, 10))
    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "stump" / "stump.glb")

    organic = validate.validate_asset("stump", "organic_prop")
    prop = validate.validate_asset("stump", "prop")

    assert [c.name for c in organic.checks if not c.passed] == ["LOD chain"]
    assert prop.ok


def _write_team_manifest(name: str, mask: np.ndarray | None) -> None:
    out_dir = config.ASSET_OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name}
    if mask is not None:
        manifest[team_colour.MANIFEST_KEY] = team_colour.manifest_entry("textures")
        (out_dir / "textures").mkdir(exist_ok=True)
        Image.fromarray(mask).save(out_dir / "textures" / team_colour.MASK_FILENAME)
    (out_dir / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _mask(size=(64, 64), painted=True) -> np.ndarray:
    mask = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    mask[..., 3] = 255
    if painted:
        mask[8:24, 8:24, 0] = 255
    return mask


def test_unit_without_team_mask_fails():
    _write_team_manifest("soldier", None)
    assert not validate._check_team_mask("soldier", "unit", 0).passed


def test_unit_without_manifest_fails():
    assert not validate._check_team_mask("ghost", "unit", 0).passed


def test_creature_without_team_mask_passes():
    _write_team_manifest("wolf", None)
    assert validate._check_team_mask("wolf", "creature", 0).passed


def test_unit_with_painted_mask_passes():
    _write_team_manifest("soldier", _mask())
    result = validate._check_team_mask("soldier", "unit", 0)
    assert result.passed, result.detail


def test_empty_team_mask_fails():
    _write_team_manifest("soldier", _mask(painted=False))
    assert not validate._check_team_mask("soldier", "unit", 0).passed


def test_non_power_of_two_team_mask_fails():
    _write_team_manifest("soldier", _mask(size=(60, 64)))
    assert not validate._check_team_mask("soldier", "unit", 0).passed


def test_team_mask_needs_uvs_on_every_primitive():
    _write_team_manifest("soldier", _mask())
    assert not validate._check_team_mask("soldier", "unit", 1).passed


def test_team_mask_manifest_entry_points_into_textures():
    entry = team_colour.manifest_entry("textures")
    assert entry["mask"] == "textures/team_mask.png"
    assert entry["texcoord"] == 0
    assert entry["channel"] == "r"


ATLAS_TEST_PX = 256


def _two_quads(gap_px: float):
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]], dtype=np.float64
    )
    indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    left = 40.0
    width = 60.0
    right = left + width + gap_px
    pixel_uvs = np.array(
        [
            [left, 40], [left + width, 40], [left + width, 100], [left, 100],
            [right, 40], [right + width, 40], [right + width, 100], [right, 100],
        ],
        dtype=np.float64,
    )
    return positions, indices, pixel_uvs / ATLAS_TEST_PX


def _painted(uvs: np.ndarray, indices: np.ndarray, colours, spill_px: int, background=(0.0, 0.0, 0.0)) -> np.ndarray:
    image = np.full((ATLAS_TEST_PX, ATLAS_TEST_PX, 3), background, dtype=np.float32)
    for quad, colour in enumerate(colours):
        corners = uvs[indices[quad * 2:quad * 2 + 2]].reshape(-1, 2) * ATLAS_TEST_PX
        x0, y0 = np.floor(corners.min(axis=0)).astype(int) - spill_px
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + spill_px
        image[y0:y1, x0:x1] = colour
    return image


def test_atlas_gutters_pass_for_far_apart_islands_padded_with_their_own_colour():
    positions, indices, uvs = _two_quads(gap_px=40.0)
    image = _painted(uvs, indices, [(1, 0, 0), (0, 0, 1)], spill_px=8)
    result = validate._check_atlas_gutters("m", positions, indices, uvs, image)[0]
    assert result.passed, result.detail


def test_atlas_gutters_flag_islands_a_pixel_apart():
    positions, indices, uvs = _two_quads(gap_px=1.0)
    image = _painted(uvs, indices, [(1, 0, 0), (0, 0, 1)], spill_px=0)
    result = validate._check_atlas_gutters("m", positions, indices, uvs, image)[0]
    assert not result.passed
    assert "closer than" in result.detail


def test_atlas_gutters_flag_a_gutter_painted_with_another_colour():
    positions, indices, uvs = _two_quads(gap_px=40.0)
    image = _painted(uvs, indices, [(1, 0, 0), (0, 0, 1)], spill_px=0, background=(0.0, 1.0, 0.0))
    result = validate._check_atlas_gutters("m", positions, indices, uvs, image)[0]
    assert not result.passed
    assert "carry another colour" in result.detail


def test_atlas_gutters_skip_without_a_texture():
    positions, indices, uvs = _two_quads(gap_px=1.0)
    assert validate._check_atlas_gutters("m", positions, indices, uvs, None)[0].passed


def test_uv_islands_join_a_quad_and_ignore_collapsed_faces():
    positions, indices, uvs = _two_quads(gap_px=40.0)
    uvs[4:] = uvs[4]
    islands = validate._uv_islands(positions, indices, uvs)
    assert (islands[2:] == -1).all()
    assert islands[0] == islands[1] >= 0


def test_texel_density_passes_for_uniform_scale():
    positions, indices, uvs = _two_quads(gap_px=40.0)
    assert validate._check_texel_density("m", positions, indices, uvs)[0].passed


def test_texel_density_flags_one_island_scaled_far_down():
    positions, indices, uvs = _two_quads(gap_px=40.0)
    uvs[4:] = uvs[4] + (uvs[4:] - uvs[4]) * 0.3
    assert not validate._check_texel_density("m", positions, indices, uvs)[0].passed


def _write_animated_manifest(name: str, animations: object) -> None:
    out_dir = config.ASSET_OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "animations": animations}
    (out_dir / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_static_asset_passes_animation_manifest_check():
    _write_team_manifest("crate", None)
    assert validate._check_animation_manifest("crate")[0].passed


def test_valid_animations_pass():
    clip = {
        "name": "working_wheel_spin",
        "state": "working",
        "kind": "rigid",
        "duration_s": 2.0,
        "fps": 24,
        "loop": True,
        "nodes": ["Wheel_ANIM"],
    }
    _write_animated_manifest("mill", [clip])
    result, entries = validate._check_animation_manifest("mill")
    assert result.passed, result.detail


def test_malformed_animations_fail():
    _write_animated_manifest("mill", [{"name": "spin"}])
    assert not validate._check_animation_manifest("mill")[0].passed


def _write_fx_asset(sockets: tuple[dict, ...], fx: list) -> None:
    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "kiln" / "kiln.glb", extra_nodes=sockets)
    manifest = {"name": "kiln", "fx": fx}
    (config.ASSET_OUT_DIR / "kiln" / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _fx_check(sockets: tuple[dict, ...], fx: list) -> validate.CheckResult:
    _write_fx_asset(sockets, fx)
    return next(check for check in validate.validate_asset("kiln", "prop").checks if check.name == "fx sockets")


SMOKE = fx_manifest.fx_entry("FX_smoke_chimney", "smoke", "idle", 8.0, 0.6, [0.4, 0.4, 0.4])


def test_static_asset_without_fx_passes_the_socket_check():
    result = _fx_check((), [])
    assert result.passed, result.detail


def test_socket_on_the_bbox_top_passes():
    result = _fx_check(({"name": "FX_smoke_chimney", "translation": [0.0, 1.0, 0.0]},), [SMOKE])
    assert result.passed, result.detail


def test_fx_entry_without_its_node_fails():
    result = _fx_check((), [SMOKE])
    assert not result.passed
    assert "no such node" in result.detail


def test_socket_outside_the_bbox_fails():
    result = _fx_check(({"name": "FX_smoke_chimney", "translation": [0.0, 3.0, 0.0]},), [SMOKE])
    assert not result.passed
    assert "outside" in result.detail


def test_socket_node_without_an_entry_fails():
    result = _fx_check(({"name": "FX_fire", "translation": [0.0, 0.5, 0.0]},), [])
    assert not result.passed
    assert "FX_fire" in result.detail


def test_socket_carrying_a_mesh_fails():
    result = _fx_check(({"name": "FX_smoke_chimney", "mesh": 0},), [SMOKE])
    assert not result.passed


def test_malformed_fx_entry_fails():
    result = _fx_check(({"name": "FX_smoke_chimney"},), [{"node": "FX_smoke_chimney"}])
    assert not result.passed
