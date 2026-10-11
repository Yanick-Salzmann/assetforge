from __future__ import annotations

import math

import numpy as np
import pytest

from assets import flora, foliage_atlas
from terrain import vegetation


@pytest.fixture(scope="module")
def meshes() -> dict[str, tuple[str, flora.MeshData]]:
    return {name: (kind, build()) for name, (kind, build) in flora.MODELS.items()}


def test_every_model_fits_its_kind_budget(meshes):
    for name, (kind, mesh) in meshes.items():
        assert kind in vegetation.KIND_TRI_BUDGET, name
        assert 0 < mesh.triangles() <= vegetation.KIND_TRI_BUDGET[kind], name


def test_every_model_stands_on_the_ground_at_the_origin(meshes):
    for name, (_, mesh) in meshes.items():
        low, high = mesh.bounds()
        assert low[2] == pytest.approx(0.0), name
        assert abs(low[0] + high[0]) * 0.5 < 0.6, name
        assert abs(low[1] + high[1]) * 0.5 < 0.6, name


def test_corner_attributes_match_faces(meshes):
    for name, (_, mesh) in meshes.items():
        assert len(mesh.uvs) == len(mesh.faces) == len(mesh.normals), name
        for face, uvs, normals in zip(mesh.faces, mesh.uvs, mesh.normals):
            assert len(uvs) == len(normals) == len(face)


def test_uvs_stay_inside_the_atlas_and_one_rect(meshes):
    rects = [foliage_atlas.bark_rect()] + [foliage_atlas.cell_rect(name) for name in foliage_atlas.CELL_NAMES]
    for name, (_, mesh) in meshes.items():
        for uvs in mesh.uvs:
            assert any(
                all(rect.u0 - 1e-9 <= u <= rect.u1 + 1e-9 and rect.v0 - 1e-9 <= v <= rect.v1 + 1e-9 for u, v in uvs)
                for rect in rects
            ), name


def test_normals_are_unit_and_agree_with_winding(meshes):
    for name, (_, mesh) in meshes.items():
        for face, normals in zip(mesh.faces, mesh.normals):
            a, b, c = (mesh.vertices[index] for index in face[:3])
            winding = flora.cross(flora.sub(b, a), flora.sub(c, a))
            if flora.length(winding) < 1e-12:
                continue
            for normal in normals:
                assert flora.length(normal) == pytest.approx(1.0, abs=1e-6)
            assert flora.dot(winding, flora.add(flora.add(normals[0], normals[1]), normals[2])) > 0.0, name


def test_models_are_deterministic():
    _, build = flora.MODELS["oak_a"]
    assert build() == build()


def test_cells_tile_the_atlas_without_overlap():
    rects = [foliage_atlas.bark_rect()] + [foliage_atlas.cell_rect(name) for name in foliage_atlas.CELL_NAMES]
    for rect in rects:
        assert 0.0 <= rect.u0 < rect.u1 <= 1.0
        assert 0.0 <= rect.v0 < rect.v1 <= 1.0
    for index, a in enumerate(rects):
        for b in rects[index + 1:]:
            assert a.u1 <= b.u0 or b.u1 <= a.u0 or a.v1 <= b.v0 or b.v1 <= a.v0


@pytest.mark.parametrize("name", ["grass", "oak"])
def test_a_cell_has_a_transparent_gutter_and_bled_colour(name):
    cell = foliage_atlas.paint_cell(name, seed=3)
    gutter = foliage_atlas.GUTTER
    assert cell.shape == (foliage_atlas.CELL_SIZE, foliage_atlas.CELL_SIZE, 4)
    assert np.all(cell[:gutter, :, 3] == 0.0)
    assert np.all(cell[-gutter:, :, 3] == 0.0)
    assert np.all(cell[:, :gutter, 3] == 0.0)
    assert np.all(cell[:, -gutter:, 3] == 0.0)
    opaque = cell[..., 3] >= foliage_atlas.ALPHA_CUTOFF
    assert 0.05 < opaque.mean() < 0.9
    assert np.all(cell[~opaque, :3].sum(axis=-1) > 0.0)
    assert np.array_equal(cell, foliage_atlas.paint_cell(name, seed=3))


def test_bleed_keeps_alpha_and_fills_colour():
    pixels = np.zeros((8, 8, 4), dtype=np.float32)
    pixels[3, 3] = (0.2, 0.6, 0.1, 1.0)
    bled = foliage_atlas.bleed(pixels)
    assert np.array_equal(bled[..., 3], pixels[..., 3])
    assert np.allclose(bled[0, 0, :3], (0.2, 0.6, 0.1))


def test_bark_tiles_down_the_strip():
    tile = np.random.default_rng(0).random((foliage_atlas.BARK_WIDTH, foliage_atlas.BARK_WIDTH, 4)).astype(np.float32)
    strip = foliage_atlas.tile_bark(tile)
    assert strip.shape == (foliage_atlas.ATLAS_SIZE, foliage_atlas.BARK_WIDTH, 4)
    assert np.array_equal(strip[foliage_atlas.BARK_WIDTH:2 * foliage_atlas.BARK_WIDTH, :, :3], tile[..., :3])
    assert np.all(strip[..., 3] == 1.0)


def test_card_droop_bends_the_tip_down():
    builder = flora.Builder()
    rect = foliage_atlas.cell_rect("fern")
    builder.card((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 0.5, rect, outward=(0.0, 0.0, 1.0), segments=3, droop=math.pi / 3)
    tip_z = builder.mesh.vertices[-1][2]
    assert tip_z < 1.0
    assert builder.mesh.triangles() == 6
