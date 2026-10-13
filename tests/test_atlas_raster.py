from __future__ import annotations

import numpy as np

from assets import atlas_raster


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[[x0, y0], [x1, y0], [x1, y1]], [[x0, y0], [x1, y1], [x0, y1]]], dtype=np.float64)


def test_rasterize_covers_pixel_centres_strictly_inside():
    labels = atlas_raster.rasterize(_square(2, 2, 6, 6), np.array([3, 3]), 8, 8)
    assert (labels[2:6, 2:6] == 3).sum() >= 12
    assert (labels[:2] == -1).all()
    assert (labels[:, 6:] == -1).all()


def test_rasterize_ignores_zero_area_triangles():
    collapsed = np.array([[[4.5, 4.5], [4.5, 4.5], [4.5, 4.5]]], dtype=np.float64)
    labels = atlas_raster.rasterize(collapsed, np.array([0]), 8, 8)
    assert (labels == -1).all()


def test_rasterize_is_winding_independent():
    forward = atlas_raster.rasterize(np.array([[[1, 1], [7, 1], [1, 7]]], dtype=np.float64), np.array([0]), 8, 8)
    backward = atlas_raster.rasterize(np.array([[[1, 1], [1, 7], [7, 1]]], dtype=np.float64), np.array([0]), 8, 8)
    assert (forward == backward).all()
    assert (forward == 0).any()


def test_shifted_moves_and_pads():
    array = np.arange(9).reshape(3, 3)
    moved = atlas_raster.shifted(array, 1, 0, -1)
    assert (moved[0] == -1).all()
    assert (moved[1:] == array[:2]).all()


def test_grow_sources_reports_islands_that_meet():
    labels = np.full((1, 12), -1, dtype=np.int32)
    labels[0, 0] = 0
    labels[0, 5] = 1
    _, _, touching = atlas_raster.grow_sources(labels, 2)
    assert touching == {(0, 1)}
    _, _, apart = atlas_raster.grow_sources(labels, 1)
    assert apart == set()


def test_grow_sources_points_every_grown_pixel_at_a_covered_pixel():
    labels = np.full((5, 5), -1, dtype=np.int32)
    labels[2, 2] = 0
    grown, source, _ = atlas_raster.grow_sources(labels)
    assert (grown == 0).all()
    assert (source == 2 * 5 + 2).all()


def test_dilate_fills_each_footprint_only_from_its_own_island():
    pixels = np.zeros((8, 16, 3), dtype=np.float32)
    labels = np.full((8, 16), -1, dtype=np.int32)
    labels[3:5, 2:5] = 0
    labels[3:5, 11:14] = 1
    pixels[labels == 0] = (1.0, 0.0, 0.0)
    pixels[labels == 1] = (0.0, 0.0, 1.0)
    atlas_raster.dilate(pixels, labels, [(0, 0, 8, 8), (8, 0, 16, 8)])
    assert (pixels[:, :8] == (1.0, 0.0, 0.0)).all()
    assert (pixels[:, 8:] == (0.0, 0.0, 1.0)).all()


def test_dilate_with_a_shared_fill_map_matches_computing_it_per_image():
    labels = np.full((8, 16), -1, dtype=np.int32)
    labels[2:6, 1:4] = 0
    labels[3:5, 10:15] = 1
    footprints = [(0, 0, 8, 8), (8, 0, 16, 8)]
    rng = np.random.default_rng(3)
    original = rng.random((8, 16, 3)).astype(np.float32)
    direct = original.copy()
    shared = original.copy()
    atlas_raster.dilate(direct, labels, footprints)
    atlas_raster.dilate(shared, labels, footprints, atlas_raster.fill_map(labels, footprints))
    assert (direct == shared).all()
    assert (direct[labels >= 0] == original[labels >= 0]).all()


def test_dilate_gives_empty_footprints_and_unowned_pixels_the_mean_colour():
    pixels = np.zeros((4, 12, 3), dtype=np.float32)
    labels = np.full((4, 12), -1, dtype=np.int32)
    labels[1:3, 1:3] = 0
    pixels[labels == 0] = (0.8, 0.4, 0.2)
    atlas_raster.dilate(pixels, labels, [(0, 0, 4, 4), (4, 0, 8, 4)])
    assert np.allclose(pixels[:, 4:], (0.8, 0.4, 0.2))
