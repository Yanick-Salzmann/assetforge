from __future__ import annotations

import random

import pytest

from assets import atlas_pack


def _many_parts(seed: int = 7, count: int = 387) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    sizes = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.6:
            sizes.append((rng.uniform(0.02, 0.2), rng.uniform(0.02, 0.2)))
        elif kind < 0.9:
            sizes.append((rng.uniform(0.3, 1.2), rng.uniform(0.05, 0.3)))
        else:
            sizes.append((rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)))
    return sizes


def _overlaps(a, b) -> bool:
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_pack_square_fills_both_axes_of_the_atlas():
    sizes = _many_parts()
    margin = 0.01
    offsets, side = atlas_pack.pack_square(sizes, margin)
    used_x = max(x + w for (x, _), (w, _) in zip(offsets, sizes)) / side
    used_y = max(y + h for (_, y), (_, h) in zip(offsets, sizes)) / side
    assert used_x > 0.9
    assert used_y > 0.9


def test_pack_square_never_overlaps_and_stays_inside_the_square():
    sizes = _many_parts(seed=3, count=200)
    offsets, side = atlas_pack.pack_square(sizes, 0.01)
    rects = [(x, y, w, h) for (x, y), (w, h) in zip(offsets, sizes)]
    for x, y, w, h in rects:
        assert x >= 0.0 and y >= 0.0
        assert x + w <= side + 1e-9 and y + h <= side + 1e-9
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlaps(rects[i], rects[j])


def test_pack_square_beats_the_old_fixed_width_guess():
    sizes = _many_parts()
    margin = 0.01
    total_area = sum(w * h for w, h in sizes)
    _, max_x, total_height = atlas_pack.shelf_pack(sizes, total_area**0.5 * 1.3, margin)
    _, side = atlas_pack.pack_square(sizes, margin)
    assert side <= max(max_x, total_height)


def test_pack_square_handles_one_wide_island():
    offsets, side = atlas_pack.pack_square([(5.0, 0.1), (0.1, 0.1)], 0.01)
    assert side >= 5.0
    assert len(offsets) == 2


def test_pack_square_of_nothing():
    offsets, side = atlas_pack.pack_square([], 0.01)
    assert offsets == []
    assert side > 0.0


def test_pack_square_is_deterministic():
    sizes = _many_parts()
    assert atlas_pack.pack_square(sizes, 0.01) == atlas_pack.pack_square(sizes, 0.01)


def _pixel_rects_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_pack_pixels_footprints_tile_without_overlap_inside_the_atlas():
    packing = atlas_pack.pack_pixels(_many_parts(seed=3, count=300), 2048, 16, 4)
    for x0, y0, x1, y1 in packing.footprints:
        assert 0 <= x0 < x1 <= 2048
        assert 0 <= y0 < y1 <= 2048
    rects = packing.footprints
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _pixel_rects_overlap(rects[i], rects[j])


def test_pack_pixels_keeps_a_full_gutter_between_island_contents():
    sizes = _many_parts(seed=5, count=200)
    packing = atlas_pack.pack_pixels(sizes, 2048, 16, 4)
    contents = [(x, y, x + w * packing.density, y + h * packing.density) for (x, y), (w, h) in zip(packing.content_origins, sizes)]
    for i in range(len(contents)):
        a = contents[i]
        for j in range(i + 1, len(contents)):
            b = contents[j]
            gap_x = max(b[0] - a[2], a[0] - b[2])
            gap_y = max(b[1] - a[3], a[1] - b[3])
            assert max(gap_x, gap_y) >= 16 - 1e-6


def test_pack_pixels_aligns_footprints_and_contents_to_the_block_grid():
    packing = atlas_pack.pack_pixels(_many_parts(), 2048, 16, 4)
    for (x, y), footprint in zip(packing.content_origins, packing.footprints):
        assert x % 4 == 0 and y % 4 == 0
        assert all(edge % 4 == 0 for edge in footprint)


def test_pack_pixels_uses_one_density_and_fills_most_of_the_atlas():
    packing = atlas_pack.pack_pixels(_many_parts(), 2048, 16, 4)
    used = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in packing.footprints)
    assert packing.density > 0.0
    assert used > 0.6 * 2048 * 2048


def test_pack_pixels_gutter_is_independent_of_source_units():
    sizes = _many_parts(seed=11, count=120)
    metres = atlas_pack.pack_pixels(sizes, 1024, 8, 4)
    millimetres = atlas_pack.pack_pixels([(w * 1000.0, h * 1000.0) for w, h in sizes], 1024, 8, 4)
    assert abs(metres.density - millimetres.density * 1000.0) <= metres.density * 1e-3
    assert metres.gutter_px == millimetres.gutter_px == 8


def test_pack_pixels_rejects_a_gutter_off_the_block_grid():
    with pytest.raises(ValueError):
        atlas_pack.pack_pixels([(1.0, 1.0)], 256, 6, 4)


def test_pack_pixels_raises_when_islands_cannot_fit_at_any_size():
    with pytest.raises(ValueError):
        atlas_pack.pack_pixels([(1.0, 1.0)] * 400, 64, 8, 4)


def test_pack_pixels_of_nothing():
    assert atlas_pack.pack_pixels([], 2048, 16, 4).footprints == ()

