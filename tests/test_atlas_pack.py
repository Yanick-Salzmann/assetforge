from __future__ import annotations

import random

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

