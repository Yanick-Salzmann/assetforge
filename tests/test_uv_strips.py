from __future__ import annotations

import math

import numpy as np
import pytest

from assets import uv_strips


def _lathe(profile, segments, rotation=None, offset=(0.0, 0.0, 0.0)):
    positions = []
    rings = []
    for radius, z in profile:
        if radius == 0.0:
            positions.append((0.0, 0.0, z))
            rings.append([len(positions) - 1] * segments)
            continue
        ring = []
        for k in range(segments):
            angle = math.pi / segments + k * 2.0 * math.pi / segments
            positions.append((radius * math.cos(angle), radius * math.sin(angle), z))
            ring.append(len(positions) - 1)
        rings.append(ring)
    triangles = []
    for lower, upper in zip(rings, rings[1:]):
        for k in range(segments):
            k2 = (k + 1) % segments
            quad = [lower[k], lower[k2], upper[k2], upper[k]]
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                if len(set(tri)) == 3:
                    triangles.append(tri)
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        if len(set(ring)) == 1:
            continue
        positions.append((0.0, 0.0, positions[ring[0]][2]))
        centre = len(positions) - 1
        for k in range(segments):
            k2 = (k + 1) % segments
            triangles.append((centre, ring[k2], ring[k]) if flip else (centre, ring[k], ring[k2]))
    points = np.asarray(positions, dtype=np.float64)
    if rotation is not None:
        points = points @ rotation.T
    return points + np.asarray(offset), np.asarray(triangles, dtype=np.int64)


def _box():
    positions = np.array([(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (0.0, 2.0)])
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6)]
    triangles = []
    for a, b, c, d in quads:
        triangles.extend([(a, b, c), (a, c, d)])
    return positions, np.asarray(triangles, dtype=np.int64)


def _areas(positions, triangles):
    corners = positions[triangles]
    return 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)


def _uv_area(points):
    (x0, y0), (x1, y1), (x2, y2) = points
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _assert_valid(strips, positions, triangles):
    areas = _areas(positions, triangles)
    seen = set()
    for strip in strips:
        assert not seen & set(strip.triangles)
        seen |= set(strip.triangles)
        assert not uv_strips.any_overlap(strip.points)
        for row, points in zip(strip.triangles, strip.points):
            ratio = _uv_area(points) / areas[row]
            assert uv_strips.BAND_MIN_AREA_RATIO <= ratio <= uv_strips.BAND_MAX_AREA_RATIO
        _assert_continuous(strip, triangles)


def _assert_continuous(strip, triangles):
    period = 2.0 * math.pi * strip.radius
    seen = {}
    for row, points in zip(strip.triangles, strip.points):
        for vertex, (u, v) in zip(triangles[row], points):
            first = seen.setdefault(int(vertex), (u, v))
            assert v == pytest.approx(first[1], abs=1e-9)
            turns = round((u - first[0]) / period)
            assert u - first[0] - turns * period == pytest.approx(0.0, abs=1e-9 * period)


def test_straight_cylinder_side_is_one_strip_and_caps_are_left_out():
    positions, triangles = _lathe([(0.5, 0.0), (0.5, 2.0)], segments=16)
    strips = uv_strips.strip_islands(positions, triangles)
    assert len(strips) == 1
    assert len(strips[0].triangles) == 32
    assert strips[0].radius == pytest.approx(0.5 * math.cos(math.pi / 16), rel=0.05)
    _assert_valid(strips, positions, triangles)
    us = [u for points in strips[0].points for u, _ in points]
    vs = [v for points in strips[0].points for _, v in points]
    assert max(us) - min(us) == pytest.approx(2.0 * math.pi * strips[0].radius, rel=0.02)
    assert max(vs) - min(vs) == pytest.approx(2.0, rel=1e-6)


def test_eight_segment_frustum_is_unwrapped():
    positions, triangles = _lathe([(0.4, 0.0), (0.3, 1.5)], segments=8)
    strips = uv_strips.strip_islands(positions, triangles)
    assert len(strips) == 1
    assert len(strips[0].triangles) == 16
    _assert_valid(strips, positions, triangles)


def test_pot_profile_gives_a_few_bands_covering_every_side_face():
    profile = [(0.0, 0.0), (0.25, 0.02), (0.35, 0.15), (0.4, 0.3), (0.38, 0.45), (0.3, 0.55), (0.2, 0.62), (0.22, 0.7), (0.0, 0.7)]
    positions, triangles = _lathe(profile, segments=24)
    strips = uv_strips.strip_islands(positions, triangles)
    _assert_valid(strips, positions, triangles)
    assert 2 <= len(strips) <= 6
    assert sum(len(s.triangles) for s in strips) >= 24 * 2 * 5


def test_tilted_offset_log_is_found_through_its_principal_axis():
    angle = math.radians(37.0)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])
    positions, triangles = _lathe([(0.15, 0.0), (0.15, 3.0)], segments=12, rotation=rotation, offset=(4.0, -2.0, 1.0))
    strips = uv_strips.strip_islands(positions, triangles)
    assert len(strips) == 1
    assert len(strips[0].triangles) == 24
    _assert_valid(strips, positions, triangles)


def test_cone_side_keeps_its_apex_in_one_place():
    positions, triangles = _lathe([(0.5, 0.0), (0.0, 1.0)], segments=16)
    strips = uv_strips.strip_islands(positions, triangles)
    _assert_valid(strips, positions, triangles)


def test_box_and_hexagonal_prism_are_left_to_axis_projection():
    assert uv_strips.strip_islands(*_box()) == []
    assert uv_strips.strip_islands(*_lathe([(0.5, 0.0), (0.5, 1.0)], segments=6)) == []


def test_parts_are_handled_independently_and_rows_restrict_the_search():
    cylinder, cylinder_triangles = _lathe([(0.5, 0.0), (0.5, 2.0)], segments=16, offset=(5.0, 0.0, 0.0))
    box, box_triangles = _box()
    positions = np.concatenate([box, cylinder])
    triangles = np.concatenate([box_triangles, cylinder_triangles + len(box)])
    strips = uv_strips.strip_islands(positions, triangles)
    assert len(strips) == 1
    assert min(strips[0].triangles) >= len(box_triangles)
    excluded = set(strips[0].triangles[:4])
    rows = [r for r in range(len(triangles)) if r not in excluded]
    restricted = uv_strips.strip_islands(positions, triangles, rows)
    assert all(not excluded & set(s.triangles) for s in restricted)


def test_wobbled_pot_still_unwraps_without_overlap():
    profile = [(0.0, 0.0), (0.3, 0.02), (0.4, 0.3), (0.3, 0.55), (0.22, 0.7), (0.0, 0.7)]
    positions, triangles = _lathe(profile, segments=20)
    rng = np.random.default_rng(3)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=3)
    positions = positions + 0.01 * np.sin(positions * 9.0 + phases)
    strips = uv_strips.strip_islands(positions, triangles)
    assert strips
    _assert_valid(strips, positions, triangles)


def test_shapes_overlap_treats_touching_as_separate():
    a = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    b = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    c = [(0.2, 0.2), (1.0, 0.2), (0.2, 1.0)]
    assert not uv_strips.shapes_overlap(a, b)
    assert uv_strips.shapes_overlap(a, c)
