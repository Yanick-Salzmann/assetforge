from __future__ import annotations

import itertools

import numpy as np
import pytest

from assets import impostor


def test_frames_look_from_the_upper_hemisphere():
    for column, row in itertools.product(range(impostor.FRAMES), repeat=2):
        direction = impostor.frame_direction(column, row)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        assert direction[1] > 0.0


def test_encode_inverts_decode():
    for column, row in itertools.product(range(impostor.FRAMES), repeat=2):
        u, v = impostor.encode(impostor.frame_direction(column, row))
        assert (u, v) == pytest.approx(impostor.frame_uv(column, row))


def test_the_centre_frames_look_down_and_the_corners_along_the_horizon():
    centre = impostor.frame_direction(impostor.FRAMES // 2, impostor.FRAMES // 2)
    corner = impostor.frame_direction(0, 0)
    assert centre[1] > 0.95
    assert corner[1] < 0.2


def test_frame_rotation_turns_the_view_to_z_and_its_up_to_y():
    for column, row in itertools.product(range(impostor.FRAMES), repeat=2):
        direction = impostor.gltf_to_blender(impostor.frame_direction(column, row))
        rotation = impostor.frame_rotation(direction)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert rotation @ np.asarray(direction) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
        _, up = impostor.view_basis(direction)
        assert rotation @ up == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_axis_conversion_round_trips():
    vector = (0.3, -0.5, 0.8)
    assert impostor.blender_to_gltf(impostor.gltf_to_blender(vector)) == vector
    assert impostor.blender_to_gltf((0.0, 0.0, 1.0)) == (0.0, 1.0, -0.0)
