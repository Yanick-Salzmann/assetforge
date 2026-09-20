from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from terrain import normal, preview
from terrain.config import MapConfig

CPU = "cpu"


def cfg(resolution: int = 32, **kwargs) -> MapConfig:
    base = {"name": "normal-unit", "resolution": resolution, "world_size_m": 4096.0, "seed": 7}
    base.update(kwargs)
    return MapConfig(**base)


def ramp(resolution: int = 32, axis: int = 1) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, resolution, device=CPU, dtype=torch.float32)
    field = line.reshape(1, -1).expand(resolution, resolution).contiguous()
    return field if axis == 1 else field.T.contiguous()


def test_flat_field_normal_points_straight_up():
    flat = torch.full((16, 16), 0.4, device=CPU)
    n = normal.compute(flat, cfg(16))
    assert n.shape == (16, 16, 3)
    assert torch.allclose(n[..., 0], torch.zeros(16, 16), atol=1e-6)
    assert torch.allclose(n[..., 1], torch.zeros(16, 16), atol=1e-6)
    assert torch.allclose(n[..., 2], torch.ones(16, 16), atol=1e-6)


def test_normals_are_unit_length():
    c = cfg(32, height_range_m=600.0)
    n = normal.compute(ramp(32), c)
    lengths = n.norm(dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)


def test_slope_tilts_the_normal_away_from_up():
    c = cfg(32, height_range_m=600.0)
    n = normal.compute(ramp(32), c)
    assert float(n[16, 16, 0].abs()) > 1e-3
    assert float(n[16, 16, 2]) < 1.0


def test_normal_matches_hillshade_gradient_sign():
    c = cfg(32, height_range_m=600.0)
    height = ramp(32)
    n = normal.compute(height, c)
    metres = height.mul(c.height_range_m)
    dzdx, dzdy = preview.gradients(metres, c.metres_per_pixel)
    unnormalised = torch.stack((dzdx.neg(), dzdy.neg(), torch.ones_like(dzdx)), dim=-1)
    expected = unnormalised / unnormalised.norm(dim=-1, keepdim=True)
    assert torch.allclose(n, expected, atol=1e-5)


def test_does_not_mutate_input():
    height = ramp()
    before = height.clone()
    normal.compute(height, cfg())
    assert torch.equal(height, before)


def test_to_image_maps_up_normal_to_mid_blue():
    flat = torch.full((8, 8), 0.5, device=CPU)
    image = normal.to_image(normal.compute(flat, cfg(8)))
    assert image.dtype == np.uint8
    assert image.shape == (8, 8, 3)
    assert tuple(image[0, 0]) == (127, 127, 255) or tuple(image[0, 0]) == (128, 128, 255)


def test_write_produces_full_resolution_png(tmp_path):
    c = cfg(16)
    target = normal.write(ramp(16), c, tmp_path / "normal.png")
    assert target.is_file()
    with Image.open(target) as image:
        assert image.size == (16, 16)
        assert image.mode == "RGB"
