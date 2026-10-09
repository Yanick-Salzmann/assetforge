from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from terrain import channels as channel_module
from terrain import macro
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 32


def cfg() -> MapConfig:
    return MapConfig(name="unit", resolution=SIZE, world_size_m=512.0, seed=5)


def random_stack(seed: int) -> channel_module.ChannelStack:
    c = cfg()
    built = channel_module.ChannelStack(c, CPU)
    generator = torch.Generator(device=CPU).manual_seed(seed)
    for name in CHANNEL_NAMES:
        built[name] = torch.rand(c.shape, generator=generator, device=CPU)
    return built


def uniform_stack(**values: float) -> channel_module.ChannelStack:
    c = cfg()
    built = channel_module.ChannelStack(c, CPU)
    for name, value in values.items():
        built[name] = torch.full(c.shape, value)
    return built


def test_neutral_channels_give_a_neutral_multiplier():
    tint = macro.compute(uniform_stack(patchiness_coarse=0.5, patchiness_mid=0.5, moisture=0.5, wetness=0.5))
    assert torch.allclose(tint, torch.ones(SIZE, SIZE, 3))


def test_missing_channels_fall_back_to_neutral():
    tint = macro.compute(uniform_stack())
    assert torch.allclose(tint, torch.ones(SIZE, SIZE, 3))


def test_wet_ground_is_darker_and_cooler_than_dry_ground():
    wet = macro.compute(uniform_stack(moisture=1.0, wetness=1.0))[0, 0]
    dry = macro.compute(uniform_stack(moisture=0.0, wetness=0.0))[0, 0]
    assert wet.mean() < dry.mean()
    assert wet[0] / wet[2] < dry[0] / dry[2]


def test_coarse_patchiness_varies_value():
    bright = macro.compute(uniform_stack(patchiness_coarse=1.0))[0, 0]
    dull = macro.compute(uniform_stack(patchiness_coarse=0.0))[0, 0]
    assert bright.mean() > dull.mean()


def test_multiplier_stays_a_gentle_tint():
    tint = macro.compute(random_stack(3))
    assert float(tint.min()) > 0.6
    assert float(tint.max()) < 1.4


def test_tint_is_low_frequency():
    built = random_stack(4)
    tint = macro.compute(built)[:, :, 1]
    raw_step = built["patchiness_coarse"].diff(dim=1).abs().mean()
    tint_step = tint.diff(dim=1).abs().mean()
    assert tint_step < 0.1 * raw_step


def test_write_is_deterministic_rgb_at_resolution(tmp_path):
    first = macro.write(random_stack(7), tmp_path / "a" / macro.COLOUR_MACRO_NAME)
    second = macro.write(random_stack(7), tmp_path / "b" / macro.COLOUR_MACRO_NAME)
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.mode == "RGB"
        assert image.size == (SIZE, SIZE)


def test_neutral_encodes_to_128():
    packed = macro.to_image(torch.ones(2, 2, 3))
    assert (packed == macro.NEUTRAL_BYTE).all()


def test_params_reject_out_of_range_values():
    with pytest.raises(MapConfigError):
        macro.MacroParams(near_strength=1.5)
    with pytest.raises(MapConfigError):
        macro.MacroParams(near_distance_m=900.0, far_distance_m=600.0)
    with pytest.raises(MapConfigError):
        macro.MacroParams(hue=0.9)
