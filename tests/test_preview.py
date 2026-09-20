from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from terrain import preview, synth
from terrain.config import MapConfig, MapConfigError

CPU = "cpu"


def cfg(resolution: int = 64, **kwargs) -> MapConfig:
    base = {"name": "preview-unit", "resolution": resolution, "world_size_m": 4096.0, "seed": 7}
    base.update(kwargs)
    return MapConfig(**base)


def ramp(resolution: int = 64, axis: int = 1) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, resolution, device=CPU, dtype=torch.float32)
    field = line.reshape(1, -1).expand(resolution, resolution).contiguous()
    return field if axis == 1 else field.T.contiguous()


def test_apply_colormap_spans_the_table():
    field = torch.tensor([[0.0, 0.5, 1.0]], device=CPU)
    rgb = preview.apply_colormap(field, "grey")
    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[0, 2]) == (255, 255, 255)
    assert 120 <= int(rgb[0, 1, 0]) <= 135


def test_apply_colormap_clamps_and_rejects_unknown_names():
    field = torch.tensor([[-4.0, 9.0]], device=CPU)
    rgb = preview.apply_colormap(field, "grey")
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[0, 1]) == (255, 255, 255)
    with pytest.raises(MapConfigError):
        preview.apply_colormap(field, "nope")


def test_renderers_do_not_mutate_their_input():
    height = ramp()
    before = height.clone()
    preview.compose_hillshade(height, cfg())
    preview.apply_colormap(height, "terrain")
    preview.contour_masks(height, cfg(), 25.0)
    assert torch.equal(height, before)


def test_gradients_match_the_world_scale():
    c = cfg(32, world_size_m=320.0)
    metres = ramp(32) * 100.0
    dzdx, dzdy = preview.gradients(metres, c.metres_per_pixel)
    expected = (100.0 / 31.0) / c.metres_per_pixel
    assert float(dzdx[16, 16]) == pytest.approx(expected, rel=1e-4)
    assert float(dzdy.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_hillshade_is_bounded_and_lights_from_the_azimuth():
    c = cfg(64, height_range_m=600.0)
    shade = preview.hillshade(ramp(), c)
    assert shade.shape == c.shape
    assert float(shade.min()) >= preview.DEFAULT_AMBIENT - 1e-6
    assert float(shade.max()) <= 1.0
    west = preview.hillshade(ramp(), c, azimuth_deg=270.0)
    east = preview.hillshade(ramp(), c, azimuth_deg=90.0)
    assert float(west.mean()) != pytest.approx(float(east.mean()))


def test_hillshade_of_a_flat_field_is_uniform():
    flat = torch.full((32, 32), 0.4, device=CPU)
    shade = preview.hillshade(flat, cfg(32))
    assert float(shade.std()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"altitude_deg": 0.0},
        {"altitude_deg": 91.0},
        {"ambient": 1.0},
        {"ambient": -0.1},
        {"z_scale": 0.0},
    ],
)
def test_hillshade_validates_its_arguments(kwargs):
    with pytest.raises(MapConfigError):
        preview.hillshade(ramp(16), cfg(16), **kwargs)


def test_nice_interval_rounds_up_to_a_readable_step():
    assert preview.nice_interval(24.0) == 25.0
    assert preview.nice_interval(1.0) == 1.0
    assert preview.nice_interval(0.03) == pytest.approx(0.05)
    assert preview.nice_interval(600.0) == 1000.0
    with pytest.raises(MapConfigError):
        preview.nice_interval(0.0)


def test_default_contour_interval_gives_roughly_the_requested_count():
    c = cfg(height_range_m=600.0)
    interval = preview.default_contour_interval(c, 24)
    assert 0.5 <= (c.height_range_m / interval) / 24.0 <= 2.0


def test_contour_masks_are_exclusive_and_track_the_interval():
    c = cfg(64, height_range_m=1000.0)
    minor, major = preview.contour_masks(ramp(), c, 100.0, index_every=5)
    assert minor.dtype == torch.bool
    assert not bool((minor & major).any())
    assert int(minor.sum()) > 0
    assert int(major.sum()) > 0
    coarse_minor, _ = preview.contour_masks(ramp(), c, 250.0, index_every=5)
    assert int(coarse_minor.sum()) < int(minor.sum())


def test_contour_masks_are_empty_on_a_flat_field():
    flat = torch.full((32, 32), 0.25, device=CPU)
    minor, major = preview.contour_masks(flat, cfg(32), 10.0)
    assert not bool(minor.any())
    assert not bool(major.any())


def test_compose_hillshade_marks_the_sea():
    c = cfg(64, height_range_m=600.0, sea_level_m=180.0)
    rgb = preview.compose_hillshade(ramp(), c, relief_tint=0.0)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    low = rgb[32, 2]
    high = rgb[32, 61]
    assert int(low[2]) > int(low[0])
    assert int(high[2]) <= int(high[0]) + 2


def test_render_hillshade_writes_a_png_within_the_size_budget(tmp_path):
    c = cfg(128)
    height = synth.heightfield(c, "continental", device=CPU)
    path = preview.render_hillshade(height, c, tmp_path / "hillshade.png", max_size=64)
    assert path.exists()
    with Image.open(path) as image:
        assert image.size == (64, 64)
        assert image.mode == "RGB"


def test_render_hillshade_defaults_to_the_map_out_dir():
    c = cfg(32)
    assert preview.render_hillshade.__doc__
    expected = c.out_dir() / preview.HILLSHADE_NAME
    assert expected.name == "preview_hillshade.png"


def test_channel_sheet_from_a_mapping(tmp_path):
    channels = {"height": ramp(32), "slope": ramp(32, axis=0), "flow": ramp(32) * 0.25}
    path = preview.render_channel_sheet(channels, tmp_path / "channels.png", tile=32, columns=2)
    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.width >= 2 * 32
        assert image.height >= 2 * (32 + preview.SHEET_LABEL_HEIGHT)


def test_channel_sheet_from_a_stack_uses_the_canonical_names():
    stack = torch.stack([ramp(16), ramp(16, axis=0), ramp(16) * 0.5], dim=2)
    sheet = preview.compose_channel_sheet(stack, tile=16, columns=3)
    assert sheet.size[0] >= 3 * 16
    with pytest.raises(MapConfigError):
        preview.compose_channel_sheet(stack, names=["only-one"])
    with pytest.raises(MapConfigError):
        preview.compose_channel_sheet(ramp(16))


def test_channel_sheet_rejects_an_empty_stack():
    with pytest.raises(MapConfigError):
        preview.compose_channel_sheet({})


def test_channel_colormap_falls_back_to_viridis():
    assert preview.channel_colormap("height") == "terrain"
    assert preview.channel_colormap("strata") == "viridis"


def test_render_channel_rescales_a_narrow_field(tmp_path):
    field = torch.full((32, 32), 0.5, device=CPU)
    field[:, 16:] = 0.51
    path = preview.render_channel(field, tmp_path / "flat.png", "slope", max_size=None)
    with Image.open(path) as image:
        pixels = np.asarray(image)
    assert not np.array_equal(pixels[0, 0], pixels[0, 31])


def weights3(resolution: int = 16) -> torch.Tensor:
    a = ramp(resolution)
    b = 1.0 - a
    zero = torch.zeros_like(a)
    return torch.stack([a, b, zero], dim=-1)


def test_compose_splat_composite_matches_a_hand_computed_pixel():
    weights = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    palette = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    rgb = preview.compose_splat_composite(weights, palette)
    assert tuple(int(v) for v in rgb[0, 0]) == (255, 0, 0)
    assert tuple(int(v) for v in rgb[0, 1]) == (0, 0, 255)
    assert tuple(int(v) for v in rgb[0, 2]) == (128, 0, 128)


def test_compose_splat_composite_cycles_the_palette_past_its_length():
    weights = torch.zeros((1, 1, 9))
    weights[0, 0, 8] = 1.0
    rgb = preview.compose_splat_composite(weights)
    expected = preview._to_uint8(np.asarray(preview.SPLAT_PALETTE[0], dtype=np.float32)[None])
    assert tuple(int(v) for v in rgb[0, 0]) == tuple(int(v) for v in expected[0])


def test_compose_splat_composite_rejects_a_wrong_shape():
    with pytest.raises(MapConfigError):
        preview.compose_splat_composite(torch.ones((4, 4)))


def test_render_splat_composite_writes_a_png(tmp_path):
    path = preview.render_splat_composite(weights3(16), tmp_path / "splat.png")
    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 16)


def test_render_splat_composite_defaults_to_the_map_out_dir():
    c = cfg(16)
    path = preview.render_splat_composite(weights3(16), cfg=c)
    assert path == c.out_dir() / preview.SPLAT_NAME
    assert path.exists()


def test_render_splat_composite_needs_a_path_or_cfg():
    with pytest.raises(MapConfigError):
        preview.render_splat_composite(weights3(16))


def test_splat_composite_preview_is_within_budget(tmp_path):
    result = preview.splat_composite_preview(weights3(64), tmp_path / "splat.png")
    assert result.path.exists()
    assert result.estimated_tokens <= preview.DEFAULT_BUDGET.max_tokens


def test_coverage_table_lists_every_layer_and_percentages():
    coverage = (
        {"layer": "silt_flat", "material": "silt", "mean": 0.625, "dominant": 0.7},
        {"layer": "exposed_rock", "material": "cliff_rock", "mean": 0.375, "dominant": 0.3},
    )
    table = preview.coverage_table(coverage)
    lines = table.splitlines()
    assert lines[0].split() == ["layer", "material", "mean", "%", "dominant", "%"]
    assert "silt_flat" in lines[1]
    assert "62.50" in lines[1]
    assert "70.00" in lines[1]
    assert "exposed_rock" in lines[2]


def test_coverage_table_rejects_an_empty_table():
    with pytest.raises(MapConfigError):
        preview.coverage_table(())
