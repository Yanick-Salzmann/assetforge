from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from terrain import channels as channel_module
from terrain import scatter
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 8


def cfg(**kwargs) -> MapConfig:
    base = {"name": "scatter-unit", "resolution": SIZE, "world_size_m": 64.0, "seed": 5}
    base.update(kwargs)
    return MapConfig(**base)


def quadrants(**values: float) -> channel_module.ChannelStack:
    """An 8x8 stack with four 4x4 quadrants: open_water, channel_bed, steep, safe."""
    built = channel_module.ChannelStack(cfg(), CPU)
    for name in CHANNEL_NAMES:
        built[name] = torch.zeros(built.cfg.shape, device=CPU)
    for name, value in values.items():
        field = torch.zeros(built.cfg.shape, device=CPU)
        if name == "water":
            field[:4, :4] = value
        elif name == "flow":
            field[:4, 4:] = value
        elif name == "slope":
            field[4:, :4] = value
        else:
            field[4:, 4:] = value
        built[name] = field
    return built


def safe_stack(**extra: float) -> channel_module.ChannelStack:
    built = quadrants(water=1.0, flow=1.0, slope=1.0, wear=1.0)
    for name, value in extra.items():
        field = torch.zeros(built.cfg.shape, device=CPU)
        field[4:, 4:] = value
        built[name] = field
    return built


def test_avoid_zeroes_out_water_channel_bed_and_steep_rock():
    built = quadrants(water=1.0, flow=1.0, slope=1.0)
    safe = scatter._avoid(built, scatter.ScatterParams())
    assert float(safe[0, 0]) == pytest.approx(0.0, abs=1e-4)
    assert float(safe[0, 4]) == pytest.approx(0.0, abs=1e-4)
    assert float(safe[4, 0]) == pytest.approx(0.0, abs=1e-4)
    assert float(safe[4, 4]) == pytest.approx(1.0, abs=1e-4)


def test_density_rock_lights_up_only_on_wear_in_the_safe_quadrant():
    built = safe_stack()
    density = scatter.density_rock(built)
    assert float(density[4, 4]) == pytest.approx(1.0, abs=1e-3)
    assert float(density[0, 0]) == pytest.approx(0.0, abs=1e-3)
    assert float(density[0, 4]) == pytest.approx(0.0, abs=1e-3)
    assert float(density[4, 0]) == pytest.approx(0.0, abs=1e-3)


def test_density_rock_also_lights_up_on_bare_bedrock():
    built = quadrants(water=1.0, flow=1.0, slope=1.0)
    field = torch.zeros(built.cfg.shape, device=CPU)
    field[4:, 4:] = 1.0
    built["bedrock"] = field
    density = scatter.density_rock(built)
    assert float(density[4, 4]) == pytest.approx(1.0, abs=1e-3)


def test_density_tree_needs_moisture_and_gentle_slope():
    built = safe_stack(moisture=1.0)
    density = scatter.density_tree(built, scatter.ScatterParams(tree_water_buffer_m=0.0))
    assert float(density[4, 4]) == pytest.approx(1.0, abs=1e-3)
    assert float(density[0, 0]) == pytest.approx(0.0, abs=1e-3)


def test_density_tree_respects_the_water_buffer():
    built = channel_module.ChannelStack(cfg(world_size_m=8.0), CPU)
    for name in CHANNEL_NAMES:
        built[name] = torch.zeros(built.cfg.shape, device=CPU)
    water = torch.zeros(built.cfg.shape, device=CPU)
    water[0, 0] = 1.0
    built["water"] = water
    built["moisture"] = torch.ones(built.cfg.shape, device=CPU)
    params = scatter.ScatterParams(tree_water_buffer_m=3.0)
    density = scatter.density_tree(built, params)
    assert float(density[0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(density[0, 7]) > 0.0


def test_density_grass_uses_the_provided_weight_and_thins_on_slope():
    built = quadrants(slope=1.0)
    weight = torch.full(built.cfg.shape, 0.8, device=CPU)
    flat = scatter.density_grass(built, weight)
    assert float(flat[0, 0]) == pytest.approx(0.8, abs=1e-3)
    assert float(flat[4, 0]) == pytest.approx(0.0, abs=1e-3)


def test_density_debris_responds_to_deposition_and_flow_edges():
    built = quadrants(water=1.0, flow=1.0, slope=1.0)
    field = torch.zeros(built.cfg.shape, device=CPU)
    field[4:, 4:] = 1.0
    built["deposition"] = field
    density = scatter.density_debris(built)
    assert float(density[4, 4]) == pytest.approx(1.0, abs=1e-3)
    assert float(density[0, 0]) == pytest.approx(0.0, abs=1e-3)


def test_density_debris_picks_up_a_flow_discontinuity():
    built = channel_module.ChannelStack(cfg(), CPU)
    for name in CHANNEL_NAMES:
        built[name] = torch.zeros(built.cfg.shape, device=CPU)
    flow = torch.zeros(built.cfg.shape, device=CPU)
    flow[:, 4:] = 0.3
    built["flow"] = flow
    density = scatter.density_debris(built)
    assert float(density[0, 3]) > float(density[0, 0])


def test_density_tree_needs_a_stack_with_cfg():
    plain = {name: torch.zeros((SIZE, SIZE), device=CPU) for name in CHANNEL_NAMES}
    with pytest.raises(MapConfigError, match="carries cfg"):
        scatter.density_tree(plain)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_bed_low": 0.8, "channel_bed_high": 0.5},
        {"steep_low": 0.5, "steep_high": 0.5},
        {"tree_water_buffer_m": -1.0},
    ],
)
def test_scatter_params_rejects_bad_ranges(kwargs):
    with pytest.raises(MapConfigError):
        scatter.ScatterParams(**kwargs)


def test_quantise_rounds_to_the_nearest_step():
    density = torch.tensor([0.0, 0.5, 1.0, 1.5])
    quantised = scatter.quantise(density)
    assert quantised.dtype == torch.uint8
    assert torch.equal(quantised, torch.tensor([0, 128, 255, 255], dtype=torch.uint8))


def test_build_returns_all_four_masks():
    built = safe_stack(moisture=1.0)
    weight = torch.full(built.cfg.shape, 0.5, device=CPU)
    masks = scatter.build(built, weight)
    assert set(masks) == {"rock", "tree", "grass", "debris"}


def test_write_produces_grayscale_pngs_with_expected_names(tmp_path):
    built = safe_stack(moisture=1.0)
    weight = torch.full(built.cfg.shape, 0.5, device=CPU)
    masks = scatter.build(built, weight)
    written = scatter.write(masks, tmp_path)
    names = {path.name for path in written}
    assert names == {scatter.ROCK_NAME, scatter.TREE_NAME, scatter.GRASS_NAME, scatter.DEBRIS_NAME}
    for path in written:
        with Image.open(path) as image:
            assert image.mode == "L"
            assert image.size == (SIZE, SIZE)


def test_write_rejects_an_unknown_mask_name(tmp_path):
    with pytest.raises(MapConfigError):
        scatter.write({"lava": torch.zeros((SIZE, SIZE))}, tmp_path)


def test_coverage_reports_mean_and_threshold_fraction():
    masks = {"rock": torch.tensor([[0.0, 1.0], [1.0, 1.0]])}
    table = scatter.coverage(masks, threshold=0.5)
    assert table[0]["mask"] == "rock"
    assert table[0]["mean"] == pytest.approx(0.75)
    assert table[0]["coverage"] == pytest.approx(0.75)
