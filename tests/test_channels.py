from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from terrain import channels, erosion, synth
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError
from terrain.device import resolve as resolve_device

CPU = "cpu"


def cfg(resolution: int = 64, **kwargs) -> MapConfig:
    base = {"name": "unit", "resolution": resolution, "world_size_m": 1024.0, "seed": 7}
    base.update(kwargs)
    return MapConfig(**base)


def eroded(c: MapConfig) -> erosion.ErosionResult:
    height = synth.heightfield(c, "continental", CPU)
    params = erosion.ErosionParams(iterations=20, settle_iterations=5)
    return erosion.erode(c, height, params, device=CPU)


def ramp(c: MapConfig, rise: float = 1.0) -> torch.Tensor:
    axis = torch.linspace(0.0, rise, c.resolution, device=CPU, dtype=torch.float32)
    return axis.reshape(-1, 1).expand(c.shape).contiguous()


def test_channel_index_covers_every_name():
    assert tuple(channels.CHANNEL_INDEX) == CHANNEL_NAMES
    assert set(channels.GEOMETRIC_CHANNELS) <= set(CHANNEL_NAMES)
    assert set(channels.EROSION_DERIVED_CHANNELS) <= set(CHANNEL_NAMES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"curvature_radius_m": 0.0},
        {"curvature_gain": 0.0},
        {"soil_depth_m": -1.0},
        {"soil_smooth_m": 0.0},
        {"soil_slope_deg": 90.0},
        {"strata_thickness_m": 0.0},
        {"strata_dip_deg": 90.0},
        {"strata_warp_m": -1.0},
        {"strata_warp_feature_size_m": 0.0},
        {"strata_edge": 0.0},
        {"wetness_min_tilt": 0.0},
    ],
)
def test_geometry_params_reject_impossible_values(kwargs):
    with pytest.raises(MapConfigError):
        channels.GeometryParams(**kwargs)


def test_smooth_preserves_a_constant_field():
    field = torch.full((48, 48), 0.25)
    assert torch.allclose(channels.smooth(field, 6.0), field, atol=1e-5)


def test_smooth_reduces_variance_and_does_not_mutate():
    field = torch.rand((64, 64), generator=torch.Generator().manual_seed(3))
    original = field.clone()
    blurred = channels.smooth(field, 4.0)
    assert torch.equal(field, original)
    assert float(blurred.var()) < float(field.var())
    assert float(blurred.mean()) == pytest.approx(float(field.mean()), abs=0.02)


def test_smooth_with_zero_sigma_copies():
    field = torch.rand((16, 16))
    result = channels.smooth(field, 0.0)
    assert result is not field
    assert torch.equal(result, field)


def test_gradient_matches_the_analytic_ramp():
    c = cfg(64, height_range_m=100.0)
    grad_x, grad_y = channels.gradient(c, ramp(c))
    expected = 100.0 / ((c.resolution - 1) * c.metres_per_pixel)
    interior = grad_y[1:-1, 1:-1]
    assert float(grad_x.abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert float(interior.mean()) == pytest.approx(expected, rel=1e-4)


def test_slope_channel_is_the_angle_over_ninety_degrees():
    c = cfg(64, height_range_m=100.0)
    height = ramp(c)
    angle = channels.slope_angle_deg(c, height)
    field = channels.slope(c, height)
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0
    assert torch.allclose(field, angle / 90.0, atol=1e-5)


def test_slope_of_flat_ground_is_zero():
    c = cfg(32)
    assert float(channels.slope(c, torch.full(c.shape, 0.4)).max()) == pytest.approx(0.0)


def test_curvature_separates_hollows_from_ridges():
    c = cfg(128, height_range_m=200.0)
    axis = torch.linspace(-1.0, 1.0, c.resolution)
    bowl = (axis.reshape(-1, 1) ** 2 + axis.reshape(1, -1) ** 2).div(2.0)
    params = channels.GeometryParams(curvature_radius_m=8.0)
    hollow = channels.curvature_per_m(c, bowl, params)
    ridge = channels.curvature_per_m(c, 1.0 - bowl, params)
    assert float(hollow[32:96, 32:96].mean()) > 0.0
    assert float(ridge[32:96, 32:96].mean()) < 0.0
    field = channels.curvature(c, bowl, params)
    assert 0.0 <= float(field.min()) and float(field.max()) <= 1.0


def test_curvature_of_a_plane_sits_at_the_midpoint():
    c = cfg(64)
    field = channels.curvature(c, ramp(c))
    assert float(field.mean()) == pytest.approx(0.5, abs=1e-3)


def test_bedrock_sits_under_the_height_and_strips_on_steep_ground():
    c = cfg(128, height_range_m=300.0)
    height = synth.heightfield(c, "highland", CPU)
    params = channels.GeometryParams(soil_depth_m=12.0)
    rock = channels.bedrock(c, height, params)
    depth = channels.soil_depth_m(c, height, params)
    assert float((rock - height).max()) <= 1e-6
    assert float(depth.max()) <= params.soil_depth_m + 1e-5
    steep = channels.slope_angle_deg(c, height) > params.soil_slope_deg
    if bool(steep.any()):
        assert float(depth[steep].mean()) < float(depth[~steep].mean())


def test_bedrock_accepts_an_explicit_soil_field():
    c = cfg(32, height_range_m=100.0)
    height = torch.full(c.shape, 0.5)
    soil = torch.full(c.shape, 10.0)
    rock = channels.bedrock(c, height, soil_m=soil)
    assert float(rock.mean()) == pytest.approx(0.4, abs=1e-6)
    assert float(soil.mean()) == pytest.approx(10.0)


def test_strata_is_banded_and_deterministic():
    c = cfg(128)
    height = synth.heightfield(c, "continental", CPU)
    first = channels.strata(c, height, device=CPU)
    second = channels.strata(c, height, device=CPU)
    assert torch.equal(first, second)
    assert 0.0 <= float(first.min()) and float(first.max()) <= 1.0
    assert float(first.std()) > 0.05
    thick = channels.strata(c, height, channels.GeometryParams(strata_thickness_m=400.0), CPU)
    assert float(thick.std()) < float(first.std())


def test_strata_follows_the_dip_plane():
    c = cfg(64, height_range_m=1.0)
    flat = torch.zeros(c.shape)
    params = channels.GeometryParams(
        strata_dip_deg=45.0, strata_strike_deg=0.0, strata_warp_m=0.0, strata_thickness_m=64.0
    )
    field = channels.strata(c, flat, params, CPU)
    assert float(field.std(dim=0).max()) == pytest.approx(0.0, abs=1e-6)
    assert float(field.std(dim=1).mean()) > 0.05


def test_strata_seed_changes_the_bands():
    c = cfg(64)
    flat = torch.zeros(c.shape)
    other = cfg(64, seed=8)
    assert not torch.equal(channels.strata(c, flat, device=CPU), channels.strata(other, flat, device=CPU))


def test_wetness_rises_with_flow_and_falls_with_slope():
    c = cfg(64, height_range_m=100.0)
    flow = torch.linspace(0.0, 1e5, c.resolution).reshape(1, -1).expand(c.shape).contiguous()
    field = channels.wetness(c, flow, torch.full(c.shape, 0.5))
    assert 0.0 <= float(field.min()) and float(field.max()) <= 1.0
    assert float(field[:, -1].mean()) > float(field[:, 0].mean())
    steep = channels.wetness(c, flow, ramp(c, 1.0))
    assert float(steep.sum()) < float(field.sum())


def test_wetness_rejects_a_zero_tilt_floor():
    c = cfg(32)
    with pytest.raises(MapConfigError):
        channels.wetness(c, torch.zeros(c.shape), torch.zeros(c.shape), min_tilt=0.0)


def test_geometric_channels_are_named_and_normalised():
    c = cfg(128)
    height = synth.heightfield(c, "coastal_range", CPU)
    fields = channels.geometric_channels(c, height, device=CPU)
    assert tuple(fields) == channels.GEOMETRIC_CHANNELS
    for name, field in fields.items():
        assert field.shape == c.shape, name
        assert float(field.min()) >= 0.0, name
        assert float(field.max()) <= 1.0, name


def test_geometric_channels_reject_a_mismatched_height():
    c = cfg(32)
    with pytest.raises(MapConfigError):
        channels.geometric_channels(c, torch.zeros((16, 16)), device=CPU)


def test_erosion_channels_add_wetness():
    c = cfg()
    result = eroded(c)
    fields = channels.erosion_channels(c, result)
    assert set(channels.EROSION_DERIVED_CHANNELS) <= set(fields)
    for name, field in fields.items():
        assert float(field.min()) >= 0.0, name
        assert float(field.max()) <= 1.0, name


def test_stack_holds_named_views_of_one_buffer():
    c = cfg(32)
    stack = channels.ChannelStack(c, CPU)
    assert len(stack) == len(CHANNEL_NAMES)
    assert stack.data.shape == (c.resolution, c.resolution, len(CHANNEL_NAMES))
    assert stack.nbytes == c.channel_stack_bytes()
    assert stack.filled() == ()
    assert stack.missing() == CHANNEL_NAMES
    stack["slope"] = torch.full(c.shape, 0.25)
    assert "slope" in stack
    assert float(stack["slope"].mean()) == pytest.approx(0.25)
    assert stack.filled() == ("slope",)
    assert "slope" not in stack.missing()
    assert tuple(stack.as_dict()) == ("slope",)


def test_stack_rejects_unknown_shapes_and_ranges():
    c = cfg(32)
    stack = channels.ChannelStack(c, CPU)
    with pytest.raises(MapConfigError):
        stack["gravel"] = torch.zeros(c.shape)
    with pytest.raises(MapConfigError):
        stack["slope"] = torch.zeros((8, 8))
    with pytest.raises(MapConfigError):
        stack["slope"] = torch.full(c.shape, 1.5)
    with pytest.raises(MapConfigError):
        channels.ChannelStack(c, CPU, ("slope", "slope"))
    with pytest.raises(MapConfigError):
        channels.ChannelStack(c, CPU, ("slope", "gravel"))


def test_stack_can_hold_a_subset_of_channels():
    c = cfg(32)
    stack = channels.ChannelStack(c, CPU, channels.GEOMETRIC_CHANNELS)
    assert stack.data.shape[2] == len(channels.GEOMETRIC_CHANNELS)
    assert stack.nbytes == c.channel_stack_bytes(len(channels.GEOMETRIC_CHANNELS))


def test_build_fills_every_channel():
    c = cfg()
    stack = channels.build(c, eroded(c), device=CPU)
    filled = set(stack.filled())
    for group in (
        channels.GEOMETRIC_CHANNELS,
        channels.EROSION_DERIVED_CHANNELS,
        channels.WATER_CHANNELS,
        channels.CLIMATE_CHANNELS,
        channels.PATCHINESS_CHANNELS,
    ):
        assert set(group) <= filled
    assert "height" in filled
    assert stack.missing() == ()
    assert float(stack.data.min()) >= 0.0
    assert float(stack.data.max()) <= 1.0


def test_build_is_deterministic():
    c = cfg()
    result = eroded(c)
    first = channels.build(c, result, device=CPU)
    second = channels.build(c, result, device=CPU)
    assert torch.equal(first.data, second.data)


@pytest.mark.gpu
def test_channels_follow_the_resolved_device():
    device = resolve_device()
    if not device.is_accelerator:
        pytest.skip("no accelerator available")
    c = cfg(64)
    height = synth.heightfield(c, "continental", device)
    result = erosion.erode(c, height, erosion.ErosionParams(iterations=10), device=device)
    stack = channels.build(c, result, device=device)
    assert stack.data.device.type == device.backend
    reference = channels.geometric_channels(c, result.height.cpu(), device=CPU)
    assert torch.allclose(stack["slope"].cpu(), reference["slope"], atol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sea_level_quantile": 1.0},
        {"fill_passes": -1},
        {"flow_exponent": 0.0},
        {"convergence_stride": 0},
        {"river_area_fraction": 0.001},
        {"river_core_fraction": 0.0},
        {"river_depth_m": -1.0},
        {"river_depth_exponent": 0.0},
        {"river_width_m": -1.0},
        {"shore_depth_m": -1.0},
        {"full_depth_m": 0.0},
        {"depth_reference_m": 0.0},
        {"surface_smooth_m": -1.0},
    ],
)
def test_water_params_reject_impossible_values(kwargs):
    with pytest.raises(MapConfigError):
        channels.WaterParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature_feature_size_m": 0.0},
        {"moisture_feature_size_m": 0.0},
        {"patchiness_fine_m": 0.0},
        {"patchiness_mid_m": 0.0},
        {"patchiness_coarse_m": 0.0},
        {"temperature_max_c": -30.0},
        {"temperature_noise_c": -1.0},
        {"moisture_warp_m": -1.0},
        {"moisture_octaves": 0},
        {"moisture_wetness_smooth_m": -1.0},
        {"moisture_wetness_weight": 1.5},
        {"patchiness_octaves": 0},
        {"patchiness_cell_weight": 1.5},
        {"patchiness_warp_ratio": -1.0},
        {"patchiness_warp_octaves": 0},
    ],
)
def test_climate_params_reject_impossible_values(kwargs):
    with pytest.raises(MapConfigError):
        channels.ClimateParams(**kwargs)


def test_sea_level_follows_the_config_then_the_override():
    c = cfg(32, sea_level_m=40.0)
    height = ramp(c)
    assert channels.sea_level_m(c, height) == 40.0
    assert channels.sea_level_m(c, height, channels.WaterParams(sea_level_m=90.0)) == 90.0
    half = channels.WaterParams(sea_level_quantile=0.5)
    assert channels.sea_level_m(c, height, half) == pytest.approx(0.5 * c.height_range_m, abs=20.0)


def bowl(c: MapConfig, floor: float = 0.2, rim: float = 0.8) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, c.resolution, device=CPU, dtype=torch.float32)
    radius = torch.hypot(axis.reshape(-1, 1), axis.reshape(1, -1)).clamp_(0.0, 1.0)
    return radius.mul_(rim - floor).add_(floor)


def moat(c: MapConfig, pit: float = 0.1, wall: float = 0.8, outer: float = 0.3) -> torch.Tensor:
    height = torch.full(c.shape, outer, device=CPU, dtype=torch.float32)
    n = c.resolution
    a, b = n // 4, 3 * n // 4
    height[a:b, a:b] = wall
    p, q = n // 3, 2 * n // 3
    height[p:q, p:q] = pit
    return height


def nested_moat(c: MapConfig) -> torch.Tensor:
    """Two closed basins in series: a plaza behind one wall, a pit behind a second, taller one.

    The inner wall is taller than the outer one, so fill_depressions gives the pit and the plaza
    two different flood levels - two separate groups for _lake_escape to consider. One round can
    only breach the outer wall, since the pit's own escape search can walk out through the (still
    dry) inner wall but not through the plaza beyond it, which is still classified flooded; only
    once that round's breach lets fill_depressions reclassify most of the plaza as draining on its
    own does a further round's search reach past it to the true low ground outside.
    """
    height = torch.full(c.shape, 0.15, device=CPU, dtype=torch.float32)
    height[8:56, 8:56] = 0.7
    height[14:50, 14:50] = 0.45
    height[22:42, 22:42] = 0.75
    height[26:38, 26:38] = 0.1
    return height


def test_breach_depressions_lowers_the_wall_toward_the_plain_it_blocks():
    c = cfg(64)
    height = moat(c)
    breached = channels.breach_depressions(c, height, smooth_m=16.0)
    n = c.resolution
    a, b = n // 4, 3 * n // 4
    p, q = n // 3, 2 * n // 3
    wall = torch.zeros(c.shape, dtype=torch.bool)
    wall[a:b, a:b] = True
    wall[p:q, p:q] = False
    assert float(breached[wall].mean()) < float(height[wall].mean())
    assert float(breached[wall].max()) < 0.8


def test_breach_depressions_never_raises_the_terrain():
    c = cfg(64)
    height = moat(c)
    breached = channels.breach_depressions(c, height)
    assert bool((breached <= height + 1e-6).all())


def test_breach_depressions_leaves_a_draining_slope_untouched():
    c = cfg(64)
    height = ramp(c)
    breached = channels.breach_depressions(c, height)
    assert torch.equal(breached, height)


def test_breach_depressions_leaves_a_true_endorheic_basin_alone():
    c = cfg(64)
    height = bowl(c)
    breached = channels.breach_depressions(c, height)
    assert torch.allclose(breached, height, atol=1e-5)


def test_breach_depressions_shrinks_what_fill_depressions_still_needs_to_flood():
    c = cfg(64)
    height = moat(c)
    filled_before = channels.fill_depressions(c, height)
    breached = channels.breach_depressions(c, height)
    filled_after = channels.fill_depressions(c, breached)
    assert float((filled_before - height).max()) > 0.3
    assert float((filled_after - breached).max()) < float((filled_before - height).max())


def test_breach_depressions_is_deterministic():
    c = cfg(64)
    height = moat(c)
    first = channels.breach_depressions(c, height)
    second = channels.breach_depressions(c, height)
    assert torch.equal(first, second)


def test_breach_depressions_zero_rounds_is_a_no_op():
    c = cfg(64)
    height = moat(c)
    assert torch.equal(channels.breach_depressions(c, height, rounds=0), height)


def test_breach_depressions_more_rounds_carves_deeper():
    c = cfg(64)
    height = nested_moat(c)
    shallow = channels.breach_depressions(c, height, rounds=1)
    deeper = channels.breach_depressions(c, height, rounds=3)
    assert bool((deeper <= shallow + 1e-6).all())
    assert float((shallow - deeper).max()) > 0.0


def test_fill_depressions_raises_a_pit_to_its_rim():
    c = cfg(64)
    height = bowl(c)
    filled = channels.fill_depressions(c, height)
    lifted = filled - height
    assert float(lifted.min()) >= 0.0
    assert float(lifted[c.resolution // 2, c.resolution // 2]) > 0.3
    assert float(lifted[0, 0]) == 0.0


def test_fill_depressions_leaves_a_draining_slope_untouched():
    c = cfg(64)
    height = ramp(c)
    filled = channels.fill_depressions(c, height)
    assert float((filled - height).max()) < 1e-3


def test_flow_accumulation_grows_downhill():
    c = cfg(64)
    height = ramp(c)
    accumulated = channels.flow_accumulation(c, channels.fill_depressions(c, height))
    rows = accumulated.mean(dim=1)
    assert float(accumulated.min()) >= 1.0
    assert float(rows[1]) > 4.0 * float(rows[-2])


def test_fill_depressions_leaves_one_exact_lake_surface_not_an_epsilon_ladder():
    c = cfg(64)
    height = bowl(c)
    filled = channels.fill_depressions(c, height)
    surface = filled[filled > height]
    assert surface.numel() > 0
    assert float(surface.max()) - float(surface.min()) == 0.0


def test_fill_depressions_matches_the_single_level_solve_the_pyramid_replaces():
    c = cfg(64)
    height = bowl(c)
    params = channels.WaterParams()
    level = c.sea_level_m / c.height_range_m
    flat = channels._settle_fill(
        height,
        level,
        torch.full_like(height, channels._FILL_CEILING),
        2 * c.resolution,
        params.convergence_stride,
    )
    assert torch.equal(channels.fill_depressions(c, height, params=params), flat)


def plateau(c: MapConfig, rise: float = 1.0) -> torch.Tensor:
    height = ramp(c, rise).clone()
    low, high = c.resolution // 4, 3 * c.resolution // 4
    height[low:high, low:high] = float(height[high - 1, low])
    return height


def test_flat_gradient_is_empty_where_every_cell_already_drains():
    c = cfg(64)
    filled = channels.fill_depressions(c, ramp(c))
    assert float(channels.flat_gradient(c, filled).max()) == 0.0


def test_flat_gradient_leaves_no_flat_cell_without_a_lower_neighbour():
    c = cfg(64)
    filled = channels.fill_depressions(c, plateau(c))
    guide = channels.flat_gradient(c, filled)
    flat = guide > 0.0
    assert float(flat.float().mean()) > 0.1
    neighbours = channels._neighbourhood(filled)
    across = guide.sub(channels._neighbourhood(guide)).clamp_(min=0.0).mul_(neighbours <= filled)
    assert not bool((flat & (across.sum(0) <= 0.0)).any())


def test_flat_gradient_steps_stay_far_above_the_float_resolution_an_epsilon_fill_had():
    c = cfg(64)
    filled = channels.fill_depressions(c, plateau(c))
    guide = channels.flat_gradient(c, filled)
    flat = guide > 0.0
    drops = channels._neighbourhood(guide).sub_(guide).abs_()
    steps = drops[(drops > 0.0) & channels._neighbourhood(flat) & flat]
    assert float(steps.min()) >= 1.0


def test_flow_accumulation_fans_across_a_flat_rather_than_following_one_chain():
    c = cfg(64)
    filled = channels.fill_depressions(c, plateau(c))
    guide = channels.flat_gradient(c, filled)
    flat = guide > 0.0
    drops = guide.sub(channels._neighbourhood(guide)).clamp_(min=0.0)
    receivers = (drops > 0.0).sum(0)
    assert float((receivers[flat] > 1).float().mean()) > 0.5


def test_flow_accumulation_carries_a_flat_plateau_off_its_rim():
    c = cfg(64)
    height = plateau(c)
    filled = channels.fill_depressions(c, height)
    accumulated = channels.flow_accumulation(c, filled)
    guide = channels.flat_gradient(c, filled)
    flat = guide > 0.0
    assert float(accumulated[flat].mean()) > 2.0
    assert float(accumulated[1].mean()) > float(accumulated[flat].mean())


def test_water_fills_a_bowl_to_its_rim():
    c = cfg(64)
    height = bowl(c)
    depth = channels.water_depth_metres(c, height)
    mask = channels.water_mask(depth)
    assert float(mask[c.resolution // 2, c.resolution // 2]) == pytest.approx(1.0)
    assert float(mask[0, 0]) == 0.0
    assert float(depth.max()) == pytest.approx(0.6 * c.height_range_m, rel=0.1)


def test_water_covers_everything_below_the_sea_level():
    c = cfg(64, sea_level_m=300.0)
    height = ramp(c)
    channelled = channels.water_channels(c, height)
    submerged = height < c.sea_level_normalised - 0.02
    assert float(channelled["water"][submerged].min()) == pytest.approx(1.0)
    assert float(channelled["water"][-1, -1]) == 0.0
    assert float(channelled["water_depth"].max()) == pytest.approx(1.0)


def test_water_summary_records_the_level_it_used():
    c = cfg(64, sea_level_m=120.0)
    level = channels.summarise_water(c, eroded(c))
    assert level.sea_level_m == 120.0
    assert 0.0 <= level.covered_fraction <= 1.0
    assert level.max_depth_m >= level.mean_depth_m
    assert set(level.as_dict()) == {
        "sea_level_m",
        "covered_fraction",
        "mean_depth_m",
        "max_depth_m",
        "depth_reference_m",
    }


def test_temperature_falls_with_altitude_and_latitude():
    c = cfg(64)
    params = channels.ClimateParams(temperature_noise_c=0.0, latitude_gradient_c=0.0)
    flat = torch.zeros(c.shape, device=CPU)
    high = torch.ones(c.shape, device=CPU)
    lapse = float(channels.temperature_c(c, flat, params).mean()) - float(
        channels.temperature_c(c, high, params).mean()
    )
    assert lapse == pytest.approx(params.lapse_rate_c_per_km * c.height_range_m / 1000.0, rel=1e-3)
    gradient = channels.ClimateParams(temperature_noise_c=0.0)
    field = channels.temperature_c(c, flat, gradient)
    assert float(field[0].mean()) > float(field[-1].mean())


def test_temperature_channel_stays_inside_its_stated_range():
    c = cfg(64)
    params = channels.ClimateParams(temperature_min_c=0.0, temperature_max_c=10.0)
    field = channels.temperature(c, ramp(c), params)
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0


def test_moisture_without_wetness_is_pure_noise():
    c = cfg(64)
    plain = channels.moisture(c, device=CPU)
    assert float(plain.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(plain.max()) == pytest.approx(1.0, abs=1e-5)
    wet = ramp(c)
    biased = channels.moisture(c, wet, device=CPU)
    assert float(biased[-1].mean()) - float(plain[-1].mean()) > 0.05
    assert float(biased[0].mean()) < float(plain[0].mean()) + 0.05


def test_patchiness_scales_differ_and_stay_in_range():
    c = cfg(64)
    fields = channels.patchiness_channels(c, device=CPU)
    assert set(fields) == set(channels.PATCHINESS_CHANNELS)
    for field in fields.values():
        assert float(field.min()) >= 0.0
        assert float(field.max()) <= 1.0
    coarse = fields["patchiness_coarse"]
    fine = fields["patchiness_fine"]
    assert not torch.allclose(coarse, fine)
    assert float(channels.smooth(coarse, 2.0).std()) > float(channels.smooth(fine, 2.0).std())


def test_patchiness_warp_octaves_change_the_field_but_stay_in_range():
    c = cfg(64)
    low = channels.patchiness(c, 110.0, "mid", channels.ClimateParams(patchiness_warp_octaves=1), CPU)
    high = channels.patchiness(c, 110.0, "mid", channels.ClimateParams(patchiness_warp_octaves=4), CPU)
    assert float(low.min()) >= 0.0
    assert float(low.max()) <= 1.0
    assert not torch.allclose(low, high)


def test_climate_channels_reject_a_mismatched_height():
    c = cfg(32)
    with pytest.raises(MapConfigError):
        channels.climate_channels(c, torch.zeros((8, 8), device=CPU))


def test_drainage_shares_one_fill_and_accumulation():
    c = cfg(64)
    height = bowl(c)
    basin = channels.drainage(c, height)
    assert torch.equal(basin.filled, channels.fill_depressions(c, height, basin.level_m))
    assert torch.equal(
        basin.accumulated,
        channels.flow_accumulation(c, basin.filled, level_m=basin.level_m),
    )
    assert torch.equal(
        channels.water_depth_metres(c, height, basin=basin),
        channels.water_depth_metres(c, height),
    )


def test_flow_channel_comes_from_drainage_not_the_sim():
    c = cfg(64)
    result = eroded(c)
    fields = channels.erosion_channels(c, result)
    assert "flow" not in result.channels()
    assert torch.equal(fields["flow"], channels.drainage(c, result.height).flow)


def contrast(field: torch.Tensor) -> float:
    return float(field.max()) / max(float(field.mean()), 1e-8)


def test_flow_channel_carries_the_dynamic_range_the_sim_flow_volume_lost():
    c = cfg(128)
    result = eroded(c)
    basin = channels.drainage(c, result.height)
    assert contrast(result.flow_volume_m3) < 10.0
    assert contrast(basin.accumulated) > 100.0
    flow = channels.erosion_channels(c, result, basin=basin)["flow"]
    assert float(flow.quantile(0.5)) < 0.5 * float(flow.quantile(0.999))


def pipeline(c: MapConfig, shape: str = "continental") -> channels.ChannelStack:
    height = synth.heightfield(c, shape, CPU)
    params = erosion.ErosionParams(iterations=20, settle_iterations=5)
    return channels.build(c, erosion.erode(c, height, params, device=CPU), device=CPU)


def test_pipeline_reproduces_the_whole_stack_from_a_seed():
    c = cfg(64)
    assert torch.equal(pipeline(c).data, pipeline(c).data)


def test_pipeline_diverges_on_a_different_seed():
    first = pipeline(cfg(64, seed=7))
    second = pipeline(cfg(64, seed=8))
    assert not torch.equal(first.data, second.data)
    assert first.filled() == second.filled()


def test_every_stack_channel_is_finite_and_normalised():
    stack = pipeline(cfg(64))
    for name in stack.filled():
        field = stack[name]
        assert torch.isfinite(field).all(), name
        assert field.shape == stack.cfg.shape, name
        assert 0.0 <= float(field.min()), name
        assert float(field.max()) <= 1.0, name
