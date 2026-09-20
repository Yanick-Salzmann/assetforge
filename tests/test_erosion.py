from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from terrain import erosion, synth
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError
from terrain.device import resolve as resolve_device

CPU = "cpu"


def cfg(resolution: int = 64, **kwargs) -> MapConfig:
    base = {"name": "unit", "resolution": resolution, "world_size_m": 1024.0, "seed": 99}
    base.update(kwargs)
    return MapConfig(**base)


def quick(**kwargs) -> erosion.ErosionParams:
    base = {"iterations": 20, "settle_iterations": 5}
    base.update(kwargs)
    return erosion.ErosionParams(**base)


def ramp(c: MapConfig) -> torch.Tensor:
    axis = torch.linspace(1.0, 0.0, c.resolution, device=CPU, dtype=torch.float32)
    return axis.reshape(-1, 1).expand(c.shape).contiguous()


def test_presets_are_named_and_valid():
    assert "default" in erosion.erosion_preset_names()
    for name in erosion.erosion_preset_names():
        assert isinstance(erosion.erosion_preset(name), erosion.ErosionParams)
    with pytest.raises(MapConfigError):
        erosion.erosion_preset("swamp")


def test_resolve_params_accepts_name_instance_or_none():
    assert erosion.resolve_params() is erosion.EROSION_PRESETS[erosion.DEFAULT_PRESET]
    assert erosion.resolve_params("light") is erosion.EROSION_PRESETS["light"]
    params = quick()
    assert erosion.resolve_params(params) is params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"settle_iterations": -1},
        {"dt": 0.0},
        {"rain_rate": -1.0},
        {"evaporation_rate": 25.0},
        {"pipe_area": 0.0},
        {"gravity": -9.81},
        {"sediment_capacity": -1.0},
        {"min_tilt": 1.5},
        {"deep_water_depth": 0.0},
        {"min_velocity_depth": 0.0},
        {"max_courant": 0.0},
        {"max_courant": 2.0},
        {"max_height_step_m": 0.0},
    ],
)
def test_params_reject_unusable_values(kwargs):
    with pytest.raises(MapConfigError):
        erosion.ErosionParams(**kwargs)


def test_shape_mismatch_is_rejected():
    c = cfg()
    with pytest.raises(MapConfigError):
        erosion.erode(c, torch.zeros(8, 8), quick(), device=CPU)
    with pytest.raises(MapConfigError):
        erosion.erode(c, ramp(c), quick(), rain=torch.ones(8, 8), device=CPU)


def test_erosion_is_deterministic():
    c = cfg()
    height = synth.heightfield(c, "continental", CPU)
    first = erosion.erode(c, height, quick(), device=CPU)
    second = erosion.erode(c, height, quick(), device=CPU)
    assert torch.equal(first.height, second.height)
    assert torch.equal(first.erosion_m, second.erosion_m)
    assert torch.equal(first.flow_volume_m3, second.flow_volume_m3)


def test_input_height_is_not_mutated():
    c = cfg()
    height = synth.heightfield(c, "continental", CPU)
    before = height.clone()
    erosion.erode(c, height, quick(), device=CPU)
    assert torch.equal(height, before)


def test_result_fields_are_finite_and_bounded():
    c = cfg()
    result = erosion.erode(c, synth.heightfield(c, "continental", CPU), quick(), device=CPU)
    assert float(result.height.min()) >= 0.0
    assert float(result.height.max()) <= 1.0
    for field in (
        result.height,
        result.water_depth_m,
        result.sediment_m,
        result.speed_m_per_s,
        result.flux_m3_per_s,
        result.flow_volume_m3,
        result.erosion_m,
        result.deposition_m,
    ):
        assert torch.isfinite(field).all()
        assert float(field.min()) >= 0.0


def test_moved_material_is_conserved():
    c = cfg()
    result = erosion.erode(c, synth.heightfield(c, "continental", CPU), quick(), device=CPU)
    eroded = float(result.erosion_m.sum())
    balance = float(result.deposition_m.sum()) + float(result.sediment_m.sum())
    assert eroded > 0.0
    assert balance == pytest.approx(eroded, rel=1e-3)


def test_height_change_matches_the_accumulators():
    c = cfg()
    height = synth.heightfield(c, "continental", CPU)
    result = erosion.erode(c, height, quick(thermal_cadence=0), device=CPU)
    change = (result.height - height) * c.height_range_m
    expected = result.deposition_m - result.erosion_m
    assert torch.allclose(change, expected, atol=1e-2)


def test_closed_boundary_keeps_every_drop():
    c = cfg(32)
    params = quick(iterations=15, settle_iterations=0, evaporation_rate=0.0)
    result = erosion.erode(c, synth.heightfield(c, "continental", CPU), params, device=CPU)
    rained = params.iterations * params.rain_rate * params.dt * c.resolution**2
    assert float(result.water_depth_m.sum()) / c.height_range_m == pytest.approx(rained, rel=1e-3)


def test_flat_terrain_erodes_nothing():
    c = cfg(32)
    flat = torch.full(c.shape, 0.5, device=CPU, dtype=torch.float32)
    result = erosion.erode(c, flat, quick(), device=CPU)
    assert float(result.erosion_m.max()) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(result.height, flat, atol=1e-6)


def test_erosion_incises_a_slope():
    c = cfg(128)
    height = synth.heightfield(c, "highland", CPU)
    result = erosion.erode(c, height, quick(iterations=300, settle_iterations=30), device=CPU)
    ranked = result.erosion_m.flatten().sort(descending=True).values
    head = max(1, ranked.numel() // 20)
    assert float(ranked[:head].sum()) > 0.10 * float(ranked.sum())
    assert float(result.flow_volume_m3.max()) > 0.0


def test_rain_map_scales_the_water_supply():
    c = cfg(32)
    height = synth.heightfield(c, "continental", CPU)
    params = quick(settle_iterations=0, evaporation_rate=0.0)
    dry = erosion.erode(c, height, params, rain=torch.zeros(c.shape), device=CPU)
    half = erosion.erode(c, height, params, rain=torch.full(c.shape, 0.5), device=CPU)
    wet = erosion.erode(c, height, params, device=CPU)
    assert float(dry.water_depth_m.sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(half.water_depth_m.sum()) == pytest.approx(
        0.5 * float(wet.water_depth_m.sum()), rel=1e-3
    )


def test_channels_are_named_and_normalised():
    c = cfg()
    result = erosion.erode(c, synth.heightfield(c, "continental", CPU), quick(), device=CPU)
    channels = result.channels()
    assert set(channels) <= set(CHANNEL_NAMES)
    for name, field in channels.items():
        assert field.shape == c.shape, name
        assert float(field.min()) >= 0.0, name
        assert float(field.max()) <= 1.0, name


def test_stats_report_metres():
    c = cfg()
    result = erosion.erode(c, synth.heightfield(c, "continental", CPU), quick(), device=CPU)
    stats = result.stats()
    assert stats["iterations"] == float(result.params.iterations)
    assert stats["eroded_max_m"] >= stats["eroded_mean_m"] > 0.0
    assert stats["net_change_m"] == pytest.approx(
        stats["deposited_mean_m"] - stats["eroded_mean_m"]
    )


@pytest.mark.gpu
def test_erosion_runs_on_the_resolved_device():
    device = resolve_device()
    if not device.is_accelerator:
        pytest.skip("no accelerator available")
    c = cfg(128)
    height = synth.heightfield(c, "continental", device)
    result = erosion.erode(c, height, quick(), device=device)
    assert result.height.device.type == device.backend
    reference = erosion.erode(c, height.cpu(), quick(), device=CPU)
    assert torch.allclose(result.height.cpu(), reference.height, atol=1e-3)


@pytest.mark.gpu
def test_accelerator_and_cpu_agree_on_every_erosion_field():
    device = resolve_device()
    if not device.is_accelerator:
        pytest.skip("no accelerator available")
    c = cfg(64)
    height = synth.heightfield(c, "continental", device)
    accelerated = erosion.erode(c, height, quick(), device=device)
    reference = erosion.erode(c, height.cpu(), quick(), device=CPU)
    for name in (
        "height",
        "water_depth_m",
        "sediment_m",
        "flow_volume_m3",
        "erosion_m",
        "deposition_m",
        "talus_m",
    ):
        moved = getattr(accelerated, name).cpu()
        expected = getattr(reference, name)
        tolerance = 1e-3 * max(1.0, float(expected.abs().max()))
        assert torch.allclose(moved, expected, atol=tolerance), name


def steep_step(c: MapConfig) -> torch.Tensor:
    field = torch.zeros(c.shape, device=CPU, dtype=torch.float32)
    field[: c.resolution // 2, :] = 1.0
    return field


def test_thermal_params_reject_impossible_values():
    for kwargs in (
        {"talus_angle_deg": 0.0},
        {"talus_angle_deg": 90.0},
        {"thermal_rate": 0.0},
        {"thermal_rate": 1.5},
        {"thermal_cadence": -1},
        {"thermal_settle_passes": -1},
    ):
        with pytest.raises(MapConfigError):
            erosion.ErosionParams(**kwargs)


def test_thermal_relaxes_slopes_below_the_repose_angle():
    c = cfg(64, height_range_m=100.0)
    height = steep_step(c)
    relaxed, shed = erosion.thermal(c, height, angle_deg=34.0, iterations=400, device=CPU)
    limit = c.talus_delta(34.0)
    assert float(drops(relaxed).max()) <= limit * 1.05
    assert float(shed.max()) > 0.0


def test_thermal_conserves_mass_and_leaves_gentle_ground_alone():
    c = cfg(64, height_range_m=100.0)
    height = steep_step(c)
    relaxed, _ = erosion.thermal(c, height, iterations=50, device=CPU)
    assert float(relaxed.sum()) == pytest.approx(float(height.sum()), rel=1e-5)
    gentle = torch.full(c.shape, 0.5, device=CPU)
    settled, shed = erosion.thermal(c, gentle, iterations=10, device=CPU)
    assert torch.allclose(settled, gentle)
    assert float(shed.max()) == pytest.approx(0.0)


def test_thermal_does_not_mutate_the_input():
    c = cfg(32, height_range_m=100.0)
    height = steep_step(c)
    original = height.clone()
    erosion.thermal(c, height, iterations=20, device=CPU)
    assert torch.equal(height, original)


def test_thermal_rejects_bad_arguments():
    c = cfg(32)
    with pytest.raises(MapConfigError):
        erosion.thermal(c, torch.zeros((8, 8)), device=CPU)
    with pytest.raises(MapConfigError):
        erosion.thermal(c, torch.zeros(c.shape), iterations=-1, device=CPU)
    with pytest.raises(MapConfigError):
        erosion.thermal(c, torch.zeros(c.shape), rate=0.0, device=CPU)
    with pytest.raises(MapConfigError):
        erosion.thermal(c, torch.zeros(c.shape), angle_deg=0.0, device=CPU)


def drops(height: torch.Tensor) -> torch.Tensor:
    return torch.stack([erosion._neighbour(height, d).sub(height).abs() for d in range(4)])


def test_erode_interleaves_the_talus_pass_at_the_configured_cadence():
    c = cfg(64, world_size_m=8192.0)
    height = synth.heightfield(c, "highland", CPU)
    off = erosion.erode(c, height, quick(thermal_cadence=0), device=CPU)
    on = erosion.erode(c, height, quick(thermal_cadence=2, thermal_settle_passes=64), device=CPU)
    assert float(off.talus_m.max()) == pytest.approx(0.0)
    assert float(on.talus_m.max()) > 0.0
    assert on.stats()["talus_max_m"] >= on.stats()["talus_mean_m"] > 0.0
    limit = c.talus_delta(on.params.talus_angle_deg)
    assert float(drops(off.height).max()) > limit
    assert float(drops(on.height).max()) <= limit * 1.01
    assert float((drops(on.height) > limit * 1.01).float().mean()) < float(
        (drops(off.height) > limit * 1.01).float().mean()
    )


def log_uniform(resolution: int = 64, decades: float = 6.0) -> torch.Tensor:
    exponent = torch.linspace(0.0, decades, resolution * resolution, device=CPU)
    return torch.pow(10.0, exponent).reshape(resolution, resolution)


def test_robust_normalise_spans_the_unit_range_and_ignores_one_outlier():
    field = torch.rand((64, 64), device=CPU)
    field[0, 0] = 1e6
    normalised = erosion.robust_normalise(field)
    assert float(normalised.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalised.max()) == pytest.approx(1.0)
    assert float(normalised.quantile(0.5)) == pytest.approx(0.5, abs=0.1)


def test_robust_normalise_flattens_a_constant_field():
    assert float(erosion.robust_normalise(torch.full((8, 8), 3.0, device=CPU)).max()) == 0.0


def test_log_normalise_maps_a_log_uniform_field_onto_a_near_uniform_range():
    field = log_uniform()
    compressed = erosion.log_normalise(field)
    linear = erosion.robust_normalise(field)
    assert float(compressed.min()) >= 0.0
    assert float(compressed.max()) == pytest.approx(1.0)
    for fraction in (0.25, 0.5, 0.75):
        lifted = float(compressed.quantile(fraction))
        assert lifted == pytest.approx(fraction, abs=0.05), fraction
        assert lifted > 10.0 * float(linear.quantile(fraction)), fraction


def test_log_normalise_is_insensitive_to_the_unit_the_accumulator_is_counted_in():
    field = log_uniform()
    assert float(erosion.log_normalise(field).quantile(0.5)) == pytest.approx(
        float(erosion.log_normalise(field * 1000.0).quantile(0.5)), abs=0.05
    )


def test_log_normalise_preserves_order():
    ordered = erosion.log_normalise(log_uniform().reshape(-1))
    assert bool((ordered.diff() >= -1e-6).all())


def test_log_normalise_flattens_an_empty_accumulator():
    assert float(erosion.log_normalise(torch.zeros((8, 8), device=CPU)).max()) == 0.0
