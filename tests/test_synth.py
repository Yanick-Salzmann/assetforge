from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from terrain import synth
from terrain.config import MapConfig, MapConfigError
from terrain.device import resolve as resolve_device
from terrain.synth import FractalParams

CPU = "cpu"


def cfg(resolution: int = 128, **kwargs) -> MapConfig:
    base = {"name": "unit", "resolution": resolution, "world_size_m": 4096.0, "seed": 1234}
    base.update(kwargs)
    return MapConfig(**base)


def coords(count: int = 4096, span: float = 4096.0):
    axis = torch.linspace(0.0, span, count, device=CPU, dtype=torch.float32)
    return axis, axis.flip(0)


def test_grid_covers_the_world_in_metres():
    c = cfg(256)
    x, y = synth.grid(c, CPU)
    assert x.shape == y.shape == c.shape
    assert x.dtype == torch.float32
    half = 0.5 * c.metres_per_pixel
    assert float(x.min()) == pytest.approx(half)
    assert float(x.max()) == pytest.approx(c.world_size_m - half)
    assert float(x[0, 1] - x[0, 0]) == pytest.approx(c.metres_per_pixel)
    assert float(y[1, 0] - y[0, 0]) == pytest.approx(c.metres_per_pixel)
    assert torch.equal(x, y.T)


def test_perlin_is_bounded_and_deterministic():
    x, y = coords()
    a = synth.perlin(x, y, 99)
    b = synth.perlin(x, y, 99)
    assert torch.equal(a, b)
    assert float(a.min()) >= -1.0
    assert float(a.max()) <= 1.0
    assert float(a.std()) > 0.05
    assert not torch.equal(a, synth.perlin(x, y, 100))


def test_perlin_is_zero_on_the_lattice():
    lattice = torch.arange(0.0, 64.0, device=CPU, dtype=torch.float32)
    values = synth.perlin(lattice, lattice, 3)
    assert float(values.abs().max()) < 1e-5


def test_perlin_is_continuous_across_cell_boundaries():
    step = 1e-4
    left = torch.tensor([8.0 - step, 12.0 - step], device=CPU)
    right = torch.tensor([8.0 + step, 12.0 + step], device=CPU)
    y = torch.tensor([3.25, 7.75], device=CPU)
    delta = (synth.perlin(right, y, 11) - synth.perlin(left, y, 11)).abs()
    assert float(delta.max()) < 1e-3


def test_fbm_is_bounded_and_octaves_add_detail():
    x, y = coords()
    smooth = synth.fbm(x, y, 5, 1.0 / 512.0, octaves=1)
    rough = synth.fbm(x, y, 5, 1.0 / 512.0, octaves=6)
    assert float(rough.min()) >= -1.0
    assert float(rough.max()) <= 1.0
    assert float(rough.diff().abs().mean()) > float(smooth.diff().abs().mean())


def test_fbm_lacunarity_and_gain_change_the_field():
    x, y = coords()
    base = synth.fbm(x, y, 5, 1.0 / 512.0, octaves=4)
    assert not torch.equal(base, synth.fbm(x, y, 5, 1.0 / 512.0, octaves=4, lacunarity=2.5))
    assert not torch.equal(base, synth.fbm(x, y, 5, 1.0 / 512.0, octaves=4, gain=0.7))


def test_ridged_is_in_unit_range_and_crest_heavy():
    x, y = coords()
    field = synth.ridged(x, y, 17, 1.0 / 512.0, octaves=5)
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0
    assert float(field.std()) > 0.05
    assert not torch.equal(field, synth.ridged(x, y, 17, 1.0 / 512.0, octaves=5, sharpness=4.0))


@pytest.mark.parametrize("feature", ["f1", "f2", "f2f1"])
def test_worley_is_in_unit_range_and_deterministic(feature):
    x, y = coords()
    field = synth.worley(x, y, 23, 1.0 / 256.0, feature)
    assert torch.equal(field, synth.worley(x, y, 23, 1.0 / 256.0, feature))
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0
    assert float(field.std()) > 0.02


def test_worley_f2_is_never_nearer_than_f1():
    x, y = coords()
    f1 = synth.worley(x, y, 23, 1.0 / 256.0, "f1")
    f2 = synth.worley(x, y, 23, 1.0 / 256.0, "f2")
    assert float((f2 * (synth._WORLEY_F2_BOUND / synth._WORLEY_F1_BOUND) - f1).min()) >= -1e-6


def test_worley_rejects_an_unknown_feature():
    x, y = coords(16)
    with pytest.raises(MapConfigError):
        synth.worley(x, y, 1, 1.0 / 256.0, "f3")


def test_domain_warp_displaces_by_at_most_the_strength():
    x, y = coords()
    strength = 120.0
    wx, wy = synth.domain_warp(x, y, 31, strength, 1.0 / 1024.0)
    assert float((wx - x).abs().max()) <= strength + 1e-3
    assert float((wy - y).abs().max()) <= strength + 1e-3
    assert float((wx - x).abs().mean()) > 1.0
    assert not torch.equal(wx - x, wy - y)


def test_domain_warp_of_zero_strength_is_the_identity():
    x, y = coords(64)
    wx, wy = synth.domain_warp(x, y, 31, 0.0, 1.0 / 1024.0)
    assert wx is x
    assert wy is y
    with pytest.raises(MapConfigError):
        synth.domain_warp(x, y, 31, -1.0, 1.0 / 1024.0)


def test_field_helpers_are_world_anchored_not_pixel_anchored():
    x, y = coords()
    field = synth.fbm(x, y, 7, 1.0 / 1024.0, octaves=4)
    assert torch.equal(field, synth.fbm(x.clone(), y.clone(), 7, 1.0 / 1024.0, octaves=4))
    doubled = synth.fbm(x, y, 7, 1.0 / 2048.0, octaves=4)
    assert float(doubled.diff().abs().mean()) < float(field.diff().abs().mean())


@pytest.mark.parametrize(
    "builder",
    [
        lambda c: synth.warped_fbm(c, FractalParams(feature_size_m=1024.0), 150.0, device=CPU),
        lambda c: synth.warped_ridged(c, FractalParams(feature_size_m=1024.0), 150.0, device=CPU),
        lambda c: synth.cellular(c, 512.0, "f1", 150.0, device=CPU),
    ],
)
def test_map_level_fields_are_unit_ranged_float32_and_reproducible(builder):
    c = cfg(128)
    field = builder(c)
    assert field.shape == c.shape
    assert field.dtype == torch.float32
    assert 0.0 <= float(field.min())
    assert float(field.max()) <= 1.0
    assert float(field.std()) > 0.01
    assert torch.equal(field, builder(c))
    assert not torch.equal(field, builder(cfg(128, seed=1235)))


def test_map_level_fields_stay_on_the_requested_device():
    c = cfg(64)
    assert synth.warped_fbm(c, device=CPU).device.type == "cpu"
    assert synth.grid(c, torch.device(CPU))[0].device.type == "cpu"


def test_warp_strength_actually_bends_the_field():
    c = cfg(128)
    params = FractalParams(feature_size_m=1024.0)
    straight = synth.warped_fbm(c, params, 0.0, device=CPU)
    bent = synth.warped_fbm(c, params, 400.0, device=CPU)
    assert float((straight - bent).abs().mean()) > 0.01


def test_to01_and_normalise():
    signed = torch.tensor([-1.0, 0.0, 1.0], device=CPU)
    assert torch.allclose(synth.to01(signed), torch.tensor([0.0, 0.5, 1.0], device=CPU))
    assert torch.allclose(synth.normalise(signed * 0.3), torch.tensor([0.0, 0.5, 1.0], device=CPU))
    flat = torch.full((4,), 2.0, device=CPU)
    assert float(synth.normalise(flat).abs().max()) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_size_m": 0.0},
        {"octaves": 0},
        {"lacunarity": 1.0},
        {"gain": 0.0},
        {"gain": 1.0},
    ],
)
def test_fractal_params_are_validated(kwargs):
    with pytest.raises(MapConfigError):
        FractalParams(**kwargs)


def test_cellular_rejects_a_non_positive_cell_size():
    with pytest.raises(MapConfigError):
        synth.cellular(cfg(64), 0.0, device=CPU)


def _hash2_reference_int64(ix: torch.Tensor, iy: torch.Tensor, seed: int) -> torch.Tensor:
    """Independent int64 reimplementation of the pre-optimisation _hash2, as ground truth."""
    mask32 = 0xFFFFFFFF

    def mix(value: torch.Tensor) -> torch.Tensor:
        value = value & mask32
        value = ((value ^ (value >> 16)) * 0x85EBCA6B) & mask32
        value = ((value ^ (value >> 13)) * 0xC2B2AE35) & mask32
        return (value ^ (value >> 16)) & mask32

    key = (ix & mask32) * 0x1F1F1F1F + (iy & mask32) * 0x2545F491 + (seed & mask32)
    return mix(key)


def test_hash2_int32_pipeline_matches_int64_reference():
    torch.manual_seed(7)
    ix = torch.randint(-1_000_000, 1_000_000, (200_000,), dtype=torch.int64)
    iy = torch.randint(-1_000_000, 1_000_000, (200_000,), dtype=torch.int64)
    for seed in (0, 1, 2_147_483_647, 3_000_000_000):
        expected = _hash2_reference_int64(ix, iy, seed)
        actual = synth._hash2(ix, iy, seed).to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(actual, expected)


@pytest.mark.gpu
def test_accelerator_matches_cpu_within_float_tolerance():
    device = resolve_device()
    if not device.is_accelerator:
        pytest.skip("no accelerator resolved")
    c = cfg(128)
    params = FractalParams(feature_size_m=1024.0)
    on_cpu = synth.warped_fbm(c, params, 150.0, device=CPU)
    on_gpu = synth.warped_fbm(c, params, 150.0, device=device).cpu()
    assert torch.allclose(on_cpu, on_gpu, atol=1e-4)


def test_smoothstep_fades_between_its_edges():
    values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], device=CPU)
    faded = synth.smoothstep(0.0, 1.0, values)
    assert float(faded[0]) == 0.0
    assert float(faded[1]) == 0.0
    assert float(faded[2]) == pytest.approx(0.5)
    assert float(faded[3]) == 1.0
    assert float(faded[4]) == 1.0
    descending = synth.smoothstep(1.0, 0.0, values)
    assert float(descending[0]) == 1.0
    assert float(descending[4]) == 0.0
    with pytest.raises(MapConfigError):
        synth.smoothstep(1.0, 1.0, values)


def test_shape_presets_resolve_by_name_and_instance():
    names = synth.shape_preset_names()
    assert "continental" in names
    assert synth.DEFAULT_SHAPE in names
    for name in names:
        assert isinstance(synth.shape_preset(name), synth.ShapeParams)
    params = synth.ShapeParams(belt_weight=0.1)
    assert synth.resolve_shape(params) is params
    assert synth.resolve_shape(None) is synth.SHAPE_PRESETS[synth.DEFAULT_SHAPE]
    with pytest.raises(MapConfigError):
        synth.shape_preset("volcano")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"continent_feature_size_m": 0.0},
        {"belt_cell_size_m": -1.0},
        {"basin_feature_size_m": 0.0},
        {"continent_warp_m": -1.0},
        {"belt_weight": -0.1},
        {"continent_octaves": 0},
        {"continent_bias": 2.0},
        {"belt_sharpness": 0.0},
        {"belt_width": 0.0},
        {"belt_width": 1.5},
        {"belt_chain_scale": 0.5},
        {"belt_chain_cover": 0.0},
        {"belt_chain_cover": 1.5},
        {"shelf": "peninsula"},
        {"shelf_margin": 0.0},
        {"shelf_margin": 0.5},
        {"shelf_power": 0.0},
        {"relief_floor": 1.5},
    ],
)
def test_shape_params_are_validated(kwargs):
    with pytest.raises(MapConfigError):
        synth.ShapeParams(**kwargs)


def test_shelf_mask_is_open_without_a_shelf():
    mask = synth.shelf_mask(cfg(64), synth.ShapeParams(), CPU)
    assert float(mask.min()) == 1.0
    assert float(mask.max()) == 1.0


def test_island_shelf_drops_the_map_edges_to_sea():
    c = cfg(128)
    mask = synth.shelf_mask(c, "island", CPU)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
    corners = torch.tensor(
        [float(mask[0, 0]), float(mask[0, -1]), float(mask[-1, 0]), float(mask[-1, -1])]
    )
    assert float(corners.max()) == 0.0
    centre = mask[48:80, 48:80]
    assert float(centre.mean()) > 0.8


def test_coast_shelf_runs_along_one_axis():
    c = cfg(128)
    mask = synth.shelf_mask(c, "coastal_range", CPU)
    assert float(mask[:, :8].mean()) < 0.2
    assert float(mask[:, -8:].mean()) > 0.9


def test_uplift_is_normalised_and_low_frequency():
    c = cfg(128)
    field = synth.uplift(c, "continental", CPU)
    assert field.shape == c.shape
    assert field.dtype == torch.float32
    assert float(field.min()) == pytest.approx(0.0)
    assert float(field.max()) == pytest.approx(1.0)
    neighbour_step = (field[:, 1:] - field[:, :-1]).abs().mean()
    assert float(neighbour_step) < 0.02


def test_uplift_belts_add_structure_over_the_continent_alone():
    c = cfg(128)
    flat = synth.ShapeParams(belt_weight=0.0, basin_weight=0.0)
    belted = synth.ShapeParams(belt_weight=0.9, basin_weight=0.0)
    assert float(synth.uplift(c, belted, CPU).std()) != pytest.approx(
        float(synth.uplift(c, flat, CPU).std()), abs=1e-4
    )


def test_heightfield_is_deterministic_per_seed_and_shape():
    c = cfg(128)
    first = synth.heightfield(c, "continental", CPU)
    assert torch.equal(first, synth.heightfield(c, "continental", CPU))
    assert not torch.equal(first, synth.heightfield(cfg(128, seed=99), "continental", CPU))
    assert not torch.equal(first, synth.heightfield(c, "highland", CPU))


def test_heightfield_spans_the_unit_range_for_every_preset():
    c = cfg(128)
    for name in synth.shape_preset_names():
        field = synth.heightfield(c, name, CPU)
        assert field.shape == c.shape
        assert float(field.min()) == pytest.approx(0.0)
        assert float(field.max()) == pytest.approx(1.0)
        assert float(field.std()) > 0.05
        assert bool(torch.isfinite(field).all())


def test_heightfield_detail_rides_on_the_uplift():
    c = cfg(128)
    bare = synth.ShapeParams(ridge_weight=0.0, detail_weight=0.0)
    assert torch.allclose(synth.heightfield(c, bare, CPU), synth.uplift(c, bare, CPU), atol=1e-5)
    detailed = synth.ShapeParams(ridge_weight=0.6, detail_weight=0.3)
    field = synth.heightfield(c, detailed, CPU)
    roughness = (field[:, 1:] - field[:, :-1]).abs().mean()
    assert float(roughness) > float((synth.uplift(c, detailed, CPU).diff(dim=1)).abs().mean())


def test_island_heightfield_keeps_the_edges_low():
    c = cfg(128)
    field = synth.heightfield(c, "island", CPU)
    edge = torch.cat([field[0, :], field[-1, :], field[:, 0], field[:, -1]])
    assert float(edge.mean()) < float(field[32:96, 32:96].mean())
