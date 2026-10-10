from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from library import materials
from terrain import channels as channel_module
from terrain import scatter, splat
from terrain.config import CHANNEL_NAMES, SCATTER_KINDS, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 8

RULES = """
[biome]
name = "unit"

[layer.lush_grass]
material = "lush_grass"
weight = "moisture"

[layer.silt_flat]
material = "silt"
weight = "1 - moisture"

[species.tuft]
kind = "grass"
density = "lush_grass * (1 - slope)"
spacing_m = 1.5
scale = [0.7, 1.3]

[species.boulder]
kind = "rock"
density = "wear"
spacing_m = 5.0
scale = [0.6, 2.0]
water_buffer_m = 3.0
"""


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def biome(index) -> splat.Biome:
    return splat.parse_biome(RULES, index)


def cfg(**kwargs) -> MapConfig:
    base = {"name": "scatter-unit", "resolution": SIZE, "world_size_m": 64.0, "seed": 5}
    base.update(kwargs)
    return MapConfig(**base)


def blank(**kwargs) -> channel_module.ChannelStack:
    built = channel_module.ChannelStack(cfg(**kwargs), CPU)
    for name in CHANNEL_NAMES:
        built[name] = torch.zeros(built.cfg.shape, device=CPU)
    return built


def random_stack(seed: int = 3) -> channel_module.ChannelStack:
    built = channel_module.ChannelStack(cfg(), CPU)
    generator = torch.Generator(device=CPU).manual_seed(seed)
    for name in CHANNEL_NAMES:
        built[name] = torch.rand(built.cfg.shape, generator=generator, device=CPU)
    return built


def test_density_reads_splat_layer_weights_by_name(biome):
    built = blank()
    built["moisture"] = torch.ones(built.cfg.shape)
    result = splat.render(biome, built)
    field = scatter.density(biome.species[0], built, scatter.layer_weights(result))
    assert torch.allclose(field, result.weights[:, :, 0])


def test_density_is_cut_out_of_open_water(biome):
    built = blank()
    built["wear"] = torch.ones(built.cfg.shape)
    water = torch.zeros(built.cfg.shape)
    water[:4] = 1.0
    built["water"] = water
    field = scatter.density(biome.species[1], built)
    assert float(field[:4].max()) == 0.0


def test_water_buffer_widens_the_cut_in_metres(biome):
    built = blank(world_size_m=8.0)
    built["wear"] = torch.ones(built.cfg.shape)
    water = torch.zeros(built.cfg.shape)
    water[0, 0] = 1.0
    built["water"] = water
    field = scatter.density(biome.species[1], built)
    assert float(field[0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert float(field[0, 7]) == pytest.approx(1.0)


def test_water_buffer_needs_a_stack_with_cfg(biome):
    plain = {name: torch.zeros((SIZE, SIZE)) for name in CHANNEL_NAMES}
    with pytest.raises(MapConfigError, match="carries cfg"):
        scatter.density(biome.species[1], plain)


def test_density_is_clamped_to_the_unit_range(index):
    text = RULES.replace('density = "wear"', 'density = "wear * 4 - 1"')
    built = blank()
    built["wear"] = torch.linspace(0.0, 1.0, SIZE * SIZE).reshape(SIZE, SIZE)
    field = scatter.density(splat.parse_biome(text, index).species[1], built)
    assert float(field.min()) == 0.0
    assert float(field.max()) == 1.0


def test_build_returns_one_mask_per_species_in_order(biome):
    built = random_stack()
    masks = scatter.build(built, splat.render(biome, built))
    assert list(masks) == ["tuft", "boulder"]


def test_build_is_deterministic(biome):
    first = random_stack()
    second = random_stack()
    one = scatter.build(first, splat.render(biome, first))
    two = scatter.build(second, splat.render(biome, second))
    for name in one:
        assert torch.equal(scatter.quantise(one[name]), scatter.quantise(two[name]))


def test_quantise_rounds_to_the_nearest_step():
    field = torch.tensor([0.0, 0.5, 1.0, 1.5])
    quantised = scatter.quantise(field)
    assert quantised.dtype == torch.uint8
    assert torch.equal(quantised, torch.tensor([0, 128, 255, 255], dtype=torch.uint8))


def test_write_produces_grayscale_pngs_named_after_the_species(tmp_path, biome):
    built = random_stack()
    written = scatter.write(scatter.build(built, splat.render(biome, built)), tmp_path)
    assert {path.name for path in written} == {"scatter_tuft.png", "scatter_boulder.png"}
    for path in written:
        with Image.open(path) as image:
            assert image.mode == "L"
            assert image.size == (SIZE, SIZE)


def test_coverage_reports_mean_and_threshold_fraction():
    masks = {"boulder": torch.tensor([[0.0, 1.0], [1.0, 1.0]])}
    table = scatter.coverage(masks, threshold=0.5)
    assert table[0]["mask"] == "boulder"
    assert table[0]["mean"] == pytest.approx(0.75)
    assert table[0]["coverage"] == pytest.approx(0.75)


def test_species_are_parsed_with_defaults(biome):
    tuft, boulder = biome.species
    assert tuft.kind == "grass"
    assert tuft.scale == (0.7, 1.3)
    assert tuft.slope_align == 0.0
    assert tuft.exclusion_m == 0.0
    assert boulder.water_buffer_m == pytest.approx(3.0)
    assert biome.as_dict()["species"]["boulder"]["density"] == "wear"


SPECIES_HEAD = "[layer.lush_grass]\nmaterial = 'lush_grass'\nweight = 'moisture'\n"


@pytest.mark.parametrize(
    "species",
    [
        "[species.a]\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\n",
        "[species.a]\nkind = 'tree'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\n",
        "[species.a]\nkind = 'rock'\ndensity = 'snow_layer'\nspacing_m = 1.0\nscale = [1.0, 1.0]\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 0.0\nscale = [1.0, 1.0]\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [2.0, 1.0]\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = 1.0\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\nslope_align = 2.0\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\nexclusion_m = -1.0\n",
        "[species.a]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\ncolour = 'red'\n",
        "[species.Big-Rock]\nkind = 'rock'\ndensity = 'wear'\nspacing_m = 1.0\nscale = [1.0, 1.0]\n",
    ],
)
def test_bad_species_are_rejected(species: str, index):
    with pytest.raises(MapConfigError):
        splat.parse_biome(SPECIES_HEAD + species, index)


def test_layer_names_may_not_shadow_channels(index):
    with pytest.raises(MapConfigError, match="clash"):
        splat.parse_biome("[layer.wear]\nmaterial = 'silt'\nweight = 'flow'\n", index)


@pytest.mark.parametrize("name", ["temperate", "alpine", "arid"])
def test_shipped_biomes_declare_at_least_six_species(name, index):
    loaded = splat.biome(name, index=index)
    assert len(loaded.species) >= 6
    assert {species.kind for species in loaded.species} <= set(SCATTER_KINDS)
