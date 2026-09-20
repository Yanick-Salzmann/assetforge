from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from library import materials
from terrain import channels as channel_module
from terrain import splat
from terrain.config import CHANNEL_NAMES, MAX_SPLAT_LAYERS, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 8

RULES = """
[biome]
name = "temperate"
description = "test rules"
sharpness = 6.0

[layer.riverbed_gravel]
material = "river_rock"
weight = "smoothstep(flow, 0.55, 0.8) * (1 - wetness*0.3)"

[layer.silt_flat]
material = "silt"
weight = "deposition^1.5 * step(slope < 0.12) * patchiness_mid"
tiling_m = 5.0

[layer.exposed_rock]
material = "cliff_rock"
weight = "max(step(slope > 0.6), bedrock * wear)"
"""


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


def rules(tmp_path: Path, text: str, name: str = "temperate") -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def stack() -> channel_module.ChannelStack:
    cfg = MapConfig(name="unit", resolution=SIZE, world_size_m=512.0, seed=3)
    built = channel_module.ChannelStack(cfg, CPU)
    axis = torch.linspace(0.0, 1.0, SIZE, device=CPU, dtype=torch.float32)
    field = axis.reshape(-1, 1).expand(SIZE, SIZE).contiguous()
    for name in CHANNEL_NAMES:
        built[name] = field
    return built


def test_layers_keep_declaration_order(index):
    biome = splat.parse_biome(RULES, index)
    assert biome.names() == ("riverbed_gravel", "silt_flat", "exposed_rock")
    assert biome[0].name == "riverbed_gravel"
    assert biome["exposed_rock"].material == "cliff_rock"
    assert len(biome) == 3


def test_metadata_is_read(index):
    biome = splat.parse_biome(RULES, index)
    assert biome.name == "temperate"
    assert biome.description == "test rules"
    assert biome.sharpness == pytest.approx(6.0)


def test_tiling_defaults_to_the_material_scale(index):
    biome = splat.parse_biome(RULES, index)
    assert biome["riverbed_gravel"].tiling_m == pytest.approx(index["river_rock"].tiling_m)
    assert biome["silt_flat"].tiling_m == pytest.approx(5.0)


def test_variation_defaults_to_off(index):
    biome = splat.parse_biome(RULES, index)
    for layer in biome:
        assert layer.variation == pytest.approx(0.0)
        assert layer.variation_m == pytest.approx(splat.DEFAULT_VARIATION_M)


def test_variation_is_read_from_the_layer_table(index):
    text = "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\nvariation = 0.4\nvariation_m = 300.0\n"
    biome = splat.parse_biome(text, index)
    assert biome["a"].variation == pytest.approx(0.4)
    assert biome["a"].variation_m == pytest.approx(300.0)


@pytest.mark.parametrize(
    "text",
    [
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\nvariation = -0.1\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\nvariation = 1.1\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\nvariation_m = 1.0\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\nvariation_m = 999999.0\n",
    ],
)
def test_variation_out_of_range_is_rejected(text, index):
    with pytest.raises(MapConfigError):
        splat.parse_biome(text, index)


def test_referenced_channels_and_materials_are_reported(index):
    biome = splat.parse_biome(RULES, index)
    assert biome.materials() == ("cliff_rock", "river_rock", "silt")
    assert biome.channels() == (
        "slope",
        "flow",
        "deposition",
        "wear",
        "wetness",
        "bedrock",
        "patchiness_mid",
    )


def test_layers_evaluate_over_the_channel_stack(index):
    biome = splat.parse_biome(RULES, index)
    built = stack()
    for layer in biome:
        weight = layer.evaluate(built)
        assert weight.shape == built.cfg.shape
        assert weight.dtype == torch.float32


def test_as_dict_round_trips_the_sources(index):
    biome = splat.parse_biome(RULES, index)
    data = biome.as_dict()
    assert data["layers"]["silt_flat"]["material"] == "silt"
    assert data["layers"]["silt_flat"]["weight"].startswith("deposition^1.5")
    assert data["layers"]["silt_flat"]["variation"] == pytest.approx(0.0)
    assert data["layers"]["silt_flat"]["variation_m"] == pytest.approx(splat.DEFAULT_VARIATION_M)
    assert data["sharpness"] == pytest.approx(6.0)


def test_unknown_material_lists_the_known_ones(index):
    text = "[layer.a]\nmaterial = 'basalt'\nweight = 'flow'\n"
    with pytest.raises(MapConfigError) as error:
        splat.parse_biome(text, index)
    assert "basalt" in str(error.value)
    assert "cliff_rock" in str(error.value)


def test_unknown_channel_names_the_layer(index):
    text = "[layer.a]\nmaterial = 'silt'\nweight = 'elevation * 2'\n"
    with pytest.raises(MapConfigError) as error:
        splat.parse_biome(text, index)
    assert "[layer.a]" in str(error.value)
    assert "elevation" in str(error.value)


@pytest.mark.parametrize(
    "text",
    [
        "[biome]\nname = 'x'\n",
        "[layer.a]\nmaterial = 'silt'\n",
        "[layer.a]\nweight = 'flow'\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\ncolour = 'red'\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\ntiling_m = 0.0\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\ntiling_m = 900.0\n",
        "[biome]\nsharpness = 0.0\n[layer.a]\nmaterial = 'silt'\nweight = 'flow'\n",
        "[biome]\nsharpness = 500.0\n[layer.a]\nmaterial = 'silt'\nweight = 'flow'\n",
        "[biome]\nsharpen = 2.0\n[layer.a]\nmaterial = 'silt'\nweight = 'flow'\n",
        "[palette.a]\nmaterial = 'silt'\n",
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow +'\n",
        "this is not toml",
    ],
)
def test_rejected_rule_sets(text: str, index):
    with pytest.raises(MapConfigError):
        splat.parse_biome(text, index)


def test_too_many_layers_is_rejected(index):
    text = "".join(
        f"[layer.l{n}]\nmaterial = 'silt'\nweight = 'flow'\n" for n in range(MAX_SPLAT_LAYERS + 1)
    )
    with pytest.raises(MapConfigError) as error:
        splat.parse_biome(text, index)
    assert str(MAX_SPLAT_LAYERS) in str(error.value)


def test_exactly_the_texture_budget_is_accepted(index):
    text = "".join(
        f"[layer.l{n}]\nmaterial = 'silt'\nweight = 'flow'\n" for n in range(MAX_SPLAT_LAYERS)
    )
    assert len(splat.parse_biome(text, index)) == MAX_SPLAT_LAYERS


def test_load_names_the_file_in_errors(tmp_path, index):
    path = rules(tmp_path, "[layer.a]\nmaterial = 'basalt'\nweight = 'flow'\n", "broken")
    with pytest.raises(MapConfigError) as error:
        splat.load_biome(path, index)
    assert "broken.toml" in str(error.value)


def test_name_falls_back_to_the_file_stem(tmp_path, index):
    path = rules(tmp_path, "[layer.a]\nmaterial = 'silt'\nweight = 'flow'\n", "alpine")
    assert splat.load_biome(path, index).name == "alpine"
    assert splat.load_biome(path, index).sharpness == pytest.approx(splat.DEFAULT_SHARPNESS)


def test_biome_by_name_and_available(tmp_path, index):
    rules(tmp_path, RULES, "temperate")
    rules(tmp_path, RULES, "arid")
    assert splat.available(tmp_path) == ("arid", "temperate")
    assert splat.biome("arid", tmp_path, index).name == "temperate"


def test_unknown_biome_lists_what_is_there(tmp_path, index):
    rules(tmp_path, RULES, "temperate")
    with pytest.raises(MapConfigError) as error:
        splat.biome("tundra", tmp_path, index)
    assert "temperate" in str(error.value)


def test_available_on_an_empty_directory(tmp_path):
    assert splat.available(tmp_path) == ()
    assert splat.available(tmp_path / "nothing") == ()


def test_missing_file_is_reported(tmp_path, index):
    with pytest.raises(MapConfigError):
        splat.load_biome(tmp_path / "nothing.toml", index)


def test_every_shipped_biome_loads():
    for name in splat.available():
        biome = splat.biome(name)
        assert len(biome) >= 1
        assert set(biome.materials()) <= set(materials.load().names())
