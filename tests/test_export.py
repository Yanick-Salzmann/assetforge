from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from library import materials
from terrain import channels as channel_module
from terrain import export
from terrain import manifest
from terrain import splat
from terrain.channels import WaterLevel
from terrain.config import CHANNEL_NAMES, MapConfig

CPU = "cpu"
SIZE = 8

TWO_LAYER = """
[biome]
name = "flat"

[layer.dry_grass]
material = "dry_grass"
weight = "1 - moisture"

[layer.silt_flat]
material = "silt"
weight = "moisture"
"""

NO_GRASS = """
[biome]
name = "bare"

[layer.silt_flat]
material = "silt"
weight = "1 - slope"
"""


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def two_layer_biome(index) -> splat.Biome:
    return splat.parse_biome(TWO_LAYER, index)


@pytest.fixture(scope="module")
def no_grass_biome(index) -> splat.Biome:
    return splat.parse_biome(NO_GRASS, index)


def cfg() -> MapConfig:
    return MapConfig(name="unit", resolution=SIZE, world_size_m=512.0, seed=5)


def stack(c: MapConfig) -> channel_module.ChannelStack:
    built = channel_module.ChannelStack(c, CPU)
    generator = torch.Generator(device=CPU).manual_seed(3)
    for name in CHANNEL_NAMES:
        built[name] = torch.rand(c.shape, generator=generator, device=CPU)
    return built


def water() -> WaterLevel:
    return WaterLevel(
        sea_level_m=12.0,
        covered_fraction=0.2,
        mean_depth_m=1.5,
        max_depth_m=4.0,
        depth_reference_m=12.0,
    )


def test_grass_weight_sums_layers_named_grass(two_layer_biome):
    c = cfg()
    result = splat.render(two_layer_biome, stack(c))
    weight = export.grass_weight(result)
    assert torch.allclose(weight, result.weights[:, :, 0])


def test_grass_weight_is_zero_without_a_grass_layer(no_grass_biome):
    c = cfg()
    result = splat.render(no_grass_biome, stack(c))
    weight = export.grass_weight(result)
    assert torch.equal(weight, torch.zeros(c.shape))


def test_write_produces_the_full_deliverable_set(tmp_path, two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    outcome = export.write(c, built, water(), result, out_dir=tmp_path)

    names = {path.name for path in outcome.files}
    assert names == {
        manifest.HEIGHTMAP_NAME,
        manifest.WATER_MASK_NAME,
        "splat_0.png",
        "scatter_rock.png",
        "scatter_tree.png",
        "scatter_grass.png",
        "scatter_debris.png",
        "normal.png",
        "colour_macro.png",
    }
    for path in outcome.files:
        with Image.open(path) as image:
            assert image.size == (SIZE, SIZE)

    assert outcome.manifest_path == tmp_path / manifest.MANIFEST_NAME
    assert manifest.load(outcome.manifest_path) == outcome.manifest
    assert outcome.manifest["normal_map"] == "normal.png"
    assert outcome.manifest["colour_macro"]["path"] == "colour_macro.png"
    assert {entry["kind"] for entry in outcome.manifest["scatter"]} == {
        "rock",
        "tree",
        "grass",
        "debris",
    }


def test_write_skips_the_normal_map_when_disabled(tmp_path, two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    outcome = export.write(c, built, water(), result, out_dir=tmp_path, write_normal=False)

    assert outcome.manifest["normal_map"] is None
    assert not (tmp_path / "normal.png").exists()
    assert "normal.png" not in {path.name for path in outcome.files}


def test_write_skips_the_colour_macro_when_disabled(tmp_path, two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    outcome = export.write(c, built, water(), result, out_dir=tmp_path, write_colour_macro=False)

    assert outcome.manifest["colour_macro"] is None
    assert not (tmp_path / "colour_macro.png").exists()


def test_write_records_the_water_surface_when_given(tmp_path, two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    surface_m = built["height"].mul(c.height_range_m).add(2.0)
    outcome = export.write(c, built, water(), result, out_dir=tmp_path, water_surface_m=surface_m)

    assert outcome.manifest["water_surface"] == manifest.WATER_SURFACE_NAME
    with Image.open(tmp_path / manifest.WATER_SURFACE_NAME) as image:
        assert image.mode in ("I;16", "I")
        written = torch.from_numpy(np.asarray(image, dtype=np.float32))
    expected = surface_m.div(c.height_range_m).clamp(0.0, 1.0).mul(65535.0)
    assert torch.allclose(written, expected, atol=1.0)


def test_write_omits_the_water_surface_by_default(tmp_path, two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    outcome = export.write(c, built, water(), result, out_dir=tmp_path)

    assert outcome.manifest["water_surface"] is None
    assert not (tmp_path / manifest.WATER_SURFACE_NAME).exists()


def test_write_defaults_to_cfg_out_dir(two_layer_biome):
    c = cfg()
    built = stack(c)
    result = splat.render(two_layer_biome, built)
    try:
        outcome = export.write(c, built, water(), result)
        assert outcome.out_dir == c.out_dir()
    finally:
        import shutil

        shutil.rmtree(c.out_dir(), ignore_errors=True)


def test_write_rejects_a_stack_missing_height_or_water(tmp_path, two_layer_biome):
    c = cfg()
    built = channel_module.ChannelStack(c, CPU)
    for name in CHANNEL_NAMES:
        if name in ("height", "water"):
            continue
        built[name] = torch.zeros(c.shape, device=CPU)
    result_stack = stack(c)
    result = splat.render(two_layer_biome, result_stack)
    with pytest.raises(export.ExportError):
        export.write(c, built, water(), result, out_dir=tmp_path)
