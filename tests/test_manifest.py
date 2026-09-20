from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from library import materials
from terrain import channels as channel_module
from terrain import manifest
from terrain import splat
from terrain.channels import WaterLevel
from terrain.config import CHANNEL_NAMES, MapConfig

CPU = "cpu"
SIZE = 8

ONE_LAYER = """
[biome]
name = "flat"

[layer.silt_flat]
material = "silt"
weight = "1 - slope"
"""


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def biome(index) -> splat.Biome:
    return splat.parse_biome(ONE_LAYER, index)


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


def test_build_carries_world_scale_and_water(biome):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    assert payload["schema_version"] == manifest.SCHEMA_VERSION
    assert payload["world_size_m"] == c.world_size_m
    assert payload["height_range_m"] == c.height_range_m
    assert payload["metres_per_pixel"] == pytest.approx(c.metres_per_pixel)
    assert payload["water"]["sea_level_m"] == 12.0
    assert payload["seed"] == c.seed


def test_build_carries_layer_material_and_tiling(biome, index):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    layer = payload["splat"]["layers"][0]
    assert layer["material"] == "silt"
    assert layer["tiling_m"] == pytest.approx(index["silt"].tiling_m)
    assert layer["texture"] == "splat_0.png"
    assert layer["channel"] == "r"


def test_build_picks_up_the_biome_rule_path(biome, index, tmp_path):
    rule = tmp_path / "flat.toml"
    rule.write_text(ONE_LAYER, encoding="utf-8")
    loaded = splat.load_biome(rule, index)
    c = cfg()
    payload = manifest.build(c, water(), splat.render(loaded, stack(c)))
    assert payload["rule_path"] == rule.as_posix()


def test_build_defaults_scatter_and_normal_map_to_empty(biome):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    assert payload["scatter"] == []
    assert payload["normal_map"] is None


def test_build_accepts_scatter_masks_and_normal_map(biome):
    c = cfg()
    masks = [manifest.ScatterMask("rock", "scatter_rock.png")]
    payload = manifest.build(
        c, water(), splat.render(biome, stack(c)), normal_map="normal.png", scatter=masks
    )
    assert payload["scatter"] == [{"kind": "rock", "path": "scatter_rock.png"}]
    assert payload["normal_map"] == "normal.png"


def test_scatter_mask_rejects_unknown_kind():
    with pytest.raises(manifest.ManifestError):
        manifest.ScatterMask("cloud", "scatter_cloud.png")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("world_size_m"),
        lambda payload: payload.__setitem__("schema_version", 2),
        lambda payload: payload.__setitem__("resolution", "big"),
        lambda payload: payload.__setitem__("unknown_key", 1),
        lambda payload: payload["water"].pop("sea_level_m"),
        lambda payload: payload["splat"]["layers"][0].pop("material"),
        lambda payload: payload.__setitem__("scatter", [{"kind": "cloud", "path": "x.png"}]),
        lambda payload: payload.__setitem__("normal_map", 3),
    ],
)
def test_validate_rejects_malformed_payloads(biome, mutate):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    mutate(payload)
    with pytest.raises(manifest.ManifestError):
        manifest.validate(payload)


def test_write_then_load_round_trips(tmp_path, biome):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    target = manifest.write(payload, tmp_path)
    assert target.name == manifest.MANIFEST_NAME
    assert manifest.load(target) == payload


def test_verify_passes_when_every_declared_file_exists_at_resolution(tmp_path, biome, index):
    c = cfg()
    result = splat.render(biome, stack(c))
    splat.write(result, tmp_path)
    Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8), "L").save(tmp_path / manifest.HEIGHTMAP_NAME)
    Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8), "L").save(tmp_path / manifest.WATER_MASK_NAME)
    payload = manifest.build(c, water(), result)
    manifest.verify(payload, tmp_path)


def test_verify_fails_when_a_declared_file_is_missing(tmp_path, biome):
    c = cfg()
    payload = manifest.build(c, water(), splat.render(biome, stack(c)))
    with pytest.raises(manifest.ManifestError):
        manifest.verify(payload, tmp_path)


def test_verify_fails_when_a_declared_file_has_the_wrong_size(tmp_path, biome):
    c = cfg()
    result = splat.render(biome, stack(c))
    splat.write(result, tmp_path)
    Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8), "L").save(tmp_path / manifest.WATER_MASK_NAME)
    Image.fromarray(np.zeros((SIZE + 1, SIZE + 1), dtype=np.uint8), "L").save(
        tmp_path / manifest.HEIGHTMAP_NAME
    )
    payload = manifest.build(c, water(), result)
    with pytest.raises(manifest.ManifestError):
        manifest.verify(payload, tmp_path)
