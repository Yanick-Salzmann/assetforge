from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from assets import blender as blender_discovery
from library import materials
from terrain import beauty
from terrain import channels as channel_module
from terrain import splat
from terrain.config import CHANNEL_NAMES, MapConfig

CPU = "cpu"
SIZE = 32

BIOME_TEXT = """
[biome]
name = "beauty-unit"
sharpness = 2.0

[layer.riverbed_gravel]
material = "river_rock"
weight = "flow"

[layer.silt_flat]
material = "silt"
weight = "1 - flow"
"""


def cfg(resolution: int = SIZE) -> MapConfig:
    return MapConfig(name="beauty-unit", resolution=resolution, world_size_m=512.0, height_range_m=300.0, seed=5)


def stack(resolution: int = SIZE) -> channel_module.ChannelStack:
    built = channel_module.ChannelStack(cfg(resolution), CPU)
    generator = torch.Generator(device=CPU).manual_seed(3)
    for name in CHANNEL_NAMES:
        built[name] = torch.rand(built.cfg.shape, generator=generator, device=CPU)
    return built


@pytest.fixture(scope="module")
def material_index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def splat_result(material_index) -> splat.SplatResult:
    biome = splat.parse_biome(BIOME_TEXT, material_index)
    return splat.render(biome, stack())


def test_texture_index_reads_the_splat_filename():
    assert beauty._texture_index("splat_0.png") == 0
    assert beauty._texture_index("splat_1.png") == 1


def test_material_metadata_reports_missing_maps(material_index):
    broken = dataclasses.replace(
        material_index["silt"], maps={"albedo": "materials/does-not-exist/albedo.jpg"}
    )
    index = materials.MaterialIndex(material_index.version, {**material_index.materials, "silt": broken})
    with pytest.raises(beauty.BeautyRenderError, match="silt"):
        beauty._material_metadata(["silt"], index)


def test_render_rejects_a_height_field_of_the_wrong_shape(material_index, splat_result):
    wrong = torch.rand((SIZE + 1, SIZE + 1), device=CPU)
    with pytest.raises(beauty.BeautyRenderError, match="does not match"):
        beauty.render(cfg(), wrong, splat_result, material_index)


def test_render_rejects_an_out_of_range_centre(material_index, splat_result):
    with pytest.raises(beauty.BeautyRenderError, match="centre"):
        beauty.render(cfg(), stack()["height"], splat_result, material_index, centre=(1.5, 0.5))


def test_render_rejects_a_non_positive_patch_size(material_index, splat_result):
    with pytest.raises(beauty.BeautyRenderError, match="patch_size_m"):
        beauty.render(cfg(), stack()["height"], splat_result, material_index, patch_size_m=0.0)


def test_write_height_png_flips_vertically_and_scales_to_16_bit(tmp_path):
    height = torch.zeros((4, 4), device=CPU)
    height[0, :] = 1.0
    path = beauty._write_height_png(height, tmp_path / "height.png")
    written = np.array(Image.open(path))
    assert written.dtype == np.uint16
    assert written[0, 0] == 0
    assert written[-1, 0] == 65535


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_render_produces_both_views_end_to_end(material_index, splat_result, tmp_path):
    result = beauty.render(
        cfg(),
        stack()["height"],
        splat_result,
        material_index,
        out_dir=tmp_path,
        grid_resolution=16,
        samples=8,
        resolution=(160, 90),
        timeout_s=180.0,
    )
    assert result.three_quarter.is_file()
    assert result.ground.is_file()
    with Image.open(result.three_quarter) as image:
        assert image.size == (160, 90)
    with Image.open(result.ground) as image:
        assert image.size == (160, 90)
