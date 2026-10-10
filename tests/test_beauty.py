from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from assets import blender as blender_discovery
from library import materials
from terrain import atmosphere, beauty, placement, vegetation
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


SPECIES_TEXT = """
[species.scrub]
kind = "shrub"
density = "1"
spacing_m = 20.0
scale = [1.0, 1.0]
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


@pytest.fixture(scope="module")
def shrub(material_index) -> splat.Species:
    return splat.parse_biome(BIOME_TEXT + SPECIES_TEXT, material_index).species[0]


def shrub_placement(species: splat.Species, rows: list[list[float]]) -> placement.Placement:
    return placement.Placement(species, np.array(rows, dtype="<f4"), 20.0)


def flat_height(level: float = 0.0) -> np.ndarray:
    return np.full((SIZE, SIZE), int(level * 65535), dtype=np.uint16)


def test_scatter_batches_move_instances_into_blender_axes(shrub):
    placed = shrub_placement(shrub, [[10.0, -100.0, 42.0, 0.0, 2.0, 0.0]])
    (batch,) = beauty.scatter_batches(cfg(), [placed], {}, flat_height(), (256.0, 256.0))
    x, y, z, rx, ry, rz, scale = batch.records[0]
    assert (x, y) == pytest.approx((266.0, 356.0))
    assert z == pytest.approx(42.0 - beauty.KIND_SINK_M["shrub"] * 2.0)
    assert (rx, ry, rz) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert scale == pytest.approx(2.0)
    assert batch.glb is None


def test_scatter_batches_drop_instances_beyond_the_kind_reach(shrub):
    reach = beauty.KIND_REACH_M["shrub"]
    placed = shrub_placement(shrub, [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    far = (256.0 + reach + 1.0, 256.0)
    assert beauty.scatter_batches(cfg(), [placed], {}, flat_height(), far) == ()


def test_scatter_batches_split_variants_and_fold_in_unit_scale(shrub, tmp_path):
    glb = tmp_path / "kits" / "a.glb"
    glb.parent.mkdir()
    glb.write_bytes(b"glTF")
    variants = {
        "scrub": (
            vegetation.Variant("pack/a", "kits/a.glb", 100, 0.5),
            vegetation.Variant("pack/b", "kits/missing.glb", 100, 2.0),
        )
    }
    placed = shrub_placement(shrub, [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [5.0, 5.0, 0.0, 0.0, 1.0, 1.0]])
    first, second = beauty.scatter_batches(cfg(), [placed], variants, flat_height(), (256.0, 256.0), tmp_path)
    assert (first.variant, first.glb, float(first.records[0, 6])) == (0, glb, 0.5)
    assert (second.variant, second.glb, float(second.records[0, 6])) == (1, None, 2.0)


def test_instance_euler_leans_up_onto_the_normal_and_keeps_yaw():
    normal = np.array([[0.3, -0.2, 1.0]])
    normal /= np.linalg.norm(normal)
    yaw = np.array([0.7])
    rx, ry, rz = beauty.instance_euler(normal, yaw)[0]
    def axis(angle, index):
        c, s = np.cos(angle), np.sin(angle)
        m = np.eye(3)
        i, j = [k for k in range(3) if k != index]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        return m if index != 1 else m.T
    rotation = axis(rz, 2) @ axis(ry, 1) @ axis(rx, 0)
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(normal[0])
    flat = beauty.instance_euler(np.array([[0.0, 0.0, 1.0]]), yaw)[0]
    assert flat == pytest.approx((0.0, 0.0, 0.7))


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


def test_lighting_args_key_the_sun_by_day():
    lighting = beauty.lighting_args(atmosphere.Atmosphere(time_of_day_h=10.0))
    assert lighting["summary"]["key"] == "sun"
    assert lighting["sky"]["sun_elevation_rad"] > 0.0
    assert lighting["moon"]["energy"] == 0.0
    assert lighting["exposure"]["key"] == pytest.approx(beauty.EXPOSURE_KEY)


def test_lighting_args_key_the_moon_by_night():
    lighting = beauty.lighting_args(atmosphere.Atmosphere(time_of_day_h=23.0))
    assert lighting["summary"]["key"] == "moon"
    assert lighting["moon"]["energy"] == pytest.approx(beauty.MOON_IRRADIANCE)
    assert lighting["exposure"]["key"] < beauty.EXPOSURE_KEY


def test_lighting_args_scale_sky_densities_from_the_atmosphere():
    clean = beauty.lighting_args(atmosphere.Atmosphere(turbidity=1.0, rayleigh=1.5))["sky"]
    hazy = beauty.lighting_args(atmosphere.Atmosphere(turbidity=8.0, rayleigh=3.0))["sky"]
    assert clean["air_density"] == pytest.approx(1.0)
    assert hazy["air_density"] == pytest.approx(2.0)
    assert hazy["aerosol_density"] > clean["aerosol_density"]


def test_render_rejects_an_out_of_range_time_of_day(material_index, splat_result):
    with pytest.raises(atmosphere.AtmosphereError):
        beauty.render(cfg(), stack()["height"], splat_result, material_index, time_of_day_h=30.0)


def test_write_height_png_keeps_row_order_and_scales_to_16_bit(tmp_path):
    height = torch.zeros((4, 4), device=CPU)
    height[0, :] = 1.0
    path = beauty._write_height_png(height, tmp_path / "height.png")
    written = np.array(Image.open(path))
    assert written.dtype == np.uint16
    assert written[0, 0] == 65535
    assert written[-1, 0] == 0


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_render_produces_both_views_end_to_end(material_index, splat_result, shrub, tmp_path):
    result = beauty.render(
        cfg(),
        stack()["height"],
        splat_result,
        material_index,
        out_dir=tmp_path,
        placements=[shrub_placement(shrub, [[0.0, 0.0, 10.0, 0.0, 1.0, 0.0], [8.0, 8.0, 10.0, 1.0, 1.0, 0.0]])],
        variants={},
        grid_resolution=16,
        samples=8,
        resolution=(160, 90),
        timeout_s=180.0,
    )
    assert result.lighting["key"] in ("sun", "moon")
    assert result.scatter == {"scrub": 2}
    assert set(result.lighting["exposure_stops"]) == {"three_quarter", "ground"}
    assert result.three_quarter.is_file()
    assert result.ground.is_file()
    with Image.open(result.three_quarter) as image:
        assert image.size == (160, 90)
    with Image.open(result.ground) as image:
        assert image.size == (160, 90)
