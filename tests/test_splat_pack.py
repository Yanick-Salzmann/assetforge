from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from library import materials
from terrain import channels as channel_module
from terrain import splat
from terrain.config import CHANNEL_NAMES, SPLAT_LAYERS_PER_TEXTURE, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 16

THREE_LAYERS = """
[biome]
name = "three"
sharpness = 2.0

[layer.silt_flat]
material = "silt"
weight = "1 - slope"

[layer.riverbed_gravel]
material = "river_rock"
weight = "flow"

[layer.exposed_rock]
material = "cliff_rock"
weight = "slope^2"
"""


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def biome(index) -> splat.Biome:
    return splat.parse_biome(THREE_LAYERS, index)


def eight_layers(index) -> splat.Biome:
    text = "".join(
        f"[layer.l{n}]\nmaterial = 'silt'\nweight = 'flow + {n * 0.1}'\n" for n in range(8)
    )
    return splat.parse_biome(text, index)


def stack() -> channel_module.ChannelStack:
    cfg = MapConfig(name="unit", resolution=SIZE, world_size_m=512.0, seed=3)
    built = channel_module.ChannelStack(cfg, CPU)
    generator = torch.Generator(device=CPU).manual_seed(11)
    for name in CHANNEL_NAMES:
        built[name] = torch.rand(cfg.shape, generator=generator, device=CPU)
    return built


def test_weights_are_stacked_in_declaration_order(biome):
    built = stack()
    raw = splat.weights(biome, built)
    assert tuple(raw.shape) == (SIZE, SIZE, 3)
    assert torch.allclose(raw[:, :, 1], built["flow"])
    assert float(raw.min()) >= 0.0


def test_blend_sums_to_one_per_pixel(biome):
    blended, silent = splat.blend(splat.weights(biome, stack()), biome.sharpness)
    assert torch.allclose(blended.sum(dim=-1), torch.ones((SIZE, SIZE)), atol=1e-5)
    assert not bool(silent.any())


def test_blend_keeps_a_zero_weight_at_zero():
    raw = torch.tensor([[[0.0, 0.5, 0.5]]])
    blended, _ = splat.blend(raw, 4.0)
    assert float(blended[0, 0, 0]) == 0.0
    assert float(blended[0, 0, 1]) == pytest.approx(0.5)


def test_sharpness_one_is_plain_normalisation():
    raw = torch.tensor([[[1.0, 3.0]]])
    blended, _ = splat.blend(raw, 1.0)
    assert float(blended[0, 0, 0]) == pytest.approx(0.25)


def variation_biome(index, strength: float, variation_m: float = 300.0) -> splat.Biome:
    text = (
        "[layer.a]\nmaterial = 'silt'\nweight = 'flow + 0.5'\n"
        f"variation = {strength}\nvariation_m = {variation_m}\n"
        "[layer.b]\nmaterial = 'river_rock'\nweight = 'flow + 0.5'\n"
    )
    return splat.parse_biome(text, index)


def test_zero_variation_leaves_the_weight_untouched(index):
    biome = variation_biome(index, 0.0)
    built = stack()
    raw = splat.weights(biome, built)
    assert torch.allclose(raw[:, :, 0], built["flow"] + 0.5)


def test_nonzero_variation_perturbs_the_weight(index):
    biome = variation_biome(index, 1.0)
    built = stack()
    raw = splat.weights(biome, built)
    plain = built["flow"] + 0.5
    assert not torch.allclose(raw[:, :, 0], plain)
    assert float(raw[:, :, 0].min()) >= 0.0
    assert torch.allclose(raw[:, :, 1], plain)


def test_variation_is_deterministic(index):
    biome = variation_biome(index, 0.6)
    built = stack()
    first = splat.weights(biome, built)
    second = splat.weights(biome, built)
    assert torch.equal(first, second)


def test_variation_differs_by_layer_name(index):
    text = (
        "[layer.a]\nmaterial = 'silt'\nweight = '0.5'\nvariation = 1.0\nvariation_m = 300.0\n"
        "[layer.b]\nmaterial = 'river_rock'\nweight = '0.5'\nvariation = 1.0\nvariation_m = 300.0\n"
    )
    biome = splat.parse_biome(text, index)
    built = stack()
    raw = splat.weights(biome, built)
    assert not torch.allclose(raw[:, :, 0], raw[:, :, 1])


def test_variation_requires_a_channel_stack_with_cfg(index):
    biome = variation_biome(index, 1.0)
    plain = {name: torch.full((SIZE, SIZE), 0.5) for name in CHANNEL_NAMES}
    with pytest.raises(MapConfigError, match="carries no cfg"):
        splat.weights(biome, plain)


def test_higher_sharpness_favours_the_leader():
    raw = torch.tensor([[[1.0, 2.0]]])
    soft, _ = splat.blend(raw, 1.0)
    hard, _ = splat.blend(raw, 8.0)
    assert float(hard[0, 0, 1]) > float(soft[0, 0, 1])
    assert float(hard[0, 0, 1]) > 0.99


def test_silent_pixels_fall_back_to_the_first_layer():
    raw = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    blended, silent = splat.blend(raw, 2.0)
    assert torch.equal(silent, torch.tensor([[True, False]]))
    assert torch.allclose(blended[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(blended[0, 1], torch.tensor([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("sharpness", [0.0, 1000.0])
def test_sharpness_outside_the_range_is_rejected(sharpness):
    with pytest.raises(MapConfigError):
        splat.blend(torch.ones((2, 2, 2)), sharpness)


def test_blend_rejects_a_wrong_shape():
    with pytest.raises(MapConfigError):
        splat.blend(torch.ones((2, 2)), 2.0)


def test_quantised_layers_sum_to_255_everywhere(biome):
    blended, _ = splat.blend(splat.weights(biome, stack()), biome.sharpness)
    quantised = splat.quantise(blended)
    assert quantised.dtype == torch.uint8
    assert torch.equal(
        quantised.to(torch.int32).sum(dim=-1),
        torch.full((SIZE, SIZE), 255, dtype=torch.int32),
    )


def test_quantisation_error_stays_within_one_step(biome):
    blended, _ = splat.blend(splat.weights(biome, stack()), biome.sharpness)
    quantised = splat.quantise(blended).to(torch.float32) / 255.0
    assert float((quantised - blended).abs().max()) <= 1.0 / 255.0 + 1e-6


def test_quantisation_is_exact_on_a_clean_split():
    blended = torch.tensor([[[0.2, 0.8]]])
    assert torch.equal(splat.quantise(blended), torch.tensor([[[51, 204]]], dtype=torch.uint8))


def test_quantisation_of_a_single_dominant_layer():
    blended = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    assert torch.equal(
        splat.quantise(blended), torch.tensor([[[255, 0, 0, 0]]], dtype=torch.uint8)
    )


def test_three_layers_pack_into_one_rgba_texture(biome):
    result = splat.render(biome, stack())
    assert len(result.textures) == 1
    texture = result.textures[0]
    assert texture.shape == (SIZE, SIZE, SPLAT_LAYERS_PER_TEXTURE)
    assert texture.dtype == np.uint8
    assert not texture[:, :, 3].any()


def test_eight_layers_pack_into_two_textures(index):
    result = splat.render(eight_layers(index), stack())
    assert len(result.textures) == 2
    total = sum(int(texture.astype(np.int32).sum()) for texture in result.textures)
    assert total == 255 * SIZE * SIZE


def test_packing_keeps_the_declaration_order(index):
    biome = eight_layers(index)
    result = splat.render(biome, stack())
    quantised = splat.quantise(result.weights).numpy()
    assert np.array_equal(result.textures[0], quantised[:, :, 0:4])
    assert np.array_equal(result.textures[1], quantised[:, :, 4:8])


def test_assignment_maps_layers_to_texture_and_channel(index):
    placed = splat.assignment(eight_layers(index))
    assert placed[0]["texture"] == "splat_0.png"
    assert placed[0]["channel"] == "r"
    assert placed[3]["channel"] == "a"
    assert placed[4]["texture"] == "splat_1.png"
    assert placed[4]["channel"] == "r"
    assert [entry["index"] for entry in placed] == list(range(8))


def test_assignment_carries_the_material_and_tiling(biome, index):
    placed = splat.assignment(biome)
    assert placed[1]["material"] == "river_rock"
    assert placed[1]["tiling_m"] == pytest.approx(index["river_rock"].tiling_m)


def test_coverage_sums_to_one_and_names_the_layers(biome):
    result = splat.render(biome, stack())
    assert [entry["layer"] for entry in result.coverage] == list(biome.names())
    assert sum(entry["mean"] for entry in result.coverage) == pytest.approx(1.0, abs=1e-5)
    assert sum(entry["dominant"] for entry in result.coverage) == pytest.approx(1.0)


def test_render_honours_a_sharpness_override(biome):
    built = stack()
    assert splat.render(biome, built).sharpness == pytest.approx(biome.sharpness)
    assert splat.render(biome, built, 9.0).sharpness == pytest.approx(9.0)


def test_render_is_deterministic(biome):
    built = stack()
    first = splat.render(biome, built)
    second = splat.render(biome, built)
    assert np.array_equal(first.textures[0], second.textures[0])
    assert first.as_dict() == second.as_dict()


def test_as_dict_describes_the_export(biome):
    data = splat.render(biome, stack()).as_dict()
    assert data["biome"] == "three"
    assert data["textures"] == ["splat_0.png"]
    assert len(data["layers"]) == 3
    assert data["silent_fraction"] == 0.0


def test_write_produces_readable_rgba_files(tmp_path, index):
    result = splat.render(eight_layers(index), stack())
    written = splat.write(result, tmp_path)
    assert [path.name for path in written] == ["splat_0.png", "splat_1.png"]
    for path, block in zip(written, result.textures):
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size == (SIZE, SIZE)
            assert np.array_equal(np.array(image), block)
