from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from terrain import budget
from terrain.config import MapConfigError


def noise(width: int, height: int, seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def flat(width: int, height: int, value: int = 128) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8), "RGB")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_edge": 8},
        {"min_edge": 8},
        {"min_edge": 4096},
        {"max_bytes": 16},
        {"quality": 99},
        {"min_quality": 0},
        {"quality": 30, "min_quality": 60},
    ],
)
def test_budget_rejects_impossible_values(kwargs):
    with pytest.raises(MapConfigError):
        budget.PreviewBudget(**kwargs)


def test_fit_leaves_a_small_image_alone():
    small = flat(200, 120)
    assert budget.PreviewBudget(max_edge=768).fit(small) is small


def test_fit_caps_the_long_edge_and_keeps_the_aspect():
    fitted = budget.PreviewBudget(max_edge=256).fit(flat(2048, 1024))
    assert fitted.size == (256, 128)


def test_encode_stays_under_the_byte_ceiling():
    limit = budget.PreviewBudget(max_edge=512, max_bytes=40_000)
    payload, shown = limit.encode(noise(2048, 2048))
    assert len(payload) <= limit.max_bytes
    assert max(shown.size) <= limit.max_edge
    assert payload[:2] == b"\xff\xd8"


def test_encode_drops_pixels_only_when_quality_is_not_enough():
    tight = budget.PreviewBudget(max_edge=512, max_bytes=6_000, min_edge=64)
    _, shown = tight.encode(noise(1024, 1024))
    assert max(shown.size) < 512
    _, roomy = budget.PreviewBudget(max_edge=512, max_bytes=400_000).encode(noise(1024, 1024))
    assert max(roomy.size) == 512


def test_encode_gives_up_at_the_minimum_edge():
    impossible = budget.PreviewBudget(max_edge=128, max_bytes=1024, min_edge=128, min_quality=40)
    payload, shown = impossible.encode(noise(512, 512))
    assert max(shown.size) == 128
    assert len(payload) > 0


def test_deliver_reports_the_source_size_and_token_cost(tmp_path):
    preview = budget.deliver(noise(2048, 2048), tmp_path / "map.png")
    assert (preview.source_width, preview.source_height) == (2048, 2048)
    assert preview.width == budget.DEFAULT_MAX_EDGE
    assert preview.downscaled
    assert preview.estimated_tokens == budget.DEFAULT_BUDGET.max_tokens
    assert preview.as_dict()["path"].endswith("map.png")
    assert preview.as_base64()


def test_deliver_crops_to_a_fractional_region():
    preview = budget.deliver(noise(1000, 800), region=(0.25, 0.5, 0.75, 1.0))
    assert preview.region == (250, 400, 750, 800)
    assert (preview.source_width, preview.source_height) == (500, 400)
    assert not preview.downscaled
    assert preview.path is None


def test_a_crop_costs_far_less_than_the_whole_map():
    whole = budget.deliver(noise(2048, 2048))
    crop = budget.deliver(noise(2048, 2048), region=budget.detail_region((0.5, 0.5), 0.2))
    assert crop.estimated_tokens < whole.estimated_tokens


@pytest.mark.parametrize(
    "region",
    [(0.0, 0.0, 0.0, 1.0), (0.2, 0.3, 0.1, 0.9), (-0.1, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0)],
)
def test_region_box_rejects_an_impossible_region(region):
    with pytest.raises(MapConfigError):
        budget.region_box((256, 256), region)


def test_region_box_is_none_without_a_region():
    assert budget.region_box((256, 256)) is None


def test_detail_region_stays_on_the_map():
    assert budget.detail_region((0.0, 1.0), 0.4) == pytest.approx((0.0, 0.6, 0.4, 1.0))
    assert budget.detail_region((0.5, 0.5), 1.0) == pytest.approx((0.0, 0.0, 1.0, 1.0))


@pytest.mark.parametrize("kwargs", [{"span": 0.0}, {"span": 1.5}, {"centre": (0.5,)}])
def test_detail_region_rejects_impossible_values(kwargs):
    with pytest.raises(MapConfigError):
        budget.detail_region(**kwargs)


def test_deliver_file_reads_an_image_off_disk(tmp_path):
    path = tmp_path / "render.png"
    noise(900, 600).save(path)
    preview = budget.deliver_file(path)
    assert preview.path == path
    assert (preview.source_width, preview.source_height) == (900, 600)
    assert preview.media_type == budget.JPEG_MEDIA_TYPE


def test_deliver_file_rejects_a_missing_image(tmp_path):
    with pytest.raises(MapConfigError):
        budget.deliver_file(tmp_path / "absent.png")


def test_deliver_flattens_an_alpha_image():
    preview = budget.deliver(Image.new("RGBA", (64, 64), (10, 20, 30, 128)))
    assert preview.width == 64
