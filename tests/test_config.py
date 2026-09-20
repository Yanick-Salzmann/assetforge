from __future__ import annotations

import math
from pathlib import Path

import pytest

from terrain import config
from terrain.config import MapConfig, MapConfigError


def test_repo_root_contains_pyproject():
    assert (config.REPO_ROOT / "pyproject.toml").is_file()


def test_metres_per_pixel():
    cfg = MapConfig(name="unit", resolution=1024, world_size_m=2048.0)
    assert cfg.metres_per_pixel == 2.0
    assert cfg.pixels_per_metre == 0.5


def test_out_dir_is_under_out_terrain():
    cfg = MapConfig(name="unit")
    assert cfg.out_dir() == config.TERRAIN_OUT_DIR / "unit"
    assert isinstance(cfg.out_dir(), Path)


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("ASSETFORGE_DEVICE", "cpu")
    assert config.device() == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": 1000},
        {"resolution": 8192},
        {"resolution": 1},
        {"world_size_m": 0.0},
        {"height_range_m": -1.0},
        {"seed": -1},
        {"name": ""},
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    base = {"name": "unit"}
    base.update(kwargs)
    with pytest.raises(MapConfigError):
        MapConfig(**base)


def test_height_metre_round_trip():
    cfg = MapConfig(name="unit", height_range_m=600.0)
    assert cfg.height_to_metres(0.5) == 300.0
    assert cfg.metres_to_height(300.0) == 0.5
    assert MapConfig(name="unit", height_range_m=600.0, sea_level_m=60.0).sea_level_normalised == 0.1


def test_slope_is_a_world_gradient():
    cfg = MapConfig(name="unit", resolution=1024, world_size_m=1024.0, height_range_m=100.0)
    assert cfg.slope_scale() == 100.0
    assert cfg.slope_from_normalised(0.01) == pytest.approx(1.0)
    assert cfg.talus_angle_deg(0.01) == pytest.approx(45.0)


def test_talus_angle_survives_a_resolution_change():
    coarse = MapConfig(name="unit", resolution=1024, world_size_m=4096.0, height_range_m=600.0)
    fine = coarse.at_resolution(4096)
    angle = 37.0
    assert fine.talus_delta(angle) == pytest.approx(coarse.talus_delta(angle) / 4.0)
    for cfg in (coarse, fine):
        assert cfg.talus_angle_deg(cfg.talus_delta(angle)) == pytest.approx(angle)


def test_slope_survives_a_resolution_change():
    coarse = MapConfig(name="unit", resolution=512, world_size_m=4096.0, height_range_m=600.0)
    fine = coarse.at_resolution(2048)
    rise_m = 30.0
    run_px_coarse = 4
    run_px_fine = run_px_coarse * (fine.resolution // coarse.resolution)
    delta_coarse = coarse.metres_to_height(rise_m) / run_px_coarse
    delta_fine = fine.metres_to_height(rise_m) / run_px_fine
    assert coarse.slope_from_normalised(delta_coarse) == pytest.approx(
        fine.slope_from_normalised(delta_fine)
    )


def test_tiling_scale_is_resolution_independent():
    coarse = MapConfig(name="unit", resolution=1024, world_size_m=4096.0)
    fine = coarse.at_resolution(4096)
    assert coarse.tiling_scale(8.0) == fine.tiling_scale(8.0) == 512.0
    assert coarse.texture_size_m(512.0) == 8.0


def test_tiling_scale_tracks_world_size():
    small = MapConfig(name="unit", world_size_m=1024.0)
    large = MapConfig(name="unit", world_size_m=4096.0)
    assert large.tiling_scale(4.0) == small.tiling_scale(4.0) * 4.0


def test_channel_stack_bytes_matches_the_measured_ceiling():
    assert MapConfig(name="unit", resolution=2048).channel_stack_bytes() == 256 * 1024 * 1024
    assert MapConfig(name="unit", resolution=4096).channel_stack_bytes() == 1024 * 1024 * 1024


def test_round_trips_through_a_dict():
    cfg = MapConfig(name="unit", resolution=512, world_size_m=1024.0, height_range_m=250.0, sea_level_m=12.5, seed=7)
    data = cfg.as_dict()
    assert data["metres_per_pixel"] == 2.0
    assert MapConfig.from_dict(data) == cfg


def test_derived_seeds_are_deterministic_and_distinct():
    cfg = MapConfig(name="unit", seed=1234)
    assert cfg.derive_seed("erosion") == cfg.derive_seed("erosion")
    assert cfg.derive_seed("erosion") != cfg.derive_seed("moisture")
    assert cfg.derive_seed("warp", 3) != cfg.derive_seed("warp", 4)
    assert MapConfig(name="unit", seed=1235).derive_seed("erosion") != cfg.derive_seed("erosion")


def test_bad_talus_and_tiling_inputs_are_rejected():
    cfg = MapConfig(name="unit")
    with pytest.raises(MapConfigError):
        cfg.talus_delta(0.0)
    with pytest.raises(MapConfigError):
        cfg.talus_delta(90.0)
    with pytest.raises(MapConfigError):
        cfg.tiling_scale(0.0)


def test_channel_names_are_the_documented_sixteen():
    assert len(config.CHANNEL_NAMES) == 16
    assert config.CHANNEL_NAMES[0] == "height"
    assert all(name == name.lower() for name in config.CHANNEL_NAMES)
    assert math.isclose(config.HEIGHTMAP_MAX, 65535)
