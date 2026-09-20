from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from scripts import regen_diff
from terrain import config
from terrain import session as session_mod

RESOLUTION = 64
FAST_EROSION = {"iterations": 8, "settle_iterations": 2, "thermal_settle_passes": 2}
BIOME_NAME = "temperate"
NAME = "regen-probe"


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TERRAIN_OUT_DIR", tmp_path / "terrain")
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path / "assets")
    for name in session_mod.resident():
        session_mod.forget(name)
    yield
    for name in session_mod.resident():
        session_mod.forget(name)


def _build_and_export(name: str = NAME, seed: int = 5) -> None:
    from terrain import export

    session = session_mod.create(name=name, seed=seed, resolution=RESOLUTION, world_size_m=256.0)
    session.erosion_overrides = dict(FAST_EROSION)
    session.erode()
    session.build_channels()
    session.apply_biome(BIOME_NAME)
    export.write(session.cfg, session.channels(), session.water, session.splat_result())


def test_diff_terrain_reports_no_drift_for_an_untouched_export():
    _build_and_export()
    drift = regen_diff.diff_terrain(NAME)
    assert drift.ok, drift.as_dict()


def test_diff_terrain_reports_drift_when_a_stored_file_is_tampered_with():
    _build_and_export()
    height_path = config.TERRAIN_OUT_DIR / NAME / "height.png"
    original = height_path.read_bytes()
    height_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))

    drift = regen_diff.diff_terrain(NAME)
    assert not drift.ok
    assert "height.png" in drift.file_drift


def test_diff_terrain_reports_an_error_when_no_manifest_exists():
    session_mod.create(name=NAME, seed=5, resolution=RESOLUTION, world_size_m=256.0)
    drift = regen_diff.diff_terrain(NAME)
    assert not drift.ok
    assert drift.error is not None


def test_list_terrains_and_assets_reflect_disk_state():
    _build_and_export()
    assert NAME in regen_diff.list_terrains()
    assert regen_diff.list_assets() == []


def test_regenerate_asset_reports_missing_recipe():
    (config.ASSET_OUT_DIR / "prop-crate").mkdir(parents=True)
    result = regen_diff.regenerate_asset("prop-crate")
    assert result["ok"] is False
    assert "no recipe.py" in result["error"]


def test_run_skips_terrain_and_assets_when_asked():
    report = regen_diff.run([], [], skip_terrain=True, skip_assets=True)
    assert report == {"terrain": [], "assets": []}
