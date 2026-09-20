from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from terrain import config, session
from terrain.config import MapConfigError

RESOLUTION = 64
FAST_EROSION = {"iterations": 8, "settle_iterations": 2, "thermal_settle_passes": 2}


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TERRAIN_OUT_DIR", tmp_path / "terrain")
    for name in session.resident():
        session.forget(name)
    yield
    for name in session.resident():
        session.forget(name)


def make(name: str = "probe", **kwargs) -> session.TerrainSession:
    kwargs.setdefault("resolution", RESOLUTION)
    kwargs.setdefault("world_size_m", 256.0)
    return session.create(name, **kwargs)


@pytest.mark.parametrize("name", ["", "Probe", "../escape", "a/b", "probe.name", "x" * 65])
def test_bad_names_are_refused(name):
    with pytest.raises(MapConfigError):
        session.validate_name(name)


def test_create_writes_a_manifest_and_a_height_cache():
    current = make(seed=3)
    target = current.out_dir()
    assert current.stage == "synth"
    assert (target / session.SESSION_NAME).is_file()
    assert (target / session.HEIGHT_CACHE).is_file()
    payload = json.loads((target / session.SESSION_NAME).read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 3
    assert payload["stage"] == "synth"


def test_create_refuses_to_clobber_without_overwrite():
    make()
    with pytest.raises(MapConfigError):
        make()
    assert make(overwrite=True).stage == "synth"


def test_create_rejects_an_unknown_shape():
    with pytest.raises(MapConfigError):
        make(shape="nowhere")


def test_erosion_moves_the_height_and_records_its_stats():
    current = make()
    base = current.base.clone()
    current.erosion_overrides = dict(FAST_EROSION)
    current.erode()
    assert current.stage == "eroded"
    assert not current.height.cpu().equal(base.cpu())
    assert current.manifest()["erosion_stats"]["iterations"] == float(FAST_EROSION["iterations"])


def test_channels_fill_the_whole_stack_in_range():
    current = make()
    current.erosion_overrides = dict(FAST_EROSION)
    current.build_channels()
    stack = current.channels()
    assert stack.missing() == ()
    assert set(stack.filled()) == set(config.CHANNEL_NAMES)
    assert float(stack.data.min()) >= 0.0
    assert float(stack.data.max()) <= 1.0
    assert current.water is not None
    assert 0.0 <= current.water.covered_fraction <= 1.0


def test_channel_stats_cover_every_filled_channel():
    current = make()
    current.erosion_overrides = dict(FAST_EROSION)
    current.build_channels()
    stats = session.channel_stats(current)
    assert set(stats) == set(config.CHANNEL_NAMES)
    assert all(row["min"] <= row["mean"] <= row["max"] for row in stats.values())


def test_reload_restores_the_height_without_replaying_synthesis():
    current = make(seed=11)
    height = current.current().cpu().clone()
    session.forget(current.name)
    reloaded = session.open_session(current.name)
    assert reloaded is not current
    assert reloaded.cfg == current.cfg
    assert reloaded.current().cpu().equal(height)


def test_reload_replays_erosion_deterministically():
    current = make(seed=5)
    current.erosion_overrides = dict(FAST_EROSION)
    current.erode()
    eroded = current.height.cpu().clone()
    session.forget(current.name)
    reloaded = session.open_session(current.name)
    assert reloaded.erosion_overrides == FAST_EROSION
    assert reloaded.eroded().height.cpu().equal(eroded)


def test_open_refuses_an_unknown_terrain():
    with pytest.raises(MapConfigError):
        session.open_session("missing")


def test_overrides_reject_unknown_keys():
    assert session.overrides(None, session.EROSION_FIELDS, "erosion params") == {}
    assert session.overrides({"iterations": 4}, session.EROSION_FIELDS, "erosion params") == {
        "iterations": 4
    }
    with pytest.raises(MapConfigError):
        session.overrides({"itterations": 4}, session.EROSION_FIELDS, "erosion params")


def test_hillshade_preview_stays_inside_the_budget():
    current = make()
    view = session.hillshade(current)
    assert view.path.is_file()
    assert view.estimated_tokens <= session.DEFAULT_BUDGET.max_tokens
    assert len(view.data) <= session.DEFAULT_BUDGET.max_bytes


def test_a_crop_costs_less_than_the_whole_map():
    current = make(resolution=256, world_size_m=1024.0)
    whole = session.hillshade(current)
    crop = session.hillshade(current, centre=[0.5, 0.5], span=0.25)
    assert crop.region == (96, 96, 160, 160)
    assert crop.estimated_tokens < whole.estimated_tokens


def test_channel_view_reads_height_without_building_the_stack():
    current = make()
    view, stats = session.channel_view(current, "height")
    assert current.stack is None
    assert view.path.is_file()
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_channel_view_rejects_an_unknown_channel():
    current = make()
    current.erosion_overrides = dict(FAST_EROSION)
    with pytest.raises(MapConfigError):
        session.channel_view(current, "elevation")


def test_stored_lists_only_directories_with_a_manifest():
    make("alpha")
    make("beta")
    (config.TERRAIN_OUT_DIR / "stray").mkdir(parents=True)
    assert [name for name, _ in session.stored()] == ["alpha", "beta"]
