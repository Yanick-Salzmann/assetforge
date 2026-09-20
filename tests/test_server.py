from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL.Image")
Client = pytest.importorskip("fastmcp").Client

from assets import blender as blender_discovery
from terrain import config, session
from terrain.server import mcp

RESOLUTION = 64
FAST_EROSION = {"iterations": 8, "settle_iterations": 2, "thermal_settle_passes": 2}
BIOME_NAME = "temperate"

NEW_TERRAIN = {
    "name": "probe",
    "seed": 3,
    "resolution": RESOLUTION,
    "world_size_m": 256.0,
    "shape": "island",
}


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TERRAIN_OUT_DIR", tmp_path / "terrain")
    for name in session.resident():
        session.forget(name)
    yield
    for name in session.resident():
        session.forget(name)


async def call(tool: str, arguments: dict) -> tuple[dict, list]:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    texts = [block for block in result.content if block.type == "text"]
    images = [block for block in result.content if block.type == "image"]
    return json.loads(texts[0].text), images


async def terrain(name: str = "probe", **overrides) -> dict:
    payload = dict(NEW_TERRAIN) | overrides | {"name": name}
    summary, _ = await call("new_terrain", payload)
    return summary


@pytest.mark.anyio
async def test_status_lists_the_presets_and_parameters_the_other_tools_take():
    async with Client(mcp) as client:
        result = await client.call_tool("terrain_status", {})
    status = result.structured_content
    assert status["server"] == "terrain-forge"
    assert "continental" in status["shape_presets"]
    assert "canyon" in status["erosion_presets"]
    assert "iterations" in status["erosion_params"]
    assert "river_depth_m" in status["water_params"]
    assert status["channels"] == list(config.CHANNEL_NAMES)


@pytest.mark.anyio
async def test_new_terrain_returns_a_summary_and_one_jpeg():
    summary, images = await call("new_terrain", dict(NEW_TERRAIN))
    assert summary["stage"] == "synth"
    assert summary["config"]["seed"] == NEW_TERRAIN["seed"]
    assert len(images) == 1
    assert images[0].mime_type == "image/jpeg"
    assert summary["preview"]["path"].endswith("preview_hillshade.png")
    assert summary["preview"]["estimated_tokens"] <= session.DEFAULT_BUDGET.max_tokens


@pytest.mark.anyio
async def test_new_terrain_refuses_a_name_that_is_not_a_directory():
    with pytest.raises(Exception, match="terrain name"):
        await terrain("../escape")


@pytest.mark.anyio
async def test_run_erosion_records_the_resolved_parameters():
    await terrain()
    summary, images = await call(
        "run_erosion", {"name": "probe", "preset": "light", "params": FAST_EROSION}
    )
    assert summary["stage"] == "eroded"
    assert summary["erosion"] == "light"
    assert summary["params"]["iterations"] == FAST_EROSION["iterations"]
    assert summary["params"]["sediment_capacity"] == 0.5
    assert summary["erosion_stats"]["eroded_max_m"] > 0.0
    assert len(images) == 1


@pytest.mark.anyio
async def test_run_erosion_rejects_an_unknown_override():
    await terrain()
    with pytest.raises(Exception, match="unknown erosion params"):
        await call("run_erosion", {"name": "probe", "params": {"itterations": 4}})


@pytest.mark.anyio
async def test_build_channels_fills_the_stack_and_reports_every_range():
    await terrain()
    await call("run_erosion", {"name": "probe", "params": FAST_EROSION})
    summary, images = await call("build_channels", {"name": "probe"})
    assert summary["stage"] == "channels"
    assert summary["channels"] == list(config.CHANNEL_NAMES)
    assert set(summary["channel_stats"]) == set(config.CHANNEL_NAMES)
    assert summary["stack_bytes"] == RESOLUTION * RESOLUTION * len(config.CHANNEL_NAMES) * 4
    assert 0.0 <= summary["water"]["covered_fraction"] <= 1.0
    assert len(images) == 1


@pytest.mark.anyio
async def test_build_channels_replays_erosion_when_the_server_has_restarted():
    await terrain()
    await call("run_erosion", {"name": "probe", "params": FAST_EROSION})
    session.forget("probe")
    summary, _ = await call("build_channels", {"name": "probe"})
    assert summary["erosion_overrides"] == FAST_EROSION
    assert summary["channels"] == list(config.CHANNEL_NAMES)


@pytest.mark.anyio
async def test_preview_hillshade_crops_to_a_centre_and_span():
    await terrain(resolution=256, world_size_m=1024.0)
    whole, _ = await call("preview_hillshade", {"name": "probe"})
    crop, images = await call(
        "preview_hillshade", {"name": "probe", "centre": [0.25, 0.25], "span": 0.25}
    )
    assert crop["preview"]["region"] == [32, 32, 96, 96]
    assert crop["preview"]["estimated_tokens"] < whole["preview"]["estimated_tokens"]
    assert len(images) == 1


@pytest.mark.anyio
async def test_inspect_channel_reports_the_range_it_rendered():
    await terrain()
    await call("run_erosion", {"name": "probe", "params": FAST_EROSION})
    summary, images = await call("inspect_channel", {"name": "probe", "channel": "flow"})
    assert summary["channel"] == "flow"
    assert summary["range"]["min"] <= summary["range"]["mean"] <= summary["range"]["max"]
    assert summary["preview"]["path"].endswith("channel_flow.png")
    assert len(images) == 1


@pytest.mark.anyio
async def test_inspect_channel_rejects_an_unknown_channel():
    await terrain()
    with pytest.raises(Exception, match="unknown channel"):
        await call("inspect_channel", {"name": "probe", "channel": "elevation"})


@pytest.mark.anyio
async def test_apply_rules_reports_coverage_and_returns_the_composite():
    await terrain()
    await call("run_erosion", {"name": "probe", "params": FAST_EROSION})
    await call("build_channels", {"name": "probe"})
    summary, images = await call("apply_rules", {"name": "probe", "biome_file": BIOME_NAME})
    assert summary["stage"] == "splat"
    assert summary["splat"]["biome"] == BIOME_NAME
    assert "riverbed_gravel" in summary["coverage_table"]
    assert len(images) == 1


@pytest.mark.anyio
async def test_apply_rules_rejects_an_unknown_biome():
    await terrain()
    await call("build_channels", {"name": "probe"})
    with pytest.raises(Exception, match="unknown biome"):
        await call("apply_rules", {"name": "probe", "biome_file": "does-not-exist"})


@pytest.mark.anyio
async def test_preview_splat_requires_a_biome_first():
    await terrain()
    await call("build_channels", {"name": "probe"})
    with pytest.raises(Exception, match="no biome applied"):
        await call("preview_splat", {"name": "probe"})


@pytest.mark.anyio
async def test_preview_splat_replays_after_the_server_restarts():
    await terrain()
    await call("build_channels", {"name": "probe"})
    await call("apply_rules", {"name": "probe", "biome_file": BIOME_NAME})
    session.forget("probe")
    summary, images = await call("preview_splat", {"name": "probe"})
    assert summary["biome_file"] == BIOME_NAME
    assert len(images) == 1


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
@pytest.mark.anyio
async def test_preview_beauty_renders_the_requested_view():
    await terrain()
    await call("run_erosion", {"name": "probe", "params": FAST_EROSION})
    await call("build_channels", {"name": "probe"})
    await call("apply_rules", {"name": "probe", "biome_file": BIOME_NAME})
    summary, images = await call(
        "preview_beauty", {"name": "probe", "view": "ground", "samples": 8}
    )
    assert summary["view"] == "ground"
    assert len(images) == 1


@pytest.mark.anyio
async def test_list_terrains_reports_the_stage_of_each_map():
    await terrain("alpha")
    await terrain("beta")
    async with Client(mcp) as client:
        result = await client.call_tool("list_terrains", {})
    entries = {entry["name"]: entry for entry in result.structured_content["result"]}
    assert set(entries) == {"alpha", "beta"}
    assert entries["alpha"]["stage"] == "synth"
    assert entries["beta"]["resident"] is True
