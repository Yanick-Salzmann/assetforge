from __future__ import annotations

import json

import pytest

Client = pytest.importorskip("fastmcp").Client

from assets import blender as blender_discovery
from assets import gate
from assets.server import mcp
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    gate._GATES.clear()
    yield
    gate._GATES.clear()


async def call(tool: str, arguments: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    texts = [block for block in result.content if block.type == "text"]
    return json.loads(texts[0].text)


async def call_with_images(tool: str, arguments: dict) -> tuple[dict, list]:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    texts = [block for block in result.content if block.type == "text"]
    images = [block for block in result.content if block.type == "image"]
    return json.loads(texts[0].text), images


@pytest.mark.anyio
async def test_record_patch_counts_toward_the_cap():
    status = await call("record_patch", {"name": "hero-crate"})
    assert status["patch_count"] == 1
    assert status["patches_remaining"] == gate.MAX_PATCH_ITERATIONS - 1


@pytest.mark.anyio
async def test_a_fifth_patch_is_refused():
    for _ in range(gate.MAX_PATCH_ITERATIONS):
        await call("record_patch", {"name": "hero-crate"})
    with pytest.raises(Exception, match="4 allowed patch"):
        await call("record_patch", {"name": "hero-crate"})


@pytest.mark.anyio
async def test_approve_asset_records_approval():
    status = await call("approve_asset", {"name": "prop-barrel", "note": "ship it"})
    assert status["approved"] is True
    assert status["approval_note"] == "ship it"


@pytest.mark.anyio
async def test_gate_status_reflects_persisted_state():
    await call("record_patch", {"name": "prop-crate"})
    await call("approve_asset", {"name": "prop-crate"})
    status = await call("asset_gate_status", {"name": "prop-crate"})
    assert status["patch_count"] == 1
    assert status["approved"] is True


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
@pytest.mark.anyio
async def test_render_contact_sheet_renders_every_view():
    from tests.test_contact_sheet import _write_tetrahedron_glb

    _write_tetrahedron_glb(config.ASSET_OUT_DIR / "test-prop" / "test-prop.glb")

    summary, images = await call_with_images("render_contact_sheet", {"name": "test-prop"})

    assert summary["name"] == "test-prop"
    assert set(summary["views"]) == {
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
        "three_quarter",
        "worm_eye",
    }
    assert len(images) == 1
