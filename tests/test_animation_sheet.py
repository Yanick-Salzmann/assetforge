from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
Client = pytest.importorskip("fastmcp").Client

from assets import animation_manifest
from assets import animation_sheet
from assets import blender as blender_discovery
from assets.server import mcp
from terrain import config

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SELFTEST = REPO_ROOT / "assets" / "blender_scripts" / "export_selftest.py"


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path / "out")
    yield


def _write_manifest(name: str, animations: list) -> None:
    directory = config.ASSET_OUT_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / animation_sheet.MANIFEST_NAME).write_text(json.dumps({"name": name, "animations": animations}), encoding="utf-8")


def test_find_clip_returns_the_manifest_entry():
    entry = animation_manifest.manifest_entry("idle", "spin", "rigid", 2.0, 24, ["Rotor_ANIM"])
    _write_manifest("mill", [entry])
    assert animation_sheet.find_clip("mill", "idle_spin") == entry


def test_find_clip_names_the_clips_it_has():
    entry = animation_manifest.manifest_entry("idle", "spin", "rigid", 2.0, 24, ["Rotor_ANIM"])
    _write_manifest("mill", [entry])
    with pytest.raises(animation_sheet.AnimationSheetError, match="idle_spin"):
        animation_sheet.find_clip("mill", "working_grind")


def test_find_clip_rejects_a_static_asset():
    _write_manifest("crate", [])
    with pytest.raises(animation_sheet.AnimationSheetError, match="no clip"):
        animation_sheet.find_clip("crate", "idle_spin")


def test_find_clip_rejects_a_missing_manifest():
    with pytest.raises(animation_sheet.AnimationSheetError, match="does not exist"):
        animation_sheet.find_clip("nothing", "idle_spin")


def test_render_rejects_a_missing_glb():
    entry = animation_manifest.manifest_entry("idle", "spin", "rigid", 2.0, 24, ["Rotor_ANIM"])
    _write_manifest("mill", [entry])
    with pytest.raises(animation_sheet.AnimationSheetError, match="glb does not exist"):
        animation_sheet.render("mill", "idle_spin")


def test_render_rejects_an_unknown_focus():
    with pytest.raises(animation_sheet.AnimationSheetError, match="focus"):
        animation_sheet.render("mill", "idle_spin", focus="roof")


def test_compose_lays_eight_frames_and_a_trail_on_a_three_by_three_grid():
    colours = [(index * 28, 255 - index * 28, 90) for index in range(9)]
    tiles = [(str(index), Image.new("RGB", (32, 32), colour)) for index, colour in enumerate(colours)]
    sheet = animation_sheet.compose(tiles, tile=32)
    cell_w = 32 + animation_sheet.TILE_PADDING
    cell_h = 32 + animation_sheet.LABEL_HEIGHT + animation_sheet.TILE_PADDING
    for index, colour in enumerate(colours):
        x = animation_sheet.TILE_PADDING + (index % 3) * cell_w + 16
        y = animation_sheet.TILE_PADDING + (index // 3) * cell_h + 16
        assert sheet.getpixel((x, y)) == colour
    assert sheet.size == (3 * cell_w + animation_sheet.TILE_PADDING, 3 * cell_h + animation_sheet.TILE_PADDING)


def _frames_with_moving_square(count: int) -> list:
    frames = []
    for index in range(count):
        frame = Image.new("RGB", (64, 64), (0, 0, 0))
        frame.paste((255, 255, 255), (index * 4, 0, index * 4 + 8, 8))
        frames.append(frame)
    return frames


def test_motion_trail_weights_every_frame_equally():
    frames = _frames_with_moving_square(4)
    trail = animation_sheet.motion_trail(frames)
    assert trail.getpixel((1, 1))[0] == pytest.approx(255 / 4, abs=2)
    assert trail.getpixel((60, 60)) == (0, 0, 0)


def test_moving_fraction_is_zero_for_a_still_clip_and_counts_swept_pixels():
    still = [Image.new("RGB", (64, 64), (40, 40, 40)) for _ in range(8)]
    assert animation_sheet.moving_fraction(still) == 0.0
    moving = _frames_with_moving_square(8)
    assert animation_sheet.moving_fraction(moving) == pytest.approx((8 + 7 * 4) * 8 / (64 * 64))


def test_frame_label_shows_frame_and_time():
    assert animation_sheet.frame_label(12, 24) == "f12  t=0.50s"


async def _call(tool: str, arguments: dict) -> tuple[dict, list]:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    texts = [block for block in result.content if block.type == "text"]
    images = [block for block in result.content if block.type == "image"]
    return json.loads(texts[0].text), images


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "clip"),
    [("export-selftest-rigid", "idle_rotor_spin"), ("export-selftest-skinned", "idle_strip_wave")],
)
async def test_render_animation_sheet_returns_a_budgeted_filmstrip(tmp_path, name, clip):
    result_path = tmp_path / "export.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(EXPORT_SELFTEST),
            "--",
            str(REPO_ROOT),
            str(config.ASSET_OUT_DIR),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert json.loads(result_path.read_text(encoding="utf-8")).get("ok"), completed.stdout + completed.stderr

    summary, images = await _call("render_animation_sheet", {"name": name, "clip": clip})

    assert summary["clip"]["name"] == clip
    assert len(summary["frames"]) == animation_sheet.FRAME_COUNT
    assert summary["frames"][0] == 0
    assert summary["clip"]["moving_fraction"] > 0.001
    assert Path(summary["sheet_path"]).is_file()
    assert len(images) == 1
