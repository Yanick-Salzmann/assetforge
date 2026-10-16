from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from assets import animation_manifest
from assets.blender import require_blender
from assets.recipe import validate_name
from terrain import config
from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, deliver_file

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_scripts" / "animation_sheet_render.py"
MANIFEST_NAME = "asset.json"

FRAME_COUNT = 8
TRAIL_LABEL = "motion trail"
DEFAULT_RESOLUTION = (384, 384)
DEFAULT_SAMPLES = 32
DEFAULT_MARGIN = 1.15
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_DEVICE = "OPTIX"

TILE_PADDING = 12
LABEL_HEIGHT = 26
SHEET_BACKGROUND = (24, 24, 28)
LABEL_COLOR = (232, 232, 236)
LOG_TAIL_CHARS = 4000


class AnimationSheetError(RuntimeError):
    """Raised when an animation filmstrip cannot be produced."""


@dataclass(frozen=True)
class AnimationSheet:
    sheet: Path
    clip: dict
    frames: list[int]
    device: dict[str, Any]
    log: str


def sheet_name(clip: str) -> str:
    return f"animation_{clip}.png"


def find_clip(name: str, clip: str) -> dict:
    manifest_path = config.ASSET_OUT_DIR / name / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AnimationSheetError(f"{manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        clips = animation_manifest.load(manifest)
    except animation_manifest.AnimationManifestError as error:
        raise AnimationSheetError(str(error)) from error
    for entry in clips:
        if entry["name"] == clip:
            return entry
    raise AnimationSheetError(f"{name} has no clip {clip!r}; it has {[entry['name'] for entry in clips]}")


def motion_trail(frames: list[Image.Image]) -> Image.Image:
    """Every frame laid over the first at equal weight: still parts stay sharp, moving parts
    smear along their path, so the swept volume reads at a glance."""
    trail = frames[0].convert("RGB")
    for index, frame in enumerate(frames[1:], start=2):
        trail = Image.blend(trail, frame.convert("RGB"), 1.0 / index)
    return trail


def moving_fraction(frames: list[Image.Image], threshold: int = 8) -> float:
    """Share of pixels that differ between the first frame and any later one."""
    first = frames[0].convert("L")
    moved = Image.new("L", first.size, 0)
    for frame in frames[1:]:
        moved = ImageChops.lighter(moved, ImageChops.difference(first, frame.convert("L")))
    mask = moved.point(lambda value: 255 if value > threshold else 0)
    return mask.histogram()[255] / (first.size[0] * first.size[1])


def compose(tiles: list[tuple[str, Image.Image]], tile: int) -> Image.Image:
    columns = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    cell_w = tile + TILE_PADDING
    cell_h = tile + LABEL_HEIGHT + TILE_PADDING
    sheet = Image.new("RGB", (columns * cell_w + TILE_PADDING, rows * cell_h + TILE_PADDING), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(tiles):
        scale = tile / max(image.size)
        resized = image.convert("RGB").resize(
            (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.LANCZOS
        )
        x = TILE_PADDING + (index % columns) * cell_w
        y = TILE_PADDING + (index // columns) * cell_h
        sheet.paste(resized, (x, y))
        draw.text((x + 2, y + tile + 2), label, fill=LABEL_COLOR)
    return sheet


def _load(path: str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def frame_label(frame: int, fps: int) -> str:
    return f"f{frame}  t={frame / fps:.2f}s"


def render(
    name: str,
    clip: str,
    trail: bool = True,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    samples: int = DEFAULT_SAMPLES,
    margin: float = DEFAULT_MARGIN,
    device: str = DEFAULT_DEVICE,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AnimationSheet:
    """Blender headless: play one clip of out/assets/<name>/<name>.glb and render one 3/4 view at
    FRAME_COUNT evenly spaced frames, composed with an optional motion-trail tile into
    out/assets/<name>/animation_<clip>.png. The agent reads this before calling motion good."""
    validate_name(name)
    entry = find_clip(name, clip)
    glb_path = config.ASSET_OUT_DIR / name / f"{name}.glb"
    if not glb_path.is_file():
        raise AnimationSheetError(f"{glb_path} does not exist")
    if resolution[0] < 16 or resolution[1] < 16:
        raise AnimationSheetError(f"resolution {tuple(resolution)} must be at least 16x16")
    executable = blender_executable or require_blender()

    with tempfile.TemporaryDirectory(prefix="animation_sheet_") as scratch_name:
        scratch = Path(scratch_name)
        outputs = [str(scratch / f"frame_{index}.png") for index in range(FRAME_COUNT)]
        result_json = scratch / "result.json"
        args = {
            "glb_path": str(glb_path),
            "clip": clip,
            "fps": entry["fps"],
            "frame_count": animation_manifest.frame_count(entry),
            "resolution": list(resolution),
            "samples": samples,
            "margin": margin,
            "device": device,
            "outputs": outputs,
            "result_json": str(result_json),
        }
        args_path = scratch / "args.json"
        args_path.write_text(json.dumps(args), encoding="utf-8")
        completed = subprocess.run(
            [executable, "--background", "--factory-startup", "--python", str(SCRIPT_PATH), "--", str(args_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = completed.stdout + completed.stderr
        missing = [path for path in outputs if not Path(path).is_file()]
        if completed.returncode != 0 or missing or not result_json.is_file():
            raise AnimationSheetError(
                f"Blender animation sheet render failed (exit {completed.returncode}, {len(missing)} frames missing):\n"
                f"{log[-LOG_TAIL_CHARS:]}"
            )
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        images = [_load(path) for path in outputs]

    frames = payload["frames"]
    tiles = [(frame_label(frame, entry["fps"]), image) for frame, image in zip(frames, images)]
    if trail:
        tiles.append((TRAIL_LABEL, motion_trail(images)))
    sheet_path = config.ASSET_OUT_DIR / name / sheet_name(clip)
    compose(tiles, max(resolution)).save(sheet_path, optimize=True)
    return AnimationSheet(sheet=sheet_path, clip=entry | {"moving_fraction": moving_fraction(images)}, frames=frames, device=payload["device"], log=log)


def animation_sheet_preview(
    name: str,
    clip: str,
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> tuple[AnimationSheet, Preview]:
    result = render(name, clip, **kwargs)
    return result, deliver_file(result.sheet, budget)
