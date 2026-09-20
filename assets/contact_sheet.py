from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from assets.blender import require_blender
from assets.recipe import validate_name
from terrain import config
from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, deliver_file

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_scripts" / "contact_sheet_render.py"

SHEET_NAME = "contact_sheet.png"

VIEWS = ("front", "back", "left", "right", "top", "bottom", "three_quarter", "worm_eye")
GRID_COLUMNS = 4

DEFAULT_VIEW_RESOLUTION = (512, 512)
DEFAULT_SAMPLES = 64
DEFAULT_HUMAN_HEIGHT_M = 1.8
DEFAULT_MARGIN = 1.15
DEFAULT_TIMEOUT_S = 240.0
DEFAULT_DEVICE = "OPTIX"

TILE_PADDING = 12
LABEL_HEIGHT = 26
SHEET_BACKGROUND = (24, 24, 28)
LABEL_COLOR = (232, 232, 236)
LOG_TAIL_CHARS = 4000


class ContactSheetError(RuntimeError):
    """Raised when the headless Blender contact-sheet render cannot be produced."""


@dataclass(frozen=True)
class ContactSheetRender:
    """The composed contact sheet, the eight source view renders, and what rendered them."""

    sheet: Path
    views: dict[str, Path]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    device: dict[str, Any]
    log: str


def _find_glb(name: str) -> Path:
    path = config.ASSET_OUT_DIR / name / f"{name}.glb"
    if not path.is_file():
        raise ContactSheetError(f"{path} does not exist")
    return path


def _tile_size(view_resolution: tuple[int, int], tile: int) -> tuple[int, int]:
    width, height = view_resolution
    scale = tile / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _compose_sheet(view_paths: dict[str, Path], tile: int) -> Image.Image:
    columns = min(GRID_COLUMNS, len(VIEWS))
    rows = math.ceil(len(VIEWS) / columns)
    cell_w = tile + TILE_PADDING
    cell_h = tile + LABEL_HEIGHT + TILE_PADDING
    sheet = Image.new(
        "RGB", (columns * cell_w + TILE_PADDING, rows * cell_h + TILE_PADDING), SHEET_BACKGROUND
    )
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(VIEWS):
        with Image.open(view_paths[view]) as opened:
            image = opened.convert("RGB")
            size = _tile_size(image.size, tile)
            image = image.resize(size, Image.LANCZOS)
        x = TILE_PADDING + (index % columns) * cell_w
        y = TILE_PADDING + (index // columns) * cell_h
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + tile + 2), view.replace("_", " "), fill=LABEL_COLOR)
    return sheet


def render(
    name: str,
    out_dir: Path | None = None,
    resolution: tuple[int, int] = DEFAULT_VIEW_RESOLUTION,
    tile: int | None = None,
    samples: int = DEFAULT_SAMPLES,
    human_height_m: float = DEFAULT_HUMAN_HEIGHT_M,
    margin: float = DEFAULT_MARGIN,
    device: str = DEFAULT_DEVICE,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ContactSheetRender:
    """Blender headless: import the exported glb, stand a 1.8 m scale reference beside it, and
    render 6 orthographic views plus a 3/4 beauty shot and a worm's-eye shot under flat, even
    lighting - one PNG per view, composed here into a single labelled contact sheet.

    This is how the agent sees the asset: it must expose missing back faces, flipped normals
    and one-angle-only detail.
    """
    validate_name(name)
    glb_path = _find_glb(name)
    if resolution[0] < 16 or resolution[1] < 16:
        raise ContactSheetError(f"resolution {tuple(resolution)} must be at least 16x16")
    if samples < 1:
        raise ContactSheetError(f"samples {samples} must be at least 1")
    if human_height_m <= 0.0:
        raise ContactSheetError(f"human_height_m {human_height_m} must be positive")
    if margin < 1.0:
        raise ContactSheetError(f"margin {margin} must be at least 1.0")

    target_dir = (config.ASSET_OUT_DIR / name) if out_dir is None else Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    executable = blender_executable or require_blender()

    with tempfile.TemporaryDirectory(prefix="contact_sheet_") as scratch_name:
        scratch = Path(scratch_name)
        view_outputs = {view: str(scratch / f"{view}.png") for view in VIEWS}
        result_json = scratch / "result.json"
        args = {
            "glb_path": str(glb_path),
            "views": list(VIEWS),
            "resolution": list(resolution),
            "samples": samples,
            "device": device,
            "human_height_m": human_height_m,
            "margin": margin,
            "outputs": view_outputs,
            "result_json": str(result_json),
        }
        args_path = scratch / "args.json"
        args_path.write_text(json.dumps(args), encoding="utf-8")

        completed = subprocess.run(
            [
                executable,
                "--background",
                "--factory-startup",
                "--python",
                str(SCRIPT_PATH),
                "--",
                str(args_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = completed.stdout + completed.stderr
        missing = [view for view, path in view_outputs.items() if not Path(path).is_file()]
        if completed.returncode != 0 or missing:
            raise ContactSheetError(
                f"Blender contact sheet render failed (exit {completed.returncode}, missing {missing}):\n"
                f"{log[-LOG_TAIL_CHARS:]}"
            )
        payload = json.loads(result_json.read_text(encoding="utf-8")) if result_json.is_file() else {}

        sheet_tile = tile or max(resolution)
        sheet_image = _compose_sheet({view: Path(path) for view, path in view_outputs.items()}, sheet_tile)
        sheet_path = target_dir / SHEET_NAME
        sheet_image.save(sheet_path, optimize=True)

        persisted_views: dict[str, Path] = {}
        for view, path in view_outputs.items():
            destination = target_dir / f"contact_sheet_{view}.png"
            destination.write_bytes(Path(path).read_bytes())
            persisted_views[view] = destination

    return ContactSheetRender(
        sheet=sheet_path,
        views=persisted_views,
        bbox_min=tuple(payload.get("bbox_min", (0.0, 0.0, 0.0))),
        bbox_max=tuple(payload.get("bbox_max", (0.0, 0.0, 0.0))),
        device=payload.get("device", {}),
        log=log,
    )


def contact_sheet_preview(
    name: str,
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> tuple[ContactSheetRender, Preview]:
    """Render the sheet and hand back a budgeted preview of it, agent-ready."""
    result = render(name, **kwargs)
    return result, deliver_file(result.sheet, budget)
