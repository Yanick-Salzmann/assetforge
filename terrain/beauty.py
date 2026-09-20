from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from assets.blender import require_blender
from library.materials import MaterialIndex
from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, deliver_file
from terrain.config import HEIGHTMAP_MAX, MapConfig, MapConfigError
from terrain.preview import DEFAULT_ALTITUDE_DEG, DEFAULT_AZIMUTH_DEG
from terrain.splat import SplatResult

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_scripts" / "beauty_render.py"

THREE_QUARTER_NAME = "preview_beauty_3q.png"
GROUND_NAME = "preview_beauty_ground.png"

DEFAULT_GRID_RESOLUTION = 512
DEFAULT_PATCH_SIZE_M = 200.0
DEFAULT_SAMPLES = 128
DEFAULT_RESOLUTION = (960, 540)
DEFAULT_TIMEOUT_S = 240.0
DEFAULT_THREE_QUARTER_AZIMUTH_DEG = 45.0
DEFAULT_THREE_QUARTER_ALTITUDE_DEG = 35.0
DEFAULT_GROUND_EYE_HEIGHT_M = 1.8
DEFAULT_DEVICE = "OPTIX"
LOG_TAIL_CHARS = 4000


class BeautyRenderError(MapConfigError):
    """Raised when the headless Blender beauty render cannot be produced."""


@dataclass(frozen=True)
class BeautyRender:
    """The two rendered views, where the camera ended up, and which compute device rendered them."""

    three_quarter: Path
    ground: Path
    centre_world_m: tuple[float, float, float]
    device: dict[str, Any]
    log: str


def _texture_index(texture_name: str) -> int:
    return int(texture_name.removeprefix("splat_").removesuffix(".png"))


def _write_height_png(height: torch.Tensor, path: Path) -> Path:
    """A 16-bit scratch heightmap for the Displace modifier, not the canonical export deliverable.

    Flipped vertically: a PNG's row 0 is the image top, but Blender samples UV v=0 from a
    texture's bottom row, so leaving this unflipped would mirror the terrain top-to-bottom
    relative to every other preview in the pipeline.
    """
    values = height.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    scaled = np.flipud(values * HEIGHTMAP_MAX + 0.5).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(scaled)).save(path)
    return path


def _write_splat_png(block: np.ndarray, path: Path) -> Path:
    flipped = np.ascontiguousarray(np.flipud(block))
    Image.fromarray(flipped, "RGBA").save(path)
    return path


def _material_metadata(names: Sequence[str], index: MaterialIndex) -> dict[str, dict[str, Any]]:
    metadata = {}
    for name in names:
        material = index[name]
        missing = material.missing()
        if missing:
            raise BeautyRenderError(
                f"material {name!r} is missing {', '.join(missing)} on disk; re-run the pull script"
            )
        paths = material.paths()
        metadata[name] = {
            "tiling_m": material.tiling_m,
            "albedo": str(paths["albedo"]),
            "normal": str(paths["normal"]),
            "roughness": str(paths["roughness"]),
        }
    return metadata


def _validate_centre(centre: Sequence[float]) -> tuple[float, float]:
    if len(centre) != 2:
        raise BeautyRenderError(f"centre {tuple(centre)} must hold two fractions")
    x, y = float(centre[0]), float(centre[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise BeautyRenderError(f"centre {(x, y)} must lie within [0, 1]")
    return x, y


def render(
    cfg: MapConfig,
    height: torch.Tensor,
    splat: SplatResult,
    material_index: MaterialIndex,
    out_dir: Path | None = None,
    centre: Sequence[float] = (0.5, 0.5),
    patch_size_m: float = DEFAULT_PATCH_SIZE_M,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    samples: int = DEFAULT_SAMPLES,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    sun_azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    sun_altitude_deg: float = DEFAULT_ALTITUDE_DEG,
    three_quarter_azimuth_deg: float = DEFAULT_THREE_QUARTER_AZIMUTH_DEG,
    three_quarter_altitude_deg: float = DEFAULT_THREE_QUARTER_ALTITUDE_DEG,
    ground_eye_height_m: float = DEFAULT_GROUND_EYE_HEIGHT_M,
    device: str = DEFAULT_DEVICE,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> BeautyRender:
    """Blender headless: displace a grid by the height field, bind the splat's real tiling
    materials, and render one 3/4 view and one ground-level view at a chosen map position.

    Reads only what is handed in - no dependency on export.py's terrain.json, since this is the
    Phase 2 gate and must run before the export contract exists.
    """
    if tuple(height.shape) != cfg.shape:
        raise BeautyRenderError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    centre_x, centre_y = _validate_centre(centre)
    if grid_resolution < 2:
        raise BeautyRenderError(f"grid_resolution {grid_resolution} must be at least 2")
    if samples < 1:
        raise BeautyRenderError(f"samples {samples} must be at least 1")
    if patch_size_m <= 0.0:
        raise BeautyRenderError(f"patch_size_m {patch_size_m} must be positive")

    target_dir = cfg.out_dir() if out_dir is None else Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    executable = blender_executable or require_blender()
    assignment = splat.assignment()
    materials_used = sorted({entry["material"] for entry in assignment})
    materials_meta = _material_metadata(materials_used, material_index)

    with tempfile.TemporaryDirectory(prefix="beauty_") as scratch_name:
        scratch = Path(scratch_name)
        height_png = _write_height_png(height, scratch / "height.png")
        splat_textures = {
            str(texture_index): str(_write_splat_png(block, scratch / f"splat_{texture_index}.png"))
            for texture_index, block in enumerate(splat.textures)
        }

        three_quarter_path = target_dir / THREE_QUARTER_NAME
        ground_path = target_dir / GROUND_NAME
        result_json = scratch / "result.json"
        args = {
            "world_size_m": cfg.world_size_m,
            "height_range_m": cfg.height_range_m,
            "height_png": str(height_png),
            "grid_resolution": min(grid_resolution, cfg.resolution),
            "splat_textures": splat_textures,
            "materials": materials_meta,
            "layers": [
                {
                    "material": entry["material"],
                    "texture": _texture_index(entry["texture"]),
                    "channel": entry["channel"],
                }
                for entry in assignment
            ],
            "sun": {
                "azimuth_deg": sun_azimuth_deg,
                "altitude_deg": sun_altitude_deg,
                "energy": 4.0,
            },
            "camera": {
                "centre_frac": [centre_x, centre_y],
                "patch_size_m": patch_size_m,
                "three_quarter": {
                    "azimuth_deg": three_quarter_azimuth_deg,
                    "altitude_deg": three_quarter_altitude_deg,
                },
                "ground": {
                    "eye_height_m": ground_eye_height_m,
                    "look_azimuth_deg": three_quarter_azimuth_deg,
                    "back_distance_m": max(10.0, patch_size_m * 0.25),
                },
            },
            "render": {
                "samples": samples,
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "device": device,
            },
            "output": {"three_quarter": str(three_quarter_path), "ground": str(ground_path)},
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
        if completed.returncode != 0 or not three_quarter_path.is_file() or not ground_path.is_file():
            raise BeautyRenderError(
                f"Blender beauty render failed (exit {completed.returncode}):\n{log[-LOG_TAIL_CHARS:]}"
            )
        payload = json.loads(result_json.read_text(encoding="utf-8")) if result_json.is_file() else {}

    return BeautyRender(
        three_quarter=three_quarter_path,
        ground=ground_path,
        centre_world_m=tuple(payload.get("centre_world_m", (0.0, 0.0, 0.0))),
        device=payload.get("device", {}),
        log=log,
    )


def beauty_previews(
    cfg: MapConfig,
    height: torch.Tensor,
    splat: SplatResult,
    material_index: MaterialIndex,
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> tuple[BeautyRender, Preview, Preview]:
    """Render both views and hand back budgeted previews of each, agent-ready."""
    result = render(cfg, height, splat, material_index, **kwargs)
    return result, deliver_file(result.three_quarter, budget), deliver_file(result.ground, budget)
