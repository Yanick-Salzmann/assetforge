from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assets import blender as blender_discovery
from assets import contact_sheet
from assets.recipe import validate_name
from terrain import config

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_scripts" / "demo_showcase_build.py"

DEFAULT_NAME = "hardsurface-demo"
DEFAULT_TIMEOUT_S = 180.0
LOG_TAIL_CHARS = 4000


class DemoBuildError(RuntimeError):
    """Raised when the headless hard-surface capability demo cannot be built."""


@dataclass(frozen=True)
class DemoBuildResult:
    """The exported demo glb, its triangle count, and what built it."""

    glb_path: Path
    triangle_count: int
    object_names: list[str]
    log: str


def build(
    name: str = DEFAULT_NAME,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> DemoBuildResult:
    """Headless Blender: assemble a small shed from every assets/hardsurface.py helper, run it
    through assets/game_ready.py, and export the glb.

    This is a visual smoke test of the current hard-surface pipeline, independent of any live
    MCP session - the assets counterpart to terrain's synth/erosion smoke previews.
    """
    validate_name(name)
    executable = blender_executable or blender_discovery.require_blender()
    out_dir = (config.ASSET_OUT_DIR / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / f"{name}.glb"
    result_path = out_dir / "demo_build_result.json"

    completed = subprocess.run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(config.REPO_ROOT),
            str(glb_path),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    log = completed.stdout + completed.stderr
    payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    if completed.returncode != 0 or not glb_path.is_file() or not payload.get("ok"):
        raise DemoBuildError(
            f"headless demo build failed (exit {completed.returncode}):\n{log[-LOG_TAIL_CHARS:]}\n{payload.get('error', '')}"
        )
    return DemoBuildResult(
        glb_path=glb_path,
        triangle_count=int(payload.get("triangle_count", 0)),
        object_names=list(payload.get("object_names", [])),
        log=log,
    )


def build_and_render(
    name: str = DEFAULT_NAME,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[DemoBuildResult, contact_sheet.ContactSheetRender]:
    """Build the demo asset, then render its 8-view contact sheet in one call."""
    built = build(name, blender_executable=blender_executable, timeout_s=timeout_s)
    rendered = contact_sheet.render(name, blender_executable=blender_executable, timeout_s=timeout_s)
    return built, rendered
