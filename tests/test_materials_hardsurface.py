from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from library import materials as materials_lib

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "materials_selftest.py"
CC0_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "materials_cc0_selftest.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_selftest(script_path: Path, tmp_path: Path) -> dict:
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            str(REPO_ROOT),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8"))


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_hardsurface_materials_assign_and_stay_procedural(tmp_path):
    payload = _run_selftest(SCRIPT_PATH, tmp_path)
    assert payload.get("ok"), payload


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
@pytest.mark.skipif(not materials_lib.ASSET_INDEX_FILE.is_file(), reason="asset material index not pulled")
def test_cc0_materials_assign_alongside_procedural(tmp_path):
    payload = _run_selftest(CC0_SCRIPT_PATH, tmp_path)
    assert payload.get("ok"), payload
