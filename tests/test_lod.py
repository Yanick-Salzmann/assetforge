from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "lod_selftest.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_generate_lods_decimates_and_bakes_normal_maps(tmp_path):
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(REPO_ROOT),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload.get("ok"), payload
