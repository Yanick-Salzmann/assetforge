from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from assets import validate
from terrain import config

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "export_selftest.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_export_asset_writes_glb_textures_and_manifest(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    out_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(REPO_ROOT),
            str(out_dir),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload.get("ok"), payload
    monkeypatch.setattr(config, "ASSET_OUT_DIR", out_dir)
    checks = {check.name: check for check in validate.validate_asset("export-selftest-fx", "prop").checks}
    assert checks["fx sockets"].passed, checks["fx sockets"]
    report = validate.validate_asset("export-selftest-animated-lod", "prop")
    assert list(report.lod_triangle_counts) == ["LOD0", "LOD1", "LOD2"], report.lod_triangle_counts
    animation_checks = [
        check
        for check in report.checks
        if check.name in ("animations manifest", "animation clips") or check.name.startswith(("idle_", "Wheel_"))
    ]
    assert len(animation_checks) >= 6, [check.name for check in report.checks]
    assert all(check.passed for check in animation_checks), [check for check in animation_checks if not check.passed]
    cloth_report = validate.validate_asset("export-selftest-cloth-lod", "prop")
    failed = [f"{check.name}: {check.detail}" for check in cloth_report.checks if not check.passed]
    assert not failed, failed
    assert list(cloth_report.lod_triangle_counts) == ["LOD0", "LOD1", "LOD2"], cloth_report.lod_triangle_counts
    if not payload["prop_check"].get("skipped"):
        prop_checks = {check.name: check for check in validate.validate_asset("export-selftest-props", "prop").checks}
        assert prop_checks["kit props"].passed, prop_checks["kit props"]
        assert "nature/pot_small" in prop_checks["kit props"].detail
