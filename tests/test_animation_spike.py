from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "animation_spike.py"


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_animation_spike_exports_named_rigid_and_skinned_clips(tmp_path):
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(tmp_path / "out"),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload.get("ok"), payload
    for mode in ("ACTIONS", "NLA_TRACKS"):
        summary = payload["modes"][mode]
        assert summary["materials"] == ["spike_mat"]
        assert summary["skins"] == [{"joints": ["seg0", "seg1", "seg2", "seg3"]}]
        nodes = {node["name"]: node for node in summary["nodes"]}
        assert nodes["Base"]["children"] == ["Rotor_ANIM"]
        assert nodes["Strip"]["skin"]
        clips = {animation["name"]: animation["channels"] for animation in summary["animations"]}
        assert set(clips) == {"rotor_spin", "strip_wave"}
        rotor = clips["rotor_spin"][0]
        assert (rotor["node"], rotor["path"]) == ("Rotor_ANIM", "rotation")
        assert rotor["t_min"] == [0]
        assert rotor["t_max"] == [2]
        assert {channel["node"] for channel in clips["strip_wave"]} == {"seg0", "seg1", "seg2", "seg3"}
    assert payload["modes"]["ACTIVE_ACTIONS"]["animations"] == []
