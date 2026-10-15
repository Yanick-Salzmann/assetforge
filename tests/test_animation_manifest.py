from __future__ import annotations

import pytest

from assets import animation_manifest as am


def _rigid(**overrides) -> dict:
    entry = {
        "name": "working_wheel_spin",
        "state": "working",
        "kind": "rigid",
        "duration_s": 2.0,
        "fps": 24,
        "loop": True,
        "nodes": ["Wheel_ANIM"],
    }
    entry.update(overrides)
    return entry


def _skinned(**overrides) -> dict:
    entry = {
        "name": "idle_banner_wave",
        "state": "idle",
        "kind": "skinned",
        "duration_s": 1.5,
        "fps": 24,
        "loop": True,
        "nodes": ["seg0", "seg1", "seg2", "seg3"],
    }
    entry.update(overrides)
    return entry


def test_valid_entries_load():
    animations = am.load({"animations": [_rigid(), _skinned()]})
    assert [entry["name"] for entry in animations] == ["working_wheel_spin", "idle_banner_wave"]
    assert am.frame_count(animations[0]) == 48


def test_missing_key_is_static():
    assert am.load({"name": "crate"}) == []


def test_manifest_entry_builds_a_valid_entry():
    entry = am.manifest_entry("idle", "flag_wave", "skinned", 1.5, 24, ["seg0", "seg1"])
    assert entry["name"] == "idle_flag_wave"
    assert am.entry_errors(entry) == []


def test_anim_node_name():
    assert am.anim_node_name("Wheel") == "Wheel_ANIM"
    with pytest.raises(am.AnimationManifestError):
        am.anim_node_name("bad name")


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        (_rigid(state="running"), "state"),
        (_rigid(kind="cloth"), "kind"),
        (_rigid(name="wheel_spin"), "<state>_<clip>"),
        (_rigid(name="idle_wheel_spin"), "start with its state"),
        (_rigid(name="working_Wheel"), "<state>_<clip>"),
        (_rigid(fps=0), "fps"),
        (_rigid(fps=24.0), "fps"),
        (_rigid(fps=True), "fps"),
        (_rigid(duration_s=0), "positive"),
        (_rigid(duration_s=float("nan")), "positive"),
        (_rigid(duration_s="2"), "number"),
        (_rigid(duration_s=2.01), "whole number of frames"),
        (_rigid(loop="yes"), "loop"),
        (_skinned(loop=False), "must loop"),
        (_rigid(nodes=[]), "non-empty list"),
        (_rigid(nodes=[""]), "non-empty strings"),
        (_rigid(nodes=["Wheel"]), "_ANIM"),
        (_rigid(nodes=["Wheel_ANIM", "Wheel_ANIM"]), "repeat"),
        (_rigid(extra=1), "unknown field"),
    ],
)
def test_entry_errors(entry, fragment):
    errors = am.entry_errors(entry)
    assert any(fragment in error for error in errors), errors


def test_missing_fields_reported():
    entry = _rigid()
    del entry["fps"]
    assert am.entry_errors(entry) == ["working_wheel_spin: missing fps"]


def test_working_clips_may_play_once():
    assert am.entry_errors(_rigid(loop=False)) == []


def test_skinned_nodes_need_no_anim_suffix():
    assert am.entry_errors(_skinned()) == []


def test_repeated_clip_names_rejected():
    errors = am.manifest_errors([_rigid(), _rigid()])
    assert errors == ["clip name(s) repeat: working_wheel_spin"]


def test_non_list_rejected():
    with pytest.raises(am.AnimationManifestError, match="must be a list"):
        am.load({"animations": {"name": "idle_x"}})


def test_non_object_entry_rejected():
    assert am.manifest_errors(["idle_x"]) == ["animation entry must be an object, got str"]


def test_builder_rejects_bad_state():
    with pytest.raises(am.AnimationManifestError):
        am.manifest_entry("sleeping", "x", "rigid", 1.0, 24, ["X_ANIM"])
