from __future__ import annotations

import pytest

from assets import gate
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    gate._GATES.clear()
    yield
    gate._GATES.clear()


def test_patches_are_allowed_up_to_the_cap():
    asset_gate = gate.open_gate("hero-crate")
    for _ in range(gate.MAX_PATCH_ITERATIONS):
        asset_gate.record_patch()
    assert asset_gate.patch_count == gate.MAX_PATCH_ITERATIONS


def test_a_patch_past_the_cap_is_refused():
    asset_gate = gate.open_gate("hero-crate")
    for _ in range(gate.MAX_PATCH_ITERATIONS):
        asset_gate.record_patch()
    with pytest.raises(gate.GateError, match="4 allowed patch"):
        asset_gate.record_patch()


@pytest.mark.parametrize("kind", ["hero_building", "hero_vehicle"])
def test_hero_kinds_get_the_wider_cap(kind):
    asset_gate = gate.open_gate("hero-tractor")
    for _ in range(gate.HERO_PATCH_ITERATIONS):
        asset_gate.record_patch(kind)
    assert asset_gate.as_dict()["patches_remaining"] == 0
    with pytest.raises(gate.GateError, match=f"{gate.HERO_PATCH_ITERATIONS} allowed patch"):
        asset_gate.record_patch(kind)


def test_kind_is_remembered_across_reload():
    gate.open_gate("hero-tractor").record_patch("hero_vehicle")
    gate.forget("hero-tractor")
    reloaded = gate.open_gate("hero-tractor")
    assert reloaded.kind == "hero_vehicle"
    assert reloaded.patch_cap() == gate.HERO_PATCH_ITERATIONS


def test_non_hero_kinds_keep_the_default_cap():
    assert gate.patch_cap("vehicle") == gate.MAX_PATCH_ITERATIONS
    assert gate.patch_cap("prop") == gate.MAX_PATCH_ITERATIONS
    assert gate.patch_cap("") == gate.MAX_PATCH_ITERATIONS


def test_export_is_blocked_until_approved():
    asset_gate = gate.open_gate("prop-barrel")
    with pytest.raises(gate.GateError, match="not been approved"):
        asset_gate.require_approved()
    asset_gate.approve("looks good")
    asset_gate.require_approved()


def test_gate_reloads_from_disk_after_forget():
    asset_gate = gate.open_gate("prop-crate")
    asset_gate.record_patch()
    asset_gate.approve("ok")
    gate.forget("prop-crate")
    reloaded = gate.open_gate("prop-crate")
    assert reloaded.patch_count == 1
    assert reloaded.approved is True
    assert reloaded.approval_note == "ok"


@pytest.mark.parametrize("name", ["Hero", "has space", "-leading-dash", ""])
def test_invalid_names_are_rejected(name):
    with pytest.raises(Exception):
        gate.open_gate(name)


def test_clips_need_their_own_approval():
    asset = gate.open_gate("wheel-shed")
    asset.approve("form ok")
    asset.require_approved()
    with pytest.raises(gate.GateError, match="idle_wheel_coast"):
        asset.require_approved(clips=["idle_wheel_coast"])
    asset.approve("form and motion ok", clips=["working_wheel_spin", "idle_wheel_coast"])
    asset.require_approved(clips=["idle_wheel_coast", "working_wheel_spin"])
    with pytest.raises(gate.GateError, match="working_kick"):
        asset.require_approved(clips=["working_kick"])


def test_approved_clips_survive_reload():
    gate.open_gate("wheel-shed").approve("ok", clips=["idle_wheel_coast"])
    gate.forget("wheel-shed")
    reloaded = gate.open_gate("wheel-shed")
    assert reloaded.approved_clips == ["idle_wheel_coast"]
    assert reloaded.as_dict()["approved_clips"] == ["idle_wheel_coast"]
