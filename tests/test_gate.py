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
