from __future__ import annotations

import pytest

from assets import fx_manifest


def _entry(**overrides):
    entry = {"node": "FX_fire_forge", "type": "fire", "state": "working", "rate": 30.0, "size": 0.4, "colour": [1.0, 0.5, 0.1]}
    entry.update(overrides)
    return entry


def test_valid_entry_has_no_errors():
    assert fx_manifest.entry_errors(_entry()) == []


@pytest.mark.parametrize("fx_type", fx_manifest.TYPES)
def test_every_type_names_a_valid_socket(fx_type):
    assert fx_manifest.NODE_NAME.match(fx_manifest.fx_node_name(fx_type))
    assert fx_manifest.NODE_NAME.match(fx_manifest.fx_node_name(fx_type, "left.001"))


def test_fx_node_name_rejects_unknown_type_and_bad_label():
    with pytest.raises(fx_manifest.FxManifestError):
        fx_manifest.fx_node_name("lava")
    with pytest.raises(fx_manifest.FxManifestError):
        fx_manifest.fx_node_name("fire", "bad label")


def test_fx_entry_builds_a_validated_entry():
    entry = fx_manifest.fx_entry("FX_flicker_light", "flicker_light", "idle", 6, 3, (1, 0.8, 0.5))
    assert entry == {"node": "FX_flicker_light", "type": "flicker_light", "state": "idle", "rate": 6.0, "size": 3.0, "colour": [1.0, 0.8, 0.5]}


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"type": "lava"}, "type"),
        ({"node": "fire_forge"}, "node must be named"),
        ({"node": "FX_smoke_forge"}, "does not carry its type"),
        ({"state": "night"}, "state"),
        ({"rate": 0}, "rate"),
        ({"rate": True}, "rate"),
        ({"rate": fx_manifest.MAX_RATE * 2}, "rate"),
        ({"size": -1.0}, "size"),
        ({"size": float("nan")}, "size"),
        ({"colour": [1.0, 0.5]}, "colour"),
        ({"colour": [1.0, 0.5, 2.0]}, "colour"),
        ({"colour": "orange"}, "colour"),
        ({"extra": 1}, "unknown field"),
    ],
)
def test_invalid_entries_are_reported(overrides, fragment):
    errors = fx_manifest.entry_errors(_entry(**overrides))
    assert any(fragment in error for error in errors), errors


def test_missing_fields_are_reported():
    assert "missing" in fx_manifest.entry_errors({"node": "FX_fire"})[0]


def test_manifest_must_be_a_list_without_repeated_nodes():
    assert fx_manifest.manifest_errors({}) == ["fx must be a list"]
    assert any("repeat" in error for error in fx_manifest.manifest_errors([_entry(), _entry()]))


def test_load_returns_entries_and_raises_on_errors():
    assert fx_manifest.load({}) == []
    assert fx_manifest.load({"fx": [_entry()]}) == [_entry()]
    with pytest.raises(fx_manifest.FxManifestError):
        fx_manifest.load({"fx": [_entry(rate=-1)]})
