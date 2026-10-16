from __future__ import annotations

import math

import numpy as np
import pytest

from assets import animation_checks, mechanism_checks, mechanism_manifest

FPS = 24
CRANK_RADIUS_M = 0.3


def _entry(name, nodes, duration_s=1.0):
    return {"name": name, "state": "working", "kind": "rigid", "duration_s": duration_s, "fps": FPS, "loop": True, "nodes": list(nodes)}


def _times(duration_s=1.0):
    return np.linspace(0.0, duration_s, round(duration_s * FPS) + 1)


def _rotation(angles, axis=(0.0, 1.0, 0.0)):
    axis = np.asarray(axis, dtype=np.float64)
    halves = np.asarray(angles)[:, None] / 2.0
    return np.hstack([np.sin(halves) * axis[None, :], np.cos(halves)])


def _rotation_channel(node, angles, axis=(0.0, 1.0, 0.0)):
    return animation_checks.Channel(node, "rotation", "LINEAR", _times(), _rotation(angles, axis))


def _translation_channel(node, positions):
    return animation_checks.Channel(node, "translation", "LINEAR", _times(), np.asarray(positions, dtype=np.float64))


def _document(nodes):
    return {"scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes}


def _gear_document():
    return _document([{"name": "gear0_ANIM", "translation": [0.0, 1.0, 0.0]}, {"name": "gear1_ANIM", "translation": [0.5, 1.0, 0.0]}])


def _gears(driven_angles, driven_axis=(0.0, 1.0, 0.0)):
    driver = 2.0 * math.pi * _times()
    clips = {"working_gears": [_rotation_channel(0, driver), _rotation_channel(1, driven_angles, driven_axis)]}
    entries = [_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"])]
    mechanisms = [mechanism_manifest.gears_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"], [12, 18])]
    return mechanism_checks.mechanism_checks(_gear_document(), entries, mechanisms, clips)[0]


def test_meshing_gears_at_the_tooth_ratio_pass():
    finding = _gears(-2.0 * math.pi * _times() * 12 / 18)
    assert finding[1], finding


def test_slipping_gear_fails():
    finding = _gears(-2.0 * math.pi * _times() * 12 / 16)
    assert not finding[1]
    assert "gear1_ANIM slips" in finding[2]


def test_gear_turning_with_its_neighbour_fails():
    finding = _gears(2.0 * math.pi * _times() * 12 / 18)
    assert not finding[1]
    assert "the same way as" in finding[2]


def test_gear_on_a_crossed_axis_fails():
    finding = _gears(-2.0 * math.pi * _times() * 12 / 18, driven_axis=(1.0, 0.0, 0.0))
    assert not finding[1]
    assert "off the chain axis" in finding[2]


def test_static_driver_fails():
    clips = {"working_gears": [_rotation_channel(0, np.zeros(FPS + 1)), _rotation_channel(1, np.zeros(FPS + 1))]}
    entries = [_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"])]
    mechanisms = [mechanism_manifest.gears_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"], [12, 18])]
    finding = mechanism_checks.mechanism_checks(_gear_document(), entries, mechanisms, clips)[0]
    assert not finding[1]
    assert "does not turn" in finding[2]


def _crank(rod_positions):
    document = _document([{"name": "crank_ANIM", "translation": [0.0, 1.0, 0.0], "children": [1]}, {"name": "rod_ANIM"}])
    document["scenes"][0]["nodes"] = [0]
    angles = 2.0 * math.pi * _times()
    clips = {"working_crank": [_rotation_channel(0, angles), _translation_channel(1, rod_positions)]}
    entries = [_entry("working_crank", ["crank_ANIM", "rod_ANIM"])]
    mechanisms = [mechanism_manifest.linkage_entry("working_crank", [("rod_ANIM", "crank_ANIM")])]
    return mechanism_checks.mechanism_checks(document, entries, mechanisms, clips)[0]


def test_rod_riding_its_crank_pin_passes():
    finding = _crank(np.tile([CRANK_RADIUS_M, 0.0, 0.0], (FPS + 1, 1)))
    assert finding[1], finding


def test_detached_rod_fails():
    angles = 2.0 * math.pi * _times()
    counter_turned = np.stack([CRANK_RADIUS_M * np.cos(2.0 * angles), np.zeros_like(angles), CRANK_RADIUS_M * np.sin(2.0 * angles)], axis=1)
    finding = _crank(counter_turned)
    assert not finding[1]
    assert "rod_ANIM comes" in finding[2]
    assert "off its pin on crank_ANIM" in finding[2]


def test_world_space_crank_slider_passes():
    angles = 2.0 * math.pi * _times()
    rod_length = 0.9
    pins = np.stack([CRANK_RADIUS_M * np.cos(angles), np.zeros_like(angles), -CRANK_RADIUS_M * np.sin(angles)], axis=1)
    slide = pins[:, 0] + np.sqrt(rod_length**2 - pins[:, 2] ** 2)
    rod_angles = np.arctan2(pins[:, 2], slide - pins[:, 0])
    document = _document([{"name": "crank_ANIM"}, {"name": "rod_ANIM"}, {"name": "piston_ANIM"}])
    clips = {
        "working_crank": [
            _rotation_channel(0, angles),
            _rotation_channel(1, rod_angles),
            _translation_channel(1, pins),
            _translation_channel(2, np.stack([slide, np.zeros_like(slide), np.zeros_like(slide)], axis=1)),
        ]
    }
    piston_on_rod = mechanism_manifest.linkage_entry("working_crank", [("rod_ANIM", "crank_ANIM"), ("piston_ANIM", "rod_ANIM")])
    entries = [_entry("working_crank", ["crank_ANIM", "rod_ANIM", "piston_ANIM"])]
    rod_tip = np.stack([pins[:, 0] + rod_length * np.cos(rod_angles), pins[:, 2] - rod_length * np.sin(rod_angles)], axis=1)
    assert np.allclose(rod_tip[:, 0], slide) and np.allclose(rod_tip[:, 1], 0.0)
    finding = mechanism_checks.mechanism_checks(document, entries, [piston_on_rod], clips)[0]
    assert finding[1], finding


def test_mechanism_on_a_missing_clip_fails():
    entries = [_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"])]
    mechanisms = [mechanism_manifest.gears_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"], [12, 18])]
    finding = mechanism_checks.mechanism_checks(_gear_document(), entries, mechanisms, {})[0]
    assert not finding[1]
    assert "no such clip" in finding[2]


def test_no_mechanisms_reports_nothing():
    assert mechanism_checks.mechanism_checks(_gear_document(), [], [], {}) == []


def test_manifest_rejects_bad_entries():
    animations = [_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"]), {**_entry("idle_wind", ["bone0"]), "kind": "skinned", "state": "idle"}]
    errors = mechanism_manifest.manifest_errors(
        [
            {"type": "belt", "clip": "working_gears"},
            {"type": "gears", "clip": "working_gears", "gears": [{"node": "gear0_ANIM", "teeth": 12}]},
            {"type": "gears", "clip": "working_gears", "gears": [{"node": "gear0_ANIM", "teeth": 12}, {"node": "gear9_ANIM", "teeth": 18}]},
            {"type": "linkage", "clip": "working_absent", "joints": [{"node": "rod_ANIM", "on": "crank_ANIM"}]},
            {"type": "linkage", "clip": "idle_wind", "joints": [{"node": "bone0", "on": "bone1"}]},
            {"type": "linkage", "clip": "working_gears", "joints": [{"node": "rod_ANIM", "on": "rod_ANIM"}], "extra": 1},
        ],
        animations,
    )
    text = "; ".join(errors)
    assert "type 'belt'" in text
    assert "at least two meshing gears" in text
    assert "gear9_ANIM not animated by the clip" in text
    assert "no animations[] entry" in text
    assert "rigid clips only" in text
    assert "pinned to itself" in text
    assert "unknown field(s) extra" in text


def test_gears_entry_needs_one_tooth_count_per_gear():
    with pytest.raises(mechanism_manifest.MechanismManifestError):
        mechanism_manifest.gears_entry("working_gears", ["gear0_ANIM", "gear1_ANIM"], [12])
