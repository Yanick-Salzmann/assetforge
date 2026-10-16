from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from assets import animation_checks, validate
from library import gltf
from terrain import config

FPS = 24
CUBE_FACES = [
    (0, 2, 1), (0, 3, 2),
    (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4),
    (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6),
    (3, 0, 4), (3, 4, 7),
]


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    yield


def _box(low, high) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = low, high
    positions = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        dtype=np.float64,
    )
    return positions, np.array(CUBE_FACES, dtype=np.int64)[:, ::-1].copy()


def _entry(name, nodes, kind="rigid", duration_s=1.0, loop=True):
    return {
        "name": name,
        "state": name.split("_", 1)[0],
        "kind": kind,
        "duration_s": duration_s,
        "fps": FPS,
        "loop": loop,
        "nodes": list(nodes),
    }


def _times(duration_s=1.0):
    return np.linspace(0.0, duration_s, round(duration_s * FPS) + 1)


def _y_rotation(angles: np.ndarray) -> np.ndarray:
    return np.stack([np.zeros_like(angles), np.sin(angles / 2), np.zeros_like(angles), np.cos(angles / 2)], axis=1)


def _rotation_channel(node, angles, times=None):
    times = _times() if times is None else times
    return animation_checks.Channel(node, "rotation", "LINEAR", times, _y_rotation(angles))


def _translation_channel(node, offsets, times=None):
    times = _times() if times is None else times
    values = np.zeros((len(times), 3))
    values[:, 0] = offsets
    return animation_checks.Channel(node, "translation", "LINEAR", times, values)


def _document(nodes, skins=None):
    document = {"scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes}
    if skins:
        document["skins"] = skins
    return document


def test_matching_clips_pass():
    document = _document([{"name": "arm_ANIM"}])
    clips = {"working_swing": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.clip_checks(document, [_entry("working_swing", ["arm_ANIM"])], clips)[0]
    assert finding[1], finding


def test_entry_without_clip_fails():
    document = _document([{"name": "arm_ANIM"}])
    finding = animation_checks.clip_checks(document, [_entry("working_swing", ["arm_ANIM"])], {})[0]
    assert not finding[1]
    assert "no glTF clip" in finding[2]


def test_clip_without_entry_fails():
    document = _document([{"name": "arm_ANIM"}])
    clips = {"idle_sway": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.clip_checks(document, [], clips)[0]
    assert not finding[1]
    assert "idle_sway" in finding[2]


def test_clip_that_skips_a_listed_node_fails():
    document = _document([{"name": "arm_ANIM"}, {"name": "lid_ANIM"}])
    clips = {"working_swing": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.clip_checks(document, [_entry("working_swing", ["arm_ANIM", "lid_ANIM"])], clips)[0]
    assert not finding[1]
    assert "lid_ANIM" in finding[2]


def test_clip_length_must_match_duration():
    document = _document([{"name": "arm_ANIM"}])
    clips = {"working_swing": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.clip_checks(document, [_entry("working_swing", ["arm_ANIM"], duration_s=2.0)], clips)[0]
    assert not finding[1]
    assert "runs 1.0000 s" in finding[2]


def test_skinned_nodes_must_be_skin_joints():
    document = _document([{"name": "bone0"}, {"name": "bone1"}], skins=[{"joints": [0]}])
    clips = {"idle_wind": [_rotation_channel(0, np.zeros(FPS + 1)), _rotation_channel(1, np.zeros(FPS + 1))]}
    finding = animation_checks.clip_checks(document, [_entry("idle_wind", ["bone0", "bone1"], kind="skinned")], clips)[0]
    assert not finding[1]
    assert "not skin joints: bone1" in finding[2]


def test_rest_transform_at_pivot_passes():
    document = _document([{"name": "arm_ANIM", "translation": [0.5, 1.0, 0.0]}])
    finding = animation_checks.rest_checks(document, [_entry("working_swing", ["arm_ANIM"])])[0]
    assert finding[1], finding


@pytest.mark.parametrize(
    "node",
    [
        {"name": "arm_ANIM", "rotation": [0.0, math.sin(0.25), 0.0, math.cos(0.25)]},
        {"name": "arm_ANIM", "scale": [1.0, 2.0, 1.0]},
        {"name": "arm_ANIM", "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
        {"name": "other"},
    ],
)
def test_rest_transform_failures(node):
    finding = animation_checks.rest_checks(_document([node]), [_entry("working_swing", ["arm_ANIM"])])[0]
    assert not finding[1]


def test_skinned_joints_may_rest_rotated():
    document = _document([{"name": "bone0", "rotation": [0.5, 0.5, 0.5, 0.5]}], skins=[{"joints": [0]}])
    assert animation_checks.rest_checks(document, [_entry("idle_wind", ["bone0"], kind="skinned")]) == []


def _seam(channels, loop=True):
    document = _document([{"name": "arm_ANIM"}])
    entry = _entry("idle_sway", ["arm_ANIM"], loop=loop) if loop else _entry("working_lift", ["arm_ANIM"], loop=False)
    return animation_checks.seam_checks(document, [entry], {entry["name"]: channels})


def test_cosine_loop_closes_its_seam():
    finding = _seam([_translation_channel(0, 0.3 * np.cos(2 * np.pi * _times()))])[0]
    assert finding[1], finding


def test_full_turn_spin_closes_its_seam_despite_quaternion_sign_flips():
    finding = _seam([_rotation_channel(0, 2 * np.pi * _times())])[0]
    assert finding[1], finding


def test_open_ended_loop_fails():
    finding = _seam([_translation_channel(0, 0.3 * _times())])[0]
    assert not finding[1]
    assert "ends" in finding[2]


def test_velocity_kink_at_the_seam_fails():
    finding = _seam([_translation_channel(0, 0.3 * np.sin(np.pi * _times()))])[0]
    assert not finding[1]
    assert "velocity" in finding[2]


def test_rotation_velocity_kink_at_the_seam_fails():
    finding = _seam([_rotation_channel(0, 0.6 * np.sin(np.pi * _times()))])[0]
    assert not finding[1]


def test_non_looping_clips_skip_the_seam_check():
    assert _seam([_translation_channel(0, 0.3 * _times())], loop=False) == []


def test_held_channel_that_ends_early_still_closes():
    times = np.array([0.0, 0.5])
    finding = _seam([_translation_channel(0, np.array([0.0, 0.0]), times=times), _rotation_channel(0, 2 * np.pi * _times())])[0]
    assert finding[1], finding


def test_budget_passes_for_a_small_clip():
    clips = {"idle_sway": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.budget_checks(_document([{"name": "a"}]), [_entry("idle_sway", ["a"])], clips, "prop")[0]
    assert finding[1], finding


def test_budget_fails_for_too_many_bones():
    count = animation_checks.MAX_BONES["prop"] + 1
    document = _document([{"name": f"bone{i}"} for i in range(count)], skins=[{"joints": list(range(count))}])
    finding = animation_checks.budget_checks(document, [], {}, "prop")[0]
    assert not finding[1]
    assert "bones" in finding[2]


def test_budget_fails_for_an_overlong_clip():
    duration = animation_checks.CLIP_MAX_S + 1.0
    finding = animation_checks.budget_checks(_document([{"name": "a"}]), [_entry("idle_sway", ["a"], duration_s=duration)], {}, "prop")[0]
    assert not finding[1]


def test_budget_fails_for_too_many_keyframes(monkeypatch):
    monkeypatch.setattr(animation_checks, "CLIP_MAX_KEYFRAMES", 10)
    clips = {"idle_sway": [_rotation_channel(0, np.zeros(FPS + 1))]}
    finding = animation_checks.budget_checks(_document([{"name": "a"}]), [_entry("idle_sway", ["a"])], clips, "prop")[0]
    assert not finding[1]
    assert "keyframes" in finding[2]


def test_solid_winding_and_tolerance():
    solid = animation_checks.Solid(*_box((-1, -1, -1), (1, 1, 1)))
    points = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.7], [1.5, 0.0, 0.0], [0.0, 2.0, 0.0], [0.999, 0.0, 0.0]])
    assert (solid.winding(points) > 0).tolist() == [True, True, False, False, True]
    assert solid.deep_inside(points, 0.01).tolist() == [True, True, False, False, False]


def test_overlapping_shells_still_read_inside():
    first = _box((-1, -1, -1), (1, 1, 1))
    second = _box((0, -0.5, -0.5), (2, 0.5, 0.5))
    positions = np.vstack([first[0], second[0]])
    indices = np.vstack([first[1], second[1] + 8])
    solid = animation_checks.Solid(positions, indices)
    assert (solid.winding(np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])) > 0).tolist() == [True, True]


def _arm_scene(post_x: float):
    nodes = [{"name": "arm_ANIM", "translation": [0.0, 0.5, 0.0]}, {"name": "post", "translation": [post_x, 0.0, 0.0]}]
    parts = {0: _box((0.0, -0.05, -0.05), (1.0, 0.05, 0.05)), 1: _box((-0.1, 0.0, -0.1), (0.1, 1.0, 0.1))}
    return _document(nodes), parts


def test_arm_swinging_through_a_post_collides():
    document, parts = _arm_scene(post_x=0.0)
    document["nodes"][1]["translation"] = [0.0, 0.0, 0.6]
    clips = {"idle_swing": [_rotation_channel(0, 2 * np.pi * _times())]}
    finding = animation_checks.sweep_checks(document, [_entry("idle_swing", ["arm_ANIM"])], clips, parts)[0]
    assert not finding[1]
    assert "arm_ANIM into post" in finding[2] or "post into arm_ANIM" in finding[2]


def test_arm_swinging_clear_of_a_post_passes():
    document, parts = _arm_scene(post_x=0.0)
    document["nodes"][1]["translation"] = [0.0, 0.0, 1.5]
    clips = {"idle_swing": [_rotation_channel(0, 2 * np.pi * _times())]}
    finding = animation_checks.sweep_checks(document, [_entry("idle_swing", ["arm_ANIM"])], clips, parts)[0]
    assert finding[1], finding


def test_axle_spinning_in_its_bearing_is_designed_contact():
    nodes = [{"name": "axle_ANIM", "translation": [0.0, 1.0, 0.0]}, {"name": "bearing"}]
    parts = {0: _box((-0.05, -0.05, -0.6), (0.05, 0.05, 0.6)), 1: _box((-0.3, 0.5, -0.2), (0.3, 1.5, 0.2))}
    angles = 2 * np.pi * _times()
    rotation = np.stack([np.zeros_like(angles), np.zeros_like(angles), np.sin(angles / 2), np.cos(angles / 2)], axis=1)
    clips = {"working_spin": [animation_checks.Channel(0, "rotation", "LINEAR", _times(), rotation)]}
    finding = animation_checks.sweep_checks(_document(nodes), [_entry("working_spin", ["axle_ANIM"])], clips, parts)[0]
    assert finding[1], finding


def test_skinned_clips_skip_the_sweep():
    document, parts = _arm_scene(post_x=0.0)
    clips = {"idle_wind": [_rotation_channel(0, 2 * np.pi * _times())]}
    assert animation_checks.sweep_checks(document, [_entry("idle_wind", ["arm_ANIM"], kind="skinned")], clips, parts) == []


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _write_glb(path, nodes, meshes, animations) -> None:
    binary = b""
    accessors = []
    views = []

    def add(array: np.ndarray, accessor_type: str, component: int, extra: dict | None = None) -> int:
        nonlocal binary
        data = array.astype(np.uint16 if component == 5123 else np.float32).tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += _pad4(data)
        accessor = {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": accessor_type}
        accessor.update(extra or {})
        accessors.append(accessor)
        return len(accessors) - 1

    mesh_entries = []
    for positions, indices in meshes:
        position = add(positions, "VEC3", 5126, {"min": positions.min(0).tolist(), "max": positions.max(0).tolist()})
        index = add(indices.reshape(-1), "SCALAR", 5123)
        mesh_entries.append({"primitives": [{"attributes": {"POSITION": position}, "indices": index, "material": 0}]})
    animation_entries = []
    for name, channels in animations:
        samplers = []
        targets = []
        for channel in channels:
            times = add(channel.times, "SCALAR", 5126, {"min": [float(channel.times[0])], "max": [float(channel.times[-1])]})
            values = add(channel.values, "VEC4" if channel.path == "rotation" else "VEC3", 5126)
            samplers.append({"input": times, "output": values, "interpolation": channel.interpolation})
            targets.append({"sampler": len(samplers) - 1, "target": {"node": channel.node, "path": channel.path}})
        animation_entries.append({"name": name, "samplers": samplers, "channels": targets})
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes)]}],
        "nodes": nodes + [{"name": "root", "children": list(range(len(nodes)))}],
        "meshes": mesh_entries,
        "materials": [{"name": "atlas"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "animations": animation_entries,
    }
    json_chunk = _pad4(json.dumps(document).encode("utf-8"), fill=b" ")
    header = struct.pack("<III", gltf.GLB_MAGIC, 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        header + struct.pack("<II", len(json_chunk), gltf.CHUNK_JSON) + json_chunk + struct.pack("<II", len(binary), gltf.CHUNK_BIN) + binary
    )


def _write_swing_asset(name: str, post_z: float, animations_entry: bool) -> None:
    nodes = [{"name": "arm_ANIM", "mesh": 0, "translation": [0.0, 0.5, 0.0]}, {"name": "post", "mesh": 1, "translation": [0.0, 0.0, post_z]}]
    meshes = [_box((0.0, -0.05, -0.05), (1.0, 0.05, 0.05)), _box((-0.1, 0.0, -0.1), (0.1, 1.0, 0.1))]
    channels = [_rotation_channel(0, 2 * np.pi * _times())]
    _write_glb(config.ASSET_OUT_DIR / name / f"{name}.glb", nodes, meshes, [("idle_swing", channels)])
    manifest = {"name": name}
    if animations_entry:
        manifest["animations"] = [_entry("idle_swing", ["arm_ANIM"])]
    (config.ASSET_OUT_DIR / name / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _checks(name: str) -> dict[str, validate.CheckResult]:
    return {check.name: check for check in validate.validate_asset(name, "prop").checks}


def test_validate_asset_runs_the_animation_checks_end_to_end():
    _write_swing_asset("swing", post_z=1.5, animations_entry=True)
    checks = _checks("swing")
    for label in ("animation clips", "arm_ANIM: rest transform", "idle_swing: loop seam", "animation budget", "idle_swing: swept collision"):
        assert checks[label].passed, checks[label]


def test_validate_asset_flags_a_colliding_clip():
    _write_swing_asset("swing", post_z=0.6, animations_entry=True)
    assert not _checks("swing")["idle_swing: swept collision"].passed


def test_validate_asset_flags_glb_clips_missing_from_the_manifest():
    _write_swing_asset("swing", post_z=1.5, animations_entry=False)
    assert not _checks("swing")["animation clips"].passed


def test_read_clips_returns_channels_per_clip():
    _write_swing_asset("swing", post_z=1.5, animations_entry=True)
    document, buffers = gltf.read_document_with_buffers(config.ASSET_OUT_DIR / "swing" / "swing.glb")
    clips = animation_checks.read_clips(document, buffers)
    channel = clips["idle_swing"][0]
    assert (channel.node, channel.path, channel.values.shape) == (0, "rotation", (FPS + 1, 4))
