from __future__ import annotations

import json

from library import generated
from library.kenney import PackSpec
from tests.test_quaternius import _minimal_glb


def _asset(root, name, approved):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / f"{name}.glb").write_bytes(_minimal_glb())
    (directory / generated.GATE_NAME).write_text(json.dumps({"approved": approved}), encoding="utf-8")


def test_pull_locks_only_approved_assets_under_the_prefix(tmp_path):
    assets = tmp_path / "assets"
    _asset(assets, "scatter_rock_a", True)
    _asset(assets, "scatter_rock_b", False)
    _asset(assets, "other_rock", True)
    spec = PackSpec(name="scatter", slug="scatter", category="vegetation", source="assetforge")
    lock_path = tmp_path / "kits.lock.json"
    report = generated.pull([spec], tmp_path / "kits", lock_path, asset_dir=assets)
    assert report["pulled"] == ["scatter"]
    entry = json.loads(lock_path.read_text())["packs"]["scatter"]
    assert entry["source"] == "assetforge"
    assert list(entry["models"]) == ["rock_a"]
    model = entry["models"]["rock_a"]
    assert model["path"] == "kits/scatter/rock_a.glb"
    assert model["tris"] == 1
    assert len(model["sha256"]) == 64
    assert (tmp_path / "kits" / "scatter" / "rock_a.glb").is_file()


def test_pull_reports_a_pack_with_nothing_approved(tmp_path):
    spec = PackSpec(name="scatter", slug="scatter", category="vegetation", source="assetforge")
    report = generated.pull([spec], tmp_path / "kits", tmp_path / "lock.json", asset_dir=tmp_path / "assets")
    assert report["failed"][0]["pack"] == "scatter"


def test_pull_locks_lods_and_copies_the_impostor(tmp_path):
    assets = tmp_path / "assets"
    _asset(assets, "flora_oak_a", True)
    directory = assets / "flora_oak_a"
    (directory / "textures").mkdir()
    for key in generated.IMPOSTOR_MAPS:
        (directory / "textures" / f"impostor_{key}.png").write_bytes(key.encode())
    manifest = {
        "triangle_counts": {"LOD0": 9, "LOD2": 2, "LOD1": 4},
        "lod": {"switch_distances_m": [40.0, 100.0, 250.0]},
        "impostor": {
            "albedo": "textures/impostor_albedo.png",
            "normal": "textures/impostor_normal.png",
            "layout": "hemi_octahedral",
            "frames": 8,
            "frame_px": 128,
            "centre": [0.0, 4.0, 0.0],
            "radius": 5.0,
        },
    }
    (directory / generated.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    spec = PackSpec(name="flora", slug="flora", category="vegetation", source="assetforge")
    lock_path = tmp_path / "kits.lock.json"
    generated.pull([spec], tmp_path / "kits", lock_path, asset_dir=assets)
    model = json.loads(lock_path.read_text())["packs"]["flora"]["models"]["oak_a"]
    assert model["lod_tris"] == [9, 4, 2]
    assert model["lod_distances_m"] == [40.0, 100.0, 250.0]
    assert model["impostor"]["albedo"] == "kits/flora/oak_a_impostor_albedo.png"
    assert model["impostor"]["frames"] == 8
    assert len(model["impostor"]["normal_sha256"]) == 64
    assert (tmp_path / "kits" / "flora" / "oak_a_impostor_normal.png").read_bytes() == b"normal"
