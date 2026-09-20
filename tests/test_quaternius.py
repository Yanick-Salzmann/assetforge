from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

from library import gltf, quaternius
from library.kenney import PackSpec


def _minimal_glb() -> bytes:
    document = {
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [
            {
                "count": 3,
                "type": "VEC3",
                "componentType": 5126,
                "min": [-1.0, 0.0, -1.0],
                "max": [1.0, 2.0, 1.0],
            }
        ],
    }
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((-len(payload)) % 4)
    header = struct.pack("<III", gltf.GLB_MAGIC, 2, 12 + 8 + len(payload))
    chunk = struct.pack("<II", len(payload), gltf.CHUNK_JSON)
    return header + chunk + payload


def _fake_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Quaternius_Pack/Models/GLTF/house.glb", _minimal_glb())
        archive.writestr("Quaternius_Pack/readme.txt", "Free assets by Quaternius")
    return buffer.getvalue()


@pytest.fixture
def spec():
    return PackSpec(name="test_pack", slug="test-pack", category="prop", source="quaternius")


@pytest.fixture
def staged(tmp_path):
    staging_dir = tmp_path / "_staging"
    staging_dir.mkdir()
    (staging_dir / "test-pack.zip").write_bytes(_fake_zip())
    return staging_dir


def test_pull_extracts_models_and_stats(tmp_path, spec, staged):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    report = quaternius.pull([spec], kits_dir, lock_path, staging_dir=staged)
    assert report["pulled"] == ["test_pack"]
    assert (kits_dir / "test_pack" / "Quaternius_Pack" / "Models" / "GLTF" / "house.glb").is_file()

    entry = json.loads(lock_path.read_text())["packs"]["test_pack"]
    assert entry["source"] == "quaternius"
    assert entry["license"] == "CC0"
    assert entry["category"] == "prop"
    model = entry["models"]["house"]
    assert model["tris"] == 1
    assert model["bbox_min_m"] == [-1.0, 0.0, -1.0]
    assert model["bbox_max_m"] == [1.0, 2.0, 1.0]


def test_pull_reports_a_missing_staged_zip(tmp_path, spec):
    report = quaternius.pull([spec], tmp_path / "kits", tmp_path / "lock.json", staging_dir=tmp_path / "_staging")
    assert report["failed"][0]["pack"] == "test_pack"
    assert "no staged zip" in report["failed"][0]["error"]


def test_pull_keeps_when_the_staged_zip_is_unchanged(tmp_path, spec, staged):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    quaternius.pull([spec], kits_dir, lock_path, staging_dir=staged)

    report = quaternius.pull([spec], kits_dir, lock_path, staging_dir=staged)
    assert report["kept"] == ["test_pack"]


def test_pull_ignores_non_quaternius_specs(tmp_path, staged):
    kenney_spec = PackSpec(name="other", slug="other", category="prop", source="kenney")
    report = quaternius.pull([kenney_spec], tmp_path / "kits", tmp_path / "lock.json", staging_dir=staged)
    assert report == {"pulled": [], "kept": [], "failed": []}


def test_pull_drops_quaternius_packs_removed_from_the_set(tmp_path, spec, staged):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    quaternius.pull([spec], kits_dir, lock_path, staging_dir=staged)
    quaternius.pull([], kits_dir, lock_path, staging_dir=staged)
    assert json.loads(lock_path.read_text())["packs"] == {}


def test_pull_leaves_kenney_sourced_locked_packs_alone(tmp_path, spec, staged):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    lock_path.write_text(
        json.dumps({"version": 1, "packs": {"kenney_pack": {"source": "kenney", "models": {}}}}),
        encoding="utf-8",
    )
    quaternius.pull([spec], kits_dir, lock_path, staging_dir=staged)
    packs = json.loads(lock_path.read_text())["packs"]
    assert "kenney_pack" in packs
    assert "test_pack" in packs
