from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

from library import gltf, kenney

ZIP_URL = "https://kenney.nl/media/pages/assets/test-pack/x/kenney_test-pack.zip"


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


def _fake_zip(license_text: str = "Free Pack\nLicense: CC0\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("License.txt", license_text)
        archive.writestr("Models/GLB format/foo.glb", _minimal_glb())
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, text: str = "", content: bytes = b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, zip_bytes: bytes, zip_url: str = ZIP_URL):
        self.zip_bytes = zip_bytes
        self.zip_url = zip_url
        self.get_calls: list[str] = []

    def get(self, url, **kwargs):
        self.get_calls.append(url)
        if url == self.zip_url:
            return FakeResponse(content=self.zip_bytes)
        return FakeResponse(text=f"<a href='{self.zip_url}'>download</a>")


@pytest.fixture
def spec():
    return kenney.PackSpec(name="test_pack", slug="test-pack", category="prop")


@pytest.fixture
def session():
    return FakeSession(_fake_zip())


def test_load_set_reads_the_pinned_file():
    specs = kenney.load_set()
    names = {s.name for s in specs}
    assert {"city_kit_commercial", "nature", "graveyard"} <= names
    assert all(s.slug and s.category for s in specs)


def test_resolve_zip_url_finds_the_matching_href(spec, session):
    assert kenney.resolve_zip_url(spec.slug, session) == session.zip_url


def test_resolve_zip_url_ignores_unrelated_zips(spec):
    class Session:
        def get(self, url, **kwargs):
            return FakeResponse(text="<a href='https://kenney.itch.io/other.zip'>x</a>")

    with pytest.raises(kenney.KenneyError):
        kenney.resolve_zip_url(spec.slug, Session())


def test_pull_extracts_models_and_stats(tmp_path, spec, session):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    report = kenney.pull([spec], kits_dir, lock_path, session=session)
    assert report["pulled"] == ["test_pack"]
    assert (kits_dir / "test_pack" / "foo.glb").is_file()

    entry = json.loads(lock_path.read_text())["packs"]["test_pack"]
    assert entry["license"] == "CC0"
    assert entry["category"] == "prop"
    model = entry["models"]["foo"]
    assert model["tris"] == 1
    assert model["bbox_min_m"] == [-1.0, 0.0, -1.0]
    assert model["bbox_max_m"] == [1.0, 2.0, 1.0]
    assert model["path"] == "kits/test_pack/foo.glb"


def test_pull_keeps_when_zip_url_is_unchanged(tmp_path, spec, session):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    kenney.pull([spec], kits_dir, lock_path, session=session)
    session.get_calls.clear()

    report = kenney.pull([spec], kits_dir, lock_path, session=session)
    assert report["kept"] == ["test_pack"]
    assert all(not call.endswith(".zip") for call in session.get_calls)


def test_pull_rejects_a_non_cc0_pack(tmp_path, spec):
    session = FakeSession(_fake_zip(license_text="All rights reserved"))
    report = kenney.pull([spec], tmp_path / "kits", tmp_path / "lock.json", session=session)
    assert report["failed"][0]["pack"] == "test_pack"
    assert "CC0" in report["failed"][0]["error"]


def test_pull_drops_packs_removed_from_the_set(tmp_path, spec, session):
    kits_dir = tmp_path / "kits"
    lock_path = tmp_path / "kits.lock.json"
    kenney.pull([spec], kits_dir, lock_path, session=session)
    kenney.pull([], kits_dir, lock_path, session=session)
    assert json.loads(lock_path.read_text())["packs"] == {}


def test_committed_lock_covers_the_pinned_set():
    lock = kenney.load_lock()
    specs = kenney.load_set()
    assert set(lock["packs"]) == {s.name for s in specs}
    for spec in specs:
        entry = lock["packs"][spec.name]
        assert entry["slug"] == spec.slug
        assert entry["category"] == spec.category
        assert entry["license"] == "CC0"
        assert entry["models"]
