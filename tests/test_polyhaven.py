import json

import pytest

from library import polyhaven


class FakeResponse:
    def __init__(self, payload=None, body=b""):
        self._payload = payload
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, payload, body, info=None):
        self.payload = payload
        self.body = body
        self.info = {"dimensions": [2000, 2000], "categories": ["floor"]} if info is None else info
        self.downloads = 0

    def get(self, url, **kwargs):
        if url.startswith(f"{polyhaven.API_ROOT}/info/"):
            return FakeResponse(payload=self.info)
        if url.startswith(polyhaven.API_ROOT):
            return FakeResponse(payload=self.payload)
        self.downloads += 1
        return FakeResponse(body=self.body)


@pytest.fixture
def spec():
    return polyhaven.MaterialSpec(
        name="gravel", slug="gravel", resolution="2k", fmt="jpg", maps=("Diffuse", "Rough")
    )


@pytest.fixture
def session():
    payload = {
        "Diffuse": {"2k": {"jpg": {"url": "https://x/diff.jpg", "size": 3, "md5": "a"}}},
        "Rough": {"2k": {"jpg": {"url": "https://x/rough.jpg", "size": 3, "md5": "b"}}},
    }
    return FakeSession(payload, b"abc")


def test_load_set_reads_the_pinned_file():
    specs = polyhaven.load_set()
    names = {s.name for s in specs}
    assert {"lush_grass", "cliff_rock", "gravel", "snow"} <= names
    assert all(s.resolution and s.fmt and s.maps for s in specs)


def test_every_pinned_map_has_a_role():
    for spec in polyhaven.load_set():
        for map_name in spec.maps:
            assert map_name in polyhaven.MAP_ROLES


def test_pull_downloads_then_keeps(tmp_path, spec, session):
    materials = tmp_path / "materials"
    lock = tmp_path / "lock.json"

    first = polyhaven.pull([spec], materials, lock, session=session)
    assert sorted(first["downloaded"]) == ["gravel/albedo", "gravel/roughness"]
    assert (materials / "gravel" / "albedo.jpg").read_bytes() == b"abc"
    assert session.downloads == 2

    second = polyhaven.pull([spec], materials, lock, session=session)
    assert second["downloaded"] == []
    assert sorted(second["kept"]) == ["gravel/albedo", "gravel/roughness"]
    assert session.downloads == 2


def test_lock_records_provenance(tmp_path, spec, session):
    lock_path = tmp_path / "lock.json"
    polyhaven.pull([spec], tmp_path / "materials", lock_path, session=session)
    entry = json.loads(lock_path.read_text())["materials"]["gravel"]
    assert entry["license"] == "CC0"
    assert entry["source"].endswith("/gravel")
    assert entry["dimensions_mm"] == [2000.0, 2000.0]
    assert entry["categories"] == ["floor"]
    albedo = entry["files"]["albedo"]
    assert albedo["sha256"] == polyhaven.hashlib.sha256(b"abc").hexdigest()
    assert albedo["url"] == "https://x/diff.jpg"


def test_verify_redownloads_corrupted_file(tmp_path, spec, session):
    materials = tmp_path / "materials"
    lock = tmp_path / "lock.json"
    polyhaven.pull([spec], materials, lock, session=session)
    (materials / "gravel" / "albedo.jpg").write_bytes(b"corrupt")

    report = polyhaven.pull([spec], materials, lock, verify=True, session=session)
    assert report["downloaded"] == ["gravel/albedo"]
    assert (materials / "gravel" / "albedo.jpg").read_bytes() == b"abc"


def test_pull_drops_materials_removed_from_the_set(tmp_path, spec, session):
    materials = tmp_path / "materials"
    lock = tmp_path / "lock.json"
    polyhaven.pull([spec], materials, lock, session=session)
    polyhaven.pull([], materials, lock, session=session)
    assert json.loads(lock.read_text())["materials"] == {}


def test_missing_map_is_reported_not_raised(tmp_path, session):
    spec = polyhaven.MaterialSpec("x", "x", "8k", "jpg", ("Diffuse",))
    report = polyhaven.pull([spec], tmp_path / "m", tmp_path / "l.json", session=session)
    assert report["failed"][0]["material"] == "x"
    assert report["downloaded"] == []


def test_committed_lock_covers_the_pinned_set():
    lock = polyhaven.load_lock()
    specs = polyhaven.load_set()
    assert set(lock["materials"]) == {s.name for s in specs}
    for spec in specs:
        entry = lock["materials"][spec.name]
        assert entry["slug"] == spec.slug
        assert entry["resolution"] == spec.resolution
        assert entry["format"] == spec.fmt
        assert sorted(entry["files"]) == sorted(polyhaven.MAP_ROLES[m] for m in spec.maps)
        for meta in entry["files"].values():
            assert len(meta["sha256"]) == 64


def test_search_textures_filters_by_keyword(tmp_path):
    cache = tmp_path / "catalog.json"
    catalog = {
        "weathered_planks": {"name": "Weathered Planks", "categories": ["wood", "floor"]},
        "concrete_wall_001": {"name": "Concrete Wall", "categories": ["concrete", "wall"]},
    }
    cache.write_text(json.dumps(catalog), encoding="utf-8")

    results = polyhaven.search_textures("wood", cache_path=cache)
    assert [r["slug"] for r in results] == ["weathered_planks"]

    results = polyhaven.search_textures("", cache_path=cache)
    assert {r["slug"] for r in results} == {"weathered_planks", "concrete_wall_001"}


def test_search_textures_matches_category_and_name(tmp_path):
    cache = tmp_path / "catalog.json"
    cache.write_text(
        json.dumps({"gravel_02": {"name": "Gravel", "categories": ["outdoor", "ground"]}}),
        encoding="utf-8",
    )
    assert [r["slug"] for r in polyhaven.search_textures("ground", cache_path=cache)] == ["gravel_02"]
    assert [r["slug"] for r in polyhaven.search_textures("Gravel", cache_path=cache)] == ["gravel_02"]


def test_fetch_texture_catalog_uses_fresh_cache_without_a_network_call(tmp_path):
    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"a": {"name": "a", "categories": []}}), encoding="utf-8")

    class ExplodingSession:
        def get(self, *args, **kwargs):
            raise AssertionError("should not hit the network when the cache is fresh")

    catalog = polyhaven.fetch_texture_catalog(session=ExplodingSession(), cache_path=cache, max_age_s=3600)
    assert catalog == {"a": {"name": "a", "categories": []}}


def test_fetch_texture_catalog_refetches_when_the_cache_is_stale(tmp_path):
    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"old": {}}), encoding="utf-8")
    fresh = {"new": {"name": "new", "categories": []}}

    class StaleSession:
        def get(self, url, **kwargs):
            return FakeResponse(payload=fresh)

    catalog = polyhaven.fetch_texture_catalog(session=StaleSession(), cache_path=cache, max_age_s=-1)
    assert catalog == fresh
    assert json.loads(cache.read_text(encoding="utf-8")) == fresh


def test_missing_dimensions_is_reported_not_raised(tmp_path, spec, session):
    session.info = {"categories": ["floor"]}
    report = polyhaven.pull([spec], tmp_path / "m", tmp_path / "l.json", session=session)
    assert report["failed"][0]["material"] == "gravel"
    assert report["downloaded"] == []


def test_resolving_again_keeps_the_recorded_hashes(tmp_path, spec, session):
    materials = tmp_path / "materials"
    lock = tmp_path / "lock.json"
    polyhaven.pull([spec], materials, lock, session=session)
    stored = json.loads(lock.read_text())["materials"]["gravel"]
    del stored["dimensions_mm"]
    lock.write_text(json.dumps({"version": 1, "materials": {"gravel": stored}}))

    report = polyhaven.pull([spec], materials, lock, session=session)
    assert report["downloaded"] == []
    assert sorted(report["kept"]) == ["gravel/albedo", "gravel/roughness"]
    entry = json.loads(lock.read_text())["materials"]["gravel"]
    assert entry["dimensions_mm"] == [2000.0, 2000.0]
    assert entry["files"]["albedo"]["sha256"] == polyhaven.hashlib.sha256(b"abc").hexdigest()


def test_committed_lock_records_physical_size():
    lock = polyhaven.load_lock()
    for name, entry in lock["materials"].items():
        assert len(entry["dimensions_mm"]) == 2, name
        assert entry["dimensions_mm"][0] > 0.0, name
