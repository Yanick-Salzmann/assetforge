from __future__ import annotations

import json
from pathlib import Path

import pytest

from library import kenney, kits


def lock_entry(**overrides) -> dict:
    entry = {
        "source": "kenney",
        "slug": "test-pack",
        "category": "prop",
        "license": "CC0",
        "source_url": "https://kenney.nl/assets/test-pack",
        "zip_url": "https://kenney.nl/media/pages/assets/test-pack/x/kenney_test-pack.zip",
        "zip_sha256": "0" * 64,
        "models": {
            "foo": {
                "path": "kits/test_pack/foo.glb",
                "tris": 120,
                "bbox_min_m": [-0.5, 0.0, -0.5],
                "bbox_max_m": [0.5, 1.0, 0.5],
            }
        },
    }
    entry.update(overrides)
    return entry


def written(tmp_path: Path, toml: str, entries: dict) -> tuple[Path, Path]:
    set_path = tmp_path / "kits.toml"
    lock_path = tmp_path / "kits.lock.json"
    set_path.write_text(toml, encoding="utf-8")
    lock_path.write_text(json.dumps({"version": 1, "packs": entries}), encoding="utf-8")
    return set_path, lock_path


BASE_TOML = """
[defaults]
category = "prop"

[pack.test_pack]
source = "kenney"
slug = "test-pack"
"""


def test_committed_index_matches_the_lockfile():
    assert kits.load().as_dict() == kits.build().as_dict()


def test_committed_index_covers_the_pinned_packs():
    lock = kenney.load_lock()
    expected = sum(len(pack["models"]) for pack in lock["packs"].values())
    assert len(kits.load()) == expected


def test_committed_index_files_exist_on_disk():
    assert kits.load().missing() == ()


def test_every_prop_carries_provenance():
    index = kits.load()
    for name in index:
        prop = index[name]
        assert prop.licence == "CC0"
        assert prop.source == "kenney"
        assert prop.category in index.categories()
        assert prop.tris > 0


def test_dimensions_are_non_negative():
    index = kits.load()
    for name in index:
        assert all(dimension >= 0.0 for dimension in index[name].dimensions_m)


def test_find_filters_by_category_tris_and_size(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"test_pack": lock_entry()})
    index = kits.build(set_path, lock_path)
    assert index.find(category="prop") == (index["test_pack/foo"],)
    assert index.find(category="building") == ()
    assert index.find(max_tris=50) == ()
    assert index.find(max_dimension_m=0.5) == ()
    assert index.find(max_dimension_m=1.0) == (index["test_pack/foo"],)


def test_find_sorts_by_tris_ascending(tmp_path):
    entries = {
        "test_pack": lock_entry(
            models={
                "big": {
                    "path": "kits/test_pack/big.glb",
                    "tris": 900,
                    "bbox_min_m": [0.0, 0.0, 0.0],
                    "bbox_max_m": [1.0, 1.0, 1.0],
                },
                "small": {
                    "path": "kits/test_pack/small.glb",
                    "tris": 100,
                    "bbox_min_m": [0.0, 0.0, 0.0],
                    "bbox_max_m": [1.0, 1.0, 1.0],
                },
            }
        )
    }
    set_path, lock_path = written(tmp_path, BASE_TOML, entries)
    index = kits.build(set_path, lock_path)
    ordered = index.find(category="prop")
    assert [prop.name for prop in ordered] == ["test_pack/small", "test_pack/big"]


def test_pinned_but_unlocked_pack_is_an_error(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {})
    with pytest.raises(kits.KitError):
        kits.build(set_path, lock_path)


def test_unknown_prop_name_lists_the_known_ones(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"test_pack": lock_entry()})
    index = kits.build(set_path, lock_path)
    with pytest.raises(kits.KitError) as error:
        index["nope"]
    assert "test_pack/foo" in str(error.value)


def test_round_trip_through_json(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"test_pack": lock_entry()})
    index = kits.build(set_path, lock_path)
    out = kits.write(index, tmp_path / "kits.json")
    assert kits.load(out).as_dict() == index.as_dict()


def test_load_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "kits.json"
    path.write_text(json.dumps({"version": 99, "props": {}}), encoding="utf-8")
    with pytest.raises(kits.KitError):
        kits.load(path)


def test_load_reports_a_missing_index(tmp_path):
    with pytest.raises(kits.KitError):
        kits.load(tmp_path / "nothing.json")


def test_missing_reports_absent_props(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"test_pack": lock_entry()})
    index = kits.build(set_path, lock_path)
    assert index.missing(tmp_path) == ("test_pack/foo",)
