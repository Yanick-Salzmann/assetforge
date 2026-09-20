from __future__ import annotations

import json
from pathlib import Path

import pytest

from library import materials, polyhaven


def lock_entry(**overrides) -> dict:
    entry = {
        "slug": "gravel",
        "resolution": "2k",
        "format": "jpg",
        "license": "CC0",
        "source": "https://polyhaven.com/a/gravel",
        "dimensions_mm": [2000.0, 2000.0],
        "categories": ["floor", "outdoor"],
        "files": {
            role: {"path": f"gravel/{role}.jpg", "url": f"https://x/{role}.jpg", "sha256": "0" * 64}
            for role in ("albedo", "normal", "roughness")
        },
    }
    entry.update(overrides)
    return entry


def written(tmp_path: Path, toml: str, entries: dict) -> tuple[Path, Path]:
    set_path = tmp_path / "materials.toml"
    lock_path = tmp_path / "materials.lock.json"
    set_path.write_text(toml, encoding="utf-8")
    lock_path.write_text(json.dumps({"version": 1, "materials": entries}), encoding="utf-8")
    return set_path, lock_path


BASE_TOML = """
[defaults]
resolution = "2k"
format = "jpg"
maps = ["Diffuse", "nor_gl", "Rough"]

[material.gravel]
slug = "gravel"
"""


def test_committed_index_matches_the_lockfile():
    assert materials.load().as_dict() == materials.build().as_dict()


def test_committed_index_covers_the_pinned_set():
    index = materials.load()
    assert index.names() == tuple(sorted(spec.name for spec in polyhaven.load_set()))
    assert len(index) == 10


def test_committed_index_files_exist_on_disk():
    assert materials.load().missing() == {}


def test_every_material_carries_provenance_and_scale():
    for name in materials.load():
        material = materials.load()[name]
        assert material.licence == "CC0"
        assert material.source.startswith("https://polyhaven.com/a/")
        assert materials.MIN_TILING_M <= material.tiling_m <= materials.MAX_TILING_M
        assert set(materials.REQUIRED_ROLES) <= set(material.maps)
        assert set(material.maps) <= set(materials.MAP_ROLES)


def test_tiling_comes_from_the_physical_dimensions(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    assert materials.build(set_path, lock_path)["gravel"].tiling_m == pytest.approx(2.0)


def test_toml_overrides_the_derived_tiling(tmp_path):
    set_path, lock_path = written(
        tmp_path, BASE_TOML + "tiling_m = 6.5\n", {"gravel": lock_entry()}
    )
    assert materials.build(set_path, lock_path)["gravel"].tiling_m == pytest.approx(6.5)


def test_absurd_tiling_is_rejected(tmp_path):
    set_path, lock_path = written(
        tmp_path, BASE_TOML + "tiling_m = 5000.0\n", {"gravel": lock_entry()}
    )
    with pytest.raises(materials.MaterialError):
        materials.build(set_path, lock_path)


def test_missing_dimensions_without_an_override_is_an_error(tmp_path):
    set_path, lock_path = written(
        tmp_path, BASE_TOML, {"gravel": lock_entry(dimensions_mm=None)}
    )
    with pytest.raises(materials.MaterialError):
        materials.build(set_path, lock_path)


def test_pinned_but_unlocked_material_is_an_error(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {})
    with pytest.raises(materials.MaterialError):
        materials.build(set_path, lock_path)


def test_missing_required_map_is_an_error(tmp_path):
    entry = lock_entry()
    del entry["files"]["normal"]
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": entry})
    with pytest.raises(materials.MaterialError):
        materials.build(set_path, lock_path)


def test_unknown_map_role_is_an_error(tmp_path):
    entry = lock_entry()
    entry["files"]["specular"] = {"path": "gravel/specular.jpg", "url": "x", "sha256": "0" * 64}
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": entry})
    with pytest.raises(materials.MaterialError):
        materials.build(set_path, lock_path)


def test_unknown_material_name_lists_the_known_ones(tmp_path):
    index = materials.load()
    with pytest.raises(materials.MaterialError) as error:
        index["basalt"]
    assert "lush_grass" in str(error.value)


def test_unknown_map_role_lists_the_known_ones():
    material = materials.load()["gravel"]
    with pytest.raises(materials.MaterialError) as error:
        material.path("specular")
    assert "albedo" in str(error.value)


def test_paths_resolve_under_the_library():
    material = materials.load()["gravel"]
    albedo = material.path("albedo")
    assert albedo.is_file()
    assert albedo.parent.name == "gravel"
    assert albedo.parent.parent == materials.MATERIALS_DIR


def test_maps_are_relative_so_the_index_is_portable():
    for relative in materials.load()["gravel"].maps.values():
        assert not Path(relative).is_absolute()
        assert relative.startswith("materials/gravel/")


def test_missing_reports_the_absent_roles(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    index = materials.build(set_path, lock_path)
    assert index.missing(tmp_path) == {"gravel": ("albedo", "normal", "roughness")}


def test_round_trip_through_json(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    index = materials.build(set_path, lock_path)
    out = materials.write(index, tmp_path / "materials.json")
    assert materials.load(out).as_dict() == index.as_dict()


def test_load_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "materials.json"
    path.write_text(json.dumps({"version": 99, "materials": {}}), encoding="utf-8")
    with pytest.raises(materials.MaterialError):
        materials.load(path)


def test_load_reports_a_missing_index(tmp_path):
    with pytest.raises(materials.MaterialError):
        materials.load(tmp_path / "nothing.json")


def test_build_prefixes_maps_with_the_given_materials_dir(tmp_path):
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    other_dir = tmp_path / "asset_materials"
    index = materials.build(set_path, lock_path, other_dir)
    assert index["gravel"].maps["albedo"] == "asset_materials/gravel/albedo.jpg"


def test_committed_asset_index_matches_its_lockfile():
    assert materials.load_assets().as_dict() == materials.build_assets().as_dict()


def test_committed_asset_index_covers_the_pinned_set():
    index = materials.load_assets()
    asset_specs = polyhaven.load_set(materials.ASSET_SET_FILE)
    assert index.names() == tuple(sorted(spec.name for spec in asset_specs))
    assert {"concrete_cc0", "brick_cc0", "metal_cc0"} <= set(index.names())


def test_committed_asset_index_files_exist_on_disk():
    assert materials.load_assets().missing() == {}


def test_asset_maps_are_prefixed_with_asset_materials():
    for material_name in materials.load_assets():
        for relative in materials.load_assets()[material_name].maps.values():
            assert relative.startswith("asset_materials/")


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
    def __init__(self, ok: bool = True):
        self.ok = ok

    def get(self, url, **kwargs):
        if url.startswith(f"{polyhaven.API_ROOT}/info/"):
            return FakeResponse(payload={"dimensions": [2000, 2000], "categories": ["wood"]})
        if url.startswith(polyhaven.API_ROOT):
            if not self.ok:
                raise polyhaven.requests.RequestException("boom")
            return FakeResponse(
                payload={
                    "Diffuse": {"2k": {"jpg": {"url": "https://x/diff.jpg", "size": 3, "md5": "a"}}},
                    "nor_gl": {"2k": {"jpg": {"url": "https://x/nor.jpg", "size": 3, "md5": "b"}}},
                    "Rough": {"2k": {"jpg": {"url": "https://x/rough.jpg", "size": 3, "md5": "c"}}},
                }
            )
        return FakeResponse(body=b"abc")


def asset_test_dirs(tmp_path: Path) -> dict:
    set_path, lock_path = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    return {
        "set_path": set_path,
        "materials_dir": tmp_path / "materials",
        "lock_path": lock_path,
        "index_path": tmp_path / "materials.json",
    }


def test_add_material_spec_appends_a_new_entry(tmp_path):
    set_path, _ = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    materials.add_material_spec("wood_cc0", "weathered_planks", set_path=set_path)
    specs = {s.name: s for s in polyhaven.load_set(set_path)}
    assert specs["wood_cc0"].slug == "weathered_planks"
    assert specs["gravel"].slug == "gravel"


def test_add_material_spec_can_override_tiling(tmp_path):
    set_path, _ = written(tmp_path, BASE_TOML, {"gravel": lock_entry()})
    materials.add_material_spec("wood_cc0", "weathered_planks", tiling_m=1.5, set_path=set_path)
    specs = {s.name: s for s in polyhaven.load_set(set_path)}
    assert specs["wood_cc0"].tiling_m == pytest.approx(1.5)


def test_pull_asset_material_downloads_locks_and_reindexes(tmp_path):
    dirs = asset_test_dirs(tmp_path)
    result = materials.pull_asset_material(
        "wood_cc0", "weathered_planks", session=FakeSession(), **dirs
    )
    assert result["material"]["slug"] == "weathered_planks"

    index = materials.build(dirs["set_path"], dirs["lock_path"], dirs["materials_dir"])
    assert "wood_cc0" in index.names()
    assert (dirs["materials_dir"] / "wood_cc0" / "albedo.jpg").is_file()
    on_disk = json.loads(dirs["index_path"].read_text(encoding="utf-8"))
    assert "wood_cc0" in on_disk["materials"]


def test_pull_asset_material_reverts_the_toml_on_failure(tmp_path):
    dirs = asset_test_dirs(tmp_path)
    original = dirs["set_path"].read_text(encoding="utf-8")

    with pytest.raises(Exception):
        materials.pull_asset_material(
            "wood_cc0", "missing_slug", session=FakeSession(ok=False), **dirs
        )

    assert dirs["set_path"].read_text(encoding="utf-8") == original
    assert "wood_cc0" not in {s.name for s in polyhaven.load_set(dirs["set_path"])}
