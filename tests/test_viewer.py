from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from viewer import server as viewer


def _write_asset(root, name, kind="prop", counts=None, glb=True):
    directory = root / name
    directory.mkdir()
    manifest = {"name": name, "kind": kind, "glb": f"{name}.glb", "triangle_counts": counts or {"LOD0": 12}}
    (directory / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")
    if glb:
        (directory / f"{name}.glb").write_bytes(b"glTF-bytes")
    return directory


def test_list_assets_reports_exported_assets_with_ordered_lods(tmp_path):
    _write_asset(tmp_path, "tavern", kind="hero_building", counts={"LOD2": 10, "LOD0": 100, "LOD1": 50})
    _write_asset(tmp_path, "barrel")
    assets = viewer.list_assets(tmp_path)
    assert [asset["name"] for asset in assets] == ["barrel", "tavern"]
    tavern = assets[1]
    assert tavern["kind"] == "hero_building"
    assert tavern["lods"] == ["LOD0", "LOD1", "LOD2"]
    assert tavern["glb"] == "/files/assets/tavern/tavern.glb"


def test_list_assets_skips_directories_without_manifest_or_glb(tmp_path):
    (tmp_path / "empty").mkdir()
    _write_asset(tmp_path, "unexported", glb=False)
    assert viewer.list_assets(tmp_path) == []


def test_list_assets_of_a_missing_root_is_empty(tmp_path):
    assert viewer.list_assets(tmp_path / "absent") == []


def test_resolve_under_rejects_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    assert viewer.resolve_under(root, "inside.txt") == (root / "inside.txt").resolve()
    assert viewer.resolve_under(root, "../outside.txt") is None
    assert viewer.resolve_under(root, "%2e%2e/outside.txt") is None
    assert viewer.resolve_under(root, "missing.txt") is None


@pytest.fixture
def running(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    _write_asset(asset_root, "barrel")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    httpd = viewer.make_server(port=0, asset_root=asset_root)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def test_server_serves_index_listing_and_glb(running):
    status, content_type, body = _get(f"{running}/")
    assert status == 200 and "text/html" in content_type and b"viewer.js" in body
    status, content_type, body = _get(f"{running}/api/assets")
    assert json.loads(body)[0]["name"] == "barrel"
    status, content_type, body = _get(f"{running}/files/assets/barrel/barrel.glb")
    assert content_type == "model/gltf-binary" and body == b"glTF-bytes"
    status, content_type, _ = _get(f"{running}/static/viewer.js")
    assert "javascript" in content_type


def test_server_refuses_paths_outside_its_roots(running):
    for path in ("/files/assets/../secret.txt", "/files/assets/%2e%2e/secret.txt", "/static/../server.py", "/nope"):
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(f"{running}{path}")
        assert error.value.code == 404


def _write_terrain(root, name, resolution=4):
    import numpy as np
    from PIL import Image

    directory = root / name
    directory.mkdir(parents=True)
    height = (np.arange(resolution * resolution, dtype=np.uint16).reshape(resolution, resolution) * 1000).astype(np.uint16)
    Image.fromarray(height).save(directory / "height.png")
    Image.fromarray(np.full((resolution, resolution), 7, np.uint8), "L").save(directory / "water.png")
    splat = np.zeros((resolution, resolution, 4), np.uint8)
    splat[..., 0] = 200
    splat[..., 3] = 55
    Image.fromarray(splat, "RGBA").save(directory / "splat_0.png")
    Image.fromarray(np.full((resolution, resolution), 128, np.uint8), "L").save(directory / "scatter_rock.png")
    manifest = {
        "schema_version": 1,
        "name": name,
        "seed": 3,
        "resolution": resolution,
        "world_size_m": 16.0,
        "height_range_m": 100.0,
        "metres_per_pixel": 4.0,
        "heightmap": "height.png",
        "water_mask": "water.png",
        "water": {"sea_level_m": 0.0, "covered_fraction": 0.0, "mean_depth_m": 0.0, "max_depth_m": 0.0, "depth_reference_m": 12.0},
        "splat": {
            "biome": "test",
            "sharpness": 3.0,
            "textures": ["splat_0.png"],
            "layers": [
                {"layer": "rock", "material": "cliff_rock", "tiling_m": 2.0, "index": 0, "texture": "splat_0.png", "channel": "r"},
                {"layer": "sand", "material": "unknown_sand", "tiling_m": 4.0, "index": 1, "texture": "splat_0.png", "channel": "a"},
            ],
        },
        "scatter": [{"kind": "rock", "path": "scatter_rock.png"}],
        "normal_map": None,
        "rule_path": None,
    }
    (directory / "terrain.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, height


def _write_library(root):
    (root / "materials" / "cliff_rock").mkdir(parents=True)
    (root / "materials" / "cliff_rock" / "albedo.jpg").write_bytes(b"jpg")
    index = {
        "version": 1,
        "materials": {
            "cliff_rock": {
                "maps": {
                    "albedo": "materials/cliff_rock/albedo.jpg",
                    "normal": "materials/cliff_rock/normal.jpg",
                }
            }
        },
    }
    (root / "materials.json").write_text(json.dumps(index), encoding="utf-8")


def test_list_terrains_reports_exported_terrains_with_resolved_materials(tmp_path):
    terrain_root = tmp_path / "terrain"
    library_root = tmp_path / "library"
    _write_terrain(terrain_root, "vale")
    (terrain_root / "unexported").mkdir()
    broken = terrain_root / "broken"
    broken.mkdir()
    (broken / "terrain.json").write_text("{}", encoding="utf-8")
    _write_library(library_root)
    terrains = viewer.list_terrains(terrain_root, library_root)
    assert [terrain["name"] for terrain in terrains] == ["vale"]
    vale = terrains[0]
    assert vale["files"] == "/files/terrain/vale/"
    assert vale["raw"] == "/api/raw/terrain/vale/"
    assert vale["manifest"]["world_size_m"] == 16.0
    assert vale["materials"] == {
        "cliff_rock": {"albedo": "/files/library/materials/cliff_rock/albedo.jpg"},
        "unknown_sand": {},
    }


def test_list_terrains_skips_a_manifest_whose_images_are_missing(tmp_path):
    directory, _ = _write_terrain(tmp_path, "vale")
    (directory / "scatter_rock.png").unlink()
    assert viewer.list_terrains(tmp_path, tmp_path / "library") == []


def test_raw_image_preserves_sixteen_bit_and_rgba_samples(tmp_path):
    import numpy as np

    directory, height = _write_terrain(tmp_path, "vale")
    body, headers = viewer.raw_image(directory / "height.png")
    assert headers == {"X-Width": "4", "X-Height": "4", "X-Channels": "1", "X-Bits": "16"}
    assert np.array_equal(np.frombuffer(body, "<u2").reshape(4, 4), height)
    body, headers = viewer.raw_image(directory / "splat_0.png")
    assert headers["X-Channels"] == "4" and headers["X-Bits"] == "8"
    samples = np.frombuffer(body, np.uint8).reshape(4, 4, 4)
    assert samples[0, 0].tolist() == [200, 0, 0, 55]
    body, headers = viewer.raw_image(directory / "water.png")
    assert headers["X-Channels"] == "1" and set(body) == {7}


@pytest.fixture
def running_terrain(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    terrain_root = tmp_path / "terrain"
    library_root = tmp_path / "library"
    _write_terrain(terrain_root, "vale")
    _write_library(library_root)
    (tmp_path / "secret.png").write_bytes(b"secret")
    httpd = viewer.make_server(port=0, asset_root=asset_root, terrain_root=terrain_root, library_root=library_root)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_server_serves_terrain_listing_raw_samples_and_library_maps(running_terrain):
    _, _, body = _get(f"{running_terrain}/api/terrains")
    assert json.loads(body)[0]["name"] == "vale"
    with urllib.request.urlopen(f"{running_terrain}/api/raw/terrain/vale/height.png", timeout=5) as response:
        assert response.headers["X-Bits"] == "16"
        assert len(response.read()) == 4 * 4 * 2
    status, content_type, _ = _get(f"{running_terrain}/files/terrain/vale/terrain.json")
    assert status == 200 and "json" in content_type
    _, _, body = _get(f"{running_terrain}/files/library/materials/cliff_rock/albedo.jpg")
    assert body == b"jpg"


def test_server_refuses_terrain_paths_outside_its_roots(running_terrain):
    for path in (
        "/api/raw/terrain/../secret.png",
        "/api/raw/terrain/%2e%2e/secret.png",
        "/api/raw/terrain/vale/terrain.json",
        "/files/terrain/../secret.png",
        "/files/library/%2e%2e/secret.png",
    ):
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(f"{running_terrain}{path}")
        assert error.value.code == 404
