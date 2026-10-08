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
