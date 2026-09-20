from __future__ import annotations

import pytest

from assets import blender as blender_discovery
from assets import demo
from assets.recipe import RecipeError
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    yield


def test_build_rejects_a_bad_name():
    with pytest.raises(RecipeError):
        demo.build("Not Valid")


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_build_exports_a_glb_end_to_end():
    result = demo.build("demo-selftest", timeout_s=180.0)

    assert result.glb_path.is_file()
    assert result.triangle_count > 0
    assert result.object_names
