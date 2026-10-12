from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from assets import prop_set
from assets.recipe import RecipeError

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "assets" / "blender_scripts" / "prop_set_build.py"


class FakeObject(dict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name


def test_group_follows_tags_not_names():
    lid = prop_set.tag(FakeObject("tavern_crate_lid"), "tavern_crate")
    body = prop_set.tag(FakeObject("tavern_crate_body"), "tavern_crate")
    barrel = prop_set.tag(FakeObject("tavern_old_barrel_hoop"), "tavern_old_barrel")
    stray = FakeObject("tavern_crate_helper")

    groups = prop_set.group([lid, barrel, stray, body])

    assert list(groups) == ["tavern_crate", "tavern_old_barrel"]
    assert groups["tavern_crate"] == [lid, body]
    assert groups["tavern_old_barrel"] == [barrel]


def test_tag_rejects_names_that_cannot_be_asset_directories():
    with pytest.raises(RecipeError):
        prop_set.tag(FakeObject("x"), "Tavern Crate")


SCENE_SETUP = """
import sys
sys.path.insert(0, {repo!r})
import bpy
from assets import hardsurface as hs, materials, prop_set
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
crate = hs.bevelled_box("set_a_crate_body", size=(0.6, 0.6, 0.6), bevel_width=0.02)
materials.assign_material(crate, None, "wood")
prop_set.tag(crate, "set_a_crate")
lid = hs.bevelled_box("set_a_crate_lid", size=(0.62, 0.62, 0.05), bevel_width=0.01, location=(0.0, 0.0, 0.6))
materials.assign_material(lid, None, "metal")
prop_set.tag(lid, "set_a_crate")
post = hs.frustum("set_a_post_shaft", 0.08, 0.06, 1.2, location=(2.0, 0.0, 0.0))
materials.assign_material(post, None, "metal")
prop_set.tag(post, "set_a_post")
hs.bevelled_box("untagged_helper", size=(0.1, 0.1, 0.1), location=(5.0, 0.0, 0.0))
bpy.ops.wm.save_as_mainfile(filepath={blend!r})
"""


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_prop_set_build_bakes_one_atlas_and_exports_each_prop(tmp_path):
    blender = blender_discovery.require_blender()
    blend_path = tmp_path / "set_a.blend"
    result_path = tmp_path / "result.json"
    env = {**os.environ, "ASSETFORGE_WORKSPACE": str(tmp_path)}
    setup = subprocess.run(
        [blender, "--background", "--factory-startup", "--python-expr", SCENE_SETUP.format(repo=str(REPO_ROOT), blend=str(blend_path))],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert blend_path.is_file(), setup.stdout + setup.stderr

    completed = subprocess.run(
        [blender, "--background", str(blend_path), "--factory-startup", "--python", str(SCRIPT_PATH), "--", str(REPO_ROOT), str(result_path)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"], payload
    assert sorted(payload["props"]) == ["set_a_crate", "set_a_post"]
    for prop in ("set_a_crate", "set_a_post"):
        assert (tmp_path / "out" / "assets" / prop / f"{prop}.glb").is_file()
    assert not (tmp_path / "out" / "assets" / "untagged_helper").exists()
