from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from assets import prop_manifest
from library import kits
from terrain import config

REPO_ROOT = Path(__file__).resolve().parent.parent
BEAUTY_SCRIPT = REPO_ROOT / "terrain" / "blender_scripts" / "beauty_render.py"
POT = "nature/pot_small"


def entry(**overrides):
    base = {"mesh": POT, "translation": [1.0, 0.0, -2.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": 1.0}
    return base | overrides


def test_prop_entry_turns_blender_design_coordinates_into_the_glb_frame():
    placed = prop_manifest.prop_entry(POT, (4.0, 2.5, 0.25), (0.0, 0.0, math.pi / 2.0), 2.0, shift=(3.0, 2.0, 0.0))
    assert placed["translation"] == pytest.approx([1.0, 0.25, -0.5])
    assert placed["rotation"] == pytest.approx([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    assert placed["scale"] == 2.0


def test_prop_entry_turns_a_tilt_about_blender_x_into_a_tilt_about_gltf_x():
    placed = prop_manifest.prop_entry(POT, (0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0))
    assert placed["rotation"] == pytest.approx([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])


def test_prop_entry_turns_a_tilt_about_blender_y_into_a_tilt_about_gltf_minus_z():
    placed = prop_manifest.prop_entry(POT, (0.0, 0.0, 0.0), (0.0, math.pi / 2.0, 0.0))
    assert placed["rotation"] == pytest.approx([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)])


def test_prop_entry_refuses_a_bad_mesh_name_or_scale():
    with pytest.raises(prop_manifest.PropManifestError, match="mesh"):
        prop_manifest.prop_entry("pot_small", (0.0, 0.0, 0.0))
    with pytest.raises(prop_manifest.PropManifestError, match="scale"):
        prop_manifest.prop_entry(POT, (0.0, 0.0, 0.0), scale=0.0)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"rotation": [0.0, 0.0, 0.5, 0.5]}, "unit quaternion"),
        ({"rotation": [0.0, 0.0, 1.0]}, "quaternion of four"),
        ({"translation": [0.0, math.nan, 0.0]}, "translation"),
        ({"scale": True}, "scale"),
        ({"colour": [1, 1, 1]}, "unknown field"),
    ],
)
def test_entry_errors_name_the_bad_field(overrides, fragment):
    errors = prop_manifest.entry_errors(entry(**overrides))
    assert any(fragment in error for error in errors), errors


def test_entry_errors_report_missing_fields():
    assert prop_manifest.entry_errors({"mesh": POT}) == [f"{POT}: missing translation, rotation, scale"]


def test_manifest_errors_flag_kit_props_the_library_does_not_carry():
    errors = prop_manifest.manifest_errors([entry(), entry(mesh="nature/unobtainium")], {POT})
    assert errors == ["unknown kit prop(s) nature/unobtainium"]
    assert prop_manifest.manifest_errors({}) == ["props must be a list"]


def test_outside_errors_allow_the_yard_but_not_the_next_field():
    bbox_min = [-2.0, 0.0, -2.0]
    bbox_max = [2.0, 5.0, 2.0]
    near = entry(translation=[2.0 + prop_manifest.REACH_M - 0.1, 0.0, 0.0])
    far = entry(translation=[2.0 + prop_manifest.REACH_M + 0.1, 0.0, 0.0])
    sunk = entry(translation=[0.0, -prop_manifest.SINK_M - 0.1, 0.0])
    assert prop_manifest.outside_errors([near], bbox_min, bbox_max) == []
    assert len(prop_manifest.outside_errors([far, sunk], bbox_min, bbox_max)) == 2


def test_load_rejects_a_malformed_list():
    with pytest.raises(prop_manifest.PropManifestError):
        prop_manifest.load({"props": [entry(scale=-1.0)]})
    assert prop_manifest.load({}) == []


def test_glb_paths_address_every_locked_model_by_pack_and_model(tmp_path):
    lock = tmp_path / "kits.lock.json"
    lock.write_text(
        json.dumps({"version": 1, "packs": {"nature": {"models": {"pot_small": {"path": "kits/nature/pot_small.glb"}}}}}),
        encoding="utf-8",
    )
    assert kits.glb_paths(lock) == {POT: "kits/nature/pot_small.glb"}
    assert kits.glb_paths(tmp_path / "absent.json") == {}


def _pot_glb() -> Path | None:
    relative = kits.glb_paths().get(POT)
    path = config.LIBRARY_DIR / relative if relative else None
    return path if path is not None and path.is_file() else None


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
@pytest.mark.skipif(_pot_glb() is None, reason="nature kit pack not pulled")
def test_a_prop_placed_in_blender_comes_back_where_it_was_placed_through_the_beauty_render(tmp_path):
    result_path = tmp_path / "props.json"
    expression = (
        "import json, os, runpy, sys\n"
        f"os.environ['ASSETFORGE_WORKSPACE'] = r'{REPO_ROOT}'\n"
        f"sys.path.insert(0, r'{REPO_ROOT}')\n"
        "import bpy\n"
        "from mathutils import Vector\n"
        "from assets import prop_manifest, props\n"
        f"module = runpy.run_path(r'{BEAUTY_SCRIPT}', run_name='beauty_render')\n"
        f"empty, _ = props.add_prop('{POT}', (4.0, 2.5, 0.3), (0.2, -0.1, 0.6), 1.3, label='pot')\n"
        "(placement,) = props.recorded_props()\n"
        "shift = (3.0, 2.0, 0.1)\n"
        "placed = prop_manifest.prop_entry(placement['mesh'], placement['location'], placement['rotation'], placement['scale'], shift)\n"
        "root = bpy.data.objects.new('Building_test', None)\n"
        "bpy.context.scene.collection.objects.link(root)\n"
        "root.location = Vector(shift)\n"
        "bpy.context.view_layer.update()\n"
        f"module['_build_props'](root, [dict(placed, glb=r'{_pot_glb()}')], {{}})\n"
        "bpy.context.view_layer.update()\n"
        "instance = bpy.data.objects['Building_test_prop0']\n"
        "difference = max(abs(a - b) for row_a, row_b in zip(instance.matrix_world, empty.matrix_world) for a, b in zip(row_a, row_b))\n"
        "shown = len(instance.instance_collection.all_objects)\n"
        f"open(r'{result_path}', 'w').write(json.dumps({{'difference': difference, 'shown': shown, 'recorded': placement['mesh']}}))\n"
    )
    completed = subprocess.run(
        [blender_discovery.require_blender(), "--background", "--factory-startup", "--python-expr", expression],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "ASSETFORGE_WORKSPACE": str(REPO_ROOT)},
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["recorded"] == POT
    assert result["shown"] >= 1
    assert result["difference"] < 1e-5
