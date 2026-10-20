from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from assets import blender as blender_discovery
from assets import creature_spec

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "blender_scripts" / "creature_selftest.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _minimal() -> dict:
    return {
        "name": "biped_test",
        "seed": 1,
        "stance": "biped",
        "palette": {"hide": [0.5, 0.4, 0.3]},
        "bones": [
            {"name": "hips", "head": [0, 0, 0.5], "tail": [0, 0, 0.8], "radius": [0.1, 0.1]},
            {"name": "head", "parent": "hips", "head": [0, 0, 0.85], "tail": [0, 0, 0.95], "radius": [0.08, 0.08]},
            {"name": "leg.L", "parent": "hips", "head": [0.06, 0, 0.5], "tail": [0.06, 0, 0.05], "radius": [0.05, 0.05]},
        ],
    }


def test_catalogue_specs_are_valid_and_cover_the_acceptance_set():
    catalogue = creature_spec.catalogue()
    stances = [species.stance for species in catalogue.values()]
    assert stances.count("quadruped") >= 3
    assert stances.count("biped") >= 1
    for species in catalogue.values():
        assert creature_spec.TRI_RANGE[0] <= species.target_tris <= creature_spec.TRI_RANGE[1]
        assert len(species.feet()) >= creature_spec.STANCES[species.stance]


def test_left_bones_mirror_to_right_with_mirrored_parents():
    species = creature_spec.parse(_minimal())
    right = species.bone("leg.R")
    assert right.head == (-0.06, 0.0, 0.5)
    assert right.parent == "hips"
    deer = creature_spec.catalogue()["deer"]
    assert deer.bone("tine_top.R").parent == "antler.R"


def test_minimal_spec_parses_with_defaults():
    species = creature_spec.parse(_minimal())
    assert [bone.name for bone in species.bones] == ["hips", "head", "leg.L", "leg.R"]
    assert species.regions() == ["hide"]
    low, high = species.bounds()
    assert low[2] == pytest.approx(0.0)
    assert high[2] == pytest.approx(1.03)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda spec: spec["bones"].append({"name": "arm.R", "parent": "hips", "head": [-0.1, 0, 0.7], "tail": [-0.3, 0, 0.6], "radius": [0.03, 0.03]}), "mirrored"),
        (lambda spec: spec["bones"][1].update(head=[0, 0, 1.5], tail=[0, 0, 1.6]), "outside its parent"),
        (lambda spec: spec["bones"][1].update(parent="nobody"), "listed before"),
        (lambda spec: spec["bones"][1].pop("parent"), "exactly one root"),
        (lambda spec: spec["bones"][2].update(head=[-0.06, 0, 0.5]), "x > 0"),
        (lambda spec: spec["bones"][2].update(tail=[0.06, 0, 0.3]), "ground"),
        (lambda spec: spec.update(stance="quadruped"), "quadruped needs 4"),
        (lambda spec: spec["bones"][1].update(region="horn"), "no palette colour"),
        (lambda spec: spec["bones"][1].update(region="scales"), "region 'scales'"),
        (lambda spec: spec.update(target_tris=20000), "target_tris"),
        (lambda spec: spec["palette"].update(hide=[2, 0, 0]), "[0, 1]"),
        (lambda spec: spec.update(voxel_m=0.045) or spec["bones"][2].update(radius=[0.03, 0.03]), "below voxel_m"),
        (lambda spec: spec.update(name="Wolf"), "lowercase"),
        (lambda spec: spec.update(legs=4), "unknown keys"),
    ],
)
def test_errors_name_the_problem(mutate, fragment):
    spec = copy.deepcopy(_minimal())
    mutate(spec)
    problems = creature_spec.errors(spec)
    assert any(fragment in problem for problem in problems), problems
    with pytest.raises(creature_spec.SpecError):
        creature_spec.parse(spec)


def test_region_at_picks_nearest_bone_and_underside_by_normal():
    spec = _minimal()
    spec["palette"]["underside"] = [0.8, 0.8, 0.7]
    spec["palette"]["dark"] = [0.1, 0.1, 0.1]
    spec["bones"][1]["region"] = "dark"
    species = creature_spec.parse(spec)
    assert species.region_at((0.0, 0.0, 1.03), 1.0) == "dark"
    assert species.region_at((0.1, 0.0, 0.65), 0.0) == "hide"
    assert species.region_at((0.1, 0.0, 0.65), -0.9) == "underside"


def test_schema_lists_every_spec_key_and_region():
    properties = creature_spec.SPEC_SCHEMA["properties"]
    assert set(properties["palette"]["properties"]) == set(creature_spec.REGIONS)
    json.dumps(creature_spec.SPEC_SCHEMA)
    for path in creature_spec.SPECS_DIR.glob("*.toml"):
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert set(data) <= set(properties)
    assert "quadruped" in creature_spec.draft_prompt("a wolf", "quadruped")


def test_json_spec_loads(tmp_path):
    path = tmp_path / "biped_test.json"
    path.write_text(json.dumps(_minimal()), encoding="utf-8")
    assert creature_spec.load(path).name == "biped_test"


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.skipif(blender_discovery.find_blender() is None, reason="Blender executable not found")
def test_creature_builder_leaves_closed_grounded_painted_geometry(tmp_path):
    result_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            blender_discovery.require_blender(),
            "--background",
            "--factory-startup",
            "--python",
            str(SCRIPT_PATH),
            "--",
            str(REPO_ROOT),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result_path.is_file(), completed.stdout + completed.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload.get("ok"), payload
