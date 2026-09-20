from __future__ import annotations

import pytest

from assets import recipe
from terrain import config


@pytest.fixture(autouse=True)
def isolated_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ASSET_OUT_DIR", tmp_path)
    recipe._SESSIONS.clear()
    yield
    recipe._SESSIONS.clear()


def test_record_appends_in_order():
    session = recipe.open_session("hero-crate")
    session.record("bpy.ops.mesh.primitive_cube_add()")
    session.record("bpy.context.object.name = 'crate'")
    path = session.out_dir() / recipe.RECIPE_NAME
    body = path.read_text(encoding="utf-8")
    assert body.startswith(recipe.HEADER)
    assert body.index("primitive_cube_add") < body.index("name = 'crate'")


def test_recipe_is_runnable_shaped():
    session = recipe.open_session("hero-crate")
    session.record("bpy.ops.mesh.primitive_cube_add()")
    body = session.out_dir().joinpath(recipe.RECIPE_NAME).read_text(encoding="utf-8")
    assert "import bpy" in body


def test_open_session_reloads_snippets_after_forget():
    session = recipe.open_session("prop-barrel")
    session.record("bpy.ops.mesh.primitive_cylinder_add()")
    recipe.forget("prop-barrel")
    reloaded = recipe.open_session("prop-barrel")
    assert reloaded.snippets == ["bpy.ops.mesh.primitive_cylinder_add()"]


def test_empty_snippet_is_rejected():
    session = recipe.open_session("prop-crate")
    with pytest.raises(recipe.RecipeError):
        session.record("   ")


@pytest.mark.parametrize("name", ["Hero", "has space", "-leading-dash", ""])
def test_invalid_names_are_rejected(name):
    with pytest.raises(recipe.RecipeError):
        recipe.open_session(name)
