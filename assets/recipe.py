from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from terrain import config

RECIPE_NAME = "recipe.py"
SNIPPETS_NAME = ".recipe_snippets.json"
HEADER = '"""Runnable under: blender --background --python recipe.py"""\n\nimport bpy\n'

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class RecipeError(ValueError):
    """Raised when an asset name or recipe snippet cannot be recorded."""


def validate_name(name: str) -> str:
    if not NAME_PATTERN.match(name):
        raise RecipeError(
            f"asset name {name!r} must be lowercase alphanumerics, dashes or underscores"
        )
    return name


@dataclass
class RecipeSession:
    """Every bpy snippet one live MCP session executed for one asset, in order."""

    name: str
    snippets: list[str] = field(default_factory=list)

    def out_dir(self) -> Path:
        return config.ASSET_OUT_DIR / self.name

    def record(self, code: str) -> RecipeSession:
        snippet = code.strip()
        if not snippet:
            raise RecipeError("recipe snippet must not be empty")
        self.snippets.append(snippet)
        self.write()
        return self

    def render(self) -> str:
        return HEADER + "".join(f"\n\n{snippet}" for snippet in self.snippets)

    def write(self) -> Path:
        target = self.out_dir()
        target.mkdir(parents=True, exist_ok=True)
        (target / SNIPPETS_NAME).write_text(json.dumps(self.snippets), encoding="utf-8")
        path = target / RECIPE_NAME
        path.write_text(self.render() + "\n", encoding="utf-8")
        return path


_SESSIONS: dict[str, RecipeSession] = {}


def open_session(name: str) -> RecipeSession:
    """The resident recipe session for this asset, reloaded from disk if the server restarted."""
    validate_name(name)
    session = _SESSIONS.get(name)
    if session is not None:
        return session
    session = RecipeSession(name=name, snippets=_load_snippets(name))
    _SESSIONS[name] = session
    return session


def _load_snippets(name: str) -> list[str]:
    path = config.ASSET_OUT_DIR / name / SNIPPETS_NAME
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def forget(name: str) -> None:
    _SESSIONS.pop(name, None)
