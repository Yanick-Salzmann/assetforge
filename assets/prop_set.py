"""Which prop of a set each live-scene object belongs to.

A prop set is modelled as one scene and baked onto one shared atlas, then exported one prop at a
time. Every part carries its prop's asset name in the PROP_PROPERTY custom property, set by tag,
so grouping never depends on how object or prop names are spelled. Pure Python, duck-typed over
bpy objects, so it runs both inside Blender and under pytest.
"""

from __future__ import annotations

from typing import Iterable, Protocol, TypeVar

from assets.recipe import validate_name

PROP_PROPERTY = "assetforge_prop"


class Taggable(Protocol):
    name: str

    def get(self, key: str, default=None): ...

    def __setitem__(self, key: str, value) -> None: ...


T = TypeVar("T", bound=Taggable)


def tag(obj: T, prop: str) -> T:
    """Mark obj as a part of the prop exported as out/assets/<prop>/."""
    obj[PROP_PROPERTY] = validate_name(prop)
    return obj


def group(objects: Iterable[T]) -> dict[str, list[T]]:
    """Tagged objects by prop name, props in first-seen order; untagged objects are left out."""
    groups: dict[str, list[T]] = {}
    for obj in objects:
        prop = obj.get(PROP_PROPERTY)
        if prop:
            groups.setdefault(str(prop), []).append(obj)
    return groups
