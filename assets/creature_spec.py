"""Species specs for procedural creatures. Stdlib only, so Blender's interpreter can import it.

A spec is a skeleton of tapered capsules in metres (Blender axes: +Z up, +Y forward, origin at the
base centre between the feet) plus a palette. Bones named `<part>.L` sit on +X and are mirrored to
`<part>.R`. assets/creature.py skins the capsules into one closed body; this module only parses,
validates and measures specs, so an LLM draft can be checked against SPEC_SCHEMA and errors()
before Blender ever sees it.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

Vec = tuple[float, float, float]

STANCES = {"quadruped": 4, "biped": 2}
REGIONS = ("hide", "underside", "mane", "dark", "horn", "eye")
MIRROR_SUFFIX = ".L"
MIRRORED_SUFFIX = ".R"
TRI_RANGE = (500, 8000)
GROUND_CONTACT_M = 0.03
GROUND_TOLERANCE_M = 0.05
MIN_RADIUS_M = 0.004
MAX_EXTENT_M = 12.0
VOXEL_RANGE_M = (0.004, 0.05)
SPECS_DIR = Path(__file__).resolve().parent / "creature_specs"


class SpecError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    head: Vec
    tail: Vec
    radius: tuple[float, float]
    region: str = "hide"

    def length(self) -> float:
        return _distance(self.head, self.tail)

    def mirrored(self) -> Bone:
        parent = self.parent
        if parent is not None and parent.endswith(MIRROR_SUFFIX):
            parent = parent[: -len(MIRROR_SUFFIX)] + MIRRORED_SUFFIX
        return Bone(
            name=self.name[: -len(MIRROR_SUFFIX)] + MIRRORED_SUFFIX,
            parent=parent,
            head=_flip(self.head),
            tail=_flip(self.tail),
            radius=self.radius,
            region=self.region,
        )

    def lowest(self) -> float:
        return min(self.head[2] - self.radius[0], self.tail[2] - self.radius[1])

    def closest(self, point: Vec) -> tuple[float, float]:
        axis = _sub(self.tail, self.head)
        span = _dot(axis, axis)
        t = 0.0 if span <= 0.0 else max(0.0, min(1.0, _dot(_sub(point, self.head), axis) / span))
        on_axis = _add(self.head, _scale(axis, t))
        return t, _distance(point, on_axis)

    def surface_distance(self, point: Vec) -> float:
        t, distance = self.closest(point)
        return distance - (self.radius[0] + (self.radius[1] - self.radius[0]) * t)


@dataclass(frozen=True)
class Species:
    name: str
    seed: int
    stance: str
    bones: tuple[Bone, ...]
    palette: Mapping[str, Vec]
    target_tris: int = 3000
    voxel_m: float = 0.015
    smooth: int = 6
    noise_m: float = 0.006
    noise_scale_m: float = 0.08
    sink_m: float = 0.01
    description: str = ""

    def bone(self, name: str) -> Bone:
        for bone in self.bones:
            if bone.name == name:
                return bone
        raise KeyError(name)

    def regions(self) -> list[str]:
        used = {bone.region for bone in self.bones}
        if "underside" in self.palette:
            used.add("underside")
        return [region for region in REGIONS if region in used]

    def region_at(self, point: Vec, normal_z: float) -> str:
        nearest = min(self.bones, key=lambda bone: bone.surface_distance(point))
        if nearest.region == "hide" and normal_z < -0.45 and "underside" in self.palette:
            return "underside"
        return nearest.region

    def bounds(self) -> tuple[Vec, Vec]:
        low = [math.inf] * 3
        high = [-math.inf] * 3
        for bone in self.bones:
            for point, radius in ((bone.head, bone.radius[0]), (bone.tail, bone.radius[1])):
                for axis in range(3):
                    low[axis] = min(low[axis], point[axis] - radius)
                    high[axis] = max(high[axis], point[axis] + radius)
        return (low[0], low[1], low[2]), (high[0], high[1], high[2])

    def feet(self) -> list[str]:
        return [bone.name for bone in self.bones if bone.lowest() <= GROUND_CONTACT_M]


SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "seed", "stance", "bones", "palette"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "description": {"type": "string"},
        "seed": {"type": "integer"},
        "stance": {"enum": sorted(STANCES)},
        "target_tris": {"type": "integer", "minimum": TRI_RANGE[0], "maximum": TRI_RANGE[1]},
        "voxel_m": {"type": "number", "minimum": VOXEL_RANGE_M[0], "maximum": VOXEL_RANGE_M[1]},
        "smooth": {"type": "integer", "minimum": 0, "maximum": 30},
        "noise_m": {"type": "number", "minimum": 0.0, "maximum": 0.05},
        "noise_scale_m": {"type": "number", "exclusiveMinimum": 0.0},
        "sink_m": {"type": "number", "minimum": 0.0, "maximum": 0.05},
        "palette": {
            "type": "object",
            "required": ["hide"],
            "properties": {
                region: {"type": "array", "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "minItems": 3, "maxItems": 3}
                for region in REGIONS
            },
            "additionalProperties": False,
        },
        "bones": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "required": ["name", "head", "tail", "radius"],
                "properties": {
                    "name": {"type": "string"},
                    "parent": {"type": "string"},
                    "head": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "tail": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "radius": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0.0}, "minItems": 2, "maxItems": 2},
                    "region": {"enum": list(REGIONS)},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

DRAFT_SYSTEM = (
    "You draft creature skeleton specs for a procedural modeller. Reply with JSON only. "
    "Units are metres. Axes: +Z up, +Y is the direction the creature faces, origin on the ground "
    "between the feet. Each bone is a tapered capsule from head to tail with radius [at head, at tail]. "
    "Every bone except one root names a parent and starts inside the parent's capsule. "
    "Name left-side bones '<part>.L' with x > 0; they are mirrored to '.R' automatically, so never "
    "list '.R' bones. Feet must reach the ground (lowest capsule point at z = 0). "
    f"Regions colour the surface: {', '.join(REGIONS)}."
)


def draft_prompt(description: str, stance: str) -> str:
    return (
        f"Draft a {stance} creature: {description}. "
        "Use 10-30 bones (spine, neck, head, jaw, tail, legs in 2-3 segments, ears, horns as needed). "
        "Give a palette of linear RGB colours for hide plus any other regions used."
    )


def _distance(a: Vec, b: Vec) -> float:
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in range(3)))


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec, factor: float) -> Vec:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _flip(a: Vec) -> Vec:
    return (-a[0], a[1], a[2])


def _vector(value: Any, length: int, label: str, problems: list[str]) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        problems.append(f"{label} must be a list of {length} numbers")
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in value):
        problems.append(f"{label} must hold finite numbers")
        return None
    return tuple(float(item) for item in value)


def _parse_bones(raw: Any, problems: list[str]) -> list[Bone]:
    if not isinstance(raw, list) or len(raw) < 3:
        problems.append("bones must be a list of at least 3 bones")
        return []
    bones: list[Bone] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            problems.append(f"bones[{index}] must be a table")
            continue
        name = entry.get("name")
        label = f"bone {name!r}" if isinstance(name, str) else f"bones[{index}]"
        unknown = set(entry) - {"name", "parent", "head", "tail", "radius", "region"}
        if unknown:
            problems.append(f"{label} has unknown keys {sorted(unknown)}")
        if not isinstance(name, str) or not name:
            problems.append(f"{label} needs a name")
            continue
        if name.endswith(MIRRORED_SUFFIX):
            problems.append(f"{label}: list only {MIRROR_SUFFIX} bones, {MIRRORED_SUFFIX} is mirrored from them")
        parent = entry.get("parent")
        if parent is not None and not isinstance(parent, str):
            problems.append(f"{label} parent must be a bone name")
            parent = None
        head = _vector(entry.get("head"), 3, f"{label} head", problems)
        tail = _vector(entry.get("tail"), 3, f"{label} tail", problems)
        radius = _vector(entry.get("radius"), 2, f"{label} radius", problems)
        region = entry.get("region", "hide")
        if region not in REGIONS:
            problems.append(f"{label} region {region!r} is not one of {list(REGIONS)}")
        if head is None or tail is None or radius is None:
            continue
        if min(radius) < MIN_RADIUS_M:
            problems.append(f"{label} radius below {MIN_RADIUS_M} m")
        bone = Bone(name, parent or None, head, tail, (radius[0], radius[1]), region if region in REGIONS else "hide")
        if bone.length() <= 0.0:
            problems.append(f"{label} has zero length")
        if name.endswith(MIRROR_SUFFIX) and min(head[0], tail[0]) <= 0.0:
            problems.append(f"{label} is a left-side bone and must lie at x > 0")
        bones.append(bone)
    return bones


def _expand(bones: list[Bone]) -> list[Bone]:
    expanded = []
    for bone in bones:
        expanded.append(bone)
        if bone.name.endswith(MIRROR_SUFFIX):
            expanded.append(bone.mirrored())
    return expanded


def _check_skeleton(bones: list[Bone], problems: list[str]) -> None:
    names = [bone.name for bone in bones]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate bones {duplicates}")
    roots = [bone.name for bone in bones if bone.parent is None]
    if len(roots) != 1:
        problems.append(f"need exactly one root bone, found {roots}")
    by_name = {bone.name: bone for bone in bones}
    seen: set[str] = set()
    for bone in bones:
        if bone.parent is not None:
            parent = by_name.get(bone.parent)
            if parent is None or bone.parent not in seen:
                problems.append(f"bone {bone.name!r} parent {bone.parent!r} must be listed before it")
            elif parent.surface_distance(bone.head) > bone.radius[0]:
                problems.append(f"bone {bone.name!r} starts outside its parent {bone.parent!r}; the body would come apart")
        seen.add(bone.name)
    for bone in bones:
        for point in (bone.head, bone.tail):
            if max(abs(value) for value in point) > MAX_EXTENT_M:
                problems.append(f"bone {bone.name!r} reaches beyond {MAX_EXTENT_M} m")
                break


def _check_stance(stance: str, bones: list[Bone], problems: list[str]) -> None:
    lowest = min(bone.lowest() for bone in bones)
    if abs(lowest) > GROUND_TOLERANCE_M:
        problems.append(f"lowest capsule point is at z = {lowest:.3f} m; feet must reach the ground at z = 0")
    feet = [bone for bone in bones if bone.lowest() <= lowest + GROUND_CONTACT_M]
    needed = STANCES[stance]
    if len(feet) < needed:
        problems.append(f"a {stance} needs {needed} bones on the ground, found {[bone.name for bone in feet]}")


def _palette(raw: Any, problems: list[str]) -> dict[str, Vec]:
    if not isinstance(raw, Mapping):
        problems.append("palette must be a table of region colours")
        return {}
    palette: dict[str, Vec] = {}
    for region, value in raw.items():
        if region not in REGIONS:
            problems.append(f"palette region {region!r} is not one of {list(REGIONS)}")
            continue
        colour = _vector(value, 3, f"palette {region}", problems)
        if colour is None:
            continue
        if not all(0.0 <= channel <= 1.0 for channel in colour):
            problems.append(f"palette {region} channels must lie in [0, 1]")
        palette[region] = (colour[0], colour[1], colour[2])
    if "hide" not in palette:
        problems.append("palette needs a hide colour")
    return palette


def _number(data: Mapping[str, Any], key: str, default: float, low: float, high: float, problems: list[str]) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
        problems.append(f"{key} must be a number in [{low}, {high}]")
        return default
    return float(value)


def _parse(data: Mapping[str, Any]) -> tuple[Species | None, list[str]]:
    problems: list[str] = []
    if not isinstance(data, Mapping):
        return None, ["spec must be a table"]
    unknown = set(data) - set(SPEC_SCHEMA["properties"])
    if unknown:
        problems.append(f"unknown keys {sorted(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not name or not name[0].isalpha() or not all(ch.islower() or ch.isdigit() or ch == "_" for ch in name):
        problems.append("name must be lowercase letters, digits and underscores")
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("seed must be an integer")
    stance = data.get("stance")
    if stance not in STANCES:
        problems.append(f"stance must be one of {sorted(STANCES)}")
    target_tris = data.get("target_tris", 3000)
    if not isinstance(target_tris, int) or isinstance(target_tris, bool) or not TRI_RANGE[0] <= target_tris <= TRI_RANGE[1]:
        problems.append(f"target_tris must be an integer in {list(TRI_RANGE)}")
    smooth = data.get("smooth", 6)
    if not isinstance(smooth, int) or isinstance(smooth, bool) or not 0 <= smooth <= 30:
        problems.append("smooth must be an integer in [0, 30]")
    voxel_m = _number(data, "voxel_m", 0.015, *VOXEL_RANGE_M, problems)
    noise_m = _number(data, "noise_m", 0.006, 0.0, 0.05, problems)
    noise_scale_m = _number(data, "noise_scale_m", 0.08, 1e-3, 10.0, problems)
    sink_m = _number(data, "sink_m", 0.01, 0.0, 0.05, problems)
    description = data.get("description", "")
    if not isinstance(description, str):
        problems.append("description must be a string")
    palette = _palette(data.get("palette"), problems)
    bones = _expand(_parse_bones(data.get("bones"), problems))
    if bones:
        _check_skeleton(bones, problems)
        if stance in STANCES:
            _check_stance(stance, bones, problems)
        for bone in bones:
            if bone.region != "hide" and bone.region not in palette:
                problems.append(f"bone {bone.name!r} region {bone.region!r} has no palette colour")
                break
        thinnest = min(min(bone.radius) for bone in bones)
        if thinnest < voxel_m:
            problems.append(f"thinnest radius {thinnest} m is below voxel_m {voxel_m} m; the remesh would drop it")
    if problems:
        return None, problems
    return (
        Species(
            name=name,
            seed=seed,
            stance=stance,
            bones=tuple(bones),
            palette=palette,
            target_tris=target_tris,
            voxel_m=voxel_m,
            smooth=smooth,
            noise_m=noise_m,
            noise_scale_m=noise_scale_m,
            sink_m=sink_m,
            description=description,
        ),
        [],
    )


def errors(data: Mapping[str, Any]) -> list[str]:
    return _parse(data)[1]


def parse(data: Mapping[str, Any]) -> Species:
    species, problems = _parse(data)
    if species is None:
        raise SpecError(problems)
    return species


def load(path: Path | str) -> Species:
    path = Path(path)
    if path.suffix == ".json":
        return parse(json.loads(path.read_text(encoding="utf-8")))
    with path.open("rb") as handle:
        return parse(tomllib.load(handle))


def catalogue(directory: Path = SPECS_DIR) -> dict[str, Species]:
    return {path.stem: load(path) for path in sorted(directory.glob("*.toml"))}
