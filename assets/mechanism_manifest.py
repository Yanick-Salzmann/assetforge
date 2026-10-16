"""asset.json `mechanisms[]` schema. Stdlib only, so Blender's interpreter can import it.

Each entry declares how the moving parts of one animations[] clip are coupled, so validate_asset
can check the clip honours it:

- `{"type": "gears", "clip": <clip>, "gears": [{"node": <part>_ANIM, "teeth": n}, ...]}` - a chain
  in which each gear meshes with the next: neighbours turn about parallel axes, in opposite
  directions, at the inverse ratio of their tooth counts.
- `{"type": "linkage", "clip": <clip>, "joints": [{"node": <a>, "on": <b>}, ...]}` - the origin of
  node `a` (its pivot) is pinned to node `b`: it stays at one point in `b`'s frame over the whole
  clip, e.g. a connecting rod's crank-pin end on the crank and a piston's wrist pin on the rod.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from assets import animation_manifest

MANIFEST_KEY = "mechanisms"
TYPES = ("gears", "linkage")
FIELDS = {"gears": ("type", "clip", "gears"), "linkage": ("type", "clip", "joints")}
GEAR_FIELDS = ("node", "teeth")
JOINT_FIELDS = ("node", "on")


class MechanismManifestError(ValueError):
    pass


def gears_entry(clip: str, nodes: Sequence[str], teeth: Sequence[int]) -> dict:
    if len(nodes) != len(teeth):
        raise MechanismManifestError("one tooth count per gear")
    entry = {"type": "gears", "clip": clip, "gears": [{"node": node, "teeth": count} for node, count in zip(nodes, teeth)]}
    errors = entry_errors(entry)
    if errors:
        raise MechanismManifestError("; ".join(errors))
    return entry


def linkage_entry(clip: str, joints: Sequence[tuple[str, str]]) -> dict:
    entry = {"type": "linkage", "clip": clip, "joints": [{"node": node, "on": on} for node, on in joints]}
    errors = entry_errors(entry)
    if errors:
        raise MechanismManifestError("; ".join(errors))
    return entry


def _named(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _gear_errors(label: str, gears: object) -> list[str]:
    if not isinstance(gears, list) or len(gears) < 2:
        return [f"{label}: gears must list at least two meshing gears"]
    errors = []
    for gear in gears:
        if not isinstance(gear, Mapping) or sorted(gear) != sorted(GEAR_FIELDS):
            errors.append(f"{label}: each gear is an object with exactly {', '.join(GEAR_FIELDS)}")
            continue
        if not _named(gear["node"]):
            errors.append(f"{label}: gear node must be a non-empty string")
        teeth = gear["teeth"]
        if isinstance(teeth, bool) or not isinstance(teeth, int) or teeth < 3:
            errors.append(f"{label}: {gear['node']} teeth must be an integer of at least 3")
    nodes = [gear.get("node") for gear in gears if isinstance(gear, Mapping)]
    if len(set(map(str, nodes))) != len(nodes):
        errors.append(f"{label}: gear nodes repeat")
    return errors


def _joint_errors(label: str, joints: object) -> list[str]:
    if not isinstance(joints, list) or not joints:
        return [f"{label}: joints must be a non-empty list"]
    errors = []
    for joint in joints:
        if not isinstance(joint, Mapping) or sorted(joint) != sorted(JOINT_FIELDS):
            errors.append(f"{label}: each joint is an object with exactly {', '.join(JOINT_FIELDS)}")
            continue
        if not (_named(joint["node"]) and _named(joint["on"])):
            errors.append(f"{label}: joint node and on must be non-empty strings")
        elif joint["node"] == joint["on"]:
            errors.append(f"{label}: {joint['node']} cannot be pinned to itself")
    return errors


def entry_errors(entry: object) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"mechanism entry must be an object, got {type(entry).__name__}"]
    kind = entry.get("type")
    if kind not in TYPES:
        return [f"mechanism type {kind!r} not in {TYPES}"]
    label = f"{kind} on {entry.get('clip', '<no clip>')}"
    fields = FIELDS[kind]
    missing = [key for key in fields if key not in entry]
    if missing:
        return [f"{label}: missing {', '.join(missing)}"]
    unknown = sorted(set(entry) - set(fields))
    errors = [f"{label}: unknown field(s) {', '.join(unknown)}"] if unknown else []
    if not isinstance(entry["clip"], str) or not animation_manifest.CLIP_NAME.match(entry["clip"]):
        errors.append(f"{label}: clip must be a <state>_<clip> name")
    if kind == "gears":
        errors.extend(_gear_errors(label, entry["gears"]))
    else:
        errors.extend(_joint_errors(label, entry["joints"]))
    return errors


def mechanism_nodes(entry: Mapping) -> list[str]:
    if entry["type"] == "gears":
        return [gear["node"] for gear in entry["gears"]]
    return [name for joint in entry["joints"] for name in (joint["node"], joint["on"])]


def manifest_errors(mechanisms: object, animations: Sequence[Mapping] = ()) -> list[str]:
    if not isinstance(mechanisms, list):
        return [f"{MANIFEST_KEY} must be a list"]
    clips = {entry.get("name"): entry for entry in animations if isinstance(entry, Mapping)}
    errors = []
    for entry in mechanisms:
        problems = entry_errors(entry)
        errors.extend(problems)
        if problems:
            continue
        clip = clips.get(entry["clip"])
        if clip is None:
            errors.append(f"{entry['type']} on {entry['clip']}: no animations[] entry of that name")
        elif clip.get("kind") != "rigid":
            errors.append(f"{entry['type']} on {entry['clip']}: mechanisms couple rigid clips only")
        elif entry["type"] == "gears":
            loose = [gear["node"] for gear in entry["gears"] if gear["node"] not in clip.get("nodes", [])]
            if loose:
                errors.append(f"gears on {entry['clip']}: {', '.join(loose)} not animated by the clip")
    return errors


def load(manifest: Mapping) -> list[dict]:
    mechanisms = manifest.get(MANIFEST_KEY, [])
    errors = manifest_errors(mechanisms, manifest.get(animation_manifest.MANIFEST_KEY, []))
    if errors:
        raise MechanismManifestError("; ".join(errors))
    return [dict(entry) for entry in mechanisms]
