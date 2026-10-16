"""asset.json `animations[]` schema. Stdlib only, so Blender's interpreter can import it.

Each entry describes one glTF clip. `name` is `<state>_<clip>` and equals the Blender action
(NLA strip) name, which the glTF exporter uses as the clip name. `state` is `idle` (an ambient
loop that always plays) or `working` (plays only while the building is occupied). `kind` is
`rigid` (object TRS animation on `<part>_ANIM` nodes, each with its origin at its pivot) or
`skinned` (bone animation; `nodes` lists the animated joints). An asset without the key is static.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

MANIFEST_KEY = "animations"
STATES = ("idle", "working")
KINDS = ("rigid", "skinned")
LOOPING_STATES = ("idle",)
ANIM_SUFFIX = "_ANIM"
CLIP_NAME = re.compile(r"^(idle|working)_[a-z0-9][a-z0-9_]{0,62}$")
ANIM_NODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}_ANIM$")
LOD_NODE_SUFFIX = re.compile(r"_LOD(\d+)(?:_ANIM)?$")
FIELDS = ("name", "state", "kind", "duration_s", "fps", "loop", "nodes")
FRAME_TOLERANCE = 1e-3
MAX_FPS = 120


class AnimationManifestError(ValueError):
    pass


def anim_node_name(part: str) -> str:
    name = f"{part}{ANIM_SUFFIX}"
    if not ANIM_NODE_NAME.match(name):
        raise AnimationManifestError(f"part {part!r} does not make a valid animated node name")
    return name


def lod_copy_name(node: str, lod: int) -> str:
    """The reduced LOD's copy of rigid node <part>_ANIM: <part>_LOD<lod>_ANIM, keyed by the same clip."""
    return anim_node_name(f"{node.removesuffix(ANIM_SUFFIX)}_LOD{lod}")


def lod_index(node: str) -> int:
    """The LOD a node belongs to: the n of a _LOD<n> or _LOD<n>_ANIM suffix, else 0."""
    match = LOD_NODE_SUFFIX.search(node)
    return int(match.group(1)) if match else 0


def with_lod_copies(entry: Mapping, names: Sequence[str]) -> dict:
    """entry with every reduced-LOD copy of its rigid nodes found in names appended to nodes."""
    if entry["kind"] != "rigid":
        return dict(entry)
    present = set(names)
    copies = []
    for node in entry["nodes"]:
        lod = 1
        while lod_copy_name(node, lod) in present:
            copies.append(lod_copy_name(node, lod))
            lod += 1
    extra = [copy for copy in copies if copy not in entry["nodes"]]
    return {**entry, "nodes": [*entry["nodes"], *extra]}


def clip_name(state: str, clip: str) -> str:
    name = f"{state}_{clip}"
    if state not in STATES or not CLIP_NAME.match(name):
        raise AnimationManifestError(f"{name!r} is not a valid <state>_<clip> name; states are {STATES}")
    return name


def manifest_entry(
    state: str,
    clip: str,
    kind: str,
    duration_s: float,
    fps: int,
    nodes: Sequence[str],
    loop: bool = True,
) -> dict:
    entry = {
        "name": clip_name(state, clip),
        "state": state,
        "kind": kind,
        "duration_s": float(duration_s),
        "fps": fps,
        "loop": loop,
        "nodes": list(nodes),
    }
    errors = entry_errors(entry)
    if errors:
        raise AnimationManifestError("; ".join(errors))
    return entry


def frame_count(entry: Mapping) -> int:
    return round(entry["duration_s"] * entry["fps"])


def entry_errors(entry: object) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"animation entry must be an object, got {type(entry).__name__}"]
    label = entry.get("name", "<unnamed>")
    missing = [key for key in FIELDS if key not in entry]
    if missing:
        return [f"{label}: missing {', '.join(missing)}"]
    unknown = sorted(set(entry) - set(FIELDS))
    errors = [f"{label}: unknown field(s) {', '.join(unknown)}"] if unknown else []
    name = entry["name"]
    state = entry["state"]
    kind = entry["kind"]
    if state not in STATES:
        errors.append(f"{label}: state {state!r} not in {STATES}")
    if kind not in KINDS:
        errors.append(f"{label}: kind {kind!r} not in {KINDS}")
    if not isinstance(name, str) or not CLIP_NAME.match(name):
        errors.append(f"{label}: name must match <state>_<clip> ({CLIP_NAME.pattern})")
    elif state in STATES and not name.startswith(f"{state}_"):
        errors.append(f"{label}: name must start with its state {state!r}")
    errors.extend(_timing_errors(label, entry))
    if not isinstance(entry["loop"], bool):
        errors.append(f"{label}: loop must be true or false")
    elif state in LOOPING_STATES and not entry["loop"]:
        errors.append(f"{label}: {state} clips are ambient and must loop")
    errors.extend(_node_errors(label, kind, entry["nodes"]))
    return errors


def _timing_errors(label: str, entry: Mapping) -> list[str]:
    fps = entry["fps"]
    duration_s = entry["duration_s"]
    if isinstance(fps, bool) or not isinstance(fps, int) or not 0 < fps <= MAX_FPS:
        return [f"{label}: fps must be an integer in 1..{MAX_FPS}"]
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        return [f"{label}: duration_s must be a number"]
    if not math.isfinite(duration_s) or duration_s <= 0:
        return [f"{label}: duration_s must be positive"]
    frames = duration_s * fps
    if abs(frames - round(frames)) > FRAME_TOLERANCE:
        return [f"{label}: duration_s {duration_s} is not a whole number of frames at {fps} fps"]
    return []


def _node_errors(label: str, kind: str, nodes: object) -> list[str]:
    if not isinstance(nodes, list) or not nodes:
        return [f"{label}: nodes must be a non-empty list"]
    if not all(isinstance(node, str) and node for node in nodes):
        return [f"{label}: nodes must be non-empty strings"]
    errors = []
    if len(set(nodes)) != len(nodes):
        errors.append(f"{label}: nodes repeat")
    if kind == "rigid":
        bad = [node for node in nodes if not ANIM_NODE_NAME.match(node)]
        if bad:
            errors.append(f"{label}: rigid nodes must be named <part>{ANIM_SUFFIX}: {', '.join(bad)}")
    return errors


def manifest_errors(animations: object) -> list[str]:
    if not isinstance(animations, list):
        return [f"{MANIFEST_KEY} must be a list"]
    errors = []
    for entry in animations:
        errors.extend(entry_errors(entry))
    names = [entry.get("name") for entry in animations if isinstance(entry, Mapping)]
    repeated = sorted({name for name in names if names.count(name) > 1 and isinstance(name, str)})
    if repeated:
        errors.append(f"clip name(s) repeat: {', '.join(repeated)}")
    return errors


def load(manifest: Mapping) -> list[dict]:
    animations = manifest.get(MANIFEST_KEY, [])
    errors = manifest_errors(animations)
    if errors:
        raise AnimationManifestError("; ".join(errors))
    return [dict(entry) for entry in animations]
