"""asset.json `fx[]` schema. Stdlib only, so Blender's interpreter can import it.

Each entry names one particle or light emitter an engine attaches at a socket: an empty node in
the glb named `FX_<type>` or `FX_<type>_<label>` whose transform is the emitter's (local +Y, the
glTF up axis, is the emit direction). `type` is one of `TYPES`; `state` is `idle` (always on) or
`working` (only while the building is occupied), as for animations; `rate` is particles per
second, or flickers per second for `flicker_light`; `size` is the particle size or light radius in
metres; `colour` is a linear RGB tint in [0, 1]. An asset without the key has no effects.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from assets import animation_manifest

MANIFEST_KEY = "fx"
TYPES = ("fire", "smoke", "sparks", "steam", "flicker_light")
STATES = animation_manifest.STATES
FIELDS = ("node", "type", "state", "rate", "size", "colour")
NODE_PREFIX = "FX_"
LABEL = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}"
NODE_NAME = re.compile(rf"^{NODE_PREFIX}({'|'.join(TYPES)})(?:_{LABEL})?$")
MAX_RATE = 1000.0
MAX_SIZE_M = 20.0


class FxManifestError(ValueError):
    pass


def fx_node_name(fx_type: str, label: str = "") -> str:
    if fx_type not in TYPES:
        raise FxManifestError(f"fx type {fx_type!r} not in {TYPES}")
    name = f"{NODE_PREFIX}{fx_type}_{label}" if label else f"{NODE_PREFIX}{fx_type}"
    if not node_matches_type(name, fx_type):
        raise FxManifestError(f"label {label!r} does not make a valid fx node name")
    return name


def node_matches_type(node: str, fx_type: str) -> bool:
    return re.fullmatch(rf"{NODE_PREFIX}{fx_type}(?:_{LABEL})?", node) is not None


def is_fx_node(name: str) -> bool:
    return name.startswith(NODE_PREFIX)


def fx_entry(
    node: str,
    fx_type: str,
    state: str,
    rate: float,
    size: float,
    colour: Sequence[float],
) -> dict:
    entry = {
        "node": node,
        "type": fx_type,
        "state": state,
        "rate": float(rate),
        "size": float(size),
        "colour": [float(channel) for channel in colour],
    }
    errors = entry_errors(entry)
    if errors:
        raise FxManifestError("; ".join(errors))
    return entry


def _positive_number(value: object, ceiling: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0 < value <= ceiling


def entry_errors(entry: object) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"fx entry must be an object, got {type(entry).__name__}"]
    label = entry.get("node", "<unnamed>")
    missing = [key for key in FIELDS if key not in entry]
    if missing:
        return [f"{label}: missing {', '.join(missing)}"]
    unknown = sorted(set(entry) - set(FIELDS))
    errors = [f"{label}: unknown field(s) {', '.join(unknown)}"] if unknown else []
    node = entry["node"]
    fx_type = entry["type"]
    if fx_type not in TYPES:
        errors.append(f"{label}: type {fx_type!r} not in {TYPES}")
    if not isinstance(node, str) or not NODE_NAME.match(node):
        errors.append(f"{label}: node must be named {NODE_PREFIX}<type>[_<label>] ({NODE_NAME.pattern})")
    elif fx_type in TYPES and not node_matches_type(node, fx_type):
        errors.append(f"{label}: node name does not carry its type {fx_type!r}")
    if entry["state"] not in STATES:
        errors.append(f"{label}: state {entry['state']!r} not in {STATES}")
    if not _positive_number(entry["rate"], MAX_RATE):
        errors.append(f"{label}: rate must be a number in (0, {MAX_RATE}]")
    if not _positive_number(entry["size"], MAX_SIZE_M):
        errors.append(f"{label}: size must be a number of metres in (0, {MAX_SIZE_M}]")
    colour = entry["colour"]
    valid_colour = (
        isinstance(colour, list)
        and len(colour) == 3
        and all(not isinstance(c, bool) and isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in colour)
    )
    if not valid_colour:
        errors.append(f"{label}: colour must be three linear RGB numbers in [0, 1]")
    return errors


def manifest_errors(fx: object) -> list[str]:
    if not isinstance(fx, list):
        return [f"{MANIFEST_KEY} must be a list"]
    errors = []
    for entry in fx:
        errors.extend(entry_errors(entry))
    nodes = [entry.get("node") for entry in fx if isinstance(entry, Mapping)]
    repeated = sorted({node for node in nodes if isinstance(node, str) and nodes.count(node) > 1})
    if repeated:
        errors.append(f"fx node(s) repeat: {', '.join(repeated)}")
    return errors


def load(manifest: Mapping) -> list[dict]:
    fx = manifest.get(MANIFEST_KEY, [])
    errors = manifest_errors(fx)
    if errors:
        raise FxManifestError("; ".join(errors))
    return [dict(entry) for entry in fx]
