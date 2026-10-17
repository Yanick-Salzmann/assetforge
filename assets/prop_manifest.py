"""asset.json `props[]` schema. Stdlib only, so Blender's interpreter can import it.

A hero building ships its architecture in the glb and its small dressing props (pots, jars, logs,
sherds) as a placement list the engine instances from the kit library, like scatter. Each entry
names one kit prop as `<pack>/<model>` from library/kits.lock.json and its transform in the glb's
own frame (glTF Y-up, metres from the asset origin at base centre, the frame the building's root
node is in): `translation` [x, y, z], `rotation` a unit quaternion [x, y, z, w] and a uniform
`scale`, applied to the kit glb as modelled. An asset without the key has no props.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

MANIFEST_KEY = "props"
FIELDS = ("mesh", "translation", "rotation", "scale")
MESH_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_SCALE = 100.0
MAX_PROPS = 256
ROTATION_TOLERANCE = 1e-3
REACH_M = 4.0
SINK_M = 0.5
BLENDER_TO_GLTF = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


class PropManifestError(ValueError):
    pass


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _axis_quaternion(axis: int, angle: float) -> tuple[float, float, float, float]:
    vector = [0.0, 0.0, 0.0]
    vector[axis] = math.sin(angle / 2.0)
    return (vector[0], vector[1], vector[2], math.cos(angle / 2.0))


def euler_quaternion(rotation: Sequence[float]) -> tuple[float, float, float, float]:
    """Blender's XYZ Euler (radians) as an [x, y, z, w] quaternion: X applied first, Z last."""
    x, y, z = (float(angle) for angle in rotation)
    return _multiply(_axis_quaternion(2, z), _multiply(_axis_quaternion(1, y), _axis_quaternion(0, x)))


def gltf_translation(location: Sequence[float]) -> list[float]:
    """A Blender Z-up point in the glTF exporter's Y-up frame."""
    x, y, z = (float(value) for value in location)
    return [x, z, -y]


def gltf_rotation(quaternion: Sequence[float]) -> list[float]:
    """A Blender-frame rotation quaternion [x, y, z, w] in the glTF exporter's Y-up frame."""
    inverse = (-BLENDER_TO_GLTF[0], -BLENDER_TO_GLTF[1], -BLENDER_TO_GLTF[2], BLENDER_TO_GLTF[3])
    turned = _multiply(BLENDER_TO_GLTF, _multiply(tuple(float(value) for value in quaternion), inverse))
    if turned[3] < 0.0:
        turned = tuple(-value for value in turned)
    return [value + 0.0 for value in turned]


def prop_entry(
    mesh: str,
    location: Sequence[float],
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    shift: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict:
    """The props[] entry for a kit prop placed in Blender design coordinates: location (Z-up
    metres), rotation (XYZ Euler radians) and uniform scale, less the base-centre shift export
    applies to the asset."""
    entry = {
        "mesh": mesh,
        "translation": gltf_translation([value - offset for value, offset in zip(location, shift)]),
        "rotation": gltf_rotation(euler_quaternion(rotation)),
        "scale": float(scale),
    }
    errors = entry_errors(entry)
    if errors:
        raise PropManifestError("; ".join(errors))
    return entry


def _number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _vector(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_number(component) for component in value)


def entry_errors(entry: object) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"prop entry must be an object, got {type(entry).__name__}"]
    label = entry.get("mesh", "<unnamed>")
    missing = [key for key in FIELDS if key not in entry]
    if missing:
        return [f"{label}: missing {', '.join(missing)}"]
    unknown = sorted(set(entry) - set(FIELDS))
    errors = [f"{label}: unknown field(s) {', '.join(unknown)}"] if unknown else []
    if not isinstance(entry["mesh"], str) or not MESH_NAME.match(entry["mesh"]):
        errors.append(f"{label}: mesh must be a kit prop named <pack>/<model>")
    if not _vector(entry["translation"], 3):
        errors.append(f"{label}: translation must be three finite numbers of metres")
    rotation = entry["rotation"]
    if not _vector(rotation, 4):
        errors.append(f"{label}: rotation must be a quaternion of four finite numbers [x, y, z, w]")
    elif abs(math.sqrt(sum(component * component for component in rotation)) - 1.0) > ROTATION_TOLERANCE:
        errors.append(f"{label}: rotation must be a unit quaternion")
    scale = entry["scale"]
    if not _number(scale) or not 0.0 < scale <= MAX_SCALE:
        errors.append(f"{label}: scale must be a number in (0, {MAX_SCALE}]")
    return errors


def manifest_errors(props: object, known_meshes: object = None) -> list[str]:
    """Schema errors in a props[] list; with known_meshes (a container of `<pack>/<model>` names),
    also every mesh the kit library does not carry."""
    if not isinstance(props, list):
        return [f"{MANIFEST_KEY} must be a list"]
    if len(props) > MAX_PROPS:
        return [f"{MANIFEST_KEY} carries {len(props)} entries; at most {MAX_PROPS}"]
    errors = []
    for entry in props:
        errors.extend(entry_errors(entry))
    if known_meshes is not None:
        unknown = sorted(
            {
                entry["mesh"]
                for entry in props
                if isinstance(entry, Mapping) and isinstance(entry.get("mesh"), str) and entry["mesh"] not in known_meshes
            }
        )
        if unknown:
            errors.append(f"unknown kit prop(s) {', '.join(unknown)}")
    return errors


def outside_errors(props: Sequence[Mapping], bbox_min: Sequence[float], bbox_max: Sequence[float]) -> list[str]:
    """Props standing more than REACH_M outside the asset bbox across the ground, or more than
    SINK_M below or anywhere above it."""
    errors = []
    for entry in props:
        x, y, z = entry["translation"]
        across = (
            x < bbox_min[0] - REACH_M
            or x > bbox_max[0] + REACH_M
            or z < bbox_min[2] - REACH_M
            or z > bbox_max[2] + REACH_M
        )
        if across or y < bbox_min[1] - SINK_M or y > bbox_max[1]:
            errors.append(f"{entry['mesh']}: placed at {[round(value, 3) for value in entry['translation']]} outside the asset bbox")
    return errors


def load(manifest: Mapping) -> list[dict]:
    props = manifest.get(MANIFEST_KEY, [])
    errors = manifest_errors(props)
    if errors:
        raise PropManifestError("; ".join(errors))
    return [dict(entry) for entry in props]
