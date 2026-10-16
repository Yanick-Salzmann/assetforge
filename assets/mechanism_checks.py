"""Mechanism checks for validate_asset: does each clip honour the couplings asset.json
`mechanisms[]` declares for it (assets.mechanism_manifest)?

Every clip is sampled once per frame. Gears: each gear's world rotation relative to its first
sample must stay a turn about one axis shared by the whole chain, and every meshing neighbour's
accumulated angle must equal minus the driver's times the tooth ratio to within
`GEAR_SLIP_TEETH` of a tooth. Linkage: a joint's node origin, carried in the frame of the node it
is pinned on, must not drift more than `LINKAGE_TOLERANCE_M` from where it sat at the first sample.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from assets import animation_checks, animation_manifest

Finding = tuple[str, bool, str]

GEAR_SLIP_TEETH = 0.05
GEAR_AXIS_TOLERANCE_DEG = 2.0
GEAR_MIN_TURN_RAD = 1e-3
AXIS_SIN_FLOOR = 0.05
LINKAGE_TOLERANCE_M = 0.002


def sample_times(entry: Mapping) -> np.ndarray:
    frames = animation_manifest.frame_count(entry)
    return np.arange(frames + 1, dtype=np.float64) / entry["fps"]


def clip_worlds(document: dict, channels: Sequence[animation_checks.Channel], times: np.ndarray) -> list[dict[int, np.ndarray]]:
    return [animation_checks.world_matrices(document, animation_checks.clip_pose(channels, float(time_s))) for time_s in times]


def _rotation(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    return linear / np.maximum(np.linalg.norm(linear, axis=0), 1e-12)[None, :]


def _vee(delta: np.ndarray) -> np.ndarray:
    return np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])


def _deltas(worlds: Sequence[Mapping[int, np.ndarray]], node: int) -> list[np.ndarray]:
    start = _rotation(worlds[0][node])
    return [_rotation(world[node]) @ start.T for world in worlds]


def _chain_axis(deltas: Sequence[np.ndarray]) -> np.ndarray | None:
    vees = [_vee(delta) for delta in deltas]
    strongest = max(vees, key=lambda vee: float(np.linalg.norm(vee)))
    length = float(np.linalg.norm(strongest))
    if length / 2.0 < AXIS_SIN_FLOOR:
        return None
    return strongest / length


def _signed_angles(deltas: Sequence[np.ndarray], axis: np.ndarray) -> np.ndarray:
    angles = [math.atan2(float(np.dot(axis, _vee(delta))) / 2.0, (float(np.trace(delta)) - 1.0) / 2.0) for delta in deltas]
    return np.unwrap(np.array(angles))


def _off_axis_deg(deltas: Sequence[np.ndarray], axis: np.ndarray) -> float:
    worst = 0.0
    for delta in deltas:
        vee = _vee(delta)
        length = float(np.linalg.norm(vee))
        if length / 2.0 < AXIS_SIN_FLOOR:
            continue
        cosine = min(1.0, abs(float(np.dot(axis, vee))) / length)
        worst = max(worst, math.degrees(math.acos(cosine)))
    return worst


def gear_errors(entry: Mapping, worlds: Sequence[Mapping[int, np.ndarray]], index_by_name: Mapping[str, int]) -> list[str]:
    label = f"gears on {entry['clip']}"
    gears = entry["gears"]
    absent = [gear["node"] for gear in gears if gear["node"] not in index_by_name]
    if absent:
        return [f"{label}: no node {', '.join(absent)} in the glb"]
    deltas = [_deltas(worlds, index_by_name[gear["node"]]) for gear in gears]
    axis = _chain_axis(deltas[0])
    if axis is None:
        return [f"{label}: driver {gears[0]['node']} does not turn"]
    errors = []
    for gear, gear_deltas in zip(gears, deltas):
        off_axis = _off_axis_deg(gear_deltas, axis)
        if off_axis > GEAR_AXIS_TOLERANCE_DEG:
            errors.append(f"{gear['node']} turns {off_axis:.1f} deg off the chain axis")
    angles = [_signed_angles(gear_deltas, axis) for gear_deltas in deltas]
    for (driver, driven), (driver_angle, driven_angle) in zip(zip(gears, gears[1:]), zip(angles, angles[1:])):
        expected = -driver_angle * driver["teeth"] / driven["teeth"]
        slip = float(np.max(np.abs(driven_angle - expected))) * driven["teeth"] / (2.0 * math.pi)
        if slip > GEAR_SLIP_TEETH:
            turned = "the same way as" if float(np.dot(driven_angle[1:], driver_angle[1:])) > 0.0 else "against"
            errors.append(
                f"{driven['node']} slips {slip:.2f} teeth against {driver['node']} "
                f"(ratio {driver['teeth']}:{driven['teeth']}, turns {turned} it)"
            )
    return [f"{label}: {error}" for error in errors]


def linkage_errors(entry: Mapping, worlds: Sequence[Mapping[int, np.ndarray]], index_by_name: Mapping[str, int]) -> list[str]:
    label = f"linkage on {entry['clip']}"
    errors = []
    for joint in entry["joints"]:
        absent = [name for name in (joint["node"], joint["on"]) if name not in index_by_name]
        if absent:
            errors.append(f"no node {', '.join(absent)} in the glb")
            continue
        node = index_by_name[joint["node"]]
        on = index_by_name[joint["on"]]
        pinned = np.linalg.inv(worlds[0][on]) @ worlds[0][node][:, 3]
        drift = max(float(np.linalg.norm(world[on] @ pinned - world[node][:, 3])) for world in worlds)
        if drift > LINKAGE_TOLERANCE_M:
            errors.append(f"{joint['node']} comes {drift * 1000:.1f} mm off its pin on {joint['on']}")
    return [f"{label}: {error}" for error in errors]


def mechanism_checks(
    document: dict,
    entries: Sequence[Mapping],
    mechanisms: Sequence[Mapping],
    clips: Mapping[str, Sequence[animation_checks.Channel]],
) -> list[Finding]:
    if not mechanisms:
        return []
    index_by_name = {name: index for index, name in enumerate(animation_checks.node_names(document))}
    by_clip = {entry["name"]: entry for entry in entries}
    errors = []
    sampled: dict[str, list[dict[int, np.ndarray]]] = {}
    for mechanism in mechanisms:
        name = mechanism["clip"]
        if name not in clips or name not in by_clip:
            errors.append(f"{mechanism['type']} on {name}: no such clip in the glb")
            continue
        if name not in sampled:
            sampled[name] = clip_worlds(document, clips[name], sample_times(by_clip[name]))
        check = gear_errors if mechanism["type"] == "gears" else linkage_errors
        errors.extend(check(mechanism, sampled[name], index_by_name))
    if errors:
        return [("mechanisms", False, "; ".join(errors))]
    summary = ", ".join(f"{mechanism['type']} on {mechanism['clip']}" for mechanism in mechanisms)
    return [("mechanisms", True, f"{len(mechanisms)} coupling(s) hold: {summary}")]
