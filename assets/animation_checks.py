"""Animation checks for validate_asset over a parsed glTF document.

Every animations[] entry is matched against the glb's clips; rigid `<part>_ANIM` nodes must rest
at their pivot with identity rotation and scale; looping clips must close their seam in value and
in velocity; clip length, keyframe count and bone count stay within budget; and every rigid clip
is sampled and its moving parts ray-cast against every other LOD0 part, flagging points that sink
deeper than `CONTACT_TOLERANCE_M` into another solid where they did not already sit at rest
(designed contact such as an axle in its bearing is present at rest and so tolerated).

Each check returns `(name, passed, detail)` findings; matrices are 4x4 numpy arrays acting on
column vectors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from library import gltf

Finding = tuple[str, bool, str]

MAX_BONES: dict[str, int] = {
    "hero_building": 32,
    "prop": 16,
    "vehicle": 32,
    "hero_vehicle": 64,
    "vegetation": 16,
    "unit": 64,
    "creature": 96,
    "organic_prop": 16,
}
CLIP_MAX_S = 30.0
CLIP_MAX_KEYFRAMES = 25_000
REST_ROTATION_TOLERANCE_RAD = 1e-3
REST_SCALE_TOLERANCE = 1e-3
SEAM_TOLERANCE = {"translation": 1e-3, "rotation": 1e-3, "scale": 1e-3, "weights": 1e-3}
SEAM_VELOCITY_FACTOR = 3.0
SEAM_VELOCITY_EPS = 1e-2
SWEEP_SAMPLES = 24
CONTACT_TOLERANCE_M = 0.005
SWEEP_GRID_CELLS = 32
SWEEP_REPORT_LIMIT = 5
MOTION_EPS = 1e-6
TIME_EPS = 1e-6
RAY_JITTER = np.array([1.37e-7, 0.0, 2.71e-7])
IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Channel:
    node: int
    path: str
    interpolation: str
    times: np.ndarray
    values: np.ndarray


def read_clips(document: dict, buffers: list[bytes]) -> dict[str, list[Channel]]:
    clips: dict[str, list[Channel]] = {}
    for animation in document.get("animations", []):
        channels = []
        for channel in animation["channels"]:
            target = channel["target"]
            if "node" not in target:
                continue
            sampler = animation["samplers"][channel["sampler"]]
            interpolation = sampler.get("interpolation", "LINEAR")
            times = gltf.read_accessor(document, buffers, sampler["input"])[:, 0].astype(np.float64)
            raw = gltf.read_accessor(document, buffers, sampler["output"]).astype(np.float64)
            if interpolation == "CUBICSPLINE":
                values = raw.reshape(len(times), 3, -1)[:, 1]
            else:
                values = raw.reshape(len(times), -1)
            channels.append(Channel(target["node"], target["path"], interpolation, times, values))
        clips[animation.get("name", "")] = channels
    return clips


def node_names(document: dict) -> list[str]:
    return [node.get("name", "") for node in document.get("nodes", [])]


def joint_names(document: dict) -> set[str]:
    names = node_names(document)
    return {names[joint] for skin in document.get("skins", []) for joint in skin["joints"]}


def trs_matrix(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x, y, z, w = rotation / max(np.linalg.norm(rotation), 1e-12)
    linear = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = linear * np.asarray(scale, dtype=np.float64)[None, :]
    matrix[:3, 3] = translation
    return matrix


def _local_matrix(node: dict, pose: Mapping[str, np.ndarray]) -> np.ndarray:
    if "matrix" in node and not pose:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = pose.get("translation", np.array(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64))
    rotation = pose.get("rotation", np.array(node.get("rotation", IDENTITY_QUATERNION), dtype=np.float64))
    scale = pose.get("scale", np.array(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64))
    return trs_matrix(translation, rotation, scale)


def world_matrices(document: dict, poses: Mapping[int, Mapping[str, np.ndarray]] | None = None) -> dict[int, np.ndarray]:
    poses = poses or {}
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not scenes:
        raise gltf.GltfError("document has no scenes")
    worlds: dict[int, np.ndarray] = {}
    stack = [(root, np.eye(4)) for root in scenes[document.get("scene", 0)]["nodes"]]
    while stack:
        index, parent = stack.pop()
        world = parent @ _local_matrix(nodes[index], poses.get(index, {}))
        worlds[index] = world
        stack.extend((child, world) for child in nodes[index].get("children", ()))
    return worlds


def skin_bind_matrix(document: dict, buffers: list[bytes], skin_index: int) -> np.ndarray:
    skin = document["skins"][skin_index]
    joint = skin["joints"][0]
    inverse_bind = np.eye(4)
    if "inverseBindMatrices" in skin:
        inverse_bind = gltf.read_accessor(document, buffers, skin["inverseBindMatrices"])[0].astype(np.float64).reshape(4, 4).T
    return world_matrices(document)[joint] @ inverse_bind


def _nlerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    if np.dot(a, b) < 0.0:
        b = -b
    blended = a + (b - a) * u
    return blended / max(np.linalg.norm(blended), 1e-12)


def sample_channel(channel: Channel, time_s: float) -> np.ndarray:
    times = channel.times
    values = channel.values
    if time_s <= times[0]:
        return values[0]
    if time_s >= times[-1]:
        return values[-1]
    index = int(np.searchsorted(times, time_s, side="right")) - 1
    if channel.interpolation == "STEP":
        return values[index]
    u = (time_s - times[index]) / (times[index + 1] - times[index])
    if channel.path == "rotation":
        return _nlerp(values[index], values[index + 1], u)
    return values[index] + (values[index + 1] - values[index]) * u


def clip_pose(channels: Sequence[Channel], time_s: float) -> dict[int, dict[str, np.ndarray]]:
    poses: dict[int, dict[str, np.ndarray]] = {}
    for channel in channels:
        if channel.path in ("translation", "rotation", "scale"):
            poses.setdefault(channel.node, {})[channel.path] = sample_channel(channel, time_s)
    return poses


def _clip_end(channels: Sequence[Channel]) -> float:
    return max((float(channel.times[-1]) for channel in channels), default=0.0)


def clip_checks(document: dict, entries: Sequence[Mapping], clips: Mapping[str, Sequence[Channel]]) -> list[Finding]:
    names = node_names(document)
    joints = joint_names(document)
    expected = [entry["name"] for entry in entries]
    errors = []
    missing = sorted(set(expected) - set(clips))
    if missing:
        errors.append(f"no glTF clip for {', '.join(missing)}")
    extra = sorted(set(clips) - set(expected))
    if extra:
        errors.append(f"glTF clip(s) {', '.join(extra)} have no animations[] entry")
    for entry in entries:
        channels = clips.get(entry["name"])
        if channels is None:
            continue
        targets = {names[channel.node] for channel in channels}
        absent = [node for node in entry["nodes"] if node not in targets]
        if absent:
            errors.append(f"{entry['name']}: clip does not animate {', '.join(absent)}")
        if entry["kind"] == "skinned":
            loose = [node for node in entry["nodes"] if node not in joints]
            if loose:
                errors.append(f"{entry['name']}: skinned nodes are not skin joints: {', '.join(loose)}")
        end_s = _clip_end(channels)
        if not math.isclose(end_s, entry["duration_s"], abs_tol=0.5 / entry["fps"]):
            errors.append(f"{entry['name']}: clip runs {end_s:.4f} s, animations[] says {entry['duration_s']} s")
    if errors:
        return [("animation clips", False, "; ".join(errors))]
    return [("animation clips", True, f"{len(expected)} clip(s) match animations[]")]


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def rest_checks(document: dict, entries: Sequence[Mapping]) -> list[Finding]:
    nodes = document.get("nodes", [])
    index_by_name = {node.get("name", ""): index for index, node in enumerate(nodes)}
    rigid = sorted({node for entry in entries if entry["kind"] == "rigid" for node in entry["nodes"]})
    findings = []
    for name in rigid:
        label = f"{name}: rest transform"
        index = index_by_name.get(name)
        if index is None:
            findings.append((label, False, "no node of that name in the glb"))
            continue
        node = nodes[index]
        if "matrix" in node:
            findings.append((label, False, "rest transform is a matrix, which glTF cannot animate"))
            continue
        issues = []
        angle = quaternion_angle(np.array(node.get("rotation", IDENTITY_QUATERNION), dtype=np.float64), IDENTITY_QUATERNION)
        if angle > REST_ROTATION_TOLERANCE_RAD:
            issues.append(f"rest rotation is {angle:.4f} rad from identity")
        scale = np.array(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        if np.max(np.abs(scale - 1.0)) > REST_SCALE_TOLERANCE:
            issues.append(f"rest scale {scale.tolist()} is not (1, 1, 1)")
        if issues:
            findings.append((label, False, "; ".join(issues)))
            continue
        pivot = [round(float(v), 4) for v in node.get("translation", (0.0, 0.0, 0.0))]
        findings.append((label, True, f"pivot at {pivot}, identity rotation and scale"))
    return findings


def _rotation_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = -a[:, 0], -a[:, 1], -a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    relative = np.stack(
        [
            bw * ax + bx * aw + by * az - bz * ay,
            bw * ay - bx * az + by * aw + bz * ax,
            bw * az + bx * ay - by * ax + bz * aw,
            bw * aw - bx * ax - by * ay - bz * az,
        ],
        axis=1,
    )
    relative[relative[:, 3] < 0.0] *= -1.0
    vector = relative[:, :3]
    sine = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(sine, relative[:, 3])
    factor = np.where(sine > 1e-12, angle / np.maximum(sine, 1e-12), 2.0)
    return vector * factor[:, None]


def _seam_keys(channel: Channel, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    times = channel.times
    values = channel.values
    if times[0] > TIME_EPS:
        times = np.concatenate([[0.0], times])
        values = np.vstack([values[:1], values])
    if times[-1] < end_s - TIME_EPS:
        times = np.concatenate([times, [end_s]])
        values = np.vstack([values, values[-1:]])
    return times, values


def _step_velocities(path: str, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    durations = np.diff(times)
    keep = durations > TIME_EPS
    if path == "rotation":
        steps = _rotation_vectors(values[:-1], values[1:])
    else:
        steps = np.diff(values, axis=0)
    return steps[keep] / durations[keep, None]


def velocity_jumps(path: str, times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    velocities = _step_velocities(path, times, values)
    if len(velocities) < 2:
        return 0.0, 0.0
    seam = float(np.linalg.norm(velocities[0] - velocities[-1]))
    interior = float(np.max(np.linalg.norm(np.diff(velocities, axis=0), axis=1)))
    return seam, interior


def _value_gap(path: str, first: np.ndarray, last: np.ndarray) -> float:
    if path == "rotation":
        return quaternion_angle(first, last)
    return float(np.max(np.abs(last - first)))


def seam_checks(document: dict, entries: Sequence[Mapping], clips: Mapping[str, Sequence[Channel]]) -> list[Finding]:
    names = node_names(document)
    findings = []
    for entry in entries:
        channels = clips.get(entry["name"])
        if not entry["loop"] or channels is None:
            continue
        end_s = _clip_end(channels)
        problems = []
        for channel in channels:
            target = f"{names[channel.node]}.{channel.path}"
            times, values = _seam_keys(channel, end_s)
            gap = _value_gap(channel.path, values[0], values[-1])
            if gap > SEAM_TOLERANCE.get(channel.path, SEAM_TOLERANCE["translation"]):
                problems.append(f"{target} ends {gap:.4g} from where it starts")
                continue
            if channel.interpolation == "STEP" or len(times) < 3:
                continue
            seam, interior = velocity_jumps(channel.path, times, values)
            if seam > SEAM_VELOCITY_FACTOR * interior + SEAM_VELOCITY_EPS:
                problems.append(f"{target} velocity jumps {seam:.4g}/s across the seam (largest interior jump {interior:.4g}/s)")
        label = f"{entry['name']}: loop seam"
        if problems:
            findings.append((label, False, "; ".join(problems)))
        else:
            findings.append((label, True, f"{len(channels)} channel(s) close in value and velocity"))
    return findings


def budget_checks(document: dict, entries: Sequence[Mapping], clips: Mapping[str, Sequence[Channel]], kind: str) -> list[Finding]:
    errors = []
    bones = max((len(skin["joints"]) for skin in document.get("skins", [])), default=0)
    max_bones = MAX_BONES.get(kind, min(MAX_BONES.values()))
    if bones > max_bones:
        errors.append(f"{bones} bones in one skin, {kind} allows {max_bones}")
    keyframes = {}
    for entry in entries:
        if entry["duration_s"] > CLIP_MAX_S:
            errors.append(f"{entry['name']} runs {entry['duration_s']} s, longer than {CLIP_MAX_S} s")
        count = sum(len(channel.times) for channel in clips.get(entry["name"], ()))
        keyframes[entry["name"]] = count
        if count > CLIP_MAX_KEYFRAMES:
            errors.append(f"{entry['name']} carries {count} keyframes, more than {CLIP_MAX_KEYFRAMES}")
    if errors:
        return [("animation budget", False, "; ".join(errors))]
    busiest = max(keyframes.values(), default=0)
    detail = f"{bones}/{max_bones} bones, clips at most {CLIP_MAX_S} s, busiest clip {busiest} keyframes"
    return [("animation budget", True, detail)]


class Solid:
    """A closed mesh in its node's local frame, answering point-inside queries by a signed ray
    crossing count along +Y over an XZ bucket grid, so overlapping closed shells still read as
    inside."""

    def __init__(self, positions: np.ndarray, indices: np.ndarray) -> None:
        triangles = positions[indices]
        self.probes = np.vstack([positions, triangles.mean(axis=1)])
        normal_y = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 1]
        facing = np.abs(normal_y) > 1e-12
        self.triangles = triangles[facing]
        self.signs = np.sign(normal_y[facing])
        self.low = positions.min(axis=0)
        self.high = positions.max(axis=0)
        self.cell = np.maximum(self.high[[0, 2]] - self.low[[0, 2]], 1e-9) / SWEEP_GRID_CELLS
        self.buckets = self._bucket(self.triangles[:, :, [0, 2]])

    def _cells(self, xz: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((xz - self.low[[0, 2]]) / self.cell).astype(np.int64), 0, SWEEP_GRID_CELLS - 1)

    def _bucket(self, xz: np.ndarray) -> dict[int, np.ndarray]:
        lows = self._cells(xz.min(axis=1))
        highs = self._cells(xz.max(axis=1))
        buckets: dict[int, list[int]] = {}
        for index, (low, high) in enumerate(zip(lows, highs)):
            for i in range(low[0], high[0] + 1):
                for j in range(low[1], high[1] + 1):
                    buckets.setdefault(i * SWEEP_GRID_CELLS + j, []).append(index)
        return {key: np.array(members, dtype=np.int64) for key, members in buckets.items()}

    def winding(self, points: np.ndarray) -> np.ndarray:
        points = points + RAY_JITTER
        result = np.zeros(len(points), dtype=np.int64)
        within = np.flatnonzero(np.all((points >= self.low) & (points <= self.high), axis=1))
        if len(within) == 0:
            return result
        cells = self._cells(points[within][:, [0, 2]])
        keys = cells[:, 0] * SWEEP_GRID_CELLS + cells[:, 1]
        for key in np.unique(keys):
            members = self.buckets.get(int(key))
            if members is None:
                continue
            index = within[keys == key]
            result[index] = self._crossings(points[index], self.triangles[members], self.signs[members])
        return result

    @staticmethod
    def _crossings(points: np.ndarray, triangles: np.ndarray, signs: np.ndarray) -> np.ndarray:
        a = triangles[:, 0]
        edge_b = triangles[:, 1] - a
        edge_c = triangles[:, 2] - a
        offset = points[:, None, :] - a[None, :, :]
        denominator = edge_b[:, 0] * edge_c[:, 2] - edge_c[:, 0] * edge_b[:, 2]
        u = (offset[:, :, 0] * edge_c[:, 2] - edge_c[:, 0] * offset[:, :, 2]) / denominator
        v = (edge_b[:, 0] * offset[:, :, 2] - offset[:, :, 0] * edge_b[:, 2]) / denominator
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        height = a[:, 1] + u * edge_b[:, 1] + v * edge_c[:, 1]
        above = height > points[:, None, 1]
        return np.sum(np.where(hit & above, signs[None, :], 0.0), axis=1).round().astype(np.int64)

    def deep_inside(self, points: np.ndarray, tolerance: float) -> np.ndarray:
        inside = self.winding(points) > 0
        for axis in range(3):
            for direction in (-1.0, 1.0):
                index = np.flatnonzero(inside)
                if len(index) == 0:
                    return inside
                shift = np.zeros(3)
                shift[axis] = direction * tolerance
                inside[index] = self.winding(points[index] + shift) > 0
        return inside


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _sinking(probe: Solid, solid: Solid, probe_world: np.ndarray, solid_world: np.ndarray) -> np.ndarray:
    relative = np.linalg.inv(solid_world) @ probe_world
    return solid.deep_inside(_transform(relative, probe.probes), CONTACT_TOLERANCE_M)


def sweep_checks(
    document: dict,
    entries: Sequence[Mapping],
    clips: Mapping[str, Sequence[Channel]],
    parts: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> list[Finding]:
    rigid = [entry for entry in entries if entry["kind"] == "rigid" and entry["name"] in clips]
    if not rigid or len(parts) < 2:
        return []
    names = node_names(document)
    solids = {index: Solid(positions, indices) for index, (positions, indices) in parts.items()}
    rest = world_matrices(document)
    findings = []
    for entry in rigid:
        channels = clips[entry["name"]]
        times = np.linspace(0.0, entry["duration_s"], SWEEP_SAMPLES, endpoint=False)
        frames = [world_matrices(document, clip_pose(channels, float(time_s))) for time_s in times]
        moving = [
            index
            for index in solids
            if any(np.max(np.abs(frame[index] - rest[index])) > MOTION_EPS for frame in frames)
        ]
        pairs = []
        for probe in moving:
            for other in solids:
                if other != probe:
                    pairs.append((probe, other))
                    if other not in moving:
                        pairs.append((other, probe))
        collisions = []
        for probe, solid in pairs:
            at_rest = _sinking(solids[probe], solids[solid], rest[probe], rest[solid])
            rest_relative = np.linalg.inv(rest[solid]) @ rest[probe]
            for time_s, frame in zip(times, frames):
                relative = np.linalg.inv(frame[solid]) @ frame[probe]
                if np.max(np.abs(relative - rest_relative)) <= MOTION_EPS:
                    continue
                fresh = int(np.count_nonzero(_sinking(solids[probe], solids[solid], frame[probe], frame[solid]) & ~at_rest))
                if fresh:
                    collisions.append((fresh, float(time_s), names[probe], names[solid]))
        label = f"{entry['name']}: swept collision"
        if collisions:
            worst = sorted(collisions, key=lambda item: (-item[0], item[1]))[:SWEEP_REPORT_LIMIT]
            detail = "; ".join(
                f"{probe} into {solid} at {time_s:.3f} s ({count} point(s) deeper than {CONTACT_TOLERANCE_M} m)"
                for count, time_s, probe, solid in worst
            )
            findings.append((label, False, detail))
        else:
            detail = f"{len(moving)} moving part(s) clear of the other {len(solids) - 1} over {SWEEP_SAMPLES} samples"
            findings.append((label, True, detail))
    return findings
