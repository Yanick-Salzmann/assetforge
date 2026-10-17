"""Scatter vegetation geometry: bark tubes and alpha-cutout foliage cards over the flora atlas.

Pure Python (math and random only), so tri counts and placement are testable without Blender;
assets/blender_scripts/flora_build.py turns each MeshData into a bpy mesh.

Every builder takes lod (0..LODS-1). A reduced LOD replays LOD0's random draws and keeps every
LOD_STRIDES[lod]-th card, widened by LOD_WIDEN[lod], with fewer tube sides and card segments.
A crown (broadleaf, bush) keeps its outermost cards instead; ferns only lose card segments, and
grass and flower tufts lose segments first and cards (TUFT_STRIDES) only at LOD2. Blender axes: Z up,
origin at base centre, metres. Cards are single-sided quads rendered double-sided; each faces
outward from its plant and carries custom normals bent toward that outward direction, so a crown
shades as a volume rather than as a heap of flat planes. Card UVs map a whole foliage cell of
assets/foliage_atlas.py, bottom centre at the attachment point.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from assets.foliage_atlas import Rect, bark_rect, cell_rect

Vec = tuple[float, float, float]

Z: Vec = (0.0, 0.0, 1.0)
NORMAL_BEND = 0.9
GROUND_Z = 0.0
BARK_TILES = 4
LOD_STRIDES: tuple[int, ...] = (1, 2, 3)
LOD_WIDEN: tuple[float, ...] = (1.0, 1.4, 1.9)
CROWN_WIDEN: tuple[float, ...] = (1.0, 1.25, 1.3)
TUFT_STRIDES: tuple[int, ...] = (1, 1, 2)
SPRUCE_MIN_REACH = 0.035
SPRUCE_CARD_WIDTH = 1.1
SPRUCE_ROLL = (0.55, 1.25)
SPRUCE_SEGMENTED_BELOW = 0.3
SPRUCE_LEADER_SPRAYS = 3
SPRUCE_TRUNK_REACH = 0.9
TUFT_WIDEN: tuple[float, ...] = (1.0, 1.0, 1.4)
LODS = len(LOD_STRIDES)


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec) -> Vec:
    size = length(a)
    if size < 1e-9:
        return Z
    return mul(a, 1.0 / size)


def rotate(v: Vec, axis: Vec, angle: float) -> Vec:
    k = unit(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1.0 - c)))


def perpendicular(a: Vec, b: Vec) -> Vec:
    """b with its component along unit a removed, normalised."""
    return unit(sub(b, mul(a, dot(a, b))))


def kept(index: int, lod: int, strides: tuple[int, ...] = LOD_STRIDES) -> bool:
    return index % strides[lod] == 0


def segments_at(segments: int, lod: int) -> int:
    return max(1, segments - lod)


def sides_at(sides: int, lod: int) -> int:
    return max(3, sides - 2 * lod)


@dataclass
class MeshData:
    """Faces index vertices; uvs and normals are per face corner."""

    vertices: list[Vec] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    uvs: list[list[tuple[float, float]]] = field(default_factory=list)
    normals: list[list[Vec]] = field(default_factory=list)

    def triangles(self) -> int:
        return sum(len(face) - 2 for face in self.faces)

    def bounds(self) -> tuple[Vec, Vec]:
        xs, ys, zs = zip(*self.vertices)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


class Builder:
    def __init__(self) -> None:
        self.mesh = MeshData()

    def _vertex(self, point: Vec) -> int:
        self.mesh.vertices.append((point[0], point[1], max(point[2], GROUND_Z)))
        return len(self.mesh.vertices) - 1

    def _face(self, indices: tuple[int, ...], uvs: list[tuple[float, float]], normals: list[Vec]) -> None:
        self.mesh.faces.append(indices)
        self.mesh.uvs.append(uvs)
        self.mesh.normals.append(normals)

    def card(
        self,
        base: Vec,
        up: Vec,
        side: Vec,
        size: float,
        width: float,
        rect: Rect,
        outward: Vec,
        segments: int = 1,
        droop: float = 0.0,
    ) -> None:
        """A strip of quads growing from base along up, bending down by droop radians at the tip.
        Its front faces outward; corner normals lean toward outward by NORMAL_BEND."""
        up = unit(up)
        side = perpendicular(up, side)
        if dot(cross(side, up), outward) < 0.0:
            side = mul(side, -1.0)
        spine = [base]
        directions = []
        for index in range(segments):
            t = (index + 0.5) / segments
            pitch_axis = cross(up, Z)
            direction = rotate(up, pitch_axis, -droop * t) if length(pitch_axis) > 1e-6 else up
            directions.append(direction)
            spine.append(add(spine[-1], mul(direction, size / segments)))
        directions.append(directions[-1])
        half = width * 0.5
        rows = []
        for index, point in enumerate(spine):
            across = perpendicular(directions[index], side)
            rows.append((self._vertex(sub(point, mul(across, half))), self._vertex(add(point, mul(across, half)))))
        bent = unit(add(unit(cross(side, up)), mul(unit(outward), NORMAL_BEND)))
        for index in range(segments):
            v0 = index / segments
            v1 = (index + 1) / segments
            left0, right0 = rows[index]
            left1, right1 = rows[index + 1]
            self._face(
                (left0, right0, right1, left1),
                [rect.at(0.0, v0), rect.at(1.0, v0), rect.at(1.0, v1), rect.at(0.0, v1)],
                [bent, bent, bent, bent],
            )

    def tube(self, path: list[Vec], radii: list[float], sides: int, rect: Rect) -> None:
        """An open bark tube along path (no caps - the base sits in the ground, the tip is a
        point). u wraps the circumference once; v runs BARK_TILES square bark tiles per
        base circumference, squeezed to fit if the tube is longer than the strip."""
        circumference = 2.0 * math.pi * radii[0]
        span = BARK_TILES * circumference
        total = sum(length(sub(b, a)) for a, b in zip(path, path[1:]))
        squeeze = min(1.0, span / max(total, 1e-6))
        rings = []
        normals = []
        travelled = 0.0
        vs = []
        reference = None
        for index, point in enumerate(path):
            ahead = path[min(index + 1, len(path) - 1)]
            behind = path[max(index - 1, 0)]
            tangent = unit(sub(ahead, behind))
            reference = perpendicular(tangent, reference if reference is not None else (1.0, 0.0, 0.0) if abs(tangent[0]) < 0.9 else (0.0, 1.0, 0.0))
            other = cross(tangent, reference)
            if index:
                travelled += length(sub(point, path[index - 1]))
            vs.append(travelled / span * squeeze)
            ring = []
            ring_normals = []
            for step in range(sides + 1):
                angle = 2.0 * math.pi * step / sides
                radial = add(mul(reference, math.cos(angle)), mul(other, math.sin(angle)))
                ring.append(self._vertex(add(point, mul(radial, radii[index]))))
                ring_normals.append(radial)
            rings.append(ring)
            normals.append(ring_normals)
        for index in range(len(path) - 1):
            for step in range(sides):
                u0 = step / sides
                u1 = (step + 1) / sides
                lower = rings[index]
                upper = rings[index + 1]
                self._face(
                    (lower[step], lower[step + 1], upper[step + 1], upper[step]),
                    [rect.at(u0, vs[index]), rect.at(u1, vs[index]), rect.at(u1, vs[index + 1]), rect.at(u0, vs[index + 1])],
                    [normals[index][step], normals[index][step + 1], normals[index + 1][step + 1], normals[index + 1][step]],
                )


def _trunk_path(rng: random.Random, height: float, segments: int, lean: float) -> list[Vec]:
    heading = rng.uniform(0.0, 2.0 * math.pi)
    points = []
    for index in range(segments + 1):
        t = index / segments
        sway = lean * height * t * t
        wobble = 0.015 * height * math.sin(t * 7.0 + heading)
        points.append((math.cos(heading) * sway + wobble * 0.3, math.sin(heading) * sway, height * t))
    return points


def _point_on(path: list[Vec], z: float) -> Vec:
    for a, b in zip(path, path[1:]):
        if a[2] <= z <= b[2]:
            t = (z - a[2]) / max(b[2] - a[2], 1e-9)
            return add(a, mul(sub(b, a), t))
    return path[-1]


def spruce(
    seed: int,
    height: float,
    crown_radius: float,
    cell: str = "spruce",
    crown_base: float = 0.18,
    tiers: int = 19,
    per_tier: int = 13,
    lod: int = 0,
) -> MeshData:
    """A spruce: a tapering trunk and whorls of needle-spray cards, long and drooping at the
    bottom, short and lifted near the leader, with a leader of upturned sprays hiding the tip."""
    rng = random.Random(seed)
    builder = Builder()
    rect = cell_rect(cell)
    path = _trunk_path(rng, height * SPRUCE_TRUNK_REACH, 4, 0.01)
    base_radius = 0.022 * height
    builder.tube(path, [base_radius * 1.25, base_radius * 0.75, base_radius * 0.42, base_radius * 0.16, 0.01], sides_at(6, lod), bark_rect())
    widen = LOD_WIDEN[lod]
    for tier in range(tiers):
        f = tier / (tiers - 1)
        z = height * (crown_base + (0.93 - crown_base) * f)
        reach = crown_radius * (1.0 - f) ** 0.85 * rng.uniform(0.85, 1.1) + SPRUCE_MIN_REACH * height
        count = max(3, round(per_tier * (1.0 - 0.45 * f)))
        phase = rng.uniform(0.0, 2.0 * math.pi)
        centre = _point_on(path, z)
        for index in range(count):
            azimuth = phase + 2.0 * math.pi * (index + rng.uniform(-0.3, 0.3)) / count
            horizontal = (math.cos(azimuth), math.sin(azimuth), 0.0)
            pitch = -0.2 + 0.55 * f + rng.uniform(-0.08, 0.08)
            up = unit(add(mul(horizontal, math.cos(pitch)), mul(Z, math.sin(pitch))))
            side = rotate((-horizontal[1], horizontal[0], 0.0), up, rng.choice((-1.0, 1.0)) * rng.uniform(*SPRUCE_ROLL))
            if not kept(index + tier, lod):
                continue
            builder.card(
                add(centre, mul(horizontal, 0.04)),
                up,
                side,
                reach,
                reach * SPRUCE_CARD_WIDTH * widen,
                rect,
                outward=unit(add(horizontal, mul(Z, 0.35))),
                segments=segments_at(2 if f < SPRUCE_SEGMENTED_BELOW else 1, lod),
                droop=0.45 * (1.0 - f),
            )
    sprays = SPRUCE_LEADER_SPRAYS if lod < 2 else SPRUCE_LEADER_SPRAYS - 1
    base = _point_on(path, height * 0.85)
    phase = rng.uniform(0.0, math.pi)
    for index in range(sprays):
        angle = phase + math.pi * index / sprays
        side = (math.cos(angle), math.sin(angle), 0.0)
        length = height * 0.16
        builder.card(
            base,
            unit(add(Z, mul(cross(side, Z), 0.04))),
            side,
            length,
            length * 0.32 * widen,
            rect,
            outward=cross(side, Z),
        )
    return builder.mesh


def broadleaf(
    seed: int,
    trunk_height: float,
    crown: Vec,
    cell: str,
    cards: int,
    card_size: float,
    trunk_radius: float,
    branches: int = 5,
    branch_lift: float = 0.6,
    lod: int = 0,
) -> MeshData:
    """A broadleaf tree: a trunk forking into branches that reach into an ellipsoidal crown of
    leaf-cluster cards, denser toward the shell than the core."""
    rng = random.Random(seed)
    builder = Builder()
    rect = cell_rect(cell)
    path = _trunk_path(rng, trunk_height * 1.1, 3, 0.03)
    builder.tube(path, [trunk_radius * 1.35, trunk_radius, trunk_radius * 0.85, trunk_radius * 0.7], sides_at(7, lod), bark_rect())
    centre = (path[-1][0], path[-1][1], trunk_height + crown[2] * 0.85)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for index in range(branches):
        azimuth = phase + 2.0 * math.pi * (index + rng.uniform(-0.25, 0.25)) / branches
        start = _point_on(path, trunk_height * rng.uniform(0.75, 1.0))
        direction = unit((math.cos(azimuth), math.sin(azimuth), branch_lift))
        tip = add(centre, (direction[0] * crown[0] * 0.75, direction[1] * crown[1] * 0.75, (direction[2] - 0.2) * crown[2] * 0.6))
        middle = add(mul(add(start, tip), 0.5), (0.0, 0.0, crown[2] * 0.12))
        builder.tube([start, middle, tip], [trunk_radius * 0.6, trunk_radius * 0.35, 0.02], sides_at(5, lod), bark_rect())
    shell = []
    for _ in range(cards):
        direction = unit((rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), abs(rng.gauss(0.0, 1.0)) * 1.2 - 0.35))
        reach = 0.5 + 0.5 * math.sqrt(rng.random())
        position = add(centre, (direction[0] * crown[0] * reach, direction[1] * crown[1] * reach, direction[2] * crown[2] * reach))
        facing = unit(add(direction, (rng.gauss(0.0, 0.5), rng.gauss(0.0, 0.5), rng.gauss(0.0, 0.5))))
        up = rotate(perpendicular(facing, Z), facing, rng.uniform(-0.7, 0.7))
        size = card_size * rng.uniform(0.8, 1.2)
        shell.append(CrownCard(reach, position, direction, up, facing, size))
    _crown(builder, shell, rect, lod)
    return builder.mesh


@dataclass(frozen=True)
class CrownCard:
    reach: float
    position: Vec
    direction: Vec
    up: Vec
    facing: Vec
    size: float


def _crown(builder: Builder, shell: list[CrownCard], rect: Rect, lod: int) -> None:
    """Square cluster cards centred on their positions. A reduced LOD keeps the outermost
    1 / LOD_STRIDES[lod] of them by reach - the inner ones are hidden behind the shell anyway -
    widened by CROWN_WIDEN[lod]."""
    count = math.ceil(len(shell) / LOD_STRIDES[lod])
    order = sorted(range(len(shell)), key=lambda index: (-shell[index].reach, index))
    widen = CROWN_WIDEN[lod]
    for index in sorted(order[:count]):
        card = shell[index]
        size = card.size * widen
        builder.card(sub(card.position, mul(card.up, size * 0.5)), card.up, cross(card.up, card.facing), size, size, rect, outward=card.direction)


def bush(seed: int, radii: Vec, cell: str, cards: int, card_size: float, lod: int = 0) -> MeshData:
    """A shrub: a dome of cluster cards rooted near the ground, no trunk (the cell's own twigs
    read as stems)."""
    rng = random.Random(seed)
    builder = Builder()
    rect = cell_rect(cell)
    centre = (0.0, 0.0, radii[2] * 0.45)
    shell = []
    for _ in range(cards):
        direction = unit((rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), abs(rng.gauss(0.0, 1.0)) * 1.4 - 0.2))
        reach = 0.5 + 0.5 * math.sqrt(rng.random())
        position = add(centre, (direction[0] * radii[0] * reach, direction[1] * radii[1] * reach, direction[2] * radii[2] * reach))
        facing = unit(add((direction[0], direction[1], direction[2] * 0.35), (rng.gauss(0.0, 0.3), rng.gauss(0.0, 0.3), rng.gauss(0.0, 0.2))))
        up = rotate(perpendicular(facing, Z), facing, rng.uniform(-0.6, 0.6))
        size = card_size * rng.uniform(0.8, 1.2)
        shell.append(CrownCard(reach, position, direction, up, facing, size))
    _crown(builder, shell, rect, lod)
    return builder.mesh


def fern(seed: int, frond_length: float, fronds: int, upright: int = 0, cell: str = "fern", lod: int = 0) -> MeshData:
    """A fern: arching frond cards radiating from the crown, plus optional young upright ones."""
    rng = random.Random(seed)
    builder = Builder()
    rect = cell_rect(cell)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for index in range(fronds + upright):
        young = index >= fronds
        count = upright if young else fronds
        azimuth = phase + 2.0 * math.pi * (index + rng.uniform(-0.25, 0.25)) / count
        horizontal = (math.cos(azimuth), math.sin(azimuth), 0.0)
        pitch = rng.uniform(1.25, 1.45) if young else rng.uniform(0.95, 1.2)
        up = unit(add(mul(horizontal, math.cos(pitch)), mul(Z, math.sin(pitch))))
        size = frond_length * (rng.uniform(0.55, 0.7) if young else rng.uniform(0.85, 1.1))
        side = rotate((-horizontal[1], horizontal[0], 0.0), up, rng.uniform(-0.3, 0.3))
        builder.card(
            add((0.0, 0.0, 0.02), mul(horizontal, 0.04)),
            up,
            side,
            size,
            size * 0.9,
            rect,
            outward=unit(add(horizontal, Z)),
            segments=segments_at(3, lod),
            droop=0.4 if young else 1.0,
        )
    return builder.mesh


def tuft(seed: int, height: float, cell: str, cards: int = 10, width: float = 1.0, lean: float = 0.3, lod: int = 0) -> MeshData:
    """Grass or flowering stems: crossed upright cards leaning slightly out from a small base."""
    rng = random.Random(seed)
    builder = Builder()
    rect = cell_rect(cell)
    phase = rng.uniform(0.0, math.pi)
    for index in range(cards):
        angle = phase + math.pi * (index + rng.uniform(-0.2, 0.2)) / cards
        side = (math.cos(angle), math.sin(angle), 0.0)
        facing = (-side[1], side[0], 0.0)
        if index % 2:
            facing = mul(facing, -1.0)
        offset = mul(facing, rng.uniform(0.0, 0.06) * height)
        up = unit(add(Z, mul(facing, rng.uniform(0.0, lean))))
        size = height * rng.uniform(0.85, 1.15)
        droop = rng.uniform(0.05, 0.25)
        if kept(index, lod, TUFT_STRIDES):
            builder.card(offset, up, side, size, size * width * TUFT_WIDEN[lod], rect, outward=unit(add(facing, Z)), segments=segments_at(2, lod), droop=droop)
    return builder.mesh


MODELS: dict[str, tuple[str, Callable[..., MeshData]]] = {
    "spruce_a": ("conifer", lambda lod=0: spruce(101, 11.0, 2.6, lod=lod)),
    "spruce_b": ("conifer", lambda lod=0: spruce(102, 8.5, 2.1, tiers=18, lod=lod)),
    "spruce_c": ("conifer", lambda lod=0: spruce(103, 5.0, 1.6, cell="spruce_alpine", crown_base=0.08, tiers=17, per_tier=14, lod=lod)),
    "spruce_d": ("conifer", lambda lod=0: spruce(104, 6.5, 1.7, cell="spruce_alpine", crown_base=0.1, tiers=18, lod=lod)),
    "oak_a": ("broadleaf", lambda lod=0: broadleaf(201, 2.6, (3.4, 3.2, 2.9), "oak", 210, 1.7, 0.3, lod=lod)),
    "oak_b": ("broadleaf", lambda lod=0: broadleaf(202, 3.0, (2.8, 2.6, 3.4), "oak", 210, 1.6, 0.27, lod=lod)),
    "mesquite_a": ("broadleaf", lambda lod=0: broadleaf(301, 1.2, (3.2, 3.0, 1.5), "mesquite", 200, 1.5, 0.2, branch_lift=0.35, lod=lod)),
    "mesquite_b": ("broadleaf", lambda lod=0: broadleaf(302, 1.5, (2.6, 2.4, 1.8), "mesquite", 200, 1.4, 0.18, branches=4, branch_lift=0.45, lod=lod)),
    "hazel_a": ("shrub", lambda lod=0: bush(401, (0.8, 0.75, 0.75), "hazel", 95, 0.8, lod=lod)),
    "hazel_b": ("shrub", lambda lod=0: bush(402, (0.6, 0.6, 0.55), "hazel", 90, 0.65, lod=lod)),
    "sage_a": ("shrub", lambda lod=0: bush(501, (0.5, 0.45, 0.4), "sage", 80, 0.5, lod=lod)),
    "sage_b": ("shrub", lambda lod=0: bush(502, (0.4, 0.4, 0.45), "sage", 70, 0.45, lod=lod)),
    "juniper_a": ("shrub", lambda lod=0: bush(601, (0.85, 0.75, 0.3), "spruce_alpine", 90, 0.7, lod=lod)),
    "juniper_b": ("shrub", lambda lod=0: bush(602, (0.6, 0.55, 0.35), "spruce_alpine", 80, 0.55, lod=lod)),
    "fern_a": ("shrub", lambda lod=0: fern(701, 0.9, 9, lod=lod)),
    "fern_b": ("shrub", lambda lod=0: fern(702, 0.7, 7, upright=4, lod=lod)),
    "grass_a": ("grass", lambda lod=0: tuft(801, 0.5, "grass", cards=12, lod=lod)),
    "grass_b": ("grass", lambda lod=0: tuft(802, 0.35, "grass", cards=10, lod=lod)),
    "grass_dry_a": ("grass", lambda lod=0: tuft(803, 0.55, "grass_dry", cards=12, lod=lod)),
    "grass_dry_b": ("grass", lambda lod=0: tuft(804, 0.4, "grass_dry", cards=10, lod=lod)),
    "flower_purple": ("flower", lambda lod=0: tuft(901, 0.5, "flower_purple", cards=6, lean=0.2, lod=lod)),
    "flower_yellow": ("flower", lambda lod=0: tuft(902, 0.45, "flower_yellow", cards=6, lean=0.2, lod=lod)),
    "flower_red": ("flower", lambda lod=0: tuft(903, 0.45, "flower_red", cards=6, lean=0.2, lod=lod)),
}
