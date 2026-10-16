"""Cloth panels for flags, pennants, banners and awnings: closed thin slabs with a bone chain.

Pure Python (math only), so geometry, skin weights and layouts are testable without Blender;
assets/cloth.py turns a ClothLayout into a skinned bpy mesh on its rods. Blender axes: Z up,
metres, +Y forward at facing_deg 0.

A panel lives in a local frame: s runs along the free axis from the attached edge (s = 0) to the
free edge (s = length), t runs across it centred on the attachment, n through its thickness. The
slab is closed and manifold, so it is double-sided by construction: a front face at +n/2, a back
face at -n/2 and a rim joining them. It is subdivided along s for skinning. The bone chain runs
along s from the attached edge, each bone's local Z along the cross axis, so bending a bone about
its local Z swings the cloth through its thickness (assets.motion.animate_wind's default
bend_axis) and twisting about local Y (along the bone) flutters it. Skin weights blend linearly
between neighbouring bone centres and are pinned to the first bone at the attached edge, which
therefore never moves.

UVs give the front and back their own islands at one uniform texel density; the rim (one
thickness wide) is collapsed onto the front island's border so it owns no texels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec = tuple[float, float, float]
UV = tuple[float, float]

STANDOFF_M = 0.002
UV_MARGIN = 0.02
DEFAULT_THICKNESS_M = 0.01
SEGMENTS_PER_BONE = 4
CROSS_SEGMENTS = 2
PENNANT_TAPER = 0.92
FLAG_FINIAL_GAP_M = 0.05
ROD_OVERHANG_M = 0.08
MAX_BEND = {"flag": 0.8, "pennant": 0.8, "banner": 0.35, "awning": 0.12}


class ClothError(ValueError):
    pass


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


def normalised(a: Vec) -> Vec:
    length = math.sqrt(dot(a, a))
    if length < 1e-12:
        raise ClothError(f"direction {a!r} has zero length")
    return mul(a, 1.0 / length)


def yawed(a: Vec, facing_deg: float) -> Vec:
    angle = math.radians(facing_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c, a[2])


@dataclass(frozen=True)
class Panel:
    vertices: list[Vec]
    faces: list[tuple[int, int, int, int]]
    uvs: list[tuple[UV, UV, UV, UV]]
    weights: list[list[tuple[int, float]]]
    bone_heads: list[Vec]
    bone_tails: list[Vec]
    free: Vec
    cross_axis: Vec
    normal: Vec

    @property
    def triangle_count(self) -> int:
        return 2 * len(self.faces)


@dataclass(frozen=True)
class Rod:
    start: Vec
    end: Vec
    radius: float


@dataclass(frozen=True)
class ClothLayout:
    kind: str
    panel: Panel
    rods: list[Rod]
    max_bend: float


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ClothError(f"{name} must be positive, got {value!r}")


def _count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ClothError(f"{name} must be a positive integer, got {value!r}")


def chain_weights(s: float, bones: int, length: float) -> list[tuple[int, float]]:
    """Skin weights at distance s along the free axis: linear between neighbouring bone centres,
    the first bone alone up to its centre and the last bone alone past its centre."""
    segment = length / bones
    position = s / segment - 0.5
    if position <= 0.0:
        return [(0, 1.0)]
    if position >= bones - 1:
        return [(bones - 1, 1.0)]
    lower = int(math.floor(position))
    blend = position - lower
    if blend < 1e-9:
        return [(lower, 1.0)]
    return [(lower, 1.0 - blend), (lower + 1, blend)]


def panel(
    origin: Vec,
    free: Vec,
    cross_axis: Vec,
    length: float,
    height: float,
    bones: int,
    thickness: float = DEFAULT_THICKNESS_M,
    taper: float = 0.0,
    segments_per_bone: int = SEGMENTS_PER_BONE,
    cross_segments: int = CROSS_SEGMENTS,
) -> Panel:
    """A closed slab hanging from origin (the middle of its attached edge) along free, height
    across cross_axis, narrowing linearly to (1 - taper) * height at the free edge."""
    _positive("length", length)
    _positive("height", height)
    _positive("thickness", thickness)
    _count("bones", bones)
    _count("segments_per_bone", segments_per_bone)
    _count("cross_segments", cross_segments)
    if not 0.0 <= taper < 1.0:
        raise ClothError(f"taper must be in [0, 1), got {taper!r}")
    free_axis = normalised(free)
    across = normalised(sub(cross_axis, mul(free_axis, dot(cross_axis, free_axis))))
    normal = cross(free_axis, across)
    columns = bones * segments_per_bone
    rows = cross_segments
    half = thickness * 0.5

    def grid(i: int, j: int) -> tuple[float, float]:
        s = length * i / columns
        span = height * (1.0 - taper * s / length)
        return s, span * (j / rows - 0.5)

    def point(s: float, t: float, n: float) -> Vec:
        return add(origin, add(mul(free_axis, s), add(mul(across, t), mul(normal, n))))

    per_face = (columns + 1) * (rows + 1)
    vertices: list[Vec] = []
    weights: list[list[tuple[int, float]]] = []
    for n in (half, -half):
        for i in range(columns + 1):
            for j in range(rows + 1):
                s, t = grid(i, j)
                vertices.append(point(s, t, n))
                weights.append(chain_weights(s, bones, length))

    scale = min((1.0 - 2.0 * UV_MARGIN) / length, (1.0 - 3.0 * UV_MARGIN) / (2.0 * height))

    def front_uv(i: int, j: int) -> UV:
        s, t = grid(i, j)
        return (UV_MARGIN + s * scale, UV_MARGIN + (t + 0.5 * height) * scale)

    def back_uv(i: int, j: int) -> UV:
        u, v = front_uv(i, j)
        return (u, v + height * scale + UV_MARGIN)

    def front(i: int, j: int) -> int:
        return i * (rows + 1) + j

    def back(i: int, j: int) -> int:
        return per_face + front(i, j)

    faces: list[tuple[int, int, int, int]] = []
    uvs: list[tuple[UV, UV, UV, UV]] = []
    for i in range(columns):
        for j in range(rows):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            faces.append(tuple(front(*corner) for corner in corners))
            uvs.append(tuple(front_uv(*corner) for corner in corners))
            reverse = tuple(reversed(corners))
            faces.append(tuple(back(*corner) for corner in reverse))
            uvs.append(tuple(back_uv(*corner) for corner in reverse))

    loop = (
        [(i, 0) for i in range(columns)]
        + [(columns, j) for j in range(rows)]
        + [(i, rows) for i in range(columns, 0, -1)]
        + [(0, j) for j in range(rows, 0, -1)]
    )
    for index, a in enumerate(loop):
        b = loop[(index + 1) % len(loop)]
        faces.append((front(*a), back(*a), back(*b), front(*b)))
        uvs.append((front_uv(*a), front_uv(*a), front_uv(*b), front_uv(*b)))

    segment = length / bones
    heads = [point(index * segment, 0.0, 0.0) for index in range(bones)]
    tails = [point((index + 1) * segment, 0.0, 0.0) for index in range(bones)]
    return Panel(vertices, faces, uvs, weights, heads, tails, free_axis, across, normal)


def flag(
    base: Vec = (0.0, 0.0, 0.0),
    pole_height: float = 3.0,
    width: float = 1.2,
    height: float = 0.8,
    facing_deg: float = 0.0,
    pole_radius: float = 0.03,
    bones: int = 6,
    thickness: float = DEFAULT_THICKNESS_M,
    taper: float = 0.0,
    kind: str = "flag",
) -> ClothLayout:
    """A flag flying from a vertical pole standing on base, out along +X turned by facing_deg,
    its top FLAG_FINIAL_GAP_M under the pole top and its pole edge STANDOFF_M off the pole."""
    _positive("pole_height", pole_height)
    _positive("pole_radius", pole_radius)
    if height + FLAG_FINIAL_GAP_M >= pole_height:
        raise ClothError(f"a {height} m flag does not fit on a {pole_height} m pole")
    free = yawed((1.0, 0.0, 0.0), facing_deg)
    top = base[2] + pole_height - FLAG_FINIAL_GAP_M
    origin = add((base[0], base[1], top - 0.5 * height), mul(free, pole_radius + STANDOFF_M))
    cloth = panel(origin, free, (0.0, 0.0, 1.0), width, height, bones, thickness, taper)
    pole = Rod(base, (base[0], base[1], base[2] + pole_height), pole_radius)
    return ClothLayout(kind, cloth, [pole], MAX_BEND[kind])


def pennant(
    base: Vec = (0.0, 0.0, 0.0),
    pole_height: float = 3.0,
    width: float = 1.0,
    height: float = 0.4,
    facing_deg: float = 0.0,
    pole_radius: float = 0.03,
    bones: int = 6,
    thickness: float = DEFAULT_THICKNESS_M,
) -> ClothLayout:
    """A flag tapering to a point at its free end."""
    return flag(base, pole_height, width, height, facing_deg, pole_radius, bones, thickness, PENNANT_TAPER, "pennant")


def banner(
    top_centre: Vec = (0.0, 0.0, 3.0),
    width: float = 0.8,
    length: float = 1.6,
    facing_deg: float = 0.0,
    rod_radius: float = 0.02,
    bones: int = 6,
    thickness: float = DEFAULT_THICKNESS_M,
    taper: float = 0.0,
) -> ClothLayout:
    """A banner hanging down from a horizontal rod through top_centre, its front facing +Y
    turned by facing_deg, the rod overhanging each side by ROD_OVERHANG_M."""
    _positive("rod_radius", rod_radius)
    across = yawed((-1.0, 0.0, 0.0), facing_deg)
    origin = sub(top_centre, (0.0, 0.0, rod_radius + STANDOFF_M))
    cloth = panel(origin, (0.0, 0.0, -1.0), across, length, width, bones, thickness, taper)
    reach = mul(across, 0.5 * width + ROD_OVERHANG_M)
    rod = Rod(sub(top_centre, reach), add(top_centre, reach), rod_radius)
    return ClothLayout("banner", cloth, [rod], MAX_BEND["banner"])


def awning(
    edge_centre: Vec = (0.0, 0.0, 2.4),
    width: float = 2.0,
    depth: float = 1.0,
    slope_deg: float = 25.0,
    facing_deg: float = 0.0,
    rod_radius: float = 0.025,
    bones: int = 4,
    thickness: float = DEFAULT_THICKNESS_M,
) -> ClothLayout:
    """An awning running out and down from a horizontal wall rod through edge_centre, sloping
    slope_deg below horizontal towards +Y turned by facing_deg, its front face upward."""
    _positive("rod_radius", rod_radius)
    if not 0.0 <= slope_deg < 90.0:
        raise ClothError(f"slope_deg must be in [0, 90), got {slope_deg!r}")
    slope = math.radians(slope_deg)
    free = yawed((0.0, math.cos(slope), -math.sin(slope)), facing_deg)
    across = yawed((-1.0, 0.0, 0.0), facing_deg)
    origin = add(edge_centre, mul(free, rod_radius + STANDOFF_M))
    cloth = panel(origin, free, across, depth, width, bones, thickness)
    reach = mul(across, 0.5 * width + ROD_OVERHANG_M)
    rod = Rod(sub(edge_centre, reach), add(edge_centre, reach), rod_radius)
    return ClothLayout("awning", cloth, [rod], MAX_BEND["awning"])


BUILDERS = {"flag": flag, "pennant": pennant, "banner": banner, "awning": awning}
