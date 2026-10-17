"""Hemi-octahedral impostor layout: the far LOD of scatter vegetation (af-8m0.9).

An impostor atlas is FRAMES x FRAMES orthographic views of LOD0 from the upper hemisphere. Frame
(column, row), row 0 at the bottom of the image (v up, like glTF UVs read with flipY), looks along
frame_direction(column, row): with u, v the frame centre mapped to [-1, 1],
a = (u + v) / 2, b = (u - v) / 2 and h = 1 - |a| - |b|, the direction toward the camera is
normalise(a, h, b) in glTF axes (Y up). encode() is the inverse a viewer uses to pick frames.

Each frame is framed on LOD0's bounding sphere (centre, radius): the frame spans 2 * radius
across, centred on centre, with its up being glTF +Y projected onto the view plane, or glTF -Z
when looking straight down. Blender is Z up, so a glTF (x, y, z) is Blender (x, -z, y).
"""

from __future__ import annotations

import math

import numpy as np

FRAMES = 8
FRAME_PX = 128
LAYOUT = "hemi_octahedral"

Vec = tuple[float, float, float]


def frame_uv(column: int, row: int, frames: int = FRAMES) -> tuple[float, float]:
    return (column + 0.5) / frames * 2.0 - 1.0, (row + 0.5) / frames * 2.0 - 1.0


def decode(u: float, v: float) -> Vec:
    """The glTF direction toward the camera for a point (u, v) of [-1, 1]^2."""
    a = (u + v) * 0.5
    b = (u - v) * 0.5
    h = 1.0 - abs(a) - abs(b)
    size = math.sqrt(a * a + h * h + b * b)
    return a / size, h / size, b / size


def encode(direction: Vec) -> tuple[float, float]:
    """(u, v) in [-1, 1]^2 of a glTF direction; below-horizon directions fold onto the rim."""
    x, y, z = direction
    y = max(y, 0.0)
    total = abs(x) + y + abs(z)
    a = x / total
    b = z / total
    return a + b, a - b


def frame_direction(column: int, row: int, frames: int = FRAMES) -> Vec:
    return decode(*frame_uv(column, row, frames))


def gltf_to_blender(vector: Vec) -> Vec:
    return vector[0], -vector[2], vector[1]


def blender_to_gltf(vector: Vec) -> Vec:
    return vector[0], vector[2], -vector[1]


def view_basis(direction: Vec) -> tuple[np.ndarray, np.ndarray]:
    """Screen right and up, Blender axes, for a camera looking back along direction (Blender
    axes): up is +Z projected onto the view plane, or +Y (glTF -Z) when looking straight down."""
    view = np.asarray(direction, dtype=np.float64)
    view = view / np.linalg.norm(view)
    world_up = np.array((0.0, 0.0, 1.0))
    if abs(view @ world_up) > 0.999:
        world_up = np.array((0.0, 1.0, 0.0))
    up = world_up - view * (view @ world_up)
    up /= np.linalg.norm(up)
    right = np.cross(up, view)
    return right, up


def frame_rotation(direction: Vec) -> np.ndarray:
    """The rotation (Blender axes) that turns direction to +Z and its screen up to +Y, so a copy
    of the model rotated by it and seen from above by a +Y-up camera shows that frame."""
    right, up = view_basis(direction)
    view = np.asarray(direction, dtype=np.float64)
    return np.stack([right, up, view / np.linalg.norm(view)])
