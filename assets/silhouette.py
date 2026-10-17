"""Orthographic alpha-cutout silhouettes of a flora mesh, in numpy, for comparing LODs.

A silhouette is the boolean coverage of a MeshData seen along a direction, its triangles textured
by an alpha atlas (rows bottom-up, like assets/foliage_atlas.py) and cut at ALPHA_CUTOFF. Every
LOD of a model is framed by LOD0's bounding sphere, so masks of different LODs line up pixel for
pixel and their disagreement is the silhouette pop a LOD switch shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from assets.flora import MeshData, Vec
from assets.impostor import view_basis

ALPHA_CUTOFF = 0.5
VIEWS: dict[str, Vec] = {
    "front": (0.0, -1.0, 0.0),
    "side": (1.0, 0.0, 0.0),
    "three_quarter": (0.6, -0.6, 0.53),
    "top": (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class Frame:
    centre: Vec
    radius: float

    @classmethod
    def around(cls, mesh: MeshData) -> Frame:
        points = np.asarray(mesh.vertices, dtype=np.float64)
        centre = (points.min(axis=0) + points.max(axis=0)) * 0.5
        radius = float(np.linalg.norm(points - centre, axis=1).max())
        return cls(tuple(centre), radius)


def coverage(mesh: MeshData, alpha: np.ndarray, direction: Vec, size: int, frame: Frame) -> np.ndarray:
    """bool[size, size], row 0 at the top: which pixels the cut-out mesh covers."""
    right, up = view_basis(direction)
    points = np.asarray(mesh.vertices, dtype=np.float64) - np.asarray(frame.centre)
    scale = size / (2.0 * frame.radius)
    xs = (points @ right) * scale + size * 0.5
    ys = size * 0.5 - (points @ up) * scale
    height, width = alpha.shape
    mask = np.zeros((size, size), dtype=bool)
    for face, uvs in zip(mesh.faces, mesh.uvs):
        for corner in range(1, len(face) - 1):
            picks = (0, corner, corner + 1)
            tri_x = xs[[face[pick] for pick in picks]]
            tri_y = ys[[face[pick] for pick in picks]]
            tri_uv = np.array([uvs[pick] for pick in picks], dtype=np.float64)
            _fill(mask, tri_x, tri_y, tri_uv, alpha, width, height)
    return mask


def _fill(mask: np.ndarray, tri_x: np.ndarray, tri_y: np.ndarray, tri_uv: np.ndarray, alpha: np.ndarray, width: int, height: int) -> None:
    size = mask.shape[0]
    x0 = max(int(math.floor(tri_x.min())), 0)
    x1 = min(int(math.ceil(tri_x.max())), size - 1)
    y0 = max(int(math.floor(tri_y.min())), 0)
    y1 = min(int(math.ceil(tri_y.max())), size - 1)
    if x0 > x1 or y0 > y1:
        return
    area = (tri_x[1] - tri_x[0]) * (tri_y[2] - tri_y[0]) - (tri_x[2] - tri_x[0]) * (tri_y[1] - tri_y[0])
    if abs(area) < 1e-12:
        return
    px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    w1 = ((px - tri_x[0]) * (tri_y[2] - tri_y[0]) - (tri_x[2] - tri_x[0]) * (py - tri_y[0])) / area
    w2 = ((tri_x[1] - tri_x[0]) * (py - tri_y[0]) - (px - tri_x[0]) * (tri_y[1] - tri_y[0])) / area
    w0 = 1.0 - w1 - w2
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not inside.any():
        return
    u = w0 * tri_uv[0, 0] + w1 * tri_uv[1, 0] + w2 * tri_uv[2, 0]
    v = w0 * tri_uv[0, 1] + w1 * tri_uv[1, 1] + w2 * tri_uv[2, 1]
    columns = np.clip((u * width).astype(np.int64), 0, width - 1)
    rows = np.clip((v * height).astype(np.int64), 0, height - 1)
    opaque = inside & (alpha[rows, columns] >= ALPHA_CUTOFF)
    mask[y0:y1 + 1, x0:x1 + 1] |= opaque


def pop(reference: np.ndarray, other: np.ndarray) -> float:
    """Pixels the two masks disagree on, as a fraction of the reference's coverage."""
    covered = int(reference.sum())
    if covered == 0:
        return 0.0 if not other.any() else 1.0
    return float((reference ^ other).sum()) / covered


def blurred_pop(reference: np.ndarray, other: np.ndarray, pixels: int) -> float:
    """pop of the two masks box-averaged down to pixels x pixels: coverage differences finer
    than one screen pixel at that size read as the same grey and do not count."""
    return float(np.abs(_downsample(reference, pixels) - _downsample(other, pixels)).sum() / max(_downsample(reference, pixels).sum(), 1e-9))


def _downsample(mask: np.ndarray, pixels: int) -> np.ndarray:
    factor = mask.shape[0] // pixels
    trimmed = mask[: factor * pixels, : factor * pixels].astype(np.float64)
    return trimmed.reshape(pixels, factor, pixels, factor).mean(axis=(1, 3))
