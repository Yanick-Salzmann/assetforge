"""The flora atlas: one 2048 RGBA texture shared by every generated vegetation prop.

Pure numpy, no bpy and no I/O, so the same code runs inside Blender's interpreter and under the
uv test suite. Pixel rows run bottom-up (row 0 is v = 0), the order bpy.types.Image.pixels uses.
The left strip holds tiled bark; the rest is a grid of foliage cells drawn procedurally -
needle sprays, leaf clusters, fern fronds, grass blades and flowering stems - each with a hard
alpha cutout and its colour bled into the transparent margin so mipmaps do not fringe dark.
Every cell grows from its bottom centre upward, which is how foliage.py maps its cards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

ATLAS_SIZE = 2048
BARK_WIDTH = 512
CELL_SIZE = 512
COLUMNS = 3
ROWS = 4
ALPHA_CUTOFF = 0.5
GUTTER = 8
BLEED_PASSES = 24

Colour = tuple[float, float, float]

TWIG: Colour = (0.23, 0.16, 0.10)
CELL_NAMES = (
    "spruce",
    "spruce_alpine",
    "oak",
    "mesquite",
    "hazel",
    "sage",
    "fern",
    "grass",
    "grass_dry",
    "flower_purple",
    "flower_yellow",
    "flower_red",
)


@dataclass(frozen=True)
class Rect:
    """A UV rectangle in [0, 1]^2, v up."""

    u0: float
    v0: float
    u1: float
    v1: float

    def at(self, u: float, v: float) -> tuple[float, float]:
        return self.u0 + (self.u1 - self.u0) * u, self.v0 + (self.v1 - self.v0) * v


def cell_rect(name: str) -> Rect:
    index = CELL_NAMES.index(name)
    column = index % COLUMNS
    row = index // COLUMNS
    x0 = BARK_WIDTH + column * CELL_SIZE + GUTTER
    y0 = row * CELL_SIZE + GUTTER
    inner = CELL_SIZE - 2 * GUTTER
    return Rect(x0 / ATLAS_SIZE, y0 / ATLAS_SIZE, (x0 + inner) / ATLAS_SIZE, (y0 + inner) / ATLAS_SIZE)


def bark_rect() -> Rect:
    inset = 2.0 / ATLAS_SIZE
    return Rect(inset, 0.0, BARK_WIDTH / ATLAS_SIZE - inset, 1.0)


def _jitter(rng: random.Random, colour: Colour, amount: float) -> np.ndarray:
    shade = rng.uniform(1.0 - amount, 1.0 + amount)
    hue = [rng.uniform(-amount, amount) * 0.25 for _ in range(3)]
    return np.clip(np.array([channel * shade + offset for channel, offset in zip(colour, hue)], dtype=np.float32), 0.0, 1.0)


class Canvas:
    """An RGBA float canvas, y up, with painter's-order shape rasterisers."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.pixels = np.zeros((size, size, 4), dtype=np.float32)

    def _window(self, x0: float, y0: float, x1: float, y1: float):
        ix0 = max(int(math.floor(x0)), 0)
        iy0 = max(int(math.floor(y0)), 0)
        ix1 = min(int(math.ceil(x1)) + 1, self.size)
        iy1 = min(int(math.ceil(y1)) + 1, self.size)
        if ix0 >= ix1 or iy0 >= iy1:
            return None
        ys, xs = np.mgrid[iy0:iy1, ix0:ix1].astype(np.float32)
        return (iy0, iy1, ix0, ix1), xs + 0.5, ys + 0.5

    def _paint(self, window, mask: np.ndarray, colour: np.ndarray) -> None:
        (iy0, iy1, ix0, ix1) = window
        region = self.pixels[iy0:iy1, ix0:ix1]
        if colour.ndim == 1:
            region[mask, :3] = colour
        else:
            region[mask, :3] = colour[mask]
        region[mask, 3] = 1.0

    def stroke(self, a: tuple[float, float], b: tuple[float, float], width_a: float, width_b: float, colour: np.ndarray) -> None:
        reach = max(width_a, width_b)
        window = self._window(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach, max(a[0], b[0]) + reach, max(a[1], b[1]) + reach)
        if window is None:
            return
        bounds, xs, ys = window
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length_sq = max(dx * dx + dy * dy, 1e-6)
        t = np.clip(((xs - a[0]) * dx + (ys - a[1]) * dy) / length_sq, 0.0, 1.0)
        distance = np.hypot(xs - (a[0] + t * dx), ys - (a[1] + t * dy))
        half = 0.5 * (width_a + (width_b - width_a) * t)
        self._paint(bounds, distance <= np.maximum(half, 0.5), colour)

    def polyline(self, points: list[tuple[float, float]], width_start: float, width_end: float, colour: np.ndarray) -> None:
        count = len(points) - 1
        for index in range(count):
            w0 = width_start + (width_end - width_start) * index / count
            w1 = width_start + (width_end - width_start) * (index + 1) / count
            self.stroke(points[index], points[index + 1], w0, w1, colour)

    def leaf(
        self,
        base: tuple[float, float],
        angle: float,
        length: float,
        width: float,
        colour: np.ndarray,
        fullness: float = 0.8,
        rib: float = 0.18,
        tip: float = 0.6,
    ) -> None:
        """A pointed leaf blade from base along angle: width profile sin(pi t)^fullness, skewed
        toward the base by tip, shaded light to dark across the blade with a darker midrib."""
        ax = math.cos(angle)
        ay = math.sin(angle)
        far = (base[0] + ax * length, base[1] + ay * length)
        reach = width
        window = self._window(min(base[0], far[0]) - reach, min(base[1], far[1]) - reach, max(base[0], far[0]) + reach, max(base[1], far[1]) + reach)
        if window is None:
            return
        bounds, xs, ys = window
        rx = xs - base[0]
        ry = ys - base[1]
        along = (rx * ax + ry * ay) / length
        across = (-rx * ay + ry * ax) / (0.5 * width)
        skew = np.clip(along, 0.0, 1.0) ** tip
        profile = np.maximum(np.sin(np.pi * skew), 0.0) ** fullness
        inside = (along >= 0.0) & (along <= 1.0) & (np.abs(across) <= profile)
        if not inside.any():
            return
        side = np.where(across > 0.0, 1.08, 0.86)
        midrib = np.where(np.abs(across) < rib * np.maximum(profile, 1e-3) * 0.35, 0.82, 1.0)
        tone = (0.82 + 0.3 * along) * side * midrib
        shaded = np.clip(colour[None, None, :] * tone[..., None], 0.0, 1.0)
        self._paint(bounds, inside, shaded)

    def disc(self, centre: tuple[float, float], radius: float, colour: np.ndarray) -> None:
        window = self._window(centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius)
        if window is None:
            return
        bounds, xs, ys = window
        distance = np.hypot(xs - centre[0], ys - centre[1])
        inside = distance <= radius
        shaded = np.clip(colour[None, None, :] * (1.1 - 0.35 * distance / radius)[..., None], 0.0, 1.0)
        self._paint(bounds, inside, shaded)


def _curve(rng: random.Random, start: tuple[float, float], angle: float, length: float, bend: float, steps: int) -> list[tuple[float, float]]:
    points = [start]
    x, y = start
    heading = angle
    for _ in range(steps):
        heading += bend / steps + rng.uniform(-0.05, 0.05)
        x += math.cos(heading) * length / steps
        y += math.sin(heading) * length / steps
        points.append((x, y))
    return points


def _spruce(canvas: Canvas, rng: random.Random, needle: Colour) -> None:
    size = canvas.size
    twig = np.array(TWIG, dtype=np.float32)
    axis = _curve(rng, (size * 0.5, 0.0), math.pi / 2, size * 0.97, rng.uniform(-0.15, 0.15), 24)
    stems = [(axis, 1.0)]
    for index in range(4, len(axis) - 2, 2):
        fraction = index / len(axis)
        spread = math.sin(math.pi * min(fraction * 1.25, 1.0)) * size * 0.42 * rng.uniform(0.8, 1.1)
        for side in (-1.0, 1.0):
            angle = math.pi / 2 - side * rng.uniform(0.75, 1.05)
            stems.append((_curve(rng, axis[index], angle, spread, side * 0.25, 8), 0.6))
    for stem, weight in stems:
        canvas.polyline(stem, 7.0 * weight, 2.0, twig)
    for stem, weight in stems:
        for index in range(1, len(stem)):
            ax0, ay0 = stem[index - 1]
            ax1, ay1 = stem[index]
            heading = math.atan2(ay1 - ay0, ax1 - ax0)
            for _ in range(int(12 * weight) + 6):
                t = rng.random()
                origin = (ax0 + (ax1 - ax0) * t, ay0 + (ay1 - ay0) * t)
                side = rng.choice((-1.0, 1.0))
                angle = heading + side * rng.uniform(0.6, 1.1)
                length = rng.uniform(16.0, 30.0) * (0.75 + 0.5 * weight)
                tip = (origin[0] + math.cos(angle) * length, origin[1] + math.sin(angle) * length)
                canvas.stroke(origin, tip, 3.2, 1.2, _jitter(rng, needle, 0.22))


def _broadleaf(
    canvas: Canvas,
    rng: random.Random,
    leaf: Colour,
    leaves: int,
    leaf_length: tuple[float, float],
    aspect: float,
    radius: float,
    fullness: float = 0.8,
) -> None:
    size = canvas.size
    twig = np.array(TWIG, dtype=np.float32)
    centre = (size * 0.5, size * 0.52)
    tips = []
    for _ in range(7):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        reach = rng.uniform(0.4, 0.95) * radius * size
        tips.append((centre[0] + math.cos(angle) * reach, centre[1] + math.sin(angle) * reach * 0.9))
    for tip in tips:
        canvas.polyline([(size * 0.5, 0.0), (size * 0.5 + (tip[0] - size * 0.5) * 0.3, size * 0.25), centre, tip], 9.0, 2.5, twig)
    placed = 0
    while placed < leaves:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        reach = math.sqrt(rng.random()) * radius * size
        base = (centre[0] + math.cos(angle) * reach, centre[1] + math.sin(angle) * reach * 0.9)
        heading = angle + rng.uniform(-0.8, 0.8) if reach > 8 else rng.uniform(0.0, 2.0 * math.pi)
        length = rng.uniform(*leaf_length)
        far = (base[0] + math.cos(heading) * length, base[1] + math.sin(heading) * length)
        if not (8 <= far[0] <= size - 8 and 8 <= far[1] <= size - 8):
            continue
        canvas.leaf(base, heading, length, length * aspect, _jitter(rng, leaf, 0.2), fullness=fullness)
        placed += 1


def _sage(canvas: Canvas, rng: random.Random, leaf: Colour) -> None:
    size = canvas.size
    twig = np.array((0.36, 0.30, 0.24), dtype=np.float32)
    stems = []
    for _ in range(22):
        angle = math.pi / 2 + rng.uniform(-0.9, 0.9)
        stems.append(_curve(rng, (size * 0.5 + rng.uniform(-30, 30), 0.0), angle, size * rng.uniform(0.55, 0.9), rng.uniform(-0.6, 0.6), 10))
    for stem in stems:
        canvas.polyline(stem, 5.0, 1.5, twig)
    for stem in stems:
        for index in range(2, len(stem)):
            for _ in range(12):
                origin = stem[index]
                angle = rng.uniform(0.0, 2.0 * math.pi)
                length = rng.uniform(12.0, 22.0)
                canvas.leaf(origin, angle, length, length * 0.32, _jitter(rng, leaf, 0.15), fullness=0.6)


def _fern(canvas: Canvas, rng: random.Random, leaf: Colour) -> None:
    size = canvas.size
    rachis = _curve(rng, (size * 0.5, 0.0), math.pi / 2, size * 0.98, rng.uniform(-0.12, 0.12), 30)
    canvas.polyline(rachis, 6.0, 1.5, np.array((0.28, 0.32, 0.12), dtype=np.float32))
    for index in range(3, len(rachis) - 1, 2):
        fraction = index / len(rachis)
        reach = size * 0.44 * math.sin(math.pi * min(0.15 + fraction, 1.0)) ** 0.8
        x0, y0 = rachis[index - 1]
        x1, y1 = rachis[index]
        heading = math.atan2(y1 - y0, x1 - x0)
        for side in (-1.0, 1.0):
            angle = heading + side * 1.0
            canvas.leaf(rachis[index], angle, reach, reach * 0.16, _jitter(rng, leaf, 0.12), fullness=0.55, tip=0.9)
            for lobe in range(1, 5):
                origin = (rachis[index][0] + math.cos(angle) * reach * lobe / 5.5, rachis[index][1] + math.sin(angle) * reach * lobe / 5.5)
                lobe_length = reach * 0.22 * (1.0 - lobe / 6.0)
                for lobe_side in (-1.0, 1.0):
                    canvas.leaf(origin, angle + lobe_side * 0.9, lobe_length, lobe_length * 0.5, _jitter(rng, leaf, 0.12))


def _grass(canvas: Canvas, rng: random.Random, blade: Colour, tip: Colour, count: int) -> None:
    size = canvas.size
    for _ in range(count):
        x = size * 0.5 + rng.gauss(0.0, size * 0.17)
        angle = math.pi / 2 + (x - size * 0.5) / size * 1.2 + rng.uniform(-0.25, 0.25)
        length = size * rng.uniform(0.45, 0.97)
        path = _curve(rng, (x, 0.0), angle, length, rng.uniform(-0.7, 0.7), 10)
        base = _jitter(rng, blade, 0.18)
        top = _jitter(rng, tip, 0.18)
        width = rng.uniform(7.0, 12.0)
        for index in range(len(path) - 1):
            t = index / (len(path) - 1)
            colour = base * (1.0 - t) + top * t
            canvas.stroke(path[index], path[index + 1], width * (1.0 - t * 0.85), width * (1.0 - (t + 0.1) * 0.85), colour)


def _flower(canvas: Canvas, rng: random.Random, petal: Colour, centre: Colour, form: str) -> None:
    size = canvas.size
    stem_colour = np.array((0.22, 0.36, 0.12), dtype=np.float32)
    _grass(canvas, rng, (0.20, 0.34, 0.10), (0.32, 0.46, 0.16), 22)
    canvas.pixels[int(size * 0.62):] = 0.0
    for _ in range(5):
        x = size * 0.5 + rng.uniform(-size * 0.28, size * 0.28)
        height = size * rng.uniform(0.6, 0.9)
        stem = _curve(rng, (size * 0.5 + rng.uniform(-20, 20), 0.0), math.atan2(height, x - size * 0.5), math.hypot(x - size * 0.5, height), rng.uniform(-0.2, 0.2), 8)
        canvas.polyline(stem, 6.0, 3.5, stem_colour)
        for index in (3, 5):
            canvas.leaf(stem[index], math.pi / 2 + rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.0), size * 0.1, size * 0.035, _jitter(rng, (0.22, 0.38, 0.12), 0.1))
        head = stem[-1]
        if form == "spike":
            for step in range(16):
                y = head[1] - step * 9.0
                for side in (-1.0, 1.0):
                    canvas.disc((head[0] + side * (10.0 - step * 0.2), y), 8.0 - step * 0.18, _jitter(rng, petal, 0.12))
        elif form == "daisy":
            radius = rng.uniform(22.0, 30.0)
            for step in range(9):
                angle = step / 9.0 * 2.0 * math.pi + rng.uniform(-0.1, 0.1)
                canvas.leaf(head, angle, radius, radius * 0.55, _jitter(rng, petal, 0.08), fullness=0.6, rib=0.0)
            canvas.disc(head, radius * 0.28, np.array(centre, dtype=np.float32))
        else:
            radius = rng.uniform(26.0, 34.0)
            for step in range(5):
                angle = math.pi / 2 + (step - 2) * 0.55
                offset = (head[0] + math.cos(angle) * radius * 0.35, head[1] + math.sin(angle) * radius * 0.35 - radius * 0.2)
                canvas.disc(offset, radius * 0.62, _jitter(rng, petal, 0.1))
            canvas.disc((head[0], head[1] - radius * 0.1), radius * 0.2, np.array(centre, dtype=np.float32))


PAINTERS: dict[str, Callable[[Canvas, random.Random], None]] = {
    "spruce": lambda canvas, rng: _spruce(canvas, rng, (0.10, 0.20, 0.08)),
    "spruce_alpine": lambda canvas, rng: _spruce(canvas, rng, (0.11, 0.21, 0.17)),
    "oak": lambda canvas, rng: _broadleaf(canvas, rng, (0.22, 0.34, 0.10), 280, (34.0, 56.0), 0.55, 0.42),
    "mesquite": lambda canvas, rng: _broadleaf(canvas, rng, (0.34, 0.38, 0.17), 650, (14.0, 24.0), 0.38, 0.44, fullness=0.6),
    "hazel": lambda canvas, rng: _broadleaf(canvas, rng, (0.25, 0.39, 0.12), 210, (40.0, 62.0), 0.75, 0.40, fullness=0.9),
    "sage": lambda canvas, rng: _sage(canvas, rng, (0.46, 0.51, 0.40)),
    "fern": lambda canvas, rng: _fern(canvas, rng, (0.22, 0.40, 0.11)),
    "grass": lambda canvas, rng: _grass(canvas, rng, (0.20, 0.33, 0.10), (0.42, 0.52, 0.20), 70),
    "grass_dry": lambda canvas, rng: _grass(canvas, rng, (0.46, 0.42, 0.24), (0.72, 0.64, 0.42), 70),
    "flower_purple": lambda canvas, rng: _flower(canvas, rng, (0.44, 0.26, 0.62), (0.9, 0.8, 0.3), "spike"),
    "flower_yellow": lambda canvas, rng: _flower(canvas, rng, (0.95, 0.80, 0.18), (0.65, 0.45, 0.10), "daisy"),
    "flower_red": lambda canvas, rng: _flower(canvas, rng, (0.78, 0.12, 0.08), (0.12, 0.08, 0.06), "cup"),
}


def bleed(pixels: np.ndarray, passes: int = BLEED_PASSES) -> np.ndarray:
    """Spread opaque colour into transparent texels so mip levels and bilinear taps near the
    cutout edge average toward the foliage colour instead of black. Alpha is untouched."""
    out = pixels.copy()
    known = out[..., 3] >= ALPHA_CUTOFF
    out[~known, :3] = 0.0
    for _ in range(passes):
        if known.all():
            break
        weight = known.astype(np.float32)
        colour_sum = np.zeros_like(out[..., :3])
        weight_sum = np.zeros_like(weight)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                colour_sum += np.roll(out[..., :3] * weight[..., None], (dy, dx), axis=(0, 1))
                weight_sum += np.roll(weight, (dy, dx), axis=(0, 1))
        grow = (~known) & (weight_sum > 0.0)
        out[grow, :3] = colour_sum[grow] / weight_sum[grow, None]
        known = known | grow
    if (~known).any() and known.any():
        out[~known, :3] = out[known, :3].mean(axis=0)
    return out


def paint_cell(name: str, seed: int, size: int = CELL_SIZE) -> np.ndarray:
    """One foliage cell: painted on the inner square, inside a transparent GUTTER frame so a
    bilinear or mip tap at a card's edge never reaches the neighbouring cell."""
    inner = size - 2 * GUTTER
    canvas = Canvas(inner)
    PAINTERS[name](canvas, random.Random(f"{seed}:{name}"))
    cell = np.zeros((size, size, 4), dtype=np.float32)
    cell[GUTTER:GUTTER + inner, GUTTER:GUTTER + inner] = canvas.pixels
    return bleed(cell)


def procedural_bark(seed: int, width: int = BARK_WIDTH, height: int = ATLAS_SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    furrows = rng.random(width // 8)
    columns = np.interp(np.arange(width), np.linspace(0, width, len(furrows)), furrows)
    streak = rng.random(height // 32)
    rows = np.interp(np.arange(height), np.linspace(0, height, len(streak)), streak)
    tone = 0.6 + 0.4 * (0.7 * columns[None, :] + 0.3 * rows[:, None])
    bark = np.ones((height, width, 4), dtype=np.float32)
    bark[..., :3] = np.array((0.30, 0.22, 0.15), dtype=np.float32) * tone[..., None]
    return bark


def tile_bark(tile: np.ndarray, width: int = BARK_WIDTH, height: int = ATLAS_SIZE) -> np.ndarray:
    """Repeat a square bark tile (already width x width) down the bark strip."""
    repeats = math.ceil(height / tile.shape[0])
    strip = np.concatenate([tile] * repeats, axis=0)[:height, :width].copy()
    strip[..., 3] = 1.0
    return strip


def build(seed: int, bark: np.ndarray | None = None) -> np.ndarray:
    """The whole atlas, float32[ATLAS_SIZE, ATLAS_SIZE, 4], rows bottom-up, sRGB-encoded colour."""
    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.float32)
    atlas[:, :BARK_WIDTH] = procedural_bark(seed) if bark is None else bark
    for name in CELL_NAMES:
        index = CELL_NAMES.index(name)
        x0 = BARK_WIDTH + (index % COLUMNS) * CELL_SIZE
        y0 = (index // COLUMNS) * CELL_SIZE
        atlas[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE] = paint_cell(name, seed)
    return atlas
