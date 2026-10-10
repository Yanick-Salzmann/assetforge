from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from terrain.config import HEIGHTMAP_MAX, MapConfig, MapConfigError
from terrain.splat import Species

INSTANCE_FIELDS = ("x", "y", "z", "yaw", "scale")
INSTANCE_LAYOUT = (
    "little-endian float32 records of (x, y, z, yaw, scale): x along image columns and y along "
    "image rows in metres from the map centre, z the elevation in metres sampled bilinearly from "
    "height.png, yaw in radians about the up axis, scale a uniform multiplier"
)
INSTANCE_SUFFIX = ".bin"
KIND_CAPS: dict[str, int] = {
    "conifer": 50_000,
    "broadleaf": 40_000,
    "cactus": 30_000,
    "shrub": 50_000,
    "rock": 40_000,
    "grass": 80_000,
    "flower": 60_000,
    "debris": 40_000,
}
MAX_CELLS = 2_000_000
PACKING = 0.6

_OFFSETS = tuple((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0))


def instance_name(species: str) -> str:
    return f"scatter_{species}{INSTANCE_SUFFIX}"


@dataclass(frozen=True)
class Placement:
    """One species' instances as an [N, 5] float32 array and the spacing actually used."""

    species: Species
    instances: np.ndarray
    spacing_m: float

    @property
    def count(self) -> int:
        return int(self.instances.shape[0])


def _shifted(grid: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """grid[i + dy, j + dx] at [i, j], with fill where that neighbour falls off the grid."""
    out = np.full_like(grid, fill)
    rows, cols = grid.shape
    out[max(0, -dy) : rows - max(0, dy), max(0, -dx) : cols - max(0, dx)] = grid[
        max(0, dy) : rows - max(0, -dy), max(0, dx) : cols - max(0, -dx)
    ]
    return out


def poisson_disc(rng: np.random.Generator, world_m: float, radius_m: float) -> np.ndarray:
    """A maximal set of points at least radius_m apart over [0, world_m)^2, as [N, 2] (x, y).

    One jittered candidate per r/sqrt(2) cell, then a Luby-style parallel independent set: in
    every round each undecided candidate that outranks every conflicting undecided neighbour by
    a random priority is accepted, and its conflicts are dropped. Fully vectorised and exactly
    reproducible from the generator state.
    """
    cells = max(1, int(world_m / (radius_m / math.sqrt(2.0))))
    cell_m = world_m / cells
    jitter = rng.random((2, cells, cells))
    priority = rng.permutation(cells * cells).reshape(cells, cells)
    index = np.arange(cells, dtype=np.float64)
    xs = (index[None, :] + jitter[0]) * cell_m
    ys = (index[:, None] + jitter[1]) * cell_m
    limit = radius_m * radius_m

    neighbours = []
    for dy, dx in _OFFSETS:
        nx = _shifted(xs, dy, dx, np.inf)
        ny = _shifted(ys, dy, dx, np.inf)
        close = (nx - xs) ** 2 + (ny - ys) ** 2 < limit
        neighbours.append((dy, dx, close, _shifted(priority, dy, dx, -1)))

    undecided = np.ones((cells, cells), dtype=bool)
    accepted = np.zeros((cells, cells), dtype=bool)
    while undecided.any():
        winner = undecided.copy()
        for dy, dx, close, other in neighbours:
            rival = _shifted(undecided, dy, dx, False) & close & (other > priority)
            winner &= ~rival
        accepted |= winner
        undecided &= ~winner
        for dy, dx, close, _ in neighbours:
            undecided &= ~(_shifted(winner, dy, dx, False) & close)
    return np.stack([xs[accepted], ys[accepted]], axis=-1)


def _pixel(coordinate: np.ndarray, metres_per_pixel: float, resolution: int) -> np.ndarray:
    return np.clip(np.rint(coordinate / metres_per_pixel - 0.5), 0, resolution - 1).astype(np.int64)


def sample_mask(mask: np.ndarray, points: np.ndarray, metres_per_pixel: float) -> np.ndarray:
    """Nearest-pixel 8-bit density at each (x, y) point, as [0, 1]."""
    resolution = mask.shape[0]
    px = _pixel(points[:, 0], metres_per_pixel, resolution)
    py = _pixel(points[:, 1], metres_per_pixel, resolution)
    return mask[py, px].astype(np.float64) / 255.0


def sample_height(height_u16: np.ndarray, points: np.ndarray, cfg: MapConfig) -> np.ndarray:
    """Elevation in metres, bilinear over the exported 16-bit heightmap exactly as the viewer samples it."""
    last = height_u16.shape[0] - 1
    px = np.clip(points[:, 0] / cfg.metres_per_pixel - 0.5, 0, last)
    py = np.clip(points[:, 1] / cfg.metres_per_pixel - 0.5, 0, last)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.minimum(x0 + 1, last)
    y1 = np.minimum(y0 + 1, last)
    fx = px - x0
    fy = py - y0
    samples = height_u16.astype(np.float64)
    top = samples[y0, x0] * (1 - fx) + samples[y0, x1] * fx
    bottom = samples[y1, x0] * (1 - fx) + samples[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy) / HEIGHTMAP_MAX * cfg.height_range_m


def effective_spacing(species: Species, mean_density: float, world_m: float) -> float:
    """The species' own spacing, widened until the expected count fits its kind's cap and the grid fits MAX_CELLS."""
    cap = KIND_CAPS[species.kind]
    spacing = species.spacing_m
    expected = PACKING * mean_density * (world_m / spacing) ** 2
    if expected > cap:
        spacing *= math.sqrt(expected / cap)
    return max(spacing, world_m * math.sqrt(2.0 / MAX_CELLS))


class _Occupied:
    """Instances already placed, with their exclusion radii, hashed on a uniform grid."""

    def __init__(self) -> None:
        self.points = np.zeros((0, 2))
        self.radii = np.zeros(0)

    def add(self, points: np.ndarray, radius: float) -> None:
        if radius <= 0.0 or points.shape[0] == 0:
            return
        self.points = np.concatenate([self.points, points])
        self.radii = np.concatenate([self.radii, np.full(points.shape[0], radius)])

    def clear_of(self, points: np.ndarray, radius: float) -> np.ndarray:
        """True where a point keeps max(own radius, placed radius) from every placed instance."""
        keep = np.ones(points.shape[0], dtype=bool)
        if self.points.shape[0] == 0 or points.shape[0] == 0:
            return keep
        reach = max(radius, float(self.radii.max()))
        if reach <= 0.0:
            return keep
        placed_cells = np.floor(self.points / reach).astype(np.int64)
        cells = np.floor(points / reach).astype(np.int64)
        span = int(max(placed_cells.max(), cells.max())) + 3
        placed_keys = (placed_cells[:, 1] + 1) * span + placed_cells[:, 0] + 1
        order = np.argsort(placed_keys, kind="stable")
        sorted_keys = placed_keys[order]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                keys = (cells[:, 1] + dy + 1) * span + cells[:, 0] + dx + 1
                start = np.searchsorted(sorted_keys, keys, side="left")
                stop = np.searchsorted(sorted_keys, keys, side="right")
                depth = int((stop - start).max(initial=0))
                for step in range(depth):
                    slot = start + step
                    valid = slot < stop
                    if not valid.any():
                        break
                    other = order[np.minimum(slot, len(order) - 1)]
                    gap = np.maximum(self.radii[other], radius)
                    distance_sq = ((self.points[other] - points) ** 2).sum(axis=1)
                    keep &= ~(valid & (distance_sq < gap * gap))
        return keep


def place(
    cfg: MapConfig,
    species: Sequence[Species],
    masks: Mapping[str, np.ndarray],
    height_u16: np.ndarray,
) -> tuple[Placement, ...]:
    """Poisson-disc instances per species in declaration order, thinned by the 8-bit mask.

    Each species draws from its own generator seeded by cfg.derive_seed("scatter", name), so
    adding a species never moves another's candidates. A candidate is dropped if it lies within
    max(own exclusion_m, other exclusion_m) of an instance an earlier species already placed.
    """
    world_m = cfg.world_size_m
    occupied = _Occupied()
    placed = []
    for entry in species:
        if entry.name not in masks:
            raise MapConfigError(f"no density mask for species {entry.name!r}")
        mask = masks[entry.name]
        rng = np.random.Generator(np.random.PCG64(cfg.derive_seed("scatter", entry.name)))
        spacing = effective_spacing(entry, float(mask.mean()) / 255.0, world_m)
        points = poisson_disc(rng, world_m, spacing)
        rolls = rng.random((3, points.shape[0]))
        kept = rolls[0] < sample_mask(mask, points, cfg.metres_per_pixel)
        kept &= occupied.clear_of(points, entry.exclusion_m)
        cap = KIND_CAPS[entry.kind]
        survivors = np.flatnonzero(kept)
        if survivors.shape[0] > cap:
            kept[survivors[np.argsort(rolls[0][survivors], kind="stable")[cap:]]] = False
        points = points[kept]
        yaw = rolls[1][kept] * 2.0 * math.pi
        low, high = entry.scale
        scale = low + (high - low) * rolls[2][kept]
        elevation = sample_height(height_u16, points, cfg)
        centred = points - world_m / 2.0
        instances = np.stack(
            [centred[:, 0], centred[:, 1], elevation, yaw, scale], axis=-1
        ).astype("<f4")
        occupied.add(points, entry.exclusion_m)
        placed.append(Placement(entry, np.ascontiguousarray(instances), spacing))
    return tuple(placed)


def write(placements: Sequence[Placement], out_dir: Path) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for placement in placements:
        target = out_dir / instance_name(placement.species.name)
        target.write_bytes(placement.instances.tobytes())
        written.append(target)
    return tuple(written)


def read(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(-1, len(INSTANCE_FIELDS))
