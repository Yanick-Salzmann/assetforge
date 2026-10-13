"""Atlas coverage rasterisation and island-aware gutter fill. numpy and the stdlib only, so the bake
(assets/atlas_bake.py, inside Blender) and the validator (assets/validate.py, in the uv venv) share
one implementation of which pixel belongs to which island.

Pixel convention: a triangle's (x, y) are in pixels along the image array's columns and rows, and a
pixel (row, column) is covered when its centre (column + 0.5, row + 0.5) lies strictly inside a
triangle - the same rule Cycles uses to decide which texels a bake writes. Strictly inside means a
pixel on a shared edge belongs to neither triangle, so coverage is a subset of what the bake wrote
and every filled pixel copies a texel the bake really produced.
"""

from __future__ import annotations

import numpy as np

EDGE_EPSILON = 1e-6
MIN_TRIANGLE_AREA = 1e-12
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def rasterize(triangles: np.ndarray, owners: np.ndarray, height: int, width: int) -> np.ndarray:
    """(height, width) int32 map of the owner of the triangle covering each pixel centre, -1 where
    none does. triangles is (T, 3, 2) in pixels, owners (T,) non-negative ints."""
    labels = np.full((height, width), -1, dtype=np.int32)
    for triangle, owner in zip(np.asarray(triangles, dtype=np.float64), owners):
        xs = triangle[:, 0]
        ys = triangle[:, 1]
        x0 = max(int(np.floor(xs.min())), 0)
        x1 = min(int(np.ceil(xs.max())), width)
        y0 = max(int(np.floor(ys.min())), 0)
        y1 = min(int(np.ceil(ys.max())), height)
        if x1 <= x0 or y1 <= y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = triangle
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < MIN_TRIANGLE_AREA:
            continue
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        sign = 1.0 if area > 0.0 else -1.0
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * sign
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * sign
        w2 = ((ax - px) * (by - py) - (ay - py) * (bx - px)) * sign
        inside = (w0 > EDGE_EPSILON) & (w1 > EDGE_EPSILON) & (w2 > EDGE_EPSILON)
        labels[y0:y1, x0:x1][inside] = owner
    return labels


def shifted(array: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    """array moved by (dy, dx): result[y, x] = array[y - dy, x - dx], fill where that is outside."""
    height, width = array.shape
    result = np.full_like(array, fill)
    result[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = array[
        max(-dy, 0):height + min(-dy, 0), max(-dx, 0):width + min(-dx, 0)
    ]
    return result


def grow_sources(labels: np.ndarray, steps: int | None = None, track_meetings: bool = True) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    """Grow every labelled region outward one pixel ring per step (8-connected), each new pixel
    taking the label and source pixel of the first labelled neighbour in NEIGHBOURS order. Returns
    (grown labels, flat index of each pixel's covered source pixel or -1, label pairs whose grown
    regions touched). Stops after steps rings, or when nothing is left to grow."""
    height, width = labels.shape
    grown = labels.copy()
    source = np.where(labels >= 0, np.arange(height * width, dtype=np.int64).reshape(height, width), -1)
    touching: set[tuple[int, int]] = set()

    def record_meetings(current: np.ndarray) -> None:
        for dy, dx in NEIGHBOURS:
            neighbour = shifted(current, dy, dx, -1)
            meeting = (current >= 0) & (neighbour >= 0) & (current != neighbour)
            if meeting.any():
                low = np.minimum(current[meeting], neighbour[meeting])
                high = np.maximum(current[meeting], neighbour[meeting])
                touching.update(zip(low.tolist(), high.tolist()))

    if track_meetings:
        record_meetings(grown)
    step = 0
    while steps is None or step < steps:
        step += 1
        next_grown = grown.copy()
        next_source = source.copy()
        changed = False
        for dy, dx in NEIGHBOURS:
            neighbour = shifted(grown, dy, dx, -1)
            neighbour_source = shifted(source, dy, dx, -1)
            take = (next_grown < 0) & (neighbour >= 0)
            if take.any():
                next_grown[take] = neighbour[take]
                next_source[take] = neighbour_source[take]
                changed = True
        grown = next_grown
        source = next_source
        if not changed:
            break
        if track_meetings:
            record_meetings(grown)
    return grown, source, touching


def fill_map(labels: np.ndarray, footprints) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where dilate copies from, computed once per layout and reused for every image baked onto
    it: (target flat indices, source flat indices, flat mask of pixels left for the mean colour).
    Each uncovered pixel of a footprint maps to the nearest covered pixel of the island owning it
    (labels value == footprint index)."""
    height, width = labels.shape
    covered = labels >= 0
    filled = covered.copy()
    targets: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for index, (x0, y0, x1, y1) in enumerate(footprints):
        own = labels[y0:y1, x0:x1] == index
        if not own.any():
            continue
        region = np.where(own, 0, -1).astype(np.int32)
        _, source, _ = grow_sources(region, track_meetings=False)
        take = (~own) & (source >= 0)
        if not take.any():
            continue
        block_width = x1 - x0
        target_rows, target_columns = np.nonzero(take)
        source_rows, source_columns = np.divmod(source[take], block_width)
        targets.append((target_rows + y0) * width + target_columns + x0)
        sources.append((source_rows + y0) * width + source_columns + x0)
        filled[y0:y1, x0:x1] |= take
    empty = np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(targets) if targets else empty,
        np.concatenate(sources) if sources else empty,
        ~filled.ravel(),
    )


def dilate(pixels: np.ndarray, labels: np.ndarray, footprints, fill: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
    """Fill every footprint pixel no triangle covers, in place, with the colour of the nearest
    covered pixel of the island that owns the footprint, so a gutter only ever carries its own
    island's colours however far a mip level reaches into it. Footprints with no covered pixel
    (collapsed hidden faces, sub-pixel islands) and pixels outside every footprint take the mean
    covered colour. Pass fill (from fill_map) to reuse one map across images of the same layout."""
    targets, sources, unfilled = fill if fill is not None else fill_map(labels, footprints)
    flat = pixels.reshape(-1, pixels.shape[-1])
    flat[targets] = flat[sources]
    covered = (labels >= 0).ravel()
    if covered.any():
        flat[unfilled] = flat[covered].mean(axis=0)
    pixels[...] = flat.reshape(pixels.shape)
