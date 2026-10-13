"""Deterministic shelf packing of UV island bounding boxes into a square atlas. Stdlib only, so
assets/game_ready.py can call it inside Blender and the uv venv can test it directly."""

from __future__ import annotations

import math
from dataclasses import dataclass

WIDTH_CANDIDATES = 48
MAX_WIDTH_FACTOR = 2.5
DENSITY_SEARCH_STEPS = 48


@dataclass(frozen=True)
class PixelPacking:
    """Islands placed on an atlas_px square in whole pixels. Each island owns a footprint - its
    content rect rounded up to block_px plus gutter_px, half on every side - and footprints tile
    without overlapping, so every gutter pixel belongs to exactly one island. Content origins and
    footprints sit on the block_px grid, so no two islands share a block-compressed (BC7 4x4) block
    or a texel of the first log2(gutter_px / 2 / block_px) + log2(block_px) mip levels. density is
    the uniform pixels per source unit every island was scaled by."""

    atlas_px: int
    gutter_px: int
    block_px: int
    density: float
    content_origins: tuple[tuple[int, int], ...]
    footprints: tuple[tuple[int, int, int, int], ...]


def _ceil_to(value: float, step: int) -> int:
    return max(step, int(math.ceil(value / step - 1e-9)) * step)


def _shelf_pixels(
    footprints: list[tuple[int, int]], atlas_px: int
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(footprints)), key=lambda i: (footprints[i][1], footprints[i][0]), reverse=True)
    origins: list[tuple[int, int]] = [(0, 0)] * len(footprints)
    cursor_x = 0
    cursor_y = 0
    row_height = 0
    for i in order:
        width, height = footprints[i]
        if width > atlas_px:
            return None
        if cursor_x > 0 and cursor_x + width > atlas_px:
            cursor_x = 0
            cursor_y += row_height
            row_height = 0
        if cursor_y + height > atlas_px:
            return None
        origins[i] = (cursor_x, cursor_y)
        cursor_x += width
        row_height = max(row_height, height)
    return origins


def pack_pixels(
    sizes: list[tuple[float, float]], atlas_px: int, gutter_px: int, block_px: int = 4
) -> PixelPacking:
    """Pack (width, height) islands, in any source unit, onto an atlas_px square at the largest
    uniform density that fits, with gutter_px between any two islands' content measured in atlas
    pixels rather than source units - a gutter in source units shrinks to nothing on a large asset
    (af-alp: 0.01 m was ~1 px on a 12 m building, so bake margins and mips bled across islands)."""
    if gutter_px % (2 * block_px) != 0:
        raise ValueError(f"gutter_px {gutter_px} must be a multiple of 2 * block_px ({2 * block_px})")
    if not sizes:
        return PixelPacking(atlas_px, gutter_px, block_px, 0.0, (), ())

    def footprint_sizes(density: float) -> list[tuple[int, int]]:
        return [(_ceil_to(w * density, block_px) + gutter_px, _ceil_to(h * density, block_px) + gutter_px) for w, h in sizes]

    usable = atlas_px - gutter_px
    longest = max(max(w, h) for w, h in sizes)
    area = sum(w * h for w, h in sizes)
    high = min(usable / max(longest, 1e-12), atlas_px / math.sqrt(max(area, 1e-24)))
    best: tuple[float, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    def attempt(density: float) -> bool:
        nonlocal best
        footprints = footprint_sizes(density)
        origins = _shelf_pixels(footprints, atlas_px)
        if origins is None:
            return False
        if best is None or density > best[0]:
            best = (density, footprints, origins)
        return True

    low = 0.0
    if not attempt(high):
        for _ in range(DENSITY_SEARCH_STEPS):
            middle = (low + high) * 0.5
            if attempt(middle):
                low = middle
            else:
                high = middle
            if high - low <= high * 1e-4:
                break
    if best is None:
        footprints = footprint_sizes(0.0)
        origins = _shelf_pixels(footprints, atlas_px)
        if origins is None:
            raise ValueError(f"{len(sizes)} islands do not fit a {atlas_px}px atlas even at minimum size with a {gutter_px}px gutter")
        best = (0.0, footprints, origins)

    density, footprints, origins = best
    half = gutter_px // 2
    return PixelPacking(
        atlas_px=atlas_px,
        gutter_px=gutter_px,
        block_px=block_px,
        density=density,
        content_origins=tuple((x + half, y + half) for x, y in origins),
        footprints=tuple((x, y, x + fw, y + fh) for (x, y), (fw, fh) in zip(origins, footprints)),
    )


def shelf_pack(
    sizes: list[tuple[float, float]], canvas_width: float, margin: float
) -> tuple[list[tuple[float, float]], float, float]:
    """Place (width, height) rects left-to-right, wrapping to a new row below when a row would
    exceed canvas_width - the tallest-first order keeps rows dense. Every rect is placed strictly
    right of, or strictly below, every earlier rect in its row/column, so no two rects ever
    overlap regardless of how good the canvas_width guess turns out to be."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][1], reverse=True)
    offsets = [(0.0, 0.0)] * len(sizes)
    cursor_x = margin
    cursor_y = margin
    row_height = 0.0
    max_x = margin
    for i in order:
        width, height = sizes[i]
        if cursor_x > margin and cursor_x + width + margin > canvas_width:
            cursor_x = margin
            cursor_y += row_height + margin
            row_height = 0.0
        offsets[i] = (cursor_x, cursor_y)
        max_x = max(max_x, cursor_x + width + margin)
        cursor_x += width + margin
        row_height = max(row_height, height)
    total_height = cursor_y + row_height + margin
    return offsets, max_x, total_height


def pack_square(
    sizes: list[tuple[float, float]], margin: float
) -> tuple[list[tuple[float, float]], float]:
    """Shelf-pack sizes at the row width whose packing comes out closest to square, and return the
    offsets with the square side they fit in. A single fixed width guess leaves a band of the
    square atlas empty whenever the rows come out shorter (or longer) than they are wide - af-ggx,
    where a 387-part tractor filled only the bottom 75%. Islands keep their relative scale, so
    texel density stays uniform; only the row width is searched."""
    if not sizes:
        return [], max(margin * 4.0, 1e-6)
    total_area = sum((w + margin) * (h + margin) for w, h in sizes)
    narrowest = max(w for w, _ in sizes) + 2.0 * margin
    low = max(narrowest, total_area**0.5)
    high = max(low, total_area**0.5 * MAX_WIDTH_FACTOR)
    best_offsets: list[tuple[float, float]] = []
    best_side = float("inf")
    for step in range(WIDTH_CANDIDATES):
        canvas_width = low * (high / low) ** (step / (WIDTH_CANDIDATES - 1))
        offsets, max_x, total_height = shelf_pack(sizes, canvas_width, margin)
        side = max(max_x, total_height)
        if side < best_side:
            best_side = side
            best_offsets = offsets
    return best_offsets, max(best_side, 1e-6)
