"""Deterministic shelf packing of UV island bounding boxes into a square atlas. Stdlib only, so
assets/game_ready.py can call it inside Blender and the uv venv can test it directly."""

from __future__ import annotations

WIDTH_CANDIDATES = 48
MAX_WIDTH_FACTOR = 2.5


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
