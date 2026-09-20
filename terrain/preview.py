from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, deliver, deliver_file
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError

HILLSHADE_NAME = "preview_hillshade.png"
CHANNELS_NAME = "preview_channels.png"
SPLAT_NAME = "preview_splat.png"

DEFAULT_AZIMUTH_DEG = 315.0
DEFAULT_ALTITUDE_DEG = 45.0
DEFAULT_AMBIENT = 0.18
DEFAULT_CONTOUR_COUNT = 16
DEFAULT_INDEX_EVERY = 5
DEFAULT_MAX_SIZE = None
DEFAULT_TILE = 256
DEFAULT_COLUMNS = 4

MINOR_CONTOUR_TINT = 0.55
MAJOR_CONTOUR_TINT = 0.28
RELIEF_TINT = 0.35
SEA_RGB = (0.16, 0.32, 0.52)
SHEET_BACKGROUND = (18, 18, 22)
SHEET_LABEL_HEIGHT = 16
SHEET_PADDING = 4

_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)

COLORMAPS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "grey": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "viridis": (
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ),
    "magma": (
        (0.001, 0.000, 0.014),
        (0.116, 0.066, 0.276),
        (0.316, 0.072, 0.485),
        (0.517, 0.132, 0.507),
        (0.716, 0.215, 0.475),
        (0.888, 0.352, 0.391),
        (0.983, 0.554, 0.382),
        (0.996, 0.764, 0.529),
        (0.987, 0.991, 0.750),
    ),
    "terrain": (
        (0.10, 0.22, 0.42),
        (0.20, 0.44, 0.62),
        (0.34, 0.55, 0.36),
        (0.56, 0.66, 0.38),
        (0.76, 0.69, 0.45),
        (0.62, 0.51, 0.41),
        (0.80, 0.80, 0.80),
        (1.00, 1.00, 1.00),
    ),
    "flow": (
        (0.05, 0.05, 0.10),
        (0.10, 0.25, 0.45),
        (0.15, 0.50, 0.70),
        (0.45, 0.80, 0.90),
        (0.95, 0.99, 1.00),
    ),
}

SPLAT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.25, 0.25),
    (0.25, 0.65, 0.35),
    (0.25, 0.45, 0.85),
    (0.95, 0.75, 0.20),
    (0.60, 0.30, 0.75),
    (0.20, 0.75, 0.75),
    (0.90, 0.45, 0.15),
    (0.55, 0.55, 0.55),
)

CHANNEL_COLORMAPS: dict[str, str] = {
    "height": "terrain",
    "flow": "flow",
    "water": "flow",
    "water_depth": "flow",
    "wetness": "flow",
    "moisture": "flow",
    "temperature": "magma",
    "deposition": "magma",
    "wear": "magma",
}


def _cpu(field: torch.Tensor) -> torch.Tensor:
    """A private CPU float32 copy, so every renderer may work in place on it."""
    return field.detach().to(device="cpu", dtype=torch.float32, copy=True)


def colormap(name: str) -> tuple[tuple[float, float, float], ...]:
    try:
        return COLORMAPS[name]
    except KeyError:
        known = ", ".join(sorted(COLORMAPS))
        raise MapConfigError(f"unknown colormap {name!r}; expected one of {known}") from None


def apply_colormap(field: torch.Tensor, name: str = "viridis") -> np.ndarray:
    """Map a [0, 1] field onto an RGB uint8 image through a named colormap."""
    table = np.asarray(colormap(name), dtype=np.float32)
    values = _cpu(field).clamp_(0.0, 1.0).numpy()
    position = values * (len(table) - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, len(table) - 1)
    weight = (position - low)[..., None]
    rgb = table[low] * (1.0 - weight) + table[high] * weight
    return _to_uint8(rgb)


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def gradients(metres: torch.Tensor, metres_per_pixel: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference surface gradient in metres per metre, replicate padded."""
    padded = torch.nn.functional.pad(metres[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    scale = 0.5 / metres_per_pixel
    dzdx = padded[1:-1, 2:].sub(padded[1:-1, :-2]).mul_(scale)
    dzdy = padded[2:, 1:-1].sub(padded[:-2, 1:-1]).mul_(scale)
    return dzdx, dzdy


def hillshade(
    height: torch.Tensor,
    cfg: MapConfig,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    altitude_deg: float = DEFAULT_ALTITUDE_DEG,
    z_scale: float = 1.0,
    ambient: float = DEFAULT_AMBIENT,
) -> torch.Tensor:
    """Lambertian hillshade of a normalised heightfield, returned in [ambient, 1]."""
    if not 0.0 < altitude_deg <= 90.0:
        raise MapConfigError(f"altitude_deg {altitude_deg} must lie within (0, 90]")
    if not 0.0 <= ambient < 1.0:
        raise MapConfigError(f"ambient {ambient} must lie within [0, 1)")
    if z_scale <= 0.0:
        raise MapConfigError(f"z_scale {z_scale} must be positive")
    metres = _cpu(height).mul_(cfg.height_range_m * z_scale)
    dzdx, dzdy = gradients(metres, cfg.metres_per_pixel)
    slope = torch.atan(torch.hypot(dzdx, dzdy))
    aspect = torch.atan2(dzdy, dzdx.neg_())
    zenith = math.radians(90.0 - altitude_deg)
    light = math.radians(360.0 - azimuth_deg + 90.0)
    shade = torch.cos(slope).mul_(math.cos(zenith))
    shade.add_(torch.sin(slope).mul_(torch.cos(aspect.sub_(light))).mul_(math.sin(zenith)))
    return shade.clamp_(0.0, 1.0).mul_(1.0 - ambient).add_(ambient)


def nice_interval(rough_m: float) -> float:
    """Round a contour spacing up to the nearest 1, 2, 2.5 or 5 times a power of ten."""
    if rough_m <= 0.0:
        raise MapConfigError(f"rough_m {rough_m} must be positive")
    decade = 10.0 ** math.floor(math.log10(rough_m))
    for step in _NICE_STEPS:
        candidate = step * decade
        if candidate >= rough_m:
            return candidate
    return 10.0 * decade


def default_contour_interval(cfg: MapConfig, count: int = DEFAULT_CONTOUR_COUNT) -> float:
    if count < 1:
        raise MapConfigError(f"contour count {count} must be at least 1")
    return nice_interval(cfg.height_range_m / count)


def contour_masks(
    height: torch.Tensor,
    cfg: MapConfig,
    interval_m: float,
    index_every: int = DEFAULT_INDEX_EVERY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean minor and major contour line masks, one pixel wide, mutually exclusive."""
    if interval_m <= 0.0:
        raise MapConfigError(f"interval_m {interval_m} must be positive")
    if index_every < 1:
        raise MapConfigError(f"index_every {index_every} must be at least 1")
    level = _cpu(height).mul_(cfg.height_range_m).div_(interval_m).floor_()
    minor = torch.zeros_like(level, dtype=torch.bool)
    major = torch.zeros_like(level, dtype=torch.bool)
    right = level[:, 1:] != level[:, :-1]
    right_level = torch.maximum(level[:, 1:], level[:, :-1])
    down = level[1:, :] != level[:-1, :]
    down_level = torch.maximum(level[1:, :], level[:-1, :])
    minor[:, :-1] |= right
    minor[:-1, :] |= down
    major[:, :-1] |= right & (torch.remainder(right_level, index_every) == 0)
    major[:-1, :] |= down & (torch.remainder(down_level, index_every) == 0)
    return minor & ~major, major


def compose_hillshade(
    height: torch.Tensor,
    cfg: MapConfig,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    altitude_deg: float = DEFAULT_ALTITUDE_DEG,
    z_scale: float = 1.0,
    contour_interval_m: float | None = None,
    index_every: int = DEFAULT_INDEX_EVERY,
    relief_tint: float = RELIEF_TINT,
) -> np.ndarray:
    """Hillshade, hypsometric tint, contour overlay and sea mask as one RGB uint8 image."""
    if not 0.0 <= relief_tint <= 1.0:
        raise MapConfigError(f"relief_tint {relief_tint} must lie within [0, 1]")
    shade = hillshade(height, cfg, azimuth_deg, altitude_deg, z_scale).numpy()[..., None]
    rgb = np.repeat(shade, 3, axis=2)
    if relief_tint > 0.0:
        tint = apply_colormap(height, "terrain").astype(np.float32) / 255.0
        rgb = rgb * (1.0 - relief_tint) + rgb * tint * (relief_tint * 2.0)
    sea_level = cfg.sea_level_normalised
    if sea_level > 0.0:
        below = (_cpu(height) < sea_level).numpy()[..., None]
        sea = np.asarray(SEA_RGB, dtype=np.float32) * (shade * 0.5 + 0.5)
        rgb = np.where(below, sea, rgb)
    interval = contour_interval_m if contour_interval_m is not None else default_contour_interval(cfg)
    minor, major = contour_masks(height, cfg, interval, index_every)
    rgb = np.where(minor.numpy()[..., None], rgb * MINOR_CONTOUR_TINT, rgb)
    rgb = np.where(major.numpy()[..., None], rgb * MAJOR_CONTOUR_TINT, rgb)
    return _to_uint8(rgb)


def _fit(image: Image.Image, max_size: int | None) -> Image.Image:
    if max_size is None or max(image.size) <= max_size:
        return image
    if max_size < 1:
        raise MapConfigError(f"max_size {max_size} must be at least 1")
    scale = max_size / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def write_png(rgb: np.ndarray, path: str | Path, max_size: int | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _fit(Image.fromarray(rgb, "RGB"), max_size).save(target, optimize=True)
    return target


def render_hillshade(
    height: torch.Tensor,
    cfg: MapConfig,
    path: str | Path | None = None,
    max_size: int | None = DEFAULT_MAX_SIZE,
    **kwargs: Any,
) -> Path:
    """Write the hillshade-and-contour preview at full resolution."""
    target = Path(path) if path is not None else cfg.out_dir() / HILLSHADE_NAME
    return write_png(compose_hillshade(height, cfg, **kwargs), target, max_size)


def hillshade_preview(
    height: torch.Tensor,
    cfg: MapConfig,
    path: str | Path | None = None,
    budget: PreviewBudget = DEFAULT_BUDGET,
    region: Sequence[float] | None = None,
    **kwargs: Any,
) -> Preview:
    """Write the full-resolution hillshade and hand back a budgeted view of it."""
    target = Path(path) if path is not None else cfg.out_dir() / HILLSHADE_NAME
    rgb = compose_hillshade(height, cfg, **kwargs)
    written = write_png(rgb, target)
    return deliver(Image.fromarray(rgb, "RGB"), written, budget, region)


def _named_channels(
    channels: Mapping[str, torch.Tensor] | torch.Tensor,
    names: Sequence[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if isinstance(channels, torch.Tensor):
        if channels.ndim != 3:
            raise MapConfigError(f"channel stack must be [H, W, C], got {tuple(channels.shape)}")
        labels = list(names) if names is not None else list(CHANNEL_NAMES[: channels.shape[2]])
        if len(labels) != channels.shape[2]:
            raise MapConfigError(
                f"{len(labels)} channel names for a stack of {channels.shape[2]} channels"
            )
        return [(label, channels[:, :, index]) for index, label in enumerate(labels)]
    return [(str(name), field) for name, field in channels.items()]


def channel_colormap(name: str) -> str:
    return CHANNEL_COLORMAPS.get(name, "viridis")


def _tile(field: torch.Tensor, name: str, size: int, rescale: bool) -> tuple[Image.Image, str]:
    values = _cpu(field)
    low = float(values.min())
    high = float(values.max())
    shown = values
    if rescale and high > low:
        shown = (values - low) / (high - low)
    image = Image.fromarray(apply_colormap(shown, channel_colormap(name)), "RGB")
    return image.resize((size, size), Image.LANCZOS), f"{name}  {low:.3f}-{high:.3f}"


def compose_channel_sheet(
    channels: Mapping[str, torch.Tensor] | torch.Tensor,
    names: Sequence[str] | None = None,
    tile: int = DEFAULT_TILE,
    columns: int = DEFAULT_COLUMNS,
    rescale: bool = True,
) -> Image.Image:
    """False-colour contact sheet of a channel stack, one labelled tile per channel."""
    entries = _named_channels(channels, names)
    if not entries:
        raise MapConfigError("channel sheet needs at least one channel")
    if tile < 16:
        raise MapConfigError(f"tile {tile} must be at least 16 px")
    if columns < 1:
        raise MapConfigError(f"columns {columns} must be at least 1")
    columns = min(columns, len(entries))
    rows = math.ceil(len(entries) / columns)
    cell_w = tile + SHEET_PADDING
    cell_h = tile + SHEET_LABEL_HEIGHT + SHEET_PADDING
    sheet = Image.new(
        "RGB",
        (columns * cell_w + SHEET_PADDING, rows * cell_h + SHEET_PADDING),
        SHEET_BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, field) in enumerate(entries):
        image, label = _tile(field, name, tile, rescale)
        x = SHEET_PADDING + (index % columns) * cell_w
        y = SHEET_PADDING + (index // columns) * cell_h
        sheet.paste(image, (x, y))
        draw.text((x + 2, y + tile + 2), label, fill=(232, 232, 236))
    return sheet


def render_channel_sheet(
    channels: Mapping[str, torch.Tensor] | torch.Tensor,
    path: str | Path,
    names: Sequence[str] | None = None,
    tile: int = DEFAULT_TILE,
    columns: int = DEFAULT_COLUMNS,
    rescale: bool = True,
    max_size: int | None = None,
) -> Path:
    """Write the per-channel false-colour dump used to debug the stack."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet = compose_channel_sheet(channels, names, tile, columns, rescale)
    _fit(sheet, max_size).save(target, optimize=True)
    return target


def channel_sheet_preview(
    channels: Mapping[str, torch.Tensor] | torch.Tensor,
    path: str | Path,
    names: Sequence[str] | None = None,
    tile: int = DEFAULT_TILE,
    columns: int = DEFAULT_COLUMNS,
    rescale: bool = True,
    budget: PreviewBudget = DEFAULT_BUDGET,
) -> Preview:
    """Write the full contact sheet and hand back a budgeted view of it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet = compose_channel_sheet(channels, names, tile, columns, rescale)
    sheet.save(target, optimize=True)
    return deliver(sheet, target, budget)


def channel_preview(
    field: torch.Tensor,
    path: str | Path,
    name: str = "field",
    rescale: bool = True,
    budget: PreviewBudget = DEFAULT_BUDGET,
    region: Sequence[float] | None = None,
) -> Preview:
    """Write one full-resolution false-colour channel and hand back a budgeted view of it."""
    written = render_channel(field, path, name, rescale, max_size=None)
    return deliver_file(written, budget, region)


def render_channel(
    field: torch.Tensor,
    path: str | Path,
    name: str = "field",
    rescale: bool = True,
    max_size: int | None = DEFAULT_MAX_SIZE,
) -> Path:
    """Write a single channel as a false-colour PNG."""
    values = _cpu(field)
    if rescale:
        low = float(values.min())
        high = float(values.max())
        if high > low:
            values = (values - low) / (high - low)
    return write_png(apply_colormap(values, channel_colormap(name)), path, max_size)


def splat_palette_color(index: int, palette: Sequence[tuple[float, float, float]] = SPLAT_PALETTE) -> tuple[float, float, float]:
    return palette[index % len(palette)]


def compose_splat_composite(
    weights: torch.Tensor,
    palette: Sequence[tuple[float, float, float]] = SPLAT_PALETTE,
) -> np.ndarray:
    """One flat colour per layer, blended per pixel by its normalised weight.

    `weights` is the blended [H, W, L] stack (splat.blend's first return value): it sums to
    one per pixel, so the composite is directly legible as "which layer(s) won here."
    """
    if weights.ndim != 3:
        raise MapConfigError(f"splat weights must be [H, W, L], got {tuple(weights.shape)}")
    layers = weights.shape[2]
    if layers < 1:
        raise MapConfigError("splat weights need at least one layer")
    table = np.asarray([palette[index % len(palette)] for index in range(layers)], dtype=np.float32)
    values = _cpu(weights).numpy()
    rgb = values @ table
    return _to_uint8(rgb)


def render_splat_composite(
    weights: torch.Tensor,
    path: str | Path | None = None,
    cfg: MapConfig | None = None,
    max_size: int | None = DEFAULT_MAX_SIZE,
    palette: Sequence[tuple[float, float, float]] = SPLAT_PALETTE,
) -> Path:
    """Write the full-resolution false-colour splat composite."""
    if path is None:
        if cfg is None:
            raise MapConfigError("render_splat_composite needs either path or cfg")
        path = cfg.out_dir() / SPLAT_NAME
    return write_png(compose_splat_composite(weights, palette), path, max_size)


def splat_composite_preview(
    weights: torch.Tensor,
    path: str | Path | None = None,
    cfg: MapConfig | None = None,
    budget: PreviewBudget = DEFAULT_BUDGET,
    region: Sequence[float] | None = None,
    palette: Sequence[tuple[float, float, float]] = SPLAT_PALETTE,
) -> Preview:
    """Write the full-resolution composite and hand back a budgeted view of it."""
    if path is None:
        if cfg is None:
            raise MapConfigError("splat_composite_preview needs either path or cfg")
        path = cfg.out_dir() / SPLAT_NAME
    rgb = compose_splat_composite(weights, palette)
    written = write_png(rgb, path)
    return deliver(Image.fromarray(rgb, "RGB"), written, budget, region)


COVERAGE_COLUMNS = ("layer", "material", "mean %", "dominant %")


def coverage_table(coverage: Sequence[Mapping[str, Any]]) -> str:
    """Render splat.coverage()'s per-layer stats as an agent-legible fixed-width table."""
    if not coverage:
        raise MapConfigError("coverage table needs at least one layer")
    rows = [
        (
            str(entry["layer"]),
            str(entry["material"]),
            f"{float(entry['mean']) * 100.0:.2f}",
            f"{float(entry['dominant']) * 100.0:.2f}",
        )
        for entry in coverage
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(COVERAGE_COLUMNS)
    ]
    lines = ["  ".join(column.ljust(width) for column, width in zip(COVERAGE_COLUMNS, widths))]
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    return "\n".join(lines)
