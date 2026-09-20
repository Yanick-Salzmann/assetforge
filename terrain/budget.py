from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageFile

from terrain.config import MapConfigError

DEFAULT_MAX_EDGE = 768
DEFAULT_MAX_BYTES = 320_000
DEFAULT_QUALITY = 82
DEFAULT_MIN_QUALITY = 40
DEFAULT_MIN_EDGE = 256
DEFAULT_DETAIL_SPAN = 0.25

PIXELS_PER_TOKEN = 750.0
JPEG_MEDIA_TYPE = "image/jpeg"
QUALITY_STEP = 12
EDGE_STEP = 0.75


@dataclass(frozen=True)
class PreviewBudget:
    """What one image returned to an agent may cost: pixels on the long edge, then bytes."""

    max_edge: int = DEFAULT_MAX_EDGE
    max_bytes: int = DEFAULT_MAX_BYTES
    quality: int = DEFAULT_QUALITY
    min_quality: int = DEFAULT_MIN_QUALITY
    min_edge: int = DEFAULT_MIN_EDGE

    def __post_init__(self) -> None:
        if self.max_edge < 16:
            raise MapConfigError(f"max_edge {self.max_edge} must be at least 16 px")
        if self.min_edge < 16 or self.min_edge > self.max_edge:
            raise MapConfigError(f"min_edge {self.min_edge} must lie within [16, {self.max_edge}]")
        if self.max_bytes < 1024:
            raise MapConfigError(f"max_bytes {self.max_bytes} must be at least 1024")
        if not 1 <= self.min_quality <= self.quality <= 95:
            raise MapConfigError(
                f"quality band [{self.min_quality}, {self.quality}] must rise within [1, 95]"
            )

    @property
    def max_tokens(self) -> int:
        return int(self.max_edge * self.max_edge / PIXELS_PER_TOKEN)

    def fit(self, image: Image.Image) -> Image.Image:
        """Downscale to the long-edge budget. An image already inside it is returned untouched."""
        longest = max(image.size)
        if longest <= self.max_edge:
            return image
        scale = self.max_edge / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.LANCZOS)

    def encode(self, image: Image.Image) -> tuple[bytes, Image.Image]:
        """Encode within the byte ceiling, spending quality first and pixels only if that is not enough."""
        current = self.fit(_opaque(image))
        while True:
            quality = self.quality
            while True:
                payload = _jpeg(current, quality)
                if len(payload) <= self.max_bytes or quality <= self.min_quality:
                    break
                quality = max(self.min_quality, quality - QUALITY_STEP)
            if len(payload) <= self.max_bytes or max(current.size) <= self.min_edge:
                return payload, current
            edge = max(self.min_edge, int(max(current.size) * EDGE_STEP))
            current = PreviewBudget(
                edge, self.max_bytes, self.quality, self.min_quality, self.min_edge
            ).fit(current)


DEFAULT_BUDGET = PreviewBudget()


@dataclass(frozen=True)
class Preview:
    """One budgeted image: the full-resolution file stays on disk, the payload is what the agent reads."""

    data: bytes
    width: int
    height: int
    source_width: int
    source_height: int
    path: Path | None = None
    region: tuple[int, int, int, int] | None = None
    media_type: str = JPEG_MEDIA_TYPE

    @property
    def estimated_tokens(self) -> int:
        return int(self.width * self.height / PIXELS_PER_TOKEN)

    @property
    def downscaled(self) -> bool:
        return (self.width, self.height) != (self.source_width, self.source_height)

    def as_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def as_dict(self) -> dict[str, Any]:
        """Everything about the image except the payload, for the JSON half of a tool result."""
        return {
            "path": None if self.path is None else str(self.path),
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "bytes": len(self.data),
            "estimated_tokens": self.estimated_tokens,
            "downscaled": self.downscaled,
            "region": None if self.region is None else list(self.region),
        }


def _opaque(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, image.width * image.height + 1024)
    image.save(buffer, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return buffer.getvalue()


def region_box(
    size: tuple[int, int],
    region: Sequence[float] | None = None,
) -> tuple[int, int, int, int] | None:
    """Turn a fractional (x0, y0, x1, y1) region into a pixel box inside an image of this size."""
    if region is None:
        return None
    if len(region) != 4:
        raise MapConfigError(f"region {tuple(region)} must hold four fractions")
    x0, y0, x1, y1 = (float(value) for value in region)
    if not 0.0 <= x0 < x1 <= 1.0 or not 0.0 <= y0 < y1 <= 1.0:
        raise MapConfigError(f"region {tuple(region)} must rise within [0, 1]")
    width, height = size
    box = (
        int(x0 * width),
        int(y0 * height),
        max(int(x0 * width) + 1, int(round(x1 * width))),
        max(int(y0 * height) + 1, int(round(y1 * height))),
    )
    return box


def detail_region(
    centre: Sequence[float] = (0.5, 0.5),
    span: float = DEFAULT_DETAIL_SPAN,
) -> tuple[float, float, float, float]:
    """A square fractional region around a point, clamped to stay on the map."""
    if not 0.0 < span <= 1.0:
        raise MapConfigError(f"span {span} must lie within (0, 1]")
    if len(centre) != 2:
        raise MapConfigError(f"centre {tuple(centre)} must hold two fractions")
    half = 0.5 * span
    x = min(max(float(centre[0]), half), 1.0 - half)
    y = min(max(float(centre[1]), half), 1.0 - half)
    return (x - half, y - half, x + half, y + half)


def deliver(
    image: Image.Image,
    path: str | Path | None = None,
    budget: PreviewBudget = DEFAULT_BUDGET,
    region: Sequence[float] | None = None,
) -> Preview:
    """Budget one already-rendered image, optionally cropping to a fractional region first."""
    full = _opaque(image)
    box = region_box(full.size, region)
    cropped = full if box is None else full.crop(box)
    payload, shown = budget.encode(cropped)
    return Preview(
        data=payload,
        width=shown.width,
        height=shown.height,
        source_width=cropped.width,
        source_height=cropped.height,
        path=None if path is None else Path(path),
        region=box,
    )


def deliver_file(
    path: str | Path,
    budget: PreviewBudget = DEFAULT_BUDGET,
    region: Sequence[float] | None = None,
) -> Preview:
    """Budget an image already written to disk: Blender renders and contact sheets come this way."""
    target = Path(path)
    if not target.is_file():
        raise MapConfigError(f"no image at {target}")
    with Image.open(target) as opened:
        return deliver(opened, target, budget, region)
