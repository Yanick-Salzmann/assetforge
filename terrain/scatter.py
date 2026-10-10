from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from terrain.channels import dilate
from terrain.config import MapConfigError
from terrain.splat import SplatResult, Species

SCATTER_QUANTUM = 255
MASK_PREFIX = "scatter_"


def mask_name(species: str) -> str:
    return f"{MASK_PREFIX}{species}.png"


class _Scope:
    """Channel stack plus the biome's blended splat layer weights, looked up by name."""

    def __init__(self, channels: Any, layers: Mapping[str, torch.Tensor]) -> None:
        self.channels = channels
        self.layers = layers
        self.cfg = getattr(channels, "cfg", None)

    def __contains__(self, name: str) -> bool:
        return name in self.layers or name in self.channels

    def __getitem__(self, name: str) -> torch.Tensor:
        if name in self.layers:
            return self.layers[name]
        return self.channels[name]


def _channel(channels: Any, name: str) -> torch.Tensor:
    if name not in channels:
        raise MapConfigError(f"channel {name!r} is not available in the stack")
    return channels[name]


def _water_clear(channels: Any, buffer_m: float) -> torch.Tensor:
    """1 where planting is allowed: off the water mask widened by buffer_m."""
    water = _channel(channels, "water")
    if buffer_m <= 0.0:
        return 1.0 - water
    cfg = getattr(channels, "cfg", None)
    if cfg is None:
        raise MapConfigError("a water_buffer_m needs a channel stack that carries cfg")
    radius_px = max(1, round(buffer_m / cfg.metres_per_pixel))
    return 1.0 - dilate(water, radius_px)


def layer_weights(splat: SplatResult) -> dict[str, torch.Tensor]:
    return {layer.name: splat.weights[:, :, position] for position, layer in enumerate(splat.biome)}


def density(
    species: Species,
    channels: Any,
    layers: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One species' [0, 1] density: its rule, clamped, with the water buffer cut out."""
    scope = _Scope(channels, {} if layers is None else layers)
    signal = species.density.evaluate(scope, _channel(channels, "water")).clamp(0.0, 1.0)
    return signal.mul(_water_clear(channels, species.water_buffer_m)).clamp_(0.0, 1.0)


def build(channels: Any, splat: SplatResult) -> dict[str, torch.Tensor]:
    """Every species the biome declares, keyed by species name in declaration order."""
    layers = layer_weights(splat)
    return {species.name: density(species, channels, layers) for species in splat.biome.species}


def quantise(field: torch.Tensor) -> torch.Tensor:
    """Round an [0, 1] density field to an 8-bit mask."""
    return field.clamp(0.0, 1.0).mul(float(SCATTER_QUANTUM)).round().to(torch.uint8)


def to_image(field: torch.Tensor) -> np.ndarray:
    return quantise(field).to(device="cpu").numpy()


def write(masks: Mapping[str, torch.Tensor], out_dir: Path) -> tuple[Path, ...]:
    """Write each density mask as a single-channel 8-bit PNG named scatter_<species>.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, field in masks.items():
        target = out_dir / mask_name(name)
        Image.fromarray(to_image(field), "L").save(target, optimize=True)
        written.append(target)
    return tuple(written)


def coverage(masks: Mapping[str, torch.Tensor], threshold: float = 0.5) -> tuple[dict, ...]:
    """Per mask: mean density and the fraction of pixels visibly above threshold."""
    table = []
    for name, field in masks.items():
        table.append(
            {
                "mask": name,
                "mean": float(field.mean()),
                "coverage": float((field >= threshold).float().mean()),
            }
        )
    return tuple(table)
