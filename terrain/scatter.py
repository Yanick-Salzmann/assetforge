from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from terrain.channels import dilate
from terrain.config import MapConfig, MapConfigError
from terrain.synth import smoothstep

SCATTER_QUANTUM = 255

ROCK_NAME = "scatter_rock.png"
TREE_NAME = "scatter_tree.png"
GRASS_NAME = "scatter_grass.png"
DEBRIS_NAME = "scatter_debris.png"

MASK_NAMES: dict[str, str] = {
    "rock": ROCK_NAME,
    "tree": TREE_NAME,
    "grass": GRASS_NAME,
    "debris": DEBRIS_NAME,
}


@dataclass(frozen=True)
class ScatterParams:
    """Thresholds for the four density masks, all in the channel stack's own [0, 1] units."""

    channel_bed_low: float = 0.55
    channel_bed_high: float = 0.75
    steep_low: float = 0.5
    steep_high: float = 0.8
    rock_wear_low: float = 0.45
    rock_wear_high: float = 0.8
    tree_moisture_low: float = 0.4
    tree_moisture_high: float = 0.75
    tree_slope_low: float = 0.15
    tree_slope_high: float = 0.4
    tree_water_buffer_m: float = 8.0
    grass_slope_low: float = 0.3
    grass_slope_high: float = 0.55
    debris_deposition_low: float = 0.4
    debris_deposition_high: float = 0.8
    debris_edge_low: float = 0.05
    debris_edge_high: float = 0.25

    def __post_init__(self) -> None:
        for low, high, label in (
            (self.channel_bed_low, self.channel_bed_high, "channel_bed"),
            (self.steep_low, self.steep_high, "steep"),
            (self.rock_wear_low, self.rock_wear_high, "rock_wear"),
            (self.tree_moisture_low, self.tree_moisture_high, "tree_moisture"),
            (self.tree_slope_low, self.tree_slope_high, "tree_slope"),
            (self.grass_slope_low, self.grass_slope_high, "grass_slope"),
            (self.debris_deposition_low, self.debris_deposition_high, "debris_deposition"),
            (self.debris_edge_low, self.debris_edge_high, "debris_edge"),
        ):
            if low >= high:
                raise MapConfigError(f"{label}_low {low} must be less than {label}_high {high}")
        if self.tree_water_buffer_m < 0.0:
            raise MapConfigError(
                f"tree_water_buffer_m {self.tree_water_buffer_m} must not be negative"
            )


def _channel(channels: Any, name: str) -> torch.Tensor:
    if name not in channels:
        raise MapConfigError(f"channel {name!r} is not available in the stack")
    return channels[name]


def _gradient_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Cheap central-difference edge strength, in the field's own units per pixel."""
    padded = torch.nn.functional.pad(field[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    dx = padded[1:-1, 2:].sub(padded[1:-1, :-2])
    dy = padded[2:, 1:-1].sub(padded[:-2, 1:-1])
    return torch.hypot(dx, dy).mul_(0.5)


def _water_buffer(channels: Any, cfg: MapConfig, buffer_m: float) -> torch.Tensor:
    """The water mask widened by buffer_m, so nothing plants at the very edge of the shore."""
    water = _channel(channels, "water")
    if buffer_m <= 0.0:
        return water
    radius_px = max(1, round(buffer_m / cfg.metres_per_pixel))
    return dilate(water, radius_px)


def _avoid(channels: Any, params: ScatterParams) -> torch.Tensor:
    """Shared "safe to scatter here" multiplier: no open water, channel bed or steep rock."""
    water = _channel(channels, "water")
    flow = _channel(channels, "flow")
    slope = _channel(channels, "slope")
    channel_bed = smoothstep(params.channel_bed_low, params.channel_bed_high, flow)
    steep = smoothstep(params.steep_low, params.steep_high, slope)
    return (1.0 - water) * (1.0 - channel_bed) * (1.0 - steep)


def _cfg_of(channels: Any) -> MapConfig:
    cfg = getattr(channels, "cfg", None)
    if cfg is None:
        raise MapConfigError("scatter needs a channel stack that carries cfg")
    return cfg


def density_rock(channels: Any, params: ScatterParams = ScatterParams()) -> torch.Tensor:
    """Rocks on wear and bare bedrock, away from open water, channel beds and sheer cliffs."""
    wear = _channel(channels, "wear")
    bedrock = _channel(channels, "bedrock")
    exposure = torch.maximum(smoothstep(params.rock_wear_low, params.rock_wear_high, wear), bedrock)
    return exposure.mul(_avoid(channels, params)).clamp_(0.0, 1.0)


def density_tree(channels: Any, params: ScatterParams = ScatterParams()) -> torch.Tensor:
    """Trees on moist, gentle ground, set back from the water's edge by a real buffer."""
    cfg = _cfg_of(channels)
    moisture = _channel(channels, "moisture")
    slope = _channel(channels, "slope")
    signal = smoothstep(params.tree_moisture_low, params.tree_moisture_high, moisture)
    signal = signal.mul(1.0 - smoothstep(params.tree_slope_low, params.tree_slope_high, slope))
    clear = 1.0 - _water_buffer(channels, cfg, params.tree_water_buffer_m)
    channel_bed = smoothstep(params.channel_bed_low, params.channel_bed_high, _channel(channels, "flow"))
    steep = smoothstep(params.steep_low, params.steep_high, slope)
    return signal.mul(clear).mul(1.0 - channel_bed).mul(1.0 - steep).clamp_(0.0, 1.0)


def density_grass(
    channels: Any,
    grass_weight: torch.Tensor,
    params: ScatterParams = ScatterParams(),
) -> torch.Tensor:
    """Grass on the biome's own grass-layer splat weight, thinned out on steeper ground.

    `grass_weight` is the caller's sum of whichever splat layers count as grass for the active
    biome (e.g. dry_grass + lush_grass) - scatter.py has no biome-specific layer names of its own.
    """
    slope = _channel(channels, "slope")
    thinning = 1.0 - smoothstep(params.grass_slope_low, params.grass_slope_high, slope)
    signal = grass_weight.clamp(0.0, 1.0).mul(thinning)
    return signal.mul(_avoid(channels, params)).clamp_(0.0, 1.0)


def density_debris(channels: Any, params: ScatterParams = ScatterParams()) -> torch.Tensor:
    """Debris in deposition basins and along flow edges, away from water, beds and cliffs."""
    deposition = _channel(channels, "deposition")
    flow = _channel(channels, "flow")
    pooled = smoothstep(params.debris_deposition_low, params.debris_deposition_high, deposition)
    edge = smoothstep(params.debris_edge_low, params.debris_edge_high, _gradient_magnitude(flow))
    signal = torch.maximum(pooled, edge)
    return signal.mul(_avoid(channels, params)).clamp_(0.0, 1.0)


def build(
    channels: Any,
    grass_weight: torch.Tensor,
    params: ScatterParams = ScatterParams(),
) -> dict[str, torch.Tensor]:
    """All four density masks, keyed the same as MASK_NAMES."""
    return {
        "rock": density_rock(channels, params),
        "tree": density_tree(channels, params),
        "grass": density_grass(channels, grass_weight, params),
        "debris": density_debris(channels, params),
    }


def quantise(density: torch.Tensor) -> torch.Tensor:
    """Round an [0, 1] density field to an 8-bit mask."""
    return density.clamp(0.0, 1.0).mul(float(SCATTER_QUANTUM)).round().to(torch.uint8)


def to_image(density: torch.Tensor) -> np.ndarray:
    return quantise(density).to(device="cpu").numpy()


def write(masks: Mapping[str, torch.Tensor], out_dir: Path) -> tuple[Path, ...]:
    """Write each density mask as a single-channel 8-bit PNG named after MASK_NAMES."""
    unknown = sorted(set(masks) - set(MASK_NAMES))
    if unknown:
        raise MapConfigError(f"unknown scatter masks {', '.join(unknown)}; expected {', '.join(MASK_NAMES)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, density in masks.items():
        target = out_dir / MASK_NAMES[name]
        Image.fromarray(to_image(density), "L").save(target, optimize=True)
        written.append(target)
    return tuple(written)


def coverage(masks: Mapping[str, torch.Tensor], threshold: float = 0.5) -> tuple[dict, ...]:
    """Per mask: mean density and the fraction of pixels visibly above threshold."""
    table = []
    for name, density in masks.items():
        table.append(
            {
                "mask": name,
                "mean": float(density.mean()),
                "coverage": float((density >= threshold).float().mean()),
            }
        )
    return tuple(table)
