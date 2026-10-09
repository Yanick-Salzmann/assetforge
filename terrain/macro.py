from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from terrain.channels import ChannelStack, smooth
from terrain.config import MapConfigError

COLOUR_MACRO_NAME = "colour_macro.png"
NEUTRAL_BYTE = 128
ENCODING = "albedo_multiplier = byte / 128"
NEUTRAL_CHANNEL = 0.5

DRY_TINT = (1.0, 0.55, -0.6)
LUSH_TINT = (-0.45, 0.3, 0.05)


@dataclass(frozen=True)
class MacroParams:
    """How strongly the macro tint varies albedo, as fractions around a neutral 1.0 multiplier."""

    sigma_fraction: float = 1.0 / 192.0
    coarse_value: float = 0.2
    mid_value: float = 0.07
    wet_value: float = 0.12
    hue: float = 0.09
    near_strength: float = 0.35
    near_distance_m: float = 40.0
    far_distance_m: float = 600.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sigma_fraction <= 0.25:
            raise MapConfigError(f"sigma_fraction {self.sigma_fraction} must lie within [0, 0.25]")
        for name in ("coarse_value", "mid_value", "wet_value", "hue"):
            value = getattr(self, name)
            if not 0.0 <= value <= 0.3:
                raise MapConfigError(f"{name} {value} must lie within [0, 0.3]")
        if not 0.0 <= self.near_strength <= 1.0:
            raise MapConfigError(f"near_strength {self.near_strength} must lie within [0, 1]")
        if not 0.0 <= self.near_distance_m < self.far_distance_m:
            raise MapConfigError("near_distance_m must be non-negative and below far_distance_m")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": COLOUR_MACRO_NAME,
            "encoding": ENCODING,
            "near_strength": self.near_strength,
            "near_distance_m": self.near_distance_m,
            "far_distance_m": self.far_distance_m,
        }


def _channel(stack: ChannelStack, name: str) -> torch.Tensor:
    if name in stack:
        return stack[name].detach().to(device="cpu", dtype=torch.float32)
    return torch.full(stack.cfg.shape, NEUTRAL_CHANNEL, dtype=torch.float32)


def compute(stack: ChannelStack, params: MacroParams = MacroParams()) -> torch.Tensor:
    """Low-frequency albedo multiplier as [H, W, 3], neutral 1.0, from patchiness and climate.

    Computed on the CPU so regeneration from the seed reproduces the same bytes on any device.
    """
    sigma = params.sigma_fraction * stack.cfg.resolution
    coarse = smooth(_channel(stack, "patchiness_coarse"), sigma) - NEUTRAL_CHANNEL
    mid = smooth(_channel(stack, "patchiness_mid"), sigma) - NEUTRAL_CHANNEL
    wet = smooth(0.5 * (_channel(stack, "moisture") + _channel(stack, "wetness")), sigma) - NEUTRAL_CHANNEL
    value = 1.0 + params.coarse_value * 2.0 * coarse + params.mid_value * 2.0 * mid - params.wet_value * 2.0 * wet
    dryness = (-2.0 * wet).clamp(0.0, 1.0)
    lushness = (2.0 * wet).clamp(0.0, 1.0)
    drift = 2.0 * coarse
    tints = []
    for dry, lush in zip(DRY_TINT, LUSH_TINT):
        tints.append(value * (1.0 + params.hue * (dry * dryness + lush * lushness + 0.5 * lush * drift)))
    return torch.stack(tints, dim=-1).clamp(0.0, 255.0 / NEUTRAL_BYTE)


def to_image(multiplier: torch.Tensor) -> np.ndarray:
    packed = multiplier.mul(NEUTRAL_BYTE).round().clamp(0.0, 255.0)
    return packed.to(torch.uint8).numpy()


def write(stack: ChannelStack, path: Path, params: MacroParams = MacroParams()) -> Path:
    """Write the macro tint as 8-bit RGB, non-colour data: no lighting, no AO, only a multiplier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(to_image(compute(stack, params))), "RGB").save(path)
    return path
