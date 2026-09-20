from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from terrain.config import MapConfig
from terrain.preview import gradients

NORMAL_MAP_NAME = "normal.png"


def compute(height: torch.Tensor, cfg: MapConfig) -> torch.Tensor:
    """Unit tangent-space normals from height at world scale, as [H, W, 3] in [-1, 1].

    Uses the same gradient sign convention as preview.hillshade, so the two stay
    consistent with each other by construction.
    """
    metres = height.detach().to(device="cpu", dtype=torch.float32).mul(cfg.height_range_m)
    dzdx, dzdy = gradients(metres, cfg.metres_per_pixel)
    normal = torch.stack((dzdx.neg(), dzdy.neg(), torch.ones_like(dzdx)), dim=-1)
    return normal / normal.norm(dim=-1, keepdim=True)


def to_image(normal: torch.Tensor) -> np.ndarray:
    """Pack unit normals into an 8-bit RGB image, each axis mapped from [-1, 1] to [0, 255]."""
    packed = normal.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().clamp(0.0, 255.0)
    return packed.to(torch.uint8).numpy()


def write(height: torch.Tensor, cfg: MapConfig, path: str | Path | None = None) -> Path:
    """Write the normal map PNG at full resolution, non-colour data with no gamma tag."""
    target = Path(path) if path is not None else cfg.out_dir() / NORMAL_MAP_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_image(compute(height, cfg)), "RGB").save(target)
    return target
