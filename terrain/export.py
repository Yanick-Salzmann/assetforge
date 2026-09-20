from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from terrain import manifest as manifest_mod
from terrain import normal as normal_mod
from terrain import scatter as scatter_mod
from terrain import splat as splat_mod
from terrain.channels import ChannelStack, WaterLevel
from terrain.config import HEIGHTMAP_MAX, MapConfig, MapConfigError

WATER_MASK_MAX = 255


class ExportError(MapConfigError):
    """Raised when the terrain deliverable set cannot be written or verified."""


def write_height(height: torch.Tensor, path: Path) -> Path:
    """16-bit grayscale, non-colour: row-major, no flip. Blender's own UV convention lives in beauty.py, not here."""
    values = height.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    scaled = (values * HEIGHTMAP_MAX + 0.5).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(scaled)).save(path)
    return path


def _write_water_mask(water_mask: torch.Tensor, path: Path) -> Path:
    values = water_mask.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    scaled = (values * WATER_MASK_MAX + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(scaled), "L").save(path)
    return path


def grass_weight(result: splat_mod.SplatResult) -> torch.Tensor:
    """Sum of every splat layer whose name marks it as grass, for scatter's grass density.

    Zero everywhere when the biome declares no such layer - scatter still needs a
    same-shaped field to multiply against.
    """
    indices = [position for position, layer in enumerate(result.biome) if "grass" in layer.name]
    if not indices:
        return torch.zeros_like(result.weights[:, :, 0])
    return result.weights[:, :, indices].sum(dim=-1)


@dataclass(frozen=True)
class ExportResult:
    """Where the deliverable set landed, and the terrain.json payload written."""

    out_dir: Path
    manifest: dict[str, Any]
    manifest_path: Path
    files: tuple[Path, ...]


def write(
    cfg: MapConfig,
    stack: ChannelStack,
    water: WaterLevel,
    splat: splat_mod.SplatResult,
    *,
    out_dir: Path | None = None,
    rule_path: str | Path | None = None,
    write_normal: bool = True,
    scatter_params: scatter_mod.ScatterParams = scatter_mod.ScatterParams(),
) -> ExportResult:
    """Write the full terrain deliverable set plus terrain.json, then verify it end to end.

    Does not touch preview_*.png or anything else already in out_dir - only the files this
    module itself writes.
    """
    if "height" not in stack or "water" not in stack:
        raise ExportError("channel stack must carry filled height and water channels to export")
    target = out_dir if out_dir is not None else cfg.out_dir()
    target.mkdir(parents=True, exist_ok=True)

    written = [
        write_height(stack["height"], target / manifest_mod.HEIGHTMAP_NAME),
        _write_water_mask(stack["water"], target / manifest_mod.WATER_MASK_NAME),
    ]
    written.extend(splat_mod.write(splat, target))

    masks = scatter_mod.build(stack, grass_weight(splat), scatter_params)
    written.extend(scatter_mod.write(masks, target))
    scatter_entries = tuple(
        manifest_mod.ScatterMask(kind=kind, path=scatter_mod.MASK_NAMES[kind]) for kind in masks
    )

    normal_map_name = None
    if write_normal:
        normal_path = target / normal_mod.NORMAL_MAP_NAME
        normal_mod.write(stack["height"], cfg, normal_path)
        normal_map_name = normal_mod.NORMAL_MAP_NAME
        written.append(normal_path)

    payload = manifest_mod.build(
        cfg,
        water,
        splat,
        rule_path=rule_path,
        normal_map=normal_map_name,
        scatter=scatter_entries,
    )
    manifest_path = manifest_mod.write(payload, target)
    manifest_mod.verify(payload, target)

    return ExportResult(out_dir=target, manifest=payload, manifest_path=manifest_path, files=tuple(written))
