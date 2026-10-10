from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from terrain import macro as macro_mod
from terrain import manifest as manifest_mod
from terrain import normal as normal_mod
from terrain import placement as placement_mod
from terrain import scatter as scatter_mod
from terrain import splat as splat_mod
from terrain import vegetation as vegetation_mod
from terrain.channels import ChannelStack, WaterLevel
from terrain.config import HEIGHTMAP_MAX, MapConfig, MapConfigError

WATER_MASK_MAX = 255


class ExportError(MapConfigError):
    """Raised when the terrain deliverable set cannot be written or verified."""


def height_u16(height: torch.Tensor) -> np.ndarray:
    """The exact 16-bit samples write_height stores."""
    values = height.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    return np.ascontiguousarray((values * HEIGHTMAP_MAX + 0.5).astype(np.uint16))


def scatter_placements(
    cfg: MapConfig, stack: ChannelStack, splat: splat_mod.SplatResult
) -> tuple[dict[str, torch.Tensor], tuple[placement_mod.Placement, ...]]:
    """Every species' density mask and the instances placed from it - the exact set export writes."""
    masks = scatter_mod.build(stack, splat)
    placements = placement_mod.place(
        cfg,
        splat.biome.species,
        {name: scatter_mod.to_image(field) for name, field in masks.items()},
        height_u16(stack["height"]),
    )
    return masks, placements


def write_height(height: torch.Tensor, path: Path) -> Path:
    """16-bit grayscale, non-colour: row-major, no flip. Blender's own UV convention lives in beauty.py, not here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(height_u16(height)).save(path)
    return path


def _write_water_mask(water_mask: torch.Tensor, path: Path) -> Path:
    values = water_mask.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    scaled = (values * WATER_MASK_MAX + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(scaled), "L").save(path)
    return path


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
    write_colour_macro: bool = True,
    macro_params: macro_mod.MacroParams = macro_mod.MacroParams(),
    water_surface_m: torch.Tensor | None = None,
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

    masks, placements = scatter_placements(cfg, stack, splat)
    for pattern in ("*.png", f"*{placement_mod.INSTANCE_SUFFIX}"):
        for stale in target.glob(f"{scatter_mod.MASK_PREFIX}{pattern}"):
            stale.unlink()
    written.extend(scatter_mod.write(masks, target))
    written.extend(placement_mod.write(placements, target))
    variants = vegetation_mod.resolve_all(splat.biome.species)
    scatter_entries = tuple(
        manifest_mod.ScatterMask.of(
            placed.species,
            scatter_mod.mask_name(placed.species.name),
            placement_mod.instance_name(placed.species.name),
            placed.count,
            placed.spacing_m,
            variants[placed.species.name],
        )
        for placed in placements
    )

    normal_map_name = None
    if write_normal:
        normal_path = target / normal_mod.NORMAL_MAP_NAME
        normal_mod.write(stack["height"], cfg, normal_path)
        normal_map_name = normal_mod.NORMAL_MAP_NAME
        written.append(normal_path)

    water_surface_name = None
    if water_surface_m is not None:
        surface_path = target / manifest_mod.WATER_SURFACE_NAME
        write_height(water_surface_m.div(cfg.height_range_m), surface_path)
        water_surface_name = manifest_mod.WATER_SURFACE_NAME
        written.append(surface_path)

    colour_macro = None
    if write_colour_macro:
        written.append(macro_mod.write(stack, target / macro_mod.COLOUR_MACRO_NAME, macro_params))
        colour_macro = macro_params.as_dict()

    payload = manifest_mod.build(
        cfg,
        water,
        splat,
        rule_path=rule_path,
        normal_map=normal_map_name,
        scatter=scatter_entries,
        colour_macro=colour_macro,
        water_surface=water_surface_name,
    )
    manifest_path = manifest_mod.write(payload, target)
    manifest_mod.verify(payload, target)

    return ExportResult(out_dir=target, manifest=payload, manifest_path=manifest_path, files=tuple(written))
