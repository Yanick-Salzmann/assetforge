from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from terrain import placement as placement_mod
from terrain.config import ASSET_OUT_DIR, HEIGHTMAP_MAX, MapConfig

BUILDING_KIND = re.compile(r"building")
GROUND_FLAT_REACH_M = 20.0
GROUND_MAX_SLOPE = 0.08
GROUND_SEA_MARGIN_M = 2.0
GROUND_MAX_WET = 0.2
CANDIDATE_GRID = 256
SITE_GAP_M = 8.0
SITE_EYE_CLEARANCE_M = 14.0
SITE_MAX_RELIEF_M = 2.5
SITE_MAX_WET = 0.2
SITE_SEA_MARGIN_M = 1.0
SITE_SAMPLES = 9
SITE_SCATTER_MARGIN_M = 2.0
SURVEY_CHUNK = 512
WATER_MASK_MAX = 255.0


@dataclass(frozen=True)
class Footprint:
    """A building's disc on the ground: the asset's bbox reach from its base-centre origin."""

    name: str
    radius: float
    front: float
    glb: Path | None = None


@dataclass(frozen=True)
class Site:
    """Where the viewer stands a building: x along image columns and z along image rows in metres
    from the map centre, base the lowest ground under its footprint, yaw about up facing the eye."""

    name: str
    x: float
    z: float
    base: float
    relief: float
    radius: float
    yaw: float
    glb: Path | None = None


class HeightField:
    """The viewer's HeightField: bilinear metres over the 16-bit heightmap, pixel centres at half
    a pixel from the map edge."""

    def __init__(self, cfg: MapConfig, height_u16: np.ndarray, water_u8: np.ndarray) -> None:
        self.samples = np.asarray(height_u16, dtype=np.float64)
        self.water = np.asarray(water_u8, dtype=np.float64)
        self.resolution = int(self.samples.shape[0])
        self.world_size = float(cfg.world_size_m)
        self.metres_per_pixel = float(cfg.metres_per_pixel)
        self.range = float(cfg.height_range_m)

    def at_pixel(self, px: np.ndarray, pz: np.ndarray) -> np.ndarray:
        last = self.resolution - 1
        x = np.clip(px, 0, last)
        z = np.clip(pz, 0, last)
        x0 = np.floor(x).astype(np.int64)
        z0 = np.floor(z).astype(np.int64)
        x1 = np.minimum(x0 + 1, last)
        z1 = np.minimum(z0 + 1, last)
        fx = x - x0
        fz = z - z0
        top = self.samples[z0, x0] * (1 - fx) + self.samples[z0, x1] * fx
        bottom = self.samples[z1, x0] * (1 - fx) + self.samples[z1, x1] * fx
        return (top * (1 - fz) + bottom * fz) / HEIGHTMAP_MAX * self.range

    def to_pixel(self, metres: np.ndarray) -> np.ndarray:
        return (metres + self.world_size / 2) / self.metres_per_pixel - 0.5

    def to_world(self, pixel: np.ndarray) -> np.ndarray:
        return (pixel + 0.5) * self.metres_per_pixel - self.world_size / 2

    def at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.at_pixel(self.to_pixel(x), self.to_pixel(z))

    def wet(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        last = self.water.shape[0] - 1
        px = np.clip(np.floor(self.to_pixel(x) + 0.5), 0, last).astype(np.int64)
        pz = np.clip(np.floor(self.to_pixel(z) + 0.5), 0, last).astype(np.int64)
        return self.water[pz, px] / WATER_MASK_MAX


def _scan_grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, resolution // CANDIDATE_GRID)
    pz, px = np.meshgrid(np.arange(0, resolution, step), np.arange(0, resolution, step), indexing="ij")
    return px.ravel().astype(np.float64), pz.ravel().astype(np.float64)


def ground_point(field: HeightField, sea_level_m: float) -> tuple[float, float, float]:
    """The viewer's eye spot: the dry, gently sloped scan pixel nearest the map centre."""
    resolution = field.resolution
    centre = (resolution - 1) / 2
    px, pz = _scan_grid(resolution)
    distance = np.hypot(px - centre, pz - centre)
    elevation = field.at_pixel(px, pz)
    wet = field.water[pz.astype(np.int64), px.astype(np.int64)] / WATER_MASK_MAX
    reach = max(2, math.ceil(GROUND_FLAT_REACH_M / field.metres_per_pixel))
    slope_x = np.abs(field.at_pixel(px + reach, pz) - field.at_pixel(px - reach, pz))
    slope_z = np.abs(field.at_pixel(px, pz + reach) - field.at_pixel(px, pz - reach))
    flat = np.maximum(slope_x, slope_z) / (2 * reach * field.metres_per_pixel) <= GROUND_MAX_SLOPE
    usable = (elevation >= sea_level_m + GROUND_SEA_MARGIN_M) & (wet <= GROUND_MAX_WET) & flat
    if usable.any():
        best = int(np.argmin(np.where(usable, distance, np.inf)))
        gx, gz = px[best], pz[best]
    else:
        gx = gz = centre
    return (
        float(field.to_world(gx)),
        float(field.at_pixel(np.float64(gx), np.float64(gz))),
        float(field.to_world(gz)),
    )


def _candidates(field: HeightField, ground: tuple[float, float, float]) -> np.ndarray:
    px, pz = _scan_grid(field.resolution)
    x = field.to_world(px)
    z = field.to_world(pz)
    distance = np.hypot(x - ground[0], z - ground[2])
    order = np.argsort(distance, kind="stable")
    return np.stack([distance[order], x[order], z[order]], axis=-1)


def _survey_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    steps = (2 * np.arange(SITE_SAMPLES) / (SITE_SAMPLES - 1) - 1) * radius
    dz, dx = np.meshgrid(steps, steps, indexing="ij")
    inside = np.hypot(dx, dz) <= radius
    return dx[inside], dz[inside]


def _choose(
    footprint: Footprint,
    candidates: np.ndarray,
    field: HeightField,
    sea_level_m: float,
    ground: tuple[float, float, float],
    taken: Sequence[Site],
) -> Site | None:
    half = field.world_size / 2
    distance, x, z = candidates[:, 0], candidates[:, 1], candidates[:, 2]
    keep = distance >= footprint.radius + SITE_EYE_CLEARANCE_M
    keep &= (np.abs(x) + footprint.radius <= half) & (np.abs(z) + footprint.radius <= half)
    for site in taken:
        keep &= np.hypot(site.x - x, site.z - z) >= site.radius + footprint.radius + SITE_GAP_M
    indices = np.flatnonzero(keep)
    dx, dz = _survey_offsets(footprint.radius)
    fallback: Site | None = None
    for start in range(0, indices.size, SURVEY_CHUNK):
        chunk = indices[start : start + SURVEY_CHUNK]
        sx = x[chunk, None] + dx[None, :]
        sz = z[chunk, None] + dz[None, :]
        elevation = field.at(sx, sz)
        low = elevation.min(axis=1)
        relief = elevation.max(axis=1) - low
        wet = field.wet(sx, sz).max(axis=1)
        dry = (wet <= SITE_MAX_WET) & (low >= sea_level_m + SITE_SEA_MARGIN_M)
        for offset in np.flatnonzero(dry):
            cx = float(x[chunk[offset]])
            cz = float(z[chunk[offset]])
            site = Site(
                name=footprint.name,
                x=cx,
                z=cz,
                base=float(low[offset]),
                relief=float(relief[offset]),
                radius=footprint.radius,
                yaw=math.atan2(-(ground[0] - cx), -(ground[2] - cz)),
                glb=footprint.glb,
            )
            if site.relief <= SITE_MAX_RELIEF_M:
                return site
            if fallback is None or site.relief < fallback.relief:
                fallback = site
    return fallback


def building_sites(
    footprints: Sequence[Footprint],
    field: HeightField,
    sea_level_m: float,
    ground: tuple[float, float, float] | None = None,
) -> tuple[Site | None, ...]:
    """The viewer's buildingSites: largest footprint first, each on the nearest dry, flat enough
    spot to the eye that clears every building already placed; None where nothing fits."""
    eye = ground_point(field, sea_level_m) if ground is None else ground
    candidates = _candidates(field, eye)
    taken: list[Site] = []
    for footprint in sorted(footprints, key=lambda item: -item.radius):
        site = _choose(footprint, candidates, field, sea_level_m, eye, taken)
        if site is not None:
            taken.append(site)
    by_name = {site.name: site for site in taken}
    return tuple(by_name.get(footprint.name) for footprint in footprints)


def footprint_of(name: str, manifest: dict, glb: Path | None = None) -> Footprint:
    min_x, min_y = manifest["bbox_min"][0], manifest["bbox_min"][1]
    max_x, max_y = manifest["bbox_max"][0], manifest["bbox_max"][1]
    half_width = max(-min_x, max_x)
    half_depth = max(-min_y, max_y)
    return Footprint(name=name, radius=math.hypot(half_width, half_depth), front=half_depth, glb=glb)


def building_footprints(asset_root: Path = ASSET_OUT_DIR) -> tuple[Footprint, ...]:
    """Every exported building the viewer would stand on a terrain, in its listing order."""
    if not asset_root.is_dir():
        return ()
    footprints = []
    for directory in sorted(path for path in asset_root.iterdir() if path.is_dir()):
        manifest_path = directory / "asset.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        glb = directory / manifest.get("glb", f"{directory.name}.glb")
        if not glb.is_file() or not BUILDING_KIND.search(manifest.get("kind") or ""):
            continue
        if not manifest.get("bbox_min") or not manifest.get("bbox_max"):
            continue
        footprints.append(footprint_of(directory.name, manifest, glb))
    return tuple(footprints)


def clear_scatter(
    placements: Sequence[placement_mod.Placement], sites: Sequence[Site]
) -> tuple[placement_mod.Placement, ...]:
    """Drop every instance within a building's radius plus margin, as the viewer's clearScatter."""
    cleared = []
    for placed in placements:
        keep = np.ones(placed.count, dtype=bool)
        for site in sites:
            reach = np.hypot(placed.instances[:, 0] - site.x, placed.instances[:, 1] - site.z)
            keep &= reach >= site.radius + SITE_SCATTER_MARGIN_M
        cleared.append(replace(placed, instances=placed.instances[keep]))
    return tuple(cleared)
