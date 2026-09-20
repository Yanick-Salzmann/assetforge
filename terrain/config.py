from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from terrain.device import resolve as resolve_device

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR

BIOMES_DIR = PACKAGE_DIR / "biomes"
PACKAGE_LIBRARY_DIR = PACKAGE_DIR / "library"

WORKSPACE_ENV_VAR = "ASSETFORGE_WORKSPACE"


def resolve_workspace(override: str | None = None) -> Path:
    """Resolve the user workspace: an override, else ASSETFORGE_WORKSPACE, else cwd.

    Packaged defaults (biome presets, library pin/lock files, blender_scripts/) stay
    relative to PACKAGE_DIR regardless of the workspace - only user-owned data (out/,
    downloaded library binaries, biome overrides) lives under it.
    """
    requested = override if override is not None else os.environ.get(WORKSPACE_ENV_VAR)
    if requested is not None and not requested.strip():
        requested = None
    return Path(requested).resolve() if requested is not None else Path.cwd()


WORKSPACE_DIR = resolve_workspace()

LIBRARY_DIR = WORKSPACE_DIR / "library"
MATERIALS_DIR = LIBRARY_DIR / "materials"
KITS_DIR = LIBRARY_DIR / "kits"
OUT_DIR = WORKSPACE_DIR / "out"
TERRAIN_OUT_DIR = OUT_DIR / "terrain"
ASSET_OUT_DIR = OUT_DIR / "assets"

WORKSPACE_BIOMES_DIR = WORKSPACE_DIR / "biomes"


def resolve_biome_path(name: str) -> Path:
    """A biome rule file by name: a workspace override first, else the packaged preset."""
    override = WORKSPACE_BIOMES_DIR / f"{name}.toml"
    if override.is_file():
        return override
    return BIOMES_DIR / f"{name}.toml"


def available_biome_names() -> tuple[str, ...]:
    names: set[str] = set()
    if BIOMES_DIR.is_dir():
        names.update(path.stem for path in BIOMES_DIR.glob("*.toml"))
    if WORKSPACE_BIOMES_DIR.is_dir():
        names.update(path.stem for path in WORKSPACE_BIOMES_DIR.glob("*.toml"))
    return tuple(sorted(names))

HEIGHTMAP_BIT_DEPTH = 16
HEIGHTMAP_MAX = (1 << HEIGHTMAP_BIT_DEPTH) - 1
SPLAT_LAYERS_PER_TEXTURE = 4
SPLAT_TEXTURES = 2
MAX_SPLAT_LAYERS = SPLAT_LAYERS_PER_TEXTURE * SPLAT_TEXTURES

MAX_RESOLUTION = 4096

CHANNEL_NAMES = (
    "height",
    "slope",
    "curvature",
    "flow",
    "deposition",
    "wear",
    "wetness",
    "water",
    "water_depth",
    "bedrock",
    "strata",
    "temperature",
    "moisture",
    "patchiness_fine",
    "patchiness_mid",
    "patchiness_coarse",
)


class MapConfigError(ValueError):
    """Raised when a map config carries values the pipeline cannot honour."""


@dataclass(frozen=True)
class MapConfig:
    name: str
    resolution: int = 2048
    world_size_m: float = 4096.0
    height_range_m: float = 600.0
    sea_level_m: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise MapConfigError("map name must not be empty")
        if self.resolution < 2:
            raise MapConfigError(f"resolution {self.resolution} is below the 2 px minimum")
        if self.resolution & (self.resolution - 1):
            raise MapConfigError(f"resolution {self.resolution} is not a power of two")
        if self.resolution > MAX_RESOLUTION:
            raise MapConfigError(
                f"resolution {self.resolution} exceeds the {MAX_RESOLUTION} px VRAM ceiling"
            )
        if self.world_size_m <= 0.0:
            raise MapConfigError(f"world_size_m {self.world_size_m} must be positive")
        if self.height_range_m <= 0.0:
            raise MapConfigError(f"height_range_m {self.height_range_m} must be positive")
        if self.seed < 0:
            raise MapConfigError(f"seed {self.seed} must not be negative")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.resolution, self.resolution)

    @property
    def metres_per_pixel(self) -> float:
        return self.world_size_m / self.resolution

    @property
    def pixels_per_metre(self) -> float:
        return self.resolution / self.world_size_m

    @property
    def sea_level_normalised(self) -> float:
        return self.sea_level_m / self.height_range_m

    def height_to_metres(self, normalised: float) -> float:
        return normalised * self.height_range_m

    def metres_to_height(self, metres: float) -> float:
        return metres / self.height_range_m

    def slope_scale(self) -> float:
        """Multiplier turning a per-pixel normalised height delta into a gradient in m/m."""
        return self.height_range_m / self.metres_per_pixel

    def slope_from_normalised(self, delta_per_pixel: float) -> float:
        return delta_per_pixel * self.slope_scale()

    def talus_delta(self, angle_deg: float) -> float:
        """Largest normalised height step a pixel pair may hold at this repose angle."""
        if not 0.0 < angle_deg < 90.0:
            raise MapConfigError(f"talus angle {angle_deg} must lie strictly between 0 and 90 degrees")
        return math.tan(math.radians(angle_deg)) / self.slope_scale()

    def talus_angle_deg(self, delta_per_pixel: float) -> float:
        return math.degrees(math.atan(self.slope_from_normalised(delta_per_pixel)))

    def tiling_scale(self, texture_size_m: float) -> float:
        """UV repeats across the whole map for a material that tiles every texture_size_m."""
        if texture_size_m <= 0.0:
            raise MapConfigError(f"texture_size_m {texture_size_m} must be positive")
        return self.world_size_m / texture_size_m

    def texture_size_m(self, tiling_scale: float) -> float:
        if tiling_scale <= 0.0:
            raise MapConfigError(f"tiling_scale {tiling_scale} must be positive")
        return self.world_size_m / tiling_scale

    def channel_stack_bytes(self, channels: int = len(CHANNEL_NAMES)) -> int:
        return self.resolution * self.resolution * channels * 4

    def at_resolution(self, resolution: int) -> MapConfig:
        return replace(self, resolution=resolution)

    def derive_seed(self, *parts: str | int) -> int:
        digest = hashlib.blake2b(str(self.seed).encode("utf-8"), digest_size=8)
        for part in parts:
            digest.update(b"::")
            digest.update(str(part).encode("utf-8"))
        return int.from_bytes(digest.digest()[:4], "little")

    def out_dir(self) -> Path:
        return TERRAIN_OUT_DIR / self.name

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "resolution": self.resolution,
            "world_size_m": self.world_size_m,
            "height_range_m": self.height_range_m,
            "metres_per_pixel": self.metres_per_pixel,
            "sea_level_m": self.sea_level_m,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MapConfig:
        return cls(
            name=str(data["name"]),
            resolution=int(data["resolution"]),
            world_size_m=float(data["world_size_m"]),
            height_range_m=float(data["height_range_m"]),
            sea_level_m=float(data.get("sea_level_m", 0.0)),
            seed=int(data.get("seed", 0)),
        )


TerrainSpec = MapConfig


def device() -> str:
    return resolve_device().spec
