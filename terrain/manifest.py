from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from terrain import config
from terrain.channels import WaterLevel
from terrain.config import MapConfig, MapConfigError
from terrain.splat import SplatResult

SCHEMA_VERSION = 1

MANIFEST_NAME = "terrain.json"
HEIGHTMAP_NAME = "height.png"
WATER_MASK_NAME = "water.png"

SCATTER_KINDS = ("rock", "tree", "grass", "debris")

TOP_LEVEL_KEYS = (
    "schema_version",
    "name",
    "seed",
    "resolution",
    "world_size_m",
    "height_range_m",
    "metres_per_pixel",
    "heightmap",
    "water_mask",
    "water",
    "splat",
    "scatter",
    "normal_map",
    "rule_path",
)

WATER_KEYS = ("sea_level_m", "covered_fraction", "mean_depth_m", "max_depth_m", "depth_reference_m")
SPLAT_LAYER_KEYS = ("layer", "material", "tiling_m", "index", "texture", "channel")


class ManifestError(MapConfigError):
    """Raised when a terrain.json payload does not match the schema this module writes."""


@dataclass(frozen=True)
class ScatterMask:
    """One scatter density mask, as terrain.json records it."""

    kind: str
    path: str

    def __post_init__(self) -> None:
        if self.kind not in SCATTER_KINDS:
            raise ManifestError(
                f"scatter kind {self.kind!r} must be one of {', '.join(SCATTER_KINDS)}"
            )

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path}


def _relative(path: Path) -> str:
    try:
        return path.relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build(
    cfg: MapConfig,
    water: WaterLevel,
    splat: SplatResult,
    *,
    heightmap: str = HEIGHTMAP_NAME,
    water_mask: str = WATER_MASK_NAME,
    rule_path: str | Path | None = None,
    normal_map: str | None = None,
    scatter: Sequence[ScatterMask] = (),
) -> dict[str, Any]:
    """Assemble the terrain.json payload: the engine binding contract for one map."""
    if rule_path is not None:
        rule = rule_path if isinstance(rule_path, str) else _relative(rule_path)
    elif splat.biome.path is not None:
        rule = _relative(splat.biome.path)
    else:
        rule = None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": cfg.name,
        "seed": cfg.seed,
        "resolution": cfg.resolution,
        "world_size_m": cfg.world_size_m,
        "height_range_m": cfg.height_range_m,
        "metres_per_pixel": cfg.metres_per_pixel,
        "heightmap": heightmap,
        "water_mask": water_mask,
        "water": water.as_dict(),
        "splat": splat.as_dict(),
        "scatter": [mask.as_dict() for mask in scatter],
        "normal_map": normal_map,
        "rule_path": rule,
    }
    validate(payload)
    return payload


def _require(payload: Mapping[str, Any], key: str, kind: type | tuple[type, ...], label: str) -> Any:
    if key not in payload:
        raise ManifestError(f"{label} is missing {key!r}")
    value = payload[key]
    if kind is int and isinstance(value, bool):
        raise ManifestError(f"{label} {key!r} must be {kind}, got bool")
    if not isinstance(value, kind):
        raise ManifestError(f"{label} {key!r} must be {kind}, got {type(value).__name__}")
    return value


def _require_optional_str(payload: Mapping[str, Any], key: str, label: str) -> None:
    if key not in payload:
        raise ManifestError(f"{label} is missing {key!r}")
    value = payload[key]
    if value is not None and not isinstance(value, str):
        raise ManifestError(f"{label} {key!r} must be a string or null")


def validate(payload: Mapping[str, Any]) -> None:
    """Check a terrain.json payload against the schema this module writes, structure only."""
    label = "terrain.json"
    unknown = sorted(set(payload) - set(TOP_LEVEL_KEYS))
    if unknown:
        raise ManifestError(f"{label} carries unknown keys {', '.join(unknown)}")
    version = _require(payload, "schema_version", int, label)
    if version != SCHEMA_VERSION:
        raise ManifestError(f"{label} schema_version {version} does not match writer {SCHEMA_VERSION}")
    _require(payload, "name", str, label)
    _require(payload, "seed", int, label)
    _require(payload, "resolution", int, label)
    _require(payload, "world_size_m", (int, float), label)
    _require(payload, "height_range_m", (int, float), label)
    _require(payload, "metres_per_pixel", (int, float), label)
    _require(payload, "heightmap", str, label)
    _require(payload, "water_mask", str, label)

    water = _require(payload, "water", dict, label)
    for key in WATER_KEYS:
        _require(water, key, (int, float), "water")

    splat = _require(payload, "splat", dict, label)
    _require(splat, "biome", str, "splat")
    _require(splat, "sharpness", (int, float), "splat")
    textures = _require(splat, "textures", list, "splat")
    for texture in textures:
        if not isinstance(texture, str):
            raise ManifestError("splat textures must be strings")
    layers = _require(splat, "layers", list, "splat")
    for entry in layers:
        if not isinstance(entry, dict):
            raise ManifestError("splat layer entries must be tables")
        missing = [key for key in SPLAT_LAYER_KEYS if key not in entry]
        if missing:
            raise ManifestError(f"splat layer entry is missing {', '.join(missing)}")

    scatter = _require(payload, "scatter", list, label)
    for entry in scatter:
        if not isinstance(entry, dict) or "kind" not in entry or "path" not in entry:
            raise ManifestError("scatter entries must carry kind and path")
        if entry["kind"] not in SCATTER_KINDS:
            raise ManifestError(
                f"scatter kind {entry['kind']!r} must be one of {', '.join(SCATTER_KINDS)}"
            )
        if not isinstance(entry["path"], str):
            raise ManifestError("scatter path must be a string")

    _require_optional_str(payload, "normal_map", label)
    _require_optional_str(payload, "rule_path", label)


def write(payload: Mapping[str, Any], out_dir: Path) -> Path:
    validate(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / MANIFEST_NAME
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate(payload)
    return payload


def _verify_image(out_dir: Path, name: str, resolution: int) -> None:
    target = out_dir / name
    if not target.is_file():
        raise ManifestError(f"{name} is declared in terrain.json but missing at {target}")
    with Image.open(target) as image:
        if image.size != (resolution, resolution):
            raise ManifestError(
                f"{name} is {image.size[0]}x{image.size[1]}, expected {resolution}x{resolution}"
            )


def verify(payload: Mapping[str, Any], out_dir: Path) -> None:
    """Check every path terrain.json declares exists on disk at the declared resolution."""
    validate(payload)
    resolution = int(payload["resolution"])
    _verify_image(out_dir, payload["heightmap"], resolution)
    _verify_image(out_dir, payload["water_mask"], resolution)
    for texture in payload["splat"]["textures"]:
        _verify_image(out_dir, texture, resolution)
    for entry in payload["scatter"]:
        _verify_image(out_dir, entry["path"], resolution)
    normal_map = payload.get("normal_map")
    if normal_map is not None:
        _verify_image(out_dir, normal_map, resolution)
    rule_path = payload.get("rule_path")
    if rule_path is not None and not (config.REPO_ROOT / rule_path).is_file():
        raise ManifestError(f"rule file {rule_path} does not exist")
