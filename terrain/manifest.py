from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from terrain import atmosphere as atmosphere_mod
from terrain import config
from terrain.channels import WaterLevel
from terrain.config import MapConfig, MapConfigError
from terrain.splat import Species, SplatResult
from terrain.vegetation import Variant

SCHEMA_VERSION = 3

MANIFEST_NAME = "terrain.json"
HEIGHTMAP_NAME = "height.png"
WATER_MASK_NAME = "water.png"
WATER_SURFACE_NAME = "water_surface.png"

SCATTER_KINDS = config.SCATTER_KINDS
SCATTER_NUMBER_KEYS = ("spacing_m", "placed_spacing_m", "slope_align", "exclusion_m", "water_buffer_m")
INSTANCE_RECORD_BYTES = 24

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
    "water_surface",
    "splat",
    "scatter",
    "normal_map",
    "colour_macro",
    "atmosphere",
    "rule_path",
)

WATER_KEYS = ("sea_level_m", "covered_fraction", "mean_depth_m", "max_depth_m", "depth_reference_m")
COLOUR_MACRO_KEYS = ("path", "encoding", "near_strength", "near_distance_m", "far_distance_m")
SPLAT_LAYER_KEYS = ("layer", "material", "tiling_m", "index", "texture", "channel")


class ManifestError(MapConfigError):
    """Raised when a terrain.json payload does not match the schema this module writes."""


@dataclass(frozen=True)
class ScatterMask:
    """One species' scatter density mask and its placement parameters, as terrain.json records it."""

    species: str
    kind: str
    path: str
    density: str = ""
    spacing_m: float = 1.0
    scale: tuple[float, float] = (1.0, 1.0)
    slope_align: float = 0.0
    exclusion_m: float = 0.0
    water_buffer_m: float = 0.0
    instances: str = ""
    count: int = 0
    placed_spacing_m: float = 0.0
    variants: tuple[Variant, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SCATTER_KINDS:
            raise ManifestError(
                f"scatter kind {self.kind!r} must be one of {', '.join(SCATTER_KINDS)}"
            )

    @classmethod
    def of(
        cls,
        species: Species,
        path: str,
        instances: str = "",
        count: int = 0,
        placed_spacing_m: float = 0.0,
        variants: tuple[Variant, ...] = (),
    ) -> ScatterMask:
        return cls(
            species=species.name,
            kind=species.kind,
            path=path,
            density=species.density.source,
            spacing_m=species.spacing_m,
            scale=species.scale,
            slope_align=species.slope_align,
            exclusion_m=species.exclusion_m,
            water_buffer_m=species.water_buffer_m,
            instances=instances,
            count=count,
            placed_spacing_m=placed_spacing_m,
            variants=variants,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "kind": self.kind,
            "path": self.path,
            "density": self.density,
            "spacing_m": self.spacing_m,
            "scale": list(self.scale),
            "slope_align": self.slope_align,
            "exclusion_m": self.exclusion_m,
            "water_buffer_m": self.water_buffer_m,
            "instances": self.instances,
            "count": self.count,
            "placed_spacing_m": self.placed_spacing_m,
            "variants": [variant.as_dict() for variant in self.variants],
        }


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
    colour_macro: Mapping[str, Any] | None = None,
    water_surface: str | None = None,
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
        "water_surface": water_surface,
        "splat": splat.as_dict(),
        "scatter": [mask.as_dict() for mask in scatter],
        "normal_map": normal_map,
        "colour_macro": dict(colour_macro) if colour_macro is not None else None,
        "atmosphere": splat.biome.atmosphere.as_dict(),
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
    seen: set[str] = set()
    for entry in scatter:
        if not isinstance(entry, dict):
            raise ManifestError("scatter entries must be tables")
        species = _require(entry, "species", str, "scatter entry")
        if species in seen:
            raise ManifestError(f"scatter species {species!r} is declared twice")
        seen.add(species)
        kind = _require(entry, "kind", str, f"scatter {species!r}")
        if kind not in SCATTER_KINDS:
            raise ManifestError(
                f"scatter kind {kind!r} must be one of {', '.join(SCATTER_KINDS)}"
            )
        _require(entry, "path", str, f"scatter {species!r}")
        _require(entry, "density", str, f"scatter {species!r}")
        _require(entry, "instances", str, f"scatter {species!r}")
        count = _require(entry, "count", int, f"scatter {species!r}")
        if count < 0:
            raise ManifestError(f"scatter {species!r} count must not be negative")
        for key in SCATTER_NUMBER_KEYS:
            _require(entry, key, (int, float), f"scatter {species!r}")
        scale = _require(entry, "scale", list, f"scatter {species!r}")
        if len(scale) != 2 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in scale
        ):
            raise ManifestError(f"scatter {species!r} scale must be a [min, max] pair")
        for variant in _require(entry, "variants", list, f"scatter {species!r}"):
            if not isinstance(variant, dict):
                raise ManifestError(f"scatter {species!r} variants must be tables")
            _require(variant, "mesh", str, f"scatter {species!r} variant")
            _require(variant, "glb", str, f"scatter {species!r} variant")
            _require(variant, "tris", int, f"scatter {species!r} variant")
            unit_scale = _require(variant, "unit_scale", (int, float), f"scatter {species!r} variant")
            if unit_scale <= 0:
                raise ManifestError(f"scatter {species!r} variant unit_scale must be positive")

    _require_optional_str(payload, "normal_map", label)
    water_surface = payload.get("water_surface")
    if water_surface is not None and not isinstance(water_surface, str):
        raise ManifestError(f"{label} 'water_surface' must be a string or null")
    _require_optional_str(payload, "rule_path", label)

    colour_macro = payload.get("colour_macro")
    if colour_macro is not None:
        if not isinstance(colour_macro, dict):
            raise ManifestError(f"{label} 'colour_macro' must be a table or null")
        _require(colour_macro, "path", str, "colour_macro")
        _require(colour_macro, "encoding", str, "colour_macro")
        for key in COLOUR_MACRO_KEYS[2:]:
            _require(colour_macro, key, (int, float), "colour_macro")

    atmosphere = payload.get("atmosphere")
    if atmosphere is not None:
        try:
            atmosphere_mod.validate(atmosphere)
        except atmosphere_mod.AtmosphereError as error:
            raise ManifestError(f"{label} {error}") from None


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


def _verify_instances(out_dir: Path, name: str, count: int) -> None:
    target = out_dir / name
    if not target.is_file():
        raise ManifestError(f"{name} is declared in terrain.json but missing at {target}")
    size = target.stat().st_size
    if size != count * INSTANCE_RECORD_BYTES:
        raise ManifestError(
            f"{name} holds {size} bytes, expected {count} instances of {INSTANCE_RECORD_BYTES} bytes"
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
        _verify_instances(out_dir, entry["instances"], entry["count"])
    normal_map = payload.get("normal_map")
    if normal_map is not None:
        _verify_image(out_dir, normal_map, resolution)
    water_surface = payload.get("water_surface")
    if water_surface is not None:
        _verify_image(out_dir, water_surface, resolution)
    colour_macro = payload.get("colour_macro")
    if colour_macro is not None:
        _verify_image(out_dir, colour_macro["path"], resolution)
    rule_path = payload.get("rule_path")
    if rule_path is not None and not (config.REPO_ROOT / rule_path).is_file():
        raise ManifestError(f"rule file {rule_path} does not exist")
