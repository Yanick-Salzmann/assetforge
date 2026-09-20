from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from terrain.config import MapConfig, MapConfigError
from terrain.device import as_torch_device

_MASK32 = 0xFFFFFFFF
_INV_UINT32 = 1.0 / 4294967296.0
_PERLIN_SCALE = math.sqrt(2.0)
_WORLEY_JITTER = 0.5
_PERLIN_STD = 0.3051
_WARP_SIGMAS = 2.0
_WORLEY_F1_BOUND = 1.1
_WORLEY_F2_BOUND = 1.6


@dataclass(frozen=True)
class FractalParams:
    feature_size_m: float = 2048.0
    octaves: int = 6
    lacunarity: float = 2.0
    gain: float = 0.5

    def __post_init__(self) -> None:
        if self.feature_size_m <= 0.0:
            raise MapConfigError(f"feature_size_m {self.feature_size_m} must be positive")
        if self.octaves < 1:
            raise MapConfigError(f"octaves {self.octaves} must be at least 1")
        if self.lacunarity <= 1.0:
            raise MapConfigError(f"lacunarity {self.lacunarity} must be greater than 1")
        if not 0.0 < self.gain < 1.0:
            raise MapConfigError(f"gain {self.gain} must lie strictly between 0 and 1")


def _torch_device(device: Any | None) -> torch.device:
    return as_torch_device(device)


def _mix(value: torch.Tensor) -> torch.Tensor:
    value = value & _MASK32
    value = ((value ^ (value >> 16)) * 0x85EBCA6B) & _MASK32
    value = ((value ^ (value >> 13)) * 0xC2B2AE35) & _MASK32
    return (value ^ (value >> 16)) & _MASK32


def _to_i32(value: int) -> int:
    """Reinterpret an arbitrary-width int as the two's-complement bit pattern of an int32."""
    value &= _MASK32
    return value - 0x100000000 if value > 0x7FFFFFFF else value


_MIX32_C1 = _to_i32(0x85EBCA6B)
_MIX32_C2 = _to_i32(0xC2B2AE35)
_HASH2_C1 = _to_i32(0x1F1F1F1F)
_HASH2_C2 = _to_i32(0x2545F491)
_WORLEY_JITTER_Y_OFFSET = _to_i32(0x27D4EB2F)


def _lshr32(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical (unsigned) right shift on an int32 tensor holding an unsigned 32-bit pattern."""
    return (value >> shift) & (_MASK32 >> shift)


def _mix32(value: torch.Tensor) -> torch.Tensor:
    """Murmur3 finalizer on int32 tensors: same bit pattern as _mix, native 32-bit ops only.

    int32 wraps modulo 2**32 exactly like _mix's explicit `& _MASK32` masking does, so this is
    bit-for-bit identical to _mix - it is only faster because the GPU does not have to emulate
    64-bit integer arithmetic. Equivalence is covered by test_synth.py::test_hash2_int32_matches_int64.
    """
    value = _lshr32(value, 16) ^ value
    value = value * _MIX32_C1
    value = _lshr32(value, 13) ^ value
    value = value * _MIX32_C2
    value = _lshr32(value, 16) ^ value
    return value


def _hash2(ix: torch.Tensor, iy: torch.Tensor, seed: int) -> torch.Tensor:
    key = ix.to(torch.int32) * _HASH2_C1 + iy.to(torch.int32) * _HASH2_C2 + _to_i32(seed)
    return _mix32(key)


def _unit01(hashed: torch.Tensor) -> torch.Tensor:
    unsigned = hashed.to(torch.int64) & _MASK32
    return unsigned.to(torch.float64).mul_(_INV_UINT32).to(torch.float32)


def hash01(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic [0, 1] hash of an integer tensor, shape preserved."""
    return _unit01(_mix((values.to(torch.int64) & _MASK32) * 0x9E3779B1 + (seed & _MASK32)))


def _fade(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def grid(
    cfg: MapConfig,
    device: Any | None = None,
    origin_m: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-space sample coordinates in metres, shaped [H, W], pixel centred."""
    target = _torch_device(device)
    step = cfg.metres_per_pixel
    axis = torch.arange(cfg.resolution, device=target, dtype=torch.float32) * step + 0.5 * step
    y = (axis + origin_m[1]).reshape(-1, 1).expand(cfg.resolution, cfg.resolution)
    x = (axis + origin_m[0]).reshape(1, -1).expand(cfg.resolution, cfg.resolution)
    return x.contiguous(), y.contiguous()


def perlin(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    """Gradient noise on the unit lattice of x and y, returned in [-1, 1]."""
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = x - x0
    fy = y - y0
    ix = x0.to(torch.int64)
    iy = y0.to(torch.int64)
    ux = _fade(fx)
    uy = _fade(fy)
    corners = []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        angle = _unit01(_hash2(ix + dx, iy + dy, seed)) * (2.0 * math.pi)
        corners.append(torch.cos(angle) * (fx - dx) + torch.sin(angle) * (fy - dy))
    bottom = torch.lerp(corners[0], corners[1], ux)
    top = torch.lerp(corners[2], corners[3], ux)
    return torch.lerp(bottom, top, uy).mul_(_PERLIN_SCALE).clamp_(-1.0, 1.0)


def fbm(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    frequency: float,
    octaves: int = 6,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> torch.Tensor:
    """Fractional Brownian motion over perlin octaves, returned in [-1, 1]."""
    total = torch.zeros_like(x)
    amplitude = 1.0
    normaliser = 0.0
    for octave in range(octaves):
        scale = frequency * (lacunarity**octave)
        total.add_(perlin(x * scale, y * scale, seed + octave * 0x9E3779B1), alpha=amplitude)
        normaliser += amplitude
        amplitude *= gain
    return total.div_(normaliser)


def ridged(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    frequency: float,
    octaves: int = 6,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    sharpness: float = 2.0,
) -> torch.Tensor:
    """Ridged multifractal with per-octave weight feedback, returned in [0, 1]."""
    total = torch.zeros_like(x)
    amplitude = 1.0
    normaliser = 0.0
    weight = torch.ones_like(x)
    for octave in range(octaves):
        scale = frequency * (lacunarity**octave)
        signal = perlin(x * scale, y * scale, seed + octave * 0x85EBCA77).abs_().neg_().add_(1.0)
        signal.pow_(sharpness).mul_(weight)
        weight = signal.clamp(0.0, 1.0)
        total.add_(signal, alpha=amplitude)
        normaliser += amplitude
        amplitude *= gain
    return total.div_(normaliser).clamp_(0.0, 1.0)


def worley(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    frequency: float,
    feature: str = "f1",
) -> torch.Tensor:
    """Jittered-grid cellular noise in [0, 1]. Feature is f1, f2 or f2f1."""
    if feature not in ("f1", "f2", "f2f1"):
        raise MapConfigError(f"unknown worley feature {feature!r}; expected f1, f2 or f2f1")
    px = x * frequency
    py = y * frequency
    cx = torch.floor(px)
    cy = torch.floor(py)
    icx = cx.to(torch.int64)
    icy = cy.to(torch.int64)
    f1 = torch.full_like(px, 9.0)
    f2 = torch.full_like(px, 9.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            hashed = _hash2(icx + dx, icy + dy, seed)
            jitter_x = _unit01(hashed)
            jitter_y = _unit01(_mix32(hashed + _WORLEY_JITTER_Y_OFFSET))
            ox = cx + dx + 0.5 + (jitter_x - 0.5) * (2.0 * _WORLEY_JITTER)
            oy = cy + dy + 0.5 + (jitter_y - 0.5) * (2.0 * _WORLEY_JITTER)
            distance = torch.hypot(ox - px, oy - py)
            closer = distance < f1
            f2 = torch.where(closer, f1, torch.minimum(f2, distance))
            f1 = torch.where(closer, distance, f1)
    if feature == "f1":
        result = f1.div_(_WORLEY_F1_BOUND)
    elif feature == "f2":
        result = f2.div_(_WORLEY_F2_BOUND)
    else:
        result = (f2 - f1).div_(_WORLEY_F1_BOUND)
    return result.clamp_(0.0, 1.0)


def fbm_std(octaves: int, gain: float = 0.5) -> float:
    """Analytic standard deviation of fbm at these octave settings."""
    power = sum(gain ** (2 * octave) for octave in range(octaves))
    total = sum(gain**octave for octave in range(octaves))
    return _PERLIN_STD * math.sqrt(power) / total


def domain_warp(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    strength_m: float,
    frequency: float,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offset the sample coordinates by two independent fBm fields, in metres."""
    if strength_m < 0.0:
        raise MapConfigError(f"strength_m {strength_m} must not be negative")
    if strength_m == 0.0:
        return x, y
    spread = 1.0 / (_WARP_SIGMAS * fbm_std(octaves, gain))
    offset_x = fbm(x, y, seed + 0x1111, frequency, octaves, lacunarity, gain)
    offset_y = fbm(x, y, seed + 0x2222, frequency, octaves, lacunarity, gain)
    offset_x.mul_(spread).clamp_(-1.0, 1.0).mul_(strength_m)
    offset_y.mul_(spread).clamp_(-1.0, 1.0).mul_(strength_m)
    return x + offset_x, y + offset_y


def to01(field: torch.Tensor) -> torch.Tensor:
    return field.mul(0.5).add_(0.5).clamp_(0.0, 1.0)


def normalise(field: torch.Tensor) -> torch.Tensor:
    """Rescale an arbitrary field onto [0, 1] by its own extent."""
    low = field.min()
    span = field.max() - low
    if float(span) == 0.0:
        return torch.zeros_like(field)
    return (field - low).div_(span)


def warped_fbm(
    cfg: MapConfig,
    params: FractalParams = FractalParams(),
    warp_strength_m: float = 0.0,
    warp_feature_size_m: float | None = None,
    warp_octaves: int = 4,
    label: str = "fbm",
    device: Any | None = None,
) -> torch.Tensor:
    """Domain-warped fBm over the whole map, float32[H, W] in [0, 1]."""
    x, y = grid(cfg, device)
    warp_frequency = 1.0 / (warp_feature_size_m or params.feature_size_m)
    x, y = domain_warp(
        x, y, cfg.derive_seed(label, "warp"), warp_strength_m, warp_frequency, warp_octaves
    )
    field = fbm(
        x,
        y,
        cfg.derive_seed(label),
        1.0 / params.feature_size_m,
        params.octaves,
        params.lacunarity,
        params.gain,
    )
    return to01(field)


def warped_ridged(
    cfg: MapConfig,
    params: FractalParams = FractalParams(),
    warp_strength_m: float = 0.0,
    warp_feature_size_m: float | None = None,
    sharpness: float = 2.0,
    label: str = "ridged",
    device: Any | None = None,
) -> torch.Tensor:
    """Domain-warped ridged multifractal over the whole map, float32[H, W] in [0, 1]."""
    x, y = grid(cfg, device)
    warp_frequency = 1.0 / (warp_feature_size_m or params.feature_size_m)
    x, y = domain_warp(x, y, cfg.derive_seed(label, "warp"), warp_strength_m, warp_frequency)
    return ridged(
        x,
        y,
        cfg.derive_seed(label),
        1.0 / params.feature_size_m,
        params.octaves,
        params.lacunarity,
        params.gain,
        sharpness,
    )


def cellular(
    cfg: MapConfig,
    cell_size_m: float = 256.0,
    feature: str = "f1",
    warp_strength_m: float = 0.0,
    label: str = "worley",
    device: Any | None = None,
    warp_octaves: int = 4,
) -> torch.Tensor:
    """Cellular noise over the whole map, float32[H, W] in [0, 1]."""
    if cell_size_m <= 0.0:
        raise MapConfigError(f"cell_size_m {cell_size_m} must be positive")
    x, y = grid(cfg, device)
    x, y = domain_warp(
        x, y, cfg.derive_seed(label, "warp"), warp_strength_m, 1.0 / cell_size_m, warp_octaves
    )
    return worley(x, y, cfg.derive_seed(label), 1.0 / cell_size_m, feature)


SHELF_MODES = ("none", "island", "coast")

DEFAULT_SHAPE = "continental"


@dataclass(frozen=True)
class ShapeParams:
    continent_feature_size_m: float = 6000.0
    continent_warp_m: float = 1500.0
    continent_octaves: int = 3
    continent_bias: float = 0.0
    belt_cell_size_m: float = 3000.0
    belt_warp_m: float = 900.0
    belt_width: float = 0.3
    belt_sharpness: float = 2.0
    belt_weight: float = 0.45
    belt_chain_scale: float = 3.0
    belt_chain_cover: float = 0.45
    basin_feature_size_m: float = 3500.0
    basin_weight: float = 0.3
    shelf: str = "none"
    shelf_margin: float = 0.15
    shelf_power: float = 2.0
    relief_floor: float = 0.15
    ridge: FractalParams = FractalParams(feature_size_m=1200.0, octaves=7)
    ridge_weight: float = 0.45
    detail: FractalParams = FractalParams(feature_size_m=600.0, octaves=6)
    detail_weight: float = 0.18
    detail_warp_m: float = 120.0

    def __post_init__(self) -> None:
        for name in ("continent_feature_size_m", "belt_cell_size_m", "basin_feature_size_m"):
            value = getattr(self, name)
            if value <= 0.0:
                raise MapConfigError(f"{name} {value} must be positive")
        for name in ("continent_warp_m", "belt_warp_m", "detail_warp_m"):
            value = getattr(self, name)
            if value < 0.0:
                raise MapConfigError(f"{name} {value} must not be negative")
        for name in ("belt_weight", "basin_weight", "ridge_weight", "detail_weight"):
            value = getattr(self, name)
            if value < 0.0:
                raise MapConfigError(f"{name} {value} must not be negative")
        if self.continent_octaves < 1:
            raise MapConfigError(f"continent_octaves {self.continent_octaves} must be at least 1")
        if not -1.0 <= self.continent_bias <= 1.0:
            raise MapConfigError(f"continent_bias {self.continent_bias} must lie within [-1, 1]")
        if self.belt_chain_scale < 1.0:
            raise MapConfigError(f"belt_chain_scale {self.belt_chain_scale} must be at least 1")
        if not 0.0 < self.belt_chain_cover <= 1.0:
            raise MapConfigError(
                f"belt_chain_cover {self.belt_chain_cover} must lie within (0, 1]"
            )
        if not 0.0 < self.belt_width <= 1.0:
            raise MapConfigError(f"belt_width {self.belt_width} must lie within (0, 1]")
        if self.belt_sharpness <= 0.0:
            raise MapConfigError(f"belt_sharpness {self.belt_sharpness} must be positive")
        if self.shelf not in SHELF_MODES:
            raise MapConfigError(
                f"unknown shelf mode {self.shelf!r}; expected one of {', '.join(SHELF_MODES)}"
            )
        if not 0.0 < self.shelf_margin < 0.5:
            raise MapConfigError(
                f"shelf_margin {self.shelf_margin} must lie strictly between 0 and 0.5"
            )
        if self.shelf_power <= 0.0:
            raise MapConfigError(f"shelf_power {self.shelf_power} must be positive")
        if not 0.0 <= self.relief_floor <= 1.0:
            raise MapConfigError(f"relief_floor {self.relief_floor} must lie within [0, 1]")


SHAPE_PRESETS: dict[str, ShapeParams] = {
    "continental": ShapeParams(),
    "island": ShapeParams(
        continent_feature_size_m=5000.0,
        belt_weight=0.4,
        basin_weight=0.2,
        shelf="island",
        shelf_margin=0.18,
    ),
    "archipelago": ShapeParams(
        continent_feature_size_m=2600.0,
        continent_bias=-0.1,
        belt_cell_size_m=1800.0,
        belt_weight=0.32,
        belt_chain_cover=0.55,
        basin_feature_size_m=2200.0,
        basin_weight=0.45,
        shelf="island",
        shelf_margin=0.1,
        shelf_power=1.4,
    ),
    "coastal_range": ShapeParams(
        continent_feature_size_m=5200.0,
        belt_cell_size_m=2400.0,
        belt_sharpness=2.6,
        belt_weight=0.55,
        belt_chain_cover=0.55,
        basin_weight=0.25,
        shelf="coast",
        shelf_margin=0.14,
        ridge_weight=0.4,
    ),
    "highland": ShapeParams(
        continent_feature_size_m=7000.0,
        belt_weight=0.3,
        belt_chain_cover=0.6,
        basin_feature_size_m=2800.0,
        basin_weight=0.15,
        relief_floor=0.45,
        ridge=FractalParams(feature_size_m=900.0, octaves=7),
        ridge_weight=0.45,
    ),
    "basin": ShapeParams(
        continent_feature_size_m=6500.0,
        belt_cell_size_m=3600.0,
        belt_sharpness=2.4,
        belt_weight=0.55,
        belt_chain_cover=0.35,
        basin_feature_size_m=5000.0,
        basin_weight=0.55,
        relief_floor=0.1,
        ridge_weight=0.3,
        detail_weight=0.14,
    ),
}


def shape_preset(name: str) -> ShapeParams:
    """Look a named large-scale shape preset up, or raise."""
    try:
        return SHAPE_PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(SHAPE_PRESETS))
        raise MapConfigError(f"unknown shape preset {name!r}; expected one of {known}") from None


def shape_preset_names() -> tuple[str, ...]:
    return tuple(sorted(SHAPE_PRESETS))


def resolve_shape(shape: str | ShapeParams | None = None) -> ShapeParams:
    if shape is None:
        return SHAPE_PRESETS[DEFAULT_SHAPE]
    if isinstance(shape, ShapeParams):
        return shape
    return shape_preset(shape)


def smoothstep(edge0: float, edge1: float, value: torch.Tensor) -> torch.Tensor:
    """Hermite fade from 0 at edge0 to 1 at edge1, clamped outside."""
    if edge0 == edge1:
        raise MapConfigError(f"smoothstep edges {edge0} and {edge1} must differ")
    t = value.sub(edge0).div_(edge1 - edge0).clamp_(0.0, 1.0)
    return t.mul(t).mul_(t.mul(-2.0).add_(3.0))


def shelf_mask(
    cfg: MapConfig,
    shape: str | ShapeParams | None = None,
    device: Any | None = None,
) -> torch.Tensor:
    """Continental outline in [0, 1]: 1 inland, falling to 0 across the shelf."""
    shape = resolve_shape(shape)
    x, y = grid(cfg, device)
    if shape.shelf == "none":
        return torch.ones_like(x)
    x, y = domain_warp(
        x,
        y,
        cfg.derive_seed("shelf"),
        shape.continent_warp_m,
        1.0 / shape.continent_feature_size_m,
    )
    if shape.shelf == "island":
        half = 0.5 * cfg.world_size_m
        distance = torch.hypot(x.sub(half), y.sub(half)).div_(half)
        mask = smoothstep(1.0, 1.0 - 2.0 * shape.shelf_margin, distance)
    else:
        distance = x.div(cfg.world_size_m)
        mask = smoothstep(shape.shelf_margin, 3.0 * shape.shelf_margin, distance)
    return mask.pow_(shape.shelf_power)


def _orogeny(cfg: MapConfig, shape: ShapeParams, device: Any | None) -> torch.Tensor:
    """Low-frequency mask that breaks the cellular belt web into separate chains."""
    feature = shape.belt_cell_size_m * shape.belt_chain_scale
    params = FractalParams(feature, 3)
    mask = normalise(
        warped_fbm(cfg, params, shape.belt_warp_m, label="orogeny", device=device)
    )
    centre = 1.0 - shape.belt_chain_cover
    return smoothstep(max(0.0, centre - 0.15), min(1.0, centre + 0.15), mask)


def uplift(
    cfg: MapConfig,
    shape: str | ShapeParams | None = None,
    device: Any | None = None,
) -> torch.Tensor:
    """Low-frequency tectonic uplift: continents, mountain belts and basins, in [0, 1]."""
    shape = resolve_shape(shape)
    continent = normalise(
        warped_fbm(
            cfg,
            FractalParams(shape.continent_feature_size_m, shape.continent_octaves),
            shape.continent_warp_m,
            label="continent",
            device=device,
        )
    )
    if shape.continent_bias != 0.0:
        continent = continent.add_(shape.continent_bias).clamp_(0.0, 1.0)
    field = continent.clone()
    if shape.belt_weight > 0.0:
        crack = cellular(cfg, shape.belt_cell_size_m, "f2f1", shape.belt_warp_m, "belt", device)
        belts = smoothstep(shape.belt_width, 0.0, crack).pow_(shape.belt_sharpness)
        belts.mul_(smoothstep(0.1, 0.5, continent)).mul_(_orogeny(cfg, shape, device))
        field.add_(belts, alpha=shape.belt_weight)
    if shape.basin_weight > 0.0:
        basin = warped_fbm(
            cfg,
            FractalParams(shape.basin_feature_size_m, 3),
            shape.continent_warp_m,
            label="basin",
            device=device,
        )
        basin.mul_(continent.neg().add_(1.0))
        field.sub_(basin, alpha=shape.basin_weight)
    field = normalise(field)
    if shape.shelf != "none":
        field.mul_(shelf_mask(cfg, shape, device))
        field = normalise(field)
    return field


def heightfield(
    cfg: MapConfig,
    shape: str | ShapeParams | None = None,
    device: Any | None = None,
) -> torch.Tensor:
    """The base heightfield for a map: uplift plus relief-modulated ridge and fBm detail."""
    shape = resolve_shape(shape)
    base = uplift(cfg, shape, device)
    relief = base.mul(1.0 - shape.relief_floor).add_(shape.relief_floor)
    field = base.clone()
    if shape.ridge_weight > 0.0:
        ridge = warped_ridged(cfg, shape.ridge, shape.detail_warp_m, label="ridge", device=device)
        field.add_(ridge.sub_(0.5).mul_(relief), alpha=shape.ridge_weight)
    if shape.detail_weight > 0.0:
        detail = warped_fbm(cfg, shape.detail, shape.detail_warp_m, label="detail", device=device)
        field.add_(detail.sub_(0.5).mul_(relief), alpha=shape.detail_weight)
    return normalise(field)
