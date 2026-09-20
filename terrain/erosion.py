from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from terrain.config import MapConfig, MapConfigError
from terrain.device import as_torch_device

DIRECTIONS = ("left", "right", "top", "bottom")

_OPPOSITE = (1, 0, 3, 2)
_EPS = 1e-8
_QUANTILE_SAMPLE_LIMIT = 1 << 20
_QUANTILE_HIGH = 0.995
_WATER_SURFACE_M = 0.5


@dataclass(frozen=True)
class ErosionParams:
    """Pipe-model settings. Heights are normalised map units; only slope is read in metres."""

    iterations: int = 400
    settle_iterations: int = 40
    dt: float = 0.05
    rain_rate: float = 0.002
    evaporation_rate: float = 2.0
    pipe_area: float = 1.0
    gravity: float = 9.81
    sediment_capacity: float = 1.0
    dissolve_rate: float = 0.5
    deposition_rate: float = 0.5
    min_tilt: float = 0.005
    deep_water_depth: float = 0.004
    min_velocity_depth: float = 1e-3
    max_courant: float = 0.25
    max_height_step_m: float = 0.5
    closed_boundary: bool = True
    talus_angle_deg: float = 34.0
    thermal_rate: float = 0.5
    thermal_cadence: int = 8
    thermal_settle_passes: int = 24

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise MapConfigError(f"iterations {self.iterations} must be at least 1")
        if self.settle_iterations < 0:
            raise MapConfigError(f"settle_iterations {self.settle_iterations} must not be negative")
        if self.dt <= 0.0:
            raise MapConfigError(f"dt {self.dt} must be positive")
        if self.rain_rate < 0.0:
            raise MapConfigError(f"rain_rate {self.rain_rate} must not be negative")
        if not 0.0 <= self.evaporation_rate * self.dt < 1.0:
            raise MapConfigError(
                f"evaporation_rate {self.evaporation_rate} removes all water in one dt"
            )
        if self.pipe_area <= 0.0:
            raise MapConfigError(f"pipe_area {self.pipe_area} must be positive")
        if self.gravity <= 0.0:
            raise MapConfigError(f"gravity {self.gravity} must be positive")
        for name in ("sediment_capacity", "dissolve_rate", "deposition_rate"):
            value = getattr(self, name)
            if value < 0.0:
                raise MapConfigError(f"{name} {value} must not be negative")
        if not 0.0 <= self.min_tilt <= 1.0:
            raise MapConfigError(f"min_tilt {self.min_tilt} must lie within [0, 1]")
        if self.deep_water_depth <= 0.0:
            raise MapConfigError(f"deep_water_depth {self.deep_water_depth} must be positive")
        if self.min_velocity_depth <= 0.0:
            raise MapConfigError(f"min_velocity_depth {self.min_velocity_depth} must be positive")
        if not 0.0 < self.max_courant <= 1.0:
            raise MapConfigError(f"max_courant {self.max_courant} must lie within (0, 1]")
        if self.max_height_step_m <= 0.0:
            raise MapConfigError(f"max_height_step_m {self.max_height_step_m} must be positive")
        if not 0.0 < self.talus_angle_deg < 90.0:
            raise MapConfigError(
                f"talus_angle_deg {self.talus_angle_deg} must lie strictly between 0 and 90"
            )
        if not 0.0 < self.thermal_rate <= 1.0:
            raise MapConfigError(f"thermal_rate {self.thermal_rate} must lie within (0, 1]")
        if self.thermal_cadence < 0:
            raise MapConfigError(f"thermal_cadence {self.thermal_cadence} must not be negative")
        if self.thermal_settle_passes < 0:
            raise MapConfigError(
                f"thermal_settle_passes {self.thermal_settle_passes} must not be negative"
            )


EROSION_PRESETS: dict[str, ErosionParams] = {
    "light": ErosionParams(iterations=200, sediment_capacity=0.5),
    "default": ErosionParams(),
    "heavy": ErosionParams(iterations=700, sediment_capacity=2.0),
    "canyon": ErosionParams(
        iterations=1200,
        rain_rate=0.003,
        sediment_capacity=1.5,
        dissolve_rate=0.6,
        deposition_rate=0.2,
        deep_water_depth=0.008,
    ),
}

DEFAULT_PRESET = "default"


def erosion_preset(name: str) -> ErosionParams:
    """Look a named erosion preset up, or raise."""
    try:
        return EROSION_PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(EROSION_PRESETS))
        raise MapConfigError(f"unknown erosion preset {name!r}; expected one of {known}") from None


def erosion_preset_names() -> tuple[str, ...]:
    return tuple(sorted(EROSION_PRESETS))


def resolve_params(params: str | ErosionParams | None = None) -> ErosionParams:
    if params is None:
        return EROSION_PRESETS[DEFAULT_PRESET]
    if isinstance(params, ErosionParams):
        return params
    return erosion_preset(params)


def _neighbour(field: torch.Tensor, direction: int) -> torch.Tensor:
    """Value of the neighbour in this direction, edge-replicated so borders see no drop."""
    if direction == 0:
        return torch.cat((field[:, :1], field[:, :-1]), dim=1)
    if direction == 1:
        return torch.cat((field[:, 1:], field[:, -1:]), dim=1)
    if direction == 2:
        return torch.cat((field[:1, :], field[:-1, :]), dim=0)
    return torch.cat((field[1:, :], field[-1:, :]), dim=0)


def _gather(field: torch.Tensor, direction: int) -> torch.Tensor:
    """Value of the neighbour in this direction, zero-filled so nothing flows in from outside."""
    if direction in (0, 1):
        pad = torch.zeros_like(field[:, :1])
        if direction == 0:
            return torch.cat((pad, field[:, :-1]), dim=1)
        return torch.cat((field[:, 1:], pad), dim=1)
    pad = torch.zeros_like(field[:1, :])
    if direction == 2:
        return torch.cat((pad, field[:-1, :]), dim=0)
    return torch.cat((field[1:, :], pad), dim=0)


def _seal_edges(flux: torch.Tensor) -> None:
    flux[0, :, 0] = 0.0
    flux[1, :, -1] = 0.0
    flux[2, 0, :] = 0.0
    flux[3, -1, :] = 0.0


def _slope(terrain: torch.Tensor, cell_m: float) -> torch.Tensor:
    grad_x = (_neighbour(terrain, 1) - _neighbour(terrain, 0)).div_(2.0 * cell_m)
    grad_y = (_neighbour(terrain, 3) - _neighbour(terrain, 2)).div_(2.0 * cell_m)
    return torch.hypot(grad_x, grad_y)


def _high_quantile(field: torch.Tensor) -> float:
    flat = field.reshape(-1)
    if flat.numel() > _QUANTILE_SAMPLE_LIMIT:
        flat = flat[:: max(1, flat.numel() // _QUANTILE_SAMPLE_LIMIT)]
    return float(torch.quantile(flat.float(), _QUANTILE_HIGH))


def robust_normalise(field: torch.Tensor) -> torch.Tensor:
    """Rescale onto [0, 1] against the 99.5th percentile so one outlier cannot flatten it."""
    high = _high_quantile(field)
    low = float(field.min())
    if high - low <= _EPS:
        return torch.zeros_like(field)
    return field.sub(low).div_(high - low).clamp_(0.0, 1.0)


def log_normalise(field: torch.Tensor) -> torch.Tensor:
    """Rescale a heavy-tailed accumulator onto [0, 1] by stretching its decades, not its values."""
    return robust_normalise(torch.log1p(field.clamp(min=0.0)))


def _thermal_step(terrain: torch.Tensor, max_delta: float, rate: float) -> torch.Tensor:
    """One talus relaxation sweep in place. Returns the material each cell shed."""
    excess = torch.stack([terrain - _neighbour(terrain, direction) for direction in range(4)])
    excess.sub_(max_delta).clamp_(min=0.0)
    amount = excess.max(0).values.mul_(0.5 * rate)
    moved = excess.mul_(excess.sum(0).clamp_(min=_EPS).reciprocal_()).mul_(amount)
    outflow = moved.sum(0)
    terrain.sub_(outflow)
    for direction in range(4):
        terrain.add_(_gather(moved[_OPPOSITE[direction]], direction))
    return outflow


def thermal(
    cfg: MapConfig,
    height: torch.Tensor,
    angle_deg: float = ErosionParams.talus_angle_deg,
    rate: float = ErosionParams.thermal_rate,
    iterations: int = 32,
    device: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone talus relaxation. Returns the relaxed height and the shed material in metres."""
    if tuple(height.shape) != cfg.shape:
        raise MapConfigError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    if iterations < 0:
        raise MapConfigError(f"iterations {iterations} must not be negative")
    if not 0.0 < rate <= 1.0:
        raise MapConfigError(f"rate {rate} must lie within (0, 1]")
    target = as_torch_device(device)
    terrain = height.to(device=target, dtype=torch.float32).clone()
    max_delta = cfg.talus_delta(angle_deg)
    shed = torch.zeros_like(terrain)
    for _ in range(iterations):
        shed.add_(_thermal_step(terrain, max_delta, rate))
    return terrain.clamp_(0.0, 1.0), shed.mul_(cfg.height_range_m)


@dataclass(frozen=True)
class ErosionResult:
    height: torch.Tensor
    water_depth_m: torch.Tensor
    sediment_m: torch.Tensor
    speed_m_per_s: torch.Tensor
    flux_m3_per_s: torch.Tensor
    flow_volume_m3: torch.Tensor
    erosion_m: torch.Tensor
    deposition_m: torch.Tensor
    talus_m: torch.Tensor
    params: ErosionParams

    def channels(self) -> dict[str, torch.Tensor]:
        """The erosion-derived slice of the shared channel stack, every field in [0, 1].

        No flow: the outflow limiter leaves flow_volume_m3 near uniform, so the flow channel
        comes from channels.drainage instead.
        """
        return {
            "height": self.height,
            "deposition": log_normalise(self.deposition_m),
            "wear": log_normalise(self.erosion_m),
            "water": self.water_depth_m.div(_WATER_SURFACE_M).clamp(0.0, 1.0),
            "water_depth": robust_normalise(self.water_depth_m),
        }

    def stats(self) -> dict[str, float]:
        return {
            "iterations": float(self.params.iterations),
            "eroded_mean_m": float(self.erosion_m.mean()),
            "eroded_max_m": float(self.erosion_m.max()),
            "deposited_mean_m": float(self.deposition_m.mean()),
            "deposited_max_m": float(self.deposition_m.max()),
            "suspended_mean_m": float(self.sediment_m.mean()),
            "water_mean_m": float(self.water_depth_m.mean()),
            "water_max_m": float(self.water_depth_m.max()),
            "speed_max_m_per_s": float(self.speed_m_per_s.max()),
            "talus_mean_m": float(self.talus_m.mean()),
            "talus_max_m": float(self.talus_m.max()),
            "net_change_m": float(self.deposition_m.mean() - self.erosion_m.mean()),
        }


def erode(
    cfg: MapConfig,
    height: torch.Tensor,
    params: str | ErosionParams | None = None,
    rain: torch.Tensor | None = None,
    device: Any | None = None,
) -> ErosionResult:
    """Pipe-model hydraulic erosion. Takes and returns normalised height in [0, 1]."""
    params = resolve_params(params)
    if tuple(height.shape) != cfg.shape:
        raise MapConfigError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    target = as_torch_device(device)
    terrain = height.to(device=target, dtype=torch.float32).clone()
    slope_scale = cfg.slope_scale()
    max_step = params.max_height_step_m / cfg.height_range_m
    flow_constant = params.dt * params.pipe_area * params.gravity
    if rain is None:
        rain_step = torch.full_like(terrain, params.rain_rate * params.dt)
    else:
        if tuple(rain.shape) != cfg.shape:
            raise MapConfigError(f"rain {tuple(rain.shape)} does not match cfg.shape {cfg.shape}")
        rain_step = rain.to(device=target, dtype=torch.float32).mul(params.rain_rate * params.dt)
    water = torch.zeros_like(terrain)
    sediment = torch.zeros_like(terrain)
    flux = torch.zeros((4, cfg.resolution, cfg.resolution), device=target, dtype=torch.float32)
    flow_volume = torch.zeros_like(terrain)
    eroded_total = torch.zeros_like(terrain)
    deposited_total = torch.zeros_like(terrain)
    speed = torch.zeros_like(terrain)
    talus_total = torch.zeros_like(terrain)
    talus_delta = cfg.talus_delta(params.talus_angle_deg)
    evaporation = max(0.0, 1.0 - params.evaporation_rate * params.dt)
    courant_limit = params.max_courant / params.dt
    total_steps = params.iterations + params.settle_iterations
    for step in range(total_steps):
        if step < params.iterations:
            water.add_(rain_step)
        surface = terrain + water
        for direction in range(4):
            delta = surface - _neighbour(surface, direction)
            flux[direction].add_(delta, alpha=flow_constant).clamp_(min=0.0)
        if params.closed_boundary:
            _seal_edges(flux)
        outflow = flux.sum(0)
        limit = water.div(outflow.mul(params.dt).add_(_EPS)).clamp_(max=1.0)
        flux.mul_(limit)
        outflow.mul_(limit)
        share = flux.mul(params.dt).div_(water.clamp(min=_EPS)).clamp_(min=0.0)
        total_share = share.sum(0)
        share.mul_(total_share.clamp_(min=1.0).reciprocal_())
        inflow = torch.zeros_like(outflow)
        for direction in range(4):
            inflow.add_(_gather(flux[_OPPOSITE[direction]], direction))
        water_next = water.add(inflow.sub_(outflow), alpha=params.dt).clamp_(min=0.0)
        depth = water.add_(water_next).mul_(0.5).clamp_(min=params.min_velocity_depth)
        velocity_x = _gather(flux[1], 0).sub_(flux[0]).add_(flux[1]).sub_(_gather(flux[0], 1))
        velocity_y = _gather(flux[3], 2).sub_(flux[2]).add_(flux[3]).sub_(_gather(flux[2], 3))
        velocity_x.mul_(0.5).div_(depth)
        velocity_y.mul_(0.5).div_(depth)
        speed = torch.hypot(velocity_x, velocity_y)
        brake = speed.clamp(min=_EPS).reciprocal_().mul_(courant_limit).clamp_(max=1.0)
        velocity_x.mul_(brake)
        velocity_y.mul_(brake)
        speed.mul_(brake)
        flow_volume.add_(outflow, alpha=params.dt)
        water = water_next
        slope = _slope(terrain, 1.0).mul_(slope_scale)
        sin_tilt = slope.div(torch.sqrt(slope * slope + 1.0)).clamp_(min=params.min_tilt)
        submersion = water.div(params.deep_water_depth).neg_().add_(1.0).clamp_(0.0, 1.0)
        capacity = sin_tilt.mul_(speed).mul_(submersion).mul_(water).mul_(params.sediment_capacity)
        excess = capacity.sub_(sediment)
        dissolved = excess.clamp(min=0.0).mul_(params.dissolve_rate * params.dt).clamp_(max=max_step)
        settled = excess.neg_().clamp_(min=0.0).mul_(params.deposition_rate * params.dt)
        settled = torch.minimum(settled.clamp_(max=max_step), sediment)
        terrain.add_(settled).sub_(dissolved)
        sediment.add_(dissolved).sub_(settled)
        eroded_total.add_(dissolved)
        deposited_total.add_(settled)
        carried = share.mul(sediment)
        sediment = sediment.sub_(carried.sum(0))
        for direction in range(4):
            sediment.add_(_gather(carried[_OPPOSITE[direction]], direction))
        water.mul_(evaporation)
        if params.thermal_cadence and (step + 1) % params.thermal_cadence == 0:
            talus_total.add_(_thermal_step(terrain, talus_delta, params.thermal_rate))
    if params.thermal_cadence:
        for _ in range(params.thermal_settle_passes):
            talus_total.add_(_thermal_step(terrain, talus_delta, params.thermal_rate))
    metre = cfg.height_range_m
    cell_volume = cfg.metres_per_pixel * cfg.metres_per_pixel * metre
    return ErosionResult(
        height=terrain.clamp_(0.0, 1.0),
        water_depth_m=water.mul_(metre),
        sediment_m=sediment.mul_(metre),
        speed_m_per_s=speed.mul_(cfg.metres_per_pixel),
        flux_m3_per_s=flux.sum(0).mul_(cell_volume),
        flow_volume_m3=flow_volume.mul_(cell_volume),
        erosion_m=eroded_total.mul_(metre),
        deposition_m=deposited_total.mul_(metre),
        talus_m=talus_total.mul_(metre),
        params=params,
    )
