from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

import torch

from terrain import synth
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError
from terrain.device import as_torch_device
from terrain.erosion import ErosionResult, log_normalise, robust_normalise

CHANNEL_INDEX: dict[str, int] = {name: index for index, name in enumerate(CHANNEL_NAMES)}

GEOMETRIC_CHANNELS = ("slope", "curvature", "bedrock", "strata")
EROSION_DERIVED_CHANNELS = ("flow", "deposition", "wear", "wetness")
WATER_CHANNELS = ("water", "water_depth")
CLIMATE_CHANNELS = ("temperature", "moisture")
PATCHINESS_CHANNELS = ("patchiness_fine", "patchiness_mid", "patchiness_coarse")

_EPS = 1e-8
_QUANTILE_SAMPLE_LIMIT = 1 << 20
_RANGE_TOLERANCE = 1e-4
_HALF_PI = 0.5 * math.pi
_NEIGHBOURS = ((0, -1), (0, 1), (1, -1), (1, 1))
_ACCUMULATION_FLOOR = 1e-3
_SMOOTH_PASSES = 3
_FILL_CEILING = 2.0
_PYRAMID_FLOOR = 8
_LOWER_WEIGHT = 2.0


@dataclass(frozen=True)
class GeometryParams:
    """Settings for the channels derived from the height field alone."""

    curvature_radius_m: float = 24.0
    curvature_gain: float = 1.5
    soil_depth_m: float = 6.0
    soil_smooth_m: float = 120.0
    soil_slope_deg: float = 32.0
    strata_thickness_m: float = 18.0
    strata_dip_deg: float = 12.0
    strata_strike_deg: float = 25.0
    strata_warp_m: float = 40.0
    strata_warp_feature_size_m: float = 900.0
    strata_edge: float = 0.2
    wetness_min_tilt: float = 0.005

    def __post_init__(self) -> None:
        for name in ("curvature_radius_m", "curvature_gain", "soil_smooth_m", "strata_thickness_m"):
            value = getattr(self, name)
            if value <= 0.0:
                raise MapConfigError(f"{name} {value} must be positive")
        if self.soil_depth_m < 0.0:
            raise MapConfigError(f"soil_depth_m {self.soil_depth_m} must not be negative")
        if not 0.0 < self.soil_slope_deg < 90.0:
            raise MapConfigError(
                f"soil_slope_deg {self.soil_slope_deg} must lie strictly between 0 and 90"
            )
        if not 0.0 <= self.strata_dip_deg < 90.0:
            raise MapConfigError(f"strata_dip_deg {self.strata_dip_deg} must lie within [0, 90)")
        if self.strata_warp_m < 0.0:
            raise MapConfigError(f"strata_warp_m {self.strata_warp_m} must not be negative")
        if self.strata_warp_feature_size_m <= 0.0:
            raise MapConfigError(
                f"strata_warp_feature_size_m {self.strata_warp_feature_size_m} must be positive"
            )
        if not 0.0 < self.strata_edge <= 1.0:
            raise MapConfigError(f"strata_edge {self.strata_edge} must lie within (0, 1]")
        if not 0.0 < self.wetness_min_tilt <= 1.0:
            raise MapConfigError(f"wetness_min_tilt {self.wetness_min_tilt} must lie within (0, 1]")


def _target_device(device: Any | None, field: torch.Tensor) -> Any:
    if device is None:
        return field.device
    return as_torch_device(device)


def _shift(field: torch.Tensor, dim: int, offset: int) -> torch.Tensor:
    """Neighbour values one cell along this axis, edge replicated."""
    size = field.shape[dim]
    if offset < 0:
        return torch.cat((field.narrow(dim, 0, 1), field.narrow(dim, 0, size - 1)), dim=dim)
    return torch.cat((field.narrow(dim, 1, size - 1), field.narrow(dim, size - 1, 1)), dim=dim)


def _push(field: torch.Tensor, dim: int, offset: int) -> torch.Tensor:
    """Move every value one cell along this axis, dropping whatever leaves the map."""
    size = field.shape[dim]
    zeros = list(field.shape)
    zeros[dim] = 1
    blank = field.new_zeros(zeros)
    if offset < 0:
        return torch.cat((blank, field.narrow(dim, 0, size - 1)), dim=dim)
    return torch.cat((field.narrow(dim, 1, size - 1), blank), dim=dim)


def _box_blur(field: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    size = field.shape[dim]
    radius = max(0, min(radius, size - 1))
    if radius == 0:
        return field
    shape = list(field.shape)
    shape[dim] = radius
    lead = field.narrow(dim, 0, 1).expand(shape)
    tail = field.narrow(dim, size - 1, 1).expand(shape)
    padded = torch.cat((lead, field, tail), dim=dim)
    zeros = list(padded.shape)
    zeros[dim] = 1
    cumulative = torch.cat((padded.new_zeros(zeros), padded.cumsum(dim)), dim=dim)
    window = 2 * radius + 1
    upper = cumulative.narrow(dim, window, size)
    lower = cumulative.narrow(dim, 0, size)
    return upper.sub(lower).div_(window)


def smooth(field: torch.Tensor, sigma_px: float, passes: int = _SMOOTH_PASSES) -> torch.Tensor:
    """Gaussian-equivalent blur built from repeated box passes, cost independent of the radius."""
    if passes < 1:
        raise MapConfigError(f"passes {passes} must be at least 1")
    if sigma_px <= 0.0:
        return field.clone()
    variance = 12.0 * sigma_px * sigma_px / passes
    radius = max(1, int(round(0.5 * (math.sqrt(variance + 1.0) - 1.0))))
    result = field
    for _ in range(passes):
        result = _box_blur(_box_blur(result, radius, 0), radius, 1)
    return result


def gradient(cfg: MapConfig, height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference height gradient in metres per metre, edge replicated."""
    scale = cfg.height_range_m / (2.0 * cfg.metres_per_pixel)
    grad_x = _shift(height, 1, 1).sub(_shift(height, 1, -1)).mul_(scale)
    grad_y = _shift(height, 0, 1).sub(_shift(height, 0, -1)).mul_(scale)
    return grad_x, grad_y


def slope_gradient(cfg: MapConfig, height: torch.Tensor) -> torch.Tensor:
    """Slope as rise over run, unbounded above."""
    grad_x, grad_y = gradient(cfg, height)
    return torch.hypot(grad_x, grad_y)


def slope_angle_deg(cfg: MapConfig, height: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.atan(slope_gradient(cfg, height)))


def slope(cfg: MapConfig, height: torch.Tensor) -> torch.Tensor:
    """The slope channel: 0 flat, 1 vertical, linear in the slope angle."""
    return torch.atan(slope_gradient(cfg, height)).div_(_HALF_PI).clamp_(0.0, 1.0)


def curvature_per_m(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
) -> torch.Tensor:
    """Laplacian of the height field at the parameter radius, positive in hollows."""
    field = smooth(height, params.curvature_radius_m / cfg.metres_per_pixel)
    cell = cfg.metres_per_pixel
    stencil = _shift(field, 1, 1).add_(_shift(field, 1, -1))
    stencil.add_(_shift(field, 0, 1)).add_(_shift(field, 0, -1)).sub_(field.mul(4.0))
    return stencil.mul_(cfg.height_range_m / (cell * cell))


def curvature(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
) -> torch.Tensor:
    """The curvature channel: 0.5 flat, below 0.5 convex ridges, above 0.5 concave hollows."""
    field = curvature_per_m(cfg, height, params)
    scale = float(field.abs().mean()) / params.curvature_gain
    if scale <= _EPS:
        return torch.full_like(field, 0.5)
    return field.div_(scale).tanh_().mul_(0.5).add_(0.5)


def soil_depth_m(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
) -> torch.Tensor:
    """Smoothed soil mantle in metres: thick on gentle ground, stripped off steep faces."""
    fraction = slope_angle_deg(cfg, height).div_(params.soil_slope_deg)
    fraction.neg_().add_(1.0).clamp_(0.0, 1.0)
    return smooth(fraction, params.soil_smooth_m / cfg.metres_per_pixel).mul_(params.soil_depth_m)


def bedrock(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
    soil_m: torch.Tensor | None = None,
) -> torch.Tensor:
    """The bedrock channel: height with the soil mantle removed, in the same normalised units."""
    depth = soil_depth_m(cfg, height, params) if soil_m is None else soil_m.to(height.device)
    return height.sub(depth.div(cfg.height_range_m)).clamp_(0.0, 1.0)


def strata(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
    device: Any | None = None,
) -> torch.Tensor:
    """The strata channel: banded rock beds cut by a plane dipping at strata_dip_deg."""
    target = _target_device(device, height)
    x, y = synth.grid(cfg, target)
    dip = math.radians(params.strata_dip_deg)
    strike = math.radians(params.strata_strike_deg)
    distance = height.to(device=target, dtype=torch.float32).mul(
        cfg.height_range_m * math.cos(dip)
    )
    distance.add_(x, alpha=math.sin(dip) * math.cos(strike))
    distance.add_(y, alpha=math.sin(dip) * math.sin(strike))
    if params.strata_warp_m > 0.0:
        jitter = synth.warped_fbm(
            cfg,
            synth.FractalParams(params.strata_warp_feature_size_m, 3),
            label="strata",
            device=target,
        )
        distance.add_(jitter.sub_(0.5), alpha=2.0 * params.strata_warp_m)
    distance.div_(params.strata_thickness_m)
    band = torch.floor(distance)
    edge = distance.sub_(band).sub_(1.0 - params.strata_edge).div_(params.strata_edge)
    edge.clamp_(0.0, 1.0)
    blend = edge.mul(edge).mul_(edge.mul(-2.0).add_(3.0))
    index = band.to(torch.int64)
    seed = cfg.derive_seed("strata")
    low = synth.hash01(index, seed)
    high = synth.hash01(index + 1, seed)
    return torch.lerp(low, high, blend).clamp_(0.0, 1.0)


def wetness(
    cfg: MapConfig,
    accumulated: torch.Tensor,
    height: torch.Tensor,
    min_tilt: float = GeometryParams.wetness_min_tilt,
) -> torch.Tensor:
    """Topographic wetness index: upslope contributing area over local tilt, normalised."""
    if min_tilt <= 0.0:
        raise MapConfigError(f"min_tilt {min_tilt} must be positive")
    tilt = slope_gradient(cfg, height).clamp_(min=min_tilt)
    index = torch.log1p(accumulated.clamp(min=0.0)).sub_(torch.log(tilt))
    return robust_normalise(index)


@dataclass(frozen=True)
class WaterParams:
    """Settings for the water surface reconstructed from the eroded height and the sim's film."""

    sea_level_m: float | None = None
    sea_level_quantile: float | None = None
    fill_passes: int = 0
    accumulate_passes: int = 0
    flow_exponent: float = 1.1
    convergence_stride: int = 64
    river_area_fraction: float = 0.03
    river_core_fraction: float = 0.002
    river_depth_m: float = 1.2
    river_depth_exponent: float = 0.4
    river_width_m: float = 12.0
    shore_depth_m: float = 0.15
    full_depth_m: float = 0.6
    depth_reference_m: float = 12.0
    surface_smooth_m: float = 0.0

    def passes(self, resolution: int) -> tuple[int, int]:
        """Fill and accumulation pass budgets, both O(longest flow path) when left at zero."""
        default = 2 * resolution
        return (self.fill_passes or default, self.accumulate_passes or default)

    def __post_init__(self) -> None:
        if self.sea_level_quantile is not None and not 0.0 <= self.sea_level_quantile < 1.0:
            raise MapConfigError(
                f"sea_level_quantile {self.sea_level_quantile} must lie within [0, 1)"
            )
        if not 0.0 < self.river_core_fraction < self.river_area_fraction < 1.0:
            raise MapConfigError(
                f"river fractions {self.river_core_fraction} and {self.river_area_fraction} "
                "must rise within (0, 1)"
            )
        if self.river_depth_m < 0.0:
            raise MapConfigError(f"river_depth_m {self.river_depth_m} must not be negative")
        if self.fill_passes < 0 or self.accumulate_passes < 0:
            raise MapConfigError("pass budgets must not be negative")
        if self.flow_exponent <= 0.0:
            raise MapConfigError(f"flow_exponent {self.flow_exponent} must be positive")
        if self.convergence_stride < 1:
            raise MapConfigError(f"convergence_stride {self.convergence_stride} must be positive")
        if self.river_width_m < 0.0:
            raise MapConfigError(f"river_width_m {self.river_width_m} must not be negative")
        if self.river_depth_exponent <= 0.0:
            raise MapConfigError(
                f"river_depth_exponent {self.river_depth_exponent} must be positive"
            )
        if self.shore_depth_m < 0.0:
            raise MapConfigError(f"shore_depth_m {self.shore_depth_m} must not be negative")
        if self.full_depth_m <= self.shore_depth_m:
            raise MapConfigError(
                f"full_depth_m {self.full_depth_m} must exceed shore_depth_m {self.shore_depth_m}"
            )
        if self.depth_reference_m <= 0.0:
            raise MapConfigError(f"depth_reference_m {self.depth_reference_m} must be positive")
        if self.surface_smooth_m < 0.0:
            raise MapConfigError(f"surface_smooth_m {self.surface_smooth_m} must not be negative")


@dataclass(frozen=True)
class WaterLevel:
    """What the water pass decided, for terrain.json and for the agent reading a preview."""

    sea_level_m: float
    covered_fraction: float
    mean_depth_m: float
    max_depth_m: float
    depth_reference_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sea_level_m": self.sea_level_m,
            "covered_fraction": self.covered_fraction,
            "mean_depth_m": self.mean_depth_m,
            "max_depth_m": self.max_depth_m,
            "depth_reference_m": self.depth_reference_m,
        }


def dilate(field: torch.Tensor, radius: int) -> torch.Tensor:
    """Local maximum over a square window, the cheap way to widen a one-pixel channel."""
    if radius < 1:
        return field
    window = 2 * radius + 1
    return torch.nn.functional.max_pool2d(
        field.unsqueeze(0), window, stride=1, padding=radius
    ).squeeze(0)


def quantile(field: torch.Tensor, fraction: float) -> float:
    """Quantile of a field, subsampled above the torch element limit so 4096^2 still works."""
    flat = field.reshape(-1)
    if flat.numel() > _QUANTILE_SAMPLE_LIMIT:
        flat = flat[:: max(1, flat.numel() // _QUANTILE_SAMPLE_LIMIT)]
    return float(torch.quantile(flat.float(), fraction))


def sea_level_m(
    cfg: MapConfig,
    height: torch.Tensor,
    params: WaterParams = WaterParams(),
) -> float:
    """The still-water level in metres: explicit, or a quantile of the eroded height, else the config."""
    if params.sea_level_m is not None:
        return params.sea_level_m
    if params.sea_level_quantile is not None:
        return quantile(height, params.sea_level_quantile) * cfg.height_range_m
    return cfg.sea_level_m


def _outlets(height: torch.Tensor, level: float) -> torch.Tensor:
    outlet = height <= level
    outlet[0, :] = True
    outlet[-1, :] = True
    outlet[:, 0] = True
    outlet[:, -1] = True
    return outlet


def outlet_mask(cfg: MapConfig, height: torch.Tensor, level_m: float) -> torch.Tensor:
    """Cells water may leave the map through: the border, and anything already under the sea."""
    return _outlets(height, level_m / cfg.height_range_m)


def _neighbourhood(field: torch.Tensor) -> torch.Tensor:
    return torch.stack([_shift(field, dim, offset) for dim, offset in _NEIGHBOURS])


def _lowest(field: torch.Tensor) -> torch.Tensor:
    """Smallest of the four neighbour values, folded pairwise so no four-plane stack is materialised."""
    low = _shift(field, *_NEIGHBOURS[0])
    for dim, offset in _NEIGHBOURS[1:]:
        low = torch.minimum(low, _shift(field, dim, offset))
    return low


def _highest(field: torch.Tensor) -> torch.Tensor:
    """Largest of the four neighbour values."""
    high = _shift(field, *_NEIGHBOURS[0])
    for dim, offset in _NEIGHBOURS[1:]:
        high = torch.maximum(high, _shift(field, dim, offset))
    return high


def _collect(moved: torch.Tensor) -> torch.Tensor:
    """Sum of the four neighbour parcels landing on each cell, folded in place for the same reason."""
    landed = _push(moved[0], _NEIGHBOURS[0][0], -_NEIGHBOURS[0][1])
    for index, (dim, offset) in enumerate(_NEIGHBOURS[1:], start=1):
        landed.add_(_push(moved[index], dim, -offset))
    return landed


def _mark_neighbours(flags: torch.Tensor) -> torch.Tensor:
    """OR the four per-direction flag planes onto the actual neighbour cell each one names."""
    marked = _push(flags[0], _NEIGHBOURS[0][0], -_NEIGHBOURS[0][1])
    for index, (dim, offset) in enumerate(_NEIGHBOURS[1:], start=1):
        marked = marked | _push(flags[index], dim, -offset)
    return marked


def _settle_fill(
    height: torch.Tensor,
    level: float,
    start: torch.Tensor,
    budget: int,
    stride: int,
) -> torch.Tensor:
    outlet = _outlets(height, level)
    filled = start
    for step in range(budget):
        low = _lowest(filled)
        following = torch.where(outlet, height, torch.maximum(height, low))
        if step % stride == stride - 1 and bool(torch.equal(following, filled)):
            return following
        filled = following
    return filled


def _coarsen(field: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(field.unsqueeze(0), 2, stride=2).squeeze(0)


def _refine(field: torch.Tensor) -> torch.Tensor:
    return field.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)


def _fill(height: torch.Tensor, level: float, budget: int, stride: int) -> torch.Tensor:
    """Minimax fill solved coarse to fine, each level starting from an upper bound on the next."""
    if height.shape[0] <= _PYRAMID_FLOOR:
        start = torch.full_like(height, _FILL_CEILING)
    else:
        start = _refine(_fill(_coarsen(height), level, budget, stride))
    return _settle_fill(height, level, start, budget, stride)


def fill_depressions(
    cfg: MapConfig,
    height: torch.Tensor,
    level_m: float | None = None,
    params: WaterParams = WaterParams(),
) -> torch.Tensor:
    """Every pit raised to the lowest rim it can spill over, flats left for flat_gradient to route."""
    level = cfg.sea_level_m if level_m is None else level_m
    budget, _ = params.passes(cfg.resolution)
    return _fill(height, level / cfg.height_range_m, budget, params.convergence_stride)


_BREACH_ROUNDS = 3
_BREACH_SMOOTH_M = 24.0
_BREACH_NOTCH_M = 12.0
_BREACH_REACH_M = 200.0
_BREACH_MIN_AREA_M2 = 25600.0
_INF = float("inf")


def _dry_reach(current: torch.Tensor, dry: torch.Tensor, hops: int) -> torch.Tensor:
    """The lowest height reachable from each dry cell by walking up to `hops` steps of dry ground.

    A real drainage divide can be many cells thick - fill_depressions' own pit mask only tells a
    cell it does not need raising, not how far it is from open, actually-lower ground. This is a
    bounded multi-source relaxation of that question: seed every dry cell with its own height, then
    repeatedly let it adopt a lower value from a dry neighbour, `hops` times. A cell one step from
    open low ground sees it after one pass; a cell at the far side of a wide rim needs as many
    passes as the rim is thick. Pit cells stay at infinity throughout - water has not decided where
    they drain yet - so they never leak into a neighbour's reach and never adopt one either.
    """
    reach = torch.where(dry, current, torch.full_like(current, _INF))
    for _ in range(hops):
        reach = torch.where(dry, torch.minimum(reach, _lowest(reach)), reach)
    return reach


def _lake_escape(
    current: torch.Tensor,
    filled: torch.Tensor,
    pit: torch.Tensor,
    reach: torch.Tensor,
    min_cells: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The lowest ground each basin can reach through its rim, and where the rim should be crossed.

    fill_depressions leaves `filled` exactly constant across a connected lake (proven by
    test_fill_depressions_leaves_one_exact_lake_surface_not_an_epsilon_ladder), and adjacent pit
    cells can never disagree on it either - a lower neighbour would already have pulled a higher
    one down to match. That constant is a free grouping key: group every pit cell by its `filled`
    value and take the minimum `reach` found at any dry cell the group touches - not that dry
    cell's own height, but the lowest ground it can walk to through further dry ground (_dry_reach)
    - and the result is the true lowest point the whole lake can escape to, however thick the rim
    between them is. A basin with no bordering ground reachable below its own rim - nested behind a
    second, higher divide, or a real endorheic sink - keeps its current height; there is nothing
    lower to carve down to. That includes a basin whose rim borders dry ground only at exactly its
    own flood level (a perfectly enclosed bowl, its dry border sitting at the same height all the
    way round, with no reach beyond it either): the numeric minimum still resolves there, but it is
    not an improvement, and treating it as one would plant a crossing anywhere the lake happens to
    touch dry ground at all, not at the one real low point a genuine rim has.

    A real eroded height field carries thousands of pixel-scale pits that are numerical noise, not
    basins - each one individually bounded, but there are so many that breaching every single one
    still carves a channel through most of the map's dry ground in aggregate. `min_cells` gates
    that: a lake smaller than it is left exactly as fill_depressions would flood it, the same as a
    lake with no improving escape at all, because a puddle that small was never going to move
    covered_fraction and is not worth a channel.

    `door` marks the specific pit cell(s) that actually achieve that lake-wide minimum; `entry`
    marks the dry neighbour each one crosses into - the mouth of the notch, not the far end of it,
    since the ground beyond may be many cells of rim away yet. Every other pit cell in the lake,
    including its entire interior floor, is left out: a real lake spills through one gap in its
    rim, it does not drain from the middle.
    """
    neighbour_pit = _neighbourhood(pit)
    neighbour_reach = _neighbourhood(reach)
    shore4 = torch.where(neighbour_pit, torch.full_like(neighbour_reach, _INF), neighbour_reach)
    shore = torch.where(pit, shore4.amin(dim=0), torch.full_like(current, _INF))

    levels, inverse = torch.unique(filled[pit], return_inverse=True)
    group_escape = torch.full((levels.numel(),), _INF, dtype=current.dtype, device=current.device)
    group_escape.scatter_reduce_(0, inverse, shore[pit], reduce="amin", include_self=True)
    group_size = torch.zeros_like(group_escape)
    group_size.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=group_size.dtype))
    found = group_escape[inverse]
    improves = (found < levels[inverse].sub(_EPS)) & (group_size[inverse] >= min_cells)
    resolved = torch.where(improves, found, current[pit])

    escape = current.clone()
    escape[pit] = resolved

    door = torch.zeros_like(pit)
    door[pit] = improves & (shore[pit] == found)
    crossing = door.unsqueeze(0) & (shore4 == escape.unsqueeze(0)) & ~neighbour_pit
    entry = _mark_neighbours(crossing)
    return escape, door, entry


def _breach_round(
    cfg: MapConfig,
    current: torch.Tensor,
    level: float,
    params: WaterParams,
    smooth_m: float,
    notch_m: float,
    reach_m: float,
    min_area_m2: float,
) -> tuple[torch.Tensor, bool]:
    filled = fill_depressions(cfg, current, level, params)
    pit = filled.gt(current.add(_EPS))
    if not bool(pit.any()):
        return current, False
    dry = ~pit
    hops = max(1, int(round(reach_m / cfg.metres_per_pixel)))
    reach = _dry_reach(current, dry, hops)
    min_cells = min_area_m2 / (cfg.metres_per_pixel * cfg.metres_per_pixel)
    escape, door, entry = _lake_escape(current, filled, pit, reach, min_cells)
    if not bool(entry.any()):
        return current, False
    radius = max(1, int(round(notch_m / cfg.metres_per_pixel)))
    mouth = smooth(dilate((door | entry).to(current.dtype), radius), 0.5 * radius) > 0.0
    distance = _geodesic(entry, dry, hops, params.convergence_stride)
    cutoff = float(hops)
    softened = smooth(distance.clamp(max=cutoff + 1.0), 0.5 * radius)
    channel = dry & (softened < cutoff)
    notch = mouth | channel
    target = torch.where(notch, torch.where(pit, escape, reach), current)
    sigma_px = smooth_m / cfg.metres_per_pixel
    ramp = smooth(target, sigma_px)
    vicinity = smooth(notch.to(current.dtype), sigma_px) > 0.0
    carved = torch.where(vicinity, torch.minimum(current, ramp), current)
    return carved, bool((carved != current).any())


def breach_depressions(
    cfg: MapConfig,
    height: torch.Tensor,
    level_m: float | None = None,
    params: WaterParams = WaterParams(),
    rounds: int = _BREACH_ROUNDS,
    smooth_m: float = _BREACH_SMOOTH_M,
    notch_m: float = _BREACH_NOTCH_M,
    reach_m: float = _BREACH_REACH_M,
    min_area_m2: float = _BREACH_MIN_AREA_M2,
) -> torch.Tensor:
    """Cut a single spillway through each closed basin's rim, toward the lowest ground it can reach.

    Real drainage divides get incised by headward erosion over geologic time the sim never runs;
    fill_depressions instead floods every closed basin to its rim, which is correct for the water
    channel but leaves the exported terrain itself full of lakes a river would have cut through.
    This is the dual: for every basin fill_depressions would flood, find the lowest ground its rim
    can reach through however much dry rim separates them (_lake_escape, backed by the bounded
    multi-hop search in _dry_reach), then lower only a narrow channel there - the crossing plus a
    small fixed-radius margin at the mouth, and exactly the chain of dry cells the search actually
    walked through the rim, not the rim's full width. The lake's own floor and the rest of its
    shore are left untouched; a real lake spills through one gap, it does not drain from the
    middle, and fill_depressions recomputing afterwards is what turns a single lowered channel into
    a smaller, correctly-shaped remaining lake.
    A hard cutover to the target right at the channel draws the same axis-aligned staircase a raw
    4-connected relaxation always does - fixed once already for the flat-lake routing potential in
    flat_gradient, and it recurs here because this carves real elevation, not a potential, straight
    into the exported terrain. Blurring the target with the same box-blur `smooth` uses elsewhere
    turns that cliff into a rounded, geologic-looking ramp; because the target now only differs
    from the current height inside the one channel per lake rather than across its whole footprint,
    that blur only ever touches the immediate vicinity of the channel, not the basin's interior or
    the rest of the rim. Taking the minimum against the current height afterwards keeps the blur
    from ever raising anything: only where the smoothed target undercuts the real ground does the
    ground move.
    A basin nested behind another rim, or genuinely enclosed by higher ground on every side within
    `reach_m` of its own rim, is left as fill_depressions would still treat it - a real endorheic
    lake can't be breached away, and a divide wider than `reach_m` is left for a later round to
    find a shorter way through, not forced open in one pass. A basin smaller than `min_area_m2` is
    left too, on purpose: a real eroded height field carries thousands of pixel-scale pits that are
    numerical noise rather than basins, each individually bounded by `reach_m` but so numerous that
    breaching every one of them still carves through most of the map's dry ground in aggregate, for
    puddles too small to move covered_fraction in the first place.
    Repeats to `rounds` passes, each re-reading fill_depressions so a basin lowered enough to join
    a neighbour lets that neighbour's own rim be the next thing considered; stops early once a pass
    changes nothing.
    """
    level = cfg.sea_level_m if level_m is None else level_m
    current = height
    for _ in range(rounds):
        current, changed = _breach_round(
            cfg, current, level, params, smooth_m, notch_m, reach_m, min_area_m2
        )
        if not changed:
            break
    return current


def _geodesic(seed: torch.Tensor, region: torch.Tensor, budget: int, stride: int) -> torch.Tensor:
    """Cell counts out from the seed through region alone, unreached cells left at the ceiling.

    Four-neighbour counts on purpose: their broad contours let the drainage fan across a flat, where a
    Euclidean distance would sharpen into single rays along its own medial axis.
    """
    far = float(region.numel())
    distance = torch.where(seed, torch.zeros_like(region, dtype=torch.float32), far)
    for step in range(budget):
        low = _lowest(distance).add_(1.0).clamp_(max=far)
        following = torch.where(region, torch.minimum(distance, low), distance)
        if step % stride == stride - 1 and bool(torch.equal(following, distance)):
            return following
        distance = following
    return distance


def flat_gradient(
    cfg: MapConfig,
    filled: torch.Tensor,
    level_m: float | None = None,
    params: WaterParams = WaterParams(),
) -> torch.Tensor:
    """Routing potential over the surfaces the fill left flat: off their walls, towards their spill.

    Zero wherever the filled height already drains. It is carried beside the height rather than
    added to it, so a step between two flat cells stays exactly representable instead of rounding
    away.
    """
    level = cfg.sea_level_m if level_m is None else level_m
    outlet = _outlets(filled, level / cfg.height_range_m)
    budget, _ = params.passes(cfg.resolution)
    flat = (_lowest(filled) >= filled) & ~outlet
    if not bool(flat.any()):
        return torch.zeros_like(filled)
    neighbours = _neighbourhood(filled)
    spill = (_neighbourhood(~flat) & (neighbours <= filled)).any(0)
    wall = _highest(filled) > filled
    to_spill = _geodesic(flat & spill, flat, budget, params.convergence_stride)
    from_wall = _geodesic(flat & wall, flat, budget, params.convergence_stride)
    reached = flat & (to_spill < float(filled.numel()))
    ceiling = float(from_wall.mul(reached).max())
    guide = to_spill.add_(1.0).mul_(_LOWER_WEIGHT).add_(ceiling - from_wall.clamp_(max=ceiling))
    return guide.mul_(reached)


def flow_accumulation(
    cfg: MapConfig,
    filled: torch.Tensor,
    params: WaterParams = WaterParams(),
    guide: torch.Tensor | None = None,
    level_m: float | None = None,
) -> torch.Tensor:
    """Multiple-flow-direction drainage area in cells, routed downhill over a depressionless height."""
    guide = flat_gradient(cfg, filled, level_m, params) if guide is None else guide
    neighbours = _neighbourhood(filled)
    drops = filled.sub(neighbours).clamp_(min=0.0)
    across = guide.sub(_neighbourhood(guide)).clamp_(min=0.0).mul_(neighbours <= filled)
    drops = torch.where(guide > 0.0, across, drops)
    drops.pow_(params.flow_exponent)
    total = drops.sum(0)
    share = drops.div_(total.clamp(min=_EPS)).mul_(total.gt(_EPS))
    _, budget = params.passes(cfg.resolution)
    accumulated = torch.ones_like(filled)
    parcel = torch.ones_like(filled)
    floor = _ACCUMULATION_FLOOR * parcel.numel()
    for step in range(budget):
        parcel = _collect(share.mul(parcel))
        accumulated.add_(parcel)
        settled = step % params.convergence_stride == params.convergence_stride - 1
        if settled and float(parcel.sum()) < floor:
            break
    return accumulated


@dataclass(frozen=True)
class Drainage:
    """One depression fill and one accumulation pass, shared by the flow, wetness and water channels."""

    level_m: float
    filled: torch.Tensor
    guide: torch.Tensor
    accumulated: torch.Tensor
    flow: torch.Tensor


def drainage(
    cfg: MapConfig,
    height: torch.Tensor,
    params: WaterParams = WaterParams(),
) -> Drainage:
    """Fill the pits, route the drainage area downhill, and normalise it onto [0, 1]."""
    if tuple(height.shape) != cfg.shape:
        raise MapConfigError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    level = sea_level_m(cfg, height, params)
    filled = fill_depressions(cfg, height, level, params)
    guide = flat_gradient(cfg, filled, level, params)
    accumulated = flow_accumulation(cfg, filled, params, guide, level)
    return Drainage(
        level_m=level,
        filled=filled,
        guide=guide,
        accumulated=accumulated,
        flow=log_normalise(accumulated),
    )


def river_depth_m(
    cfg: MapConfig,
    accumulated: torch.Tensor,
    params: WaterParams = WaterParams(),
    level: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel depth in metres, cut where the drainage area says a river runs."""
    if params.river_depth_m == 0.0:
        return torch.zeros_like(accumulated)
    level = log_normalise(accumulated) if level is None else level
    low = quantile(level, 1.0 - params.river_area_fraction)
    high = quantile(level, 1.0 - params.river_core_fraction)
    if high - low <= _EPS:
        return torch.zeros_like(level)
    depth = synth.smoothstep(low, high, level)
    depth.pow_(params.river_depth_exponent).mul_(params.river_depth_m)
    radius = int(round(0.5 * params.river_width_m / cfg.metres_per_pixel))
    if radius < 1:
        return depth
    return smooth(dilate(depth, radius), 0.5 * radius)


def water_depth_metres(
    cfg: MapConfig,
    height: torch.Tensor,
    standing_m: torch.Tensor | None = None,
    params: WaterParams = WaterParams(),
    basin: Drainage | None = None,
) -> torch.Tensor:
    """Depth of sea, lakes and rivers in metres, over the eroded height."""
    basin = drainage(cfg, height, params) if basin is None else basin
    depth = basin.filled.sub(height).clamp_(min=0.0).mul_(cfg.height_range_m)
    carved = river_depth_m(cfg, basin.accumulated, params, basin.flow)
    depth = torch.maximum(depth, carved)
    if standing_m is not None:
        depth = torch.maximum(depth, standing_m.clamp(min=0.0))
    terrain_m = height.mul(cfg.height_range_m)
    surface = terrain_m.add(depth).clamp_(min=basin.level_m)
    if params.surface_smooth_m > 0.0:
        surface = smooth(surface, params.surface_smooth_m / cfg.metres_per_pixel)
    return surface.sub_(terrain_m).clamp_(min=0.0)


def water_mask(depth_m: torch.Tensor, params: WaterParams = WaterParams()) -> torch.Tensor:
    """The water channel: 0 dry ground, 1 water deep enough to read as a surface."""
    return synth.smoothstep(params.shore_depth_m, params.full_depth_m, depth_m)


def water_level(depth_m: torch.Tensor, level_m: float, params: WaterParams) -> WaterLevel:
    covered = water_mask(depth_m, params)
    return WaterLevel(
        sea_level_m=level_m,
        covered_fraction=float(covered.mean()),
        mean_depth_m=float(depth_m.mean()),
        max_depth_m=float(depth_m.max()),
        depth_reference_m=params.depth_reference_m,
    )


def water_channels(
    cfg: MapConfig,
    height: torch.Tensor,
    standing_m: torch.Tensor | None = None,
    params: WaterParams = WaterParams(),
    basin: Drainage | None = None,
) -> dict[str, torch.Tensor]:
    """The water mask and its depth, both in [0, 1], depth scaled by depth_reference_m."""
    depth = water_depth_metres(cfg, height, standing_m, params, basin)
    return {
        "water": water_mask(depth, params),
        "water_depth": depth.div(params.depth_reference_m).clamp_(0.0, 1.0),
    }


def geometric_channels(
    cfg: MapConfig,
    height: torch.Tensor,
    params: GeometryParams = GeometryParams(),
    device: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Slope, curvature, bedrock and strata derived from an eroded height field."""
    if tuple(height.shape) != cfg.shape:
        raise MapConfigError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    target = _target_device(device, height)
    field = height.to(device=target, dtype=torch.float32)
    return {
        "slope": slope(cfg, field),
        "curvature": curvature(cfg, field, params),
        "bedrock": bedrock(cfg, field, params),
        "strata": strata(cfg, field, params, target),
    }


def erosion_channels(
    cfg: MapConfig,
    result: ErosionResult,
    params: GeometryParams = GeometryParams(),
    water: WaterParams = WaterParams(),
    basin: Drainage | None = None,
) -> dict[str, torch.Tensor]:
    """Flow, deposition, wear and wetness, beside the height and the reconstructed water surface."""
    basin = drainage(cfg, result.height, water) if basin is None else basin
    channels = dict(result.channels())
    channels["flow"] = basin.flow
    channels["wetness"] = wetness(cfg, basin.accumulated, result.height, params.wetness_min_tilt)
    channels.update(water_channels(cfg, result.height, result.water_depth_m, water, basin))
    return channels


@dataclass(frozen=True)
class ClimateParams:
    """Settings for the channels that vary across the map without reading the slope."""

    sea_level_temperature_c: float = 22.0
    lapse_rate_c_per_km: float = 6.5
    latitude_gradient_c: float = 9.0
    temperature_noise_c: float = 2.0
    temperature_feature_size_m: float = 3200.0
    temperature_min_c: float = -25.0
    temperature_max_c: float = 40.0
    moisture_feature_size_m: float = 2600.0
    moisture_warp_m: float = 700.0
    moisture_octaves: int = 4
    moisture_wetness_weight: float = 0.3
    moisture_wetness_smooth_m: float = 400.0
    patchiness_fine_m: float = 24.0
    patchiness_mid_m: float = 110.0
    patchiness_coarse_m: float = 780.0
    patchiness_octaves: int = 3
    patchiness_cell_weight: float = 0.5
    patchiness_warp_ratio: float = 0.35
    patchiness_warp_octaves: int = 2

    def __post_init__(self) -> None:
        for name in (
            "temperature_feature_size_m",
            "moisture_feature_size_m",
            "patchiness_fine_m",
            "patchiness_mid_m",
            "patchiness_coarse_m",
        ):
            value = getattr(self, name)
            if value <= 0.0:
                raise MapConfigError(f"{name} {value} must be positive")
        if self.temperature_max_c <= self.temperature_min_c:
            raise MapConfigError(
                f"temperature_max_c {self.temperature_max_c} must exceed "
                f"temperature_min_c {self.temperature_min_c}"
            )
        if self.temperature_noise_c < 0.0:
            raise MapConfigError(
                f"temperature_noise_c {self.temperature_noise_c} must not be negative"
            )
        if self.moisture_warp_m < 0.0:
            raise MapConfigError(f"moisture_warp_m {self.moisture_warp_m} must not be negative")
        if self.moisture_octaves < 1:
            raise MapConfigError(f"moisture_octaves {self.moisture_octaves} must be at least 1")
        if self.patchiness_octaves < 1:
            raise MapConfigError(
                f"patchiness_octaves {self.patchiness_octaves} must be at least 1"
            )
        if self.moisture_wetness_smooth_m < 0.0:
            raise MapConfigError(
                f"moisture_wetness_smooth_m {self.moisture_wetness_smooth_m} must not be negative"
            )
        if not 0.0 <= self.moisture_wetness_weight <= 1.0:
            raise MapConfigError(
                f"moisture_wetness_weight {self.moisture_wetness_weight} must lie within [0, 1]"
            )
        if not 0.0 <= self.patchiness_cell_weight <= 1.0:
            raise MapConfigError(
                f"patchiness_cell_weight {self.patchiness_cell_weight} must lie within [0, 1]"
            )
        if self.patchiness_warp_ratio < 0.0:
            raise MapConfigError(
                f"patchiness_warp_ratio {self.patchiness_warp_ratio} must not be negative"
            )
        if self.patchiness_warp_octaves < 1:
            raise MapConfigError(
                f"patchiness_warp_octaves {self.patchiness_warp_octaves} must be at least 1"
            )


def temperature_c(
    cfg: MapConfig,
    height: torch.Tensor,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> torch.Tensor:
    """Air temperature in degrees Celsius: altitude lapse, a latitude gradient and weather noise."""
    target = _target_device(device, height)
    field = height.to(device=target, dtype=torch.float32)
    elevation_km = field.mul(cfg.height_range_m).sub_(cfg.sea_level_m).div_(1000.0)
    celsius = elevation_km.mul_(-params.lapse_rate_c_per_km).add_(params.sea_level_temperature_c)
    _, y = synth.grid(cfg, target)
    celsius.add_(
        y.div(cfg.world_size_m).sub_(0.5), alpha=-params.latitude_gradient_c
    )
    if params.temperature_noise_c > 0.0:
        weather = synth.warped_fbm(
            cfg,
            synth.FractalParams(params.temperature_feature_size_m, 3),
            warp_strength_m=params.temperature_feature_size_m * 0.2,
            label="temperature",
            device=target,
        )
        celsius.add_(weather.sub_(0.5), alpha=2.0 * params.temperature_noise_c)
    return celsius


def temperature(
    cfg: MapConfig,
    height: torch.Tensor,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> torch.Tensor:
    """The temperature channel: temperature_min_c at 0, temperature_max_c at 1."""
    span = params.temperature_max_c - params.temperature_min_c
    return temperature_c(cfg, height, params, device).sub_(params.temperature_min_c).div_(
        span
    ).clamp_(0.0, 1.0)


def moisture(
    cfg: MapConfig,
    wetness_field: torch.Tensor | None = None,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> torch.Tensor:
    """The moisture channel: domain-warped low-frequency noise pulled toward the wet ground."""
    target = as_torch_device(device) if wetness_field is None else _target_device(device, wetness_field)
    field = synth.normalise(
        synth.warped_fbm(
            cfg,
            synth.FractalParams(params.moisture_feature_size_m, params.moisture_octaves),
            warp_strength_m=params.moisture_warp_m,
            label="moisture",
            device=target,
        )
    )
    if wetness_field is None or params.moisture_wetness_weight == 0.0:
        return field.clamp_(0.0, 1.0)
    wet = wetness_field.to(device=target, dtype=torch.float32)
    if params.moisture_wetness_smooth_m > 0.0:
        wet = smooth(wet, params.moisture_wetness_smooth_m / cfg.metres_per_pixel)
    return torch.lerp(field, synth.normalise(wet), params.moisture_wetness_weight).clamp_(0.0, 1.0)


def patchiness(
    cfg: MapConfig,
    feature_size_m: float,
    label: str,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> torch.Tensor:
    """One patchiness scale: two worley octaves mixed with domain-warped fBm of the same size."""
    size = max(feature_size_m, 2.0 * cfg.metres_per_pixel)
    warp = size * params.patchiness_warp_ratio
    warp_octaves = params.patchiness_warp_octaves
    cells = synth.cellular(
        cfg, size, "f1", warp, f"{label}_cells", device, warp_octaves=warp_octaves
    )
    cells.add_(
        synth.cellular(
            cfg, 0.5 * size, "f1", 0.5 * warp, f"{label}_cells_fine", device,
            warp_octaves=warp_octaves,
        ),
        alpha=0.5,
    )
    grain = synth.warped_fbm(
        cfg,
        synth.FractalParams(size, params.patchiness_octaves),
        warp_strength_m=warp,
        warp_octaves=warp_octaves,
        label=label,
        device=device,
    )
    return torch.lerp(
        synth.normalise(grain), synth.normalise(cells), params.patchiness_cell_weight
    ).clamp_(0.0, 1.0)


def climate_channels(
    cfg: MapConfig,
    height: torch.Tensor,
    wetness_field: torch.Tensor | None = None,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Temperature and moisture, the two channels a biome rule reads before anything else."""
    if tuple(height.shape) != cfg.shape:
        raise MapConfigError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    target = _target_device(device, height)
    return {
        "temperature": temperature(cfg, height, params, target),
        "moisture": moisture(cfg, wetness_field, params, target),
    }


def patchiness_channels(
    cfg: MapConfig,
    params: ClimateParams = ClimateParams(),
    device: Any | None = None,
) -> dict[str, torch.Tensor]:
    """The three slope-free variation scales that break up flat ground."""
    target = as_torch_device(device)
    sizes = {
        "patchiness_fine": params.patchiness_fine_m,
        "patchiness_mid": params.patchiness_mid_m,
        "patchiness_coarse": params.patchiness_coarse_m,
    }
    return {
        name: patchiness(cfg, size, name, params, target) for name, size in sizes.items()
    }


class ChannelStack:
    """The shared float32[H, W, C] stack, addressed by the names in config.CHANNEL_NAMES."""

    def __init__(
        self,
        cfg: MapConfig,
        device: Any | None = None,
        names: Iterable[str] = CHANNEL_NAMES,
    ) -> None:
        ordered = tuple(names)
        unknown = [name for name in ordered if name not in CHANNEL_INDEX]
        if unknown:
            raise MapConfigError(f"unknown channels {', '.join(sorted(unknown))}")
        if len(set(ordered)) != len(ordered):
            raise MapConfigError("channel names must not repeat")
        self.cfg = cfg
        self.names = ordered
        self.index = {name: position for position, name in enumerate(ordered)}
        self.data = torch.zeros(
            (cfg.resolution, cfg.resolution, len(ordered)),
            device=as_torch_device(device),
            dtype=torch.float32,
        )
        self._filled: set[str] = set()

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._filled

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.data[:, :, self._position(name)]

    def __setitem__(self, name: str, field: torch.Tensor) -> None:
        position = self._position(name)
        if tuple(field.shape) != self.cfg.shape:
            raise MapConfigError(
                f"channel {name} has shape {tuple(field.shape)}, expected {self.cfg.shape}"
            )
        low = float(field.min())
        high = float(field.max())
        if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
            raise MapConfigError(f"channel {name} spans [{low:.4f}, {high:.4f}] outside [0, 1]")
        moved = field.to(device=self.data.device, dtype=torch.float32)
        self.data[:, :, position] = moved.clamp(0.0, 1.0)
        self._filled.add(name)

    def _position(self, name: str) -> int:
        try:
            return self.index[name]
        except KeyError:
            raise MapConfigError(
                f"unknown channel {name!r}; expected one of {', '.join(self.names)}"
            ) from None

    def update(self, channels: Mapping[str, torch.Tensor]) -> ChannelStack:
        for name, field in channels.items():
            self[name] = field
        return self

    def filled(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if name in self._filled)

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if name not in self._filled)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {name: self[name] for name in self.filled()}

    def to(self, device: Any | None) -> ChannelStack:
        moved = ChannelStack(self.cfg, device, self.names)
        moved.data = self.data.to(device=as_torch_device(device))
        moved._filled = set(self._filled)
        return moved

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


def build(
    cfg: MapConfig,
    result: ErosionResult,
    params: GeometryParams = GeometryParams(),
    device: Any | None = None,
    water: WaterParams = WaterParams(),
    climate: ClimateParams = ClimateParams(),
    basin: Drainage | None = None,
) -> ChannelStack:
    """Fill every channel an eroded height field can supply on its own."""
    target = _target_device(device, result.height)
    stack = ChannelStack(cfg, target)
    stack.update(erosion_channels(cfg, result, params, water, basin))
    stack.update(geometric_channels(cfg, result.height, params, target))
    stack.update(climate_channels(cfg, result.height, stack["wetness"], climate, target))
    stack.update(patchiness_channels(cfg, climate, target))
    return stack


def summarise_water(
    cfg: MapConfig,
    result: ErosionResult,
    water: WaterParams = WaterParams(),
    basin: Drainage | None = None,
) -> WaterLevel:
    """The water level and coverage the build used, for terrain.json and for the preview caption."""
    basin = drainage(cfg, result.height, water) if basin is None else basin
    depth = water_depth_metres(cfg, result.height, result.water_depth_m, water, basin)
    return water_level(depth, basin.level_m, water)
