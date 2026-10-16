"""Pure motion primitives for rigid loop animation. Stdlib only, so Blender's interpreter can import it.

Every primitive samples one value per frame (an angle in radians or an offset in metres) over a
clip of `duration_s` at `fps`, inclusive of both ends, so a clip of N frames carries N + 1 samples
and the last sample closes the loop: it equals the first, or differs from it by whole turns for a
`Track` with `wrap` set. The duration must be a whole number of frames and a whole number of every
component period; `loop_duration` finds the shortest such clip for a set of periods.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

TAU = 2.0 * math.pi
GRAVITY_M_S2 = 9.80665
FRAME_TOLERANCE = 1e-6
PERIOD_DENOMINATOR_LIMIT = 10_000
WIND_WAVE_PERIOD_S = 1.2
FLUTTER_HARMONIC = 3
WIND_WAVELENGTH_CHAINS = 1.5


class MotionError(ValueError):
    pass


@dataclass(frozen=True)
class Track:
    fps: int
    values: tuple[float, ...]
    wrap: float = 0.0

    @property
    def frame_count(self) -> int:
        return len(self.values) - 1

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def seam_error(self) -> float:
        drift = self.values[-1] - self.values[0]
        if self.wrap:
            drift = math.remainder(drift, self.wrap)
        return abs(drift)


@dataclass(frozen=True)
class CrankSlider:
    crank: Track
    slider: Track
    rod: Track


def frame_count(duration_s: float, fps: int) -> int:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise MotionError(f"fps must be a positive integer, got {fps!r}")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise MotionError(f"duration_s must be positive, got {duration_s!r}")
    frames = duration_s * fps
    if abs(frames - round(frames)) > FRAME_TOLERANCE * max(1.0, frames):
        raise MotionError(f"duration_s {duration_s} is not a whole number of frames at {fps} fps")
    return round(frames)


def cycles(duration_s: float, period_s: float) -> int:
    if not math.isfinite(period_s) or period_s <= 0:
        raise MotionError(f"period_s must be positive, got {period_s!r}")
    ratio = duration_s / period_s
    count = round(ratio)
    if count < 1 or abs(ratio - count) > FRAME_TOLERANCE * max(1.0, ratio):
        raise MotionError(f"duration_s {duration_s} is not a whole number of {period_s} s periods")
    return count


def loop_duration(periods_s: Sequence[float], fps: int) -> float:
    if not periods_s:
        raise MotionError("loop_duration needs at least one period")
    fractions = [Fraction(1, fps)]
    for period in periods_s:
        if not math.isfinite(period) or period <= 0:
            raise MotionError(f"period_s must be positive, got {period!r}")
        fractions.append(Fraction(period).limit_denominator(PERIOD_DENOMINATOR_LIMIT))
    numerator = 0
    denominator = 0
    for fraction in fractions:
        numerator = fraction.numerator if numerator == 0 else math.lcm(numerator, fraction.numerator)
        denominator = fraction.denominator if denominator == 0 else math.gcd(denominator, fraction.denominator)
    return numerator / denominator


def sample(duration_s: float, fps: int, phase_of: Callable[[float], float], wrap: float = 0.0) -> Track:
    frames = frame_count(duration_s, fps)
    values = tuple(phase_of(frame / fps) for frame in range(frames + 1))
    return Track(fps=fps, values=values, wrap=wrap)


def _cycle_fraction(time_s: float, period_s: float) -> float:
    fraction = time_s / period_s
    nearest = round(fraction)
    if abs(fraction - nearest) <= FRAME_TOLERANCE:
        return 0.0
    return fraction - math.floor(fraction)


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * u)


def spin(period_s: float, duration_s: float, fps: int, clockwise: bool = False, phase: float = 0.0) -> Track:
    turns = cycles(duration_s, period_s)
    frames = frame_count(duration_s, fps)
    sign = -1.0 if clockwise else 1.0
    step = sign * TAU * turns / frames
    values = tuple(phase + step * frame for frame in range(frames + 1))
    return Track(fps=fps, values=values, wrap=TAU)


def oscillate(
    amplitude: float,
    period_s: float,
    duration_s: float,
    fps: int,
    phase: float = 0.0,
    offset: float = 0.0,
) -> Track:
    cycles(duration_s, period_s)
    return sample(duration_s, fps, lambda t: offset + amplitude * math.sin(TAU * _cycle_fraction(t, period_s) + phase))


def pendulum_period(length_m: float) -> float:
    if length_m <= 0:
        raise MotionError(f"length_m must be positive, got {length_m!r}")
    return TAU * math.sqrt(length_m / GRAVITY_M_S2)


def pendulum(amplitude: float, period_s: float, duration_s: float, fps: int, phase: float = 0.0) -> Track:
    cycles(duration_s, period_s)
    return sample(duration_s, fps, lambda t: amplitude * math.cos(TAU * _cycle_fraction(t, period_s) + phase))


def hinge_swing(
    open_angle: float,
    period_s: float,
    duration_s: float,
    fps: int,
    hold_fraction: float = 0.25,
) -> Track:
    if not 0.0 <= hold_fraction < 0.5:
        raise MotionError(f"hold_fraction must be in [0, 0.5), got {hold_fraction!r}")
    cycles(duration_s, period_s)
    swing = 0.5 - hold_fraction

    def angle(time_s: float) -> float:
        u = _cycle_fraction(time_s, period_s)
        if u < swing:
            return open_angle * _ease(u / swing)
        if u < swing + hold_fraction:
            return open_angle
        if u < 2.0 * swing + hold_fraction:
            return open_angle * (1.0 - _ease((u - swing - hold_fraction) / swing))
        return 0.0

    return sample(duration_s, fps, angle)


def crank_slider(
    crank_radius_m: float,
    rod_length_m: float,
    period_s: float,
    duration_s: float,
    fps: int,
    phase: float = 0.0,
) -> CrankSlider:
    if crank_radius_m <= 0 or rod_length_m <= crank_radius_m:
        raise MotionError("crank-slider needs 0 < crank_radius_m < rod_length_m")
    crank = spin(period_s, duration_s, fps, phase=phase)
    sliders = []
    rods = []
    for theta in crank.values:
        pin_x = crank_radius_m * math.cos(theta)
        pin_y = crank_radius_m * math.sin(theta)
        reach = math.sqrt(rod_length_m * rod_length_m - pin_y * pin_y)
        sliders.append(pin_x + reach)
        rods.append(math.atan2(-pin_y, reach))
    return CrankSlider(
        crank=crank,
        slider=Track(fps=fps, values=tuple(sliders)),
        rod=Track(fps=fps, values=tuple(rods)),
    )


def cam_follower(
    lift: float,
    period_s: float,
    duration_s: float,
    fps: int,
    lobes: int = 1,
    rise_fraction: float = 0.8,
) -> Track:
    if isinstance(lobes, bool) or not isinstance(lobes, int) or lobes < 1:
        raise MotionError(f"lobes must be a positive integer, got {lobes!r}")
    if not 0.0 < rise_fraction < 1.0:
        raise MotionError(f"rise_fraction must be in (0, 1), got {rise_fraction!r}")
    cycles(duration_s, period_s)
    lobe_period = period_s / lobes

    def height(time_s: float) -> float:
        u = _cycle_fraction(time_s, lobe_period)
        if u < rise_fraction:
            return lift * _ease(u / rise_fraction)
        fall = (u - rise_fraction) / (1.0 - rise_fraction)
        return lift * (1.0 - fall * fall)

    return sample(duration_s, fps, height)


def gear_periods(teeth: Sequence[int], driver_period_s: float) -> list[float]:
    if len(teeth) < 1 or any(isinstance(count, bool) or not isinstance(count, int) or count < 3 for count in teeth):
        raise MotionError(f"teeth must be integers >= 3, got {list(teeth)!r}")
    return [driver_period_s * count / teeth[0] for count in teeth]


def gear_train_loop_duration(teeth: Sequence[int], driver_period_s: float, fps: int) -> float:
    return loop_duration(gear_periods(teeth, driver_period_s), fps)


def gear_train(
    teeth: Sequence[int],
    driver_period_s: float,
    duration_s: float,
    fps: int,
    clockwise: bool = False,
) -> list[Track]:
    tracks = []
    for index, period in enumerate(gear_periods(teeth, driver_period_s)):
        turns_clockwise = clockwise != (index % 2 == 1)
        tracks.append(spin(period, duration_s, fps, clockwise=turns_clockwise))
    return tracks


@dataclass(frozen=True)
class WindChain:
    bend: tuple[Track, ...]
    twist: tuple[Track, ...]


def wind_weights(bones: int, exponent: float = 1.5) -> list[float]:
    raw = [((index + 1) / bones) ** exponent for index in range(bones)]
    total = sum(raw)
    return [value / total for value in raw]


def wind_wave(
    bones: int,
    duration_s: float,
    fps: int,
    seed: int,
    strength: float = 1.0,
    gust_period_s: float | None = None,
    wavelength_bones: float | None = None,
    max_bend: float = 0.8,
    flutter: float = 0.35,
    gust_depth: float = 0.35,
) -> WindChain:
    """A travelling wave down a bone chain hanging from its first bone, plus a faster flutter
    twist, both under a slow gust envelope. Each bone bends by its share of max_bend * strength,
    the share growing towards the free end, so the cumulative tip deflection stays within
    max_bend * strength * (1 + gust_depth). Stronger wind runs more wave cycles per clip. Every
    period divides duration_s, so the clip closes exactly; phases come from seed alone."""
    if isinstance(bones, bool) or not isinstance(bones, int) or bones < 1:
        raise MotionError(f"bones must be a positive integer, got {bones!r}")
    if not math.isfinite(strength) or strength <= 0:
        raise MotionError(f"strength must be positive, got {strength!r}")
    if not 0.0 <= gust_depth < 1.0:
        raise MotionError(f"gust_depth must be in [0, 1), got {gust_depth!r}")
    frame_count(duration_s, fps)
    gust_cycles = cycles(duration_s, gust_period_s or duration_s)
    wave_cycles = max(1, round(duration_s * strength / WIND_WAVE_PERIOD_S))
    wavelength = wavelength_bones or WIND_WAVELENGTH_CHAINS * bones
    if not math.isfinite(wavelength) or wavelength <= 0:
        raise MotionError(f"wavelength_bones must be positive, got {wavelength_bones!r}")
    rng = random.Random(seed)
    wave_phase = rng.uniform(0.0, TAU)
    flutter_phase = rng.uniform(0.0, TAU)
    gust_phase = rng.uniform(0.0, TAU)

    def gust(time_s: float) -> float:
        return 1.0 + gust_depth * math.sin(TAU * gust_cycles * time_s / duration_s + gust_phase)

    bends = []
    twists = []
    for index, weight in enumerate(wind_weights(bones)):
        amplitude = max_bend * strength * weight
        lag = TAU * index / wavelength
        bends.append(
            sample(
                duration_s,
                fps,
                lambda t, a=amplitude, k=lag: a * gust(t) * math.sin(TAU * wave_cycles * t / duration_s - k + wave_phase),
            )
        )
        twists.append(
            sample(
                duration_s,
                fps,
                lambda t, a=amplitude * flutter, k=2.0 * lag: a
                * gust(t)
                * math.sin(TAU * FLUTTER_HARMONIC * wave_cycles * t / duration_s - k + flutter_phase),
            )
        )
    return WindChain(bend=tuple(bends), twist=tuple(twists))
