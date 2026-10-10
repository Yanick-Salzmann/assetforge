from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from terrain.config import MapConfigError

ATMOSPHERE_KEYS = (
    "latitude_deg",
    "day_of_year",
    "time_of_day_h",
    "turbidity",
    "rayleigh",
    "visibility_km",
    "haze_colour",
    "ground_albedo",
)
NUMBER_RANGES = {
    "latitude_deg": (-89.0, 89.0),
    "time_of_day_h": (0.0, 24.0),
    "turbidity": (1.0, 20.0),
    "rayleigh": (0.0, 4.0),
    "visibility_km": (1.0, 300.0),
}
DAY_RANGE = (1, 365)
AXIAL_TILT_DEG = 23.44
COLOUR_KEYS = ("haze_colour", "ground_albedo")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class AtmosphereError(MapConfigError):
    """Raised when a biome [atmosphere] table or a terrain.json atmosphere entry is malformed."""


@dataclass(frozen=True)
class Atmosphere:
    """How a biome's sky is lit: where the sun runs, how hazy the air is, what the ground bounces."""

    latitude_deg: float = 45.0
    day_of_year: int = 172
    time_of_day_h: float = 10.0
    turbidity: float = 3.0
    rayleigh: float = 1.5
    visibility_km: float = 30.0
    haze_colour: str = "#c8d2dc"
    ground_albedo: str = "#6b6a55"

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ATMOSPHERE_KEYS}


def _number(value: Any, key: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AtmosphereError(f"{label} {key} must be a number")
    low, high = NUMBER_RANGES[key]
    number = float(value)
    if not low <= number <= high:
        raise AtmosphereError(f"{label} {key} {number} lies outside [{low}, {high}]")
    return number


def _day(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AtmosphereError(f"{label} day_of_year must be an integer")
    low, high = DAY_RANGE
    if not low <= value <= high:
        raise AtmosphereError(f"{label} day_of_year {value} lies outside [{low}, {high}]")
    return value


def _colour(value: Any, key: str, label: str) -> str:
    if not isinstance(value, str) or not _HEX.match(value):
        raise AtmosphereError(f"{label} {key} must be a '#rrggbb' colour")
    return value.lower()


def parse(table: Mapping[str, Any], label: str = "[atmosphere]") -> Atmosphere:
    """Read an [atmosphere] table, filling every omitted key from the defaults."""
    if not isinstance(table, Mapping):
        raise AtmosphereError(f"{label} must be a table")
    unknown = sorted(set(table) - set(ATMOSPHERE_KEYS))
    if unknown:
        raise AtmosphereError(
            f"{label} carries unknown keys {', '.join(unknown)}; expected {', '.join(ATMOSPHERE_KEYS)}"
        )
    values: dict[str, Any] = {}
    for key, value in table.items():
        if key == "day_of_year":
            values[key] = _day(value, label)
        elif key in COLOUR_KEYS:
            values[key] = _colour(value, key, label)
        else:
            values[key] = _number(value, key, label)
    return Atmosphere(**values)


def validate(payload: Any, label: str = "atmosphere") -> None:
    """Check a recorded atmosphere entry: every key present and in range."""
    if not isinstance(payload, Mapping):
        raise AtmosphereError(f"{label} must be a table")
    missing = [key for key in ATMOSPHERE_KEYS if key not in payload]
    if missing:
        raise AtmosphereError(f"{label} is missing {', '.join(missing)}")
    parse(payload, label)


@dataclass(frozen=True)
class CelestialPosition:
    """Where a body sits in the sky: elevation above the horizon and compass azimuth, clockwise from north."""

    elevation_rad: float
    azimuth_rad: float

    def direction(self) -> tuple[float, float, float]:
        """Unit vector toward the body in map space: +x east (image columns), +y north (image top), +z up."""
        flat = math.cos(self.elevation_rad)
        return (
            math.sin(self.azimuth_rad) * flat,
            math.cos(self.azimuth_rad) * flat,
            math.sin(self.elevation_rad),
        )


def declination(day_of_year: int) -> float:
    return math.radians(-AXIAL_TILT_DEG) * math.cos(2.0 * math.pi / 365.0 * (day_of_year + 10))


def celestial_position(latitude_deg: float, declination_rad: float, hours: float) -> CelestialPosition:
    """The same local-solar-time model the viewer's sky.js uses, so both renders agree on the sky."""
    latitude = math.radians(latitude_deg)
    hour_angle = math.radians(15.0 * (hours - 12.0))
    sin_elevation = math.sin(latitude) * math.sin(declination_rad) + math.cos(latitude) * math.cos(
        declination_rad
    ) * math.cos(hour_angle)
    elevation = math.asin(max(-1.0, min(1.0, sin_elevation)))
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(declination_rad) * math.cos(latitude) - math.sin(latitude) * math.cos(hour_angle),
    )
    return CelestialPosition(elevation, azimuth % (2.0 * math.pi))


def sun_position(atmosphere: Atmosphere) -> CelestialPosition:
    return celestial_position(
        atmosphere.latitude_deg, declination(atmosphere.day_of_year), atmosphere.time_of_day_h
    )


def moon_position(atmosphere: Atmosphere) -> CelestialPosition:
    """A full moon: opposite the sun in declination and twelve hours round."""
    return celestial_position(
        atmosphere.latitude_deg, -declination(atmosphere.day_of_year), atmosphere.time_of_day_h + 12.0
    )
