from __future__ import annotations

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
