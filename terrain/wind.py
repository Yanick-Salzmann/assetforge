from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from terrain.config import MapConfigError

WIND_KEYS = ("from_deg", "strength", "gust_period_s")
NUMBER_RANGES = {
    "from_deg": (0.0, 360.0),
    "strength": (0.0, 3.0),
    "gust_period_s": (1.0, 60.0),
}


class WindError(MapConfigError):
    """Raised when a biome [wind] table or a terrain.json wind entry is malformed."""


@dataclass(frozen=True)
class Wind:
    """The prevailing wind over a map: the compass bearing it blows from (clockwise from north,
    map +y), its strength relative to the clips assets are authored at (1 plays them as keyed),
    and the period of its gusts."""

    from_deg: float = 225.0
    strength: float = 1.0
    gust_period_s: float = 6.0

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in WIND_KEYS}

    def downwind(self) -> tuple[float, float]:
        """Unit vector the wind blows toward in map space: +x east (image columns), +y north (image top)."""
        bearing = math.radians(self.from_deg)
        return (-math.sin(bearing), -math.cos(bearing))


def _number(value: Any, key: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindError(f"{label} {key} must be a number")
    low, high = NUMBER_RANGES[key]
    number = float(value)
    if not low <= number <= high:
        raise WindError(f"{label} {key} {number} lies outside [{low}, {high}]")
    return number


def parse(table: Mapping[str, Any], label: str = "[wind]") -> Wind:
    """Read a [wind] table, filling every omitted key from the defaults."""
    if not isinstance(table, Mapping):
        raise WindError(f"{label} must be a table")
    unknown = sorted(set(table) - set(WIND_KEYS))
    if unknown:
        raise WindError(f"{label} carries unknown keys {', '.join(unknown)}; expected {', '.join(WIND_KEYS)}")
    return Wind(**{key: _number(value, key, label) for key, value in table.items()})


def validate(payload: Any, label: str = "wind") -> None:
    """Check a recorded wind entry: every key present and in range."""
    if not isinstance(payload, Mapping):
        raise WindError(f"{label} must be a table")
    missing = [key for key in WIND_KEYS if key not in payload]
    if missing:
        raise WindError(f"{label} is missing {', '.join(missing)}")
    parse(payload, label)
