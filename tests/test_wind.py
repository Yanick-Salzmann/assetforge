from __future__ import annotations

import pytest

from terrain import wind


def test_parse_fills_omitted_keys_from_defaults():
    parsed = wind.parse({"strength": 2})
    assert parsed.strength == 2.0
    assert parsed.from_deg == wind.Wind().from_deg


def test_as_dict_round_trips_through_validate():
    recorded = wind.Wind(from_deg=10.0, strength=0.5, gust_period_s=3.0).as_dict()
    wind.validate(recorded)
    assert wind.parse(recorded) == wind.Wind(from_deg=10.0, strength=0.5, gust_period_s=3.0)


@pytest.mark.parametrize(
    ("from_deg", "downwind"),
    [(0.0, (0.0, -1.0)), (90.0, (-1.0, 0.0)), (180.0, (0.0, 1.0)), (270.0, (1.0, 0.0))],
)
def test_downwind_points_away_from_the_bearing_the_wind_comes_from(from_deg, downwind):
    assert wind.Wind(from_deg=from_deg).downwind() == pytest.approx(downwind, abs=1e-12)


@pytest.mark.parametrize(
    "table",
    [
        {"from_deg": -1.0},
        {"from_deg": 361.0},
        {"strength": 3.5},
        {"gust_period_s": 0.5},
        {"strength": "gale"},
        {"strength": True},
        {"gusts": 1.0},
    ],
)
def test_parse_rejects_malformed_tables(table):
    with pytest.raises(wind.WindError):
        wind.parse(table)


def test_validate_requires_every_key():
    recorded = wind.Wind().as_dict()
    recorded.pop("gust_period_s")
    with pytest.raises(wind.WindError):
        wind.validate(recorded)
