from __future__ import annotations

import pytest

from terrain import atmosphere


def test_parse_fills_omitted_keys_from_defaults():
    parsed = atmosphere.parse({"turbidity": 6})
    assert parsed.turbidity == 6.0
    assert parsed.latitude_deg == atmosphere.Atmosphere().latitude_deg


def test_parse_lowercases_colours():
    assert atmosphere.parse({"haze_colour": "#ABCDEF"}).haze_colour == "#abcdef"


def test_as_dict_round_trips_through_validate():
    recorded = atmosphere.Atmosphere(latitude_deg=-33.0, day_of_year=1, time_of_day_h=18.5).as_dict()
    atmosphere.validate(recorded)
    assert atmosphere.parse(recorded) == atmosphere.Atmosphere(latitude_deg=-33.0, day_of_year=1, time_of_day_h=18.5)


@pytest.mark.parametrize(
    "table",
    [
        {"latitude_deg": 95.0},
        {"time_of_day_h": 25.0},
        {"turbidity": 0.5},
        {"visibility_km": 0.0},
        {"rayleigh": "blue"},
        {"turbidity": True},
        {"day_of_year": 0},
        {"day_of_year": 10.5},
        {"haze_colour": "grey"},
        {"ground_albedo": "#12345"},
        {"cloud_cover": 0.5},
    ],
)
def test_parse_rejects_malformed_tables(table):
    with pytest.raises(atmosphere.AtmosphereError):
        atmosphere.parse(table)


def test_validate_requires_every_key():
    recorded = atmosphere.Atmosphere().as_dict()
    recorded.pop("rayleigh")
    with pytest.raises(atmosphere.AtmosphereError):
        atmosphere.validate(recorded)
