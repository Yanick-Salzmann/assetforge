from __future__ import annotations

import math

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


def test_sun_stands_due_south_at_solar_noon_in_the_north():
    sun = atmosphere.sun_position(atmosphere.Atmosphere(latitude_deg=45.0, day_of_year=172, time_of_day_h=12.0))
    assert math.degrees(sun.azimuth_rad) == pytest.approx(180.0)
    assert math.degrees(sun.elevation_rad) == pytest.approx(90.0 - 45.0 + 23.44, abs=0.1)


def test_sun_rises_in_the_east_and_sets_in_the_west():
    morning = atmosphere.sun_position(atmosphere.Atmosphere(time_of_day_h=7.0))
    evening = atmosphere.sun_position(atmosphere.Atmosphere(time_of_day_h=17.0))
    assert 0.0 < math.degrees(morning.azimuth_rad) < 180.0
    assert 180.0 < math.degrees(evening.azimuth_rad) < 360.0
    assert morning.direction()[0] > 0.0
    assert evening.direction()[0] < 0.0


def test_full_moon_mirrors_the_sun():
    settings = atmosphere.Atmosphere(time_of_day_h=23.0)
    sun = atmosphere.sun_position(settings)
    moon = atmosphere.moon_position(settings)
    assert sun.elevation_rad < 0.0
    assert moon.elevation_rad == pytest.approx(-sun.elevation_rad)


def test_direction_is_a_unit_vector_pointing_north_at_zero_azimuth():
    direction = atmosphere.CelestialPosition(elevation_rad=0.0, azimuth_rad=0.0).direction()
    assert direction == pytest.approx((0.0, 1.0, 0.0))
