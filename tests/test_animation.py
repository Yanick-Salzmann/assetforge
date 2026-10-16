from __future__ import annotations

import math

import pytest

from assets import animation as anim

FPS = 24
SEAM = 1e-9


def _velocity_seam(track: anim.Track) -> float:
    first = track.values[1] - track.values[0]
    last = track.values[-1] - track.values[-2]
    return abs(first - last)


def _seam_kink(track: anim.Track) -> float:
    values = track.values
    interior = max(abs(values[i + 1] - 2 * values[i] + values[i - 1]) for i in range(1, len(values) - 1))
    seam = abs(values[1] - 2 * values[0] + values[-2])
    return seam - interior


def test_frame_count_rejects_fractional_frames():
    assert anim.frame_count(2.0, FPS) == 48
    with pytest.raises(anim.MotionError):
        anim.frame_count(2.01, FPS)
    with pytest.raises(anim.MotionError):
        anim.frame_count(2.0, 0)


def test_cycles_rejects_partial_periods():
    assert anim.cycles(3.0, 1.5) == 2
    with pytest.raises(anim.MotionError):
        anim.cycles(3.0, 2.0)


def test_loop_duration_is_common_multiple_of_periods_and_frames():
    assert anim.loop_duration([1.5, 2.0], FPS) == pytest.approx(6.0)
    assert anim.loop_duration([0.3], FPS) == pytest.approx(1.5)
    duration = anim.loop_duration([0.75, 1.25, 2.0], 30)
    anim.frame_count(duration, 30)
    for period in (0.75, 1.25, 2.0):
        anim.cycles(duration, period)


def test_spin_closes_on_whole_turns():
    track = anim.spin(1.0, 3.0, FPS)
    assert track.frame_count == 72
    assert track.values[-1] - track.values[0] == pytest.approx(3 * anim.TAU)
    assert track.seam_error() < SEAM
    assert _velocity_seam(track) < SEAM


def test_spin_clockwise_runs_negative():
    track = anim.spin(2.0, 2.0, FPS, clockwise=True)
    assert track.values[-1] == pytest.approx(-anim.TAU)


@pytest.mark.parametrize(
    "track",
    [
        anim.oscillate(0.2, 1.5, 3.0, FPS, phase=0.4, offset=1.0),
        anim.pendulum(0.3, 2.0, 4.0, FPS),
        anim.hinge_swing(1.2, 2.0, 4.0, FPS),
        anim.hinge_swing(1.2, 2.0, 2.0, FPS, hold_fraction=0.0),
        anim.cam_follower(0.4, 2.0, 2.0, FPS, lobes=2),
    ],
)
def test_loop_seam_exact(track):
    assert track.seam_error() < SEAM


def test_smooth_primitives_have_continuous_velocity_across_the_seam():
    for track in (
        anim.oscillate(0.2, 1.5, 3.0, FPS),
        anim.pendulum(0.3, 2.0, 4.0, FPS),
        anim.hinge_swing(1.2, 2.0, 4.0, FPS),
    ):
        assert _seam_kink(track) < 1e-9


def test_hinge_swing_holds_open_and_closed():
    track = anim.hinge_swing(1.0, 2.0, 2.0, FPS, hold_fraction=0.25)
    assert max(track.values) == pytest.approx(1.0)
    assert min(track.values) == pytest.approx(0.0)
    assert track.values[24] == pytest.approx(1.0)
    with pytest.raises(anim.MotionError):
        anim.hinge_swing(1.0, 2.0, 2.0, FPS, hold_fraction=0.5)


def test_cam_follower_rises_slowly_and_drops_fast():
    track = anim.cam_follower(0.5, 1.0, 1.0, FPS, lobes=1, rise_fraction=0.75)
    peak = max(range(len(track.values)), key=track.values.__getitem__)
    assert track.values[peak] == pytest.approx(0.5, abs=1e-6)
    assert 16 <= peak <= 18
    drop = track.values[peak:]
    assert len(drop) - 1 <= 7
    assert all(later < earlier for earlier, later in zip(drop, drop[1:]))
    for frame in range(1, peak):
        assert track.values[frame] >= track.values[frame - 1]


def test_cam_follower_lobes_repeat():
    track = anim.cam_follower(0.5, 2.0, 2.0, FPS, lobes=2)
    assert track.values[:24] == pytest.approx(track.values[24:48])


def test_pendulum_period_matches_small_angle_formula():
    assert anim.pendulum_period(1.0) == pytest.approx(2.006, abs=1e-3)


def test_crank_slider_closure():
    radius = 0.15
    rod = 0.6
    motion = anim.crank_slider(radius, rod, 1.0, 2.0, FPS, phase=0.3)
    assert motion.crank.seam_error() < SEAM
    assert motion.slider.seam_error() < SEAM
    assert motion.rod.seam_error() < SEAM
    for theta, slider, phi in zip(motion.crank.values, motion.slider.values, motion.rod.values):
        end_x = radius * math.cos(theta) + rod * math.cos(phi)
        end_y = radius * math.sin(theta) + rod * math.sin(phi)
        assert end_x == pytest.approx(slider, abs=1e-12)
        assert end_y == pytest.approx(0.0, abs=1e-12)
    assert max(motion.slider.values) == pytest.approx(radius + rod, abs=1e-3)
    assert min(motion.slider.values) == pytest.approx(rod - radius, abs=1e-3)


def test_crank_slider_rejects_short_rod():
    with pytest.raises(anim.MotionError):
        anim.crank_slider(0.5, 0.4, 1.0, 1.0, FPS)


def test_gear_ratios_consistent():
    teeth = [12, 30, 18]
    duration = anim.gear_train_loop_duration(teeth, 1.0, FPS)
    tracks = anim.gear_train(teeth, 1.0, duration, FPS)
    assert duration == pytest.approx(15.0)
    for track in tracks:
        assert track.seam_error() < SEAM
    for driver, driven, driver_teeth, driven_teeth in zip(tracks, tracks[1:], teeth, teeth[1:]):
        for a, b in zip(driver.values, driven.values):
            assert a * driver_teeth == pytest.approx(-b * driven_teeth)


def test_gear_train_rejects_partial_turns():
    with pytest.raises(anim.MotionError):
        anim.gear_train([12, 30], 1.0, 2.0, FPS)


def test_gear_train_rejects_bad_teeth():
    with pytest.raises(anim.MotionError):
        anim.gear_periods([12, 2], 1.0)


def _wind(**overrides) -> anim.WindChain:
    arguments = {"bones": 6, "duration_s": 4.0, "fps": FPS, "seed": 11} | overrides
    return anim.wind_wave(**arguments)


def test_wind_wave_closes_the_loop_for_every_bone():
    chain = _wind(strength=1.7, gust_period_s=2.0)
    for track in chain.bend + chain.twist:
        assert track.frame_count == 96
        assert track.seam_error() < SEAM
        assert _seam_kink(track) < 1e-9


def test_wind_wave_is_deterministic_per_seed():
    assert _wind() == _wind()
    assert _wind().bend[-1].values != _wind(seed=12).bend[-1].values


def test_wind_wave_amplitude_grows_from_the_attached_edge():
    chain = _wind()
    peaks = [max(abs(value) for value in track.values) for track in chain.bend]
    assert peaks == sorted(peaks)
    assert peaks[0] < peaks[-1] / 3


def test_wind_wave_tip_deflection_stays_within_the_bend_budget():
    chain = _wind(strength=0.8, max_bend=0.5, gust_depth=0.25)
    totals = [sum(track.values[frame] for track in chain.bend) for frame in range(chain.bend[0].frame_count + 1)]
    assert max(abs(total) for total in totals) <= 0.5 * 0.8 * 1.25 + 1e-12


def test_stronger_wind_runs_more_wave_cycles():
    def crossings(track: anim.Track) -> int:
        return sum(1 for a, b in zip(track.values, track.values[1:]) if a * b < 0)

    assert crossings(_wind(strength=2.0).bend[0]) > crossings(_wind(strength=0.5).bend[0])


def test_wind_wave_flutter_is_faster_than_the_wave():
    chain = _wind(gust_depth=0.0)
    bend_turns = sum(1 for a, b in zip(chain.bend[2].values, chain.bend[2].values[1:]) if a * b < 0)
    twist_turns = sum(1 for a, b in zip(chain.twist[2].values, chain.twist[2].values[1:]) if a * b < 0)
    assert twist_turns == pytest.approx(anim.FLUTTER_HARMONIC * bend_turns, abs=1)


@pytest.mark.parametrize(
    "overrides",
    [{"bones": 0}, {"strength": 0.0}, {"gust_period_s": 3.0}, {"gust_depth": 1.0}, {"duration_s": 4.01}, {"wavelength_bones": -1.0}],
)
def test_wind_wave_rejects_bad_parameters(overrides):
    with pytest.raises(anim.MotionError):
        _wind(**overrides)
