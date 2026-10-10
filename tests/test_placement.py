from __future__ import annotations

import math

import numpy as np
import pytest

from library import materials
from terrain import placement, splat
from terrain.config import HEIGHTMAP_MAX, MapConfig, MapConfigError

RESOLUTION = 32
WORLD_M = 64.0

RULES = """
[layer.ground]
material = "silt"
weight = "1"

[species.tree]
kind = "conifer"
density = "1"
spacing_m = 4.0
scale = [0.8, 1.2]
exclusion_m = 2.0

[species.bush]
kind = "shrub"
density = "1"
spacing_m = 1.5
scale = [0.5, 1.0]
exclusion_m = 1.0

[species.tuft]
kind = "grass"
density = "1"
spacing_m = 1.0
scale = [0.7, 1.3]
"""


@pytest.fixture(scope="module")
def species() -> tuple[splat.Species, ...]:
    return splat.parse_biome(RULES, materials.load()).species


def cfg(**kwargs) -> MapConfig:
    base = {"name": "placement-unit", "resolution": RESOLUTION, "world_size_m": WORLD_M, "seed": 11}
    base.update(kwargs)
    return MapConfig(**base)


def full_masks(names, value: int = 255) -> dict[str, np.ndarray]:
    return {name: np.full((RESOLUTION, RESOLUTION), value, np.uint8) for name in names}


def flat_height(value: int = 0) -> np.ndarray:
    return np.full((RESOLUTION, RESOLUTION), value, np.uint16)


def min_pair_distance(a: np.ndarray, b: np.ndarray | None = None) -> float:
    other = a if b is None else b
    distance = np.sqrt(((a[:, None, :2] - other[None, :, :2]) ** 2).sum(axis=-1))
    if b is None:
        np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def test_poisson_disc_keeps_the_minimum_distance():
    rng = np.random.Generator(np.random.PCG64(4))
    points = placement.poisson_disc(rng, 50.0, 2.0)
    assert points.shape[0] > 0.4 * (50.0 / 2.0) ** 2
    assert min_pair_distance(points) >= 2.0
    assert points.min() >= 0.0
    assert points.max() < 50.0


def test_poisson_disc_is_reproducible_from_the_generator():
    one = placement.poisson_disc(np.random.Generator(np.random.PCG64(9)), 40.0, 1.5)
    two = placement.poisson_disc(np.random.Generator(np.random.PCG64(9)), 40.0, 1.5)
    assert np.array_equal(one, two)


def test_place_is_byte_identical_from_the_seed(species):
    names = [entry.name for entry in species]
    first = placement.place(cfg(), species, full_masks(names), flat_height())
    second = placement.place(cfg(), species, full_masks(names), flat_height())
    for one, two in zip(first, second):
        assert one.instances.tobytes() == two.instances.tobytes()


def test_a_different_seed_moves_the_instances(species):
    names = [entry.name for entry in species]
    first = placement.place(cfg(), species[:1], full_masks(names), flat_height())
    second = placement.place(cfg(seed=12), species[:1], full_masks(names), flat_height())
    assert first[0].instances.tobytes() != second[0].instances.tobytes()


def test_species_keep_their_own_spacing(species):
    names = [entry.name for entry in species]
    placed = placement.place(cfg(), species, full_masks(names), flat_height())
    for entry in placed:
        assert entry.count > 0
        assert min_pair_distance(entry.instances) >= entry.species.spacing_m - 1e-4


def test_mutually_exclusive_species_never_overlap(species):
    names = [entry.name for entry in species]
    placed = placement.place(cfg(), species, full_masks(names), flat_height())
    for index, first in enumerate(placed):
        for second in placed[index + 1 :]:
            gap = max(first.species.exclusion_m, second.species.exclusion_m)
            assert min_pair_distance(first.instances, second.instances) >= gap - 1e-4


def test_an_empty_mask_places_nothing(species):
    names = [entry.name for entry in species]
    placed = placement.place(cfg(), species[:1], full_masks(names, 0), flat_height())
    assert placed[0].count == 0
    assert placed[0].instances.shape == (0, 5)


def test_half_density_thins_the_instances(species):
    names = [entry.name for entry in species]
    full = placement.place(cfg(), species[:1], full_masks(names), flat_height())[0].count
    half = placement.place(cfg(), species[:1], full_masks(names, 128), flat_height())[0].count
    assert 0.3 * full < half < 0.7 * full


def test_instances_are_centred_scaled_and_sit_on_the_heightmap(species):
    names = [entry.name for entry in species]
    c = cfg()
    height = flat_height(HEIGHTMAP_MAX // 2)
    placed = placement.place(c, species[:1], full_masks(names), height)[0].instances
    assert placed.dtype == np.dtype("<f4")
    assert placed[:, 0].min() >= -WORLD_M / 2
    assert placed[:, 0].max() < WORLD_M / 2
    assert placed[:, 2] == pytest.approx(np.full(len(placed), (HEIGHTMAP_MAX // 2) / HEIGHTMAP_MAX * c.height_range_m), rel=1e-5)
    assert placed[:, 3].min() >= 0.0
    assert placed[:, 3].max() < 2 * math.pi
    assert placed[:, 4].min() >= 0.8
    assert placed[:, 4].max() <= 1.2


def test_sample_height_is_bilinear_between_pixel_centres():
    c = cfg()
    height = np.zeros((RESOLUTION, RESOLUTION), np.uint16)
    height[:, 1] = HEIGHTMAP_MAX
    mpp = c.metres_per_pixel
    points = np.array([[0.5 * mpp, 0.0], [1.0 * mpp, 0.0], [1.5 * mpp, 0.0]])
    sampled = placement.sample_height(height, points, c) / c.height_range_m
    assert sampled == pytest.approx([0.0, 0.5, 1.0])


def test_the_kind_cap_bounds_the_count(species, monkeypatch):
    names = [entry.name for entry in species]
    monkeypatch.setitem(placement.KIND_CAPS, "grass", 25)
    placed = placement.place(cfg(), species[2:], full_masks(names), flat_height())
    assert placed[0].count <= 25
    assert placed[0].spacing_m > species[2].spacing_m


def test_a_missing_mask_is_rejected(species):
    with pytest.raises(MapConfigError):
        placement.place(cfg(), species[:1], {}, flat_height())


def test_write_and_read_round_trip(tmp_path, species):
    names = [entry.name for entry in species]
    placed = placement.place(cfg(), species, full_masks(names), flat_height())
    written = placement.write(placed, tmp_path)
    assert [path.name for path in written] == ["scatter_tree.bin", "scatter_bush.bin", "scatter_tuft.bin"]
    for path, entry in zip(written, placed):
        assert np.array_equal(placement.read(path), entry.instances)
