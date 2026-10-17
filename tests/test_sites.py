from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from terrain import beauty, placement, sites
from terrain.config import HEIGHTMAP_MAX, MapConfig
from terrain.splat import Species

SIZE = 128
WORLD_M = 512.0
RANGE_M = 100.0


def _cfg() -> MapConfig:
    return MapConfig(name="sites-unit", resolution=SIZE, world_size_m=WORLD_M, height_range_m=RANGE_M, seed=3)


def _flat(metres: float = 20.0) -> np.ndarray:
    return np.full((SIZE, SIZE), round(metres / RANGE_M * HEIGHTMAP_MAX), dtype=np.uint16)


def _dry() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=np.uint8)


def _footprints() -> tuple[sites.Footprint, ...]:
    return (
        sites.Footprint("small", 5.0, 3.0),
        sites.Footprint("large", 15.0, 10.0),
        sites.Footprint("medium", 9.0, 6.0),
    )


def test_ground_point_is_the_dry_pixel_nearest_the_centre():
    water = _dry()
    water[:, : SIZE // 2 + 4] = 255
    field = sites.HeightField(_cfg(), _flat(), water)
    x, y, z = sites.ground_point(field, 0.0)
    assert x > 0.0
    assert abs(z) <= field.metres_per_pixel
    assert y == pytest.approx(20.0, abs=0.01)
    assert water[int(field.to_pixel(np.float64(z)) + 0.5), int(field.to_pixel(np.float64(x)) + 0.5)] == 0


def test_sites_clear_the_eye_each_other_and_the_map_edge():
    field = sites.HeightField(_cfg(), _flat(), _dry())
    ground = sites.ground_point(field, 0.0)
    placed = sites.building_sites(_footprints(), field, 0.0)
    assert [site.name for site in placed] == ["small", "large", "medium"]
    for site in placed:
        assert math.hypot(site.x - ground[0], site.z - ground[2]) >= site.radius + sites.SITE_EYE_CLEARANCE_M
        assert abs(site.x) + site.radius <= WORLD_M / 2
        assert abs(site.z) + site.radius <= WORLD_M / 2
        assert site.base == pytest.approx(20.0, abs=0.01)
        assert site.yaw == pytest.approx(math.atan2(site.x - ground[0], site.z - ground[2]))
    for first in placed:
        for second in placed:
            if first is not second:
                gap = math.hypot(first.x - second.x, first.z - second.z)
                assert gap >= first.radius + second.radius + sites.SITE_GAP_M


def test_largest_footprint_takes_the_nearest_site():
    field = sites.HeightField(_cfg(), _flat(), _dry())
    ground = sites.ground_point(field, 0.0)
    placed = {site.name: site for site in sites.building_sites(_footprints(), field, 0.0)}
    reach = {name: math.hypot(site.x - ground[0], site.z - ground[2]) - site.radius for name, site in placed.items()}
    assert reach["large"] <= reach["medium"] + 1e-6


def test_sites_avoid_water_and_the_sea():
    water = _dry()
    water[: SIZE // 2, :] = 255
    field = sites.HeightField(_cfg(), _flat(), water)
    for site in sites.building_sites(_footprints(), field, 0.0):
        assert site.z - site.radius > -field.metres_per_pixel
    field = sites.HeightField(_cfg(), _flat(5.0), _dry())
    assert sites.building_sites(_footprints(), field, 10.0) == (None, None, None)


def test_sites_prefer_flat_ground_and_fall_back_to_the_flattest():
    columns = np.arange(SIZE, dtype=np.float64)
    ramp = np.tile(10.0 + columns * 0.7, (SIZE, 1))
    plateau = ramp.copy()
    plateau[96:120, 96:120] = 80.0
    for metres, flat in ((ramp, False), (plateau, True)):
        height = np.round(metres / RANGE_M * HEIGHTMAP_MAX).astype(np.uint16)
        field = sites.HeightField(_cfg(), height, _dry())
        (site,) = sites.building_sites((sites.Footprint("only", 10.0, 4.0),), field, 0.0)
        assert site is not None
        assert (site.relief <= sites.SITE_MAX_RELIEF_M) == flat
        if flat:
            assert 96 <= field.to_pixel(np.float64(site.x)) <= 120
            assert 96 <= field.to_pixel(np.float64(site.z)) <= 120


def test_footprints_read_buildings_with_a_glb_and_bbox(tmp_path):
    def asset(name, kind, glb=True, bbox=True):
        directory = tmp_path / name
        directory.mkdir()
        manifest = {"kind": kind}
        if bbox:
            manifest |= {"bbox_min": [-3.0, -4.0, 0.0], "bbox_max": [2.0, 1.0, 5.0]}
        (directory / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")
        if glb:
            (directory / f"{name}.glb").write_bytes(b"glTF")

    asset("b_hall", "hero_building")
    asset("a_crate", "prop")
    asset("c_shed", "building", glb=False)
    asset("d_hut", "building", bbox=False)
    footprints = sites.building_footprints(tmp_path)
    assert [footprint.name for footprint in footprints] == ["b_hall"]
    assert footprints[0].radius == pytest.approx(5.0)
    assert footprints[0].front == pytest.approx(4.0)
    assert footprints[0].glb == tmp_path / "b_hall" / "b_hall.glb"


def test_clear_scatter_drops_instances_inside_a_site():
    species = Species.__new__(Species)
    instances = np.array(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [6.5, 0.0, 1.0, 0.0, 1.0, 0.0], [7.5, 0.0, 1.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    placed = placement.Placement(species=species, instances=instances, spacing_m=1.0)
    site = sites.Site("hut", 0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
    (cleared,) = sites.clear_scatter((placed,), (site,))
    assert cleared.instances[:, 0].tolist() == [7.5]


def test_building_args_move_sites_into_blenders_frame(tmp_path):
    glb = tmp_path / "hut.glb"
    glb.write_bytes(b"glTF")
    cfg = _cfg()
    site = sites.Site("hut", -10.0, 30.0, 12.5, 0.0, 4.0, 0.7, glb)
    missing = sites.Site("gone", 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, tmp_path / "gone.glb")
    (entry,) = beauty.building_args(cfg, (site, missing))
    assert entry["location"] == [WORLD_M / 2 - 10.0, WORLD_M / 2 - 30.0, 12.5]
    assert entry["yaw_rad"] == 0.7
    assert entry["glb"] == str(glb)


def test_place_buildings_reads_the_water_channel():
    cfg = _cfg()
    height = torch.full((SIZE, SIZE), 0.2)
    water = torch.zeros((SIZE, SIZE))
    water[:, : SIZE // 2 + 4] = 1.0
    placed = beauty.place_buildings(cfg, height, water, _footprints())
    assert len(placed) == 3
    for site in placed:
        assert site.x - site.radius > 0.0
