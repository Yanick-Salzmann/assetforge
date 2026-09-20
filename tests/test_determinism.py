from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL.Image")

from library import materials
from terrain import channels as channels_mod
from terrain import config, session, splat

RESOLUTION = 32
FAST_EROSION = {"iterations": 8, "settle_iterations": 2, "thermal_settle_passes": 2}

BIOME_TOML = """
[biome]
name = "repro"

[layer.silt_flat]
material = "silt"
weight = "1 - slope"

[layer.riverbed_gravel]
material = "river_rock"
weight = "flow"
"""


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TERRAIN_OUT_DIR", tmp_path / "terrain")
    for name in session.resident():
        session.forget(name)
    yield
    for name in session.resident():
        session.forget(name)


@pytest.fixture(scope="module")
def index() -> materials.MaterialIndex:
    return materials.load()


@pytest.fixture(scope="module")
def biome(index) -> splat.Biome:
    return splat.parse_biome(BIOME_TOML, index)


def run(name: str, seed: int) -> tuple[torch.Tensor, channels_mod.ChannelStack]:
    current = session.create(name, seed=seed, resolution=RESOLUTION, world_size_m=1024.0)
    current.erosion_overrides = dict(FAST_EROSION)
    current.build_channels()
    return current.height.cpu().clone(), current.stack


def test_synth_erosion_and_channels_are_byte_identical_across_independent_runs():
    height_a, stack_a = run("repro-a", seed=17)
    height_b, stack_b = run("repro-b", seed=17)
    assert torch.equal(height_a, height_b)
    assert torch.equal(stack_a.data.cpu(), stack_b.data.cpu())


def test_splat_render_is_byte_identical_given_the_same_seed_and_rule_file(biome):
    _, stack_a = run("repro-splat-a", seed=23)
    _, stack_b = run("repro-splat-b", seed=23)
    result_a = splat.render(biome, stack_a)
    result_b = splat.render(biome, stack_b)
    for texture_a, texture_b in zip(result_a.textures, result_b.textures):
        assert np.array_equal(texture_a, texture_b)


def test_different_seeds_diverge():
    height_a, _ = run("repro-diff-a", seed=1)
    height_b, _ = run("repro-diff-b", seed=2)
    assert not torch.equal(height_a, height_b)
