from __future__ import annotations

import pytest

from library import kits, materials
from terrain import splat, vegetation
from terrain.config import SCATTER_KINDS, MapConfigError

RULES = """
[layer.ground]
material = "silt"
weight = "1"

[species.tree]
kind = "conifer"
meshes = ["nature/tree_pineTallA", "nature/tree_oak"]
density = "1"
spacing_m = 4.0
scale = [0.8, 1.2]

[species.tuft]
kind = "grass"
density = "1"
spacing_m = 1.0
scale = [0.7, 1.3]
"""


@pytest.fixture(scope="module")
def index() -> kits.KitIndex:
    return kits.build()


def parse(text: str) -> tuple[splat.Species, ...]:
    return splat.parse_biome(text, materials.load()).species


def test_every_kind_has_a_size_and_a_tri_budget():
    assert set(vegetation.KIND_SIZE_M) == set(SCATTER_KINDS)
    assert set(vegetation.KIND_TRI_BUDGET) == set(SCATTER_KINDS)


def test_meshes_are_read_and_default_to_none():
    tree, tuft = parse(RULES)
    assert tree.meshes == ("nature/tree_pineTallA", "nature/tree_oak")
    assert tuft.meshes == ()
    assert tree.as_dict()["meshes"] == ["nature/tree_pineTallA", "nature/tree_oak"]


@pytest.mark.parametrize(
    "meshes",
    [
        "[]",
        '"nature/tree_oak"',
        '["tree_oak"]',
        '["nature/"]',
        '["nature/a", "nature/a"]',
        '["nature/a", "nature/b", "nature/c", "nature/d", "nature/e"]',
    ],
)
def test_malformed_meshes_are_rejected(meshes):
    text = RULES.replace('meshes = ["nature/tree_pineTallA", "nature/tree_oak"]', f"meshes = {meshes}")
    with pytest.raises(splat.SplatRuleError):
        parse(text)


def test_variants_are_scaled_to_the_kind_size(index):
    tree, _ = parse(RULES)
    variants = vegetation.resolve(tree, index)
    assert [variant.mesh for variant in variants] == list(tree.meshes)
    for variant in variants:
        prop = index[variant.mesh]
        assert variant.glb == prop.glb
        assert variant.tris == prop.tris
        assert max(prop.dimensions_m) * variant.unit_scale == pytest.approx(vegetation.KIND_SIZE_M["conifer"])


def test_an_unknown_kit_prop_is_rejected(index):
    tree, _ = parse(RULES.replace("nature/tree_oak", "nature/no_such_tree"))
    with pytest.raises(MapConfigError):
        vegetation.resolve(tree, index)


def test_species_without_meshes_need_no_kit_index():
    _, tuft = parse(RULES)
    assert vegetation.resolve_all((tuft,), index=object()) == {"tuft": ()}


@pytest.mark.parametrize("name", splat.available())
def test_shipped_species_have_locked_variants_within_budget(name, index):
    biome = splat.biome(name)
    assert biome.species
    for species in biome.species:
        variants = vegetation.resolve(species, index)
        assert len(variants) >= 2, species.name
        for variant in variants:
            assert variant.tris <= vegetation.KIND_TRI_BUDGET[species.kind], (species.name, variant.mesh)


def test_props_from_a_metric_source_keep_their_modelled_size(index):
    metric = [name for name in index if index[name].source in vegetation.METRIC_SOURCES]
    if not metric:
        pytest.skip("no generated pack is locked")
    species = parse(RULES.replace('meshes = ["nature/tree_pineTallA", "nature/tree_oak"]', f'meshes = ["{metric[0]}"]'))[0]
    assert vegetation.resolve(species, index)[0].unit_scale == 1.0
