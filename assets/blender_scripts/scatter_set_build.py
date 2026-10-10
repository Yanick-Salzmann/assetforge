from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ASSETFORGE_WORKSPACE", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bpy

from assets import atlas_bake, export, game_ready, materials, organic
from assets import hardsurface as hs

ROCKS = {
    "scatter_boulder_a": dict(seed=11, size=(1.6, 1.35, 1.1), facets=8, facet_depth=0.14, roughness=0.14, base_cut=0.22),
    "scatter_boulder_b": dict(seed=23, size=(1.6, 1.2, 0.7), facets=10, facet_depth=0.2, roughness=0.1, base_cut=0.2),
    "scatter_boulder_c": dict(seed=37, size=(1.0, 0.9, 1.7), facets=11, facet_depth=0.24, roughness=0.1, base_cut=0.15),
    "scatter_stone_a": dict(seed=41, size=(0.42, 0.34, 0.26), facets=5, facet_depth=0.1, roughness=0.1, base_cut=0.25, subdivisions=3, target_tris=260),
    "scatter_stone_b": dict(seed=53, size=(0.48, 0.38, 0.16), facets=6, facet_depth=0.12, roughness=0.08, base_cut=0.25, subdivisions=3, target_tris=260),
}


def saguaro(name: str, height: float, arms, seed: int) -> bpy.types.Object:
    trunk = organic.tube(
        name,
        [(0, 0, height * t / 8) for t in range(9)],
        [0.24, 0.23, 0.225, 0.22, 0.215, 0.21, 0.205, 0.2, 0.19],
        sides=20,
        ribs=10,
        rib_depth=0.12,
        seed=seed,
        top="dome",
    )
    parts = []
    for index, (direction, base, reach, rise) in enumerate(arms):
        dx, dy = direction
        path = [
            (dx * 0.05, dy * 0.05, base),
            (dx * reach * 0.6, dy * reach * 0.6, base + 0.05),
            (dx * reach, dy * reach, base + 0.3),
            (dx * reach * 1.03, dy * reach * 1.03, base + rise * 0.6),
            (dx * reach * 1.03, dy * reach * 1.03, base + rise),
        ]
        parts.append(
            organic.tube(
                f"{name}_arm{index}",
                path,
                [0.14, 0.14, 0.14, 0.135, 0.13],
                sides=16,
                ribs=8,
                rib_depth=0.12,
                seed=seed + index + 1,
                top="dome",
            )
        )
    return organic.union(trunk, parts)


def build_geometry() -> list[bpy.types.Object]:
    built = [organic.rock(name, **params) for name, params in ROCKS.items()]
    built.append(
        organic.tube(
            "scatter_log_a",
            [(-1.2 + 0.4 * i, 0.03 * math.sin(i * 1.3), 0.17 - 0.004 * i) for i in range(7)],
            [0.17, 0.168, 0.165, 0.16, 0.155, 0.15, 0.145],
            sides=16,
            jitter=0.05,
            seed=5,
        )
    )
    built.append(
        organic.tube(
            "scatter_stump_a",
            [(0, 0, z) for z in (0.0, 0.05, 0.12, 0.2, 0.28, 0.36, 0.44)],
            [0.42, 0.33, 0.28, 0.255, 0.245, 0.24, 0.235],
            sides=16,
            jitter=0.07,
            seed=7,
        )
    )
    built.append(
        organic.tube(
            "scatter_stump_b",
            [(0.01 * i, 0, z) for i, z in enumerate((0.0, 0.06, 0.15, 0.27, 0.4, 0.55, 0.7))],
            [0.36, 0.26, 0.22, 0.205, 0.195, 0.185, 0.17],
            sides=16,
            jitter=0.08,
            seed=9,
        )
    )
    built.append(saguaro("scatter_saguaro_a", 3.7, [((1, 0), 1.5, 0.55, 1.1), ((-1, 0.2), 1.9, 0.5, 0.9)], 13))
    built.append(saguaro("scatter_saguaro_b", 3.2, [((0.3, 1), 1.7, 0.5, 1.0)], 17))
    built.append(
        organic.tube(
            "scatter_barrel_cactus_a",
            [(0, 0, 0.0), (0, 0, 0.12), (0, 0, 0.32), (0, 0, 0.46)],
            [0.27, 0.35, 0.34, 0.27],
            sides=28,
            ribs=14,
            rib_depth=0.14,
            seed=19,
            top="dome",
        )
    )
    built.append(
        organic.tube(
            "scatter_barrel_cactus_b",
            [(0, 0, 0.0), (0.0, 0, 0.15), (0.03, 0, 0.42), (0.07, 0, 0.66)],
            [0.24, 0.3, 0.29, 0.23],
            sides=28,
            ribs=14,
            rib_depth=0.14,
            seed=23,
            top="dome",
        )
    )
    for index, obj in enumerate(built):
        obj.location.x = 2.5 * index
    return built


def endgrain_material() -> bpy.types.Material:
    material, tree, bsdf = materials._bsdf_material("endgrain")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.rings_direction = "SPHERICAL"
    wave.inputs["Scale"].default_value = 6.0
    wave.inputs["Distortion"].default_value = 3.0
    wave.inputs["Detail"].default_value = 3.0
    tree.links.new(coord.outputs["Object"], wave.inputs["Vector"])
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.30, 0.20, 0.11, 1)
    ramp.color_ramp.elements[1].color = (0.52, 0.40, 0.26, 1)
    tree.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    return material


def cactus_material() -> bpy.types.Material:
    material, tree, bsdf = materials._bsdf_material("cactus_skin")
    noise = materials._noise(tree, 18.0, 6.0)
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.11, 0.17, 0.07, 1)
    ramp.color_ramp.elements[1].color = (0.24, 0.30, 0.13, 1)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def assign_materials(props: list[bpy.types.Object]) -> None:
    terrain_index = json.loads((materials.LIBRARY_DIR / "materials.json").read_text(encoding="utf-8"))["materials"]
    rock = materials.cc0_material("rock_cc0", terrain_index["cliff_rock"])
    bark = materials.get_material("bark_cc0")
    endgrain = endgrain_material()
    cactus = cactus_material()
    for obj in props:
        obj.data.materials.clear()
        if "boulder" in obj.name or "stone" in obj.name:
            obj.data.materials.append(rock)
        elif "log" in obj.name or "stump" in obj.name:
            obj.data.materials.append(bark)
            obj.data.materials.append(endgrain)
        else:
            obj.data.materials.append(cactus)
            for polygon in obj.data.polygons:
                polygon.material_index = 0


def bake_shared_atlas(props: list[bpy.types.Object]) -> None:
    for obj in props:
        hs.apply_transforms(obj)
        hs.recalc_normals(obj)
    game_ready.uv_unwrap_and_pack(props)
    atlas_bake.prepare_bake()
    atlas_bake.bake_albedo(props, atlas_size=game_ready.ATLAS_SIZE)
    atlas_bake.bake_normal(props, atlas_size=game_ready.ATLAS_SIZE)
    atlas_bake.bake_orm(props, atlas_size=game_ready.ATLAS_SIZE)
    material = game_ready.consolidate_material(props, bake_result=atlas_bake.collect_bake_result())
    game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)
    atlas_bake.purge_stale_images()


def main() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    props = build_geometry()
    assign_materials(props)
    bake_shared_atlas(props)
    for obj in props:
        export.export_asset([obj], name=obj.name, kind="prop")


if __name__ == "__main__":
    main()
