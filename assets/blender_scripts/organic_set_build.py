from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ASSETFORGE_WORKSPACE", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bpy

from assets import export, lod, materials, organic
from assets.blender_scripts.scatter_set_build import bake_shared_atlas, endgrain_material
from assets.recipe import RecipeSession

PREFIX = "organic_"
KIND = "organic_prop"
RECIPE = (
    "import runpy\n"
    "from pathlib import Path\n\n"
    f"runpy.run_path(str(Path(r\"{REPO.as_posix()}\") / \"assets/blender_scripts/organic_set_build.py\"), run_name=\"__main__\")"
)

MODELS = {
    "stump_a": ("wood", organic.stump, dict(seed=101, radius=0.3, height=0.42)),
    "stump_b": ("wood", organic.stump, dict(seed=104, radius=0.36, height=0.75, buttresses=6, jag=0.2)),
    "roots_a": ("wood", organic.root_cluster, dict(seed=105, roots=5, reach=1.3, thickness=0.09)),
    "roots_b": ("wood", organic.root_cluster, dict(seed=107, roots=7, reach=0.9, thickness=0.06, arch=0.12)),
    "mushroom_a": ("porcini", organic.mushroom, dict(seed=109, stem_height=0.13, stem_radius=0.035, cap_radius=0.1, cap_height=0.07)),
    "mushroom_b": ("agaric", organic.mushroom, dict(seed=111, stem_height=0.2, stem_radius=0.022, cap_radius=0.085, cap_height=0.06)),
    "mushroom_cluster_a": ("tan", organic.mushroom_cluster, dict(seed=113, count=5, spread=0.1)),
    "mushroom_cluster_b": ("tan", organic.mushroom_cluster, dict(seed=115, count=7, spread=0.08, size=0.8, lean_deg=28.0)),
    "coral_branch_a": ("coral_orange", organic.branching_coral, dict(seed=117, height=0.7, depth=3, sides=8)),
    "coral_branch_b": ("coral_lilac", organic.branching_coral, dict(seed=119, height=0.45, radius=0.035, forks=(2, 2), spread_deg=48.0)),
    "coral_dome_a": ("brain", organic.dome_coral, dict(seed=121, radius=0.42, height=0.3)),
}


def _ramp_material(name: str, scale: float, low: tuple, high: tuple, roughness: float, bump: float = 0.0) -> bpy.types.Material:
    material, tree, bsdf = materials._bsdf_material(name)
    noise = materials._noise(tree, scale, 5.0)
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*low, 1.0)
    ramp.color_ramp.elements[1].color = (*high, 1.0)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    if bump > 0.0:
        bump_node = tree.nodes.new("ShaderNodeBump")
        bump_node.inputs["Strength"].default_value = bump
        tree.links.new(noise.outputs["Fac"], bump_node.inputs["Height"])
        tree.links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def agaric_material() -> bpy.types.Material:
    material, tree, bsdf = materials._bsdf_material("agaric_cap")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 38.0
    voronoi.inputs["Randomness"].default_value = 0.8
    tree.links.new(coord.outputs["Object"], voronoi.inputs["Vector"])
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0.85, 0.82, 0.74, 1.0)
    ramp.color_ramp.elements[1].position = 0.22
    ramp.color_ramp.elements[1].color = (0.62, 0.06, 0.03, 1.0)
    tree.links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return material


def brain_material() -> bpy.types.Material:
    material, tree, bsdf = materials._bsdf_material("brain_coral")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.inputs["Scale"].default_value = 9.0
    wave.inputs["Distortion"].default_value = 14.0
    wave.inputs["Detail"].default_value = 4.0
    tree.links.new(coord.outputs["Object"], wave.inputs["Vector"])
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.42, 0.40, 0.24, 1.0)
    ramp.color_ramp.elements[1].color = (0.66, 0.62, 0.42, 1.0)
    tree.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.75
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    tree.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def palettes() -> dict[str, list[bpy.types.Material]]:
    stem = _ramp_material("mushroom_stem", 30.0, (0.70, 0.64, 0.52), (0.84, 0.79, 0.68), 0.7)
    return {
        "wood": [materials.get_material("bark_cc0"), endgrain_material()],
        "porcini": [stem, _ramp_material("porcini_cap", 12.0, (0.28, 0.15, 0.07), (0.44, 0.27, 0.13), 0.5)],
        "agaric": [stem, agaric_material()],
        "tan": [stem, _ramp_material("tan_cap", 14.0, (0.50, 0.38, 0.24), (0.68, 0.54, 0.36), 0.55)],
        "coral_orange": [_ramp_material("coral_orange", 40.0, (0.70, 0.34, 0.18), (0.88, 0.52, 0.30), 0.7, bump=0.4)],
        "coral_lilac": [_ramp_material("coral_lilac", 40.0, (0.46, 0.32, 0.50), (0.64, 0.48, 0.66), 0.7, bump=0.4)],
        "brain": [brain_material()],
    }


def build() -> list[bpy.types.Object]:
    palette = palettes()
    built = []
    for index, (model, (style, builder, params)) in enumerate(MODELS.items()):
        obj = builder(f"{PREFIX}{model}", **params)
        indices = [polygon.material_index for polygon in obj.data.polygons]
        obj.data.materials.clear()
        for material in palette[style]:
            obj.data.materials.append(material)
        obj.data.polygons.foreach_set("material_index", [min(value, len(palette[style]) - 1) for value in indices])
        obj.data.update()
        obj.location.x = 2.0 * index
        built.append(obj)
    return built


def main() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    props = build()
    bake_shared_atlas(props)
    for obj in props:
        name = obj.name
        lod_set = lod.generate_lods(obj)
        result = export.export_asset(lod_set.objects, name=name, kind=KIND)
        RecipeSession(name=name, snippets=[RECIPE]).write()
        print(f"[organic] {name} {result.triangle_counts}", flush=True)


if __name__ == "__main__":
    main()
