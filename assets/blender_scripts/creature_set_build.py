from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ASSETFORGE_WORKSPACE", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bpy

from assets import creature, creature_spec, export, lod, materials
from assets.blender_scripts.organic_set_build import _ramp_material
from assets.blender_scripts.scatter_set_build import bake_shared_atlas
from assets.recipe import RecipeSession

PREFIX = "creature_"
KIND = "creature"
NOISE_SCALE = {"hide": 60.0, "underside": 50.0, "mane": 90.0, "dark": 30.0, "horn": 25.0, "eye": 5.0}
ROUGHNESS = {"hide": 0.8, "underside": 0.8, "mane": 0.85, "dark": 0.6, "horn": 0.45, "eye": 0.15}
BUMP = {"hide": 0.25, "underside": 0.2, "mane": 0.45, "dark": 0.1, "horn": 0.15, "eye": 0.0}


def _recipe(species: str) -> str:
    return (
        "import runpy\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"sys.argv = [sys.argv[0], \"--\", \"{species}\"]\n"
        f"runpy.run_path(str(Path(r\"{REPO.as_posix()}\") / \"assets/blender_scripts/creature_set_build.py\"), run_name=\"__main__\")"
    )


def region_material(species: creature_spec.Species, region: str) -> bpy.types.Material:
    colour = species.palette.get(region, species.palette["hide"])
    if region == "eye":
        material, _, bsdf = materials._bsdf_material(f"{species.name}_eye")
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
        bsdf.inputs["Roughness"].default_value = ROUGHNESS["eye"]
        return material
    low = tuple(channel * 0.75 for channel in colour)
    high = tuple(min(1.0, channel * 1.2) for channel in colour)
    return _ramp_material(f"{species.name}_{region}", NOISE_SCALE[region], low, high, ROUGHNESS[region], bump=BUMP[region])


def build(species: creature_spec.Species) -> bpy.types.Object:
    obj = creature.creature(f"{PREFIX}{species.name}", species)
    indices = [polygon.material_index for polygon in obj.data.polygons]
    obj.data.materials.clear()
    for region in species.regions():
        obj.data.materials.append(region_material(species, region))
    obj.data.polygons.foreach_set("material_index", indices)
    obj.data.update()
    return obj


def _selected(argv: list[str]) -> list[str]:
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1:]


def main() -> None:
    catalogue = creature_spec.catalogue()
    names = _selected(sys.argv) or list(catalogue)
    for species_name in names:
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        obj = build(catalogue[species_name])
        bake_shared_atlas([obj])
        name = obj.name
        lod_set = lod.generate_lods(obj)
        result = export.export_asset(lod_set.objects, name=name, kind=KIND)
        RecipeSession(name=name, snippets=[_recipe(species_name)]).write()
        print(f"[creature] {name} {result.triangle_counts}", flush=True)


if __name__ == "__main__":
    main()
