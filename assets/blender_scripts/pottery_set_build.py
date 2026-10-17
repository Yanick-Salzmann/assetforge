from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ASSETFORGE_WORKSPACE", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bpy

from assets import export, lod, materials, pottery
from assets.blender_scripts.scatter_set_build import bake_shared_atlas, endgrain_material
from assets.recipe import RecipeSession

PREFIX = "pottery_"
KIND = "organic_prop"
RECIPE = (
    "import runpy\n"
    "from pathlib import Path\n\n"
    f"runpy.run_path(str(Path(r\"{REPO.as_posix()}\") / \"assets/blender_scripts/pottery_set_build.py\"), run_name=\"__main__\")"
)

MODELS = {
    "jar_raw": ("raw", pottery.vessel, dict(profile="jar_small", seed=201, lean_deg=3.5, oval=0.07)),
    "jar_fired": ("fired", pottery.vessel, dict(profile="jar_small", seed=203, lean_deg=2.0, oval=0.05, wobble=0.035)),
    "thrown_raw": ("raw", pottery.vessel, dict(profile="thrown", seed=205, lean_deg=4.0, oval=0.08, wobble=0.04)),
    "storage_fired": ("fired", pottery.vessel, dict(profile="storage", seed=207, lean_deg=2.5, oval=0.06)),
    "bowl_raw": ("raw", pottery.vessel, dict(profile="bowl", seed=209, lean_deg=1.5, oval=0.08, wobble=0.03)),
    "bowl_fired": ("fired", pottery.vessel, dict(profile="bowl", seed=211, lean_deg=1.0, oval=0.06, wobble=0.03)),
    "amphora": ("fired", pottery.vessel, dict(profile="amphora", seed=213, segments=18, lean_deg=2.5, oval=0.05)),
    "water_jar": ("fired", pottery.vessel, dict(profile="water_jar", seed=215, segments=18, lean_deg=3.0, oval=0.06)),
    "log_pile": ("wood", pottery.log_pile, dict(seed=217)),
    "sherds": ("fired", pottery.sherd_cluster, dict(seed=219)),
}


def palettes() -> dict[str, list[bpy.types.Material]]:
    return {
        "raw": [materials.get_material("raw_clay")],
        "fired": [materials.get_material("terracotta")],
        "wood": [materials.get_material("wood"), endgrain_material()],
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
        obj.location.x = 1.5 * index
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
        print(f"[pottery] {name} {result.triangle_counts}", flush=True)


if __name__ == "__main__":
    main()
