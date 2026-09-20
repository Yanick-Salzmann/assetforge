---
name: asset-vehicle
description: Model a non-building hard-surface asset (vehicle or machinery) end to end - same loop as asset-building, adjusted brief and 5000-20000 tri budget. Brief, live bpy modelling over assets/hardsurface.py, contact sheet, written critique, validate, a patch loop capped at 4 iterations, human approval, game_ready + LODs, export. Trigger on "/asset-vehicle", "model a vehicle called X", "build a machinery asset", or any request to model a car, truck, tank, crane, or other non-building hard-surface asset.
---

# asset-vehicle

Produces one deliverable set under `out/assets/<name>/`: `<name>.glb`, `asset.json`,
`recipe.py`, `textures/`, `contact_sheet.png`. Budget: 5000-20000 tris (`vehicle` in
`TRI_BUDGETS`), one shared material, 2048 atlas, LOD0/1/2 always shipped - same as a hero
building, just wider. 1 Blender unit = 1 metre, Z-up, origin at base centre, `+Y` forward - the
contact sheet and `validate_asset` both check these, not just this doc.

This is [[asset-building]]'s loop with a different budget and brief; it is not a separate
pipeline - every helper, gate and MCP tool below is the same one that skill uses. Call
`asset_status` first if this session hasn't already.

## 1. Brief

Before touching Blender, pin down:

- **name** - lowercase alphanumerics/dashes/underscores only
  (`assets.recipe.validate_name`), becomes the output directory.
- **kind** - `vehicle` (5000-20000 tris) - covers both vehicles and machinery; there is no
  separate `machinery` budget row, use `vehicle` for both.
- **overall dimensions in metres** - length/width/height, checked against the 1.8 m human
  reference every contact sheet includes.
- **silhouette and mechanism** - body shape, wheel/track/leg count and placement, articulated
  parts (turret, boom, bucket, doors) if any, panel seams and vents, primary material read
  (painted metal panels, raw steel, rubber, glass).

State the brief back in one line before generating anything.

## 2. Model - live bpy session

Same import and helper set as `asset-building`:

```python
import sys
sys.path.insert(0, r"<repo_root from asset_status>")
from assets import hardsurface as hs
from assets import materials
```

`cylinder_between` covers axles, wheels and hydraulic rams; `array` repeats wheels/road wheels
along a side; `mirror` builds the opposite side from one modelled half; `frustum` covers tapered
noses, hulls and counterweights; `panel_cut` cuts vents, hatches and window openings;
`greeble` adds panel-line and bolt detail at range. Assign materials with `assets.materials`
(`metal_material` for panels, `glass_material` for windows/canopy, plus procedural node groups
for rubber/tread where none exists yet - add one in `assets/materials.py` rather than faking
tread with a metal tint) - leave per-part materials as-is, consolidation to one atlas is
`game_ready_pass`'s job in step 8.

`record_recipe(name, code)` after every snippet, same as building modelling - not optional.

## 3. Contact sheet

```
render_contact_sheet(name)
```

Same eight views. Wheels/tracks and any articulated parts should read correctly from the worm's-
eye and three-quarter shots too - a mechanism that only looks right from the side it was designed
in is the same failure mode a building's one-angle-only detail is.

## 4. Written critique

One paragraph against the brief: silhouette and proportions against the 1.8 m human reference,
wheel/track contact with the ground plane, no missing back or underside faces, mechanism reads
from every angle. State pass or fail plainly.

## 5. Validate

```
validate_asset(name, "vehicle")
```

Same checks as a building at the vehicle tri budget: triangle count, per-primitive material
assignment, watertight/manifold, outward normals, UV overlap, applied transforms at base-centre
origin.

## 6. Patch loop - capped at 4 iterations

Same as `asset-building` step 6: `record_patch(name)` first (raises `GateError` past the cap -
`asset_gate_status(name)` shows `patches_remaining`), fix in the live session and
`record_recipe` it, back to step 3.

## 7. Human approval

```
approve_asset(name, note)
```

Ask the user to look at the contact sheet and confirm; pass their note through verbatim.
`asset_gate_status(name)` should show `approved: true` before continuing.

## 8. game_ready + LODs + export

**Do not call `game_ready.game_ready_pass(objects)` alone** - it bakes no textures (it calls
`consolidate_material` with no `bake_result`, leaving a blank `atlas_material`). Use the full
sequence from `asset-building` step 8 - `apply_transforms`/`recalc_normals`, then
`uv_unwrap_and_pack` while parts still carry their own materials, then `atlas_bake.bake_atlas`,
then `consolidate_material(objects, bake_result=...)`, then `enforce_power_of_two_textures`, then
`bpy.ops.object.join()` (a multi-part vehicle is never auto-joined, and `generate_lods` needs one
object), then `generate_lods`, then `export_asset`:

```python
from assets import hardsurface as hs, game_ready, atlas_bake, lod, export

for obj in objects:
    hs.apply_transforms(obj)
    hs.recalc_normals(obj)

game_ready.uv_unwrap_and_pack(objects)
bake_result = atlas_bake.bake_atlas(objects, atlas_size=game_ready.ATLAS_SIZE)
material = game_ready.consolidate_material(objects, bake_result=bake_result)
game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)

bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.object.join()
lod0_object = bpy.context.view_layer.objects.active
lod_set = lod.generate_lods(lod0_object)
result = export.export_asset(lod_set.objects, name=name, kind="vehicle")
```

`record_recipe` each call. `atlas_bake.bake_atlas` runs 5 sequential full-resolution Cycles bake
passes and is long-running (minutes on a many-part asset) - `bpy.ops.object.bake` blocks
Blender's single thread for the whole call, and the `blender-mcp` bridge's transport can time out
before it returns even though Blender keeps computing correctly. That's a transport artifact, not
a failure - do not blindly retry a timed-out bake call (it duplicates images under `.001`
suffixes); check `bpy.data.images`/the material's node tree first to see what already completed.
Issuing one `execute_blender_code` call per bake pass instead of one call for all of
`bake_atlas` reduces how often this trips, at no cost to the actual bake time.

## Done

Run `validate_asset(name, "vehicle")` once more against the final exported glb, then report the
`out/assets/<name>/` file list, `asset_gate_status(name)`, and the one-line brief from step 1.
