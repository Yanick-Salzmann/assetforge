---
name: asset-prop
description: Generate a companion set of 6-12 dressing props end to end - brief, CC0 fit check against library/kits, live bpy modelling for whatever isn't already covered, one shared atlas across the whole set, per-prop contact sheet/critique/validate/patch/approve/export. Budget 200-1500 tris per prop. Trigger on "/asset-prop", "make a prop set for X", "generate dressing props", or any request for a batch of small props to go with a building or terrain.
---

# asset-prop

Produces one deliverable set of individually placeable props, each its own
`out/assets/<prop-name>/`: `<prop-name>.glb`, `asset.json`, `recipe.py`, `textures/`,
`contact_sheet.png`. Budget: 200-1500 tris per prop (`prop` in `TRI_BUDGETS`), 6-12 props per
set, one shared 2048 atlas across every modelled prop in the set. 1 Blender unit = 1 metre,
Z-up, origin at base centre, `+Y` forward - same conventions `validate_asset` and the contact
sheet check for buildings.

This reuses the [[asset-building]] loop per prop - brief, model, contact sheet, critique,
validate, patch, approve, export - with two differences: a CC0 reuse check before modelling
anything, and one shared atlas built across the whole set instead of one per prop. Call
`asset_status` first if this session hasn't already - it lists the Blender executable, output
dir and kits dir.

## 1. Brief

Before touching Blender or `library.kits`, pin down:

- **set name** - lowercase alphanumerics/dashes/underscores only
  (`assets.recipe.validate_name`), used as a prefix for prop names, not an output directory of
  its own - each prop gets its own `out/assets/<prop-name>/`.
- **6-12 props** - name, category (crate, barrel, debris, foliage, signage, ...), rough
  footprint in metres, and what it's dressing (which building or terrain biome).

State the brief back in one line before checking the kit library.

## 2. CC0 fit check - library/kits first

```python
from library import kits
index = kits.load()
index.find(category="<category>", max_tris=1500, max_dimension_m=<largest planned dimension>)
```

`find` returns candidates smallest-tris-first. For every prop in the brief with a usable
candidate, use it as-is - record its `pack`/`source`/`licence` in the set notes and drop it from
the modelling list; a CC0 prop that already fits the budget does not get re-exported through this
pipeline, it gets placed directly. Only props with no fit (`find` returns nothing at the given
category/budget) go to step 3. `index.categories()` lists what categories exist if the brief's
category guess doesn't match the index.

## 3. Model - live bpy session

Same as `asset-building` step 2: insert the repo root into `sys.path`, import
`assets.hardsurface` and `assets.materials`, build with the composable helpers
(`bevelled_box`, `frustum`, `cylinder_between`, `panel_cut`, `array`, `mirror`, `solidify`,
`greeble`, ...), assign per-part materials with `assets.materials`, `record_recipe(name, code)`
after every snippet. Model every prop that needs generating into the same live scene, one
object group per prop, named so the groups stay distinguishable (e.g. `crate_01`,
`crate_01_lid`) - step 4 needs the full set's objects together.

## 4. One shared atlas across the set

Unlike a single building, do not call `game_ready_pass` per prop. Also **do not call
`game_ready.game_ready_pass(all_prop_objects)` on its own** - it calls `consolidate_material`
with no `bake_result`, which produces a UV-correct but untextured `atlas_material` and silently
throws away every prop's CC0/procedural material. Bake the atlas explicitly instead, once across
every newly-modelled prop's objects in the same session, before any of them are exported:

```python
from assets import hardsurface as hs, game_ready, atlas_bake

all_prop_objects = [...]  # every modelled prop's objects, whole set, not yet exported
for obj in all_prop_objects:
    hs.apply_transforms(obj)
    hs.recalc_normals(obj)
game_ready.uv_unwrap_and_pack(all_prop_objects)          # while parts still carry their own materials
bake_result = atlas_bake.bake_atlas(all_prop_objects, atlas_size=game_ready.ATLAS_SIZE)
material = game_ready.consolidate_material(all_prop_objects, bake_result=bake_result)
game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)
```

This bakes transforms/normals, UV-packs one shared 0-1 atlas across the whole set, bakes every
part's original material into shared albedo/normal/orm images, and consolidates every object onto
one shared `atlas_material` wired to those images - the "one atlas per set" budget line.
`record_recipe` each call too.

`atlas_bake.bake_atlas` runs 5 sequential full-resolution Cycles bake passes and is long-running;
`bpy.ops.object.bake` blocks Blender's thread for the whole call and the `blender-mcp` bridge's
transport can time out and report "failed" before Blender actually finishes - that's a transport
artifact, not a real failure. Don't blindly retry (it duplicates images under `.001`); check
`bpy.data.images` and the material's node tree first. Issuing one `execute_blender_code` call per
bake pass instead of one call for all of `bake_atlas` reduces how often this trips.

If a later patch (step 7) changes one prop's geometry, re-run this whole sequence across the
whole set's objects again before re-exporting anything - the shared UV pack shifts when any
object in the group changes, so a stale atlas layout on an already-exported prop is a bug the
next full re-run silently fixes, not one worth chasing per-prop.

## 5. Export each prop individually

```python
from assets import export
export.export_asset(prop_objects, name=prop_name, kind="prop")
```

Once per prop, passing only that prop's own objects (not the whole set) so each prop keeps its
own origin at its own base centre. Every export re-writes the same shared atlas images into
that prop's own `textures/` - expected duplication, each prop is a self-contained deliverable.

## 6. Contact sheet + critique, per prop

```
render_contact_sheet(prop_name)
```

Same six ortho views plus three-quarter and worm's-eye, human reference beside the prop. Read
the image. Write one line judging silhouette/proportions against the brief and the human
reference, no missing back faces - props are viewed from any angle same as buildings, no
one-angle-only detail.

## 7. Validate + patch loop, per prop - capped at 4 iterations each

```
validate_asset(prop_name, "prop")
```

On a failure or a critique fail: `record_patch(prop_name)` (raises `GateError` past 4
iterations - hand that prop to a human reviewer if hit), fix in the live session
(`record_recipe` it), re-run step 4's `game_ready_pass` across the whole set if geometry
changed, re-export that prop, go back to step 6 for it.

## 8. Human approval

Once every prop's contact sheet critique and `validate_asset` pass, ask the user to look at the
set (all contact sheets together) and confirm. Then, per prop:

```
approve_asset(prop_name, note)
```

Pass the user's note through verbatim for each. `asset_gate_status(prop_name)` should show
`approved: true` for every prop in the set before calling this done.

## Done

Report the set: each prop's `out/assets/<prop-name>/` file list, its `asset_gate_status`, and
which props (if any) came from `library/kits` instead of being modelled, so the user can confirm
the set matches the brief.
