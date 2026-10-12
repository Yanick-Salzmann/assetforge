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
(`bevelled_box`, `frustum`, `cylinder_between`, `box_between`, `lathe` for barrels and turned
parts, `prism`, `torus`, `panel_cut`, `array`, `mirror`, `solidify`, `greeble`, ...), assign
per-part materials with `assets.materials`, `record_recipe(name, code)` after every snippet.
Model every prop that needs generating into the same live scene and tag every part with the prop
it belongs to - step 4 groups and exports by that tag, never by parsing object names, so set or
prop names may contain underscores freely (af-7pp):

```python
from assets import prop_set
prop_set.tag(obj, "tavern_crate")   # the prop's asset name, i.e. out/assets/tavern_crate/
```

Untagged objects (helpers, references, cutters left behind) are ignored by the build.

Follow `asset-building`'s **Live-session conventions** (step 2) - workspace env var, fresh namespace per snippet, `<name>_*` naming, cleanup before rebuilds, +Y front, overlapping joins, geometry-derived mount points, greeble face guards, materials in `MATERIAL_BUILDERS`.

## 4. Shared atlas bake + per-prop export - a background job, never a live MCP call

One atlas is baked across the whole set, then each prop is exported on its own with only its own
parts, so each keeps its origin at its own base centre. The bake passes block Blender's only
thread for minutes, longer than the `blender-mcp` bridge's transport timeout, so this never runs
inside `execute_blender_code`. The live session only saves its scene; a headless Blender started
as a background shell job does the rest, with no timeout, and the harness notifies you when it
exits.

1. In the live session, save a copy of the scene (every part tagged, each on its own per-part
   materials) and `record_recipe` it:

   ```python
   import bpy
   bpy.ops.wm.save_as_mainfile(filepath=r"<repo_root>/out/assets/.build/<set>/<set>.blend", copy=True)
   ```

2. Launch the job with the Bash tool and `run_in_background: true` (Blender from
   `asset_status`'s `blender_executable`):

   ```bash
   "<blender_executable>" --background "out/assets/.build/<set>/<set>.blend" --factory-startup \
     --python assets/blender_scripts/prop_set_build.py -- "<repo_root>" \
     "out/assets/.build/<set>/prop_set.json" 2>&1 | grep --line-buffered -E "\[prop_set\]|Error|Traceback"
   ```

   `prop_set_build.py` runs on every tagged mesh: `apply_transforms`/`recalc_normals`,
   `uv_unwrap_and_pack` while the parts still carry their own materials, the three atlas bake
   passes, `consolidate_material(objects, bake_result=...)`, `enforce_power_of_two_textures`, then
   `export_asset(parts, name=<prop>, kind="prop")` once per prop. Every export writes the same
   shared atlas images into that prop's own `textures/` - expected duplication, each prop is a
   self-contained deliverable. Do not use `game_ready.game_ready_pass(objects)` - it bakes no
   textures and leaves a blank `atlas_material`.

3. Don't poll, and don't start a second job for the same set while one runs. When the
   notification arrives, read `prop_set.json`: `{"ok": true, "props": {...}}` carries each prop's
   export result, `{"ok": false, "error": ...}` the traceback. Fix the cause and relaunch; the job
   always starts again from the saved .blend, so a failed run leaves nothing half-baked behind.

If a later patch (step 6) changes any prop's geometry, save and rerun the whole job - the shared
UV pack shifts when any object in the set changes, so every prop is re-exported against the new
atlas.

## 5. Contact sheet + critique, per prop

```
render_contact_sheet(prop_name)
```

Same six ortho views plus three-quarter and worm's-eye, human reference beside the prop. Read
the image. Write one line judging silhouette/proportions against the brief and the human
reference, no missing back faces - props are viewed from any angle same as buildings, no
one-angle-only detail.

See `asset-building` step 3, **What the contact sheet can and can't tell you**: pre-bake sheets are clay, add Workbench close-ups for mechanism detail, and check visibility.

## 6. Validate + patch loop, per prop - capped at 4 iterations each

```
validate_asset(prop_name, "prop")
```

On a failure or a critique fail: `record_patch(prop_name)` (raises `GateError` past 4
iterations - hand that prop to a human reviewer if hit), fix in the live session
(`record_recipe` it), rerun step 4's job for the whole set, go back to step 5 for it.

## 7. Human approval

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
