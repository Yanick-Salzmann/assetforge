---
name: asset-building
description: Model a hero building end to end - brief, live bpy modelling over assets/hardsurface.py, contact sheet, written critique against the brief, validate, a patch loop capped at 8 iterations, human approval, game_ready + LODs, export. Budget 4000-15000 tris, one material, 2048 atlas. Trigger on "/asset-building", "model a building called X", "build a hero building", or any request to model a building from scratch.
---

# asset-building

Produces one deliverable set under `out/assets/<name>/`: `<name>.glb`, `asset.json`,
`recipe.py`, `textures/`, `contact_sheet.png`. Budget: 4000-15000 tris (hero_building), one
shared material, 2048 atlas, LOD0/1/2 always shipped. 1 Blender unit = 1 metre, Z-up, origin at
base centre, `+Y` forward - the contact sheet and `validate_asset` both check these, not just
this doc.

All bpy modelling runs in one live Blender session over the `blender-mcp` bridge with the
`asset-forge` MCP tools alongside it for validation, gating and rendering. Call `asset_status`
first if this session hasn't already this session - it lists the Blender executable, output dir
and kits dir, and can go stale if assumed instead of called.

## 1. Brief

Before touching Blender, pin down:

- **name** - lowercase alphanumerics/dashes/underscores only (`assets.recipe.validate_name`),
  becomes the output directory.
- **kind** - `hero_building` (4000-15000 tris).
- **footprint and height in metres** - checked against the 1.8 m human reference every contact
  sheet includes.
- **key features** - roofline, window/door placement, wall material (brick, concrete, plaster),
  roofing, metal trim.

State the brief back in one line before generating anything.

## 2. Model - live bpy session

In `execute_blender_code`, insert the repo root into `sys.path` and import `assets.hardsurface`
(and `assets.materials` for procedural materials) exactly the way every
`assets/blender_scripts/*_selftest.py` does it:

```python
import sys
sys.path.insert(0, r"<repo_root from asset_status>")
from assets import hardsurface as hs
from assets import materials
```

Build with the composable helpers rather than freehand `bpy.ops.mesh.primitive_*` - they already
encode the project's conventions (origin at base centre, outward normals, applied transforms):
`bevelled_box`, `frustum`, `cylinder_between`, `panel_cut`, `array`, `mirror`, `solidify`,
`greeble`, `window_recess`, `door_frame`, `roof_edge`, `ladder_frame`, `stairs`, `cornice`,
`attach_to_surface`. Use `attach_to_surface` for every small hardware/decoration part modelled
as its own closed solid against a host wall or roof (hinges, handles, brackets, sign/lantern
hardware) instead of hand-computing an offset - it gives a `STANDOFF_M` (2 mm) clearance so the
part never sits exactly coincident with, or embedded into, the host, which is what makes
`validate_asset`'s manifold check falsely flag the joined mesh later (af-fte).
Assign materials with `assets.materials` (`concrete_material`, `brick_material`,
`plaster_material`, `metal_material`, `wood_material`, `glass_material`, `roofing_material`, or
`get_material`/`assign_material` against the CC0 index) - leave per-part materials as they are;
consolidating to one shared atlas material is `game_ready_pass`'s job in step 8, not here.

After every snippet `execute_blender_code` runs, call `record_recipe(name, code)` on
`asset-forge` with that same snippet. This is not optional - CLAUDE.md requires every live MCP
session to leave behind a re-runnable `recipe.py`, and skipping a snippet breaks that chain
silently (the gap doesn't show up until someone tries to replay it).

## 3. Contact sheet

```
render_contact_sheet(name)
```

Six orthographic views plus a three-quarter beauty shot and a worm's-eye shot, flat even
lighting, the 1.8 m human reference beside the asset. This is how the agent sees the asset -
read the returned image, don't just check that the call succeeded.

## 4. Written critique

Before moving on, write one paragraph judging the contact sheet against the brief: silhouette
and proportions read correctly next to the human reference, no missing back faces, no
one-angle-only detail, nothing that only looks right from the view it was designed in. State
pass or fail plainly. A fail here is the cheapest point to fix - cheaper than finding the same
problem after `validate_asset` or, worse, after export.

## 5. Validate

```
validate_asset(name, "hero_building")
```

Checks the triangle budget, per-primitive material assignment, watertight/manifold geometry,
outward-facing normals, UV overlap, and applied transforms at the base-centre origin. Read
`ok` and every failing check's `detail` in the `checks` array - it names the exact failure
(e.g. "N triangle(s) wind opposite their vertex normals"), not just a pass/fail flag.

## 6. Patch loop - capped at 8 iterations

If the critique or `validate_asset` found something wrong:

1. `record_patch(name, "hero_building")` on `asset-forge` first. The cap depends on the kind it
   is given (and remembers): 8 for the hero kinds (`hero_building`, `hero_vehicle`), 4 for
   everything else. It raises `GateError` once the cap
   is spent - `asset_gate_status(name)` shows `patches_remaining` at any point. If the cap is
   hit, stop and hand the asset to a human reviewer rather than continuing to patch blind.
2. Make the targeted bpy fix in the same live session (`sys.path` is already set), and
   `record_recipe` it.
3. Go back to step 3 (re-render, re-critique, re-validate) - a patch that isn't re-checked is a
   guess, not a fix.

## 7. Human approval

Once the contact sheet critique and `validate_asset` both pass:

```
approve_asset(name, note)
```

This records that a human reviewer signed off - never call it as a stand-in for that review.
Ask the user to look at the contact sheet and confirm, and pass their note through verbatim.
`asset_gate_status(name)` should show `approved: true` before continuing to export.

## 8. game_ready + LODs + export - a background job, never a live MCP call

The bake passes (atlas, then every LOD) block Blender's only thread for minutes, longer than the
`blender-mcp` bridge's transport timeout, so step 8 never runs inside `execute_blender_code`. The
live session only saves its scene; a headless Blender started as a background shell job does the
rest, with no timeout, and the harness notifies you when it exits.

1. In the live session, save a copy of the scene (every part still named `<name>_*`, each on its
   own per-part materials) and `record_recipe` it:

   ```python
   import bpy
   bpy.ops.wm.save_as_mainfile(filepath=r"<repo_root>/out/assets/<name>/.build/<name>.blend", copy=True)
   ```

2. Launch the job with the Bash tool and `run_in_background: true` (Blender from
   `asset_status`'s `blender_executable`):

   ```bash
   "<blender_executable>" --background "out/assets/<name>/.build/<name>.blend" --factory-startup \
     --python assets/blender_scripts/game_ready_build.py -- "<repo_root>" <name> <kind> \
     "out/assets/<name>/.build/game_ready.json" 2>&1 | grep --line-buffered -E "\[game_ready\]|Error|Traceback"
   ```

   `game_ready_build.py` refuses to run unless `approve_asset` has opened the gate, then runs the
   whole sequence on every `<name>_*` mesh: `apply_transforms`/`recalc_normals`,
   `uv_unwrap_and_pack` while the parts still carry their own materials, the three atlas bake
   passes, `consolidate_material(objects, bake_result=...)`, `enforce_power_of_two_textures`,
   `join` (`generate_lods` needs one object), `generate_lods`, `export_asset(kind=<kind>)`. Do not
   use `game_ready.game_ready_pass(objects)` - it bakes no textures and leaves a blank
   `atlas_material`.

3. Don't poll, and don't start a second job for the same asset while one runs. When the
   notification arrives, read `game_ready.json`: `{"ok": true, ...}` carries the export result
   (glb, textures, triangle counts per LOD), `{"ok": false, "error": ...}` the traceback. Fix the
   cause and relaunch; the job always starts again from the saved .blend, so a failed run leaves
   nothing half-baked behind.

## Done

Run `validate_asset(name, "hero_building")` once more against the final exported glb - the LOD
swap and atlas consolidation can move triangle counts or surface a failure export didn't have
before. Report the `out/assets/<name>/` file list, the gate status
(`asset_gate_status(name)`), and the one-line brief from step 1 so the user can confirm the
result matches what they asked for.
