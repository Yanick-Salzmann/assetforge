---
name: asset-vehicle
description: Model a non-building hard-surface asset (vehicle or machinery) end to end - same loop as asset-building, adjusted brief and 5000-20000 tri budget. Brief, live bpy modelling over assets/hardsurface.py, contact sheet, written critique, validate, a patch loop capped at 4 iterations (8 for hero_vehicle), human approval, game_ready + LODs, export. Trigger on "/asset-vehicle", "model a vehicle called X", "build a machinery asset", or any request to model a car, truck, tank, crane, or other non-building hard-surface asset.
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
  separate `machinery` budget row, use `vehicle` for both. A close-up centrepiece vehicle (the
  player sees it from a metre away: engine bay, hitch, hardware) is `hero_vehicle` (15000-60000
  tris); pass that kind to `validate_asset` and `export_asset` instead of `vehicle`.
- **overall dimensions in metres** - length/width/height, checked against the 1.8 m human
  reference every contact sheet includes.
- **silhouette and mechanism** - body shape, wheel/track/leg count and placement, articulated
  parts (turret, boom, bucket, doors) if any, panel seams and vents, primary material read
  (painted metal panels, raw steel, rubber, glass).
- **moving parts** - the same table as `asset-building` step 1 (part, motion, state, rate), or
  `none`: spinning wheels and rotors, a swinging boom, a cranking arm. Articulated parts that
  only pose (a turret aimed once) stay static.

State the brief back in one line before generating anything.

## 2. Model - live bpy session

Same import and helper set as `asset-building`:

```python
import sys
sys.path.insert(0, r"<repo_root from asset_status>")
from assets import hardsurface as hs
from assets import materials
```

`cylinder_between` covers axles, rims and hydraulic rams; `box_between` link arms, struts and
tie rods; `lugged_tyre` treaded tyres; `loft` hoods, hulls and housings through cross-sections;
`prism` chassis rails and profiled plates; `annular_band` mudguards, wheel arches and heat
shields; `torus` hitch eyes and lifting rings; `array` repeats wheels/road wheels
along a side; `mirror` builds the opposite side from one modelled half; `frustum` covers tapered
noses, hulls and counterweights; `panel_cut` cuts vents, hatches and window openings;
`greeble` adds panel-line and bolt detail at range. Assign materials with `assets.materials`
(`metal_material` for panels, `glass_material` for windows/canopy, plus procedural node groups
for rubber/tread where none exists yet - add one in `assets/materials.py` rather than faking
tread with a metal tint) - leave per-part materials as-is, consolidation to one atlas is
`game_ready_pass`'s job in step 8.

`record_recipe(name, code)` after every snippet, same as building modelling - not optional.

Follow `asset-building`'s **Live-session conventions** (step 2) - workspace env var, fresh namespace per snippet, `<name>_*` naming, cleanup before rebuilds, +Y front, overlapping joins, geometry-derived mount points, greeble face guards, materials in `MATERIAL_BUILDERS`.

## 3. Contact sheet

```
render_contact_sheet(name)
```

Same eight views. Wheels/tracks and any articulated parts should read correctly from the worm's-
eye and three-quarter shots too - a mechanism that only looks right from the side it was designed
in is the same failure mode a building's one-angle-only detail is.

See `asset-building` step 3, **What the contact sheet can and can't tell you**: pre-bake sheets are clay, add Workbench close-ups for mechanism detail, and check visibility.

Model moving parts as in `asset-building` step 2, **Moving parts** (`motion.pivot_part`, the
`animate_*` helpers, `gear_mesh`/`crank_linkage` for coupled parts, the preview export with
`motion.recorded_clips()`), and render `render_animation_sheet(name, clip)` for every clip.

## 4. Written critique

One paragraph against the brief: silhouette and proportions against the 1.8 m human reference,
wheel/track contact with the ground plane, no missing back or underside faces, mechanism reads
from every angle. With clips, judge every filmstrip as in `asset-building` step 4: parts turn
about their own pivots and stay attached, no interpenetration, no seam pop, coupled parts stay
coupled, rate matches the brief. State pass or fail plainly.

## 5. Validate

```
validate_asset(name, "vehicle")
```

Same checks as a building at the vehicle tri budget: triangle count, per-primitive material
assignment, watertight/manifold, outward normals, UV overlap, applied transforms at base-centre
origin, and with clips the animation and mechanism checks.

## 6. Patch loop - capped at 4 iterations, 8 for `hero_vehicle`

Same as `asset-building` step 6: `record_patch(name, kind)` first, with the kind from step 1 (raises `GateError` past the cap -
`asset_gate_status(name)` shows `patches_remaining`), fix in the live session and
`record_recipe` it, back to step 3.

## 7. Human approval

```
approve_asset(name, note)
```

Ask the user to look at the contact sheet and confirm; pass their note through verbatim.
`asset_gate_status(name)` should show `approved: true` before continuing. For an animated asset
the user also watches every filmstrip and the approval names those clips -
`approve_asset(name, note, clips=[...])`; step 8 refuses any recorded clip it does not name.

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

Run `validate_asset(name, "vehicle")` once more against the final exported glb, then report the
`out/assets/<name>/` file list, `asset_gate_status(name)`, and the one-line brief from step 1.
