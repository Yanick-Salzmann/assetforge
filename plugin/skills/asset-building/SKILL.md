---
name: asset-building
description: Model a hero building end to end - brief, live bpy modelling over assets/hardsurface.py, contact sheet, written critique against the brief, validate, a patch loop capped at 8 iterations, human approval, game_ready + LODs, export. Budget 4000-40000 tris, one material, 2048 atlas. Trigger on "/asset-building", "model a building called X", "build a hero building", or any request to model a building from scratch.
---

# asset-building

Produces one deliverable set under `out/assets/<name>/`: `<name>.glb`, `asset.json`,
`recipe.py`, `textures/`, `contact_sheet.png`. Budget: 4000-40000 tris (hero_building), one
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
- **kind** - `hero_building` (4000-40000 tris).
- **footprint and height in metres** - checked against the 1.8 m human reference every contact
  sheet includes.
- **key features** - roofline, window/door placement, wall material (brick, concrete, plaster),
  roofing, metal trim.
- **moving parts** - one row per part, or `none`:

  | part | motion | state | rate |
  |---|---|---|---|
  | `wheel` | spin about Z | `working` | 1 rev / 1.5 s |
  | `wheel` | slow oscillation (coasting) | `idle` | +-0.3 rad, 4 s period |

  `motion` is one of the `assets/motion.py` primitives (spin, oscillate, hinge, crank, gear train,
  wind on a bone chain). `state` is `idle` (ambient, always plays, must loop) or `working` (only
  while the building is occupied). `rate` is the period, rpm or amplitude the filmstrip is judged
  against. Fire, smoke and sparks are FX sockets (`assets/fx.py`), not moving parts.

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
`greeble`, `window_recess`, `door_frame`, `roof_edge`, `gable_roof`, `roof_slope`,
`triangular_gable`, `ladder_frame`, `stairs`, `cornice`, `box_between`, `lathe`, `loft`, `prism`,
`annular_band`, `torus`, `attach_to_surface`. Do not redefine any of these inside a recipe - if a
shape keeps recurring and none fits, add it to `assets/hardsurface.py` with a selftest case. Use `attach_to_surface` for every small hardware/decoration part modelled
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

### Live-session conventions

Learned the hard way across the windmill, tavern, church, tavern props and tractor recipes:

- **Workspace first.** Blender's cwd is its install dir, so `terrain.config` resolves `out/` there
  and `export_asset` fails with a permission error. The first snippet sets
  `os.environ["ASSETFORGE_WORKSPACE"] = r"<repo_root>"` before importing anything from `assets`
  or `terrain` (if `terrain.config` was already imported, `importlib.reload` it).
- **Every snippet is a fresh namespace.** Globals do not survive between `execute_blender_code`
  calls - the windmill re-declared `radius_at` five times, the tractor `bar()` three. Each snippet
  carries its own imports, constants and helper definitions; a helper you need in more than one
  snippet (or more than one asset) belongs in `assets/hardsurface.py`, not copied again.
- **Name every object `<name>_*` the moment it exists.** `door_frame` and `cornice` return
  generically named objects (`door_frame`, `cornice.001`) - rename them immediately. Step 8's
  `game_ready_build.py` collects the asset by that prefix, so a part with any other name is
  silently left out of the bake and the export.
- **Rebuild snippets clean up first.** A snippet that rebuilds a part deletes the old object and
  its mesh (`mesh.users == 0` -> `bpy.data.meshes.remove`) before creating the new one; deleting
  only the object leaves an orphan mesh and the replacement exports as `<part>_mesh.001`.
- **Front is +Y from the first snippet.** The church was modelled facing -Y and needed a late
  snippet rotating every part by pi - decide the facing in the brief and build to it.
- **Overlap, don't butt.** Two solids that only meet edge to edge leave hairline gaps in the
  render (the church needed wedge infills and a roof embed to close its gable/roof joints). Sink
  structural joins a few cm into each other; flush hardware uses `attach_to_surface` (or a
  manual `STANDOFF_M` nudge for helpers that don't, like `door_frame`).
- **Mount points come from the host's geometry.** Compute a small part's position from the same
  function/radius that built its host surface (`radius_at(z)`, a fender arc, the hood section
  table) - a hand-typed height put the tractor's fender lamps 34 cm in the air. Anything small and
  mechanical (lamps, mirrors, pipes, stacks) gets a visible mount: a bracket, stalk, clamp or
  collar tying it to the host.
- **Keep greebles off converging faces.** Pass `greeble` a `face_indices` list that excludes
  faces near an apex or seam (the windmill cap guards the top 0.8 m).
- **New materials go into `assets/materials.py`'s `MATERIAL_BUILDERS`,** never inline in a
  snippet; `importlib.reload(materials)` in the live session after editing it.

### Moving parts

Only when the brief lists any. Each moving part is its own closed object, never joined into the
body, with clearance from everything it sweeps past (`validate_asset` sweeps every clip and fails
a part that sinks into another).

```python
from assets import motion
wheel = bpy.data.objects["<name>_wheel"]
motion.pivot_part(wheel, pivot=(x, y, z), axis="Z")
spin = motion.animate_spin(wheel, period_s=1.5, state="working", clip="wheel_spin")
coast = motion.animate_oscillate(wheel, 0.3, 4.0, state="idle", clip="wheel_coast", channel="rotation")
```

- `pivot_part` moves the origin onto the pivot without moving the geometry and renames the part
  `<name>_wheel_ANIM` - build the part at rest (identity rotation and scale) and keep its
  `<name>_` prefix so step 8 collects it. Parent it to the body (or another moving part) when it
  rides on it.
- Every `animate_*` writes one `<state>_<clip>` action and records its `animations[]` entry on the
  scene (`motion.recorded_clips()`), so the saved .blend carries it to step 8. `crank_linkage` and
  `gear_mesh` record their `mechanisms[]` entries the same way. Re-running a helper with the same
  clip name replaces the clip.
- The contact sheet and filmstrip read the exported glb, so the live preview export passes the
  clips: `export.export_asset(parts, name, kind, animations=motion.recorded_clips(),
  mechanisms=motion.recorded_mechanisms())`. An animated export needs one material per LOD, so
  save the .blend first, then swap every part onto one temporary clay material for the preview -
  the pre-bake sheet is clay anyway. FX sockets from `fx.add_fx_socket` go in with
  `fx=fx.recorded_fx()` and their empties among the parts; step 8 picks them up from the scene.
- Cloth (flags, pennants, banners, awnings) comes from `assets.cloth`, named `<name>_<label>` so
  step 8 collects it: `rig = cloth.awning("<name>_awning", material=mat, edge_centre=..., width=...,
  facing_deg=...)` then `cloth.animate_wind(rig, duration_s, seed)` records an idle skinned clip.
  The rods are static parts; the cloth stays skinned to its rig through the bake and every LOD.
  Pass the rig objects (`rig.objects`) to the preview export with the other parts.
- The 3/4 filmstrip camera is fixed: a part hidden under a roof or awning shows `moving_fraction`
  0.0. Then add a review close-up (hide the occluders, put a temporary marker on the rotor) and
  show it to the reviewer alongside the filmstrips.

## 3. Contact sheet

```
render_contact_sheet(name)
```

Six orthographic views plus a three-quarter beauty shot and a worm's-eye shot, flat even
lighting, the 1.8 m human reference beside the asset. This is how the agent sees the asset -
read the returned image, don't just check that the call succeeded.

### What the contact sheet can and can't tell you

- **Before step 8 it is untextured.** Procedural node materials don't survive the glTF export the
  sheet renders from, so every pre-approval sheet is clay. Judge form and mechanism from it, and
  say in the critique that colour and material read are unverified until after the bake.
- **Its tiles are too small for mechanism detail.** For anything you changed or anything
  articulated (hitches, linkages, brackets, engine bays, window frames), add a targeted close-up:
  a temporary camera and a `BLENDER_WORKBENCH` render to `out/assets/<name>/review_<what>.png`,
  then delete the camera. Set `hide_render = True` on any non-asset object first (the default
  `Cube` renders even when hidden in the viewport). These close-ups found the floating lamps, the
  bare hitch and the hidden engine on the tractor - the sheet showed none of them.
- **Check what is actually visible.** Detail modelled behind something else is wasted triangles;
  render from where a player will stand, not only from where the part was designed.

### Filmstrips - every clip of an animated asset

```
render_animation_sheet(name, clip)
```

One per `animations[]` clip: 8 frames across the loop from one 3/4 camera plus a motion-trail
tile. The contact sheet shows the rest pose only; motion is judged here.

## 4. Written critique

Before moving on, write one paragraph judging the contact sheet against the brief: silhouette
and proportions read correctly next to the human reference, no missing back faces, no
one-angle-only detail, nothing that only looks right from the view it was designed in. For an
animated asset, judge every filmstrip against the brief's moving-parts table too: each part turns
about its own pivot and stays attached, nothing passes through anything else, the loop has no pop
at the seam (last frame flows into the first), the rate matches the brief, and idle motion is
calm enough to loop forever. State pass or fail plainly. A fail here is the cheapest point to fix - cheaper than finding the same
problem after `validate_asset` or, worse, after export.

## 5. Validate

```
validate_asset(name, "hero_building")
```

Checks the triangle budget, per-primitive material assignment, watertight/manifold geometry,
outward-facing normals, UV overlap, and applied transforms at the base-centre origin. With clips it
also checks that they match `animations[]`, every `_ANIM` node rests at its pivot with identity
rotation and scale, loops close in value and velocity, clip budgets, a swept collision over every
rigid clip, and every `mechanisms[]` coupling. Read
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

For an animated asset the user also watches every filmstrip, and the approval names those clips:

```
approve_asset(name, note, clips=["working_wheel_spin", "idle_wheel_coast"])
```

Step 8 refuses to export any recorded clip the approval does not name, so a clip added or
renamed after approval needs a fresh review and a fresh `approve_asset`.
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
   `join` of the static parts (`generate_lods` needs one body), `generate_lods`,
   `export_asset(kind=<kind>)`. Do not use `game_ready.game_ready_pass(objects)` - it bakes no
   textures and leaves a blank `atlas_material`.

   Moving parts (`<name>_*_ANIM`) keep their pivot, stay separate nodes and are never culled as
   hidden; `generate_lods(body, moving=...)` gives each LOD its own reduced `<part>_LOD<n>_ANIM`
   copy driven by the same clips, and the export carries the recorded clips and mechanisms.

3. Don't poll, and don't start a second job for the same asset while one runs. When the
   notification arrives, read `game_ready.json`: `{"ok": true, ...}` carries the export result
   (glb, textures, triangle counts per LOD), `{"ok": false, "error": ...}` the traceback. Fix the
   cause and relaunch; the job always starts again from the saved .blend, so a failed run leaves
   nothing half-baked behind.

## Done

Run `validate_asset(name, "hero_building")` once more against the final exported glb - the LOD
swap and atlas consolidation can move triangle counts or surface a failure export didn't have
before. For an animated asset, re-render one filmstrip per clip from the final glb and check the
textured parts still turn about their pivots. Report the `out/assets/<name>/` file list, the gate status
(`asset_gate_status(name)`), and the one-line brief from step 1 so the user can confirm the
result matches what they asked for.
