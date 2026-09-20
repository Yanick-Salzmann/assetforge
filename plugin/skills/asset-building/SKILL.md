---
name: asset-building
description: Model a hero building end to end - brief, live bpy modelling over assets/hardsurface.py, contact sheet, written critique against the brief, validate, a patch loop capped at 4 iterations, human approval, game_ready + LODs, export. Budget 4000-15000 tris, one material, 2048 atlas. Trigger on "/asset-building", "model a building called X", "build a hero building", or any request to model a building from scratch.
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

## 6. Patch loop - capped at 4 iterations

If the critique or `validate_asset` found something wrong:

1. `record_patch(name)` on `asset-forge` first. It raises `GateError` once the 4-iteration cap
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

## 8. game_ready + LODs + export

Same live session. **Do not call `game_ready.game_ready_pass(objects)` alone** - it bakes no
textures. Internally it calls `consolidate_material(objects, material_name)` with no
`bake_result`, which produces a UV-correct but *blank* `atlas_material` - every carefully chosen
CC0/procedural material gets silently replaced with an untextured grey slot. The sequence that
actually preserves textures (matching the working reference in
`assets/blender_scripts/demo_showcase_build.py`) is:

```python
from assets import hardsurface as hs, game_ready, atlas_bake, lod, export

for obj in objects:
    hs.apply_transforms(obj)
    hs.recalc_normals(obj)

game_ready.uv_unwrap_and_pack(objects)                        # while each part still carries its
                                                               # own per-face material
bake_result = atlas_bake.bake_atlas(objects, atlas_size=game_ready.ATLAS_SIZE)
material = game_ready.consolidate_material(objects, bake_result=bake_result)
game_ready.enforce_power_of_two_textures(material, game_ready.ATLAS_SIZE)

bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
bpy.ops.object.join()                                         # generate_lods needs ONE object;
lod0_object = bpy.context.view_layer.objects.active           # a multi-part asset is never
                                                               # auto-joined anywhere upstream
lod_set = lod.generate_lods(lod0_object)                      # LOD1/2 decimated + normal-baked from LOD0
result = export.export_asset(lod_set.objects, name=name, kind="hero_building")
```

Order matters: UV-unwrap-and-bake needs the *original* per-part materials, so it must run before
`consolidate_material` replaces them; `join` needs everything already on the one shared
`atlas_material` (do it after consolidation, not before); `generate_lods` needs a single
already-materialed LOD0 object; `export_asset` writes `textures/` from whatever materials are on
the objects it's given. `record_recipe` each of these calls too, same as every step 2 snippet.

**`atlas_bake.bake_atlas` is long-running and will likely report "failed" even when it
isn't.** It runs 5 sequential full-resolution Cycles bake passes (albedo, normal, AO, roughness,
metallic) across every object - on an asset with 100+ parts this routinely takes several minutes.
`bpy.ops.object.bake` blocks Blender's single Python/UI thread for its whole duration, and the
`blender-mcp` bridge's own transport has a shorter timeout than that, so the tool call comes back
as a connection-timeout failure while Blender keeps computing correctly in the background - this
is a transport artifact, not a crash, and does not mean the bake needs to be re-run. To avoid
tripping the transport timeout in the first place (not to make the bake faster, which isn't the
goal - only to stop it from spuriously erroring), issue one `execute_blender_code` call per bake
pass instead of one call running `bake_atlas` end to end, e.g. call `uv_unwrap_and_pack` in its
own snippet, then separately drive albedo/normal/AO/roughness/metallic (mirroring
`atlas_bake.bake_atlas`'s internal steps) each in their own call. If a call times out anyway,
do not blindly retry it - first check state with a small, fast snippet (e.g. list
`bpy.data.images` for `albedo`/`normal`/`orm`, or check whether `atlas_material`'s node tree
already has the bake wired in) before deciding whether to resume or re-run; re-running a
completed bake step duplicates images under `.001` suffixes that `export_asset` will then ship
alongside the real ones.

## Done

Run `validate_asset(name, "hero_building")` once more against the final exported glb - the LOD
swap and atlas consolidation can move triangle counts or surface a failure export didn't have
before. Report the `out/assets/<name>/` file list, the gate status
(`asset_gate_status(name)`), and the one-line brief from step 1 so the user can confirm the
result matches what they asked for.
