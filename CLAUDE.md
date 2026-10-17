# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

## Git Policy Override

Overrides the Conservative default above: after closing a bead (`bd close`), commit the
resulting changes and push, without waiting for separate approval each time. Still subordinate
to any explicit in-session "don't commit" / "don't push" instruction, and still never force-push,
skip hooks, or push to a branch other than the one currently checked out.

## Build & Test

`uv` is the only entry point. On Windows `uv` may not be on PATH; it lives at
`%APPDATA%\Python\Python314\Scripts\uv.exe`.

```bash
uv sync --extra dev      # create/refresh .venv from uv.lock
uv run pytest            # run the whole test suite
uv run pytest -m "not slow and not gpu and not blender"   # fast subset
uv run python -m terrain.server    # start the terrain-forge MCP server by hand
```

Never call `pip` or a bare `python` against the project. Add dependencies with
`uv add`, dev dependencies with `uv add --optional dev`, and commit `uv.lock`.

`torch` resolves from the `cu130` PyTorch index on Windows and Linux, and from
default PyPI on macOS. The build target is this Windows box: RTX 4070, 12 GB
VRAM, CUDA. CPU is the portable fallback — treat a CPU run as a correctness
check, not a performance measurement. Apple Silicon and MPS are a deferred
secondary target.

Never name a backend outside `terrain/device.py`. Every module that touches a
tensor takes the device that `terrain.device.resolve()` returned; the resolver
picks cuda, then mps, then cpu, and `ASSETFORGE_DEVICE` overrides it.

12 GB of VRAM is a real ceiling. A 17-channel float32 stack costs 272 MiB at
2048² and 1.1 GiB at 4096²; 8192² does not leave room for the pipe model's flux
fields beside it. 4096² is the working ceiling.

## Architecture Overview

Two subsystems that do not share a runtime. See `PLAN.md` for the full design.

- `terrain/` — headless numpy/torch pipeline. Blender is never in this loop
  except for the final beauty render. Everything downstream of erosion reads
  from one `float32[H, W, C]` channel stack.
- `assets/` — hard-surface modelling driven live in Blender 4.5 over MCP.
- `library/` — CC0 material and kit downloads. Binaries are gitignored; the
  lockfile beside them is what makes a fresh clone reproducible.
- `out/terrain/<name>/`, `out/assets/<name>/` — generated deliverables only.
  Nothing here is ever hand-edited or committed.

Three MCP servers are declared in `.mcp.json`: `terrain-forge` (ours),
`asset-forge` (ours) and `blender-mcp` (the community bridge). A server that is
built but never registered is a server that never runs.

## Conventions & Patterns

### Blender

Blender 5.2 LTS (5.1 is the floor — the official Blender Lab MCP add-on declares
`blender_version_min = "5.1.0"`). The bridge is the Blender Foundation's own
server from `projects.blender.org/lab/blender_mcp`, not the `blender-mcp`
package on PyPI, which is a different community project under the same name.
`.mcp.json` therefore installs it from git:

```
uvx --from git+https://projects.blender.org/lab/blender_mcp.git#subdirectory=mcp blender-mcp
```

The add-on lives in that repo under `addon/blender_mcp_addon/` and must be
installed and enabled in Blender before any tool works. It gives us
`execute_blender_code`, `execute_blender_code_for_cli` (background Blender),
viewport and thumbnail renders, screenshots and API/manual doc search. Contact
sheets, validation, game-ready and export are ours, in `asset-forge`.

`ASSETFORGE_BLENDER` overrides executable discovery.

### Code style

- No comments. Names carry the meaning.
- Braces on every block in brace languages; opening brace on the statement line.
- One statement per line.
- `from __future__ import annotations` at the top of every module.
- Pure numeric code stays free of I/O so it can be unit tested without fixtures.

### Determinism

Every generator takes an explicit integer `seed` and derives sub-seeds from it.
No module reads global RNG state. Regenerating from `terrain.json` must
reproduce the same bytes — Phase 5 tests this, so do not break it earlier.

### Terrain conventions

- Heightmaps: 16-bit grayscale PNG, non-color data, `0` = minimum elevation and
  `65535` = maximum. The metre range lives in `terrain.json`, never in the image.
- World scale is metres. `terrain.json` carries `world_size_m`,
  `height_range_m`, `metres_per_pixel`, `sea_level_m`.
- Channel stack names are fixed and lowercase: `height`, `slope`, `curvature`,
  `flow`, `deposition`, `wear`, `wetness`, `water`, `water_depth`, `bedrock`,
  `strata`, `temperature`, `moisture`, `patchiness_fine`, `patchiness_mid`,
  `patchiness_coarse`, `shore`. All are normalised to `[0, 1]` except `height`, which is
  normalised too — the metre mapping is applied only at export. `shore` is 1 on the
  waterline of sea, lakes and rivers and falls to 0 `WaterParams.shore_reach_m` inland
  and offshore (a blurred `water` mask), for beach and bank materials.
- Splat masks pack 4 layers per RGBA texture: `splat_0.png` holds layers 0-3,
  `splat_1.png` layers 4-7. Layer weights across all textures sum to 1 per pixel.
- Splat masks carry linear weights only. Height-based blending is a shading-time step:
  each layer's `blend_contrast` and the biome's `blend_depth` are recorded in
  `terrain.json` (`splat.layers[].blend_contrast`, `splat.blend_depth`, and the formula in
  `splat.height_blend`) so an engine can reproduce it against the material height maps.
- Scatter species are declared per biome as `[species.<name>]` tables beside the splat
  layers (`terrain/splat.py` `Species`, evaluated by `terrain/scatter.py`). Each carries a
  `kind` (one of `config.SCATTER_KINDS`: conifer, broadleaf, shrub, grass, flower, cactus,
  rock, debris), a `density` rule in the splat rule language that may also read the biome's
  blended layer weights by layer name, `spacing_m`, `scale = [min, max]`, and optional
  `slope_align` (0 upright .. 1 surface-normal), `exclusion_m` (clearance from every other
  species), `water_buffer_m` and `meshes` (1-4 `<pack>/<model>` kit props from
  `library/kits.lock.json`, the mesh variants). Layer names may not shadow channel names.
- Mesh variants are resolved by `terrain/vegetation.py`: a prop from a metric source
  (`vegetation.METRIC_SOURCES`, i.e. our own generated packs) keeps its modelled size
  (`unit_scale` 1); any other is scaled so its largest extent equals
  `vegetation.KIND_SIZE_M[kind]` at instance scale 1. Every variant must stay within
  `vegetation.KIND_TRI_BUDGET[kind]` (cactus 1200, conifer/broadleaf 600, rock 500, debris 400,
  grass 250, shrub/flower 200) - `tests/test_vegetation.py` enforces both.
- Generated scatter props are a kit pack with `source = "assetforge"` in `library/kits.toml`:
  the slug is the asset-name prefix, and `python -m library.generated` copies every approved
  `out/assets/<slug>_<model>/` into `library/kits/<pack>/<model>.glb` and locks it (sha256 and
  recipe path) in `kits.lock.json`. The `scatter` pack is rebuilt by
  `blender --background --python assets/blender_scripts/scatter_set_build.py` (geometry from
  `assets/organic.py`), then approved and pulled.
- Environment organics (stumps, root clusters, mushrooms, corals) are the `organics` pack (slug
  `organic`, kind `organic_prop`, LOD0/1/2), rebuilt by
  `blender --background --python assets/blender_scripts/organic_set_build.py` from the
  `assets/organic.py` builders (modelled into the ground and trimmed flat by `ground_cut`), then
  approved and pulled. Kit tris/bbox in `kits.lock.json` count LOD0 only (`library/gltf.py`).
- Vegetation (trees, shrubs, ferns, grass, flowers) is the `flora` pack, rebuilt by
  `blender --background --python assets/blender_scripts/flora_build.py`: bark tubes and
  alpha-cutout foliage cards from `assets/flora.py` (pure Python, `flora.MODELS` maps each model
  to its scatter kind) over one shared 2048 RGBA atlas painted by `assets/foliage_atlas.py` (bark
  strip from `bark_cc0`, 12 procedural foliage cells with an 8 px transparent gutter). One
  material, glTF `alphaMode` MASK (alpha through a Math Round node), double-sided; card normals
  lean outward from the plant. Exported with kind `vegetation`.
- Scatter masks are single-channel 8-bit, one per species: `scatter_<species>.png`.
  `terrain.json` `scatter[]` records `species`, `kind`, `path`, `density`, `spacing_m`,
  `scale`, `slope_align`, `exclusion_m`, `water_buffer_m`, `instances`, `count`,
  `placed_spacing_m`, `variants[]` (`mesh`, `glb` relative to `library/`, `tris`,
  `unit_scale`) (schema_version 3), validated by `terrain/manifest.py`. Export removes
  stale `scatter_*` files from the map directory.
- Instances are placed in Python, never in the engine or viewer (`terrain/placement.py`):
  Poisson-disc per species in declaration order, seeded by `cfg.derive_seed("scatter",
  species)`, thinned by the 8-bit mask, capped per kind (`placement.KIND_CAPS`, which can
  widen the spacing - the result is `placed_spacing_m`), and dropped within
  `max(exclusion_m, other.exclusion_m)` of any earlier species' instance. Written as
  `scatter_<species>.bin`: little-endian float32 records `(x, y, z, yaw, scale, variant)`, x along
  image columns and y along image rows in metres from the map centre, z the elevation in
  metres sampled bilinearly from `height.png`, variant an integral index into `variants[]`
  (always 0 without meshes). Slope alignment is applied at load time from the heightfield
  normal and `slope_align`. The viewer falls back to primitive geometry for any variant whose
  GLB is not on disk.
- `colour_macro.png` is an 8-bit RGB albedo multiplier (`byte / 128`, neutral 128) built
  from `patchiness_coarse`/`patchiness_mid`, `moisture` and `wetness` (`terrain/macro.py`).
  Low-frequency, no lighting or AO. `terrain.json` `colour_macro` records the path and the
  distance fade (`near_strength` up close, full strength past `far_distance_m`).
- `water_surface.png` is the sea, lake and river surface elevation, encoded exactly like
  the heightmap (16-bit, `0..65535` over `height_range_m`) and equal to the terrain where
  dry. Optional in `terrain.json` (`water_surface`); without it the viewer falls back to
  flat water at `sea_level_m`.
- Lighting is per biome: an optional `[atmosphere]` table in `biomes/*.toml`
  (`terrain/atmosphere.py`: `latitude_deg`, `day_of_year`, `time_of_day_h`, `turbidity`,
  `rayleigh`, `visibility_km`, `haze_colour`, `ground_albedo`) is recorded as `terrain.json`
  `atmosphere` (optional on read). The viewer (`viewer/static/sky.js`) derives everything else
  from it: sun and full-moon positions from latitude/day/time, sun colour from air mass, a
  single-scattering Rayleigh/Mie sky in the same units as the sun light, the sky-captured
  environment map, Koschmieder haze thinned with camera altitude, white balance and exposure
  adaptation, and shadows. `#time=<hours>` in the viewer URL pins the time of day. The
  Blender beauty render reads the same entry (`terrain/beauty.py` `lighting_args`): sun and
  moon from `terrain.atmosphere.sun_position`/`moon_position` (map +y is north, i.e. the image
  top), a multiple-scattering sky texture with its sun disc as the key light, a moon lamp at
  night, and per-view metered exposure. It is dressed with the same instances export writes
  (`export.scatter_placements`), instanced per mesh variant by geometry nodes with the viewer's
  per-kind sink, slope alignment and reach (`beauty.KIND_SINK_M`, `beauty.KIND_REACH_M`).
  Exported buildings (`out/assets/*` of a building kind) stand on the viewer's own sites: `terrain/sites.py`
  is a numpy port of the viewer's site picker (`buildingSites`, `groundPoint`, `clearScatter`) and must
  stay in step with `viewer/static/terrain.js`; `beauty.place_buildings` feeds it and the render imports
  each building's LOD0 at its site and yaw with scatter cleared around it. Its fx sockets run as stills
  (`beauty.fx_args`, `fx_state` idle by default like the viewer): smoke/steam as a Principled Volume tube
  along the viewer particle's mean path bent downwind by the biome `[wind]`, fire as an emissive volume
  plus a point light, sparks as emissive specks, flicker_light as a point light.
- Wind is per biome: an optional `[wind]` table in `biomes/*.toml` (`terrain/wind.py`: `from_deg`,
  the compass bearing it blows from, clockwise from map +y north; `strength`, 1 = clips as authored;
  `gust_period_s`) recorded as `terrain.json` `wind` (optional on read). The viewer
  (`viewer/static/wind.js`) turns every free-flying cloth chain on a terrain building (a skinned
  `<name>_bone<i>` chain whose bone Z, the cross axis, is vertical: flags and pennants, not banners or
  awnings) about its pole so its free edge points downwind, plays idle skinned clips at
  `max(0.25, strength)` speed, and leans vegetation kinds (conifer, broadleaf, shrub, grass, flower)
  downwind in the vertex shader (and in their shadow depth pass) with gusts travelling across the map; smoke and
  steam fx on terrain buildings drift downwind at 1.5 m/s per unit `strength` (`fx.js` `WIND_M_S_PER_STRENGTH`). `window.viewerState.wind`
  reports it and `buildings[].cloth` the turned rigs.
- Output path is `out/terrain/<name>/`. Previews are kept as the visual
  changelog, not deleted between iterations.

### Asset conventions

Enforced by `validate_asset`, not by prose — if you change one, change the
validator too.

- Blender is Z-up; the glTF exporter converts to Y-up on write.
- 1 Blender unit = 1 metre. Check against a 1.8 m human reference.
- Origin at base centre. `+Y` is forward.
- All transforms applied. Normals outward. Meshes closed and manifold.
- Small hardware/decoration parts modelled as their own closed solid against a host
  surface (hinges, handles, brackets, sign or lantern hardware, cornice trim) get a
  minimum 2 mm standoff from that surface — `assets.hardsurface.attach_to_surface` /
  `STANDOFF_M`. Never place one exactly coincident with, or embedded into, the host:
  the final `bpy.ops.object.join()` concatenates every part into one primitive, and
  `validate_asset`'s manifold check re-welds vertices by rounded position across that
  whole primitive, so a coincident or embedded part's vertices weld onto the host's
  own and a harmless touching seam reads as genuine non-manifold geometry (af-fte).
- Assets are viewed from any angle: no billboards, no imposters, no missing back
  faces, no one-angle-only detail.
- Exception: scatter vegetation (export/validate kind `vegetation`) is open alpha-cutout
  cards, so `validate_asset` skips the closed/manifold and UV-overlap checks for it; an
  octahedral impostor is allowed as its far LOD (af-8m0.9). Hero assets and props keep the rule.
  Cycles renders of cutout foliage need `transparent_max_bounces` well above the default 8.
- RTS kinds (`unit`, `creature`, `organic_prop`) ship LOD0/1/2: `validate_asset` fails a
  missing LOD or any step above `validate.LOD_STEP_MAX_RATIO` (0.75) of the previous LOD's tris.
- Team colour (`assets/team_colour.py`): an atlas-sized PNG at `textures/team_mask.png` on the
  asset's `TEXCOORD_0` layout, red channel = mask, written by `export_asset(..., team_mask=img)`
  and recorded in `asset.json` `team_colour` (`mask`, `texcoord`, `channel`, `formula`). The
  engine shades `albedo * mix(1, team_rgb, mask)`, so paint the masked albedo light neutral
  grey. Required for `unit`, optional for `creature`, refused for every other kind; validated
  as power-of-two with at least 0.1% coverage.
- Animation (`assets/animation_manifest.py`): `asset.json` `animations[]`, one entry per glTF clip
  with `name`, `state`, `kind`, `duration_s`, `fps`, `loop`, `nodes`; absent means static.
  `name` is `<state>_<clip>` and equals the Blender action/NLA strip name (the exporter uses it as
  the clip name). `state` is `idle` (ambient, must loop) or `working` (only while the building is
  occupied). `kind` `rigid` animates `<part>_ANIM` nodes, each with its origin at its pivot;
  `skinned` lists the animated bones. `duration_s * fps` is a whole number of frames.
  Rigid clips are built with `assets/motion.py`: `pivot_part` moves a part's origin to its pivot
  and names it `<part>_ANIM`; `animate_spin/oscillate/hinge/crank/gear_train` (and `animate_wind`, an idle skinned clip on a bone chain from `animation.wind_wave`) key the
  `assets/animation.py` primitives one sample per frame into one `<state>_<clip>` action (a slot
  per part, pushed to NLA) at the scene fps and return the `animations[]` entry.
  `export_asset(..., animations=[...])` keeps the hierarchy: top-level objects go under a root node
  named after the asset that carries the base-centre shift (measured in rest pose), one material
  per LOD, and the glb's clips must match the entries before asset.json is written.
  LODs of an animated asset: `game_ready_pass` leaves `_ANIM` transforms unapplied and never culls
  their faces as hidden; `lod.generate_lods(body, moving=[...])` reduces each `<part>_ANIM` into a
  `<part>_LOD<n>_ANIM` copy under that LOD's body (same pivot, same clips through the shared action
  slot, one material per LOD), and export appends the copies to each entry's `nodes`. `write_clip`
  records every entry on the scene (`motion.recorded_clips()`), so `game_ready_build.py` exports
  `<name>_*_ANIM` parts with their clips from the saved .blend. Export with
  `export_animation_mode="ACTIONS"`, `export_force_sampling` and `export_anim_slide_to_zero`
  (clips start at t=0); `export_apply` keeps the armature skin. `validate_asset` checks the schema and
  (`assets/animation_checks.py`) that every entry matches a glb clip and vice versa, rigid
  `_ANIM` nodes rest at their pivot with identity rotation/scale, looping clips close in value and
  velocity, bones/clip length/keyframes stay within `MAX_BONES`/`CLIP_MAX_S`/`CLIP_MAX_KEYFRAMES`,
  and a swept-collision pass over each rigid clip finds no moving part sinking more than
  `CONTACT_TOLERANCE_M` into another LOD0 part beyond what already overlaps at rest. Skinned mesh
  nodes are measured at their bind pose.
  Couplings (`assets/mechanism_manifest.py`, `assets/mechanism_checks.py`): `asset.json`
  `mechanisms[]` on rigid clips - `gears` (`gears: [{node, teeth}]`, each meshing with the next:
  parallel axes, opposite turn, tooth ratio within `GEAR_SLIP_TEETH`) and `linkage`
  (`joints: [{node, on}]`, `node`'s origin stays pinned in `on`'s frame within
  `LINKAGE_TOLERANCE_M`). `motion.gear_mesh`/`motion.crank_linkage` return the entries for
  `export_asset(..., mechanisms=[...])`; `validate_asset` samples every frame. The viewer (`viewer/static/animation.js`) plays clips by
  state: `idle` plays the idle clips, `working` the working clips plus every idle clip whose nodes
  no working clip touches; `#anim=<state>&clip=<name>` pins it, `window.viewerState.animation`
  reports it. Buildings on a terrain play idle with a per-instance phase hashed from name and index.
- Cloth (`assets/cloth_shapes.py` pure, `assets/cloth.py` bpy): `cloth.flag/pennant/banner/awning(name,
  material=..., **params)` build a closed thin-slab panel `<name>_cloth` (front/back UV islands at one
  density, rim collapsed), skinned to a bone chain `<name>_rig` / `<name>_bone<i>` running from the
  attached edge (pinned to bone 0) to the free edge, each bone's local Z across the cloth, plus closed
  rods `<name>_rod<i>` with the cloth `STANDOFF_M` off them. `cloth.animate_wind(rig, duration_s, seed)`
  keys `idle_wind_<name>` with the per-kind `cloth_shapes.MAX_BEND`; export `rig.objects` with the
  entry. `assets/blender_scripts/cloth_selftest.py` exports and validates every type. On a
  building, build it as `<asset>_<label>` so `game_ready_build.py` collects it: armature-deformed
  meshes (`game_ready.is_skinned_part`) stay out of the static join, keep their bind pose, are never
  culled as hidden, and `lod.generate_lods(skinned=[...])` gives each LOD an unreduced `<cloth>_LOD<n>`
  copy on the same rig; the rods join the body and the skinned clips export with the rigs.
- FX sockets (`assets/fx_manifest.py`, `assets/fx.py`): `asset.json` `fx[]`, one entry per emitter
  with `node`, `type` (fire, smoke, sparks, steam, flicker_light), `state` (idle/working), `rate`
  (per second), `size` (m) and `colour` (linear RGB). `node` is an empty named `FX_<type>[_<label>]`
  at the emitter transform (local +Z in Blender = emit direction), made by `fx.add_fx_socket` and
  passed to `export_asset(..., fx=[...])` among the objects; `add_fx_socket` also records the entry on
  the scene (`fx.recorded_fx()`), so `game_ready_build.py` exports the sockets from the saved .blend. `validate_asset` fails an entry without
  its node, a socket node without an entry, a socket carrying a mesh, or one outside the bbox.
  The viewer (`viewer/static/fx.js`) draws them: pooled billboard particles in the socket's frame per
  type, plus a flickering point light for `fire` and `flicker_light`; `idle` entries always run,
  `working` ones follow the state picker (also shown for assets with fx but no clips), and
  `window.viewerState.fx` reports each socket's live particle count. Terrain buildings run idle fx.
- No baked lighting and no AO darkening in the albedo. The engine lights it.
- Power-of-two textures, 2048 atlas, one material per asset.
- Atlas layout (`game_ready.uv_unwrap_and_pack`, af-alp): islands are packed in atlas pixels at
  one uniform texel density with a `game_ready.gutter_px` gutter (16 px at 2048, 8 px at 1024/512)
  on a 4 px block grid, and every bake fills each gutter from its own island only
  (`atlas_bake.dilate_gutters` with the returned `AtlasLayout`) - pass `layout=` to every bake.
  Faces nobody can see (downward on the base plane, or buried inside another part) keep their
  geometry but are collapsed to a zero-area UV point. The side of every revolved part (lathe, frustum,
  cylinder_between; 8+ segments) is unwrapped as a few continuous strips (`assets/uv_strips.py`,
  u = angle x band radius, v = profile arc length) instead of one island per normal direction. `validate_asset` fails islands closer than
  8 px (scaled to texture size), gutter texels carrying another island's colour, and a p95/p5
  texel density ratio above 2.
- Canonical export is `.glb`, with `asset.json` and `recipe.py` beside it, in
  `out/assets/<name>/`.
- Every live MCP session appends the `bpy` it executed to `recipe.py` so the
  session leaves behind something re-runnable under
  `blender --background --python`.

### Budgets

| Kind | Tris | Notes |
|---|---|---|
| Hero building | 4000-40000 | LOD0/1/2 always shipped |
| Prop | 200-1500 | 6-12 per set, one atlas per set |
| Vehicle / machinery | 5000-20000 | same path as buildings |
| Hero vehicle | 15000-60000 | `hero_vehicle`: close-up centrepiece vehicles |
| Unit | 300-4000 | `unit`: RTS humanoid units; LOD0/1/2, team mask required |
| Creature | 500-8000 | `creature`: animals and monsters; LOD0/1/2, team mask optional |
| Organic prop | 100-2000 | `organic_prop`: stumps, roots, mushrooms, corals; LOD0/1/2 |

### The visual loop

An agent cannot judge what it cannot see. Never declare a generated result good
without rendering it and reading the render back: hillshade and false-colour
splat for terrain, the 6-view contact sheet for assets, and the `render_animation_sheet` filmstrip (8 frames of one clip from one 3/4 camera plus a motion-trail tile, `assets/animation_sheet.py`) for every clip of an animated asset. Asset iteration is
capped at 4 rounds (8 for the hero kinds, `hero_building` and `hero_vehicle`; `assets/gate.py`
`PATCH_CAPS`) and export is blocked until human approval is recorded — both
enforced server-side.

Every image an MCP tool hands back goes through `terrain.budget`. The
full-resolution PNG is written to the map or asset directory; what the agent
reads is `budget.deliver(...)` or `budget.deliver_file(...)` — a JPEG capped at
768 px on the long edge and 320 KB, about 790 tokens. Reach for a crop
(`budget.detail_region(centre, span)`) rather than another full-map dump when
inspecting detail; a quarter-map crop costs a fifth of the tokens. A tool that
returns a raw PNG of a 2048² map is a tool that ends the session.
