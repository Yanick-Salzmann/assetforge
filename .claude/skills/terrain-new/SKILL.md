---
name: terrain-new
description: Walk a new terrain end to end through the terrain-forge MCP server - synth, erosion, channels, splat, scatter, export - reading and critiquing every preview before moving on. Trigger on "/terrain-new", "generate a new terrain", "make a map called X", or any request to build a terrain from scratch.
---

# terrain-new

Produces one full deliverable set under `out/terrain/<name>/`: `height.png`, `splat_*.png`,
`scatter_*.png`, `normal.png`, `water_mask.png`, `terrain.json`. Every generation tool below
returns a preview image alongside its summary - **read the image and judge it against the
acceptance check before calling the next tool.** Chaining calls without looking is how a
noise field or a honeycomb of belts survives to the final export.

All tools are on the `terrain-forge` MCP server. Call `terrain_status` first if you have not
already this session - it lists shape presets, erosion presets, channel names and the compute
device, and its output can go stale if you assume it instead of calling it.

## 1. Brief

Before touching a tool, pin down the four things every later step needs:

- **name** - lowercase alphanumerics/dashes/underscores only, becomes a directory.
- **shape** - one of the presets from `terrain_status` (`continental`, `island`, `archipelago`,
  `coastal_range`, `highland`, `basin`), or ask the user what kind of landmass they want.
- **scale** - `resolution`, `world_size_m`, `height_range_m`, `sea_level_m`. Defaults
  (1024 / 4096 m / 600 m / 0 m) are fine for a first pass; do not go past 4096 resolution
  without checking VRAM headroom (12 GB ceiling, see CLAUDE.md).
- **biome** - a file from `biomes/` (currently `temperate.toml`) or a path to a custom one.

State the brief back in one line before generating anything, so a bad assumption is cheap to
catch.

## 2. Synth - `new_terrain`

```
new_terrain(name, seed, resolution, world_size_m, height_range_m, sea_level_m, shape)
```

Returns the hillshade of the raw, uneroded landform. **Acceptance check:** the shape reads as
the requested landform (a continental mass, a chain of islands, a basin) with plausible
large-scale relief - not a flat plane, not a single blob, not an obviously cellular honeycomb
of ridges. If it looks wrong, this is the cheapest point to change `seed` or `shape` and retry;
every later stage is more expensive to redo.

## 3. Erosion - `run_erosion`

```
run_erosion(name, preset, params=None, water=None)
```

`preset` is one of `light` / `default` / `heavy` / `canyon` from `terrain_status`. Returns the
carved hillshade. **Acceptance check (Phase 1 gate):** the hillshade shows a coherent, branching
drainage network - valleys with tributaries, not radial scratches or uniform noise. If the
network looks sparse or absent, a heavier preset or a longer `iterations` override is the fix,
not re-rolling the seed.

## 4. Channels - `build_channels`

```
build_channels(name)
```

Fills the shared channel stack (`slope`, `curvature`, `flow`, `wetness`, `moisture`,
`temperature`, `patchiness_*`, etc.) and returns a labelled false-colour contact sheet.
**Acceptance check:** rivers and lakes in the wetness/flow tiles line up with the valleys seen
in the hillshade; no channel is a flat, uniform colour (a flat `wetness` or `flow` tile usually
means the water pass fell back to an all-land or all-water state - check `sea_level_m` against
the terrain's actual elevation range first). Use `inspect_channel(name, channel, rescale=True)`
to look at any one channel full-resolution before moving on if the contact sheet thumbnail is
ambiguous.

## 5. Splat - `apply_rules`

```
apply_rules(name, biome_file, sharpness=None)
```

Evaluates the biome's material rules over the channel stack, writes the splat textures, and
returns the false-colour composite plus a coverage table. **Acceptance check (Phase 2 gate -
the important one):** pick a near-flat region of the map (check the hillshade/slope channel) and
look at it - it must show at least three distinguishable materials with organic, not tiled,
boundaries. The coverage table must show no single layer above roughly 45% of the map. A single
layer dominating means the biome file's weight expressions need rebalancing, not a re-render.
Use `preview_splat(name, centre, span)` to crop into a suspect region instead of re-running
`apply_rules`.

Optionally call `preview_beauty(name, view, centre)` for a headless-Blender ground-level or
three-quarter render with real materials - the closest thing to what this terrain will look
like in the engine, and worth doing before signing off on the splat.

## 6. Scatter + export

There is no MCP tool for this step yet (scatter and export are plain functions, not exposed on
`terrain-forge`) - run it as a short script against the resident or replayed session:

```python
from terrain import export, session

current = session.open_session(name)
result = export.write(
    current.cfg,
    current.channels(),
    current.water,
    current.splat_result(),
    rule_path=current.biome_file,
)
print(result.manifest_path, result.files)
```

`export.write` builds the scatter masks from the channel stack itself (no separate scatter
call needed), writes every deliverable, builds `terrain.json`, and calls
`manifest.verify()` internally - a raised exception here means the terrain.json schema or an
on-disk file is wrong and export is not done. **Acceptance check (Phase 3 gate):**

- `height.png` opens as 16-bit grayscale with no visible banding.
- `scatter_rock.png` / `scatter_debris.png` are dark over water and the steepest slopes;
  `scatter_tree.png` / `scatter_grass.png` avoid open water and riverbeds. Spot-check with
  `inspect_channel` on `slope` or `water` next to the scatter PNGs if unsure.
- `export.write` returned without raising - that already proves `terrain.json` validated
  against its schema and every path it declares exists at the declared resolution.

## Done

Report the final `out/terrain/<name>/` file list and the one-line brief from step 1 so the
user can confirm the result matches what they asked for. Do not delete or overwrite previous
`preview_*.png` files - they are the visual changelog for this terrain, not scratch output.
