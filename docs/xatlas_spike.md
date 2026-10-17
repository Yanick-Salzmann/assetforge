# xatlas spike (af-zch.2)

Can xatlas run in Blender's Python, and does it beat our axis/strip unwrap + shelf packer
(`game_ready.uv_unwrap_and_pack`) on padding, density and island count?

## Setup

- `xatlas` 0.0.11 (MIT; xatlas itself MIT) ships a `cp313-win_amd64` wheel. Blender 5.2.1 bundles
  Python 3.13.13 and numpy, so
  `"<blender>/5.2/python/bin/python.exe" -m pip install --only-binary=:all: --target <dir> xatlas`
  plus `sys.path.insert(0, <dir>)` imports it with no build step. Program Files stays untouched.
- `assets/blender_scripts/xatlas_spike.py` takes every static and rigid `<name>_*` part of a saved
  hero-building .blend (skinned cloth left out) in world space, triangulated, and lays it out on one
  2048 atlas three ways: `ours`, `xatlas` (every face) and `xatlas_visible` (the faces our
  `hidden_faces` pass keeps). xatlas runs with `padding = gutter_px(2048) / 2` (its padding is per
  chart side, so 8 gives our 16 px between charts), `bilinear`, `blockAlign`, and a 10-step bisection
  on `texels_per_unit` for the densest layout that fits one 2048 atlas - with `texels_per_unit = 0`
  it picks a density that overflows to ~3100 px.
- `uv run python -m assets.xatlas_spike_score <dir>` scores each layout with validate_asset's own
  measures: `_uv_islands`, the p95/p5 texel-density ratio, island pairs under 8 px, coverage.

## Results (2048 atlas, 16 px gutter)

| Asset | Tris | Method | Islands | px/m | p95/p5 | Coverage | Layout s |
|---|---|---|---|---|---|---|---|
| medieval_tavern | 4290 | ours | 622 | 38.4 | 1.11 | 0.374 | 0.44 |
| | | xatlas | 840 | 38.9 | 1.22 | 0.446 | 0.37 |
| | | xatlas_visible | 625 | 43.2 | 1.15 | 0.518 | 0.32 |
| medieval_church | 5262 | ours | 826 | 30.8 | 1.14 | 0.413 | 0.66 |
| | | xatlas | 837 | 32.1 | 1.19 | 0.506 | 0.30 |
| | | xatlas_visible | 815 | 32.4 | 1.18 | 0.507 | 0.30 |
| egypt_smithy | 18378 | ours | 2217 | 35.5 | 1.07 | 0.226 | 5.01 |
| | | xatlas | 2307 | 30.1 | 1.39 | 0.265 | 0.52 |
| | | xatlas_visible | 2007 | 35.9 | 1.32 | 0.261 | 0.54 |
| egypt_grain_mill | 26194 | ours | 3502 | 27.9 | 1.06 | 0.180 | 10.48 |
| | | xatlas | 3383 | 21.6 | 1.41 | 0.179 | 0.64 |
| | | xatlas_visible | 2931 | 26.8 | 1.38 | 0.188 | 0.68 |
| egypt_pottery_workshop | 13206 | ours | 1260 | 41.4 | 1.06 | 0.299 | 2.68 |
| | | xatlas | 1589 | 35.4 | 1.29 | 0.363 | 0.43 |
| | | xatlas_visible | 1379 | 42.9 | 1.22 | 0.349 | 0.43 |
| egypt_pyramid_site | 10670 | ours | 1594 | 22.6 | 1.05 | 0.235 | 2.68 |
| | | xatlas | 1799 | 21.1 | 1.14 | 0.389 | 0.37 |
| | | xatlas_visible | 1568 | 26.4 | 1.36 | 0.348 | 0.41 |

Every layout: 0 island pairs under 8 px, 0 UVs outside the unit square. xatlas layout seconds are
one `generate`; the density bisection costs ~10 of them (4-7 s per asset).

## Findings

- **Runs in Blender's Python: yes**, from a `--target` wheel directory, no compiler.
- **xatlas alone does not beat ours.** Without our hidden-face culling it spends texels on buried
  faces and lands 7-23% below our density on the Egyptian buildings.
- **xatlas on our visible faces is roughly a tie**: density -4% to +17% (median +5%), islands -16%
  to +9%. The gain is small because ours already packs at one density with strips for revolved
  parts; xatlas's edge is chart rotation and tighter packing of odd shapes.
- **Texel density is less uniform** (p95/p5 1.15-1.38 against our 1.05-1.14): xatlas parametrises
  charts with LSCM, which stretches; ours is a rigid projection. Both stay under the 2.0 limit.
- **Faster** on big assets (0.5 s vs 5-10 s) but the fit-to-2048 bisection eats that back.
- **Integration cost**: AtlasLayout needs per-island `PixelPacking` footprints for
  `atlas_bake.dilate_gutters`, which would have to be rebuilt from xatlas charts; the 4 px block
  grid is approximate (`blockAlign` aligns chart origins, not our footprint rule); and a native
  wheel has to be provisioned into Blender's Python on every machine (af-37k distribution).

## Decision

Keep `uv_unwrap_and_pack` for hard-surface assets. The measured gain does not pay for a native
dependency, a second footprint model and worse density uniformity. xatlas stays the obvious
candidate where axis projection has no answer - welded organic and neural meshes with no planar
regions (af-w2s.6, neural mesh cleanup), which this spike did not measure.
