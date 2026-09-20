# AssetForge — an agent-driven terrain & asset pipeline for a city builder

## Context

You build your own games — mainly city builders — with your own engine, and you can't model. You
want Claude to produce the content for you. Priority order, as you stated it:

1. **Terrain generation** — heightmap output, your engine builds the mesh.
2. **Terrain texturing** — layered tiling materials driven by splat masks, and it has to look
   *interesting*. A flat region must not read as a plain sheet of grass. Realism without needing
   the terrain to be steep.
3. **Hard surfaces** — one-off hero buildings that get placed and sit there, plus prop sets to
   dress the area around them. Also non-building hard surface (vehicles, machinery, props).
4. Organic / characters — explicitly last, optional.

The design principle throughout: **an agent cannot judge what it cannot see.** Everything is built
around a loop of generate → render a preview → *look at it* → critique → adjust. For terrain that
loop lives in 2D image space, which is exactly where an LLM is strongest.

The second principle, and the answer to your "no plain grass fields" requirement: **variety comes
from simulation, not from slope**. Slope and altitude alone give you the boring result you
described. A hydraulic erosion pass produces flow accumulation, sediment deposition and wear
fields as by-products, and those are what put dry creek beds, silt fans, gravel patches and mud
into *flat* ground. That data is free once erosion runs — it just has to be kept instead of
thrown away.

## Architecture

Two subsystems, deliberately separate. **Blender is not the hub for terrain.**

```
TERRAIN  (numpy/torch, headless, no Blender in the loop)
  brief → heightfield synth → erosion sim (torch, resolved device) → channel stack
        → splat rules → masks → preview renders → agent looks → iterate
        → export: height.png(16-bit) + splat_*.png + scatter_*.png + terrain.json

ASSETS  (Blender 5.2 LTS GUI, driven live over MCP as you asked)
  brief → bpy modelling → contact sheet render → agent looks → iterate
        → game-ready pass → export .glb + asset.json + recipe.py
```

Claude Code connects to three MCP servers: `terrain-forge` (ours, Python), `asset-forge` (ours,
the render/validate/export tools the stock bridge lacks) and `blender-mcp` (the
Blender Foundation's official Blender Lab server, which requires Blender 5.1+).
All three are declared in a checked-in `.mcp.json` so a fresh clone connects them without manual
setup — a server that is built but never registered is a server that never runs.

### Repo layout

```
~/gamedev/assetforge/
  CLAUDE.md                 conventions the agent obeys every session
  pyproject.toml            uv; torch, numpy, pillow, fastmcp, requests
  terrain/
    server.py               FastMCP: the terrain toolset
    device.py               the one place that names a backend: cuda → mps → cpu
    synth.py                base heightfield: domain-warped fBm, ridged, worley, tectonic uplift
    erosion.py              pipe-model hydraulic + thermal erosion, torch on the resolved device
    channels.py             derives the channel stack from height + erosion outputs
    splat.py                rule engine: channel stack → weighted layers → packed RGBA masks
    scatter.py              density masks for rocks / grass / trees / debris
    preview.py              hillshade, false-colour splat, 3D beauty render via Blender headless
    export.py               writes the deliverable set + terrain.json
  assets/
    server.py               FastMCP: render_contact_sheet, validate, game_ready, export, recipe
    hardsurface.py          bpy helper lib the agent composes
  library/
    materials/              CC0 tiling PBR sets (Poly Haven): grass, rock, dirt, sand, gravel, ...
    kits/                   CC0 prop kits (Kenney, Quaternius), indexed
  out/terrain/<name>/       height.png splat_0.png ... scatter_*.png preview_*.png terrain.json
  out/assets/<name>/        <name>.glb asset.json recipe.py contact_sheet.png textures/
  .claude/skills/           terrain-new, terrain-tune, asset-building, asset-prop
```

---

## Part 1 — Terrain generation

### The channel stack

Everything downstream reads from one `float32[H,W,C]` stack. This is the whole trick.

| Channel | Source | What it buys you |
|---|---|---|
| `height` | fBm + ridged noise, domain-warped, plus large-scale tectonic uplift mask | the base |
| `slope`, `curvature` | gradients of eroded height | cliffs, convex ridges vs concave hollows |
| `flow` | flow accumulation from the erosion sim | **river beds, gullies, drainage lines across flat ground** |
| `deposition` | sediment dropped by the erosion sim | **alluvial fans, silt flats, valley floors** |
| `wear` | material removed by erosion | exposed bedrock, scoured rock faces |
| `wetness` | topographic wetness index (flow ÷ slope) | marsh, mud, lush vs parched vegetation |
| `water`, `water_depth` | standing water from the erosion sim against a derived sea/lake level | **where water actually is** — river surfaces, lakes, the water plane the engine binds |
| `bedrock` | height minus a smoothed soil-depth field | outcrops poking through flat meadow |
| `strata` | banded noise following an angled plane | geological banding visible in cuts and flats |
| `temperature` | altitude lapse + latitude gradient + noise | snow line, alpine transitions |
| `moisture` | domain-warped low-freq noise, biased by `wetness` | biome variation independent of terrain shape |
| `patchiness` | multi-octave worley + domain-warped fBm, several scales | **breaks up uniform regions at 3 different scales** |

The last five exist specifically to solve your flat-grass problem: they vary across the map with
no dependence on slope at all.

### Erosion

Pipe-model hydraulic erosion (grid-based, fully vectorised) plus a thermal/talus pass, implemented
in PyTorch against whatever `terrain/device.py` resolves — CUDA on the build box, CPU anywhere
without a GPU, MPS if the macOS target is ever revived. No kernel names a backend. Grid-based
rather than particle-based
because it vectorises cleanly and because the pipe model gives water flux and sediment as
first-class grid fields — which is precisely the data the splat rules need. Target: 2048² for a few
thousand iterations in well under a minute.

We write our own rather than using an existing addon (Hydra, Terrain Nodes) because the addons
give you an eroded heightmap and discard the flow/sediment fields. Those fields are the product
here, not a side effect.

### Splat rules

`splat.py` is a small declarative rule engine — each material layer is a weight expression over
the channel stack, evaluated, softmax-blended, then packed into RGBA textures (4 layers per
texture, 8 layers = 2 textures). Rules live in a readable TOML per biome, so the agent tunes text,
not code:

```toml
[layer.riverbed_gravel]
weight = "smoothstep(flow, 0.55, 0.8) * (1 - wetness*0.3)"
[layer.silt_flat]
weight = "deposition^1.5 * step(slope < 0.12) * patchiness_mid"
[layer.dry_grass]
weight = "(1 - moisture) * step(slope < 0.35) * patchiness_coarse"
[layer.lush_grass]
weight = "moisture * wetness^0.5 * step(slope < 0.35)"
[layer.exposed_rock]
weight = "max(step(slope > 0.6), bedrock * wear)"
```

A flat patch of ground therefore gets grass *or* dry grass *or* silt *or* gravel depending on
moisture, deposition and patchiness — never one uniform layer. Blend masks are anti-tiled with a
low-frequency macro-variation multiply so repetition doesn't read at distance.

### Materials

Source CC0 tiling PBR sets from Poly Haven into `library/materials/` (grass, dry grass, forest
floor, mud, silt, gravel, river rock, cliff rock, sand, snow). Each layer in a splat set names one
of these; `terrain.json` carries the layer→material mapping and per-layer UV tiling scale so your
engine can bind them.

### How the agent sees terrain

`preview.py` produces three images per iteration, all read back by the agent:

1. **Hillshade + contour** of the heightmap — reveals drainage structure, repetition, unnatural
   noise, flat dead zones.
2. **False-colour splat composite** — one distinct colour per layer, so layer distribution and
   coverage percentages are directly legible. Plus a printed coverage table.
3. **Beauty render** — Blender headless displaces a grid, binds the real tiling materials through
   the splat masks, renders one 3/4 view and one ground-level view. This is the gate that actually
   answers "does the flat part look interesting".

### Deliverables

```
height.png      16-bit grayscale, non-color
splat_0.png     RGBA, layers 0-3      splat_1.png  RGBA, layers 4-7
scatter_rock.png  scatter_tree.png  scatter_grass.png  scatter_debris.png
water.png       8-bit water mask; terrain.json carries the water level in metres
normal.png      optional, derived from height at world scale, if your engine won't derive it
terrain.json    world size (m), height range (m), m/px, water level, layer→material+tiling map,
                splat channel assignment, scatter list, seed, rule file — versioned, with a
                validator so an export can be checked rather than trusted
preview_*.png   the three previews above, kept as the visual changelog
```

---

## Part 2 — Buildings, props, hard surfaces

Live Blender over MCP, as you chose. Blender 5.2 LTS stays open, you watch, you intervene.

**These are full 3D assets, not isometric fakery.** The camera is free — orbit, tilt, zoom — so
every face has to hold up: no billboards, no imposters, no missing back walls, no detail that only
reads from one angle, no baked-in lighting or ground shadow. Closed, manifold, correctly-normalled
meshes with clean UVs on all sides, lit entirely by your engine. The validation gates and the
6-view contact sheet exist to catch exactly the shortcuts an agent would otherwise take.

That freedom raises the budgets — a building the player can push the camera up against needs real
geometry, not a texture standing in for it.

- **Hero buildings** — freeform, one per asset, agent-written `bpy` over `hardsurface.py`
  (bevelled box, panel-cut boolean, array/mirror, solidify, greeble). 4000–15000 tris, one
  material, 2048 atlas. Real modelled window recesses, door frames, roof edges and cornices —
  the silhouette must survive being viewed from below and from behind.
- **LODs** — because the camera also pulls back, each building exports LOD0/1/2 via decimate with
  normals baked from LOD0. Your engine picks; the pipeline always ships all three.
- **Prop sets** — each building ships with a companion set of 6–12 dressing props (fences, crates,
  benches, lamps, carts, vegetation) at 200–1500 tris, batched into one atlas per set. Same rule:
  fully closed 3D, viewable from any angle. Check `library/kits/` for a CC0 fit before generating.
- **Non-building hard surface** — vehicles and machinery use the same path at 5000–20000 tris.
- **Texturing** — geometry alone is not an asset. A small library of hard-surface base materials
  (CC0 pulls plus procedural node groups for concrete, plaster, brick, metal, glass, roofing) is
  assigned per face during modelling, then baked down to one albedo + normal + ORM atlas at 2048
  after the unwrap. No baked lighting and no AO darkening in the albedo — the engine lights it.

The asset loop mirrors terrain's: build → `render_contact_sheet` (6 orthographic views + a 3/4
beauty + a low-angle worm's-eye shot, one PNG) → agent reads it and critiques in writing against
the brief → `validate_asset` (tri count, **watertight/manifold check**, ngons, loose verts,
**inward-facing normals**, **missing back faces**, UV overlap, material count, bbox in metres,
origin offset, unapplied transforms) → patch → repeat, capped at 4 iterations → your approval →
texture bake → `game_ready` → LODs → export. The iteration cap and the approval step are enforced
by the server, not by the skill's prose: past the cap further patches are refused, and export fails
until approval is recorded.

**Conventions** (in `CLAUDE.md`, enforced by `validate_asset`): Blender Z-up, glTF exporter
converts to Y-up; 1 unit = 1 m; origin at base centre; +Y forward; transforms applied; normals
outward; power-of-two textures; canonical export `.glb`.

Every MCP session dumps the bpy it executed into `recipe.py`, so a live, unreproducible chat
session still leaves behind something re-runnable under `blender --background --python`. That is
the one addition to your "live MCP" choice — it costs nothing and makes assets rebuildable.

---

## Build order

**Phase 0 — Project home + environment (½ day).**

**On approval, this session does exactly two things and then stops:** create
`C:\Users\yanick\programming\assetforge\` and write this plan into it as `PLAN.md`. No code, no
scaffolding, no dependencies, no `git init`. That folder is where the *next* session starts, and
everything below — including the rest of this phase — is that session's work, not this one's.

The build target is this Windows 11 box: RTX 4070, 12 GB VRAM, CUDA via the `cu130` torch
wheels, Blender 5.2 LTS installed locally. CPU is the portable fallback — correctness only, not a
performance measurement. Apple Silicon and MPS are a deferred secondary target; nothing in the
pipeline is written against them, and reviving that path should cost one branch in
`terrain/device.py`, not a rewrite.

Next session, inside that folder: Blender 5.2 LTS, `uv`, torch with CUDA verified, the official
`blender_mcp` add-on installed, all three MCP servers declared in `.mcp.json`, the Blender binary path resolved
once for both the live GUI instance and headless renders, `pytest` scaffolded, and a Poly Haven
material pull script that pins slugs and hashes in a lockfile — the library is gitignored, so that
lockfile is the only thing keeping terrain reproducible.

**Phase 1 — Terrain core (3–4 days).** `synth.py`, `erosion.py`, `channels.py`, hillshade preview,
`terrain-forge` MCP server. Gate: erosion runs 2048² on the resolved device in under a minute and
the hillshade shows a coherent, branching drainage network.

**Phase 2 — Splat & texturing (3–4 days).** `splat.py` rule engine, TOML biome files, material
library, false-colour preview, Blender beauty render. **This is the phase that decides whether the
project works.** Gate below.

**Phase 3 — Scatter & export contract (1–2 days).** `scatter.py`, `export.py`, `terrain.json`,
16-bit PNG round-trip verified, `/terrain-new` and `/terrain-tune` skills.

**Phase 4 — Buildings and props (3–4 days).** `assets/server.py` (contact sheet, validate, game
ready, export, recipe), `hardsurface.py`, `/asset-building` and `/asset-prop` skills.

**Phase 5 — Reproducibility & engine integration (1–2 days).** Every terrain rebuildable from
seed + rule file; every asset rebuildable from `recipe.py`; a regression script that regenerates
everything and diffs against metadata. Load one terrain and three buildings in your engine.

**Phase 6 — Optional, later.** Organic/characters: local Draw Things concept image → Tripo
image-to-3D API (~$0.01/credit, 2000 free) → Blender retopo + normal bake. Deferred as you asked.

## Verification

- **Phase 1 gate:** hillshade of a generated 2048² map shows branching valleys and ridgelines, not
  noise; the agent correctly identifies a deliberately over-tiled region you introduce.
- **Phase 2 gate — the important one:** take a terrain whose central 30% is near-flat, render the
  ground-level beauty shot there, and confirm it shows at least three distinguishable surface
  materials with organic boundaries. Coverage table must show no single layer above ~45% of the
  map. This is your "no plain grass field" requirement, made testable.
- **Phase 3 gate:** heightmap survives a 16-bit write/read round-trip with no banding; scatter
  masks visibly avoid open water, channel beds and steep rock; `terrain.json` validates against its
  own schema and every path it declares exists on disk at the declared dimensions.
- **Phase 4 gate:** three buildings plus one prop set generated in a single session, all passing
  `validate_asset`, zero manual Blender edits — and each one orbited fully in the engine with the
  camera at ground level, confirming no missing back faces, flipped normals or one-angle-only
  detail.
- **Phase 5 gate:** regeneration script reproduces every artifact byte-identically from seeds.
- **Ongoing:** everything loads in your engine at correct scale against a 1.8 m reference.

## Risks, stated plainly

- **Phase 2 is the real risk.** Erosion producing *plausible* flow and sediment fields is what
  makes the splat rules interesting. If the erosion sim is weak, flat ground stays boring no
  matter how good the rules are. Budget iteration time there, and validate erosion output visually
  before writing any splat rules.
- **12 GB of VRAM is a hard ceiling.** The old plan assumed 24 GB of unified memory, where the
  whole channel stack plus every transient erosion field could sit resident at any grid size. It
  cannot here. Measured on the 4070: a 16-channel float32 stack costs 256 MiB at 2048² and 1 GiB at
  4096², both comfortable; at 8192² the stack alone is 4 GiB and the pipe model's flux fields do
  not fit beside it. Treat 4096² as the working ceiling and tile or drop flux to float16 above it.
  Benchmark the erosion kernel early — the algorithm is the same on CPU either way, so falling back
  to numpy/numba stays an option.
- **`execute_blender_code` is arbitrary code execution inside Blender.** Run a project-only
  instance, don't open scenes you care about while MCP is attached, keep everything in git.
- **Spatial reasoning is the weak link on the asset side.** Expect 2–4 loop iterations per
  building and a human approval gate, not one-shot generation.
- **Live MCP sessions aren't reproducible** — `recipe.py` capture is the mitigation, Phase 5 proves
  it.

## Sources

- [Hydra — Blender OpenGL hydraulic erosion](https://github.com/ozikazina/Hydra) · [Terrain Mixer](https://extensions.blender.org/add-ons/terrainmixer/) · [realistic_terrain (numpy + compute shader erosion)](https://github.com/TLabAltoh/realistic_terrain)
- [Blender Lab MCP server (official)](https://www.blender.org/lab/mcp-server/) · [source](https://projects.blender.org/lab/blender_mcp) · [Blender devtalk on MCP security](https://devtalk.blender.org/t/blender-mcp-server-after-claude-mcp-security-for-blender-scripting-3d-agent-notes/45131)
- [blender-building-generator (game-oriented)](https://github.com/outerreaches/blender-building-generator) · [ProceduralBuildingGenerator](https://github.com/lsimic/ProceduralBuildingGenerator)
- [Tripo developer pricing](https://developers.tripo3d.ai/en/pricing) (Phase 6 only)
