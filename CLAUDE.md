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

12 GB of VRAM is a real ceiling. A 16-channel float32 stack costs 256 MiB at
2048² and 1 GiB at 4096²; 8192² does not leave room for the pipe model's flux
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
  `patchiness_coarse`. All are normalised to `[0, 1]` except `height`, which is
  normalised too — the metre mapping is applied only at export.
- Splat masks pack 4 layers per RGBA texture: `splat_0.png` holds layers 0-3,
  `splat_1.png` layers 4-7. Layer weights across all textures sum to 1 per pixel.
- Scatter masks are single-channel 8-bit: `scatter_rock.png`, `scatter_tree.png`,
  `scatter_grass.png`, `scatter_debris.png`.
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
- No baked lighting and no AO darkening in the albedo. The engine lights it.
- Power-of-two textures, 2048 atlas, one material per asset.
- Canonical export is `.glb`, with `asset.json` and `recipe.py` beside it, in
  `out/assets/<name>/`.
- Every live MCP session appends the `bpy` it executed to `recipe.py` so the
  session leaves behind something re-runnable under
  `blender --background --python`.

### Budgets

| Kind | Tris | Notes |
|---|---|---|
| Hero building | 4000-15000 | LOD0/1/2 always shipped |
| Prop | 200-1500 | 6-12 per set, one atlas per set |
| Vehicle / machinery | 5000-20000 | same path as buildings |

### The visual loop

An agent cannot judge what it cannot see. Never declare a generated result good
without rendering it and reading the render back: hillshade and false-colour
splat for terrain, the 6-view contact sheet for assets. Asset iteration is
capped at 4 rounds and export is blocked until human approval is recorded — both
enforced server-side.

Every image an MCP tool hands back goes through `terrain.budget`. The
full-resolution PNG is written to the map or asset directory; what the agent
reads is `budget.deliver(...)` or `budget.deliver_file(...)` — a JPEG capped at
768 px on the long edge and 320 KB, about 790 tokens. Reach for a crop
(`budget.detail_region(centre, span)`) rather than another full-map dump when
inspecting detail; a quarter-map crop costs a fifth of the tokens. A tool that
returns a raw PNG of a 2048² map is a tool that ends the session.
