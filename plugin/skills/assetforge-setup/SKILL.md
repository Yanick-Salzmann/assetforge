---
name: assetforge-setup
description: Provision a fresh machine to run the assetforge plugin - workspace directory, backend venv via uv (honouring the cu130 torch index), CC0 library binaries from the committed lockfiles, the Blender MCP add-on, and the MCP server entries for this host. Trigger on "/assetforge-setup", "set up assetforge", "provision assetforge", "install the assetforge backend", or any request to get the terrain-forge / asset-forge / blender-mcp servers running for the first time.
---

# assetforge-setup

The plugin tree (`plugin/skills/`, `plugin/.mcp.json`) carries workflow only - no backend runs
without this skill. It turns a checkout of the assetforge repo plus `uv`, `git` and Blender into
three connected MCP servers and a workspace the terrain and asset skills can write into.

**Stop-on-failure rule:** run each step's command yourself, read its actual output, and if a
command fails, report that exact command and its exact error back to the user before touching the
next step. Do not paper over a failure by skipping ahead - a later step silently building on a
half-provisioned venv or an empty library index just moves the failure somewhere harder to read.

## 0. Locate the repo

This file lives at `<repo>/plugin/skills/assetforge-setup/SKILL.md` whether it was reached through
a plugin install or a local checkout. Resolve `<repo>` as three directories up from this file's own
path - every command below runs with `<repo>` as the working directory. Do not guess a path from
`cwd`; a plugin host may invoke this skill from anywhere.

## 1. Workspace directory

Ask the user for a workspace directory, or infer one (e.g. a sibling `assetforge-workspace/` next
to `<repo>`, or the current project directory if the user is clearly setting this up for a specific
game project). This is where `out/terrain/`, `out/assets/`, downloaded `library/` binaries and any
biome overrides live - never inside `<repo>` itself, so a plugin update never wipes generated work
or re-downloads CC0 binaries. State the resolved path back in one line before creating anything.

```
mkdir -p <workspace>/out/terrain <workspace>/out/assets <workspace>/biomes
```

Everything from here on runs with `ASSETFORGE_WORKSPACE=<workspace>` in the environment - every
`uv run` call below needs it set, since `terrain.config` reads it once at import time.

## 2. Provision the venv - `uv sync`

```
uv sync --extra dev
```

Run from `<repo>`, not via `uvx --from git+...` - that resolves `torch` as an ordinary dependency
and loses `[tool.uv.sources]`'s pin to the cu130 index, silently landing CPU wheels on a machine
that has a CUDA GPU. `uv sync` against the repo's own `pyproject.toml`/`uv.lock` is the only path
that honours it.

Resolve the interpreter this produced:

- Windows: `<repo>/.venv/Scripts/python.exe`
- macOS/Linux: `<repo>/.venv/bin/python`

Call this `ASSETFORGE_PYTHON` for the rest of the skill - it is the absolute path the MCP servers
must be launched with, not a bare `python` or `uv run` that depends on `<repo>` staying the cwd.

## 3. Report the device

```
ASSETFORGE_WORKSPACE=<workspace> uv run python -c "from terrain.device import resolve; d = resolve(); print(d.spec, '-', d.label)"
```

Print this line to the user immediately - `cuda - <GPU name>` or a `cpu - CPU` fallback changes
what resolution and iteration counts are sane advice for the rest of the session (CLAUDE.md's 12 GB
/ 4096² ceiling assumes CUDA; a CPU fallback is a correctness check, not a target to generate at
production resolution).

## 4. Fetch library binaries

All of these need `ASSETFORGE_WORKSPACE=<workspace>` set - they write under
`<workspace>/library/`, reading the pinned sets and lockfiles that ship inside `<repo>/library/`.

```
uv run python -m library.polyhaven
uv run python -m library.materials

uv run python -m library.polyhaven --set library/asset_materials.toml --lock library/asset_materials.lock.json --materials-dir <workspace>/library/asset_materials
uv run python -m library.materials  --set library/asset_materials.toml --lock library/asset_materials.lock.json --materials-dir <workspace>/library/asset_materials --out <workspace>/library/asset_materials.json

uv run python -m library.kenney
uv run python -m library.kits
```

The first pair pulls the terrain splat material set and rebuilds its index; the second pair does
the same for the hard-surface asset material set; the third pulls the Kenney CC0 kit packs. Each
`pull` step reads a hash out of the committed `.lock.json` and only re-downloads what is missing or
corrupt, so re-running this skill on an already-provisioned machine is cheap and safe.

Quaternius packs (`library/kits.toml` entries with `source = "quaternius"`) are gated behind an
itch.io claim flow with no stable download URL - `library/kenney.py`'s pull reports them as failed
with a message naming the zip it expects staged under `library/_staging/`. Treat that specific
failure as a known gap to report, not a blocker: tell the user which packs need a manual download
and re-run `uv run python -m library.kenney` once staged, rather than stopping the whole skill on it.

## 5. Blender extension install

The bridge is a Blender **extension** (5.1+, `blender_manifest.toml`-based), built and installed
from the command line - no manual Preferences click-through needed.

```
git clone --depth 1 https://projects.blender.org/lab/blender_mcp.git <tmp>/blender_mcp
```

Locate the Blender executable - respect `ASSETFORGE_BLENDER` if the user has it set, otherwise
discover it (`where blender` / `which blender` / the platform's default install location) and
confirm the version is 5.1.0 or newer before continuing.

```
<blender> --command extension repo-list
```

Read the local repository id from this (normally `user_default`) rather than assuming it - a
machine with customised extension repos may not have one by that name.

```
<blender> --command extension build --source-dir <tmp>/blender_mcp/addon/blender_mcp_addon --output-filepath <tmp>/blender_mcp_addon.zip
<blender> --command extension install-file -r <repo-id> -e <tmp>/blender_mcp_addon.zip
```

`-e` enables it immediately. Report the exact command and its stderr if `build` or `install-file`
fails - a manifest validation error here is the actual root cause, not something to retry blindly.

## 6. Write the MCP server entries

`plugin/.mcp.json` declares `terrain-forge` and `asset-forge` as
`${ASSETFORGE_PYTHON}` / `${ASSETFORGE_WORKSPACE}` templates - this step is what makes those
resolve on this specific machine. The mechanism differs by host:

**Claude Code** - add or update the `env` block in the project's `.claude/settings.json` (create it
if absent; merge into it if present, never overwrite unrelated keys):

```json
{
  "env": {
    "ASSETFORGE_PYTHON": "<absolute path from step 2>",
    "ASSETFORGE_WORKSPACE": "<absolute path from step 1>"
  }
}
```

**Codex** - `config.toml` does not expand `${VAR}` in TOML string values, so write the resolved
values directly into `[mcp_servers.terrain-forge]` / `[mcp_servers.asset-forge]` (in
`~/.codex/config.toml` or the project's `.codex/config.toml`, matching whatever this host already
uses for its other servers):

```toml
[mcp_servers.terrain-forge]
command = "<absolute ASSETFORGE_PYTHON path>"
args = ["-m", "terrain.server"]
[mcp_servers.terrain-forge.env]
ASSETFORGE_WORKSPACE = "<absolute workspace path>"

[mcp_servers.asset-forge]
command = "<absolute ASSETFORGE_PYTHON path>"
args = ["-m", "assets.server"]
[mcp_servers.asset-forge.env]
ASSETFORGE_WORKSPACE = "<absolute workspace path>"
```

`blender-mcp` needs no per-machine values - leave its `uvx --from git+...` entry as-is on both
hosts.

## 7. Smoke-test all three servers

For `terrain-forge` and `asset-forge`, prove the server actually constructs and answers over the
protocol - not just that the module imports - with an in-process `fastmcp` client, the same way
`tests/test_server.py` does:

```
uv run python -c "
import asyncio
from fastmcp import Client

async def check(module, attr):
    mod = __import__(module, fromlist=[attr])
    async with Client(getattr(mod, attr)) as client:
        tools = await client.list_tools()
        print(module, 'OK', len(tools), 'tools')

asyncio.run(check('terrain.server', 'mcp'))
asyncio.run(check('assets.server', 'mcp'))
"
```

Print `PASS`/`FAIL` for each of the two, with the exception text on failure.

For `blender-mcp`, this in-process trick does not apply - it is a separate process this host
launches over `uvx`. Verify it by calling one of its own tools that needs a live Blender connection
(e.g. an object or scene summary tool) through this session's MCP connection and reading the
result: a real answer is `PASS`; a connection-refused-style error is `FAIL`, and the fix is opening
Blender with the extension from step 5 enabled, not re-running this skill. If the host has not yet
reloaded its MCP config from step 6, tell the user to restart/reload the MCP connection before this
check can pass for `terrain-forge` / `asset-forge` too.

## 8. Close the loop

Once all three servers pass, generate a small terrain to prove the whole chain end to end:

```
new_terrain(name="setup-check", seed=0, resolution=512, world_size_m=2048, height_range_m=400, sea_level_m=0, shape="continental")
```

on the `terrain-forge` server. A hillshade preview coming back, written under
`<workspace>/out/terrain/setup-check/`, is the acceptance bar for this skill. Report the resolved
workspace path, the device from step 3, the pass/fail line for each of the three servers, and any
skipped library pulls (Quaternius) so the user knows exactly what state they were left in.
