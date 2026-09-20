# assetforge

Agent-driven terrain and hard-surface asset pipeline for a city builder. Two headless MCP
servers (`terrain-forge`, `asset-forge`) plus the community `blender-mcp` bridge drive a
Blender 5.1+ instance and a numpy/torch terrain pipeline from an agent session. See
[`PLAN.md`](PLAN.md) for the full design and [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md)
for the working conventions.

## Install

The plugin ships skills and MCP server wiring only — no backend runs until you also run the
`assetforge-setup` skill (below) to provision a venv, download CC0 library binaries, and install
the Blender bridge add-on. Installing is opt-in and does not auto-install: it pulls a multi-GB
venv (`torch` with CUDA wheels) the first time you run setup.

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/), `git`, Blender 5.1+, ~4 GB free disk,
CUDA optional (falls back to CPU).

**Claude Code**

```
/plugin marketplace add Yanick-Salzmann/assetforge
/plugin install assetforge@assetforge
```

**Codex**

```
codex plugin marketplace add Yanick-Salzmann/assetforge
codex plugin add assetforge@assetforge
```

Either install lands you at the `assetforge-setup` skill — run it (`/assetforge-setup` or ask
the agent to set up assetforge) to provision the backend before using any other skill.

### `ASSETFORGE_WORKSPACE`

The plugin's own tree (`plugin/skills/`, `.mcp.json`) is read-only and lives wherever your
plugin manager installs it — a cache directory that a `/plugin update` can wipe. All generated
output, downloaded CC0 library binaries, and biome overrides instead live under a separate
**workspace directory** you choose during setup, pointed at by the `ASSETFORGE_WORKSPACE`
environment variable that the setup skill writes into your MCP server config. `terrain.config`
resolves it once at import time: an explicit override, else `ASSETFORGE_WORKSPACE`, else the
current working directory. Nothing a plugin update touches ever lives inside it.

## Versioning

The plugin (`.claude-plugin/plugin.json` / `.codex-plugin/plugin.json`) is versioned
independently of the `assetforge` Python package (`pyproject.toml`). A plugin release states
which backend (Python package) versions it expects in its own changelog entry — installing a
newer plugin against an already-provisioned older backend is not guaranteed to work; re-run
`assetforge-setup` after a plugin update to reconcile the venv. See
[`CHANGELOG.md`](CHANGELOG.md) for which backend version each plugin release expects.
