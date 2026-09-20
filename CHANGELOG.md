# Changelog

Plugin releases (`.claude-plugin/plugin.json` / `.codex-plugin/plugin.json` `version`) are
independent of the `assetforge` Python package release (`pyproject.toml` `version`). Each entry
below states the backend package version range a given plugin release was built against and
expects `assetforge-setup` to provision.

## plugin 0.1.0 — 2026-09-20

- Initial dual-manifest plugin tree (Claude Code + Codex): `terrain-new`, `terrain-tune`,
  `asset-building`, `asset-prop`, `asset-vehicle`, `assetforge-setup` skills; `terrain-forge`,
  `asset-forge`, `blender-mcp` MCP servers.
- Expects backend `assetforge` package `0.1.0`.
