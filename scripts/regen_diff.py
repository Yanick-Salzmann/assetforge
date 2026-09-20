"""Regenerate every terrain and asset from its recorded seed/recipe and report drift.

Terrain: replays synth -> erode -> build_channels -> apply_biome from the seed and
params recorded in session.json, exports to a scratch dir, and diffs the result
byte-for-byte against the live out/terrain/<name>/ deliverables and terrain.json.
Determinism is the contract under test - see CLAUDE.md's Determinism section.

Asset: replays out/assets/<name>/recipe.py through a background Blender and reports
whether it completes cleanly. There is no exported asset.json/.glb contract yet
(af-5t3), so this is a replay smoke test, not a byte diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from assets import blender as blender_discovery
from assets import recipe as recipe_mod
from terrain import config, export, manifest
from terrain import session as session_mod

BLENDER_TIMEOUT_S = 300


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diff_manifest(stored: dict, fresh: dict) -> dict[str, dict]:
    drift = {}
    for key in sorted(set(stored) | set(fresh)):
        if stored.get(key) != fresh.get(key):
            drift[key] = {"stored": stored.get(key), "regenerated": fresh.get(key)}
    return drift


@dataclass
class TerrainDrift:
    name: str
    manifest_drift: dict = field(default_factory=dict)
    file_drift: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.manifest_drift and not self.file_drift

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "manifest_drift": self.manifest_drift,
            "file_drift": self.file_drift,
        }


def regenerate_terrain(name: str, scratch_dir: Path, rule_path: str | None) -> export.ExportResult:
    stored = session_mod.TerrainSession.load(name)
    fresh = session_mod.TerrainSession(
        cfg=stored.cfg,
        shape=stored.shape,
        erosion=stored.erosion,
        erosion_overrides=stored.erosion_overrides,
        water_overrides=stored.water_overrides,
    )
    fresh.synthesise().erode().build_channels()
    if stored.biome_file is None:
        raise session_mod.MapConfigError(f"terrain {name!r} has no biome recorded; cannot export")
    fresh.apply_biome(stored.biome_file, stored.sharpness_override)
    return export.write(
        fresh.cfg, fresh.channels(), fresh.water, fresh.splat_result(),
        out_dir=scratch_dir, rule_path=rule_path,
    )


def diff_terrain(name: str) -> TerrainDrift:
    live_dir = config.TERRAIN_OUT_DIR / name
    manifest_path = live_dir / manifest.MANIFEST_NAME
    if not manifest_path.is_file():
        return TerrainDrift(name=name, error=f"no {manifest.MANIFEST_NAME} under {live_dir}")
    stored_manifest = manifest.load(manifest_path)

    with TemporaryDirectory(prefix=f"regen-{name}-") as tmp:
        scratch = Path(tmp)
        result = regenerate_terrain(name, scratch, stored_manifest.get("rule_path"))
        manifest_drift = _diff_manifest(stored_manifest, result.manifest)

        file_drift: dict[str, str] = {}
        for path in result.files:
            live_path = live_dir / path.name
            if not live_path.is_file():
                file_drift[path.name] = "missing in stored output"
            elif _hash_file(path) != _hash_file(live_path):
                file_drift[path.name] = "byte content differs"

    return TerrainDrift(name=name, manifest_drift=manifest_drift, file_drift=file_drift)


def list_terrains() -> list[str]:
    return [name for name, _ in session_mod.stored()]


def list_assets() -> list[str]:
    if not config.ASSET_OUT_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in config.ASSET_OUT_DIR.iterdir()
        if path.is_dir() and (path / recipe_mod.RECIPE_NAME).is_file()
    )


def regenerate_asset(name: str) -> dict:
    asset_dir = config.ASSET_OUT_DIR / name
    recipe_path = asset_dir / recipe_mod.RECIPE_NAME
    if not recipe_path.is_file():
        return {"name": name, "ok": False, "error": f"no {recipe_mod.RECIPE_NAME} under {asset_dir}"}

    blender_exe = blender_discovery.find_blender()
    if blender_exe is None:
        return {"name": name, "ok": None, "skipped": "Blender executable not found"}

    result = subprocess.run(
        [blender_exe, "--background", "--python", str(recipe_path)],
        cwd=asset_dir,
        capture_output=True,
        text=True,
        timeout=BLENDER_TIMEOUT_S,
    )
    return {
        "name": name,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-2000:],
    }


def run(terrain_names: list[str] | None, asset_names: list[str] | None, skip_terrain: bool, skip_assets: bool) -> dict:
    report: dict = {"terrain": [], "assets": []}

    if not skip_terrain:
        for name in terrain_names if terrain_names is not None else list_terrains():
            try:
                report["terrain"].append(diff_terrain(name).as_dict())
            except Exception as error:
                report["terrain"].append({"name": name, "ok": False, "error": str(error)})

    if not skip_assets:
        for name in asset_names if asset_names is not None else list_assets():
            try:
                report["assets"].append(regenerate_asset(name))
            except Exception as error:
                report["assets"].append({"name": name, "ok": False, "error": str(error)})

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", action="append", help="terrain name to check (repeatable); default is every terrain under out/terrain/")
    parser.add_argument("--asset", action="append", help="asset name to check (repeatable); default is every asset under out/assets/")
    parser.add_argument("--skip-terrain", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    report = run(args.terrain, args.asset, args.skip_terrain, args.skip_assets)
    print(json.dumps(report, indent=2))

    failed = any(entry.get("ok") is False for entry in report["terrain"] + report["assets"])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
