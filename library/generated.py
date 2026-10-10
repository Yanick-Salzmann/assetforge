"""Index approved assetforge-generated props as a kit pack.

A pack pinned in kits.toml with source = "assetforge" names an asset prefix as its slug. Every
out/assets/<slug>_<model>/ whose .gate.json records human approval is copied into
library/kits/<pack>/<model>.glb and locked in kits.lock.json beside the downloaded packs, so
library/kits.py indexes it like any other kit. The lock records the recipe that rebuilds each
model, since generated binaries are not committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from library import gltf
from library.kenney import KITS_DIR, LOCK_FILE, PackSpec, load_lock, load_set, save_lock
from terrain.config import ASSET_OUT_DIR, REPO_ROOT

SOURCE = "assetforge"
LICENCE = "CC0"
GATE_NAME = ".gate.json"
RECIPE_NAME = "recipe.py"


class GeneratedError(ValueError):
    """Raised when a pinned generated pack has no approved assets to index."""


def _approved(asset_dir: Path) -> bool:
    gate = asset_dir / GATE_NAME
    if not gate.is_file():
        return False
    return bool(json.loads(gate.read_text(encoding="utf-8")).get("approved", False))


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _collect(spec: PackSpec, asset_dir: Path, pack_dir: Path) -> dict[str, dict]:
    prefix = f"{spec.slug}_"
    models = {}
    for source in sorted(path for path in asset_dir.glob(f"{prefix}*") if path.is_dir()):
        glb = source / f"{source.name}.glb"
        if not glb.is_file() or not _approved(source):
            continue
        model = source.name[len(prefix):]
        destination = pack_dir / f"{model}.glb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(glb, destination)
        tris, bbox_min, bbox_max = gltf.mesh_stats(destination)
        models[model] = {
            "path": destination.relative_to(pack_dir.parent.parent).as_posix(),
            "tris": tris,
            "bbox_min_m": list(bbox_min),
            "bbox_max_m": list(bbox_max),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "recipe": _relative(source / RECIPE_NAME),
        }
    if not models:
        raise GeneratedError(f"{spec.slug}: no approved assets under {asset_dir} named {prefix}*")
    return models


def pull(
    specs: list[PackSpec] | None = None,
    kits_dir: Path = KITS_DIR,
    lock_path: Path = LOCK_FILE,
    asset_dir: Path = ASSET_OUT_DIR,
) -> dict:
    """Copy and lock every approved generated asset of each pinned assetforge pack."""
    all_specs = specs if specs is not None else load_set()
    generated_specs = [spec for spec in all_specs if spec.source == SOURCE]
    lock = load_lock(lock_path)
    packs = lock.setdefault("packs", {})
    report = {"pulled": [], "kept": [], "failed": []}

    for spec in generated_specs:
        try:
            pack_dir = kits_dir / spec.name
            if pack_dir.is_dir():
                shutil.rmtree(pack_dir)
            packs[spec.name] = {
                "source": SOURCE,
                "slug": spec.slug,
                "category": spec.category,
                "license": LICENCE,
                "models": _collect(spec, asset_dir, pack_dir),
            }
            report["pulled"].append(spec.name)
        except Exception as exc:
            report["failed"].append({"pack": spec.name, "error": str(exc)})

    stale = {
        name
        for name, entry in packs.items()
        if entry.get("source") == SOURCE and name not in {spec.name for spec in generated_specs}
    }
    for name in stale:
        del packs[name]

    save_lock(lock, lock_path)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Index approved generated props as kit packs.")
    parser.parse_args()
    report = pull()
    for key in ("pulled", "kept", "failed"):
        print(f"{key}: {len(report[key])}")
    for failure in report["failed"]:
        print(f"  FAILED {failure['pack']}: {failure['error']}")


if __name__ == "__main__":
    main()
