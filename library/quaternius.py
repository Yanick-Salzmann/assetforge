"""Manual-staging puller for Quaternius packs.

quaternius.itch.io gates downloads behind a claim/session-token flow rather than a
stable public zip URL, so there is no scrape-and-fetch path like library/kenney.py's.
Instead: pin the pack in kits.toml with source = "quaternius", download the zip by
hand from quaternius.itch.io, drop it at library/_staging/<slug>.zip, then run this
module. Extraction and stats reuse library/gltf.py exactly as the Kenney path does,
and the result lands in the same kits.lock.json / kits.json that library/kits.py
already indexes, so nothing downstream needs to know a pack came from here.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from library import gltf
from library.kenney import KITS_DIR, LIBRARY_DIR, LOCK_FILE, PackSpec, load_lock, load_set, save_lock

STAGING_DIR = LIBRARY_DIR / "_staging"
MODEL_SUFFIXES = (".glb", ".gltf")
SOURCE = "quaternius"


class QuaterniusError(ValueError):
    """Raised when a staged Quaternius pack cannot be extracted or is missing."""


def _extract_pack(archive: zipfile.ZipFile, slug: str, pack_dir: Path) -> dict[str, dict]:
    models = {}
    for member in sorted(archive.namelist()):
        if member.endswith("/"):
            continue
        destination = pack_dir / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
        if member.lower().endswith(MODEL_SUFFIXES):
            tris, bbox_min, bbox_max = gltf.mesh_stats(destination)
            models[Path(member).stem] = {
                "path": destination.relative_to(pack_dir.parent.parent).as_posix(),
                "tris": tris,
                "bbox_min_m": list(bbox_min),
                "bbox_max_m": list(bbox_max),
            }
    if not models:
        raise QuaterniusError(f"{slug}: no models extracted")
    return models


def pull(
    specs: list[PackSpec] | None = None,
    kits_dir: Path = KITS_DIR,
    lock_path: Path = LOCK_FILE,
    staging_dir: Path = STAGING_DIR,
) -> dict:
    """Extract every pinned Quaternius pack whose zip has been staged, into the shared lock."""
    all_specs = specs if specs is not None else load_set()
    quaternius_specs = [spec for spec in all_specs if spec.source == SOURCE]
    lock = load_lock(lock_path)
    packs = lock.setdefault("packs", {})
    report = {"pulled": [], "kept": [], "failed": []}

    for spec in quaternius_specs:
        try:
            zip_path = staging_dir / f"{spec.slug}.zip"
            if not zip_path.is_file():
                raise QuaterniusError(
                    f"{spec.slug}: no staged zip at {zip_path}; download it from "
                    f"quaternius.itch.io and drop it there"
                )
            zip_bytes = zip_path.read_bytes()
            zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
            existing = packs.get(spec.name)
            if existing is not None and existing.get("zip_sha256") == zip_sha256 and (kits_dir / spec.name).is_dir():
                report["kept"].append(spec.name)
                continue
            archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
            pack_dir = kits_dir / spec.name
            models = _extract_pack(archive, spec.slug, pack_dir)
            packs[spec.name] = {
                "source": SOURCE,
                "slug": spec.slug,
                "category": spec.category,
                "license": "CC0",
                "source_url": f"https://quaternius.itch.io/{spec.slug}",
                "zip_url": f"file:_staging/{zip_path.name}",
                "zip_sha256": zip_sha256,
                "models": models,
            }
            report["pulled"].append(spec.name)
        except Exception as exc:
            report["failed"].append({"pack": spec.name, "error": str(exc)})

    stale = {
        name
        for name, entry in packs.items()
        if entry.get("source") == SOURCE and name not in {spec.name for spec in quaternius_specs}
    }
    for name in stale:
        del packs[name]

    save_lock(lock, lock_path)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract pinned Quaternius packs staged under library/_staging/."
    )
    parser.parse_args()
    report = pull()
    for key in ("pulled", "kept", "failed"):
        print(f"{key}: {len(report[key])}")
    for failure in report["failed"]:
        print(f"  FAILED {failure['pack']}: {failure['error']}")


if __name__ == "__main__":
    main()
