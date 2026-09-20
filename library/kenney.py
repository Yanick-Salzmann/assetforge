from __future__ import annotations

import hashlib
import io
import json
import re
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from library import gltf

ASSET_PAGE = "https://kenney.nl/assets"
LIBRARY_DIR = Path(__file__).resolve().parent
KITS_DIR = LIBRARY_DIR / "kits"
SET_FILE = LIBRARY_DIR / "kits.toml"
LOCK_FILE = LIBRARY_DIR / "kits.lock.json"
MODEL_PREFIXES = ("Models/GLB format/", "Models/GLTF format/")
MODEL_SUFFIXES = (".glb", ".gltf")
ZIP_HREF = re.compile(r'href=[\'"]([^\'"]+\.zip)[\'"]')


class KenneyError(ValueError):
    """Raised when a Kenney pack page or archive does not look like a usable CC0 pack."""


@dataclass(frozen=True)
class PackSpec:
    name: str
    slug: str
    category: str
    source: str = "kenney"


def load_set(path: Path = SET_FILE) -> list[PackSpec]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    specs = []
    for name, entry in sorted(data.get("pack", {}).items()):
        specs.append(
            PackSpec(
                name=name,
                slug=entry["slug"],
                category=entry.get("category", defaults.get("category", "prop")),
                source=entry.get("source", defaults.get("source", "kenney")),
            )
        )
    return specs


def load_lock(path: Path = LOCK_FILE) -> dict:
    if not path.is_file():
        return {"version": 1, "packs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_lock(lock: dict, path: Path = LOCK_FILE) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_zip_url(slug: str, session: requests.Session) -> str:
    response = session.get(f"{ASSET_PAGE}/{slug}", timeout=30)
    response.raise_for_status()
    matches = [href for href in ZIP_HREF.findall(response.text) if slug in href]
    if not matches:
        raise KenneyError(f"{slug}: no zip download link found on the asset page")
    return matches[-1]


def _verify_cc0(archive: zipfile.ZipFile, slug: str) -> None:
    try:
        text = archive.read("License.txt").decode("utf-8", errors="replace")
    except KeyError:
        raise KenneyError(f"{slug}: pack has no License.txt") from None
    if "CC0" not in text:
        raise KenneyError(f"{slug}: License.txt does not declare CC0")


def _model_prefix(archive: zipfile.ZipFile, slug: str) -> str:
    names = archive.namelist()
    for prefix in MODEL_PREFIXES:
        if any(name.startswith(prefix) and name.lower().endswith(MODEL_SUFFIXES) for name in names):
            return prefix
    raise KenneyError(f"{slug}: pack has no GLB or GLTF models")


def _extract_pack(archive: zipfile.ZipFile, slug: str, prefix: str, pack_dir: Path) -> dict[str, dict]:
    models = {}
    for member in sorted(archive.namelist()):
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        relative = member[len(prefix):]
        destination = pack_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
        if relative.lower().endswith(MODEL_SUFFIXES):
            tris, bbox_min, bbox_max = gltf.mesh_stats(destination)
            models[Path(relative).stem] = {
                "path": destination.relative_to(pack_dir.parent.parent).as_posix(),
                "tris": tris,
                "bbox_min_m": list(bbox_min),
                "bbox_max_m": list(bbox_max),
            }
    if not models:
        raise KenneyError(f"{slug}: no models extracted")
    return models


def pull(
    specs: list[PackSpec] | None = None,
    kits_dir: Path = KITS_DIR,
    lock_path: Path = LOCK_FILE,
    session: requests.Session | None = None,
) -> dict:
    all_specs = specs if specs is not None else load_set()
    kenney_specs = [spec for spec in all_specs if spec.source == "kenney"]
    session = session or requests.Session()
    lock = load_lock(lock_path)
    packs = lock.setdefault("packs", {})
    report = {"pulled": [], "kept": [], "failed": []}

    for spec in kenney_specs:
        try:
            zip_url = resolve_zip_url(spec.slug, session)
            existing = packs.get(spec.name)
            if existing is not None and existing.get("zip_url") == zip_url and (kits_dir / spec.name).is_dir():
                report["kept"].append(spec.name)
                continue
            response = session.get(zip_url, timeout=180)
            response.raise_for_status()
            archive = zipfile.ZipFile(io.BytesIO(response.content))
            _verify_cc0(archive, spec.slug)
            prefix = _model_prefix(archive, spec.slug)
            pack_dir = kits_dir / spec.name
            models = _extract_pack(archive, spec.slug, prefix, pack_dir)
            packs[spec.name] = {
                "source": "kenney",
                "slug": spec.slug,
                "category": spec.category,
                "license": "CC0",
                "source_url": f"{ASSET_PAGE}/{spec.slug}",
                "zip_url": zip_url,
                "zip_sha256": hashlib.sha256(response.content).hexdigest(),
                "models": models,
            }
            report["pulled"].append(spec.name)
        except Exception as exc:
            report["failed"].append({"pack": spec.name, "error": str(exc)})

    stale = {
        name
        for name, entry in packs.items()
        if entry.get("source") == "kenney" and name not in {spec.name for spec in kenney_specs}
    }
    for name in stale:
        del packs[name]

    save_lock(lock, lock_path)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download the pinned CC0 Kenney kit library.")
    parser.parse_args()
    report = pull()
    for key in ("pulled", "kept", "failed"):
        print(f"{key}: {len(report[key])}")
    for failure in report["failed"]:
        print(f"  FAILED {failure['pack']}: {failure['error']}")


if __name__ == "__main__":
    main()
