from __future__ import annotations

import hashlib
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import requests

from terrain.config import LIBRARY_DIR, MATERIALS_DIR, PACKAGE_LIBRARY_DIR

API_ROOT = "https://api.polyhaven.com"
ASSET_PAGE = "https://polyhaven.com/a"
SET_FILE = PACKAGE_LIBRARY_DIR / "materials.toml"
LOCK_FILE = PACKAGE_LIBRARY_DIR / "materials.lock.json"
CHUNK = 1 << 20

TEXTURE_CATALOG_CACHE = LIBRARY_DIR / ".texture_catalog_cache.json"
TEXTURE_CATALOG_MAX_AGE_S = 24 * 3600.0

MAP_ROLES = {
    "Diffuse": "albedo",
    "nor_gl": "normal",
    "Rough": "roughness",
    "AO": "ao",
    "Displacement": "height",
}


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    slug: str
    resolution: str
    fmt: str
    maps: tuple[str, ...]
    tiling_m: float | None = None


def load_set(path: Path = SET_FILE) -> list[MaterialSpec]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    specs = []
    for name, entry in sorted(data.get("material", {}).items()):
        specs.append(
            MaterialSpec(
                name=name,
                slug=entry["slug"],
                resolution=entry.get("resolution", defaults["resolution"]),
                fmt=entry.get("format", defaults["format"]),
                maps=tuple(entry.get("maps", defaults["maps"])),
                tiling_m=entry.get("tiling_m", defaults.get("tiling_m")),
            )
        )
    return specs


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path = LOCK_FILE) -> dict:
    if not path.is_file():
        return {"version": 1, "materials": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_lock(lock: dict, path: Path = LOCK_FILE) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_files(spec: MaterialSpec, session: requests.Session) -> dict[str, dict]:
    response = session.get(f"{API_ROOT}/files/{spec.slug}", timeout=60)
    response.raise_for_status()
    available = response.json()
    resolved = {}
    for map_name in spec.maps:
        role = MAP_ROLES[map_name]
        entry = available.get(map_name, {}).get(spec.resolution, {}).get(spec.fmt)
        if entry is None:
            raise KeyError(f"{spec.slug}: no {map_name} at {spec.resolution}/{spec.fmt}")
        resolved[role] = {
            "map": map_name,
            "url": entry["url"],
            "size": entry["size"],
            "md5": entry.get("md5"),
        }
    return resolved


def resolve_info(spec: MaterialSpec, session: requests.Session) -> dict:
    response = session.get(f"{API_ROOT}/info/{spec.slug}", timeout=60)
    response.raise_for_status()
    info = response.json()
    dimensions = info.get("dimensions")
    if not dimensions:
        raise KeyError(f"{spec.slug}: no physical dimensions in the asset info")
    return {
        "dimensions_mm": [float(value) for value in dimensions],
        "categories": sorted(info.get("categories", [])),
    }


def download(url: str, destination: Path, session: requests.Session) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_content(CHUNK):
                handle.write(block)
    partial.replace(destination)


def pull(
    specs: list[MaterialSpec] | None = None,
    materials_dir: Path = MATERIALS_DIR,
    lock_path: Path = LOCK_FILE,
    verify: bool = False,
    session: requests.Session | None = None,
) -> dict:
    specs = specs if specs is not None else load_set()
    session = session or requests.Session()
    lock = load_lock(lock_path)
    locked = lock.setdefault("materials", {})
    report = {"downloaded": [], "kept": [], "repaired": [], "failed": []}

    for spec in specs:
        entry = locked.get(spec.name)
        previous = entry
        needs_resolve = (
            entry is None
            or entry.get("slug") != spec.slug
            or entry.get("resolution") != spec.resolution
            or entry.get("format") != spec.fmt
            or entry.get("dimensions_mm") is None
            or sorted(entry.get("files", {})) != sorted(MAP_ROLES[m] for m in spec.maps)
        )
        try:
            resolved = None
            if needs_resolve:
                resolved = (resolve_files(spec, session), resolve_info(spec, session))
        except Exception as exc:
            report["failed"].append({"material": spec.name, "error": str(exc)})
            continue

        if resolved is not None:
            files, info = resolved
            entry = {
                **info,
                "slug": spec.slug,
                "resolution": spec.resolution,
                "format": spec.fmt,
                "license": "CC0",
                "source": f"{ASSET_PAGE}/{spec.slug}",
                "files": {
                    role: {**meta, "path": f"{spec.name}/{role}.{spec.fmt}", "sha256": None}
                    for role, meta in files.items()
                },
            }
            if previous is not None:
                for role, meta in entry["files"].items():
                    kept = previous.get("files", {}).get(role)
                    if kept is not None and kept.get("url") == meta["url"]:
                        meta["sha256"] = kept.get("sha256")

        for role, meta in entry["files"].items():
            destination = materials_dir / meta["path"]
            label = f"{spec.name}/{role}"
            if destination.is_file():
                if meta["sha256"] is None:
                    meta["sha256"] = sha256_of(destination)
                    report["repaired"].append(label)
                    continue
                if not verify or sha256_of(destination) == meta["sha256"]:
                    report["kept"].append(label)
                    continue
                destination.unlink()
            try:
                download(meta["url"], destination, session)
            except Exception as exc:
                report["failed"].append({"material": label, "error": str(exc)})
                continue
            meta["sha256"] = sha256_of(destination)
            report["downloaded"].append(label)

        locked[spec.name] = entry

    for stale in set(locked) - {spec.name for spec in specs}:
        del locked[stale]

    save_lock(lock, lock_path)
    return report


def fetch_texture_catalog(
    session: requests.Session | None = None,
    cache_path: Path = TEXTURE_CATALOG_CACHE,
    max_age_s: float = TEXTURE_CATALOG_MAX_AGE_S,
) -> dict[str, dict]:
    """PolyHaven's full public texture catalog (GET /assets?t=textures), cached on disk since it
    is a multi-MB response that barely changes between sessions."""
    if cache_path.is_file() and time.time() - cache_path.stat().st_mtime < max_age_s:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    session = session or requests.Session()
    response = session.get(f"{API_ROOT}/assets", params={"t": "textures"}, timeout=60)
    response.raise_for_status()
    catalog = response.json()
    cache_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog


def search_textures(
    query: str = "",
    session: requests.Session | None = None,
    cache_path: Path = TEXTURE_CATALOG_CACHE,
    max_age_s: float = TEXTURE_CATALOG_MAX_AGE_S,
) -> list[dict]:
    """Keyword-filter the texture catalog by slug/name/category, for an agent to pick a slug from
    before calling library.materials.pull_asset_material()."""
    catalog = fetch_texture_catalog(session, cache_path, max_age_s)
    keyword = query.strip().lower()
    results = []
    for slug, entry in catalog.items():
        name = entry.get("name", slug)
        categories = entry.get("categories", [])
        haystack = " ".join([slug, name, *categories]).lower()
        if keyword and keyword not in haystack:
            continue
        results.append({"slug": slug, "name": name, "categories": categories})
    results.sort(key=lambda item: item["slug"])
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download a pinned CC0 material set.")
    parser.add_argument("--set", type=Path, default=SET_FILE, help="pinned material set TOML")
    parser.add_argument("--materials-dir", type=Path, default=MATERIALS_DIR, help="where to write downloaded files")
    parser.add_argument("--lock", type=Path, default=LOCK_FILE, help="lockfile to read and update")
    parser.add_argument("--verify", action="store_true", help="rehash existing files")
    args = parser.parse_args()
    report = pull(load_set(args.set), args.materials_dir, args.lock, verify=args.verify)
    for key in ("downloaded", "repaired", "kept", "failed"):
        print(f"{key}: {len(report[key])}")
    for failure in report["failed"]:
        print(f"  FAILED {failure['material']}: {failure['error']}")


if __name__ == "__main__":
    main()
