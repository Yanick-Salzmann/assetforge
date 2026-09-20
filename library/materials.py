from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import tomli_w

from library import polyhaven
from terrain.config import PACKAGE_LIBRARY_DIR

LIBRARY_DIR = polyhaven.LIBRARY_DIR
MATERIALS_DIR = polyhaven.MATERIALS_DIR
INDEX_FILE = LIBRARY_DIR / "materials.json"

ASSET_SET_FILE = PACKAGE_LIBRARY_DIR / "asset_materials.toml"
ASSET_MATERIALS_DIR = LIBRARY_DIR / "asset_materials"
ASSET_LOCK_FILE = PACKAGE_LIBRARY_DIR / "asset_materials.lock.json"
ASSET_INDEX_FILE = LIBRARY_DIR / "asset_materials.json"

INDEX_VERSION = 1
MAP_ROLES = ("albedo", "normal", "roughness", "ao", "height")
REQUIRED_ROLES = ("albedo", "normal", "roughness")
MIN_TILING_M = 0.05
MAX_TILING_M = 256.0


class MaterialError(ValueError):
    """Raised when the material library does not describe a usable set."""


@dataclass(frozen=True)
class Material:
    """One CC0 tiling set: where its maps live and how big it is in the world."""

    name: str
    slug: str
    source: str
    licence: str
    resolution: str
    fmt: str
    tiling_m: float
    dimensions_mm: tuple[float, float]
    categories: tuple[str, ...]
    maps: Mapping[str, str]

    def path(self, role: str, root: Path = LIBRARY_DIR) -> Path:
        relative = self.maps.get(role)
        if relative is None:
            raise MaterialError(
                f"material {self.name!r} has no {role!r} map; it carries "
                f"{', '.join(sorted(self.maps))}"
            )
        return root / relative

    def paths(self, root: Path = LIBRARY_DIR) -> dict[str, Path]:
        return {role: root / relative for role, relative in self.maps.items()}

    def missing(self, root: Path = LIBRARY_DIR) -> tuple[str, ...]:
        return tuple(
            role for role, path in sorted(self.paths(root).items()) if not path.is_file()
        )

    def as_dict(self) -> dict:
        return {
            "slug": self.slug,
            "source": self.source,
            "license": self.licence,
            "resolution": self.resolution,
            "format": self.fmt,
            "tiling_m": self.tiling_m,
            "dimensions_mm": list(self.dimensions_mm),
            "categories": list(self.categories),
            "maps": dict(self.maps),
        }


@dataclass(frozen=True)
class MaterialIndex:
    """Every downloaded set, addressed by the layer-facing material name."""

    version: int
    materials: Mapping[str, Material]

    def __len__(self) -> int:
        return len(self.materials)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __contains__(self, name: str) -> bool:
        return name in self.materials

    def __getitem__(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError:
            raise MaterialError(
                f"unknown material {name!r}; expected one of {', '.join(self.names())}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.materials))

    def missing(self, root: Path = LIBRARY_DIR) -> dict[str, tuple[str, ...]]:
        found = {name: self[name].missing(root) for name in self.names()}
        return {name: roles for name, roles in found.items() if roles}

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "materials": {name: self[name].as_dict() for name in self.names()},
        }


def _tiling_m(name: str, override: float | None, dimensions_mm: list | None) -> float:
    if override is not None:
        tiling = float(override)
    elif dimensions_mm:
        tiling = float(dimensions_mm[0]) / 1000.0
    else:
        raise MaterialError(
            f"material {name!r} has no physical size; re-run the pull or set "
            f"tiling_m in materials.toml"
        )
    if not MIN_TILING_M <= tiling <= MAX_TILING_M:
        raise MaterialError(
            f"material {name!r} tiling_m {tiling} lies outside "
            f"[{MIN_TILING_M}, {MAX_TILING_M}] metres"
        )
    return tiling


def _material(name: str, entry: dict, tiling_override: float | None, materials_dir: Path) -> Material:
    files = entry.get("files", {})
    if not files:
        raise MaterialError(f"material {name!r} has no files in the lockfile")
    unknown = sorted(set(files) - set(MAP_ROLES))
    if unknown:
        raise MaterialError(f"material {name!r} carries unknown maps {', '.join(unknown)}")
    absent = [role for role in REQUIRED_ROLES if role not in files]
    if absent:
        raise MaterialError(f"material {name!r} is missing the {', '.join(absent)} map")
    dimensions = entry.get("dimensions_mm") or []
    return Material(
        name=name,
        slug=entry["slug"],
        source=entry["source"],
        licence=entry["license"],
        resolution=entry["resolution"],
        fmt=entry["format"],
        tiling_m=_tiling_m(name, tiling_override, dimensions),
        dimensions_mm=tuple(float(value) for value in dimensions[:2]),
        categories=tuple(entry.get("categories", [])),
        maps={
            role: (materials_dir.name + "/" + meta["path"])
            for role, meta in sorted(files.items())
        },
    )


def build(
    set_path: Path = polyhaven.SET_FILE,
    lock_path: Path = polyhaven.LOCK_FILE,
    materials_dir: Path = MATERIALS_DIR,
) -> MaterialIndex:
    """Derive the index from the pinned set and the lockfile that records what was pulled."""
    specs = polyhaven.load_set(set_path)
    locked = polyhaven.load_lock(lock_path).get("materials", {})
    unlocked = sorted(spec.name for spec in specs if spec.name not in locked)
    if unlocked:
        raise MaterialError(
            f"materials {', '.join(unlocked)} are pinned but not locked; run the pull script"
        )
    materials = {
        spec.name: _material(spec.name, locked[spec.name], spec.tiling_m, materials_dir)
        for spec in specs
    }
    return MaterialIndex(INDEX_VERSION, materials)


def build_assets() -> MaterialIndex:
    """build() pinned to the hard-surface asset material set instead of the terrain splat set."""
    return build(ASSET_SET_FILE, ASSET_LOCK_FILE, ASSET_MATERIALS_DIR)


def load_assets(path: Path = ASSET_INDEX_FILE) -> MaterialIndex:
    """load() pinned to the hard-surface asset material index."""
    return load(path)


def add_material_spec(
    name: str,
    slug: str,
    tiling_m: float | None = None,
    set_path: Path = ASSET_SET_FILE,
) -> None:
    """Add or update one [material.<name>] entry in an asset-material-style TOML set, the way a
    human hand-edit would."""
    data = tomllib.loads(set_path.read_text(encoding="utf-8")) if set_path.is_file() else {}
    entry: dict = {"slug": slug}
    if tiling_m is not None:
        entry["tiling_m"] = tiling_m
    data.setdefault("material", {})[name] = entry
    set_path.write_text(tomli_w.dumps(data), encoding="utf-8")


def pull_asset_material(
    name: str,
    slug: str,
    tiling_m: float | None = None,
    set_path: Path = ASSET_SET_FILE,
    materials_dir: Path = ASSET_MATERIALS_DIR,
    lock_path: Path = ASSET_LOCK_FILE,
    index_path: Path = ASSET_INDEX_FILE,
    session=None,
) -> dict:
    """Register `slug` as asset material `name`, pull the whole pinned set (catching drift the
    same way library.polyhaven's CLI would) and rebuild+write the asset material index - the
    three steps a human runs by hand, in-process for a live MCP session. Reverts the TOML edit
    and raises if the pull for this material fails, leaving the rest of the set untouched.
    """
    original = set_path.read_text(encoding="utf-8") if set_path.is_file() else None
    add_material_spec(name, slug, tiling_m, set_path)
    try:
        specs = polyhaven.load_set(set_path)
        report = polyhaven.pull(specs, materials_dir, lock_path, session=session)
        failure = next((f for f in report["failed"] if f["material"] == name), None)
        if failure is not None:
            raise MaterialError(f"pull failed for {name!r}: {failure['error']}")
        index = build(set_path, lock_path, materials_dir)
        write(index, index_path)
    except Exception:
        if original is None:
            set_path.unlink(missing_ok=True)
        else:
            set_path.write_text(original, encoding="utf-8")
        raise
    return {"report": report, "material": index[name].as_dict()}


def write(index: MaterialIndex, path: Path = INDEX_FILE) -> Path:
    path.write_text(json.dumps(index.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load(path: Path = INDEX_FILE) -> MaterialIndex:
    """Read the generated index, so consumers never need the TOML or the lockfile."""
    if not path.is_file():
        raise MaterialError(f"{path} does not exist; regenerate it with library.materials")
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != INDEX_VERSION:
        raise MaterialError(f"{path} is version {version}, expected {INDEX_VERSION}")
    materials = {}
    for name, entry in data.get("materials", {}).items():
        materials[name] = Material(
            name=name,
            slug=entry["slug"],
            source=entry["source"],
            licence=entry["license"],
            resolution=entry["resolution"],
            fmt=entry["format"],
            tiling_m=float(entry["tiling_m"]),
            dimensions_mm=tuple(float(value) for value in entry.get("dimensions_mm", ())[:2]),
            categories=tuple(entry.get("categories", [])),
            maps=dict(entry["maps"]),
        )
    return MaterialIndex(version, materials)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate a material index.")
    parser.add_argument("--set", type=Path, default=polyhaven.SET_FILE)
    parser.add_argument("--lock", type=Path, default=polyhaven.LOCK_FILE)
    parser.add_argument("--materials-dir", type=Path, default=MATERIALS_DIR)
    parser.add_argument("--check", action="store_true", help="fail if the index is out of date")
    parser.add_argument("--out", type=Path, default=INDEX_FILE)
    arguments = parser.parse_args()

    index = build(arguments.set, arguments.lock, arguments.materials_dir)
    absent = index.missing()
    for name, roles in sorted(absent.items()):
        print(f"missing on disk: {name} [{', '.join(roles)}]")
    if arguments.check:
        current = json.dumps(index.as_dict(), indent=2, sort_keys=True) + "\n"
        stored = arguments.out.read_text(encoding="utf-8") if arguments.out.is_file() else ""
        if current != stored:
            raise SystemExit(f"{arguments.out} is out of date; regenerate it")
        print(f"{arguments.out} is current, {len(index)} materials")
        return
    write(index, arguments.out)
    print(f"wrote {arguments.out} with {len(index)} materials")


if __name__ == "__main__":
    main()
