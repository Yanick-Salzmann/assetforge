from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from library import kenney
from terrain.config import LIBRARY_DIR

KITS_DIR = kenney.KITS_DIR
INDEX_FILE = LIBRARY_DIR / "kits.json"

INDEX_VERSION = 1


class KitError(ValueError):
    """Raised when the kit library does not describe a usable prop."""


@dataclass(frozen=True)
class Prop:
    """One CC0 model pulled from a prop kit: where it lives and how big it is."""

    name: str
    pack: str
    source: str
    licence: str
    category: str
    tris: int
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    glb: str

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        return tuple(mx - mn for mn, mx in zip(self.bbox_min_m, self.bbox_max_m))

    def path(self, root: Path = LIBRARY_DIR) -> Path:
        return root / self.glb

    def as_dict(self) -> dict:
        return {
            "pack": self.pack,
            "source": self.source,
            "license": self.licence,
            "category": self.category,
            "tris": self.tris,
            "bbox_min_m": list(self.bbox_min_m),
            "bbox_max_m": list(self.bbox_max_m),
            "glb": self.glb,
        }


@dataclass(frozen=True)
class KitIndex:
    """Every downloaded prop, addressed as '<pack>/<model>'."""

    version: int
    props: Mapping[str, Prop]

    def __len__(self) -> int:
        return len(self.props)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __contains__(self, name: str) -> bool:
        return name in self.props

    def __getitem__(self, name: str) -> Prop:
        try:
            return self.props[name]
        except KeyError:
            raise KitError(f"unknown prop {name!r}; expected one of {', '.join(self.names())}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.props))

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({prop.category for prop in self.props.values()}))

    def find(
        self,
        category: str | None = None,
        max_tris: int | None = None,
        max_dimension_m: float | None = None,
    ) -> tuple[Prop, ...]:
        """Candidates for reuse before generating a prop from scratch, smallest tri count first."""
        results = []
        for name in self.names():
            prop = self.props[name]
            if category is not None and prop.category != category:
                continue
            if max_tris is not None and prop.tris > max_tris:
                continue
            if max_dimension_m is not None and max(prop.dimensions_m) > max_dimension_m:
                continue
            results.append(prop)
        return tuple(sorted(results, key=lambda prop: prop.tris))

    def missing(self, root: Path = LIBRARY_DIR) -> tuple[str, ...]:
        return tuple(name for name in self.names() if not self[name].path(root).is_file())

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "props": {name: self[name].as_dict() for name in self.names()},
        }


def build(
    set_path: Path = kenney.SET_FILE,
    lock_path: Path = kenney.LOCK_FILE,
) -> KitIndex:
    """Derive the index from the pinned pack set and the lockfile that records what was pulled."""
    specs = kenney.load_set(set_path)
    locked = kenney.load_lock(lock_path).get("packs", {})
    unlocked = sorted(spec.name for spec in specs if spec.name not in locked)
    if unlocked:
        raise KitError(f"packs {', '.join(unlocked)} are pinned but not locked; run the pull script")

    props = {}
    for spec in specs:
        entry = locked[spec.name]
        for model, meta in entry["models"].items():
            name = f"{spec.name}/{model}"
            props[name] = Prop(
                name=name,
                pack=spec.name,
                source=entry["source"],
                licence=entry["license"],
                category=entry["category"],
                tris=meta["tris"],
                bbox_min_m=tuple(meta["bbox_min_m"]),
                bbox_max_m=tuple(meta["bbox_max_m"]),
                glb=meta["path"],
            )
    return KitIndex(INDEX_VERSION, props)


def write(index: KitIndex, path: Path = INDEX_FILE) -> Path:
    path.write_text(json.dumps(index.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load(path: Path = INDEX_FILE) -> KitIndex:
    """Read the generated index, so consumers never need the TOML or the lockfile."""
    if not path.is_file():
        raise KitError(f"{path} does not exist; regenerate it with library.kits")
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != INDEX_VERSION:
        raise KitError(f"{path} is version {version}, expected {INDEX_VERSION}")
    props = {}
    for name, entry in data.get("props", {}).items():
        props[name] = Prop(
            name=name,
            pack=entry["pack"],
            source=entry["source"],
            licence=entry["license"],
            category=entry["category"],
            tris=int(entry["tris"]),
            bbox_min_m=tuple(entry["bbox_min_m"]),
            bbox_max_m=tuple(entry["bbox_max_m"]),
            glb=entry["glb"],
        )
    return KitIndex(version, props)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the kit index.")
    parser.add_argument("--check", action="store_true", help="fail if the index is out of date")
    parser.add_argument("--out", type=Path, default=INDEX_FILE)
    arguments = parser.parse_args()

    index = build()
    absent = index.missing()
    for name in absent:
        print(f"missing on disk: {name}")
    if arguments.check:
        current = json.dumps(index.as_dict(), indent=2, sort_keys=True) + "\n"
        stored = arguments.out.read_text(encoding="utf-8") if arguments.out.is_file() else ""
        if current != stored:
            raise SystemExit(f"{arguments.out} is out of date; regenerate it")
        print(f"{arguments.out} is current, {len(index)} props")
        return
    write(index, arguments.out)
    print(f"wrote {arguments.out} with {len(index)} props")


if __name__ == "__main__":
    main()
