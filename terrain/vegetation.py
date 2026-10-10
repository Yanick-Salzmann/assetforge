from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from terrain.config import MapConfigError
from terrain.splat import Species

KIND_SIZE_M: dict[str, float] = {
    "conifer": 8.0,
    "broadleaf": 7.0,
    "cactus": 4.0,
    "shrub": 1.4,
    "rock": 1.6,
    "grass": 0.45,
    "flower": 0.45,
    "debris": 0.5,
}
KIND_TRI_BUDGET: dict[str, int] = {
    "conifer": 600,
    "broadleaf": 600,
    "cactus": 1200,
    "shrub": 200,
    "rock": 500,
    "grass": 250,
    "flower": 200,
    "debris": 400,
}
METRIC_SOURCES = ("assetforge",)


@dataclass(frozen=True)
class Variant:
    """One kit mesh a species' instances can use, and the factor that brings it to kind size."""

    mesh: str
    glb: str
    tris: int
    unit_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh": self.mesh,
            "glb": self.glb,
            "tris": self.tris,
            "unit_scale": self.unit_scale,
        }


def default_index() -> Any:
    from library import kits

    return kits.build()


def resolve(species: Species, index: Any) -> tuple[Variant, ...]:
    """The species' meshes as variants: props from a metric source keep their modelled size,
    any other is scaled so its largest extent is the kind's size."""
    variants = []
    for mesh in species.meshes:
        if mesh not in index:
            raise MapConfigError(f"species {species.name!r} names unknown kit prop {mesh!r}")
        prop = index[mesh]
        extent = max(prop.dimensions_m)
        if extent <= 0.0:
            raise MapConfigError(f"kit prop {mesh!r} has no extent")
        unit_scale = 1.0 if prop.source in METRIC_SOURCES else KIND_SIZE_M[species.kind] / extent
        variants.append(Variant(mesh, prop.glb, prop.tris, unit_scale))
    return tuple(variants)


def resolve_all(species: tuple[Species, ...], index: Any | None = None) -> Mapping[str, tuple[Variant, ...]]:
    if not any(entry.meshes for entry in species):
        return {entry.name: () for entry in species}
    source = default_index() if index is None else index
    return {entry.name: resolve(entry, source) for entry in species}
