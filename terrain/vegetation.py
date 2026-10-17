from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from terrain.config import MapConfigError

if TYPE_CHECKING:
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
IMPOSTOR_KINDS = ("conifer", "broadleaf", "shrub")
LOD_SCREEN_FRACTIONS: tuple[float, ...] = (0.25, 0.1, 0.04)
REFERENCE_VIEWPORT_PX = 1080
REFERENCE_FOV_DEG = 50.0


def switch_distances(diameter_m: float, transitions: int = len(LOD_SCREEN_FRACTIONS)) -> tuple[float, ...]:
    """Camera distances in metres at which a variant whose bounding sphere is diameter_m across
    shrinks below each LOD_SCREEN_FRACTIONS share of a REFERENCE_VIEWPORT_PX-tall view with the
    viewer's REFERENCE_FOV_DEG: LOD0 to LOD1, LOD1 to LOD2, then LOD2 to the impostor."""
    focal = 2.0 * math.tan(math.radians(REFERENCE_FOV_DEG) * 0.5)
    return tuple(diameter_m / (focal * fraction) for fraction in LOD_SCREEN_FRACTIONS[:transitions])


def switch_pixels(transitions: int = len(LOD_SCREEN_FRACTIONS)) -> tuple[int, ...]:
    """The bounding sphere's on-screen size at each switch distance, in reference pixels."""
    return tuple(round(fraction * REFERENCE_VIEWPORT_PX) for fraction in LOD_SCREEN_FRACTIONS[:transitions])


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
