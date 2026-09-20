from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, fields, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from library import materials as material_library
from terrain import beauty as beauty_mod
from terrain import channels as channels_mod
from terrain import config, preview, synth
from terrain import erosion as erosion_mod
from terrain import splat as splat_mod
from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, detail_region
from terrain.channels import ChannelStack, WaterLevel, WaterParams
from terrain.config import MapConfig, MapConfigError
from terrain.erosion import ErosionParams, ErosionResult
from terrain.splat import SplatResult

SESSION_NAME = "session.json"
HEIGHT_CACHE = "height.npy"
BASE_CACHE = "height_base.npy"
CHANNEL_SHEET_NAME = "preview_channels.png"

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

EROSION_FIELDS = frozenset(f.name for f in fields(ErosionParams))
WATER_FIELDS = frozenset(f.name for f in fields(WaterParams))


def validate_name(name: str) -> str:
    """Reject anything that would not survive as a directory under out/terrain/."""
    if not NAME_PATTERN.match(name):
        raise MapConfigError(
            f"terrain name {name!r} must be lowercase alphanumerics, dashes or underscores"
        )
    return name


def overrides(values: Mapping[str, Any] | None, allowed: frozenset[str], label: str) -> dict:
    if not values:
        return {}
    unknown = sorted(set(values) - allowed)
    if unknown:
        known = ", ".join(sorted(allowed))
        raise MapConfigError(f"unknown {label} {', '.join(unknown)}; expected one of {known}")
    return dict(values)


def _write_array(path: Path, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.detach().to(device="cpu", dtype=torch.float32).numpy())


def _read_array(path: Path) -> torch.Tensor | None:
    if not path.is_file():
        return None
    return torch.from_numpy(np.load(path))


def _stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


@dataclass
class TerrainSession:
    """One terrain under out/terrain/<name>/, walked from synth through erosion to channels."""

    cfg: MapConfig
    shape: str = synth.DEFAULT_SHAPE
    erosion: str = erosion_mod.DEFAULT_PRESET
    erosion_overrides: dict[str, Any] = dataclass_field(default_factory=dict)
    water_overrides: dict[str, Any] = dataclass_field(default_factory=dict)
    biome_file: str | None = None
    sharpness_override: float | None = None
    stage: str = "synth"
    base: torch.Tensor | None = None
    height: torch.Tensor | None = None
    result: ErosionResult | None = None
    stack: ChannelStack | None = None
    water: WaterLevel | None = None
    splat: SplatResult | None = None
    timings: dict[str, float] = dataclass_field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.cfg.name

    def out_dir(self) -> Path:
        return self.cfg.out_dir()

    def erosion_params(self) -> ErosionParams:
        return replace(erosion_mod.erosion_preset(self.erosion), **self.erosion_overrides)

    def water_params(self) -> WaterParams:
        return replace(WaterParams(), **self.water_overrides)

    def synthesise(self) -> TerrainSession:
        started = time.perf_counter()
        self.base = synth.heightfield(self.cfg, self.shape)
        self.height = self.base
        self.result = None
        self.stack = None
        self.water = None
        self.stage = "synth"
        self.timings = {"synth_s": time.perf_counter() - started}
        return self.save()

    def erode(self) -> TerrainSession:
        if self.base is None:
            self.synthesise()
        started = time.perf_counter()
        result = erosion_mod.erode(self.cfg, self.base, self.erosion_params())
        breached = channels_mod.breach_depressions(self.cfg, result.height)
        self.result = replace(result, height=breached)
        self.height = self.result.height
        self.stack = None
        self.water = None
        self.splat = None
        self.stage = "eroded"
        self.timings["erode_s"] = time.perf_counter() - started
        return self.save()

    def build_channels(self) -> TerrainSession:
        result = self.eroded()
        started = time.perf_counter()
        water = self.water_params()
        basin = channels_mod.drainage(self.cfg, result.height, water)
        self.stack = channels_mod.build(self.cfg, result, water=water, basin=basin)
        self.water = channels_mod.summarise_water(self.cfg, result, water, basin)
        self.splat = None
        self.stage = "channels"
        self.timings["channels_s"] = time.perf_counter() - started
        return self.save()

    def apply_biome(self, biome_file: str, sharpness: float | None = None) -> TerrainSession:
        """Evaluate a biome's rules over the channel stack and write its splat textures.

        biome_file may be a name from biomes/ or a path to a biome TOML file. Re-running with
        the same file replays deterministically, the same way erosion and channels do.
        """
        started = time.perf_counter()
        loaded = resolve_biome(biome_file, material_index())
        self.splat = splat_mod.render(loaded, self.channels(), sharpness)
        self.biome_file = biome_file
        self.sharpness_override = sharpness
        self.stage = "splat"
        self.timings["splat_s"] = time.perf_counter() - started
        splat_mod.write(self.splat, self.out_dir())
        return self.save()

    def splat_result(self) -> SplatResult:
        """The current splat, replayed from the recorded biome if it is not resident."""
        if self.splat is None:
            if self.biome_file is None:
                raise MapConfigError(
                    f"terrain {self.name!r} has no biome applied yet; call apply_rules first"
                )
            self.apply_biome(self.biome_file, self.sharpness_override)
        assert self.splat is not None
        return self.splat

    def eroded(self) -> ErosionResult:
        """The erosion result, replayed from the recorded seed and params if it is not resident."""
        if self.result is None:
            self.erode()
        assert self.result is not None
        return self.result

    def channels(self) -> ChannelStack:
        if self.stack is None:
            self.build_channels()
        assert self.stack is not None
        return self.stack

    def field(self, name: str) -> torch.Tensor:
        if name == "height":
            return self.current()
        return self.channels()[name]

    def current(self) -> torch.Tensor:
        if self.height is None:
            self.synthesise()
        assert self.height is not None
        return self.height

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": self.cfg.as_dict(),
            "shape": self.shape,
            "erosion": self.erosion,
            "erosion_overrides": self.erosion_overrides,
            "water_overrides": self.water_overrides,
            "biome_file": self.biome_file,
            "sharpness_override": self.sharpness_override,
            "stage": self.stage,
            "timings": {name: round(value, 3) for name, value in self.timings.items()},
        }
        if self.result is not None:
            payload["erosion_stats"] = self.result.stats()
        if self.water is not None:
            payload["water"] = self.water.as_dict()
        if self.stack is not None:
            payload["channels"] = list(self.stack.filled())
        if self.splat is not None:
            payload["splat"] = self.splat.as_dict()
        return payload

    def save(self) -> TerrainSession:
        target = self.out_dir()
        target.mkdir(parents=True, exist_ok=True)
        (target / SESSION_NAME).write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )
        if self.base is not None:
            _write_array(target / BASE_CACHE, self.base)
        if self.height is not None:
            _write_array(target / HEIGHT_CACHE, self.height)
        return self

    @classmethod
    def load(cls, name: str) -> TerrainSession:
        target = config.TERRAIN_OUT_DIR / validate_name(name)
        manifest = target / SESSION_NAME
        if not manifest.is_file():
            raise MapConfigError(f"no terrain session at {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        session = cls(
            cfg=MapConfig.from_dict(payload["config"]),
            shape=str(payload.get("shape", synth.DEFAULT_SHAPE)),
            erosion=str(payload.get("erosion", erosion_mod.DEFAULT_PRESET)),
            erosion_overrides=dict(payload.get("erosion_overrides", {})),
            water_overrides=dict(payload.get("water_overrides", {})),
            biome_file=payload.get("biome_file"),
            sharpness_override=(
                float(payload["sharpness_override"])
                if payload.get("sharpness_override") is not None
                else None
            ),
            stage=str(payload.get("stage", "synth")),
            timings=dict(payload.get("timings", {})),
        )
        session.base = _read_array(target / BASE_CACHE)
        session.height = _read_array(target / HEIGHT_CACHE)
        if session.height is None:
            session.synthesise()
        return session


_SESSIONS: dict[str, TerrainSession] = {}
_MATERIAL_INDEX: material_library.MaterialIndex | None = None
BEAUTY_VIEWS = ("three_quarter", "ground")


def material_index() -> material_library.MaterialIndex:
    global _MATERIAL_INDEX
    if _MATERIAL_INDEX is None:
        _MATERIAL_INDEX = material_library.load()
    return _MATERIAL_INDEX


def resolve_biome(biome_file: str, index: material_library.MaterialIndex) -> splat_mod.Biome:
    """A biome by path if biome_file names a file on disk, else by name: a workspace override
    first, then the packaged preset (config.resolve_biome_path)."""
    direct = Path(biome_file)
    if direct.is_file():
        return splat_mod.load_biome(direct, index)
    path = config.resolve_biome_path(direct.stem)
    if not path.is_file():
        known = ", ".join(config.available_biome_names()) or "none"
        raise splat_mod.SplatRuleError(f"unknown biome {direct.stem!r}; known biomes: {known}")
    return splat_mod.load_biome(path, index)


def create(
    name: str,
    seed: int = 0,
    resolution: int = 1024,
    world_size_m: float = 4096.0,
    height_range_m: float = 600.0,
    sea_level_m: float = 0.0,
    shape: str = synth.DEFAULT_SHAPE,
    overwrite: bool = False,
) -> TerrainSession:
    """Synthesise a fresh base heightfield and register it under its name."""
    validate_name(name)
    if not overwrite and (config.TERRAIN_OUT_DIR / name / SESSION_NAME).is_file():
        raise MapConfigError(f"terrain {name!r} already exists; pass overwrite=True to replace it")
    synth.shape_preset(shape)
    cfg = MapConfig(
        name=name,
        resolution=resolution,
        world_size_m=world_size_m,
        height_range_m=height_range_m,
        sea_level_m=sea_level_m,
        seed=seed,
    )
    session = TerrainSession(cfg=cfg, shape=shape).synthesise()
    _SESSIONS[name] = session
    return session


def open_session(name: str) -> TerrainSession:
    """The resident session for this name, reloaded from disk when the server has restarted."""
    validate_name(name)
    session = _SESSIONS.get(name)
    if session is None:
        session = TerrainSession.load(name)
        _SESSIONS[name] = session
    return session


def forget(name: str) -> None:
    _SESSIONS.pop(name, None)


def resident() -> tuple[str, ...]:
    return tuple(sorted(_SESSIONS))


def region_for(centre: Sequence[float] | None, span: float) -> tuple[float, ...] | None:
    return None if centre is None else detail_region(centre, span)


def hillshade(
    session: TerrainSession,
    centre: Sequence[float] | None = None,
    span: float = 0.25,
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> Preview:
    return preview.hillshade_preview(
        session.current(),
        session.cfg,
        session.out_dir() / preview.HILLSHADE_NAME,
        budget,
        region_for(centre, span),
        **kwargs,
    )


def channel_sheet(session: TerrainSession, budget: PreviewBudget = DEFAULT_BUDGET) -> Preview:
    stack = session.channels()
    return preview.channel_sheet_preview(
        stack.as_dict(),
        session.out_dir() / CHANNEL_SHEET_NAME,
        budget=budget,
    )


def channel_view(
    session: TerrainSession,
    name: str,
    centre: Sequence[float] | None = None,
    span: float = 0.25,
    rescale: bool = False,
    budget: PreviewBudget = DEFAULT_BUDGET,
) -> tuple[Preview, dict[str, float]]:
    values = session.field(name)
    view = preview.channel_preview(
        values,
        session.out_dir() / f"channel_{name}.png",
        name,
        rescale,
        budget,
        region_for(centre, span),
    )
    return view, _stats(values)


def channel_stats(session: TerrainSession) -> dict[str, dict[str, float]]:
    stack = session.channels()
    data = stack.data
    low = data.amin(dim=(0, 1))
    mean = data.mean(dim=(0, 1))
    high = data.amax(dim=(0, 1))
    return {
        name: {
            "min": float(low[position]),
            "mean": float(mean[position]),
            "max": float(high[position]),
        }
        for position, name in enumerate(stack.names)
        if name in stack
    }


def splat_view(
    session: TerrainSession,
    centre: Sequence[float] | None = None,
    span: float = 0.25,
    budget: PreviewBudget = DEFAULT_BUDGET,
) -> tuple[Preview, str]:
    """The current splat's false-colour composite and its coverage table, replaying if needed."""
    result = session.splat_result()
    view = preview.splat_composite_preview(
        result.weights,
        session.out_dir() / preview.SPLAT_NAME,
        session.cfg,
        budget,
        region_for(centre, span),
    )
    return view, preview.coverage_table(result.coverage)


def beauty_view(
    session: TerrainSession,
    view: str = "three_quarter",
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> tuple[beauty_mod.BeautyRender, Preview]:
    """Render both beauty views and hand back the budgeted preview of the one asked for."""
    if view not in BEAUTY_VIEWS:
        raise MapConfigError(f"view {view!r} must be one of {', '.join(BEAUTY_VIEWS)}")
    result = beauty_mod.render(
        session.cfg,
        session.current(),
        session.splat_result(),
        material_index(),
        **kwargs,
    )
    chosen = result.three_quarter if view == "three_quarter" else result.ground
    return result, preview.deliver_file(chosen, budget)


def stored() -> Iterator[tuple[str, dict[str, Any]]]:
    """Every terrain on disk with a session manifest, newest state as recorded."""
    if not config.TERRAIN_OUT_DIR.is_dir():
        return
    for path in sorted(config.TERRAIN_OUT_DIR.iterdir()):
        manifest = path / SESSION_NAME
        if not path.is_dir() or not manifest.is_file():
            continue
        yield path.name, json.loads(manifest.read_text(encoding="utf-8"))
