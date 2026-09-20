from __future__ import annotations

import platform
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from terrain import beauty, config, erosion, session, synth
from terrain.budget import DEFAULT_BUDGET, Preview
from terrain.device import resolve as resolve_device

mcp = FastMCP("terrain-forge")


def _image(view: Preview) -> Image:
    return Image(data=view.data, format="jpeg")


def _result(summary: dict[str, Any], view: Preview) -> list:
    return [summary | {"preview": view.as_dict()}, _image(view)]


@mcp.tool
def terrain_status() -> dict:
    """Report the terrain pipeline environment: compute device, paths, presets, defaults."""
    return {
        "server": "terrain-forge",
        "device": resolve_device().as_dict(),
        "platform": platform.platform(),
        "repo_root": str(config.REPO_ROOT),
        "workspace_dir": str(config.WORKSPACE_DIR),
        "output_dir": str(config.TERRAIN_OUT_DIR),
        "materials_dir": str(config.MATERIALS_DIR),
        "default_resolution": config.MapConfig(name="_").resolution,
        "max_resolution": config.MAX_RESOLUTION,
        "splat_layers_per_texture": config.SPLAT_LAYERS_PER_TEXTURE,
        "channels": list(config.CHANNEL_NAMES),
        "shape_presets": list(synth.shape_preset_names()),
        "erosion_presets": list(erosion.erosion_preset_names()),
        "erosion_params": sorted(session.EROSION_FIELDS),
        "water_params": sorted(session.WATER_FIELDS),
        "resident_sessions": list(session.resident()),
        "preview_max_tokens": DEFAULT_BUDGET.max_tokens,
    }


@mcp.tool
def list_terrains() -> list[dict]:
    """List every terrain already generated under out/terrain/."""
    if not config.TERRAIN_OUT_DIR.is_dir():
        return []
    manifests = dict(session.stored())
    entries = []
    for path in sorted(config.TERRAIN_OUT_DIR.iterdir()):
        if not path.is_dir():
            continue
        manifest = manifests.get(path.name, {})
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "stage": manifest.get("stage"),
                "config": manifest.get("config"),
                "resident": path.name in session.resident(),
                "has_manifest": (path / "terrain.json").is_file(),
                "files": sorted(p.name for p in path.iterdir() if p.is_file()),
            }
        )
    return entries


@mcp.tool
def new_terrain(
    name: str,
    seed: int = 0,
    resolution: int = 1024,
    world_size_m: float = 4096.0,
    height_range_m: float = 600.0,
    sea_level_m: float = 0.0,
    shape: str = synth.DEFAULT_SHAPE,
    overwrite: bool = False,
) -> list:
    """Synthesise a base heightfield under out/terrain/<name>/ and return its hillshade.

    shape is one of the presets listed by terrain_status. Nothing is eroded yet: this is the
    large-scale landform the erosion pass will carve.
    """
    current = session.create(
        name=name,
        seed=seed,
        resolution=resolution,
        world_size_m=world_size_m,
        height_range_m=height_range_m,
        sea_level_m=sea_level_m,
        shape=shape,
        overwrite=overwrite,
    )
    return _result(current.manifest(), session.hillshade(current))


@mcp.tool
def run_erosion(
    name: str,
    preset: str = erosion.DEFAULT_PRESET,
    params: dict | None = None,
    water: dict | None = None,
) -> list:
    """Run the pipe-model erosion pass over a terrain and return the carved hillshade.

    preset is one of the erosion presets from terrain_status; params overrides individual
    ErosionParams fields on top of it, water overrides WaterParams for the later channel pass.
    Re-running replaces the previous result and invalidates the channel stack.
    """
    current = session.open_session(name)
    erosion.erosion_preset(preset)
    current.erosion = preset
    current.erosion_overrides = session.overrides(params, session.EROSION_FIELDS, "erosion params")
    if water is not None:
        current.water_overrides = session.overrides(water, session.WATER_FIELDS, "water params")
    current.erode()
    summary = current.manifest()
    summary["params"] = current.erosion_params().__dict__
    return _result(summary, session.hillshade(current))


@mcp.tool
def build_channels(name: str) -> list:
    """Fill the shared channel stack from the eroded terrain and return the contact sheet.

    Erosion is replayed from the recorded seed and params if it is not already resident.
    """
    current = session.open_session(name)
    current.build_channels()
    summary = current.manifest()
    summary["channel_stats"] = session.channel_stats(current)
    summary["stack_bytes"] = current.channels().nbytes
    return _result(summary, session.channel_sheet(current))


@mcp.tool
def preview_hillshade(
    name: str,
    centre: list[float] | None = None,
    span: float = 0.25,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    contour_interval_m: float | None = None,
) -> list:
    """Render the hillshade-and-contour view of a terrain as it currently stands.

    Pass centre as [x, y] fractions with a span to crop: a quarter-map crop costs a fraction of
    the tokens a full map does and is the right way to inspect detail.
    """
    current = session.open_session(name)
    view = session.hillshade(
        current,
        centre,
        span,
        azimuth_deg=azimuth_deg,
        altitude_deg=altitude_deg,
        contour_interval_m=contour_interval_m,
    )
    summary = {
        "name": current.name,
        "stage": current.stage,
        "config": current.cfg.as_dict(),
        "azimuth_deg": azimuth_deg,
        "altitude_deg": altitude_deg,
    }
    return _result(summary, view)


@mcp.tool
def inspect_channel(
    name: str,
    channel: str,
    centre: list[float] | None = None,
    span: float = 0.25,
    rescale: bool = False,
) -> list:
    """Render one channel of a terrain as false colour, with its range.

    channel is any name from terrain_status, or "height". rescale stretches the shown range to
    the field's own extremes instead of [0, 1] — useful for channels that sit near zero.
    """
    current = session.open_session(name)
    view, stats = session.channel_view(current, channel, centre, span, rescale)
    summary = {
        "name": current.name,
        "channel": channel,
        "stage": current.stage,
        "rescaled": rescale,
        "range": stats,
    }
    return _result(summary, view)


@mcp.tool
def apply_rules(name: str, biome_file: str, sharpness: float | None = None) -> list:
    """Evaluate a biome's material rules over the channel stack and write its splat textures.

    biome_file is a name from biomes/ (e.g. "temperate") or a path to a biome TOML file.
    sharpness overrides the biome's own blend sharpness. Returns the false-colour composite
    and each layer's coverage table.
    """
    current = session.open_session(name)
    current.apply_biome(biome_file, sharpness)
    view, table = session.splat_view(current)
    summary = current.manifest()
    summary["coverage_table"] = table
    return _result(summary, view)


@mcp.tool
def preview_splat(
    name: str,
    centre: list[float] | None = None,
    span: float = 0.25,
) -> list:
    """Render the current splat's false-colour composite and coverage table.

    Replays the recorded biome from the channel stack if the server has restarted and the
    splat is not resident. Pass centre/span to crop, as in preview_hillshade.
    """
    current = session.open_session(name)
    view, table = session.splat_view(current, centre, span)
    summary = {
        "name": current.name,
        "stage": current.stage,
        "biome_file": current.biome_file,
        "coverage_table": table,
    }
    return _result(summary, view)


@mcp.tool
def preview_beauty(
    name: str,
    view: str = "three_quarter",
    centre: list[float] | None = None,
    patch_size_m: float = beauty.DEFAULT_PATCH_SIZE_M,
    samples: int = beauty.DEFAULT_SAMPLES,
) -> list:
    """Render a headless Blender beauty view of the terrain, real materials and all.

    view is "three_quarter" or "ground". centre is [x, y] fractions of the map, defaulting to
    its middle. Rendering is not cached: each call re-renders both views and returns the one
    asked for. Applies the biome from the last apply_rules call, replaying it if needed.
    """
    current = session.open_session(name)
    kwargs: dict[str, Any] = {"patch_size_m": patch_size_m, "samples": samples}
    if centre is not None:
        kwargs["centre"] = tuple(centre)
    result, rendered = session.beauty_view(current, view, **kwargs)
    summary = {
        "name": current.name,
        "view": view,
        "centre_world_m": list(result.centre_world_m),
        "device": result.device,
    }
    return _result(summary, rendered)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
