from __future__ import annotations

import platform
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from terrain import config
from terrain.budget import Preview

mcp = FastMCP("asset-forge")


def _image(view: Preview) -> Image:
    return Image(data=view.data, format="jpeg")


def _result(summary: dict[str, Any], view: Preview) -> list:
    return [summary | {"preview": view.as_dict()}, _image(view)]


@mcp.tool
def asset_status() -> dict:
    """Report the asset pipeline environment: Blender path, output and kit dirs."""
    from assets import blender

    return {
        "server": "asset-forge",
        "platform": platform.platform(),
        "repo_root": str(config.REPO_ROOT),
        "workspace_dir": str(config.WORKSPACE_DIR),
        "output_dir": str(config.ASSET_OUT_DIR),
        "kits_dir": str(config.KITS_DIR),
        "blender_executable": blender.find_blender(),
    }


@mcp.tool
def record_recipe(name: str, code: str) -> dict:
    """Append one bpy snippet the session just executed to out/assets/<name>/recipe.py."""
    from assets import recipe

    session = recipe.open_session(name).record(code)
    return {"name": name, "path": str(session.out_dir() / recipe.RECIPE_NAME), "snippet_count": len(session.snippets)}


@mcp.tool
def validate_asset(name: str, kind: str) -> dict:
    """Validate out/assets/<name>/<name>.glb: tri budget, watertight/manifold, normals, UVs, origin, transforms."""
    from assets import validate

    return validate.validate_asset(name, kind).as_dict()


@mcp.tool
def record_patch(name: str) -> dict:
    """Count one patch iteration against the 4-round cap; refuses once the cap is spent."""
    from assets import gate

    return gate.open_gate(name).record_patch().as_dict()


@mcp.tool
def approve_asset(name: str, note: str = "") -> dict:
    """Record human approval for an asset, unblocking export."""
    from assets import gate

    return gate.open_gate(name).approve(note).as_dict()


@mcp.tool
def asset_gate_status(name: str) -> dict:
    """Report patch iterations used and approval state for an asset."""
    from assets import gate

    return gate.open_gate(name).as_dict()


@mcp.tool
def build_demo_asset(name: str = "hardsurface-demo") -> list:
    """Headless capability smoke test: assemble a small shed from every assets/hardsurface.py
    helper (bevel, boolean cuts, array, mirror, solidify, greeble), run it through
    assets/game_ready.py, export the glb and render its contact sheet - independent of any live
    MCP session. Use this to see what the hard-surface pipeline can currently produce.
    """
    from assets import demo

    built, rendered = demo.build_and_render(name)
    summary = {
        "name": name,
        "glb_path": str(built.glb_path),
        "triangle_count": built.triangle_count,
        "object_names": built.object_names,
        "bbox_min": list(rendered.bbox_min),
        "bbox_max": list(rendered.bbox_max),
        "device": rendered.device,
        "sheet_path": str(rendered.sheet),
    }
    from terrain.budget import DEFAULT_BUDGET, deliver_file

    return _result(summary, deliver_file(rendered.sheet, DEFAULT_BUDGET))


@mcp.tool
def render_contact_sheet(name: str) -> list:
    """Render out/assets/<name>/<name>.glb as one contact sheet: 6 orthographic views, a 3/4
    beauty shot and a worm's-eye shot, flat even lighting, a 1.8 m human scale reference beside
    the asset. This is how the agent sees the asset - it must expose missing back faces, flipped
    normals and one-angle-only detail.
    """
    from assets import contact_sheet

    result, rendered = contact_sheet.contact_sheet_preview(name)
    summary = {
        "name": name,
        "bbox_min": list(result.bbox_min),
        "bbox_max": list(result.bbox_max),
        "device": result.device,
        "sheet_path": str(result.sheet),
        "views": {view: str(path) for view, path in result.views.items()},
    }
    return _result(summary, rendered)


@mcp.tool
def search_cc0_materials(query: str = "") -> list[dict]:
    """Search PolyHaven's public texture catalog (cached on disk) for CC0 tiling materials by
    keyword, e.g. "weathered wood plank" or "roofing shingle". Returns slug/name/categories
    candidates to pass to pull_cc0_material.
    """
    from library import polyhaven

    return polyhaven.search_textures(query)


@mcp.tool
def pull_cc0_material(name: str, slug: str, tiling_m: float | None = None) -> dict:
    """Register a PolyHaven slug (from search_cc0_materials) as asset material `name`: download
    and lock it under library/asset_materials/, update library/asset_materials.toml and
    .lock.json, and regenerate library/asset_materials.json so
    assets.materials.get_material(name) can use it in this same session.
    """
    from library import materials

    return materials.pull_asset_material(name, slug, tiling_m)


@mcp.tool
def list_assets() -> list[dict]:
    """List every asset already exported under out/assets/."""
    if not config.ASSET_OUT_DIR.is_dir():
        return []
    entries = []
    for path in sorted(config.ASSET_OUT_DIR.iterdir()):
        if not path.is_dir():
            continue
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "has_manifest": (path / "asset.json").is_file(),
                "has_recipe": (path / "recipe.py").is_file(),
                "files": sorted(p.name for p in path.iterdir() if p.is_file()),
            }
        )
    return entries


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
