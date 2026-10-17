"""af-zch.2 spike: our axis/strip unwrap + shelf packer against xatlas on one asset's real parts.

Runs inside Blender's own interpreter:
    blender --background <blend> --factory-startup --python xatlas_spike.py -- <repo_root> <xatlas_site> <name> <out_dir>

xatlas_site is a directory holding the xatlas wheel built for Blender's Python
(`python.exe -m pip install --only-binary=:all: --target <dir> xatlas`). Every static and rigid
mesh part named `<name>_*` (skinned cloth left out) is evaluated in world space and triangulated,
then laid out three ways on one 2048 atlas: `ours` (game_ready.uv_unwrap_and_pack, hidden faces
collapsed), `xatlas` (every face) and `xatlas_visible` (the faces ours keeps visible). Each layout
goes to `<out_dir>/<name>_<method>.npz` (corner positions, uvs, layout seconds) for
`uv run python -m assets.xatlas_spike_score <out_dir>` to score.
"""

import json
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np

ATLAS_PX = 2048
DENSITY_SEARCH_STEPS = 10


def _read_args() -> tuple[Path, Path, str, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, site, name, out_dir = argv[separator + 1:separator + 5]
    return Path(repo_root), Path(site), name, Path(out_dir)


def _world_copies(name: str, game_ready) -> list:
    prefix = f"{name}_"
    parts = [obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name.startswith(prefix) and not game_ready.is_skinned_part(obj)]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    copies = []
    for part in parts:
        mesh = bpy.data.meshes.new_from_object(part.evaluated_get(depsgraph))
        mesh.transform(part.matrix_world)
        copy = bpy.data.objects.new(f"spike_{part.name}", mesh)
        bpy.context.scene.collection.objects.link(copy)
        game_ready.triangulate(copy)
        copies.append(copy)
    return copies


def _corners(obj) -> tuple[np.ndarray, np.ndarray]:
    mesh = obj.data
    uv_layer = mesh.uv_layers.active.data
    positions = []
    uvs = []
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            positions.append(tuple(mesh.vertices[mesh.loops[loop_index].vertex_index].co))
            uvs.append(tuple(uv_layer[loop_index].uv))
    return np.array(positions, dtype=np.float64).reshape(-1, 3), np.array(uvs, dtype=np.float64).reshape(-1, 2)


def _ours(copies, game_ready) -> dict:
    started = time.perf_counter()
    layout = game_ready.uv_unwrap_and_pack(copies, ATLAS_PX)
    seconds = time.perf_counter() - started
    pairs = [_corners(obj) for obj in copies]
    return {
        "positions": np.concatenate([p for p, _ in pairs]),
        "uvs": np.concatenate([u for _, u in pairs]),
        "seconds": seconds,
        "islands": len(layout.packing.footprints),
        "density_px_per_m": layout.packing.density,
    }


def _hidden(copies, game_ready) -> list[set[int]]:
    meshes = []
    for obj in copies:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        meshes.append((obj, bm))
    try:
        return game_ready._hidden_in_world(meshes)
    finally:
        for _, bm in meshes:
            bm.free()


def _generate(xatlas, inputs, gutter: int, texels_per_unit: float):
    atlas = xatlas.Atlas()
    for positions, faces in inputs:
        atlas.add_mesh(positions, faces)
    pack = xatlas.PackOptions()
    pack.resolution = ATLAS_PX
    pack.padding = gutter // 2
    pack.bilinear = True
    pack.blockAlign = True
    pack.texels_per_unit = texels_per_unit
    atlas.generate(pack_options=pack)
    return atlas


def _fits(atlas) -> bool:
    return atlas.atlas_count == 1 and atlas.width <= ATLAS_PX and atlas.height <= ATLAS_PX


def _xatlas(copies, xatlas, gutter: int, skip: list[set[int]]) -> dict:
    inputs = []
    for obj, excluded in zip(copies, skip):
        mesh = obj.data
        positions = np.array([tuple(v.co) for v in mesh.vertices], dtype=np.float32).reshape(-1, 3)
        faces = [tuple(p.vertices) for p in mesh.polygons if p.index not in excluded]
        if faces:
            inputs.append((positions, np.array(faces, dtype=np.uint32).reshape(-1, 3)))
    started = time.perf_counter()
    atlas = _generate(xatlas, inputs, gutter, 0.0)
    seconds = time.perf_counter() - started
    low = 0.0
    high = atlas.texels_per_unit * 4.0
    fitted = None
    for _ in range(DENSITY_SEARCH_STEPS):
        middle = 0.5 * (low + high)
        trial = _generate(xatlas, inputs, gutter, middle)
        if _fits(trial):
            low = middle
            fitted = trial
        else:
            high = middle
    search_seconds = time.perf_counter() - started
    atlas = fitted
    corner_positions = []
    corner_uvs = []
    for index, (positions, _) in enumerate(inputs):
        vmapping, indices, uvs = atlas[index]
        corner_positions.append(positions[vmapping][indices].reshape(-1, 3).astype(np.float64))
        corner_uvs.append(uvs[indices].reshape(-1, 2).astype(np.float64) * np.array([atlas.width, atlas.height]) / ATLAS_PX)
    return {
        "positions": np.concatenate(corner_positions),
        "uvs": np.concatenate(corner_uvs),
        "seconds": seconds,
        "search_seconds": search_seconds,
        "islands": atlas.chart_count,
        "atlas_count": atlas.atlas_count,
        "atlas_px": [atlas.width, atlas.height],
        "density_px_per_m": atlas.texels_per_unit,
    }


def _save(out_dir: Path, name: str, method: str, result: dict) -> dict:
    arrays = {key: result.pop(key) for key in ("positions", "uvs")}
    np.savez_compressed(out_dir / f"{name}_{method}.npz", **arrays, meta=json.dumps(result))
    return result


def main() -> None:
    repo_root, site, name, out_dir = _read_args()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(site))
    import xatlas

    from assets import game_ready

    out_dir.mkdir(parents=True, exist_ok=True)
    copies = _world_copies(name, game_ready)
    gutter = game_ready.gutter_px(ATLAS_PX)
    summary = {"name": name, "parts": len(copies), "tris": sum(len(obj.data.polygons) for obj in copies)}
    hidden = _hidden(copies, game_ready)
    summary["hidden_faces"] = sum(len(faces) for faces in hidden)
    summary["xatlas"] = _save(out_dir, name, "xatlas", _xatlas(copies, xatlas, gutter, [set() for _ in copies]))
    summary["xatlas_visible"] = _save(out_dir, name, "xatlas_visible", _xatlas(copies, xatlas, gutter, hidden))
    summary["ours"] = _save(out_dir, name, "ours", _ours(copies, game_ready))
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


main()
