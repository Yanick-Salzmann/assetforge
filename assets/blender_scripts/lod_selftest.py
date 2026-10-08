"""Runs inside Blender's own interpreter via `blender --background --python`.

Exercises assets/lod.py's generate_lods on a small game-ready-style scene (one mesh, one
material, UVs already unwrapped) and reports triangle counts and baked-normal-map pixel
statistics as JSON, so the pytest suite gets automated coverage of bpy-only code it cannot
import directly. Invoked as:
    blender --background --factory-startup --python lod_selftest.py -- <repo_root> <result.json>
"""

import json
import statistics
import sys
import traceback
from pathlib import Path

import bmesh
import bpy


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, result_path = argv[separator + 1:separator + 3]
    return Path(repo_root), Path(result_path)


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def _detailed_box(name: str, material: "bpy.types.Material") -> "bpy.types.Object":
    """A box with enough subdivided faces that decimation and normal baking both have real
    work to do (a single-cube base would decimate to nothing and bake a uniform flat normal)."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    result = bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=6, use_grid_fill=True)
    verts = [g for g in result["geom"] if isinstance(g, bmesh.types.BMVert)]
    import random

    rng = random.Random(0)
    for v in verts:
        v.co += v.normal * rng.uniform(-0.03, 0.03)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


HINGE_VERTEX_COUNT = 8


def _thin_hinge(name: str, location: tuple) -> "bpy.types.Object":
    """A tiny disjoint box standing in for a door hinge or finial - the small, low-tri loose
    part a flat COLLAPSE ratio has historically punched holes through (or degenerated outright)
    once joined onto a much bigger mesh (af-6cu)."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.06)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def _thin_beam(name: str, location: tuple) -> "bpy.types.Object":
    """Long thin box standing in for a full-span timber beam or barge board - not small
    relative to the whole object's bbox diagonal (its own diagonal is close to 2, only a bit
    under the hero box's ~3.46), but built with too few source triangles (12, an unsubdivided
    box) to survive a flat COLLAPSE ratio without punching a hole (af-dds)."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(1.8, 0.03, 0.03), verts=bm.verts[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    return obj


def _mid_part_kit(material: "bpy.types.Material") -> "bpy.types.Object":
    """One large plate plus a row of once-subdivided boxes (48 tris each) - enough to clear the
    closed-shape floor at a 0.5 ratio but not at 0.15. Decimating LOD2 straight from LOD0
    protects every one of them at full resolution and lands LOD2 above LOD1 (af-k4g)."""
    mesh = bpy.data.meshes.new("kit_mesh")
    bm = bmesh.new()
    plate = bmesh.ops.create_grid(bm, x_segments=8, y_segments=8, size=10.0)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.2)
    for i in range(30):
        cube = bmesh.ops.create_cube(bm, size=1.0)
        cube_verts = cube["verts"]
        bmesh.ops.translate(bm, vec=(-9.0 + (i % 10) * 2.0, -4.0 + (i // 10) * 4.0, 1.0), verts=cube_verts)
        cube_edges = list({edge for vert in cube_verts for edge in vert.link_edges})
        bmesh.ops.subdivide_edges(bm, edges=cube_edges, cuts=1, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new("kit", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _trim_heavy_kit(material: "bpy.types.Material") -> "bpy.types.Object":
    """One large subdivided plate plus sixty small subdivided boxes that hold most of the
    triangles - the shape of a building whose quoins and trim outnumber its walls. A whole-mesh
    Decimate ratio pins the boxes and takes the entire cut out of the plate, deleting it."""
    mesh = bpy.data.meshes.new("trim_heavy_mesh")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=6, y_segments=6, size=10.0)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.2)
    for i in range(60):
        cube = bmesh.ops.create_cube(bm, size=0.3)
        cube_verts = cube["verts"]
        bmesh.ops.translate(bm, vec=(-9.0 + (i % 10) * 2.0, -9.0 + (i // 10) * 3.0, 0.6), verts=cube_verts)
        cube_edges = list({edge for vert in cube_verts for edge in vert.link_edges})
        bmesh.ops.subdivide_edges(bm, edges=cube_edges, cuts=1, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new("trim_heavy", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _largest_part_diagonal(obj: "bpy.types.Object", lod_module) -> float:
    parts = lod_module._connected_parts(obj.data)
    return max(lod_module._part_diagonal(obj.data, indices) for indices in parts)


def _small_part_sizes(obj: "bpy.types.Object", lod_module, count: int) -> list:
    """Vertex counts of the `count` smallest loose parts, expected to still be intact."""
    sizes = sorted(len(part) for part in lod_module._connected_parts(obj.data))
    return sizes[:count]


def _boundary_edge_count(obj: "bpy.types.Object") -> int:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    count = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    bm.free()
    return count


def _join_loose_parts(base: "bpy.types.Object", parts: list) -> None:
    for other in bpy.context.selected_objects:
        other.select_set(False)
    for part in parts:
        part.select_set(True)
    base.select_set(True)
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.join()


def _uv_overlap_triangle_count(obj: "bpy.types.Object") -> int:
    """Count triangles whose UV footprint overlaps another triangle they don't share a mesh
    vertex with - vertex-sharing pairs are legitimately UV-adjacent (a shared edge/corner), so
    only a bleed into an unrelated region of the atlas counts (af-3b4)."""
    from assets.uv_islands import shapes_overlap

    mesh = obj.data
    mesh.calc_loop_triangles()
    uv_layer = mesh.uv_layers.active.data
    triangles = [
        (tuple(sorted(tri.vertices)), [tuple(uv_layer[li].uv) for li in tri.loops])
        for tri in mesh.loop_triangles
    ]
    overlapping: set[int] = set()
    for i in range(len(triangles)):
        verts_i, uvs_i = triangles[i]
        for j in range(i + 1, len(triangles)):
            verts_j, uvs_j = triangles[j]
            if set(verts_i) & set(verts_j):
                continue
            if shapes_overlap(uvs_i, uvs_j):
                overlapping.add(i)
                overlapping.add(j)
    return len(overlapping)


def _dark_face_fraction(obj: "bpy.types.Object", image: "bpy.types.Image") -> float:
    """Fraction of obj's triangles whose UV centroid samples a near-black texel of image. LOD0's
    albedo here is a flat light grey, so any dark texel under a face means that face's UVs point
    at texels the bake never filled - the black wall patches of af-4ir.9."""
    width, height = image.size
    pixels = list(image.pixels)
    channels = image.channels
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv_layer = mesh.uv_layers.active.data
    dark = 0
    for tri in mesh.loop_triangles:
        u = sum(uv_layer[li].uv[0] for li in tri.loops) / 3.0
        v = sum(uv_layer[li].uv[1] for li in tri.loops) / 3.0
        x = min(max(int(u * width), 0), width - 1)
        y = min(max(int(v * height), 0), height - 1)
        offset = (y * width + x) * channels
        if max(pixels[offset:offset + 3]) < 0.2:
            dark += 1
    return dark / max(len(mesh.loop_triangles), 1)


def _image_stats(image: "bpy.types.Image") -> dict:
    pixels = list(image.pixels)
    channel_count = image.channels
    blue = pixels[2::channel_count]
    return {"min": min(blue), "max": max(blue), "stdev": statistics.pstdev(blue)}


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    _clear_scene()

    payload: dict = {"ok": False}
    try:
        from assets import game_ready, lod

        material = bpy.data.materials.new("scratch_atlas")
        material.use_nodes = True
        lod0 = _detailed_box("hero", material)
        hinges = [
            _thin_hinge(f"hinge_{i}", location)
            for i, location in enumerate(
                [(1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)]
            )
        ]
        beam = _thin_beam("beam", (0.0, 0.0, 0.0))
        for part in hinges + [beam]:
            part.data.materials.append(material)
        _join_loose_parts(lod0, hinges + [beam])
        game_ready.uv_unwrap_and_pack([lod0])

        lod_set = lod.generate_lods(lod0, ratios=(1.0, 0.5, 0.15), texture_size=512)
        boundary_edges = {obj.name: _boundary_edge_count(obj) for obj in lod_set.objects}
        hinge_sizes = {obj.name: _small_part_sizes(obj, lod, 5) for obj in lod_set.objects}
        uv_overlap_counts = {obj.name: _uv_overlap_triangle_count(obj) for obj in lod_set.objects}

        triangle_counts = lod_set.triangle_counts

        kit = _mid_part_kit(material)
        game_ready.uv_unwrap_and_pack([kit])
        kit_counts = lod.generate_lods(kit, ratios=(1.0, 0.5, 0.15), texture_size=64).triangle_counts
        trim_heavy = _trim_heavy_kit(material)
        game_ready.uv_unwrap_and_pack([trim_heavy])
        trim_set = lod.generate_lods(trim_heavy, ratios=(1.0, 0.5, 0.15), texture_size=64)
        trim_counts = trim_set.triangle_counts
        plate_diagonals = [_largest_part_diagonal(obj, lod) for obj in trim_set.objects]
        image_stats = {
            entry.normal: _image_stats(bpy.data.images[entry.normal])
            for entry in lod_set.textures
            if entry is not None
        }
        hero_diagonal = lod._bound_box_diagonal(lod_set.objects[0])
        deviations = {
            obj.name: lod._deviation(lod_set.objects[0].data, obj.data) / hero_diagonal
            for obj in lod_set.objects[1:]
        }
        dark_fractions = {
            obj.name: _dark_face_fraction(obj, bpy.data.images[entry.albedo])
            for obj, entry in zip(lod_set.objects, lod_set.textures)
            if entry is not None
        }

        ok = (
            triangle_counts[0] > triangle_counts[1] > triangle_counts[2] > 0
            and kit_counts[0] > kit_counts[1] > kit_counts[2] > 0
            and trim_counts[0] > trim_counts[1] > trim_counts[2] > 0
            and all(diagonal >= plate_diagonals[0] * 0.9 for diagonal in plate_diagonals)
            and len(lod_set.objects) == 3
            and lod_set.objects[0].data.materials[0].name == material.name
            and lod_set.objects[1].data.materials[0].name != material.name
            and lod_set.objects[2].data.materials[0].name != material.name
            and all(stats["stdev"] > 1e-4 for stats in image_stats.values())
            and all(count == 0 for count in boundary_edges.values())
            and all(
                size == HINGE_VERTEX_COUNT for sizes in hinge_sizes.values() for size in sizes
            )
            and all(count == 0 for count in uv_overlap_counts.values())
            and all(fraction == 0.0 for fraction in dark_fractions.values())
            and all(
                deviations[obj.name] <= lod.LOD_DEVIATION_RATIOS[level] + 1e-6
                for level, obj in enumerate(lod_set.objects[1:], start=1)
            )
        )
        payload = {
            "ok": ok,
            "report": lod_set.as_dict(),
            "kit_triangle_counts": kit_counts,
            "trim_heavy_triangle_counts": trim_counts,
            "trim_heavy_plate_diagonals": plate_diagonals,
            "image_stats": image_stats,
            "boundary_edges": boundary_edges,
            "hinge_sizes": hinge_sizes,
            "uv_overlap_counts": uv_overlap_counts,
            "dark_face_fractions": dark_fractions,
            "deviation_ratios": deviations,
        }
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
