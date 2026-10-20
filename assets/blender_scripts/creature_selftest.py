"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds every species in assets/creature_specs and reports closed/manifold geometry, a single
shell, the lowest vertex, the base-centre origin, the triangle count against the spec target and
the painted regions as JSON. Invoked as:
    blender --background --factory-startup --python creature_selftest.py -- <repo_root> <result.json>
"""

import json
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


def _shells(bm: "bmesh.types.BMesh") -> int:
    unvisited = set(bm.verts)
    count = 0
    while unvisited:
        count += 1
        stack = [unvisited.pop()]
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other in unvisited:
                    unvisited.remove(other)
                    stack.append(other)
    return count


def _report(obj: "bpy.types.Object", species) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    xs = [vertex.co.x for vertex in bm.verts]
    ys = [vertex.co.y for vertex in bm.verts]
    report = {
        "non_manifold_edges": sum(1 for edge in bm.edges if not edge.is_manifold),
        "loose_verts": sum(1 for vertex in bm.verts if not vertex.link_faces),
        "shells": _shells(bm),
        "lowest_z": min(vertex.co.z for vertex in bm.verts),
        "centre_offset": max(abs(min(xs) + max(xs)), abs(min(ys) + max(ys))) * 0.5,
        "tris": sum(len(face.verts) - 2 for face in bm.faces),
        "target_tris": species.target_tris,
        "materials": sorted({face.material_index for face in bm.faces}),
        "regions": species.regions(),
    }
    bm.free()
    return report


def _passes(report: dict) -> bool:
    return (
        report["non_manifold_edges"] == 0
        and report["loose_verts"] == 0
        and report["shells"] == 1
        and abs(report["lowest_z"]) < 1e-4
        and report["centre_offset"] < 1e-3
        and 0.9 * report["target_tris"] <= report["tris"] <= report["target_tris"]
        and report["materials"] == list(range(len(report["regions"])))
    )


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    payload: dict = {"ok": False}
    try:
        from assets import creature, creature_spec

        checks = {
            name: _report(creature.creature(name, species), species)
            for name, species in creature_spec.catalogue().items()
        }
        payload = {"ok": all(_passes(report) for report in checks.values()), "checks": checks}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
