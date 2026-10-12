"""Runs inside Blender's own interpreter via `blender --background --python`.

Builds every assets/organic.py builder and reports closed/manifold/loose-vertex geometry, the
lowest vertex and the triangle count as JSON. Invoked as:
    blender --background --factory-startup --python organic_selftest.py -- <repo_root> <result.json>
"""

import json
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

TWO_MATERIAL_BUILDS = ("stump", "stump_jagged", "mushroom", "mushroom_cluster")


def _read_args() -> tuple[Path, Path]:
    argv = sys.argv
    separator = argv.index("--")
    repo_root, result_path = argv[separator + 1:separator + 3]
    return Path(repo_root), Path(result_path)


def _mesh_report(obj: "bpy.types.Object") -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    report = {
        "non_manifold_edges": sum(1 for edge in bm.edges if not edge.is_manifold),
        "loose_verts": sum(1 for vertex in bm.verts if not vertex.link_faces),
        "degenerate_faces": sum(1 for face in bm.faces if face.calc_area() < 1e-10),
        "lowest_z": min(vertex.co.z for vertex in bm.verts),
        "tris": sum(len(face.verts) - 2 for face in bm.faces),
        "materials": sorted({face.material_index for face in bm.faces}),
    }
    bm.free()
    return report


def _builds(organic) -> dict:
    return {
        "stump": organic.stump("stump", seed=3),
        "stump_jagged": organic.stump("stump_jagged", seed=5, jag=0.12),
        "root_cluster": organic.root_cluster("root_cluster", seed=7),
        "mushroom": organic.mushroom("mushroom", seed=9),
        "mushroom_cluster": organic.mushroom_cluster("mushroom_cluster", seed=11),
        "branching_coral": organic.branching_coral("branching_coral", seed=13),
        "dome_coral": organic.dome_coral("dome_coral", seed=15),
    }


def main() -> None:
    repo_root, result_path = _read_args()
    sys.path.insert(0, str(repo_root))
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    payload: dict = {"ok": False}
    try:
        from assets import organic

        checks = {name: _mesh_report(obj) for name, obj in _builds(organic).items()}
        ok = all(
            report["non_manifold_edges"] == 0
            and report["loose_verts"] == 0
            and report["degenerate_faces"] == 0
            and abs(report["lowest_z"]) < 1e-4
            and 100 <= report["tris"] <= 2000
            for report in checks.values()
        ) and all(checks[name]["materials"] == [0, 1] for name in TWO_MATERIAL_BUILDS)
        payload = {"ok": ok, "checks": checks}
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}

    result_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
