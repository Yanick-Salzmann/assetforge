"""af-zch.2 spike scorer: rates every layout assets/blender_scripts/xatlas_spike.py wrote with the
same measures validate_asset applies to an exported atlas.

    uv run python -m assets.xatlas_spike_score <spike_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from assets import atlas_raster, validate

ATLAS_PX = 2048
METHODS = ("ours", "xatlas", "xatlas_visible")


def score(positions: np.ndarray, uvs: np.ndarray) -> dict:
    indices = np.arange(len(positions)).reshape(-1, 3)
    islands = validate._uv_islands(positions, indices, uvs)
    live = islands >= 0
    triangles = uvs[indices][live] * ATLAS_PX
    labels = atlas_raster.rasterize(triangles, islands[live], ATLAS_PX, ATLAS_PX)
    _, _, closer_than_8 = atlas_raster.grow_sources(labels, 4)
    density = validate._check_texel_density("layout", positions, indices, uvs)[0]
    return {
        "islands": int(islands.max()) + 1,
        "coverage": round(float((labels >= 0).mean()), 3),
        "pairs_under_8px": len(closer_than_8),
        "outside_unit_square": int(((uvs < 0.0) | (uvs > 1.0)).any(axis=1).sum()),
        "density": density.detail,
    }


def main(spike_dir: Path) -> None:
    rows = {}
    for summary_path in sorted(spike_dir.glob("*_summary.json")):
        summary = json.loads(summary_path.read_text())
        name = summary["name"]
        rows[name] = {"tris": summary["tris"], "hidden_faces": summary["hidden_faces"]}
        for method in METHODS:
            data = np.load(spike_dir / f"{name}_{method}.npz")
            result = score(data["positions"], data["uvs"])
            result.update(json.loads(str(data["meta"])))
            rows[name][method] = result
    (spike_dir / "scores.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
