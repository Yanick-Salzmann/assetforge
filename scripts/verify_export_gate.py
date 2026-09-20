from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import torch
from PIL import Image

from terrain import export, manifest, scatter
from terrain import session as session_mod
from terrain.config import HEIGHTMAP_MAX

DEFAULT_NAME = "gate-export-contract"
STEEP_SLOPE = 0.6
WATER_THRESHOLD = 0.5
AVOIDANCE_CEILING = 0.15


def _heightmap_check(out_dir, height_source: torch.Tensor) -> dict:
    with Image.open(out_dir / manifest.HEIGHTMAP_NAME) as image:
        mode = image.mode
        decoded = np.array(image)
    reconstructed = torch.from_numpy(decoded.astype(np.float64)) / HEIGHTMAP_MAX
    max_error = float((reconstructed - height_source.to(torch.float64)).abs().max())
    bound = 0.5 / HEIGHTMAP_MAX + 2.0**-23
    return {
        "mode": mode,
        "max_abs_error": max_error,
        "bound": bound,
        "ok": mode in ("I;16", "I") and max_error <= bound,
    }


def _scatter_avoidance_check(out_dir, stack) -> dict:
    water = stack["water"].detach().to("cpu")
    slope = stack["slope"].detach().to("cpu")
    hazard = ((water >= WATER_THRESHOLD) | (slope >= STEEP_SLOPE)).numpy()
    per_kind = {}
    all_ok = True
    for kind, filename in scatter.MASK_NAMES.items():
        with Image.open(out_dir / filename) as image:
            mask = np.array(image).astype(np.float64) / 255.0
        hazard_mean = float(mask[hazard].mean()) if hazard.any() else 0.0
        ok = hazard_mean <= AVOIDANCE_CEILING
        all_ok = all_ok and ok
        per_kind[kind] = {"hazard_mean_density": hazard_mean, "ok": ok}
    return {"per_kind": per_kind, "hazard_pixels": int(hazard.sum()), "ok": all_ok}


def run(resolution: int, seed: int, keep: bool) -> dict:
    session = session_mod.create(name=DEFAULT_NAME, seed=seed, resolution=resolution, overwrite=True)
    session.erode()
    session.build_channels()
    session.apply_biome("temperate")
    stack = session.channels()
    result = export.write(session.cfg, stack, session.water, session.splat_result())

    report: dict = {"resolution": resolution, "seed": seed, "out_dir": str(result.out_dir)}
    report["heightmap"] = _heightmap_check(result.out_dir, stack["height"].detach().to("cpu", dtype=torch.float32))
    report["scatter_avoidance"] = _scatter_avoidance_check(result.out_dir, stack)

    try:
        payload = manifest.load(result.out_dir / manifest.MANIFEST_NAME)
        report["schema"] = {"ok": True, "error": None}
    except manifest.ManifestError as error:
        payload = result.manifest
        report["schema"] = {"ok": False, "error": str(error)}

    try:
        manifest.verify(payload, result.out_dir)
        report["paths_verified"] = {"ok": True, "error": None}
    except manifest.ManifestError as error:
        report["paths_verified"] = {"ok": False, "error": str(error)}

    if not keep:
        session_mod.forget(DEFAULT_NAME)
        shutil.rmtree(session.out_dir(), ignore_errors=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep", action="store_true", help="keep the generated terrain under out/terrain/")
    args = parser.parse_args()

    report = run(args.resolution, args.seed, args.keep)
    print(json.dumps(report, indent=2))

    passed = (
        report["heightmap"]["ok"]
        and report["scatter_avoidance"]["ok"]
        and report["schema"]["ok"]
        and report["paths_verified"]["ok"]
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
