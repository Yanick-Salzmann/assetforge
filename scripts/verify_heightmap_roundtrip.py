from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from terrain import export
from terrain.config import HEIGHTMAP_MAX

NON_COLOR_CHUNK_KEYS = ("gamma", "srgb", "chromaticity", "icc_profile")


def ramp_field(resolution: int) -> torch.Tensor:
    """A full-range linear gradient: the worst case for visible banding, not a realistic map.

    float32, matching the dtype every height field actually carries through the pipeline
    (ChannelStack is float32 throughout) - comparing against a float64 source would blame
    the PNG for float32 rounding that already happened upstream of export.py.
    """
    axis = torch.linspace(0.0, 1.0, resolution, dtype=torch.float32)
    return axis.reshape(1, -1).expand(resolution, resolution).clone()


def check(resolution: int) -> dict:
    source = ramp_field(resolution)
    with tempfile.TemporaryDirectory() as scratch:
        path = export.write_height(source, Path(scratch) / "height.png")
        with Image.open(path) as image:
            mode = image.mode
            info = dict(image.info)
            decoded = np.array(image)

    reconstructed = torch.from_numpy(decoded.astype(np.float64)) / HEIGHTMAP_MAX
    error = (reconstructed - source).abs()
    max_error = float(error.max())
    quantisation_bound = 0.5 / HEIGHTMAP_MAX
    float32_slack = 2.0**-23

    row = decoded[0].astype(np.int64)
    steps = np.diff(row)
    negative_steps = int((steps < 0).sum())
    max_step = int(steps.max()) if steps.size else 0
    expected_step = HEIGHTMAP_MAX / (resolution - 1) if resolution > 1 else 0.0

    tagged_color = [key for key in NON_COLOR_CHUNK_KEYS if key in info]

    return {
        "resolution": resolution,
        "mode": mode,
        "bit_depth_ok": mode in ("I;16", "I"),
        "max_abs_error": max_error,
        "quantisation_bound": quantisation_bound,
        "within_quantisation_bound": max_error <= quantisation_bound + float32_slack,
        "unique_levels": int(np.unique(row).size),
        "negative_steps": negative_steps,
        "monotonic": negative_steps == 0,
        "max_step": max_step,
        "expected_step": expected_step,
        "no_unexpected_gap": max_step <= expected_step + 1,
        "png_info": info,
        "tagged_as_color": tagged_color,
        "non_color": not tagged_color,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    report = check(args.resolution)
    print(json.dumps(report, indent=2))

    passed = (
        report["bit_depth_ok"]
        and report["within_quantisation_bound"]
        and report["monotonic"]
        and report["no_unexpected_gap"]
        and report["non_color"]
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
