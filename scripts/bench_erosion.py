from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from terrain import channels, erosion, synth
from terrain.config import MapConfig
from terrain.device import resolve as resolve_device

WARMUP_ITERATIONS = 10


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_vram_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1 << 20)


def measure(cfg: MapConfig, params: erosion.ErosionParams, device: torch.device) -> dict:
    height = synth.heightfield(cfg, "continental", device)
    warmup = erosion.ErosionParams(iterations=WARMUP_ITERATIONS, settle_iterations=0)
    erosion.erode(cfg, height, warmup, device=device)
    synchronise(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = erosion.erode(cfg, height, params, device=device)
    synchronise(device)
    erode_s = time.perf_counter() - start
    erode_peak = peak_vram_mib(device)
    start = time.perf_counter()
    stack = channels.build(cfg, result)
    synchronise(device)
    channels_s = time.perf_counter() - start
    steps = params.iterations + params.settle_iterations
    return {
        "device": str(device),
        "resolution": cfg.resolution,
        "steps": steps,
        "erode_s": round(erode_s, 3),
        "ms_per_step": round(1000.0 * erode_s / steps, 3),
        "channels_s": round(channels_s, 3),
        "erode_peak_vram_mib": round(erode_peak, 1),
        "total_peak_vram_mib": round(peak_vram_mib(device), 1),
        "stack_mib": round(stack.nbytes / (1 << 20), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--preset", default="default")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    params = erosion.erosion_preset(args.preset)
    specs = args.devices or [resolve_device().torch_device().type]
    runs = []
    for spec in specs:
        device = resolve_device(spec).torch_device()
        for resolution in args.resolutions:
            cfg = MapConfig(name="bench", resolution=resolution, seed=args.seed)
            runs.append(measure(cfg, params, device))
            print(json.dumps(runs[-1]))
    report = {
        "preset": args.preset,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runs": runs,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
