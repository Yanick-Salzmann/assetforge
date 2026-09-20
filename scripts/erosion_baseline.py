from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from terrain import channels, erosion, preview, synth
from terrain.config import MapConfig, TERRAIN_OUT_DIR
from terrain.device import resolve as resolve_device

VALIDATION_CHANNELS = ("flow", "deposition", "wear", "wetness")


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_vram_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1 << 20)


def dry_ground_stats(
    stack: channels.ChannelStack,
    fields: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """The same channels read only where the water mask says the ground is dry.

    Splat rules threshold against these, not against the whole-map numbers (af-l90.4).
    """
    dry = stack["water"] < 0.5
    return {
        "fraction": round(float(dry.float().mean()), 5),
        **{
            channel: {
                "p50": round(float(field[dry].quantile(0.5)), 4),
                "p90": round(float(field[dry].quantile(0.9)), 4),
                "p99": round(float(field[dry].quantile(0.99)), 4),
                "p999": round(float(field[dry].quantile(0.999)), 4),
                "max": round(float(field[dry].max()), 4),
            }
            for channel, field in fields.items()
        },
    }


def run(name: str, resolution: int, preset: str, shape: str, seed: int, device_name: str | None):
    device = resolve_device(device_name).torch_device()
    cfg = MapConfig(name=name, resolution=resolution, seed=seed)
    params = erosion.erosion_preset(preset)
    out = TERRAIN_OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: dict[str, float] = {}
    start = time.perf_counter()
    height = synth.heightfield(cfg, shape, device)
    synchronise(device)
    timings["synth_s"] = time.perf_counter() - start
    start = time.perf_counter()
    result = erosion.erode(cfg, height, params, device=device)
    synchronise(device)
    timings["erode_s"] = time.perf_counter() - start
    start = time.perf_counter()
    basin = channels.drainage(cfg, result.height)
    stack = channels.build(cfg, result, basin=basin)
    synchronise(device)
    timings["channels_s"] = time.perf_counter() - start
    water = channels.summarise_water(cfg, result, basin=basin)
    preview.render_hillshade(result.height, cfg, out / preview.HILLSHADE_NAME)
    fields = {name: stack[name] for name in VALIDATION_CHANNELS}
    preview.render_channel_sheet(fields, out / "preview_erosion.png", tile=512, columns=2)
    preview.render_channel_sheet(stack.as_dict(), out / preview.CHANNELS_NAME, tile=256, columns=4)
    for channel, field in fields.items():
        preview.render_channel(field, out / f"channel_{channel}.png", channel)
    report = {
        "name": name,
        "resolution": resolution,
        "device": str(device),
        "erosion_preset": preset,
        "shape": shape,
        "seed": seed,
        "iterations": params.iterations,
        "settle_iterations": params.settle_iterations,
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "peak_vram_mib": round(peak_vram_mib(device), 1),
        "erosion_stats": {key: round(value, 6) for key, value in result.stats().items()},
        "water": water.as_dict(),
        "channel_stats": {
            channel: {
                "mean": round(float(field.mean()), 4),
                "max": round(float(field.max()), 4),
                "p50": round(float(field.quantile(0.5)), 4),
                "p99": round(float(field.quantile(0.99)), 4),
                "coverage_above_0.5": round(float((field > 0.5).float().mean()), 5),
            }
            for channel, field in fields.items()
        },
        "dry_ground": dry_ground_stats(stack, fields),
    }
    (out / "erosion_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="erosion-baseline")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--shape", default="continental")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.name, args.resolution, args.preset, args.shape, args.seed, args.device)


if __name__ == "__main__":
    main()
