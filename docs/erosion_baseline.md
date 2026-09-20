# Erosion baseline

Phase 1 gate evidence for af-g22.3 (visual validation) and af-y2c (benchmark and CPU
fallback). Regenerate with:

```bash
PYTHONPATH=. uv run python scripts/erosion_baseline.py --resolution 1024
PYTHONPATH=. uv run python scripts/bench_erosion.py --resolutions 1024 2048 --devices cuda cpu
```

Machine: Windows 11, RTX 4070 12 GB, torch 2.14.0+cu130, CUDA 13.0.

## Benchmark

`default` preset, 400 iterations plus 40 settle passes, `continental` shape, seed 7.
`channels_s` is the full `channels.build` — the depression fill, the flat routing potential and
the MFD accumulation dominate it, and all three are O(resolution) sequential passes, which is
why they outgrow the sim at 2048².

| Device | Res | Steps | Erode | ms/step | Channels | Peak VRAM (erode) | Peak VRAM (total) | Stack |
|---|---|---|---|---|---|---|---|---|
| cuda | 512 | 440 | 0.70 s | 1.59 | 0.48 s | 45 MiB | 63 MiB | 16 MiB |
| cuda | 1024 | 440 | 0.69 s | 1.57 | 0.72 s | 180 MiB | 252 MiB | 64 MiB |
| cuda | 2048 | 440 | 4.34 s | 9.87 | 7.96 s | 720 MiB | 1008 MiB | 256 MiB |
| cuda | 2048 | 2040 | 20.08 s | 9.84 | 8.18 s | 720 MiB | 1008 MiB | 256 MiB |
| cpu | 512 | 440 | 1.48 s | 3.36 | 1.10 s | — | — | 16 MiB |
| cpu | 1024 | 440 | 7.51 s | 17.07 | 7.32 s | — | — | 64 MiB |
| cpu | 2048 | 440 | 56.37 s | 128.11 | 96.22 s | — | — | 256 MiB |

The `cuda` rows predate the af-l90.5 drainage rewrite and are stale; af-av0 tracks re-running
them. Every CUDA measurement attempted during that rewrite shared the GPU with a game — 74%
utilisation, 9.5 of 12 GB resident, 512² erode reading 3.19 s against the 0.70 s above — so
none of it is reportable.

The `cpu` rows are all post-rewrite and measured on an idle CPU, and they are the honest
comparison. The sim is untouched and reads the same: 56.37 s at 2048² against 56.28 s before.
`channels.build` went from 1.06 s to 1.10 s at 512², from 10.61 s to 7.32 s at 1024² and from
108.55 s to 96.22 s at 2048², while now also carrying the flat routing potential it did not
have before. It is still 1.71x the sim's own wall time at 2048², down from 1.93x — af-g22.6
wants it under 1.0x and is not there.

Two thousand iterations at 2048² cost 20 s on the 4070, well inside the target. Peak allocation
is 1.0 GiB including the channel stack, against a 12 GB ceiling — 4096² projects to roughly
4 GiB and still fits.

CPU is a correctness fallback, not a production path. It needs no separate numpy
implementation: `terrain.device.resolve()` already falls back to `cpu` and the whole pipeline
runs unchanged on it. `test_accelerator_and_cpu_agree_on_every_erosion_field` pins CUDA and CPU
to within 1e-3 relative on every field of `ErosionResult` at 64².

### What the drainage pass costs

Measured at 2048² inside one process, so the three parts are comparable to each other even
though the absolute numbers are not comparable to the table above. The pairwise folds in
`_lowest`, `_highest` and `_collect` replaced four-plane `torch.stack` reductions and are the
only micro-optimisation that wins on both devices: `max_pool2d` is 1.4x faster than the fold on
CUDA but 22x slower on CPU, and a `conv2d` gather is 1.6x faster on CUDA but 2.3x slower on CPU.

- `fill_depressions` — solved coarse to fine. Each pyramid level starts from the level above,
  max-pooled so the coarse answer is an upper bound on the fine one, which makes the descent to
  the fixed point exact rather than approximate. Max-pooling closes narrow valleys, so the
  coarse answer is a weak bound exactly where the long chains are: the finest level still needs
  ~1360 of the ~1560 fine-equivalent passes. The pyramid is worth about 10%; dropping the
  Planchon-Darboux epsilon is worth more.
- `flat_gradient` — two geodesic sweeps over the flats, and the second-cheapest part.
- `flow_accumulation` — the most expensive part. Mass decays roughly linearly and the last
  parcel leaves near pass 2600 of the 4096 budget, so there is no early exit to win here.

## Visual validation

`out/terrain/erosion-baseline/` at 1024², default preset, seed 7. Kept as the comparison
baseline; `preview_erosion.png` is the 2x2 sheet of the four gated channels. Whole-map and
dry-ground (`water < 0.5`, 83.6% of the map) quantiles are both in `erosion_baseline.json`;
splat rules threshold against the dry-ground numbers.

- **flow** — connected dendritic networks, tributaries feeding trunks, no speckle. Comes from
  `channels.drainage` (fill plus flat routing plus MFD accumulation), not from the sim's
  `flow_volume_m3` (af-l90.3). Dry ground: p50 0.384, p90 0.564, p99 0.714, p99.9 0.786,
  max 0.819.
- **deposition** — concentrated in valley floors and basins, dark on steep faces.
  Dry ground p50 0.079, p90 0.596, p99 0.917.
- **wear** — bright on steep faces and channel walls, dark along the thalwegs and on flat
  ridge tops. Dry ground p50 0.390, p90 0.797, p99 0.978.
- **wetness** — follows the drainage, still blown out over the filled lake surfaces.
  Dry ground p50 0.381, p99 0.745.

Gate passed. Known artifacts, filed rather than fixed here:

- `flow` and `wetness` both sit near 0.4 at the median now that `log_normalise` is a true log
  stretch (af-g22.7). That is the drainage area an MFD hillslope cell genuinely carries, but it
  leaves the preview bright and the spread between hillslope and trunk narrower than it looks
  on the old previews. A gamma belongs in `terrain.preview`, not in the channel.
- Drainage crossing a filled lake follows the routing potential in `flat_gradient`, which is a
  four-neighbour cell count out from the spill minus a count away from the walls. The straight
  beams and right-angle staircases the Planchon-Darboux epsilon ladder produced are gone
  (af-l90.5); faint axis-aligned banding remains on the widest lake surfaces, and every cell
  carrying it is submerged. A Euclidean chamfer count was tried and is worse: round contours
  sharpen the descent into single straight rays along the distance field's own medial axis.
