---
name: terrain-tune
description: Load an existing terrain, adjust its biome rule file or erosion params, re-render previews, diff the coverage table against the pre-edit run, and re-export. Trigger on "/terrain-tune", "tune this terrain", "fix the splat on X", "rebalance the biome for X", or any request to iterate on a terrain that already exists under out/terrain/.
---

# terrain-tune

For iterating on a terrain [[terrain-new]] already produced, not for building one from scratch.
The point is small, targeted edits with a before/after comparison at each change - not
regenerating the whole pipeline because one material looks wrong.

## 1. Load

```
list_terrains()
```

Confirms the terrain exists and shows its `stage` (`synth` / `eroded` / `channels` / `splat`).
Any tool call with that `name` transparently replays earlier stages from the recorded seed and
params if the server has restarted and the terrain isn't resident - determinism guarantees the
replay is exact, so there is no need to manually rebuild anything before tuning.

Capture a baseline before changing anything:

```
preview_splat(name)
```

Keep its `coverage_table` - this is what every later change gets diffed against.

## 2. Decide what layer of the stack the fix belongs to

Work at the cheapest layer that can fix the problem:

- **A material dominates, boundaries look wrong, or a layer is missing where it should apply**
  -> edit the biome TOML in `biomes/` (weight expressions, blend `sharpness`). Cheapest: only
  step 3a re-runs.
- **The drainage network, slope distribution, or wetness/moisture fields themselves are wrong**
  (e.g. splat rules keyed on `flow` or `moisture` never trigger because those channels are flat
  or misplaced) -> adjust `erosion` preset/params or `water` params and re-run from erosion.
  Expensive: invalidates channels and splat, step 3b re-runs everything downstream.
- **The landform itself is wrong** (shape, overall relief) -> this is `/terrain-new` territory;
  re-synthesising is a new terrain, not a tune. Say so rather than trying to erosion your way
  out of a bad `shape`/`seed` choice.

## 3a. Tune the splat only

Edit the biome file directly (it's just a TOML file under `biomes/`, or a path you were given),
then:

```
apply_rules(name, biome_file, sharpness=None)
```

Re-running replaces the previous splat outright and returns a fresh composite and coverage
table. Diff the new `coverage_table` against the baseline: the fix should move the specific
layer that was wrong without collapsing another layer to near-zero or pushing a different
layer past ~45% coverage (Phase 2 gate). Use `preview_splat(name, centre, span)` to crop into
the exact region that motivated the change and confirm visually, not just numerically -
coverage percentages can look fine while the boundary itself still reads as a hard tile edge.

## 3b. Tune erosion/water

```
run_erosion(name, preset, params={...overrides...}, water={...overrides...})
```

Pass only the fields you're changing as overrides - `params`/`water` are deltas on top of the
named preset, not a full replacement. Re-check the hillshade (Phase 1 gate: coherent branching
drainage, not noise) before paying for anything downstream:

```
build_channels(name)
```

Then re-inspect whichever channel motivated the change with
`inspect_channel(name, channel, rescale=True)` before moving back to step 3a to re-apply the
biome - erosion/water changes invalidate the splat even if the biome file itself didn't change.

## 4. Re-export

Same as the last step of `/terrain-new` - there is no MCP tool for this, run it as a script
against the current session:

```python
from terrain import export, session

current = session.open_session(name)
result = export.write(
    current.cfg,
    current.channels(),
    current.water,
    current.splat_result(),
    rule_path=current.biome_file,
)
print(result.manifest_path, result.files)
```

This overwrites the prior deliverable set (height/splat/scatter/normal/terrain.json) in place
and re-verifies it. It does not touch `preview_*.png` - leave those as the visual changelog.

## Report

State what was changed (which file, which field, old value -> new value), the coverage table
diff, and whether re-export succeeded. If a change didn't fix what it was meant to, say so
rather than re-running the same edit with a different seed - a tune that doesn't work is
information about which layer of the stack actually owns the problem.
