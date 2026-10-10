from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from assets.blender import require_blender
from library.materials import MaterialIndex
from terrain import atmosphere as atmosphere_mod
from terrain import placement as placement_mod
from terrain import vegetation as vegetation_mod
from terrain.budget import DEFAULT_BUDGET, Preview, PreviewBudget, deliver_file
from terrain.config import HEIGHTMAP_MAX, LIBRARY_DIR, MapConfig, MapConfigError
from terrain.splat import SplatResult

SCRIPT_PATH = Path(__file__).resolve().parent / "blender_scripts" / "beauty_render.py"

THREE_QUARTER_NAME = "preview_beauty_3q.png"
GROUND_NAME = "preview_beauty_ground.png"

DEFAULT_GRID_RESOLUTION = 512
DEFAULT_PATCH_SIZE_M = 200.0
DEFAULT_SAMPLES = 128
DEFAULT_RESOLUTION = (960, 540)
DEFAULT_TIMEOUT_S = 240.0
DEFAULT_THREE_QUARTER_AZIMUTH_DEG = 45.0
DEFAULT_THREE_QUARTER_ALTITUDE_DEG = 35.0
DEFAULT_GROUND_EYE_HEIGHT_M = 1.8
DEFAULT_DEVICE = "OPTIX"
LOG_TAIL_CHARS = 4000

RAYLEIGH_REFERENCE = 1.5
MIE_CLEAN = 4e-6
MIE_PER_TURBIDITY = 7e-6
SKY_MIE_COEFFICIENT = 2e-5
SKY_DENSITY_MAX = 10.0
SUN_KEY_MIN_SIN = -0.05
MOON_IRRADIANCE = 3.3
MOON_COLOUR = (0.66, 0.76, 1.0)
EXPOSURE_KEY = 0.18
EXPOSURE_NIGHT_KEY_SCALE = 0.2
EXPOSURE_HIGHLIGHT_PERCENTILE = 0.98
EXPOSURE_HIGHLIGHT_WHITE = 2.0
EXPOSURE_MIN_STOPS = -12.0
EXPOSURE_MAX_STOPS = 10.0
EXPOSURE_PROBE_DIVISOR = 4
EXPOSURE_PROBE_SAMPLES = 16

KIND_SINK_M: dict[str, float] = {
    "conifer": 0.3,
    "broadleaf": 0.3,
    "cactus": 0.1,
    "shrub": 0.1,
    "rock": 0.35,
    "grass": 0.02,
    "flower": 0.02,
    "debris": 0.08,
}
KIND_REACH_M: dict[str, float] = {
    "conifer": math.inf,
    "broadleaf": math.inf,
    "cactus": 1500.0,
    "shrub": 600.0,
    "rock": 1500.0,
    "grass": 250.0,
    "flower": 200.0,
    "debris": 400.0,
}
SCATTER_RECORD = ("x", "y", "z", "rotation_x", "rotation_y", "rotation_z", "scale")


class BeautyRenderError(MapConfigError):
    """Raised when the headless Blender beauty render cannot be produced."""


@dataclass(frozen=True)
class BeautyRender:
    """The two rendered views, where the camera ended up, and which compute device rendered them."""

    three_quarter: Path
    ground: Path
    centre_world_m: tuple[float, float, float]
    device: dict[str, Any]
    lighting: dict[str, Any]
    scatter: dict[str, int]
    log: str


@dataclass(frozen=True)
class ScatterBatch:
    """One species variant's instances in Blender world space, and the mesh they instance.

    records holds SCATTER_RECORD rows: position in metres from the map's south-west corner
    (+x east along image columns, +y north toward the image top, z up), an XYZ Euler rotation,
    and the uniform scale with the variant's unit_scale already folded in. glb is None when
    the species has no mesh on disk; the render then stands in a primitive of kind size.
    """

    species: str
    kind: str
    variant: int
    glb: Path | None
    records: np.ndarray

    @property
    def count(self) -> int:
        return int(self.records.shape[0])


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def lighting_args(atmosphere: atmosphere_mod.Atmosphere) -> dict[str, Any]:
    """Sun, sky and moon for the Blender render, from the same atmosphere the viewer reads.

    The sky is Blender's multiple-scattering model with its sun disc as the key light, so the
    sun's colour and the sky's brightness come from one physical model. Air and aerosol
    densities scale from the viewer's Rayleigh and Mie coefficients. Once the sun is below the
    horizon a full-moon sun lamp takes over as the key, faded in exactly as the viewer fades it,
    and the exposure key drops so night still reads as night after metering.
    """
    sun = atmosphere_mod.sun_position(atmosphere)
    moon = atmosphere_mod.moon_position(atmosphere)
    mie = MIE_CLEAN + MIE_PER_TURBIDITY * (atmosphere.turbidity - 1.0)
    sin_sun = math.sin(sun.elevation_rad)
    night = 1.0 - _smoothstep(-0.15, -0.05, sin_sun)
    moon_up = _smoothstep(-0.02, 0.12, math.sin(moon.elevation_rad))
    moon_energy = MOON_IRRADIANCE * night * moon_up if sin_sun <= SUN_KEY_MIN_SIN else 0.0
    return {
        "sky": {
            "sun_elevation_rad": sun.elevation_rad,
            "sun_rotation_rad": sun.azimuth_rad,
            "air_density": min(atmosphere.rayleigh / RAYLEIGH_REFERENCE, SKY_DENSITY_MAX),
            "aerosol_density": min(mie / SKY_MIE_COEFFICIENT, SKY_DENSITY_MAX),
        },
        "moon": {
            "direction": list(moon.direction()),
            "energy": moon_energy,
            "colour": list(MOON_COLOUR),
        },
        "exposure": {
            "key": EXPOSURE_KEY * (1.0 - night * (1.0 - EXPOSURE_NIGHT_KEY_SCALE)),
            "highlight_percentile": EXPOSURE_HIGHLIGHT_PERCENTILE,
            "highlight_white": EXPOSURE_HIGHLIGHT_WHITE,
            "min_stops": EXPOSURE_MIN_STOPS,
            "max_stops": EXPOSURE_MAX_STOPS,
            "probe_divisor": EXPOSURE_PROBE_DIVISOR,
            "probe_samples": EXPOSURE_PROBE_SAMPLES,
        },
        "summary": {
            "time_of_day_h": atmosphere.time_of_day_h,
            "sun_elevation_deg": math.degrees(sun.elevation_rad),
            "sun_azimuth_deg": math.degrees(sun.azimuth_rad),
            "moon_elevation_deg": math.degrees(moon.elevation_rad),
            "key": "sun" if sin_sun > SUN_KEY_MIN_SIN else "moon",
        },
    }


def _tilt_matrices(normals: np.ndarray) -> np.ndarray:
    """Rotations taking +z onto each unit normal (Rodrigues; normals always point up)."""
    count = normals.shape[0]
    skew = np.zeros((count, 3, 3))
    skew[:, 0, 2] = normals[:, 0]
    skew[:, 1, 2] = normals[:, 1]
    skew[:, 2, 0] = -normals[:, 0]
    skew[:, 2, 1] = -normals[:, 1]
    factor = 1.0 / (1.0 + normals[:, 2])
    return np.eye(3)[None] + skew + (skew @ skew) * factor[:, None, None]


def instance_euler(normals: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """XYZ Euler angles of tilt(normal) @ Rz(yaw): yaw about the instance's own up, then the
    lean onto the surface normal - the composition the viewer's writePlacements uses."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    spin = np.zeros((yaw.shape[0], 3, 3))
    spin[:, 0, 0] = cos_yaw
    spin[:, 0, 1] = -sin_yaw
    spin[:, 1, 0] = sin_yaw
    spin[:, 1, 1] = cos_yaw
    spin[:, 2, 2] = 1.0
    rotation = _tilt_matrices(normals) @ spin
    return np.stack(
        [
            np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]),
            np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0)),
            np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]),
        ],
        axis=-1,
    )


def _surface_normals(
    cfg: MapConfig, height_u16: np.ndarray, corner_points: np.ndarray, align: float
) -> np.ndarray:
    """The viewer's alignedUp, in Blender axes: the heightfield normal leaned by slope_align."""
    count = corner_points.shape[0]
    if align <= 0.0:
        return np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    step = cfg.metres_per_pixel
    offsets = {
        "east": (step, 0.0),
        "west": (-step, 0.0),
        "south": (0.0, step),
        "north": (0.0, -step),
    }
    heights = {
        name: placement_mod.sample_height(height_u16, corner_points + np.array(offset), cfg)
        for name, offset in offsets.items()
    }
    along_columns = (heights["east"] - heights["west"]) / (2.0 * step)
    along_rows = (heights["south"] - heights["north"]) / (2.0 * step)
    normals = np.stack([-along_columns * align, along_rows * align, np.ones(count)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def scatter_batches(
    cfg: MapConfig,
    placements: Sequence[placement_mod.Placement],
    variants: Mapping[str, tuple[vegetation_mod.Variant, ...]],
    height_u16: np.ndarray,
    centre_m: tuple[float, float],
    library_dir: Path = LIBRARY_DIR,
) -> tuple[ScatterBatch, ...]:
    """The exported instances, moved into Blender's frame and split per mesh variant.

    Each kind is kept only within the viewer's own reach of the view centre, so a ground-level
    render does not pay for grass two kilometres behind the camera.
    """
    half = cfg.world_size_m / 2.0
    batches = []
    for placed in placements:
        species = placed.species
        records = placed.instances.astype(np.float64)
        reach = KIND_REACH_M[species.kind]
        corner = records[:, 0:2] + half
        world_x = corner[:, 0]
        world_y = cfg.world_size_m - corner[:, 1]
        near = np.hypot(world_x - centre_m[0], world_y - centre_m[1]) <= reach
        records, corner, world_x, world_y = records[near], corner[near], world_x[near], world_y[near]
        if records.shape[0] == 0:
            continue
        normals = _surface_normals(cfg, height_u16, corner, species.slope_align)
        euler = instance_euler(normals, records[:, 3])
        scale = records[:, 4]
        z = records[:, 2] - KIND_SINK_M[species.kind] * scale
        choices = variants.get(species.name, ()) or (None,)
        for index, variant in enumerate(choices):
            chosen = records[:, 5] == index
            if not chosen.any():
                continue
            glb = None if variant is None else library_dir / variant.glb
            unit_scale = 1.0 if variant is None else variant.unit_scale
            rows = np.stack(
                [
                    world_x[chosen],
                    world_y[chosen],
                    z[chosen],
                    euler[chosen, 0],
                    euler[chosen, 1],
                    euler[chosen, 2],
                    scale[chosen] * unit_scale,
                ],
                axis=-1,
            ).astype("<f4")
            present = glb if glb is not None and glb.is_file() else None
            batches.append(ScatterBatch(species.name, species.kind, index, present, np.ascontiguousarray(rows)))
    return tuple(batches)


def _texture_index(texture_name: str) -> int:
    return int(texture_name.removeprefix("splat_").removesuffix(".png"))


def _height_u16(height: torch.Tensor) -> np.ndarray:
    values = height.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).numpy()
    return np.ascontiguousarray((values * HEIGHTMAP_MAX + 0.5).astype(np.uint16))


def _write_height_png(height: torch.Tensor, path: Path) -> Path:
    """A 16-bit scratch heightmap for the Displace modifier, not the canonical export deliverable.

    Not flipped: Blender shows a PNG's first row at the top of the image, i.e. at UV v=1, which
    the grid puts at +y. Row 0 therefore lands on the north edge, as in the viewer and every
    other preview, and the sun's compass azimuth means the same thing in both renders.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_height_u16(height)).save(path)
    return path


def _write_records(batch: ScatterBatch, path: Path) -> Path:
    path.write_bytes(batch.records.tobytes())
    return path


def _scatter_counts(batches: Sequence[ScatterBatch]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for batch in batches:
        counts[batch.species] = counts.get(batch.species, 0) + batch.count
    return counts


def _write_splat_png(block: np.ndarray, path: Path) -> Path:
    Image.fromarray(np.ascontiguousarray(block), "RGBA").save(path)
    return path


def _material_metadata(names: Sequence[str], index: MaterialIndex) -> dict[str, dict[str, Any]]:
    metadata = {}
    for name in names:
        material = index[name]
        missing = material.missing()
        if missing:
            raise BeautyRenderError(
                f"material {name!r} is missing {', '.join(missing)} on disk; re-run the pull script"
            )
        paths = material.paths()
        metadata[name] = {
            "tiling_m": material.tiling_m,
            "albedo": str(paths["albedo"]),
            "normal": str(paths["normal"]),
            "roughness": str(paths["roughness"]),
        }
    return metadata


def _validate_centre(centre: Sequence[float]) -> tuple[float, float]:
    if len(centre) != 2:
        raise BeautyRenderError(f"centre {tuple(centre)} must hold two fractions")
    x, y = float(centre[0]), float(centre[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise BeautyRenderError(f"centre {(x, y)} must lie within [0, 1]")
    return x, y


def render(
    cfg: MapConfig,
    height: torch.Tensor,
    splat: SplatResult,
    material_index: MaterialIndex,
    out_dir: Path | None = None,
    centre: Sequence[float] = (0.5, 0.5),
    patch_size_m: float = DEFAULT_PATCH_SIZE_M,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    samples: int = DEFAULT_SAMPLES,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    atmosphere: atmosphere_mod.Atmosphere | None = None,
    time_of_day_h: float | None = None,
    placements: Sequence[placement_mod.Placement] = (),
    variants: Mapping[str, tuple[vegetation_mod.Variant, ...]] | None = None,
    three_quarter_azimuth_deg: float = DEFAULT_THREE_QUARTER_AZIMUTH_DEG,
    three_quarter_altitude_deg: float = DEFAULT_THREE_QUARTER_ALTITUDE_DEG,
    ground_eye_height_m: float = DEFAULT_GROUND_EYE_HEIGHT_M,
    device: str = DEFAULT_DEVICE,
    blender_executable: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> BeautyRender:
    """Blender headless: displace a grid by the height field, bind the splat's real tiling
    materials, and render one 3/4 view and one ground-level view at a chosen map position.

    Lit by the biome's [atmosphere] unless one is handed in; time_of_day_h overrides its time.
    Dressed with the given scatter placements (export.scatter_placements), each instancing its
    species' kit meshes as the viewer does.

    Reads only what is handed in - no dependency on export.py's terrain.json, since this is the
    Phase 2 gate and must run before the export contract exists.
    """
    if tuple(height.shape) != cfg.shape:
        raise BeautyRenderError(f"height {tuple(height.shape)} does not match cfg.shape {cfg.shape}")
    centre_x, centre_y = _validate_centre(centre)
    if grid_resolution < 2:
        raise BeautyRenderError(f"grid_resolution {grid_resolution} must be at least 2")
    if samples < 1:
        raise BeautyRenderError(f"samples {samples} must be at least 1")
    if patch_size_m <= 0.0:
        raise BeautyRenderError(f"patch_size_m {patch_size_m} must be positive")

    target_dir = cfg.out_dir() if out_dir is None else Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    executable = blender_executable or require_blender()
    assignment = splat.assignment()
    materials_used = sorted({entry["material"] for entry in assignment})
    materials_meta = _material_metadata(materials_used, material_index)
    sky = splat.biome.atmosphere if atmosphere is None else atmosphere
    if time_of_day_h is not None:
        sky = atmosphere_mod.parse({**sky.as_dict(), "time_of_day_h": time_of_day_h}, "time_of_day_h")
    lighting = lighting_args(sky)
    if placements and variants is None:
        variants = vegetation_mod.resolve_all(splat.biome.species)
    batches = scatter_batches(
        cfg,
        placements,
        variants or {},
        _height_u16(height),
        (centre_x * cfg.world_size_m, centre_y * cfg.world_size_m),
    )

    with tempfile.TemporaryDirectory(prefix="beauty_") as scratch_name:
        scratch = Path(scratch_name)
        height_png = _write_height_png(height, scratch / "height.png")
        splat_textures = {
            str(texture_index): str(_write_splat_png(block, scratch / f"splat_{texture_index}.png"))
            for texture_index, block in enumerate(splat.textures)
        }

        three_quarter_path = target_dir / THREE_QUARTER_NAME
        ground_path = target_dir / GROUND_NAME
        result_json = scratch / "result.json"
        args = {
            "world_size_m": cfg.world_size_m,
            "height_range_m": cfg.height_range_m,
            "height_png": str(height_png),
            "grid_resolution": min(grid_resolution, cfg.resolution),
            "splat_textures": splat_textures,
            "materials": materials_meta,
            "layers": [
                {
                    "material": entry["material"],
                    "texture": _texture_index(entry["texture"]),
                    "channel": entry["channel"],
                }
                for entry in assignment
            ],
            "lighting": lighting,
            "scatter": [
                {
                    "name": f"{batch.species}_{batch.variant}",
                    "kind": batch.kind,
                    "glb": None if batch.glb is None else str(batch.glb),
                    "size_m": vegetation_mod.KIND_SIZE_M[batch.kind],
                    "records": str(_write_records(batch, scratch / f"scatter_{index}.bin")),
                    "count": batch.count,
                }
                for index, batch in enumerate(batches)
            ],
            "camera": {
                "centre_frac": [centre_x, centre_y],
                "patch_size_m": patch_size_m,
                "three_quarter": {
                    "azimuth_deg": three_quarter_azimuth_deg,
                    "altitude_deg": three_quarter_altitude_deg,
                },
                "ground": {
                    "eye_height_m": ground_eye_height_m,
                    "look_azimuth_deg": three_quarter_azimuth_deg,
                    "back_distance_m": max(10.0, patch_size_m * 0.25),
                },
            },
            "render": {
                "samples": samples,
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "device": device,
            },
            "output": {"three_quarter": str(three_quarter_path), "ground": str(ground_path)},
            "result_json": str(result_json),
        }
        args_path = scratch / "args.json"
        args_path.write_text(json.dumps(args), encoding="utf-8")

        completed = subprocess.run(
            [
                executable,
                "--background",
                "--factory-startup",
                "--python",
                str(SCRIPT_PATH),
                "--",
                str(args_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = completed.stdout + completed.stderr
        if completed.returncode != 0 or not three_quarter_path.is_file() or not ground_path.is_file():
            raise BeautyRenderError(
                f"Blender beauty render failed (exit {completed.returncode}):\n{log[-LOG_TAIL_CHARS:]}"
            )
        payload = json.loads(result_json.read_text(encoding="utf-8")) if result_json.is_file() else {}

    return BeautyRender(
        three_quarter=three_quarter_path,
        ground=ground_path,
        centre_world_m=tuple(payload.get("centre_world_m", (0.0, 0.0, 0.0))),
        device=payload.get("device", {}),
        lighting={**lighting["summary"], "exposure_stops": payload.get("exposure_stops", {})},
        scatter=_scatter_counts(batches),
        log=log,
    )


def beauty_previews(
    cfg: MapConfig,
    height: torch.Tensor,
    splat: SplatResult,
    material_index: MaterialIndex,
    budget: PreviewBudget = DEFAULT_BUDGET,
    **kwargs: Any,
) -> tuple[BeautyRender, Preview, Preview]:
    """Render both views and hand back budgeted previews of each, agent-ready."""
    result = render(cfg, height, splat, material_index, **kwargs)
    return result, deliver_file(result.three_quarter, budget), deliver_file(result.ground, budget)
