"""Team-colour mask convention for RTS kinds. Stdlib only, so Blender's interpreter can import it.

The mask is an 8-bit PNG at atlas resolution sharing the asset's TEXCOORD_0 layout, written to
out/assets/<name>/textures/team_mask.png. Its red channel is the mask: 255 tints fully, 0 keeps
the albedo. The engine shades `albedo * mix(1, team_rgb, mask)`, so the albedo under the mask is
painted a light neutral grey and the team colour supplies the hue.
"""

from __future__ import annotations

MASK_FILENAME = "team_mask.png"
MANIFEST_KEY = "team_colour"
TEXCOORD = 0
CHANNEL = "r"
FORMULA = "albedo * mix(1, team_rgb, mask)"
REQUIRED_KINDS = ("unit",)
OPTIONAL_KINDS = ("creature",)
KINDS = REQUIRED_KINDS + OPTIONAL_KINDS
MIN_COVERAGE = 0.001
COVERAGE_THRESHOLD = 128


def manifest_entry(textures_dirname: str) -> dict:
    return {
        "mask": f"{textures_dirname}/{MASK_FILENAME}",
        "texcoord": TEXCOORD,
        "channel": CHANNEL,
        "formula": FORMULA,
    }
