from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV_VAR = "ASSETFORGE_BLENDER"

_CANDIDATE_GLOBS = {
    "win32": [
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    ],
    "darwin": ["/Applications/Blender.app/Contents/MacOS/Blender"],
    "linux": ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"],
}


def find_blender() -> str | None:
    override = os.environ.get(ENV_VAR)
    if override:
        return override if Path(override).exists() else None
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    for candidate in _CANDIDATE_GLOBS.get(sys.platform, []):
        if Path(candidate).exists():
            return candidate
    return None


def require_blender() -> str:
    found = find_blender()
    if found is None:
        raise RuntimeError(
            f"Blender 5.1+ not found. Install it or set {ENV_VAR} to the executable path."
        )
    return found
