from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from assets.recipe import validate_name
from terrain import config

GATE_NAME = ".gate.json"
MAX_PATCH_ITERATIONS = 4


class GateError(RuntimeError):
    """Raised when a patch or export request violates the iteration cap or approval gate."""


@dataclass
class AssetGate:
    """Per-asset patch iteration count and human approval state, persisted to disk."""

    name: str
    patch_count: int = 0
    approved: bool = False
    approval_note: str = ""

    def out_dir(self) -> Path:
        return config.ASSET_OUT_DIR / self.name

    def record_patch(self) -> AssetGate:
        if self.patch_count >= MAX_PATCH_ITERATIONS:
            raise GateError(
                f"asset {self.name!r} has used all {MAX_PATCH_ITERATIONS} allowed patch "
                "iterations; human review is required before further patches"
            )
        self.patch_count += 1
        self.write()
        return self

    def approve(self, note: str = "") -> AssetGate:
        self.approved = True
        self.approval_note = note
        self.write()
        return self

    def require_approved(self) -> None:
        if not self.approved:
            raise GateError(
                f"asset {self.name!r} has not been approved by a human reviewer; export is blocked"
            )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "patch_count": self.patch_count,
            "patches_remaining": MAX_PATCH_ITERATIONS - self.patch_count,
            "approved": self.approved,
            "approval_note": self.approval_note,
        }

    def write(self) -> None:
        target = self.out_dir()
        target.mkdir(parents=True, exist_ok=True)
        (target / GATE_NAME).write_text(json.dumps(self.as_dict()), encoding="utf-8")


_GATES: dict[str, AssetGate] = {}


def open_gate(name: str) -> AssetGate:
    """The resident gate for this asset, reloaded from disk if the server restarted."""
    validate_name(name)
    gate = _GATES.get(name)
    if gate is not None:
        return gate
    gate = _load_gate(name)
    _GATES[name] = gate
    return gate


def _load_gate(name: str) -> AssetGate:
    path = config.ASSET_OUT_DIR / name / GATE_NAME
    if not path.is_file():
        return AssetGate(name=name)
    data = json.loads(path.read_text(encoding="utf-8"))
    return AssetGate(
        name=name,
        patch_count=int(data.get("patch_count", 0)),
        approved=bool(data.get("approved", False)),
        approval_note=str(data.get("approval_note", "")),
    )


def forget(name: str) -> None:
    _GATES.pop(name, None)
