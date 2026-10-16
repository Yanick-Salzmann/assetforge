from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from assets.recipe import validate_name
from terrain import config

GATE_NAME = ".gate.json"
MAX_PATCH_ITERATIONS = 4
HERO_PATCH_ITERATIONS = 8
PATCH_CAPS: dict[str, int] = {
    "hero_building": HERO_PATCH_ITERATIONS,
    "hero_vehicle": HERO_PATCH_ITERATIONS,
}


def patch_cap(kind: str) -> int:
    return PATCH_CAPS.get(kind, MAX_PATCH_ITERATIONS)


class GateError(RuntimeError):
    """Raised when a patch or export request violates the iteration cap or approval gate."""


@dataclass
class AssetGate:
    """Per-asset patch iteration count and human approval state, persisted to disk."""

    name: str
    kind: str = ""
    patch_count: int = 0
    approved: bool = False
    approval_note: str = ""
    approved_clips: list[str] = field(default_factory=list)

    def out_dir(self) -> Path:
        return config.ASSET_OUT_DIR / self.name

    def patch_cap(self) -> int:
        return patch_cap(self.kind)

    def record_patch(self, kind: str = "") -> AssetGate:
        if kind:
            self.kind = kind
        cap = self.patch_cap()
        if self.patch_count >= cap:
            self.write()
            raise GateError(
                f"asset {self.name!r} has used all {cap} allowed patch "
                "iterations; human review is required before further patches"
            )
        self.patch_count += 1
        self.write()
        return self

    def approve(self, note: str = "", clips: Sequence[str] = ()) -> AssetGate:
        """Record human sign-off. clips names every animation clip the reviewer watched in its
        filmstrip; approval covers the motion only for those."""
        self.approved = True
        self.approval_note = note
        self.approved_clips = sorted(set(clips))
        self.write()
        return self

    def require_approved(self, clips: Sequence[str] = ()) -> None:
        if not self.approved:
            raise GateError(
                f"asset {self.name!r} has not been approved by a human reviewer; export is blocked"
            )
        unreviewed = sorted(set(clips) - set(self.approved_clips))
        if unreviewed:
            raise GateError(
                f"asset {self.name!r} was approved without clip(s) {', '.join(unreviewed)}; "
                "approve again with every clip the reviewer watched"
            )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "patch_count": self.patch_count,
            "patch_cap": self.patch_cap(),
            "patches_remaining": max(0, self.patch_cap() - self.patch_count),
            "approved": self.approved,
            "approval_note": self.approval_note,
            "approved_clips": self.approved_clips,
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
        kind=str(data.get("kind", "")),
        patch_count=int(data.get("patch_count", 0)),
        approved=bool(data.get("approved", False)),
        approval_note=str(data.get("approval_note", "")),
        approved_clips=[str(clip) for clip in data.get("approved_clips", [])],
    )


def forget(name: str) -> None:
    _GATES.pop(name, None)
