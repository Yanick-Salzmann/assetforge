from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

DEVICE_ENV_VAR = "ASSETFORGE_DEVICE"

BACKEND_PREFERENCE = ("cuda", "mps", "cpu")

_PROBE_DTYPES = ("float64", "float32", "float16", "bfloat16")


class DeviceUnavailableError(RuntimeError):
    """Raised when a backend is requested explicitly but cannot be used here."""


@dataclass(frozen=True)
class Device:
    backend: str
    index: int | None
    label: str
    total_memory_bytes: int | None
    dtypes: frozenset[str]

    @property
    def spec(self) -> str:
        if self.index is None:
            return self.backend
        return f"{self.backend}:{self.index}"

    @property
    def is_accelerator(self) -> bool:
        return self.backend != "cpu"

    def supports(self, dtype: str) -> bool:
        return dtype in self.dtypes

    def torch_device(self) -> Any:
        import torch

        return torch.device(self.spec)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.spec,
            "backend": self.backend,
            "label": self.label,
            "total_memory_bytes": self.total_memory_bytes,
            "dtypes": sorted(self.dtypes),
            "override_env_var": DEVICE_ENV_VAR,
        }

    def __str__(self) -> str:
        return self.spec


def _import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_available(torch: Any, backend: str) -> bool:
    if backend == "cpu":
        return True
    if torch is None:
        return False
    if backend == "cuda":
        return bool(torch.cuda.is_available())
    if backend == "mps":
        backends = getattr(torch.backends, "mps", None)
        return bool(backends is not None and backends.is_available())
    return False


def _label(torch: Any, backend: str, index: int | None) -> str:
    if backend == "cuda" and torch is not None:
        return str(torch.cuda.get_device_name(index or 0))
    if backend == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


def _total_memory(torch: Any, backend: str, index: int | None) -> int | None:
    if torch is None:
        return None
    if backend == "cuda":
        return int(torch.cuda.get_device_properties(index or 0).total_memory)
    if backend == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is None:
            return None
        return int(recommended())
    return None


def _probe_dtypes(torch: Any, spec: str) -> frozenset[str]:
    if torch is None:
        return frozenset({"float32"})
    supported = set()
    for name in _PROBE_DTYPES:
        dtype = getattr(torch, name, None)
        if dtype is None:
            continue
        try:
            probe = torch.ones(2, dtype=dtype, device=spec)
            if float((probe + probe).sum()) == 4.0:
                supported.add(name)
        except Exception:
            continue
    return frozenset(supported)


def _parse(requested: str) -> tuple[str, int | None]:
    backend, _, tail = requested.strip().lower().partition(":")
    if backend not in BACKEND_PREFERENCE:
        raise DeviceUnavailableError(
            f"unknown device {requested!r}; expected one of {', '.join(BACKEND_PREFERENCE)}"
        )
    if not tail:
        return backend, None
    if not tail.isdigit():
        raise DeviceUnavailableError(f"device index in {requested!r} is not a number")
    return backend, int(tail)


def _build(torch: Any, backend: str, index: int | None) -> Device:
    spec = backend if index is None else f"{backend}:{index}"
    return Device(
        backend=backend,
        index=index,
        label=_label(torch, backend, index),
        total_memory_bytes=_total_memory(torch, backend, index),
        dtypes=_probe_dtypes(torch, spec),
    )


@lru_cache(maxsize=None)
def _resolve(requested: str | None) -> Device:
    torch = _import_torch()
    if requested is not None:
        backend, index = _parse(requested)
        if not _is_available(torch, backend):
            raise DeviceUnavailableError(
                f"device {requested!r} was requested but is not available on this machine"
            )
        return _build(torch, backend, index)
    for backend in BACKEND_PREFERENCE:
        if _is_available(torch, backend):
            return _build(torch, backend, None)
    return _build(torch, "cpu", None)


def resolve(override: str | None = None) -> Device:
    """Resolve the compute device, honouring an override or ASSETFORGE_DEVICE."""
    requested = override if override is not None else os.environ.get(DEVICE_ENV_VAR)
    if requested is not None and not requested.strip():
        requested = None
    return _resolve(requested)


def as_torch_device(device: Any | None = None) -> Any:
    """Coerce a Device, a spec string or None into a torch.device."""
    import torch

    if device is None:
        return resolve().torch_device()
    if isinstance(device, Device):
        return device.torch_device()
    return torch.device(device)


def available_backends() -> tuple[str, ...]:
    torch = _import_torch()
    return tuple(name for name in BACKEND_PREFERENCE if _is_available(torch, name))


def clear_cache() -> None:
    _resolve.cache_clear()
