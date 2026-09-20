from __future__ import annotations

import re

import pytest

from terrain import config
from terrain.device import (
    BACKEND_PREFERENCE,
    DEVICE_ENV_VAR,
    Device,
    DeviceUnavailableError,
    available_backends,
    resolve,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)


def test_cpu_is_always_available():
    assert "cpu" in available_backends()


def test_resolve_returns_a_preferred_backend():
    resolved = resolve()
    assert isinstance(resolved, Device)
    assert resolved.backend in BACKEND_PREFERENCE
    assert resolved.backend == available_backends()[0]


def test_resolve_prefers_accelerators_over_cpu():
    if available_backends() == ("cpu",):
        pytest.skip("no accelerator on this machine")
    assert resolve().is_accelerator


@pytest.mark.parametrize("backend", BACKEND_PREFERENCE)
def test_env_override_forces_each_backend(monkeypatch, backend):
    monkeypatch.setenv(DEVICE_ENV_VAR, backend)
    if backend not in available_backends():
        with pytest.raises(DeviceUnavailableError):
            resolve()
        return
    resolved = resolve()
    assert resolved.backend == backend
    assert resolved.spec == backend
    assert config.device() == backend


@pytest.mark.parametrize("backend", BACKEND_PREFERENCE)
def test_argument_override_beats_env(monkeypatch, backend):
    monkeypatch.setenv(DEVICE_ENV_VAR, "cpu")
    if backend not in available_backends():
        pytest.skip(f"{backend} not available")
    assert resolve(backend).backend == backend


def test_blank_env_override_falls_back_to_autodetect(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "   ")
    assert resolve().backend == available_backends()[0]


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "tpu")
    with pytest.raises(DeviceUnavailableError):
        resolve()


def test_bad_device_index_is_rejected():
    with pytest.raises(DeviceUnavailableError):
        resolve("cpu:first")


def test_cpu_pipeline_runs_without_an_accelerator():
    torch = pytest.importorskip("torch")
    resolved = resolve("cpu")
    assert resolved.supports("float32")
    grid = torch.zeros((4, 16, 16), dtype=torch.float32, device=resolved.torch_device())
    grid[0] = 1.0
    assert grid.device.type == "cpu"
    assert float(grid.sum()) == 256.0


def test_resolved_device_reports_dtype_support():
    resolved = resolve("cpu")
    assert resolved.supports("float32")
    assert resolved.dtypes <= {"float64", "float32", "float16", "bfloat16"}


def test_as_dict_is_json_friendly():
    payload = resolve("cpu").as_dict()
    assert payload["device"] == "cpu"
    assert payload["override_env_var"] == DEVICE_ENV_VAR
    assert isinstance(payload["dtypes"], list)


def test_no_module_outside_the_resolver_names_a_backend():
    package = config.REPO_ROOT / "terrain"
    named = re.compile(r"\b(cuda|mps)\b")
    offenders = []
    for path in package.rglob("*.py"):
        if path.name == "device.py":
            continue
        if named.search(path.read_text(encoding="utf-8").lower()):
            offenders.append(path.name)
    assert offenders == []
