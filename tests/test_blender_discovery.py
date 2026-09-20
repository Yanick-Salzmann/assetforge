from assets import blender


def test_env_override_wins(monkeypatch, tmp_path):
    exe = tmp_path / "blender.exe"
    exe.write_text("")
    monkeypatch.setenv(blender.ENV_VAR, str(exe))
    assert blender.find_blender() == str(exe)


def test_missing_env_override_is_not_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(blender.ENV_VAR, str(tmp_path / "nope.exe"))
    assert blender.find_blender() is None


def test_require_blender_raises_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv(blender.ENV_VAR, str(tmp_path / "nope.exe"))
    try:
        blender.require_blender()
    except RuntimeError as exc:
        assert blender.ENV_VAR in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
