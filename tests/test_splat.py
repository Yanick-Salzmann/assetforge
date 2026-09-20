from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from terrain import channels as channel_module
from terrain import splat
from terrain.config import CHANNEL_NAMES, MapConfig, MapConfigError

CPU = "cpu"
SIZE = 8


def cfg(resolution: int = SIZE) -> MapConfig:
    return MapConfig(name="unit", resolution=resolution, world_size_m=512.0, seed=3)


def ramp(size: int = SIZE) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, size, device=CPU, dtype=torch.float32)
    return axis.reshape(-1, 1).expand(size, size).contiguous()


def stack() -> channel_module.ChannelStack:
    built = channel_module.ChannelStack(cfg(), CPU)
    field = ramp()
    for name in CHANNEL_NAMES:
        built[name] = field
    return built


def env(**fields: torch.Tensor) -> dict[str, torch.Tensor]:
    return fields


def test_parse_reports_referenced_channels():
    expression = splat.parse("smoothstep(flow, 0.55, 0.8) * (1 - wetness * 0.3)")
    assert expression.channels == ("flow", "wetness")
    assert str(expression) == "smoothstep(flow, 0.55, 0.8) * (1 - wetness * 0.3)"


def test_caret_binds_tighter_than_multiplication():
    field = torch.full((2, 2), 0.5, device=CPU)
    value = splat.evaluate("deposition^2 * 4", env(deposition=field))
    assert torch.allclose(value, torch.full((2, 2), 1.0, device=CPU))


def test_caret_and_double_star_agree():
    field = torch.full((2, 2), 0.25, device=CPU)
    caret = splat.evaluate("flow^0.5", env(flow=field))
    star = splat.evaluate("flow ** 0.5", env(flow=field))
    assert torch.allclose(caret, star)
    assert torch.allclose(caret, torch.full((2, 2), 0.5, device=CPU))


def test_arithmetic_matches_torch():
    flow = ramp()
    wetness = ramp().flip(0)
    value = splat.evaluate("(flow + wetness) / 2 - 0.1", env(flow=flow, wetness=wetness))
    assert torch.allclose(value, (flow + wetness) / 2 - 0.1)


def test_unary_minus():
    flow = ramp()
    assert torch.allclose(splat.evaluate("-flow", env(flow=flow)), -flow)


def test_smoothstep_is_clamped_and_smooth():
    field = torch.tensor([[0.0, 0.5, 0.75, 1.0]], device=CPU)
    value = splat.evaluate("smoothstep(flow, 0.5, 1.0)", env(flow=field))
    assert torch.allclose(value[0, 0], torch.tensor(0.0))
    assert torch.allclose(value[0, 1], torch.tensor(0.0))
    assert torch.allclose(value[0, 3], torch.tensor(1.0))
    assert torch.allclose(value[0, 2], torch.tensor(0.5))
    assert float(value.min()) >= 0.0
    assert float(value.max()) <= 1.0


def test_smoothstep_rejects_equal_edges():
    with pytest.raises(MapConfigError):
        splat.evaluate("smoothstep(flow, 0.5, 0.5)", env(flow=ramp()))


def test_step_of_comparison_is_a_binary_mask():
    slope = torch.tensor([[0.05, 0.12, 0.4]], device=CPU)
    value = splat.evaluate("step(slope < 0.12)", env(slope=slope))
    assert torch.equal(value, torch.tensor([[1.0, 0.0, 0.0]], device=CPU))


def test_comparison_without_step_is_already_a_mask():
    slope = torch.tensor([[0.05, 0.4]], device=CPU)
    assert torch.equal(
        splat.evaluate("slope > 0.1", env(slope=slope)),
        torch.tensor([[0.0, 1.0]], device=CPU),
    )


def test_chained_comparison():
    slope = torch.tensor([[0.05, 0.3, 0.9]], device=CPU)
    value = splat.evaluate("0.1 < slope < 0.5", env(slope=slope))
    assert torch.equal(value, torch.tensor([[0.0, 1.0, 0.0]], device=CPU))


def test_boolean_operators():
    slope = torch.tensor([[0.1, 0.1, 0.9, 0.9]], device=CPU)
    flow = torch.tensor([[0.1, 0.9, 0.1, 0.9]], device=CPU)
    both = splat.evaluate("(slope < 0.5) and (flow > 0.5)", env(slope=slope, flow=flow))
    either = splat.evaluate("(slope < 0.5) or (flow > 0.5)", env(slope=slope, flow=flow))
    assert torch.equal(both, torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=CPU))
    assert torch.equal(either, torch.tensor([[1.0, 1.0, 0.0, 1.0]], device=CPU))


def test_not_inverts_a_mask():
    slope = torch.tensor([[0.1, 0.9]], device=CPU)
    value = splat.evaluate("not (slope < 0.5)", env(slope=slope))
    assert torch.equal(value, torch.tensor([[0.0, 1.0]], device=CPU))


def test_clamp_defaults_to_the_unit_range():
    flow = torch.tensor([[-1.0, 0.5, 4.0]], device=CPU)
    assert torch.equal(
        splat.evaluate("clamp(flow * 1)", env(flow=flow)),
        torch.tensor([[0.0, 0.5, 1.0]], device=CPU),
    )
    assert torch.equal(
        splat.evaluate("clamp(flow, 0.25, 0.75)", env(flow=flow)),
        torch.tensor([[0.25, 0.5, 0.75]], device=CPU),
    )


def test_min_and_max_are_variadic():
    flow = torch.tensor([[0.2, 0.8]], device=CPU)
    wetness = torch.tensor([[0.6, 0.4]], device=CPU)
    assert torch.equal(
        splat.evaluate("min(flow, wetness, 0.5)", env(flow=flow, wetness=wetness)),
        torch.tensor([[0.2, 0.4]], device=CPU),
    )
    assert torch.equal(
        splat.evaluate("max(flow, wetness)", env(flow=flow, wetness=wetness)),
        torch.tensor([[0.6, 0.8]], device=CPU),
    )


def test_pow_function_matches_the_operator():
    flow = torch.tensor([[0.25, 0.81]], device=CPU)
    assert torch.allclose(
        splat.evaluate("pow(flow, 0.5)", env(flow=flow)),
        splat.evaluate("flow^0.5", env(flow=flow)),
    )


def test_constant_expression_takes_the_stack_shape():
    built = stack()
    value = splat.parse("0.5").evaluate(built)
    assert value.shape == built.cfg.shape
    assert torch.allclose(value, torch.full(built.cfg.shape, 0.5))


def test_constant_expression_accepts_an_explicit_reference():
    reference = torch.zeros((3, 3), device=CPU)
    value = splat.parse("1").evaluate({}, like=reference)
    assert torch.equal(value, torch.ones((3, 3), device=CPU))


def test_constant_expression_without_any_channel_is_an_error():
    with pytest.raises(MapConfigError):
        splat.parse("1").evaluate({})


def test_evaluates_against_a_channel_stack():
    built = stack()
    value = splat.evaluate("deposition^1.5 * step(slope < 0.9) * patchiness_mid", built)
    assert value.shape == built.cfg.shape
    assert value.dtype == torch.float32


def test_unfilled_channel_is_rejected():
    built = channel_module.ChannelStack(cfg(), CPU)
    built["height"] = ramp()
    with pytest.raises(MapConfigError):
        splat.evaluate("flow * 2", built)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        "elevation * 2",
        "flow +",
        "noise(flow)",
        "flow.mean()",
        "__import__('os')",
        "[flow, wetness]",
        "flow if wetness else 0",
        "lambda: 1",
        "min(flow)",
        "smoothstep(flow, 0.1)",
        "step(flow, 1)",
        "clamp(flow, 0, 1, 2)",
        "step(x=flow)",
        "flow & wetness",
        "flow | wetness",
        "flow // 2",
        "'grass'",
        "True",
        "smoothstep",
        "flow in wetness",
    ],
)
def test_rejected_sources(source: str):
    with pytest.raises(MapConfigError):
        splat.parse(source)


def test_plan_rules_all_parse():
    sources = (
        "smoothstep(flow, 0.55, 0.8) * (1 - wetness*0.3)",
        "deposition^1.5 * step(slope < 0.12) * patchiness_mid",
        "(1 - moisture) * step(slope < 0.35) * patchiness_coarse",
        "moisture * wetness^0.5 * step(slope < 0.35)",
        "max(step(slope > 0.6), bedrock * wear)",
    )
    built = stack()
    for source in sources:
        expression = splat.parse(source)
        assert set(expression.channels) <= set(CHANNEL_NAMES)
        assert expression.evaluate(built).shape == built.cfg.shape


def test_evaluation_is_device_consistent():
    built = stack()
    value = splat.evaluate("flow * wetness", built)
    assert value.device == built.data.device
