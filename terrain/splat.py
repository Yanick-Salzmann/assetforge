from __future__ import annotations

import ast
import operator
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch
from PIL import Image

from library import materials as material_library
from terrain import synth
from terrain.config import (
    BIOMES_DIR,
    CHANNEL_NAMES,
    MAX_SPLAT_LAYERS,
    SPLAT_LAYERS_PER_TEXTURE,
    MapConfigError,
)

_WEIGHT_FLOOR = 1e-12

Value = torch.Tensor | float


class SplatRuleError(MapConfigError):
    """Raised when a layer weight expression cannot be parsed or evaluated."""


def _pair(left: Value, right: Value) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left, right
    if torch.is_tensor(left):
        return left, torch.full_like(left, float(right))
    return torch.full_like(right, float(left)), right


def _minimum(left: Value, right: Value) -> Value:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.minimum(*_pair(left, right))
    return min(left, right)


def _maximum(left: Value, right: Value) -> Value:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.maximum(*_pair(left, right))
    return max(left, right)


def _mask(value: Any) -> Value:
    if torch.is_tensor(value):
        return value.to(dtype=torch.float32)
    return 1.0 if value else 0.0


def _clamp(value: Value, low: Value = 0.0, high: Value = 1.0) -> Value:
    return _minimum(_maximum(value, low), high)


def _smoothstep(value: Value, low: Value, high: Value) -> Value:
    if not torch.is_tensor(low) and not torch.is_tensor(high) and low == high:
        raise SplatRuleError(f"smoothstep edges must differ, both are {low}")
    ramp = _clamp((value - low) / (high - low))
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _step(value: Value) -> Value:
    if torch.is_tensor(value):
        return (value > 0.0).to(dtype=torch.float32)
    return 1.0 if value > 0.0 else 0.0


def _fold(call: Callable[[Value, Value], Value], values: tuple[Value, ...]) -> Value:
    result = values[0]
    for value in values[1:]:
        result = call(result, value)
    return result


def _min_call(*values: Value) -> Value:
    return _fold(_minimum, values)


def _max_call(*values: Value) -> Value:
    return _fold(_maximum, values)


def _pow_call(base: Value, exponent: Value) -> Value:
    return base**exponent


@dataclass(frozen=True)
class _Function:
    call: Callable[..., Value]
    lowest: int
    highest: int | None


FUNCTIONS: dict[str, _Function] = {
    "smoothstep": _Function(_smoothstep, 3, 3),
    "step": _Function(_step, 1, 1),
    "clamp": _Function(_clamp, 1, 3),
    "min": _Function(_min_call, 2, None),
    "max": _Function(_max_call, 2, None),
    "pow": _Function(_pow_call, 2, 2),
}

FUNCTION_NAMES = tuple(sorted(FUNCTIONS))

_BINARY: dict[type, Callable[[Value, Value], Value]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY: dict[type, Callable[[Value], Value]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_COMPARE: dict[type, Callable[[Value, Value], Any]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _describe(node: ast.AST) -> str:
    return ast.unparse(node)


def _arity(function: _Function) -> str:
    if function.highest is None:
        return f"at least {function.lowest} arguments"
    if function.highest == function.lowest:
        return f"exactly {function.lowest} arguments"
    return f"{function.lowest} to {function.highest} arguments"


def _validate(node: ast.expr, names: set[str]) -> None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SplatRuleError(f"only numeric literals are allowed, got {node.value!r}")
        return
    if isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            raise SplatRuleError(f"{node.id} is a helper function and must be called")
        if node.id not in CHANNEL_NAMES:
            raise SplatRuleError(
                f"unknown channel {node.id!r}; expected one of {', '.join(CHANNEL_NAMES)}"
            )
        names.add(node.id)
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise SplatRuleError(f"operator {_describe(node)} is not allowed")
        _validate(node.left, names)
        _validate(node.right, names)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not) and type(node.op) not in _UNARY:
            raise SplatRuleError(f"operator {_describe(node)} is not allowed")
        _validate(node.operand, names)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate(value, names)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise SplatRuleError(f"comparison {type(op).__name__.lower()} is not allowed")
        _validate(node.left, names)
        for comparator in node.comparators:
            _validate(comparator, names)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SplatRuleError(f"{_describe(node.func)} is not a helper function")
        function = FUNCTIONS.get(node.func.id)
        if function is None:
            raise SplatRuleError(
                f"unknown function {node.func.id!r}; expected one of {', '.join(FUNCTION_NAMES)}"
            )
        if node.keywords:
            raise SplatRuleError(f"{node.func.id} takes positional arguments only")
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                raise SplatRuleError(f"{node.func.id} does not accept a starred argument")
        count = len(node.args)
        if count < function.lowest or (function.highest is not None and count > function.highest):
            raise SplatRuleError(f"{node.func.id} takes {_arity(function)}, got {count}")
        for argument in node.args:
            _validate(argument, names)
        return
    raise SplatRuleError(f"{_describe(node)} is not allowed in a weight expression")


def _channel(name: str, channels: Any) -> torch.Tensor:
    if name not in channels:
        raise SplatRuleError(f"channel {name!r} is not available in the stack")
    return channels[name]


def _evaluate(node: ast.expr, channels: Any) -> Value:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return _channel(node.id, channels)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, channels)
        right = _evaluate(node.right, channels)
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, channels)
        if isinstance(node.op, ast.Not):
            return 1.0 - _mask(_step(operand))
        return _UNARY[type(node.op)](operand)
    if isinstance(node, ast.BoolOp):
        values = tuple(_mask(_evaluate(value, channels)) for value in node.values)
        return _fold(_minimum if isinstance(node.op, ast.And) else _maximum, values)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, channels)
        masks: list[Value] = []
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, channels)
            masks.append(_mask(_COMPARE[type(op)](left, right)))
            left = right
        return _fold(_minimum, tuple(masks))
    if isinstance(node, ast.Call):
        function = FUNCTIONS[node.func.id]
        return function.call(*(_evaluate(argument, channels) for argument in node.args))
    raise SplatRuleError(f"{_describe(node)} is not allowed in a weight expression")


def _reference(channels: Any) -> torch.Tensor:
    for name in CHANNEL_NAMES:
        if name in channels:
            return channels[name]
    raise SplatRuleError("a constant weight needs a filled channel to take its shape from")


@dataclass(frozen=True)
class Expression:
    """A parsed layer weight expression over the channel stack."""

    source: str
    channels: tuple[str, ...]
    tree: ast.expr

    def __str__(self) -> str:
        return self.source

    def evaluate(self, channels: Any, like: torch.Tensor | None = None) -> torch.Tensor:
        value = _evaluate(self.tree, channels)
        if torch.is_tensor(value):
            return value.to(dtype=torch.float32)
        reference = _reference(channels) if like is None else like
        return torch.full_like(reference, float(value), dtype=torch.float32)


def parse(source: str) -> Expression:
    """Compile a weight expression, rejecting everything outside the rule grammar."""
    text = source.strip()
    if not text:
        raise SplatRuleError("a weight expression must not be empty")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise SplatRuleError(f"cannot parse weight expression {source!r}: {error.msg}") from None
    names: set[str] = set()
    _validate(tree.body, names)
    return Expression(text, tuple(sorted(names)), tree.body)


def evaluate(
    source: str,
    channels: Mapping[str, torch.Tensor] | Any,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Parse and evaluate one weight expression against a channel stack."""
    return parse(source).evaluate(channels, like)


BIOME_KEYS = ("biome", "layer")
BIOME_META_KEYS = ("name", "description", "sharpness")
LAYER_KEYS = ("material", "weight", "tiling_m", "variation", "variation_m")
DEFAULT_SHARPNESS = 4.0
MIN_SHARPNESS = 0.1
MAX_SHARPNESS = 64.0
DEFAULT_VARIATION_M = 600.0
MIN_VARIATION_STRENGTH = 0.0
MAX_VARIATION_STRENGTH = 1.0
MIN_VARIATION_M = 32.0
MAX_VARIATION_M = 4096.0


@dataclass(frozen=True)
class Layer:
    """One material layer: what it is made of, how big it tiles, and where it wins."""

    name: str
    material: str
    weight: Expression
    tiling_m: float
    variation: float = 0.0
    variation_m: float = DEFAULT_VARIATION_M

    @property
    def channels(self) -> tuple[str, ...]:
        return self.weight.channels

    def evaluate(self, channels: Any, like: torch.Tensor | None = None) -> torch.Tensor:
        return self.weight.evaluate(channels, like)

    def as_dict(self) -> dict:
        return {
            "material": self.material,
            "weight": self.weight.source,
            "tiling_m": self.tiling_m,
            "variation": self.variation,
            "variation_m": self.variation_m,
        }


@dataclass(frozen=True)
class Biome:
    """A rule set: the ordered layers a map is painted with, in declaration order."""

    name: str
    description: str
    sharpness: float
    layers: tuple[Layer, ...]
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __contains__(self, name: str) -> bool:
        return any(layer.name == name for layer in self.layers)

    def __getitem__(self, key: int | str) -> Layer:
        if isinstance(key, int):
            return self.layers[key]
        for layer in self.layers:
            if layer.name == key:
                return layer
        raise SplatRuleError(f"unknown layer {key!r}; expected one of {', '.join(self.names())}")

    def names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.layers)

    def materials(self) -> tuple[str, ...]:
        return tuple(sorted({layer.material for layer in self.layers}))

    def channels(self) -> tuple[str, ...]:
        used = {name for layer in self.layers for name in layer.channels}
        return tuple(name for name in CHANNEL_NAMES if name in used)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sharpness": self.sharpness,
            "layers": {layer.name: layer.as_dict() for layer in self.layers},
        }


def _where(path: Path | None) -> str:
    return "biome" if path is None else str(path)


def _reject_unknown(table: Mapping[str, Any], allowed: tuple[str, ...], label: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise SplatRuleError(
            f"{label} carries unknown keys {', '.join(unknown)}; expected {', '.join(allowed)}"
        )


def _layer_tiling(name: str, entry: Mapping[str, Any], default: float) -> float:
    if "tiling_m" not in entry:
        return default
    tiling = float(entry["tiling_m"])
    if not material_library.MIN_TILING_M <= tiling <= material_library.MAX_TILING_M:
        raise SplatRuleError(
            f"layer {name!r} tiling_m {tiling} lies outside "
            f"[{material_library.MIN_TILING_M}, {material_library.MAX_TILING_M}] metres"
        )
    return tiling


def _layer_variation(name: str, entry: Mapping[str, Any]) -> tuple[float, float]:
    strength = float(entry.get("variation", 0.0))
    if not MIN_VARIATION_STRENGTH <= strength <= MAX_VARIATION_STRENGTH:
        raise SplatRuleError(
            f"layer {name!r} variation {strength} lies outside "
            f"[{MIN_VARIATION_STRENGTH}, {MAX_VARIATION_STRENGTH}]"
        )
    if "variation_m" not in entry:
        return strength, DEFAULT_VARIATION_M
    variation_m = float(entry["variation_m"])
    if not MIN_VARIATION_M <= variation_m <= MAX_VARIATION_M:
        raise SplatRuleError(
            f"layer {name!r} variation_m {variation_m} lies outside "
            f"[{MIN_VARIATION_M}, {MAX_VARIATION_M}] metres"
        )
    return strength, variation_m


def _layer(
    name: str,
    entry: Mapping[str, Any],
    index: material_library.MaterialIndex,
    label: str,
) -> Layer:
    if not isinstance(entry, Mapping):
        raise SplatRuleError(f"{label} must be a table, got {type(entry).__name__}")
    _reject_unknown(entry, LAYER_KEYS, label)
    for key in ("material", "weight"):
        if key not in entry:
            raise SplatRuleError(f"{label} is missing {key}")
    material = str(entry["material"])
    if material not in index:
        raise SplatRuleError(
            f"{label} names unknown material {material!r}; "
            f"expected one of {', '.join(index.names())}"
        )
    try:
        weight = parse(str(entry["weight"]))
    except SplatRuleError as error:
        raise SplatRuleError(f"{label}: {error}") from None
    variation, variation_m = _layer_variation(name, entry)
    return Layer(
        name,
        material,
        weight,
        _layer_tiling(name, entry, index[material].tiling_m),
        variation,
        variation_m,
    )


def _sharpness(meta: Mapping[str, Any], label: str) -> float:
    sharpness = float(meta.get("sharpness", DEFAULT_SHARPNESS))
    if not MIN_SHARPNESS <= sharpness <= MAX_SHARPNESS:
        raise SplatRuleError(
            f"{label} sharpness {sharpness} lies outside [{MIN_SHARPNESS}, {MAX_SHARPNESS}]"
        )
    return sharpness


def parse_biome(
    text: str,
    index: material_library.MaterialIndex | None = None,
    path: Path | None = None,
) -> Biome:
    """Read a biome rule set, validating every channel, material and tiling scale up front."""
    index = material_library.load() if index is None else index
    where = _where(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise SplatRuleError(f"cannot parse {where}: {error}") from None
    _reject_unknown(data, BIOME_KEYS, where)
    meta = data.get("biome", {})
    _reject_unknown(meta, BIOME_META_KEYS, f"{where} [biome]")
    declared = data.get("layer", {})
    if not declared:
        raise SplatRuleError(f"{where} declares no [layer.<name>] tables")
    if len(declared) > MAX_SPLAT_LAYERS:
        raise SplatRuleError(
            f"{where} declares {len(declared)} layers; the splat textures hold {MAX_SPLAT_LAYERS}"
        )
    layers = tuple(
        _layer(name, entry, index, f"{where} [layer.{name}]") for name, entry in declared.items()
    )
    name = str(meta.get("name", path.stem if path is not None else "biome"))
    return Biome(
        name=name,
        description=str(meta.get("description", "")),
        sharpness=_sharpness(meta, where),
        layers=layers,
        path=path,
    )


def load_biome(
    path: Path,
    index: material_library.MaterialIndex | None = None,
) -> Biome:
    if not path.is_file():
        raise SplatRuleError(f"{path} does not exist")
    return parse_biome(path.read_text(encoding="utf-8"), index, path)


def biome(
    name: str,
    root: Path = BIOMES_DIR,
    index: material_library.MaterialIndex | None = None,
) -> Biome:
    """Load a biome by name from the rule directory."""
    path = root / f"{name}.toml"
    if not path.is_file():
        known = ", ".join(available(root)) or "none"
        raise SplatRuleError(f"unknown biome {name!r}; {root} holds {known}")
    return load_biome(path, index)


def available(root: Path = BIOMES_DIR) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    return tuple(sorted(found.stem for found in root.glob("*.toml")))


SPLAT_CHANNELS = ("r", "g", "b", "a")
SPLAT_QUANTUM = 255


def _texture_name(texture: int) -> str:
    return f"splat_{texture}.png"


def assignment(biome: Biome) -> tuple[dict, ...]:
    """Which texture and channel each layer lands in, as terrain.json records it."""
    placed = []
    for position, layer in enumerate(biome):
        texture, channel = divmod(position, SPLAT_LAYERS_PER_TEXTURE)
        placed.append(
            {
                "layer": layer.name,
                "material": layer.material,
                "tiling_m": layer.tiling_m,
                "index": position,
                "texture": _texture_name(texture),
                "channel": SPLAT_CHANNELS[channel],
            }
        )
    return tuple(placed)


def texture_count(layers: int) -> int:
    return -(-layers // SPLAT_LAYERS_PER_TEXTURE)


def _variation_field(cfg: Any, label: str, feature_size_m: float, device: Any) -> torch.Tensor:
    """Low-frequency fBm centred so a layer's weight is perturbed, not biased, by it."""
    return synth.warped_fbm(
        cfg,
        synth.FractalParams(feature_size_m, octaves=3),
        warp_strength_m=feature_size_m * 0.3,
        label=f"splat_variation_{label}",
        device=device,
    )


def _apply_variation(raw: torch.Tensor, layer: Layer, channels: Any) -> torch.Tensor:
    if layer.variation <= 0.0:
        return raw
    cfg = getattr(channels, "cfg", None)
    if cfg is None:
        raise SplatRuleError(
            f"layer {layer.name!r} sets variation but the channel stack carries no cfg"
        )
    field = _variation_field(cfg, layer.name, layer.variation_m, raw.device)
    multiplier = 1.0 + layer.variation * (2.0 * field - 1.0)
    return raw * multiplier


def weights(biome: Biome, channels: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    """Raw, unnormalised layer weights as the rules state them, stacked as [H, W, L].

    Layers with a nonzero `variation` are multiplied by a low-frequency, per-layer noise
    field first, so the same material does not read as a flat repeat at map scale.
    """
    evaluated = [
        _apply_variation(layer.evaluate(channels, like).clamp_min(0.0), layer, channels)
        for layer in biome
    ]
    return torch.stack(evaluated, dim=-1)


def blend(raw: torch.Tensor, sharpness: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sharpen by raising to sharpness, then normalise so the layers sum to one per pixel.

    The power form keeps a zero weight at zero, which an exponential softmax would not:
    a rule that says a layer never appears here has to stay absent.
    """
    if not MIN_SHARPNESS <= sharpness <= MAX_SHARPNESS:
        raise SplatRuleError(
            f"sharpness {sharpness} lies outside [{MIN_SHARPNESS}, {MAX_SHARPNESS}]"
        )
    if raw.ndim != 3 or raw.shape[2] < 1:
        raise SplatRuleError(f"layer weights must be [H, W, L], got {tuple(raw.shape)}")
    sharpened = raw.clamp_min(0.0) ** sharpness
    total = sharpened.sum(dim=-1, keepdim=True)
    silent = total <= 0.0
    fallback = torch.zeros_like(sharpened)
    fallback[:, :, 0] = 1.0
    blended = torch.where(silent, fallback, sharpened / total.clamp_min(_WEIGHT_FLOOR))
    return blended, silent.squeeze(-1)


def quantise(blended: torch.Tensor) -> torch.Tensor:
    """Round to 0-255 by largest remainder, so the layers still sum to 255 at every pixel."""
    scaled = blended.to(device="cpu", dtype=torch.float32) * SPLAT_QUANTUM
    floored = scaled.floor()
    deficit = (SPLAT_QUANTUM - floored.sum(dim=-1)).round().clamp_min(0.0).to(torch.int64)
    order = torch.argsort(scaled - floored, dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    positions = torch.arange(blended.shape[2], device=order.device).expand(order.shape)
    rank.scatter_(-1, order, positions)
    return (floored.to(torch.int64) + (rank < deficit.unsqueeze(-1))).clamp(0, 255).to(torch.uint8)


def pack(blended: torch.Tensor) -> tuple[np.ndarray, ...]:
    """Four layers per RGBA texture: splat_0 holds layers 0-3, splat_1 layers 4-7."""
    quantised = quantise(blended).numpy()
    layers = quantised.shape[2]
    textures = []
    for texture in range(texture_count(layers)):
        first = texture * SPLAT_LAYERS_PER_TEXTURE
        block = quantised[:, :, first : first + SPLAT_LAYERS_PER_TEXTURE]
        if block.shape[2] < SPLAT_LAYERS_PER_TEXTURE:
            padding = SPLAT_LAYERS_PER_TEXTURE - block.shape[2]
            block = np.pad(block, ((0, 0), (0, 0), (0, padding)))
        textures.append(np.ascontiguousarray(block))
    return tuple(textures)


def coverage(biome: Biome, blended: torch.Tensor) -> tuple[dict, ...]:
    """Per layer: its mean weight and the share of pixels where it dominates."""
    dominant = blended.argmax(dim=-1)
    pixels = float(blended.shape[0] * blended.shape[1])
    table = []
    for position, layer in enumerate(biome):
        table.append(
            {
                "layer": layer.name,
                "material": layer.material,
                "mean": float(blended[:, :, position].mean()),
                "dominant": float((dominant == position).sum()) / pixels,
            }
        )
    return tuple(table)


@dataclass(frozen=True)
class SplatResult:
    """Blended layer weights, the packed RGBA textures and the map into terrain.json."""

    biome: Biome
    sharpness: float
    weights: torch.Tensor
    silent: torch.Tensor
    textures: tuple[np.ndarray, ...]
    coverage: tuple[dict, ...]

    @property
    def layers(self) -> int:
        return self.weights.shape[2]

    @property
    def silent_fraction(self) -> float:
        return float(self.silent.sum()) / float(self.silent.numel())

    def assignment(self) -> tuple[dict, ...]:
        return assignment(self.biome)

    def as_dict(self) -> dict:
        return {
            "biome": self.biome.name,
            "sharpness": self.sharpness,
            "textures": [_texture_name(n) for n in range(len(self.textures))],
            "layers": list(self.assignment()),
            "coverage": list(self.coverage),
            "silent_fraction": self.silent_fraction,
        }


def render(
    biome: Biome,
    channels: Any,
    sharpness: float | None = None,
    like: torch.Tensor | None = None,
) -> SplatResult:
    """Evaluate every layer, blend, pack, and report what each layer ended up covering."""
    used = biome.sharpness if sharpness is None else float(sharpness)
    blended, silent = blend(weights(biome, channels, like), used)
    return SplatResult(
        biome=biome,
        sharpness=used,
        weights=blended,
        silent=silent,
        textures=pack(blended),
        coverage=coverage(biome, blended),
    )


def write(result: SplatResult, out_dir: Path) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for texture, block in enumerate(result.textures):
        target = out_dir / _texture_name(texture)
        Image.fromarray(block, "RGBA").save(target, optimize=True)
        written.append(target)
    return tuple(written)
