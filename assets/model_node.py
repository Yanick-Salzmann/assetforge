from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import requests

IMAGE_URL_ENV = "ASSETFORGE_IMAGE_URL"
IMAGE_TOKEN_ENV = "ASSETFORGE_IMAGE_TOKEN"
IMAGE_TOKEN_FALLBACK_ENV = "IMAGE_GEN_TOKEN"
IMAGE_MODEL_ENV = "ASSETFORGE_IMAGE_MODEL"
CHAT_URL_ENV = "ASSETFORGE_CHAT_URL"
CHAT_TOKEN_ENV = "ASSETFORGE_CHAT_TOKEN"
CHAT_API_ENV = "ASSETFORGE_CHAT_API"
VLM_MODEL_ENV = "ASSETFORGE_VLM_MODEL"
LLM_MODEL_ENV = "ASSETFORGE_LLM_MODEL"
TIMEOUT_ENV = "ASSETFORGE_MODEL_TIMEOUT_S"

CHAT_APIS = ("openai", "ollama")
DEFAULT_CHAT_API = "openai"
DEFAULT_IMAGE_MODEL = "fast"
DEFAULT_VLM_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_LLM_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_TIMEOUT_S = 600.0
SIZE_STEP_PX = 64
MAX_IMAGE_PIXELS = 1024 * 1024
SCORE_MAX = 10

_ONLINE_MODEL = re.compile(r"(:|-)cloud$")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class ModelNodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Endpoint:
    url: str
    token: str | None = None
    api: str = DEFAULT_CHAT_API

    def headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}


@dataclass(frozen=True)
class NodeConfig:
    image: Endpoint | None = None
    chat: Endpoint | None = None
    image_model: str = DEFAULT_IMAGE_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes
    seed: int
    prompt: str
    model: str
    size: tuple[int, int]


@dataclass(frozen=True)
class ImageScore:
    scores: dict[str, int]
    overall: int
    notes: str
    raw: dict[str, Any] = field(default_factory=dict)


def _clean(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _base_url(value: str) -> str:
    return value.rstrip("/")


def load_config(environ: Mapping[str, str] | None = None) -> NodeConfig:
    env = os.environ if environ is None else environ
    image_url = _clean(env.get(IMAGE_URL_ENV))
    chat_url = _clean(env.get(CHAT_URL_ENV))
    api = (_clean(env.get(CHAT_API_ENV)) or DEFAULT_CHAT_API).lower()
    if api not in CHAT_APIS:
        raise ModelNodeError(f"{CHAT_API_ENV}={api!r}: expected one of {', '.join(CHAT_APIS)}")
    image_token = _clean(env.get(IMAGE_TOKEN_ENV)) or _clean(env.get(IMAGE_TOKEN_FALLBACK_ENV))
    image = Endpoint(_base_url(image_url), image_token) if image_url else None
    chat = Endpoint(_base_url(chat_url), _clean(env.get(CHAT_TOKEN_ENV)), api) if chat_url else None
    timeout = _clean(env.get(TIMEOUT_ENV))
    config = NodeConfig(
        image=image,
        chat=chat,
        image_model=_clean(env.get(IMAGE_MODEL_ENV)) or DEFAULT_IMAGE_MODEL,
        vlm_model=_clean(env.get(VLM_MODEL_ENV)) or DEFAULT_VLM_MODEL,
        llm_model=_clean(env.get(LLM_MODEL_ENV)) or DEFAULT_LLM_MODEL,
        timeout_s=float(timeout) if timeout else DEFAULT_TIMEOUT_S,
    )
    for model in (config.vlm_model, config.llm_model):
        require_local_model(model)
    return config


def require_local_model(model: str) -> str:
    if _ONLINE_MODEL.search(model):
        raise ModelNodeError(f"{model}: cloud-hosted models are not allowed, pick a local one")
    return model


def validate_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0 or width % SIZE_STEP_PX or height % SIZE_STEP_PX:
        raise ModelNodeError(f"size {width}x{height}: both sides must be positive multiples of 64")
    if width * height > MAX_IMAGE_PIXELS:
        raise ModelNodeError(f"size {width}x{height}: more than {MAX_IMAGE_PIXELS} pixels")
    return width, height


def image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ModelNodeError("image is not PNG, JPEG or WebP")


def data_url(data: bytes) -> str:
    return f"data:{image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def strip_reasoning(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def parse_json(text: str) -> Any:
    body = strip_reasoning(text)
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise ModelNodeError(f"model did not return JSON: {body[:200]!r}") from error


def score_schema(criteria: Sequence[str]) -> dict[str, Any]:
    if not criteria:
        raise ModelNodeError("score needs at least one criterion")
    score = {"type": "integer", "minimum": 0, "maximum": SCORE_MAX}
    properties = {name: dict(score) for name in criteria}
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": properties,
                "required": list(criteria),
                "additionalProperties": False,
            },
            "overall": dict(score),
            "notes": {"type": "string"},
        },
        "required": ["scores", "overall", "notes"],
        "additionalProperties": False,
    }


def score_prompt(brief: str, criteria: Sequence[str]) -> str:
    listed = "\n".join(f"- {name}" for name in criteria)
    return (
        f"You are screening a concept image against this brief:\n{brief}\n\n"
        f"Score each criterion from 0 (fails) to {SCORE_MAX} (fully meets it):\n{listed}\n\n"
        "Then give an overall score on the same scale and one or two sentences of notes naming "
        "the biggest problem. Answer with JSON only."
    )


def _clamp_score(value: Any) -> int:
    return max(0, min(SCORE_MAX, int(round(float(value)))))


def to_image_score(payload: Mapping[str, Any], criteria: Sequence[str]) -> ImageScore:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping):
        raise ModelNodeError(f"score reply has no scores object: {payload!r}")
    missing = [name for name in criteria if name not in raw_scores]
    if missing:
        raise ModelNodeError(f"score reply is missing {', '.join(missing)}")
    scores = {name: _clamp_score(raw_scores[name]) for name in criteria}
    return ImageScore(
        scores=scores,
        overall=_clamp_score(payload.get("overall", 0)),
        notes=str(payload.get("notes", "")).strip(),
        raw=dict(payload),
    )


class ModelNode:
    def __init__(self, config: NodeConfig | None = None, session: requests.Session | None = None):
        self.config = config if config is not None else load_config()
        self.session = session or requests.Session()

    def _image_endpoint(self) -> Endpoint:
        if self.config.image is None:
            raise ModelNodeError(f"no image endpoint: set {IMAGE_URL_ENV}")
        return self.config.image

    def _chat_endpoint(self) -> Endpoint:
        if self.config.chat is None:
            raise ModelNodeError(f"no chat endpoint: set {CHAT_URL_ENV}")
        return self.config.chat

    def _post(self, endpoint: Endpoint, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{endpoint.url}{path}"
        try:
            response = self.session.post(
                url, json=payload, headers=endpoint.headers(), timeout=self.config.timeout_s
            )
        except requests.RequestException as error:
            raise ModelNodeError(f"{url}: {error}") from error
        if response.status_code >= 400:
            raise ModelNodeError(f"{url}: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    def _fetch(self, endpoint: Endpoint, url: str) -> bytes:
        try:
            response = self.session.get(url, headers=endpoint.headers(), timeout=self.config.timeout_s)
        except requests.RequestException as error:
            raise ModelNodeError(f"{url}: {error}") from error
        if response.status_code >= 400:
            raise ModelNodeError(f"{url}: HTTP {response.status_code}")
        return response.content

    def generate_image(
        self,
        prompt: str,
        seed: int,
        size: tuple[int, int] = (1024, 1024),
        steps: int | None = None,
        negative_prompt: str = "",
        model: str | None = None,
    ) -> GeneratedImage:
        endpoint = self._image_endpoint()
        width, height = validate_size(size)
        chosen = model or self.config.image_model
        payload: dict[str, Any] = {
            "model": chosen,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "size": f"{width}x{height}",
            "seed": int(seed),
            "n": 1,
            "response_format": "b64_json",
        }
        if steps is not None:
            payload["steps"] = int(steps)
        reply = self._post(endpoint, "/images/generations", payload)
        data = reply.get("data") or []
        if not data:
            raise ModelNodeError(f"image reply has no data: {str(reply)[:200]}")
        entry = data[0]
        if entry.get("b64_json"):
            png = base64.b64decode(entry["b64_json"])
        elif entry.get("url"):
            png = self._fetch(endpoint, entry["url"])
        else:
            raise ModelNodeError("image reply carries neither b64_json nor url")
        image_mime(png)
        return GeneratedImage(
            png=png,
            seed=int(entry.get("seed", seed)),
            prompt=prompt,
            model=chosen,
            size=(width, height),
        )

    def chat(
        self,
        prompt: str,
        model: str,
        images: Sequence[bytes] = (),
        system: str | None = None,
        schema: Mapping[str, Any] | None = None,
        seed: int = 0,
        temperature: float = 0.0,
    ) -> str:
        endpoint = self._chat_endpoint()
        require_local_model(model)
        if endpoint.api == "ollama":
            payload = self._ollama_payload(prompt, model, images, system, schema, seed, temperature)
            reply = self._post(endpoint, "/api/chat", payload)
            content = (reply.get("message") or {}).get("content")
        else:
            payload = self._openai_payload(prompt, model, images, system, schema, seed, temperature)
            reply = self._post(endpoint, "/chat/completions", payload)
            choices = reply.get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content")
        if content is None:
            raise ModelNodeError(f"chat reply has no content: {str(reply)[:200]}")
        return strip_reasoning(content)

    @staticmethod
    def _openai_payload(prompt, model, images, system, schema, seed, temperature) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in images:
                parts.append({"type": "image_url", "image_url": {"url": data_url(image)}})
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": int(seed),
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": dict(schema), "strict": True},
            }
        return payload

    @staticmethod
    def _ollama_payload(prompt, model, images, system, schema, seed, temperature) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            for image in images:
                image_mime(image)
            user["images"] = [base64.b64encode(image).decode("ascii") for image in images]
        messages.append(user)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": int(seed)},
        }
        if schema is not None:
            payload["format"] = dict(schema)
        return payload

    def chat_json(
        self,
        prompt: str,
        model: str,
        schema: Mapping[str, Any],
        images: Sequence[bytes] = (),
        system: str | None = None,
        seed: int = 0,
    ) -> Any:
        text = self.chat(prompt, model, images=images, system=system, schema=schema, seed=seed)
        return parse_json(text)

    def describe(self, image: bytes, prompt: str = "Describe this image.", seed: int = 0) -> str:
        return self.chat(prompt, self.config.vlm_model, images=[image], seed=seed)

    def score_image(
        self, image: bytes, brief: str, criteria: Sequence[str], seed: int = 0
    ) -> ImageScore:
        names = list(criteria)
        payload = self.chat_json(
            score_prompt(brief, names),
            self.config.vlm_model,
            score_schema(names),
            images=[image],
            seed=seed,
        )
        return to_image_score(payload, names)

    def draft(
        self,
        prompt: str,
        system: str | None = None,
        schema: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> Any:
        if schema is None:
            return self.chat(prompt, self.config.llm_model, system=system, seed=seed)
        return self.chat_json(prompt, self.config.llm_model, schema, system=system, seed=seed)
