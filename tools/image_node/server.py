# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "mflux",
#     "fastapi>=0.115",
#     "uvicorn>=0.30",
# ]
# ///
"""Concept-image node for the Mac: mflux Z-Image Turbo behind the OpenAI-style
POST /v1/images/generations that assets/model_node.py calls.

    uv run tools/image_node/server.py

Environment: IMAGE_NODE_HOST (0.0.0.0), IMAGE_NODE_PORT (8189), IMAGE_NODE_QUANTIZE (8),
IMAGE_NODE_STEPS (9), IMAGE_NODE_TOKEN (optional bearer token the client must send).
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

MODEL_ID = "z-image-turbo"
MODEL_ALIASES = {"fast", MODEL_ID, "x/z-image-turbo", ""}
MAX_PIXELS = 1024 * 1024
SIDE_MULTIPLE = 16
MAX_STEPS = 50

HOST = os.environ.get("IMAGE_NODE_HOST", "0.0.0.0")
PORT = int(os.environ.get("IMAGE_NODE_PORT", "8189"))
QUANTIZE = int(os.environ.get("IMAGE_NODE_QUANTIZE", "8"))
DEFAULT_STEPS = int(os.environ.get("IMAGE_NODE_STEPS", "9"))
TOKEN = os.environ.get("IMAGE_NODE_TOKEN", "").strip()


class GenerationRequest(BaseModel):
    prompt: str
    model: str = MODEL_ID
    negative_prompt: str = ""
    size: str = "1024x1024"
    seed: int = 0
    n: int = 1
    steps: int | None = None
    response_format: str = "b64_json"


def parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError:
        raise HTTPException(400, f"size {size!r} is not WIDTHxHEIGHT")
    if width <= 0 or height <= 0 or width % SIDE_MULTIPLE or height % SIDE_MULTIPLE:
        raise HTTPException(400, f"size {size}: sides must be positive multiples of {SIDE_MULTIPLE}")
    if width * height > MAX_PIXELS:
        raise HTTPException(400, f"size {size}: more than {MAX_PIXELS} pixels")
    return width, height


class Generator:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from mflux.models.z_image import ZImageTurbo

            self._model = ZImageTurbo(quantize=QUANTIZE)
        return self._model

    def png(self, prompt: str, seed: int, width: int, height: int, steps: int) -> bytes:
        with self._lock:
            image = self._load().generate_image(
                prompt=prompt,
                seed=seed,
                num_inference_steps=steps,
                width=width,
                height=height,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "image.png"
                image.save(str(path))
                return path.read_bytes()


app = FastAPI(title="assetforge image node")
generator = Generator()


def _authorise(authorization: str | None) -> None:
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "missing or wrong bearer token")


@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)) -> dict:
    _authorise(authorization)
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mflux"}]}


@app.post("/v1/images/generations")
def generate(request: GenerationRequest, authorization: str | None = Header(default=None)) -> dict:
    _authorise(authorization)
    if request.model not in MODEL_ALIASES:
        raise HTTPException(400, f"model {request.model!r} is not served; use {MODEL_ID}")
    if request.n != 1:
        raise HTTPException(400, "n must be 1")
    if request.response_format != "b64_json":
        raise HTTPException(400, "response_format must be b64_json")
    width, height = parse_size(request.size)
    steps = request.steps or DEFAULT_STEPS
    if not 1 <= steps <= MAX_STEPS:
        raise HTTPException(400, f"steps must be in [1, {MAX_STEPS}]")
    png = generator.png(request.prompt, request.seed, width, height, steps)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(png).decode("ascii"), "seed": request.seed}],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
