# Image node (Mac)

Concept images for the model node (`assets/model_node.py`): [mflux](https://github.com/filipstrand/mflux)
(MIT) running Z-Image Turbo (Apache 2.0) on Apple Silicon, served as an OpenAI-style
`POST /v1/images/generations`. Ollama on the same Mac keeps serving the LLM/VLM on 11434.

## Setup

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
git clone https://github.com/Yanick-Salzmann/assetforge.git ~/assetforge
cd ~/assetforge
uv run tools/image_node/server.py                   # first run fetches deps and the weights (several GB)
```

`uv run` reads the inline dependency block in `server.py`, so no venv or `pip`. Weights go to the
Hugging Face cache (`~/.cache/huggingface`); the first request is slow while the model loads.

Optional settings: `IMAGE_NODE_PORT` (8189), `IMAGE_NODE_HOST` (0.0.0.0), `IMAGE_NODE_QUANTIZE`
(8; 4 is smaller and faster), `IMAGE_NODE_STEPS` (9), `IMAGE_NODE_TOKEN` (require a bearer token).

Smoke test on the Mac:

```sh
curl -s localhost:8189/v1/images/generations -H 'content-type: application/json' \
  -d '{"prompt":"a wild boar, side view, plain grey background","size":"1024x1024","seed":42}' \
  | python3 -c 'import sys,json,base64; open("boar.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## Client side (4070 box)

```
ASSETFORGE_IMAGE_URL=http://192.168.1.114:8189/v1
ASSETFORGE_IMAGE_TOKEN=<IMAGE_NODE_TOKEN, if set>
```

## Keep it running

```sh
nohup uv run tools/image_node/server.py > ~/image_node.log 2>&1 &
```

The space Ollama's unused `x/z-image-turbo` pull takes is freed with `ollama rm x/z-image-turbo`.
