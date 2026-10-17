from __future__ import annotations

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from assets import model_node
from assets.model_node import Endpoint, ModelNode, ModelNodeError, NodeConfig

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


class StubNode:
    def __init__(self):
        self.requests: list[tuple[str, dict, dict]] = []
        self.chat_content = "ok"
        self.image_entry = {"b64_json": base64.b64encode(PNG).decode("ascii"), "seed": 11}
        self.status = 200
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return None

            def _reply(self, status, payload=None, body=None, content_type="application/json"):
                data = body if body is not None else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                stub.requests.append((self.path, dict(self.headers), {}))
                self._reply(200, body=PNG, content_type="image/png")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append((self.path, dict(self.headers), payload))
                if stub.status != 200:
                    self._reply(stub.status, {"error": "boom"})
                    return
                if self.path.endswith("/images/generations"):
                    self._reply(200, {"created": 1, "data": [stub.image_entry]})
                elif self.path.endswith("/chat/completions"):
                    message = {"role": "assistant", "content": stub.chat_content}
                    self._reply(200, {"choices": [{"message": message}]})
                elif self.path == "/api/chat":
                    message = {"role": "assistant", "content": stub.chat_content}
                    self._reply(200, {"message": message, "done": True})
                else:
                    self._reply(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    node = StubNode()
    yield node
    node.close()


def make_node(stub, api="openai", image_token="img-token", chat_token=None):
    config = NodeConfig(
        image=Endpoint(f"{stub.url}/v1", image_token),
        chat=Endpoint(f"{stub.url}/v1" if api == "openai" else stub.url, chat_token, api),
        timeout_s=10,
    )
    return ModelNode(config)


def test_load_config_reads_environment():
    config = model_node.load_config(
        {
            "ASSETFORGE_IMAGE_URL": "http://mac:8088/v1/",
            "IMAGE_GEN_TOKEN": "secret",
            "ASSETFORGE_CHAT_URL": "http://mac:11434",
            "ASSETFORGE_CHAT_API": "Ollama",
            "ASSETFORGE_VLM_MODEL": "qwen3.5:9b",
            "ASSETFORGE_MODEL_TIMEOUT_S": "30",
        }
    )
    assert config.image == Endpoint("http://mac:8088/v1", "secret")
    assert config.chat == Endpoint("http://mac:11434", None, "ollama")
    assert config.vlm_model == "qwen3.5:9b"
    assert config.llm_model == model_node.DEFAULT_LLM_MODEL
    assert config.timeout_s == 30.0


def test_load_config_prefers_assetforge_token_and_allows_missing_endpoints():
    config = model_node.load_config(
        {"ASSETFORGE_IMAGE_URL": "http://mac", "ASSETFORGE_IMAGE_TOKEN": "a", "IMAGE_GEN_TOKEN": "b"}
    )
    assert config.image.token == "a"
    assert config.chat is None
    assert model_node.load_config({}).image is None


def test_load_config_rejects_unknown_api_and_cloud_models():
    with pytest.raises(ModelNodeError, match="expected one of"):
        model_node.load_config({"ASSETFORGE_CHAT_API": "grpc"})
    with pytest.raises(ModelNodeError, match="cloud"):
        model_node.load_config({"ASSETFORGE_LLM_MODEL": "glm-5:cloud"})
    with pytest.raises(ModelNodeError, match="cloud"):
        model_node.require_local_model("gpt-oss:120b-cloud")


def test_missing_endpoint_names_the_variable():
    node = ModelNode(NodeConfig())
    with pytest.raises(ModelNodeError, match="ASSETFORGE_IMAGE_URL"):
        node.generate_image("pot", seed=1)
    with pytest.raises(ModelNodeError, match="ASSETFORGE_CHAT_URL"):
        node.draft("spec")


def test_validate_size():
    assert model_node.validate_size((512, 768)) == (512, 768)
    with pytest.raises(ModelNodeError):
        model_node.validate_size((500, 512))
    with pytest.raises(ModelNodeError):
        model_node.validate_size((2048, 1024))


def test_image_mime_and_data_url():
    assert model_node.image_mime(PNG) == "image/png"
    assert model_node.image_mime(JPEG) == "image/jpeg"
    assert model_node.image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert model_node.data_url(PNG).startswith("data:image/png;base64,")
    with pytest.raises(ModelNodeError):
        model_node.image_mime(b"GIF89a")


def test_parse_json_strips_reasoning_and_fences():
    assert model_node.parse_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ModelNodeError, match="JSON"):
        model_node.parse_json("not json")


def test_generate_image_round_trip(stub):
    image = make_node(stub).generate_image("clay pot", seed=11, size=(512, 512), steps=4)
    assert image.png == PNG
    assert image.seed == 11
    assert image.size == (512, 512)
    path, headers, payload = stub.requests[-1]
    assert path == "/v1/images/generations"
    assert headers["Authorization"] == "Bearer img-token"
    assert payload["size"] == "512x512"
    assert payload["seed"] == 11
    assert payload["steps"] == 4
    assert payload["response_format"] == "b64_json"


def test_generate_image_follows_url(stub):
    stub.image_entry = {"url": f"{stub.url}/files/1.png"}
    image = make_node(stub).generate_image("clay pot", seed=3)
    assert image.png == PNG
    assert image.seed == 3
    assert stub.requests[-1][0] == "/files/1.png"


def test_http_error_is_reported(stub):
    stub.status = 500
    with pytest.raises(ModelNodeError, match="HTTP 500"):
        make_node(stub).generate_image("clay pot", seed=1)


def test_openai_chat_sends_images_and_schema(stub):
    stub.chat_content = '{"colour": "red"}'
    node = make_node(stub, chat_token="chat-token")
    schema = {"type": "object", "properties": {"colour": {"type": "string"}}}
    reply = node.chat_json("colour?", "qwen3-vl:8b-instruct", schema, images=[PNG], seed=5)
    assert reply == {"colour": "red"}
    path, headers, payload = stub.requests[-1]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer chat-token"
    parts = payload["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert payload["seed"] == 5
    assert payload["temperature"] == 0.0


def test_ollama_chat_uses_native_api(stub):
    stub.chat_content = "<think>x</think>a pot"
    node = make_node(stub, api="ollama")
    reply = node.chat("what?", "qwen3-vl:8b", images=[JPEG], system="be brief")
    assert reply == "a pot"
    path, headers, payload = stub.requests[-1]
    assert path == "/api/chat"
    assert "Authorization" not in headers
    assert payload["think"] is False
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert base64.b64decode(payload["messages"][1]["images"][0]) == JPEG


def test_chat_refuses_cloud_model(stub):
    with pytest.raises(ModelNodeError, match="cloud"):
        make_node(stub).chat("hi", "kimi-k2.6:cloud")
    assert stub.requests == []


def test_score_image_round_trip(stub):
    stub.chat_content = json.dumps(
        {"scores": {"silhouette": 8, "palette": 12}, "overall": 6.6, "notes": "too saturated"}
    )
    score = make_node(stub).score_image(PNG, "a clay pot", ["silhouette", "palette"])
    assert score.scores == {"silhouette": 8, "palette": 10}
    assert score.overall == 7
    assert score.notes == "too saturated"
    schema = stub.requests[-1][2]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["scores"]["required"] == ["silhouette", "palette"]


def test_score_image_rejects_missing_criterion(stub):
    stub.chat_content = json.dumps({"scores": {"silhouette": 8}, "overall": 6, "notes": ""})
    with pytest.raises(ModelNodeError, match="palette"):
        make_node(stub).score_image(PNG, "a clay pot", ["silhouette", "palette"])


def test_draft_returns_text_or_json(stub):
    node = make_node(stub)
    stub.chat_content = "a spec"
    assert node.draft("write a spec") == "a spec"
    stub.chat_content = '{"name": "pot"}'
    assert node.draft("write a spec", schema={"type": "object"}) == {"name": "pot"}
    assert stub.requests[-1][2]["model"] == model_node.DEFAULT_LLM_MODEL


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("ASSETFORGE_IMAGE_URL") and os.environ.get("ASSETFORGE_CHAT_URL")),
    reason="model node not configured",
)
def test_live_concept_image_and_score():
    node = ModelNode()
    image = node.generate_image("a terracotta water jar, plain grey background", seed=7, size=(512, 512), steps=4)
    assert model_node.image_mime(image.png) == "image/png"
    score = node.score_image(image.png, "a terracotta water jar", ["matches_brief", "readability"])
    assert set(score.scores) == {"matches_brief", "readability"}
    assert 0 <= score.overall <= model_node.SCORE_MAX
