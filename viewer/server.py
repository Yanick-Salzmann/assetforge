"""Local reference viewer for exported assets: a stdlib HTTP server in front of a three.js page.

Run with `uv run python -m viewer.server [--port 8765] [--open]`. Binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from terrain import config

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8765
ASSET_FILE_PREFIX = "/files/assets/"
STATIC_PREFIX = "/static/"

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("text/javascript", ".js")


def list_assets(asset_root: Path) -> list[dict]:
    if not asset_root.is_dir():
        return []
    assets: list[dict] = []
    for directory in sorted(path for path in asset_root.iterdir() if path.is_dir()):
        manifest_path = directory / "asset.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        glb_name = manifest.get("glb", f"{directory.name}.glb")
        if not (directory / glb_name).is_file():
            continue
        triangle_counts = manifest.get("triangle_counts", {})
        assets.append(
            {
                "name": directory.name,
                "kind": manifest.get("kind"),
                "glb": f"{ASSET_FILE_PREFIX}{directory.name}/{glb_name}",
                "lods": sorted(triangle_counts, key=lambda lod: int(lod[len("LOD"):])),
                "triangle_counts": triangle_counts,
                "bbox_min": manifest.get("bbox_min"),
                "bbox_max": manifest.get("bbox_max"),
            }
        )
    return assets


def resolve_under(root: Path, relative: str) -> Path | None:
    root = root.resolve()
    candidate = (root / unquote(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, asset_root: Path, static_root: Path, **kwargs) -> None:
        self.asset_root = asset_root
        self.static_root = static_root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(self.static_root / "index.html")
        elif path == "/api/assets":
            self._send_json(list_assets(self.asset_root))
        elif path.startswith(STATIC_PREFIX):
            self._send_file(resolve_under(self.static_root, path[len(STATIC_PREFIX):]))
        elif path.startswith(ASSET_FILE_PREFIX):
            self._send_file(resolve_under(self.asset_root, path[len(ASSET_FILE_PREFIX):]))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json")

    def _send_file(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_server(port: int = DEFAULT_PORT, asset_root: Path | None = None, static_root: Path = STATIC_DIR) -> ThreadingHTTPServer:
    handler = partial(ViewerHandler, asset_root=asset_root or config.ASSET_OUT_DIR, static_root=static_root)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the assetforge glTF viewer on localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args(argv)
    server = make_server(args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"assetforge viewer at {url} serving {config.ASSET_OUT_DIR}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
