"""Local reference viewer for exported assets and terrains: a stdlib HTTP server in front of a three.js page.

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
from typing import Mapping
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

from terrain import config
from terrain import manifest as terrain_manifest

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8765
ASSET_FILE_PREFIX = "/files/assets/"
TERRAIN_FILE_PREFIX = "/files/terrain/"
LIBRARY_FILE_PREFIX = "/files/library/"
RAW_IMAGE_PREFIX = "/api/raw/terrain/"
MATERIAL_INDEX_NAME = "materials.json"
MATERIAL_ROLES = ("albedo", "normal", "roughness")
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


def _material_maps(library_root: Path, material: str, index: Mapping[str, dict]) -> dict[str, str]:
    entry = index.get(material)
    if entry is None:
        return {}
    maps: dict[str, str] = {}
    for role in MATERIAL_ROLES:
        relative = entry.get("maps", {}).get(role)
        if relative and resolve_under(library_root, relative) is not None:
            maps[role] = f"{LIBRARY_FILE_PREFIX}{relative}"
    return maps


def _material_index(library_root: Path) -> dict[str, dict]:
    path = library_root / MATERIAL_INDEX_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("materials", {})


def list_terrains(terrain_root: Path, library_root: Path) -> list[dict]:
    if not terrain_root.is_dir():
        return []
    index = _material_index(library_root)
    terrains: list[dict] = []
    for directory in sorted(path for path in terrain_root.iterdir() if path.is_dir()):
        try:
            payload = terrain_manifest.load(directory / terrain_manifest.MANIFEST_NAME)
            terrain_manifest.verify(payload, directory)
        except (terrain_manifest.ManifestError, ValueError, OSError):
            continue
        base = f"{TERRAIN_FILE_PREFIX}{directory.name}/"
        raw = f"{RAW_IMAGE_PREFIX}{directory.name}/"
        terrains.append(
            {
                "name": directory.name,
                "manifest": payload,
                "files": base,
                "raw": raw,
                "materials": {
                    layer["material"]: _material_maps(library_root, layer["material"], index)
                    for layer in payload["splat"]["layers"]
                },
            }
        )
    return terrains


def raw_image(path: Path) -> tuple[bytes, dict[str, str]]:
    """Decode a PNG to tightly packed little-endian samples so the page never relies on canvas decoding."""
    with Image.open(path) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            values = np.asarray(image).astype("<u2")
            channels = 1
            bits = 16
        elif image.mode == "L":
            values = np.asarray(image, dtype=np.uint8)
            channels = 1
            bits = 8
        else:
            values = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            channels = 4
            bits = 8
        width, height = image.size
    headers = {
        "X-Width": str(width),
        "X-Height": str(height),
        "X-Channels": str(channels),
        "X-Bits": str(bits),
    }
    return np.ascontiguousarray(values).tobytes(), headers


def resolve_under(root: Path, relative: str) -> Path | None:
    root = root.resolve()
    candidate = (root / unquote(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *args,
        asset_root: Path,
        static_root: Path,
        terrain_root: Path,
        library_root: Path,
        **kwargs,
    ) -> None:
        self.asset_root = asset_root
        self.static_root = static_root
        self.terrain_root = terrain_root
        self.library_root = library_root
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(self.static_root / "index.html")
        elif path == "/api/assets":
            self._send_json(list_assets(self.asset_root))
        elif path == "/api/terrains":
            self._send_json(list_terrains(self.terrain_root, self.library_root))
        elif path.startswith(RAW_IMAGE_PREFIX):
            self._send_raw(resolve_under(self.terrain_root, path[len(RAW_IMAGE_PREFIX):]))
        elif path.startswith(STATIC_PREFIX):
            self._send_file(resolve_under(self.static_root, path[len(STATIC_PREFIX):]))
        elif path.startswith(ASSET_FILE_PREFIX):
            self._send_file(resolve_under(self.asset_root, path[len(ASSET_FILE_PREFIX):]))
        elif path.startswith(TERRAIN_FILE_PREFIX):
            self._send_file(resolve_under(self.terrain_root, path[len(TERRAIN_FILE_PREFIX):]))
        elif path.startswith(LIBRARY_FILE_PREFIX):
            self._send_file(resolve_under(self.library_root, path[len(LIBRARY_FILE_PREFIX):]))
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

    def _send_raw(self, path: Path | None) -> None:
        if path is None or path.suffix.lower() != ".png":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body, headers = raw_image(path)
        self._send_bytes(body, "application/octet-stream", headers)

    def _send_bytes(self, body: bytes, content_type: str, headers: Mapping[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_server(
    port: int = DEFAULT_PORT,
    asset_root: Path | None = None,
    static_root: Path = STATIC_DIR,
    terrain_root: Path | None = None,
    library_root: Path | None = None,
) -> ThreadingHTTPServer:
    handler = partial(
        ViewerHandler,
        asset_root=asset_root or config.ASSET_OUT_DIR,
        static_root=static_root,
        terrain_root=terrain_root or config.TERRAIN_OUT_DIR,
        library_root=library_root or config.LIBRARY_DIR,
    )
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the assetforge glTF viewer on localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args(argv)
    server = make_server(args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"assetforge viewer at {url} serving {config.ASSET_OUT_DIR} and {config.TERRAIN_OUT_DIR}", flush=True)
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
