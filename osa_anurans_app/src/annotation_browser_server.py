"""Serve saved annotations and their WAVs on the loopback interface."""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .annotation_browser import BrowserData, load_browser_data
from .config import AppConfig

APP_ROOT = Path(__file__).resolve().parents[1]
SHARED_MODULES = (
    "local-spectrogram.mjs",
    "spectrogram-renderer.mjs",
    "spectrogram-analysis.mjs",
    "observe-visible.mjs",
)


class _BrowserServer(ThreadingHTTPServer):
    # HTTPServer's address reuse permits two Windows listeners on the same port.
    allow_reuse_address = False


def _byte_range(header: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not size:
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        suffix = int(last or 0)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable range")
    return start, end


def make_server(data: BrowserData, port: int = 0) -> ThreadingHTTPServer:
    """Create a loopback server; the caller owns serving and closing it."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer between 0 and 65535")
    assets = {
        "/": (APP_ROOT / "browser" / "index.html", "text/html; charset=utf-8"),
        "/index.html": (APP_ROOT / "browser" / "index.html", "text/html; charset=utf-8"),
        "/browser.mjs": (APP_ROOT / "browser" / "browser.mjs", "text/javascript"),
        "/browser.css": (APP_ROOT / "browser" / "browser.css", "text/css"),
        **{
            f"/shared/{name}": (APP_ROOT.parent / "docs" / name, "text/javascript")
            for name in SHARED_MODULES
        },
    }
    payload = json.dumps(data.payload, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self._serve(head=False)

        def do_HEAD(self):
            self._serve(head=True)

        def _headers(self, status, content_type, length, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()

        def _serve(self, *, head):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/data":
                self._headers(200, "application/json; charset=utf-8", len(payload))
                if not head:
                    self.wfile.write(payload)
                return
            if path in assets:
                asset, content_type = assets[path]
                self._file(asset, content_type, head=head)
                return
            if path.startswith("/media/") and path.endswith(".wav"):
                window_id = path[len("/media/"):-len(".wav")]
                media = data.media_paths.get(window_id)
                if media is not None:
                    self._file(media, "audio/wav", head=head, ranges=True)
                    return
            self.send_error(404, "Annotation browser resource not found")

        def _file(self, path, content_type, *, head, ranges=False):
            try:
                stream = path.open("rb")
            except OSError:
                self.send_error(404, "Local file is unavailable")
                return
            with stream:
                stream.seek(0, 2)
                size = stream.tell()
                start, end = 0, size - 1
                extra = [("Accept-Ranges", "bytes")] if ranges else []
                status = 200
                if ranges and self.headers.get("Range"):
                    try:
                        start, end = _byte_range(self.headers["Range"], size)
                    except ValueError:
                        self._headers(416, content_type, 0, [("Content-Range", f"bytes */{size}")])
                        return
                    status = 206
                    extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
                remaining = max(0, end - start + 1)
                self._headers(status, content_type, remaining, extra)
                if head:
                    return
                stream.seek(start)
                try:
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

    try:
        return _BrowserServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise OSError(f"Cannot start annotation browser on port {port}: {exc}. Try --port.") from exc


def run_browser(config: AppConfig, snapshot: Path | None = None, port: int = 8765) -> None:
    """Load once, print a URL, and serve until Ctrl+C, closing all server resources."""
    data = load_browser_data(config, snapshot)
    with make_server(data, port) as server:
        print(f"Annotation browser: http://127.0.0.1:{server.server_port}/", flush=True)
        print("Open this URL in a browser. Ctrl+C stops the server.", flush=True)
        server.serve_forever()
