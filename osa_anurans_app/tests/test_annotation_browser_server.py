import http.client
import json
from threading import Thread

import pytest
from osa_anurans_app.src.annotation_browser import BrowserData
from osa_anurans_app.src.annotation_browser_server import make_server


@pytest.fixture
def running(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"0123456789")
    data = BrowserData({"snapshot": {}, "annotations": [{"row_id": "1"}]}, {"w1": wav})
    with make_server(data) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server, wav
        finally:
            server.shutdown()
            thread.join(timeout=3)
            assert not thread.is_alive()


def request(server, path, method="GET", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_payload_head_and_loopback(running):
    server, _ = running
    assert server.server_address[0] == "127.0.0.1"
    status, headers, body = request(server, "/api/data")
    assert status == 200
    assert json.loads(body)["annotations"] == [{"row_id": "1"}]
    assert headers["Cache-Control"] == "no-store"
    assert request(server, "/api/data", "HEAD")[2] == b""


@pytest.mark.parametrize("header, expected", [
    ("bytes=2-4", b"234"), ("bytes=7-", b"789"),
    ("bytes=-3", b"789"), ("bytes=8-100", b"89"),
])
def test_media_ranges(running, header, expected):
    server, _ = running
    status, headers, body = request(server, "/media/w1.wav", headers={"Range": header})
    assert status == 206
    assert body == expected
    assert headers["Content-Range"].endswith("/10")
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.parametrize("header", ["bytes=20-", "bytes=4-1", "bytes=-0", "bytes=0-1,4-5"])
def test_invalid_ranges(running, header):
    server, _ = running
    status, headers, body = request(server, "/media/w1.wav", headers={"Range": header})
    assert status == 416
    assert headers["Content-Range"] == "bytes */10"
    assert body == b""


def test_media_head_and_missing_after_start(running):
    server, wav = running
    assert request(server, "/media/w1.wav", "HEAD")[1]["Content-Length"] == "10"
    assert request(server, "/media/w1.wav")[2] == b"0123456789"
    wav.unlink()
    assert request(server, "/media/w1.wav")[0] == 404


@pytest.mark.parametrize("path", [
    "/media/unknown.wav", "/../README.md", "/%2e%2e/README.md",
    "/media/%2e%2e%2fclip.wav", "/shared/../annotate.js", "/catalog.sqlite",
])
def test_only_allowlisted_resources(running, path):
    server, _ = running
    assert request(server, path)[0] == 404


def test_shared_modules(running):
    server, _ = running
    assert request(server, "/shared/local-spectrogram.mjs")[0] == 200


def test_invalid_port():
    with pytest.raises(ValueError, match="Port"):
        make_server(BrowserData({}, {}), port=65536)


def test_occupied_port(running):
    server, _ = running
    with pytest.raises(OSError, match="Try --port"):
        make_server(BrowserData({}, {}), port=server.server_port)
