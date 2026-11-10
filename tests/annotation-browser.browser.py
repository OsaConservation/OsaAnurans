"""Exercise the real local annotation browser with isolated data and generated audio.

Run with the app Python and Playwright on PYTHONPATH; no cloud access is used.
Pass --actual to also inspect the configured local snapshot without changing it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from browser_fixtures import AUDIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from osa_anurans_app.src.annotations import aggregate_votes, save_snapshot
from osa_anurans_app.src.catalog import Catalog
from osa_anurans_app.src.config import AppConfig, default_config, load_config
from osa_anurans_app.src.records import RecordingRecord, WindowRecord


def fixture_config(directory):
    config = AppConfig(audio_roots=(directory,), annotations_csv=None, artifacts=directory)
    wav = directory / "clip.wav"
    wav.write_bytes(AUDIO)
    windows = [WindowRecord("clip-a", "recording-a", 160000, 240000, 16000)]
    windows += [WindowRecord("clip-b", "recording-a", 240000, 320000, 16000)]
    with Catalog(directory / "catalog.sqlite") as catalog:
        catalog.add_recording(RecordingRecord(
            "recording-a", directory / "forest.wav", "fixture", 16000, 320000
        ))
        for window in windows:
            catalog.connection.execute(
                "INSERT INTO windows (window_id, recording_id, start_sample, end_sample, "
                "sample_rate, policy, audio_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (window.window_id, window.recording_id, window.start_sample,
                 window.end_sample, window.sample_rate, window.policy, str(wav)),
            )
        catalog.connection.commit()

    answers = []
    for index in range(54):
        answers.append({
            "annotation_id": "repeated-id" if index < 2 else f"answer-{index}",
            "window_id": "missing-clip" if index == 53 else (
                "clip-a" if index < 2 else "clip-b"
            ),
            "query_species_id": "FROG-A" if index < 2 else "FROG-B",
            "response": "present" if index == 0 else ("absent" if index == 1 else "unsure"),
            "annotator_name": "Alice" if index < 2 else "Bob",
            "annotator_id": f"session-{index}",
            "created_at": f"2026-11-10T12:00:{53 - index:02d}Z",
            "submission_id": f"submission-{index}",
        })
    snapshot = save_snapshot(
        directory / "annotations" / "fixture", aggregate_votes(answers), "browser-fixture",
        raw_answers=answers, windows=windows,
    )
    (directory / "latest.json").write_text(json.dumps({"snapshot": str(snapshot)}))
    return config


def fingerprint(config):
    paths = [config.artifacts / "catalog.sqlite", config.artifacts / "latest.json"]
    latest = config.artifacts / "latest.json"
    if latest.is_file():
        snapshot = Path(json.loads(latest.read_text())["snapshot"])
        paths += [snapshot / "answers.jsonl", snapshot / "snapshot.json"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", action="store_true")
    args = parser.parse_args()
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="annotation-browser-", dir=ROOT / ".tmp") as name:
        config = fixture_config(Path(name))
        # The server import is delayed so fixture helpers stay usable independently.
        from osa_anurans_app.src.annotation_browser_server import make_server
        from osa_anurans_app.src.annotation_browser import load_browser_data

        before = fingerprint(config)
        server = make_server(load_browser_data(config), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                try:
                    exercise(browser, server)
                    if args.actual:
                        actual_smoke(browser, make_server)
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        assert fingerprint(config) == before, "Browsing changed annotation/catalog data"
    print("PASS: isolated annotation browser playback, filters, pagination, spectrograms; no cloud writes")


def local_page(browser, server):
    base = f"http://127.0.0.1:{server.server_port}"
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors, external = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def local_only(route):
        parsed = urlparse(route.request.url)
        if parsed.hostname == "127.0.0.1" and parsed.port == server.server_port:
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    page.route("**/*", local_only)
    page.goto(base)
    return page, errors, external


def exercise(browser, server):
    page, errors, external = local_page(browser, server)
    try:
        rows = page.locator("#annotation-rows tr")
        expect(rows).to_have_count(50)
        expect(page.locator("#result-count")).to_have_text("54 annotations \u00b7 3 clips")
        first = rows.first.locator("button")
        first.focus()
        first.press("Enter")
        expect(first).to_be_focused()
        expect(rows.first).to_have_attribute("aria-selected", "true")
        expect(page.locator("#spectrogram img")).to_be_visible(timeout=30000)
        audio = page.locator("#clip-audio")
        page.wait_for_function("document.querySelector('#clip-audio').readyState >= 2")
        assert audio.evaluate("node => node.duration") == 5
        assert "10.0" in page.locator("#annotation-details").inner_text()
        audio.evaluate("node => node.play()")
        page.wait_for_function("document.querySelector('#clip-audio').currentTime > 0.05")
        audio.evaluate("node => node.currentTime = 2")
        page.wait_for_function("document.querySelector('#clip-audio').currentTime >= 2")
        assert not audio.evaluate("node => node.paused")
        source = audio.get_attribute("src")
        rows.nth(1).locator("button").click()
        expect(audio).to_have_attribute("src", source)
        expect(page.locator("#selected-title")).to_contain_text("absent")
        assert not audio.evaluate("node => node.paused"), "Same clip selection restarted playback"

        page.locator("#next-page").click()
        expect(rows).to_have_count(4)
        expect(page.locator("#page-status")).to_have_text("Page 2 of 2")
        expect(audio).to_be_hidden()
        assert audio.get_attribute("src") is None
        assert audio.evaluate("node => node.paused")
        rows.last.locator("button").click()
        expect(page.locator("#media-status")).to_contain_text("unavailable")
        expect(audio).to_be_hidden()
        expect(page.locator("#spectrogram img")).to_have_count(0)

        page.locator("#annotator-filter").select_option("Alice")
        expect(rows).to_have_count(2)
        expect(page.locator("#result-count")).to_have_text("2 annotations \u00b7 1 clips")
        expect(page.locator("#page-status")).to_have_text("Page 1 of 1")
        assert page.locator('#annotation-rows tr[aria-selected="true"]').count() == 0
        # Repeated annotation IDs and names with different session IDs stay separate.
        assert rows.nth(0).get_attribute("data-row-id") != rows.nth(1).get_attribute("data-row-id")
        page.locator("#response-filter").select_option("absent")
        expect(rows).to_have_count(1)
        expect(rows.first).to_contain_text("absent")
        page.locator("#response-filter").select_option("")
        page.locator("#species-filter").select_option("FROG-B")
        expect(rows).to_have_count(0)
        page.locator("#annotator-filter").select_option("")
        page.locator("#species-filter").select_option("")
        page.locator("#search").fill("missing-clip")
        expect(rows).to_have_count(1)
        page.locator("#search").fill("forest.wav")
        expect(page.locator("#result-count")).to_have_text("53 annotations \u00b7 2 clips")
        page.locator("#search").fill("")

        # Queue a stale generation then switch clips before it can paint.
        page.evaluate("""() => {
            document.querySelectorAll('.select-annotation')[0].click();
            document.querySelectorAll('.select-annotation')[2].click();
        }""")
        expect(page.locator("#selected-title")).to_contain_text("FROG-B")
        expect(page.locator("#spectrogram img")).to_have_attribute(
            "alt", "Spectrogram of clip clip-b, 0\u201312 kHz", timeout=30000)
        page.locator("#annotator-filter").select_option("Alice")
        expect(audio).to_be_hidden()
        expect(page.locator("#spectrogram img")).to_have_count(0)
        page.locator("#annotator-filter").select_option("")
        rows.first.locator("button").click()
        expect(page.locator("#spectrogram img")).to_be_visible(timeout=30000)
        evidence = ROOT / ".tmp" / "annotation-browser"
        evidence.mkdir(exist_ok=True)
        page.screenshot(path=str(evidence / "fixture-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(evidence / "fixture-mobile.png"), full_page=True)
        assert not errors, errors
        assert not external, external
    finally:
        page.close()


def actual_smoke(browser, make_server):
    from osa_anurans_app.src.annotation_browser import load_browser_data

    config = load_config(default_config())
    before = fingerprint(config)
    data = load_browser_data(config)
    server = make_server(data, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page, errors, external = local_page(browser, server)
        try:
            expected = len(data.payload["annotations"])
            expect(page.locator("#result-count")).to_contain_text(f"{expected} annotations")
            playable = next(row for row in data.payload["annotations"] if row["audio_url"])
            page.locator("#search").fill(playable["window_id"])
            page.locator(".select-annotation").first.click()
            expect(page.locator("#spectrogram img")).to_be_visible(timeout=30000)
            page.wait_for_function("document.querySelector('#clip-audio').readyState >= 2")
            page.locator("#clip-audio").evaluate("node => node.play()")
            page.wait_for_function("document.querySelector('#clip-audio').currentTime > 0.05")
            page.screenshot(path=str(ROOT / ".tmp/annotation-browser/actual-local-snapshot.png"), full_page=True)
            assert not errors, errors
            assert not external, external
            print(f"PASS: actual local snapshot {data.payload['snapshot']['path']}: {expected} rows; sampled real audio and spectrogram")
        finally:
            page.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert fingerprint(config) == before, "Actual local annotation/catalog files changed"


if __name__ == "__main__":
    main()
