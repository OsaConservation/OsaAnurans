"""Exercise delayed target-reference requests and media locally, without cloud writes."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright
from browser_fixtures import AUDIO, IMAGE, QuietHandler


root = Path(__file__).resolve().parents[1]
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
pending = {}
species = {'species_id': 'FROG', 'scientific_name': 'Reference frog', 'common_name': 'Test frog',
           'annotation_counts': {'positive': 0, 'negative': 0, 'unsure': 0, 'total': 0}}
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 390, 'height': 844})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def api(route):
            path = urlparse(route.request.url).path
            if route.request.method == 'OPTIONS':
                route.fulfill(status=204, headers={'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'})
                return
            if path == '/auth/login':
                data = {'access_token': 'test', 'token_type': 'Bearer', 'expires_at': '2099-01-01T00:00:00Z'}
            elif path == '/species':
                data = {'species': [species]}
            elif path == '/references':
                pending['references'] = route
                return
            elif path == '/candidates':
                data = {'candidates': [], 'next_cursor': None, 'urls_expire_at': '2099-01-01T00:00:00Z'}
            else:
                raise AssertionError('Unexpected API route: ' + path)
            route.fulfill(json=data, headers={'Access-Control-Allow-Origin': '*'})

        page.route('https://annotationservice-167659324212.us-south1.run.app/**', api)
        page.route('**/test-reference.png', lambda route: pending.update(image=route))
        page.route('**/test-reference.wav', lambda route: pending.update(audio=route))
        page.goto(base + '/docs/annotate.html')
        page.evaluate("() => import('./annotate.js?v=species-counts-1')")
        page.locator('#annotator-password').fill('local-test-password')
        page.locator('#login-button').click()
        notice = page.locator('#target-reference-loading')
        expect(notice).to_be_visible()
        expect(notice).to_contain_text('Loading target reference for Reference frog')
        expect(page.locator('#frog-card')).to_be_hidden()
        page.wait_for_function("document.querySelector('#target-reference-loading').hidden === false")
        assert 'references' in pending
        pending.pop('references').fulfill(json={
            'references': {'FROG': [{'reference_id': 'test-ref', 'audio_url': base + '/test-reference.wav',
                                    'spectrogram_url': base + '/test-reference.png'}]},
            'urls_expire_at': '2099-01-01T00:00:00Z',
        }, headers={'Access-Control-Allow-Origin': '*'})
        feedback = page.locator('#frog-card .target-media-status')
        expect(feedback).to_be_visible()
        expect(feedback).to_contain_text('Loading target reference')
        expect(feedback).to_contain_text('Audio loading')
        expect(feedback).to_contain_text('Spectrogram loading')
        expect(notice).to_be_hidden()
        expect(page.locator('#frog-card .reference-spectrogram')).not_to_have_attribute('open', '')
        # Audio can become playable while the mobile spectrogram remains collapsed/loading.
        page.wait_for_function("document.querySelector('#frog-card audio').currentSrc !== ''")
        assert 'audio' in pending
        pending.pop('audio').fulfill(content_type='audio/wav', body=AUDIO)
        expect(feedback).to_contain_text('Audio ready to play', timeout=15000)
        expect(feedback).to_contain_text('Spectrogram loading')
        assert 'image' in pending
        pending.pop('image').fulfill(content_type='image/png', body=IMAGE)
        expect(feedback).to_be_hidden()
        expect(page.locator('#frog-card .audio-clip')).to_have_attribute('aria-busy', 'false')
        page.locator('#frog-card audio').dispatch_event('waiting')
        expect(feedback).to_be_visible()
        expect(feedback).to_contain_text('Audio loading')
        page.locator('#frog-card audio').dispatch_event('canplay')
        expect(feedback).to_be_hidden()
        page.locator('#frog-card audio').dispatch_event('error')
        expect(feedback).to_be_visible()
        expect(feedback).to_contain_text('Target reference needs attention')
        expect(feedback).to_contain_text('Audio unavailable')
        expect(feedback).to_contain_text('Spectrogram ready')
        expect(page.locator('#frog-card .audio-clip')).to_have_attribute('aria-busy', 'false')
        page.locator('#frog-card img').dispatch_event('error')
        expect(feedback).to_contain_text('Spectrogram unavailable')
        assert not errors, errors
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        browser.close()
        print('PASS: delayed reference API, stale card hidden, separate audio/image readiness, mobile collapsed image status, buffering, audio/image errors; no page errors or cloud writes')
finally:
    server.shutdown()
    server.server_close()
