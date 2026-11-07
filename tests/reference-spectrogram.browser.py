"""Reference spectrogram fallback with real Web Audio and canvas, entirely local."""

from collections import Counter
from functools import partial
from http.server import ThreadingHTTPServer
import io
from pathlib import Path
import re
import threading
from urllib.parse import urlparse
import wave

from playwright.sync_api import expect, sync_playwright
from browser_fixtures import AUDIO, IMAGE, QuietHandler


root = Path(__file__).resolve().parents[1]
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
with wave.open(io.BytesIO(AUDIO), 'rb') as source:
    parameters, frames = source.getparams(), source.readframes(source.getnframes())
output = io.BytesIO()
with wave.open(output, 'wb') as target:
    target.setparams(parameters)
    target.writeframes(frames * 2)
long_audio = output.getvalue()

try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 390, 'height': 844})
        errors, fetches = [], Counter()
        failed_once = set()
        pending = {}
        page.on('pageerror', lambda error: errors.append(str(error)))

        def media(route):
            name = urlparse(route.request.url).path.rsplit('/', 1)[-1]
            if name.endswith('.png'):
                route.fulfill(content_type='image/png', body=IMAGE)
                return
            if route.request.resource_type == 'fetch':
                fetches[name] += 1
                if name == 'slow.wav':
                    pending[name] = route
                    return
                if name == 'retry.wav' and name not in failed_once:
                    failed_once.add(name)
                    route.fulfill(status=503, body='Temporary failure')
                    return
            route.fulfill(content_type='audio/wav', body=long_audio if name == 'full.wav' else AUDIO)

        page.route(base + '/generated-test/**', media)
        page.goto(base + '/docs/annotate.html')
        page.evaluate('''() => {
            document.querySelector('#annotator-login').hidden = true;
            document.querySelector('#annotator-workspace').hidden = false;
            window.timeTicks = [];
            const original = CanvasRenderingContext2D.prototype.fillText;
            CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
                if (this.textBaseline === 'top' && Number.isFinite(Number(text)))
                    window.timeTicks.push(Number(text));
                return original.call(this, text, ...args);
            };
        }''')
        page.evaluate('''async (base) => {
            const {renderReferences} = await import('./reference-media.mjs');
            const clip = (name, picture = null) => ({
                audioUrl: base + '/generated-test/' + name, spectrogramUrl: picture,
            });
            window.testReferences = {
                target: {id:'T',scientificName:'Target frog',commonName:'Target',clips:[
                    clip('full.wav'), clip('provided.wav', base + '/generated-test/provided.png'),
                ]},
                comparisons: [{id:'A',scientificName:'Comparison frog',commonName:'Other',clips:[
                    clip('comparison.wav'), clip('provided.wav', base + '/generated-test/provided.png'),
                ]}],
            };
            renderReferences(window.testReferences);
        }''', base)

        # Mobile targets generate even while their spectrogram details are collapsed.
        target = page.locator('#frog-card')
        details = target.locator('.reference-spectrogram')
        expect(details).not_to_have_attribute('open', '')
        generated = target.locator('img.generated-spectrogram')
        expect(generated).to_have_count(1, timeout=30000)
        expect(generated).to_have_attribute('src', re.compile('^data:image/png'))
        assert generated.evaluate('(image) => image.naturalWidth') == 1100
        assert generated.evaluate('(image) => image.naturalHeight') == 560
        assert page.evaluate('Math.max(...window.timeTicks)') == 10
        expect(target.locator('.target-media-status')).to_be_hidden()
        assert fetches['full.wav'] == 1
        assert fetches['comparison.wav'] == 0

        # Generated references support the same enlargement dialog as supplied images.
        details.locator('summary').click()
        generated.click()
        expect(page.locator('.spectrogram-preview')).to_be_visible()
        page.keyboard.press('Escape')

        # An image URL always takes precedence; navigation reuses generated images.
        target.get_by_role('button', name='Next reference').click()
        expect(target.locator('img')).to_have_attribute('src', base + '/generated-test/provided.png')
        assert fetches['provided.wav'] == 0
        target.get_by_role('button', name='Previous reference').click()
        expect(target.locator('img.generated-spectrogram')).to_have_count(1)
        assert fetches['full.wav'] == 1

        page.locator('.other-references').evaluate('(node) => node.open = true')
        comparison = page.locator('.frog-carousel .frog-card')
        comparison.scroll_into_view_if_needed()
        expect(comparison.locator('img.generated-spectrogram')).to_be_visible(timeout=30000)
        assert fetches['comparison.wav'] == 1
        comparison.get_by_role('button', name='Next reference').click()
        expect(comparison.locator('img')).to_have_attribute('src', base + '/generated-test/provided.png')
        comparison.get_by_role('button', name='Previous reference').click()
        expect(comparison.locator('img.generated-spectrogram')).to_be_visible()
        assert fetches['comparison.wav'] == 1

        # Full-duration references and explicit candidate intervals share the renderer.
        page.evaluate('''async (base) => {
            window.timeTicks = [];
            const {generateSpectrogram} = await import('./local-spectrogram.mjs?v=reference-spectrogram-1');
            await generateSpectrogram({audioUrl:base+'/generated-test/full.wav',offsets:[1,3]});
        }''', base)
        assert page.evaluate('Math.max(...window.timeTicks)') == 2
        assert fetches['full.wav'] == 1

        # A failed spectrogram download must not disable native audio playback.
        page.evaluate('''async (base) => {
            const {renderReferences} = await import('./reference-media.mjs');
            renderReferences({target:{id:'FAIL',scientificName:'Retry frog',commonName:'Retry',
                clips:[{audioUrl:base+'/generated-test/retry.wav',spectrogramUrl:null}]},comparisons:[]});
        }''', base)
        target.scroll_into_view_if_needed()
        target.locator('summary').click()
        retry = target.get_by_role('button', name='Retry spectrogram')
        expect(retry).to_be_visible(timeout=30000)
        expect(target.locator('.target-media-status')).to_contain_text('Spectrogram unavailable')
        page.wait_for_function("document.querySelector('#frog-card audio').readyState >= 3")
        retry.click()
        expect(target.locator('img.generated-spectrogram')).to_be_visible(timeout=30000)
        expect(target.locator('.target-media-status')).to_be_hidden()
        assert fetches['retry.wav'] == 2

        # Rapid navigation can return to a clip whose first queued card was removed.
        with page.expect_request(lambda request: request.resource_type == 'fetch' and request.url.endswith('/slow.wav')):
            page.evaluate('''async (base) => {
                const {renderReferences} = await import('./reference-media.mjs');
                renderReferences({target:{id:'NAV',scientificName:'Navigation frog',commonName:'Navigation',
                    clips:['slow.wav','queued.wav','third.wav'].map(name => ({
                        audioUrl:base+'/generated-test/'+name,spectrogramUrl:null,
                    }))},comparisons:[]});
            }''', base)
        target.get_by_role('button', name='Next reference').click()
        target.get_by_role('button', name='Next reference').click()
        target.get_by_role('button', name='Previous reference').click()
        pending.pop('slow.wav').fulfill(content_type='audio/wav', body=AUDIO)
        expect(target.locator('img.generated-spectrogram')).to_have_count(1, timeout=30000)
        assert fetches['queued.wav'] == 1
        assert fetches['third.wav'] == 0
        expect(target.get_by_role('button', name='Retry spectrogram')).to_have_count(0)

        for width in (1280, 390):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root / '.tmp/reference-generated-spectrogram.png'))
        assert not errors, errors
        browser.close()
        print('PASS: full-reference and candidate bounds, collapsed mobile target, comparison visibility, image precedence, cache and rapid queued navigation, preview, failure/retry, audio readiness, responsive layout; no cloud writes')
finally:
    server.shutdown()
    server.server_close()
