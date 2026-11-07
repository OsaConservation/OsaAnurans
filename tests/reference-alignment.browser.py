"""Check comparison alignment with wrapped names, mixed image sizes and clip counts."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import expect, sync_playwright
from browser_fixtures import QuietHandler

root = Path(__file__).resolve().parents[1]
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{server.server_port}/docs/annotate.html')
        page.evaluate('''async () => {
            await import('./annotate.js?v=species-counts-1');
            const {renderReferences} = await import('./reference-media.mjs');
            document.querySelector('#annotator-login').hidden = true;
            document.querySelector('#annotator-workspace').hidden = false;
            document.querySelector('.other-references').open = true;
            function picture(width, height) {
                const canvas = document.createElement('canvas');
                canvas.width = width; canvas.height = height;
                canvas.getContext('2d').fillRect(0, 0, width, height);
                return canvas.toDataURL();
            }
            const wide = {audioUrl:'fake_data/download0.wav',spectrogramUrl:picture(600,100)};
            const tall = {...wide,spectrogramUrl:picture(100,600)};
            renderReferences({target:{id:'T',scientificName:'Target',commonName:'Target',clips:[wide]},comparisons:[
                {id:'A',scientificName:'Short name',commonName:'Short',clips:[wide]},
                {id:'B',scientificName:'A much longer scientific species name that wraps',commonName:'A much longer common name that also wraps onto multiple lines',clips:[tall,wide]},
                {id:'C',scientificName:'Audio-only reference',commonName:'Another frog',clips:[{...wide,spectrogramUrl:null}]},
            ]});
        }''')

        def check_alignment():
            rows = page.locator('.frog-carousel').evaluate('''node => [...node.querySelectorAll('.frog-card')].map(card => {
                const slot = card.querySelector('.reference-spectrogram-slot').getBoundingClientRect();
                const audio = card.querySelector('audio').getBoundingClientRect();
                return {top:slot.top,height:slot.height,audioTop:audio.top,audioWidth:audio.width};
            })''')
            for row in rows:
                assert row['audioWidth'] >= 200, rows
                for key in ('top', 'height', 'audioTop'):
                    assert abs(row[key] - rows[0][key]) <= 1, rows

        for width in (1280, 390):
            page.set_viewport_size({'width': width, 'height': 900})
            check_alignment()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        expect(page.locator('.reference-spectrogram-placeholder')).to_have_text('No spectrogram available')
        page.locator('.frog-carousel .frog-card').nth(1).get_by_role('button', name='Next reference').click()
        check_alignment()
        page.locator('.frog-carousel .frog-card').nth(1).locator('img').dispatch_event('error')
        check_alignment()
        assert not errors, errors
        browser.close()
        print('PASS: wrapped names, wide/tall/missing spectrograms, single/multiple references, navigation, image errors; aligned media/audio on desktop and mobile')
finally:
    server.shutdown()
    server.server_close()
