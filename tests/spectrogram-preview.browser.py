"""Local browser acceptance check; run with Python plus Playwright and Edge."""
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import expect, sync_playwright
from browser_fixtures import QuietHandler


root = Path(__file__).resolve().parents[1]
(root / '.tmp').mkdir(exist_ok=True)
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base + '/docs/annotate.html')
        page.evaluate("() => import('./annotate.js?v=species-counts-1')")
        # Exercise the installed preview without authenticating or changing cloud data.
        page.evaluate('''() => {
            document.querySelector('#annotator-login').hidden = true;
            const workspace = document.querySelector('#annotator-workspace');
            workspace.hidden = false;
            document.querySelector('.results-list').innerHTML = `
                <li><h3>Clip 1, ID: preview-test</h3><div class="audio-clip">
                    <img alt="Candidate spectrogram" src="fake_data/output0.png">
                </div><input type="radio" id="test-answer" checked></li>`;
            document.querySelector('#frog-card').innerHTML = `
                <article class="frog-card"><h3>Reference frog</h3>
                    <img id="frog-photo" alt="Frog photo" src="fake_data/output0.png">
                    <div class="audio-clip"><img alt="Reference spectrogram" src="fake_data/output1.png"></div>
                </article>`;
        }''')
        candidate = page.locator('.results-list img').first
        dialog = page.locator('.spectrogram-preview')
        expect(candidate).to_have_attribute('role', 'button')
        page.wait_for_function("document.querySelector('.results-list img').naturalWidth > 0")
        candidate.click()
        expect(dialog).to_be_visible()
        expect(dialog.locator('h2')).to_have_text('Clip 1, ID: preview-test')
        expect(dialog.locator('img')).to_have_attribute('src', base + '/docs/fake_data/output0.png')
        expect(dialog.get_by_role('button', name='Close spectrogram preview')).to_be_focused()
        page.keyboard.press('Escape')
        expect(dialog).not_to_be_visible()
        expect(candidate).to_be_focused()
        expect(page.locator('#test-answer')).to_be_checked()
        candidate.press('Enter')
        expect(dialog).to_be_visible()
        dialog.get_by_role('button', name='Close spectrogram preview').click()
        expect(dialog).not_to_be_visible()
        candidate.press('Space')
        expect(dialog).to_be_visible()
        page.mouse.click(1, 1)
        expect(dialog).not_to_be_visible()
        reference = page.locator('#frog-card .audio-clip img')
        reference.click()
        expect(dialog.locator('h2')).to_have_text('Reference frog')
        page.keyboard.press('Escape')
        assert page.locator('#frog-photo').get_attribute('role') is None

        # Audio-only comparison references must retain a visible player width.
        page.evaluate('''async () => {
            const {sound} = await import('./reference-media.mjs');
            const card = document.createElement('article');
            card.className = 'frog-card';
            const clips = document.createElement('div');
            clips.className = 'reference-clips';
            clips.append(sound({audioUrl:'fake_data/download0.wav',spectrogramUrl:null}, 'Audio-only reference'));
            card.append(clips);
            const li = document.createElement('li');
            li.append(card);
            document.querySelector('.frog-carousel').append(li);
            document.querySelector('.other-references').open = true;
        }''')
        comparison_audio = page.locator('.frog-carousel audio')
        assert comparison_audio.bounding_box()['width'] >= 200
        page.set_viewport_size({'width': 390, 'height': 844})
        assert comparison_audio.bounding_box()['width'] >= 200
        page.set_viewport_size({'width': 1280, 'height': 900})

        # Generate an actual local spectrogram and verify dynamically inserted images.
        page.evaluate('''async () => {
            const { localSpectrogram } = await import('./local-spectrogram.mjs');
            const player = document.createElement('div');
            player.className = 'audio-clip';
            player.id = 'generated-preview-test';
            player.append(localSpectrogram({
                audioUrl: new URL('fake_data/download0.wav', document.baseURI).href,
                recordingId: 'preview-test', offsets: [0, 5],
            }, async () => true));
            document.querySelector('.results-list li').append(player);
        }''')
        generated = page.locator('#generated-preview-test img')
        page.locator('#generated-preview-test').scroll_into_view_if_needed()
        expect(generated).to_be_visible(timeout=30000)
        expect(generated).to_have_attribute('role', 'button')
        generated.click()
        expect(dialog).to_be_visible()
        assert dialog.locator('img').get_attribute('src').startswith('data:image/')
        page.set_viewport_size({'width': 390, 'height': 844})
        bounds = dialog.bounding_box()
        assert bounds['width'] <= 390 and bounds['height'] <= 844
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root / '.tmp/spectrogram-preview-mobile.png'))
        page.keyboard.press('Escape')
        expect(page.locator('#test-answer')).to_be_checked()

        # An expiring image must show a useful message, and logout closes the preview.
        page.route('**/fake_data/output0.png?expired', lambda route: route.abort())
        candidate.click()
        dialog.locator('img').evaluate("node => { node.src = 'fake_data/output0.png?expired'; }")
        expect(dialog.get_by_role('status')).to_contain_text('could not be loaded')
        expect(dialog.get_by_role('status')).to_be_visible()
        page.evaluate("document.querySelector('#logout-button').click()")
        expect(dialog).not_to_be_visible()
        expect(page.locator('#annotator-workspace')).not_to_be_visible()
        assert not errors, errors
        browser.close()
        print('PASS: reference, candidate and generated previews; keyboard, close button, backdrop, focus restoration, answers preserved, mobile, image errors, logout; no page errors')
finally:
    server.shutdown()
    server.server_close()
