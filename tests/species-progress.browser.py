"""Species switches retain batches, drafts, queue cursors, and expired-media identity."""
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright

from browser_fixtures import QuietHandler


root = Path(__file__).resolve().parents[1]
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
reads, refreshes, submissions = [], [], []


def candidate(species, index):
    return dict(candidate_id=f'{species}-{index}', window_id=f'{species}-window-{index}',
                query_species_id=species, recording_id='recording', score=index / 29,
                created_at='2026-01-01T00:00:00Z', start_seconds=0, end_seconds=5,
                recording_start_seconds=index * 5, recording_end_seconds=index * 5 + 5,
                audio_url=base + '/download0.wav', spectrogram_url=base + '/output0.png')


def api(route):
    parsed = urlparse(route.request.url)
    query = parse_qs(parsed.query)
    headers = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}
    if route.request.method == 'OPTIONS':
        route.fulfill(status=204, headers=headers)
        return
    if parsed.path == '/auth/login':
        data = dict(access_token='test', token_type='Bearer', expires_at='2099-01-01T00:00:00Z')
    elif parsed.path == '/species':
        data = {'species': [dict(species_id=s, scientific_name=f'Frog {s}', common_name=s)
                            for s in ('A', 'B')]}
    elif parsed.path == '/references':
        data = {'references': {s: [] for s in ('A', 'B')}, 'urls_expire_at': '2099-01-01T00:00:00Z'}
    elif parsed.path == '/candidates':
        species = query['species_id'][0]
        if 'candidate_id' in query:
            ids = query['candidate_id']
            refreshes.append(ids)
            items = [candidate(species, int(value.split('-')[-1])) for value in ids]
            cursor = None
        else:
            start = int(query.get('cursor', ['0'])[0])
            reads.append((species, start))
            end = min(start + 20, 30)
            items = [candidate(species, i) for i in range(start, end)]
            cursor = str(end) if end < 30 else None
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
        data = dict(candidates=items, next_cursor=cursor, urls_expire_at=expiry)
    elif parsed.path == '/annotations':
        payload = route.request.post_data_json
        submissions.append(payload)
        data = dict(submission_id=payload['submission_id'], accepted_count=len(payload['annotations']))
    else:
        raise AssertionError(parsed.path)
    route.fulfill(json=data, headers=headers)


try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('https://annotationservice-167659324212.us-south1.run.app/**', api)
        page.goto(base + '/docs/annotate.html')
        page.locator('#annotator-password').fill('test-password')
        page.locator('#login-button').click()
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-0')
        page.locator('#annotator-name').fill('Local test')

        # Submit two complete batches, then leave a draft on the third.
        for start in (0, 5):
            for radio in page.locator('.results-list input[value=yes]').all():
                radio.check()
            page.locator('#sendButton').click()
            expect(page.locator('#status')).to_contain_text('submitted successfully')
            page.get_by_role('button', name='Next batch', exact=True).click()
            expect(page.locator('.results-list h3').first).to_contain_text(f'A-window-{start + 5}')
        page.locator('.results-list input[value=no]').first.check()
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3').first).to_contain_text('B-window-0')
        page.locator('.results-list input[value=unsure]').first.check()
        page.locator('#scientific').select_option('A')
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-10')
        expect(page.locator('.results-list input[value=no]').first).to_be_checked()
        assert reads == [('A', 0), ('B', 0)], reads
        assert len(submissions) == 2

        # Restore B after expiry: refresh exact IDs without consuming its queue.
        page.evaluate('''() => { const now = Date.now; Date.now = () => now() + 3600000; }''')
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3').first).to_contain_text('B-window-0')
        expect(page.locator('.results-list input[value=unsure]').first).to_be_checked()
        assert refreshes == [[f'B-{i}' for i in range(5)]], refreshes
        assert len(reads) == 2

        # A's saved queue still advances from batch three through the API boundary.
        page.locator('#scientific').select_option('A')
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-10')
        page.locator('#sendButton').click()
        expect(page.locator('#status')).to_contain_text('submitted successfully')
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3').first).to_contain_text('B-window-0')
        page.locator('#scientific').select_option('A')
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-10')
        expect(page.locator('.results-list input[value=no]').first).to_be_checked()
        expect(page.locator('#sendButton')).to_be_disabled()
        for start in (15, 20, 25):
            page.get_by_role('button', name='Next batch', exact=True).click()
            expect(page.locator('.results-list h3').first).to_contain_text(f'A-window-{start}')
        expect(page.get_by_role('button', name='Next batch', exact=True)).to_be_disabled()
        assert reads == [('A', 0), ('B', 0), ('A', 20)], reads
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3').first).to_contain_text('B-window-0')
        page.locator('#scientific').select_option('A')
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-25')
        expect(page.get_by_role('button', name='Next batch', exact=True)).to_be_disabled()

        # Batch size belongs to the saved batch; switching back restores it too.
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3').first).to_contain_text('B-window-0')
        page.locator('#batch-size').fill('7')
        page.locator('#batch-size').dispatch_event('change')
        expect(page.locator('.results-list h3')).to_have_count(7)
        page.locator('#scientific').select_option('A')
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-25')
        assert page.locator('#batch-size').input_value() == '5'
        page.locator('#scientific').select_option('B')
        expect(page.locator('.results-list h3')).to_have_count(7)
        assert page.locator('#batch-size').input_value() == '7'

        page.on('dialog', lambda dialog: dialog.accept())
        page.reload()
        expect(page.locator('#annotator-login')).to_be_visible()
        page.locator('#annotator-password').fill('test-password')
        page.locator('#login-button').click()
        expect(page.locator('.results-list h3').first).to_contain_text('A-window-0')
        assert page.locator('#annotator-name').input_value() == ''
        assert not errors, errors
        browser.close()
        print('PASS: per-species batches/drafts, buffered cursor, exact expired-media refresh, exhausted queue, reload reset; no cloud writes')
finally:
    server.shutdown()
