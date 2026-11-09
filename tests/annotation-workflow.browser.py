"""Verify opt-in counts and buffered annotation workflow locally with mocked API/media."""
from functools import partial
import hashlib
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import expect, sync_playwright
from browser_fixtures import AUDIO, IMAGE, QuietHandler

root = Path(__file__).resolve().parents[1]
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{server.server_port}'
state = {'summary': 0, 'pages': 0, 'posts': 0, 'refresh': 0, 'expire': False, 'fail': False, 'attempts': []}
seeds = set()
orders = {}

def order_for_seed(seed):
    if seed not in orders:
        def priority(index):
            bits = int.from_bytes(hashlib.sha256(f"{seed}:FROG:{index}".encode()).digest()[:8], 'big') & (2**52 - 1)
            uniform = (bits + 0.5) / 2**52
            return -math.log(uniform) * math.exp((index / 99) / 0.5)
        orders[seed] = sorted(range(100), key=lambda index: (priority(index), index))
    return orders[seed]
species = {'species_id': 'FROG', 'scientific_name': 'Reference frog', 'common_name': 'Test frog'}

def candidate(index):
    return dict(candidate_id=f'candidate-{index}', window_id=f'window-{index}', query_species_id='FROG',
                recording_id='recording', score=index / 99, created_at='2026-01-01T00:00:00Z',
                start_seconds=0, end_seconds=5, recording_start_seconds=index * 5,
                recording_end_seconds=index * 5 + 5, audio_url=base + f'/clip-{index}.wav',
                spectrogram_url=base + '/test.png')

try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def api(route):
            parsed = urlparse(route.request.url)
            query = parse_qs(parsed.query)
            if route.request.method == 'OPTIONS':
                route.fulfill(status=204, headers={'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'})
                return
            if parsed.path == '/auth/login':
                data = {'access_token': 'test', 'token_type': 'Bearer', 'expires_at': '2099-01-01T00:00:00Z'}
            elif parsed.path == '/species':
                data = {'species': [dict(species)]}
                if query.get('include_counts') == ['true']:
                    state['summary'] += 1
                    data['species'][0]['annotation_counts'] = {'positive': state['posts'], 'negative': 0, 'unsure': 0, 'total': state['posts']}
                else:
                    assert query.get('include_counts') == ['false']
            elif parsed.path == '/references':
                data = {'references': {'FROG': [{'reference_id': 'ref', 'audio_url': base + '/reference.wav', 'spectrogram_url': base + '/test.png'}]}, 'urls_expire_at': '2099-01-01T00:00:00Z'}
            elif parsed.path == '/candidates':
                if 'candidate_id' in query:
                    state['refresh'] += 1
                    items = [candidate(int(value.split('-')[-1])) for value in query['candidate_id']]
                    cursor = None
                else:
                    state['pages'] += 1
                    assert len(query.get('seed', [])) == 1
                    seeds.add(query['seed'][0])
                    assert query['count'] == ['20']
                    start = int(query.get('cursor', ['0'])[0])
                    end = min(start + 20, 100)
                    items = [candidate(i) for i in order_for_seed(query['seed'][0])[start:end]]
                    cursor = str(end) if end < 100 else None
                expiry = '2099-01-01T00:00:00Z' if 'candidate_id' in query else (datetime.now(timezone.utc) + timedelta(seconds=61 if state['pages'] == 1 else 900)).isoformat()
                data = {'candidates': items, 'next_cursor': cursor, 'urls_expire_at': expiry}
            elif parsed.path == '/annotations':
                payload = route.request.post_data_json
                state['attempts'].append(payload)
                if state['fail']:
                    route.fulfill(status=503, json={'error': 'uncertain'}, headers={'Access-Control-Allow-Origin': '*'})
                    return
                state['posts'] += 1
                data = {'submission_id': payload['submission_id'], 'accepted_count': len(payload['annotations'])}
            else:
                raise AssertionError(parsed.path)
            route.fulfill(json=data, headers={'Access-Control-Allow-Origin': '*'})

        page.route('https://annotationservice-167659324212.us-south1.run.app/**', api)
        page.route('**/*.wav', lambda route: route.fulfill(content_type='audio/wav', body=AUDIO))
        page.route('**/test.png', lambda route: route.fulfill(content_type='image/png', body=IMAGE))
        page.goto(base + '/docs/annotate.html')
        page.locator('#annotator-password').fill('local-test-password')
        page.locator('#login-button').click()
        expect(page.locator('.results-list > li')).to_have_count(5)
        assert state['summary'] == 0 and state['pages'] == 1
        expected_order = order_for_seed(next(iter(seeds)))
        assert expected_order != list(range(100))
        expect(page.locator('#species-summary')).to_be_hidden()
        page.locator('#annotator-name').fill('Local test')
        page.locator('.results-list input[value=yes]').first.check()
        page.evaluate('''() => { const now = Date.now; Date.now = () => now() + 62000; document.dispatchEvent(new Event('visibilitychange')); }''')
        expect(page.locator('.results-list input[value=yes]').first).to_be_checked()
        page.wait_for_function("document.querySelector('#status').textContent.includes('Submit answered')")
        assert state['refresh'] == 1
        state['fail'] = True
        page.locator('#sendButton').click()
        expect(page.locator('#sendButton')).to_have_text('Retry submission')
        assert state['summary'] == 0
        expect(page.get_by_role('button', name='Next batch', exact=True)).to_be_disabled()
        state['fail'] = False
        page.locator('#sendButton').click()
        expect(page.locator('#status')).to_contain_text('submitted successfully')
        assert state['attempts'][0] == state['attempts'][1]
        assert state['summary'] == 0
        page.locator('#annotation-summary > summary').click()
        page.locator('#refresh-summary').click()
        expect(page.locator('#species-summary')).to_be_visible()
        assert state['summary'] == 1
        page.locator('.results-list input[value=no]').first.check()
        page.locator('#sendButton').click()
        expect(page.locator('#summary-status')).to_contain_text('Answers changed')
        assert state['summary'] == 1
        page.locator('#logout-button').click()
        page.locator('#annotator-password').fill('local-test-password')
        page.locator('#login-button').click()
        expect(page.locator('.results-list h3').first).to_contain_text(f'ID: window-{expected_order[0]}')
        expect(page.locator('#status')).to_contain_text('Submit answered')
        assert state['pages'] == 1 and state['summary'] == 1
        # Twenty visible batches consume five API pages. Buffered Next remains available.
        for index in range(1, 20):
            page.get_by_role('button', name='Next batch', exact=True).click()
            expect(page.locator('.results-list h3').first).to_contain_text(f'ID: window-{expected_order[index * 5]}')
        expect(page.get_by_role('button', name='Next batch', exact=True)).to_be_disabled()
        assert state['pages'] == 5, state
        assert len(seeds) == 1, seeds
        assert not errors, errors
        page.reload()
        expect(page.locator('#annotator-login')).to_be_visible()
        assert page.locator('#annotator-name').input_value() == ''
        assert state['summary'] == 1
        page.locator('#annotator-password').fill('local-test-password')
        page.locator('#login-button').click()
        expect(page.locator('.results-list > li')).to_have_count(5)
        assert len(seeds) == 2, seeds
        assert state['summary'] == 1
        browser.close()
        print('PASS: zero automatic summary queries, explicit counts/staleness, exact manual retry, buffered 100 clips/5 page reads, exact media refresh, re-login keeps slice, fresh reload; no cloud writes')
finally:
    server.shutdown()
