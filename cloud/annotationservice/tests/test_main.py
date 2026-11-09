from datetime import datetime, timedelta, timezone
import base64
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask, request

# Allow running tests from either this directory or the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as service
import media as media_service
from helpers import annotation_payload

SEED = "11111111-1111-4111-8111-111111111111"


class AnnotationServiceTests(unittest.TestCase):
    def setUp(self):
        # Endpoint behavior is exercised as an authorized reviewer. Dedicated
        # auth tests below cover real token verification and blocked cloud calls.
        access = patch.object(service.auth, 'require_access')
        access.start()
        self.addCleanup(access.stop)
        app = Flask(__name__)
        app.add_url_rule('/<path:path>', view_func=lambda path: service.annotationservice(request),
                         methods=['GET', 'POST', 'OPTIONS'])
        self.http = app.test_client()

    @patch.object(service.settings, 'get_client')
    @patch.object(service.annotate, 'submit_annotations')
    def test_annotation_http_receipt_and_cors(self, submit, get_client):
        data = annotation_payload()
        submit.return_value = {'submission_id': data['submission_id'], 'accepted_count': 1}
        response = self.http.post('/annotations', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), submit.return_value)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
        self.assertEqual(submit.call_args.args[0], data)
        self.assertEqual(submit.call_args.args[-1], 100_000_000)
        options = self.http.options('/annotations')
        self.assertEqual(options.status_code, 204)
        self.assertIn('POST', options.headers['Access-Control-Allow-Methods'])

    @patch.object(service.settings, 'get_client')
    def test_invalid_annotation_http_requests_never_reach_cloud(self, client):
        for kwargs, status in [
            ({'data': '{}'}, 415),
            ({'data': '{', 'content_type': 'application/json'}, 400),
            ({'json': {}}, 400),
            ({'json': []}, 400),
            ({'data': ' ' * 16385, 'content_type': 'application/json'}, 413),
        ]:
            with self.subTest(status=status):
                self.assertEqual(self.http.post('/annotations', **kwargs).status_code, status)
        client.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_duplicate_json_keys_and_invalid_text_are_rejected_before_cloud(self, client):
        text = json.dumps(annotation_payload())
        duplicate_root = text.replace('"annotator_name": "Maria"',
                                      '"annotator_name": "Maria", "annotator_name": "Other"')
        duplicate_answer = text.replace('"response": "present"',
                                        '"response": "present", "response": "absent"')
        for raw in [duplicate_root, duplicate_answer]:
            self.assertEqual(self.http.post('/annotations', data=raw,
                                           content_type='application/json').status_code, 400)
        for bad_text in ['a\x00b', 'a\x7fb', 'a\ud800b']:
            for field in ['annotator_name', 'candidate_id']:
                data = annotation_payload()
                if field == 'candidate_id':
                    data['annotations'][0][field] = bad_text
                else:
                    data[field] = bad_text
                with self.subTest(field=field, text=repr(bad_text)):
                    self.assertEqual(self.http.post('/annotations', json=data).status_code, 400)
        client.assert_not_called()

    @patch.object(service.settings, 'get_client')
    @patch.object(service.annotate, 'submit_annotations')
    def test_annotation_ambiguous_failure_warns_about_duplicates(self, submit, client):
        submit.side_effect = TimeoutError('private details')
        with self.assertLogs(level='ERROR'):
            response = self.http.post('/annotations', json=annotation_payload())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers['Retry-After'], '5')
        self.assertNotIn('private details', response.get_json()['error'])
        self.assertIn('duplicate', response.get_json()['error'])

    @patch.object(media_service, 'get_signing_context')
    @patch.object(service.settings, 'get_client')
    def test_references_use_parameterized_ids_and_signed_urls(self, get_client, get_context):
        row = SimpleNamespace(reference_id='AGACAL_1', species_id='AGACAL',
                              audio_gcs_uri='gs://osa-frog-data/reference_catalog/AGACAL_1.mp3',
                              spectrogram_gcs_uri=None)
        get_client.return_value.query.return_value.result.return_value = [row]
        media = Mock()
        media.bucket.return_value.blob.return_value.generate_signed_url.return_value = 'https://signed/audio'
        get_context.return_value = (media, 'signer@example.iam.gserviceaccount.com', 'token')

        response = self.http.get('/references?species_id=AGACAL&species_id=AGAANN&species_id=AGACAL')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['references']['AGAANN'], [])
        self.assertEqual(body['references']['AGACAL'], [{
            'reference_id': 'AGACAL_1', 'audio_url': 'https://signed/audio', 'spectrogram_url': None}])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')

        config = get_client.return_value.query.call_args.kwargs['job_config']
        self.assertEqual(config.query_parameters[0].values, ['AGACAL', 'AGAANN'])
        self.assertEqual(config.maximum_bytes_billed, 100_000_000)
        call = media.bucket.return_value.blob.return_value.generate_signed_url.call_args.kwargs
        self.assertEqual(call['service_account_email'], 'signer@example.iam.gserviceaccount.com')
        self.assertEqual(call['access_token'], 'token')
        self.assertEqual(call['method'], 'GET')
        self.assertEqual(call['version'], 'v4')
        expiration = datetime.fromisoformat(body['urls_expire_at'])
        self.assertEqual(call['expiration'], expiration)
        self.assertTrue(timedelta(minutes=14) < expiration - datetime.now(timezone.utc)
                        <= timedelta(minutes=15))

    @patch.object(service.settings, 'get_client')
    def test_invalid_requests_do_not_query_bigquery(self, get_client):
        for path in ['/references', '/references?species_id=',
                     '/references?' + '&'.join(['species_id=X'] * 53)]:
            self.assertEqual(self.http.get(path).status_code, 400)
        self.assertEqual(self.http.options('/references').status_code, 204)
        self.assertEqual(self.http.post('/species').status_code, 404)
        get_client.assert_not_called()

    @patch.object(media_service, 'get_signing_context')
    @patch.object(service.settings, 'get_client')
    def test_empty_references_need_no_signing_credentials(self, get_client, context):
        get_client.return_value.query.return_value.result.return_value = []
        response = self.http.get('/references?species_id=AGAANN')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['references'], {'AGAANN': []})
        context.assert_not_called()

    def test_rejects_media_outside_configured_bucket(self):
        context = (Mock(), 'signer', 'token')
        for uri in ['gs://other-bucket/audio.mp3', 'https://example.com/audio.mp3',
                    'gs://osa-frog-data/']:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                media_service.sign_media_uri(uri, datetime.now(timezone.utc), context)
        context[0].bucket.assert_not_called()

    @patch.object(media_service, 'get_media_client')
    def test_refreshes_token_before_resolving_runtime_identity(self, get_media):
        credentials = Mock(valid=False, service_account_email='default')
        def refresh(_):
            credentials.service_account_email = 'runtime@example.iam.gserviceaccount.com'
            credentials.token = 'refreshed-token'
        credentials.refresh.side_effect = refresh
        get_media.return_value = (Mock(), credentials)
        with patch.dict(os.environ, {}, clear=True):
            _, email, token = media_service.get_signing_context()
        self.assertEqual(email, 'runtime@example.iam.gserviceaccount.com')
        self.assertEqual(token, 'refreshed-token')
        credentials.refresh.assert_called_once()

    @patch.object(service.settings, 'get_client')
    def test_species_does_not_need_signing(self, get_client):
        get_client.return_value.query.return_value.result.return_value = [SimpleNamespace(
            species_id='AGACAL', scientific_name='Agalychnis callidryas', common_name='Treefrog',
            positive=4, negative=2, unsure=1, total=7)]
        with patch.object(media_service, 'get_signing_context') as context:
            response = self.http.get('/species')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['species'][0]['species_id'], 'AGACAL')
        self.assertEqual(response.get_json()['species'][0]['annotation_counts'],
                         {'positive': 4, 'negative': 2, 'unsure': 1, 'total': 7})
        context.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_species_audio_filter_is_optional(self, get_client):
        get_client.return_value.query.return_value.result.return_value = []
        for suffix, filtered in [('', False), ('?has_audio=false', False),
                                 ('?has_audio=true', True)]:
            with self.subTest(suffix=suffix):
                response = self.http.get('/species' + suffix)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {'species': []})
                query = get_client.return_value.query.call_args.args[0]
                self.assertEqual('WHERE EXISTS' in query, filtered)
                if filtered:
                    self.assertIn('r.species_id = s.species_id', query)
                    self.assertIn("NULLIF(TRIM(r.audio_gcs_uri), '') IS NOT NULL", query)

    @patch.object(service.settings, 'get_client')
    def test_species_candidate_filter_combines_with_audio(self, get_client):
        get_client.return_value.query.return_value.result.return_value = []
        for suffix, audio, candidates in [
            ('?has_candidates=true', False, True),
            ('?has_candidates=false', False, False),
            ('?has_audio=true&has_candidates=true', True, True),
        ]:
            with self.subTest(suffix=suffix):
                response = self.http.get('/species' + suffix)
                self.assertEqual(response.status_code, 200)
                query = get_client.return_value.query.call_args.args[0]
                self.assertEqual('.reference_media`' in query, audio)
                self.assertEqual('.candidates`' in query, candidates)
                if candidates:
                    self.assertIn('c.query_species_id = s.species_id', query)
                    self.assertIn('JOIN', query)
                    self.assertIn("NULLIF(TRIM(c.audio_gcs_uri), '') IS NOT NULL", query)
                if audio and candidates:
                    self.assertIn(' AND EXISTS', query)

    @patch.object(service.settings, 'get_client')
    def test_invalid_species_candidate_filter_does_not_query(self, get_client):
        for value in ['', '1', 'yes', 'TRUE', 'true&has_candidates=false']:
            response = self.http.get('/species?has_candidates=' + value)
            self.assertEqual(response.status_code, 400)
        get_client.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_invalid_species_audio_filter_does_not_query(self, get_client):
        for value in ['', '1', 'yes', 'TRUE', 'true&has_audio=false']:
            with self.subTest(value=value):
                response = self.http.get('/species?has_audio=' + value)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
        get_client.assert_not_called()

    @patch.object(media_service, 'get_signing_context')
    @patch.object(service.settings, 'get_client')
    def test_candidates_paginate_tied_scores_and_sign_shared_audio_once(self, get_client, context):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [SimpleNamespace(
            candidate_id=f'candidate_{i}', window_id=f'window_{i}',
            query_species_id='DENEBR', score=0.1, priority=0.1, created_at=created,
            recording_id='recording_1', audio_gcs_uri='gs://osa-frog-data/audio.mp3',
            start_seconds=i * 5, end_seconds=(i + 1) * 5, spectrogram_gcs_uri=None,
        ) for i in range(3)]
        client = get_client.return_value
        def query_result(data):
            job = Mock()
            job.result.return_value = data
            return job
        client.query.side_effect = [query_result(rows), query_result(rows[2:])]
        context.return_value = (Mock(), 'signer', 'token')
        with patch.object(media_service, 'sign_media_uri', side_effect=lambda uri, *_: 'signed' if uri else None) as sign:
            first = self.http.get('/candidates?species_id=DENEBR&count=2')
            self.assertEqual(first.status_code, 200)
            self.assertEqual(client.query.call_count, 1)
            body = first.get_json()
            self.assertEqual([r['candidate_id'] for r in body['candidates']], ['candidate_0', 'candidate_1'])
            self.assertEqual(body['candidates'][0]['audio_url'], 'signed')
            self.assertEqual(sign.call_count, 1)  # Shared audio signed once; null images skipped.
            self.assertEqual(first.headers['Cache-Control'], 'no-store')
            cutoff, after_priority, after_id, seed = service.candidates.decode_cursor(body['next_cursor'], 'DENEBR')
            self.assertEqual((after_priority, after_id), (0.1, 'candidate_1'))
            second = self.http.get('/candidates', query_string={
                'species_id': 'DENEBR', 'count': '2', 'cursor': body['next_cursor']})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(client.query.call_count, 2)
        self.assertEqual(second.get_json()['next_cursor'], None)
        self.assertEqual(second.get_json()['candidates'][0]['candidate_id'], 'candidate_2')
        config = client.query.call_args.kwargs['job_config']
        params = {p.name: p.value for p in config.query_parameters}
        self.assertEqual(params['cutoff'], cutoff)
        self.assertEqual(params['after_priority'], 0.1)
        self.assertEqual(params['after_id'], 'candidate_1')
        self.assertEqual(params['fetch_count'], 3)
        self.assertIn('ORDER BY priority ASC, candidate_id ASC', client.query.call_args.args[0])
        self.assertNotIn('.recordings`', client.query.call_args.args[0])
        self.assertIn('c.audio_gcs_uri', client.query.call_args.args[0])

    @patch.object(service.candidates, 'get_candidates')
    def test_candidate_default_count(self, candidates):
        candidates.return_value = {'candidates': [], 'next_cursor': None,
                                   'urls_expire_at': 'unused'}
        self.assertEqual(self.http.get('/candidates?species_id=DENEBR').status_code, 200)
        self.assertEqual(candidates.call_args.args[1], 15)
        self.assertEqual(candidates.call_args.args[3:5], (None, None))

    def test_full_final_candidate_page_has_no_cursor(self):
        # Exactly count rows means no additional page, even for a full batch.
        row = SimpleNamespace(candidate_id='one', window_id='window',
                              query_species_id='DENEBR', score=0.1,
                              created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                              recording_id='recording', audio_gcs_uri='gs://osa-frog-data/a',
                              start_seconds=0, end_seconds=5, spectrogram_gcs_uri=None)
        with patch.object(service.settings, 'get_client') as client, \
                patch.object(media_service, 'get_signing_context'), \
                patch.object(media_service, 'sign_media_uri', return_value='signed'):
            client.return_value.query.return_value.result.return_value = [row]
            body = service.candidates.get_candidates('DENEBR', 1, datetime.now(timezone.utc), None, None, SEED)
        self.assertIsNone(body['next_cursor'])

    @patch.object(media_service, 'get_signing_context')
    @patch.object(service.settings, 'get_client')
    def test_candidates_unknown_species_and_empty_queue(self, get_client, context):
        client = get_client.return_value
        client.query.return_value.result.return_value = []
        for species in ['UNKNOWN', 'AGAANN']:
            client.query.reset_mock()
            response = self.http.get('/candidates', query_string={'species_id': species, 'count': 20})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['candidates'], [])
            self.assertIsNone(response.get_json()['next_cursor'])
            client.query.assert_called_once()
            config = client.query.call_args.kwargs['job_config']
            self.assertEqual(config.query_parameters[0].value, species)
        context.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_invalid_candidate_requests_never_query(self, get_client):
        for query in ['', 'species_id=', 'species_id=X&species_id=Y',
                      'species_id=X&count=0', 'species_id=X&count=21',
                      'species_id=X&count=-1', 'species_id=X&count=1.5',
                      'species_id=X&count=', 'species_id=X&count=1&count=2',
                      'species_id=X&cursor=bad', 'species_id=X&cursor=']:
            with self.subTest(query=query):
                self.assertEqual(self.http.get('/candidates?' + query).status_code, 400)
        row = SimpleNamespace(score=0.1, priority=0.1, candidate_id='one')
        token = service.candidates.encode_cursor('DENEBR', datetime.now(timezone.utc), row, SEED)
        self.assertEqual(self.http.get('/candidates', query_string={
            'species_id': 'AGACAL', 'cursor': token}).status_code, 400)
        get_client.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_invalid_cursor_types_dates_and_text_do_not_query(self, client):
        valid = {'v': 3, 'seed': SEED, 'temperature': 0.5, 'species_id': 'DENEBR', 'cutoff': '2026-01-02T00:00:00+00:00',
                 'priority': 0.1, 'candidate_id': 'one'}
        invalid = [None, [], True, {**valid, 'v': True}, {**valid, 'v': 1},
                   {**valid, 'cutoff': '0001-01-01T00:00:00+01:00'},
                   {**valid, 'cutoff': '2026-01-01'},
                   {**valid, 'cutoff': '9999-01-01T00:00:00+00:00'},
                   {**valid, 'priority': float('nan')}, {**valid, 'priority': float('inf')},
                   {**valid, 'priority': -0.1}, {**valid, 'priority': 0},
                   {**valid, 'priority': True}, {**valid, 'priority': '0.1'},
                   {**valid, 'candidate_id': 'a\ud800b'},
                   {**valid, 'candidate_id': 'a\x00b'}]
        for data in invalid:
            token = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip('=')
            with self.subTest(data=data):
                response = self.http.get('/candidates', query_string={
                    'species_id': 'DENEBR', 'cursor': token})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
        for path in ['/candidates', '/references']:
            for species in ['X' * 129, 'A\x00B']:
                response = self.http.get(path, query_string={'species_id': species})
                self.assertEqual(response.status_code, 400)
        client.assert_not_called()

    def test_cursor_offset_is_normalized_to_utc(self):
        data = {'v': 3, 'seed': SEED, 'temperature': 0.5, 'species_id': 'DENEBR', 'cutoff': '2026-01-02T05:30:00+05:30',
                'priority': 0.01, 'candidate_id': 'one'}
        token = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip('=')
        cutoff, priority, candidate_id, seed = service.candidates.decode_cursor(token, 'DENEBR')
        self.assertEqual(cutoff, datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertEqual(priority, 0.01)
        self.assertEqual(seed, SEED)
        self.assertIs(cutoff.tzinfo, timezone.utc)
        self.assertEqual(candidate_id, 'one')

    @patch.object(service.settings, 'get_client')
    def test_sampling_seed_is_validated_and_bound_to_cursor_before_queries(self, client):
        for suffix in ['seed=', 'seed=bad', 'seed=' + 'a'*100,
                       'seed=' + SEED + '&seed=' + SEED]:
            self.assertEqual(self.http.get('/candidates?species_id=DENEBR&' + suffix).status_code, 400)
        row = SimpleNamespace(priority=0.12345678912345, candidate_id='one')
        token = service.candidates.encode_cursor('DENEBR', datetime.now(timezone.utc), row, SEED)
        self.assertEqual(self.http.get('/candidates', query_string={
            'species_id':'DENEBR', 'cursor':token,
            'seed':'22222222-2222-4222-8222-222222222222'}).status_code, 400)
        client.assert_not_called()
        from werkzeug.datastructures import MultiDict
        parsed = service.candidates.parse_candidate_request(MultiDict([
            ('species_id','DENEBR'),('cursor',token),('seed',SEED)]))
        self.assertEqual(parsed[3:], (row.priority, 'one', SEED))
        first = service.candidates.parse_candidate_request(MultiDict([('species_id','DENEBR'),('seed',SEED)]))
        self.assertEqual(first[-1], SEED)

    @patch.object(service.catalog, 'get_references', side_effect=RuntimeError('private signing details'))
    def test_signing_failure_returns_generic_error_with_cors(self, _):
        with self.assertLogs(level='ERROR'):
            response = self.http.get('/references?species_id=AGACAL')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {'error': 'Unable to fetch catalog data'})
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')


    @patch.object(service.settings, 'read_rows')
    def test_species_without_counts_does_not_read_annotations_or_windows(self, read):
        read.return_value = [SimpleNamespace(species_id='FROG', scientific_name='Frog', common_name='Frog')]
        response = self.http.get('/species?has_audio=true&has_candidates=true&include_counts=false')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('annotation_counts', response.get_json()['species'][0])
        query = read.call_args.args[0]
        self.assertNotIn('.annotations`', query)
        self.assertNotIn('.windows`', query)
        self.assertIn('.candidates`', query)
        for suffix in ['?include_counts=maybe', '?include_counts=true&include_counts=false']:
            read.reset_mock()
            self.assertEqual(self.http.get('/species' + suffix).status_code, 400)
            read.assert_not_called()

    @patch.object(service.settings, 'get_client')
    def test_invalid_exact_id_refresh_never_opens_cloud(self, client):
        for suffix in [
            '?species_id=FROG&candidate_id=a&count=5',
            '?species_id=FROG&candidate_id=a&cursor=old',
            '?species_id=FROG&candidate_id=a&candidate_id=a',
            '?species_id=FROG&species_id=OTHER&candidate_id=a',
            '?candidate_id=a', '?species_id=FROG&candidate_id=',
            '?species_id=FROG&' + '&'.join(f'candidate_id={i}' for i in range(21)),
        ]:
            with self.subTest(suffix=suffix):
                self.assertEqual(self.http.get('/candidates' + suffix).status_code, 400)
        client.assert_not_called()

    @patch.object(service.candidates, 'refresh_candidate_media')
    def test_exact_id_refresh_uses_its_own_read_path(self, refresh):
        refresh.return_value = {'candidates': [], 'next_cursor': None, 'urls_expire_at': '2099-01-01T00:00:00Z'}
        response = self.http.get('/candidates?species_id=FROG&candidate_id=a&candidate_id=b')
        self.assertEqual(response.status_code, 200)
        refresh.assert_called_once_with('FROG', ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
