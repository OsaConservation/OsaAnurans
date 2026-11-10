from datetime import datetime
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from flask import Flask, request
import jwt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auth
import main as service
from helpers import annotation_payload


# Test-only values, never production defaults.
TEST_ENV = {'ANNOTATOR_PASSWORD': 'test-reviewer-password',
            'ANNOTATOR_TOKEN_SECRET': 'test-only-signing-secret-' * 3}


class ReviewerAccessTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, TEST_ENV, clear=True)
        env.start()
        self.addCleanup(env.stop)
        app = Flask(__name__)
        app.add_url_rule('/<path:path>', view_func=lambda path: service.annotationservice(request),
                         methods=['GET', 'POST', 'OPTIONS'])
        self.http = app.test_client()

    def login(self):
        response = self.http.post('/auth/login', json={'password': TEST_ENV['ANNOTATOR_PASSWORD']})
        self.assertEqual(response.status_code, 200)
        return response

    def test_login_returns_four_hour_token_without_cloud_calls(self):
        with patch.object(service.settings, 'get_client') as client:
            response = self.login()
        client.assert_not_called()
        body = response.get_json()
        self.assertEqual(body['token_type'], 'Bearer')
        self.assertEqual(body['expires_in'], 14400)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], '*')
        claims = jwt.decode(body['access_token'], TEST_ENV['ANNOTATOR_TOKEN_SECRET'],
                            algorithms=['HS256'], audience=auth.TOKEN_AUDIENCE, issuer=auth.TOKEN_ISSUER)
        self.assertEqual(claims['exp'] - claims['iat'], 14400)
        self.assertEqual(int(datetime.fromisoformat(body['expires_at']).timestamp()), claims['exp'])
        self.assertNotIn('password', claims)

    def test_all_data_routes_reject_missing_or_invalid_tokens_before_cloud(self):
        with patch.object(service.settings, 'get_client') as client:
            for authorization in ['', 'Basic credentials', 'Bearer invalid', 'Bearer a b']:
                for path in ['/species', '/references?species_id=DENEBR', '/candidates?species_id=DENEBR']:
                    with self.subTest(path=path, authorization=authorization):
                        response = self.http.get(path, headers={'Authorization': authorization})
                        self.assertEqual(response.status_code, 401)
                        self.assertEqual(response.headers['Cache-Control'], 'no-store')
                        self.assertIn('Bearer', response.headers['WWW-Authenticate'])
                response = self.http.post('/annotations', json=annotation_payload(),
                                          headers={'Authorization': authorization})
                self.assertEqual(response.status_code, 401)
        client.assert_not_called()

    def test_valid_token_can_read_and_submit(self):
        token = self.login().get_json()['access_token']
        headers = {'Authorization': 'Bearer ' + token}
        with patch.object(service.catalog, 'get_species', return_value={'species': []}) as species:
            self.assertEqual(self.http.get('/species', headers=headers).status_code, 200)
            species.assert_called_once()
        with patch.object(service.settings, 'get_client'), \
                patch.object(service.annotate, 'submit_annotations', return_value={
                    'submission_id': annotation_payload()['submission_id'], 'accepted_count': 1}) as submit:
            self.assertEqual(self.http.post('/annotations', json=annotation_payload(), headers=headers).status_code, 200)
            submit.assert_called_once()

    def test_wrong_password_and_bad_login_bodies(self):
        for kwargs, status in [
            ({'json': {'password': 'wrong'}}, 401),
            ({'json': {'password': ''}}, 400),
            ({'json': {'password': 123}}, 400),
            ({'json': {'password': 'x', 'extra': True}}, 400),
            ({'json': []}, 400),
            ({'data': '{}'}, 415),
            ({'data': '{', 'content_type': 'application/json'}, 400),
            ({'data': '{"password":"one","password":"two"}', 'content_type': 'application/json'}, 400),
            ({'data': ' ' * 2049, 'content_type': 'application/json'}, 413),
        ]:
            with self.subTest(status=status):
                self.assertEqual(self.http.post('/auth/login', **kwargs).status_code, status)

    def test_expired_tampered_wrong_audience_and_unsigned_tokens_are_rejected(self):
        token = self.login().get_json()['access_token']
        claims = jwt.decode(token, options={'verify_signature': False})
        expired = {**claims, 'iat': claims['iat'] - 14401, 'nbf': claims['nbf'] - 14401,
                   'exp': claims['exp'] - 14401}
        missing_exp = {k: v for k, v in claims.items() if k != 'exp'}
        bad_claims = [expired, {**claims, 'aud': 'another-api'}, {**claims, 'iss': 'another-issuer'},
                      {**claims, 'sub': 'another-role'}, {**claims, 'exp': claims['exp'] + 1}, missing_exp]
        invalid_tokens = [jwt.encode(c, TEST_ENV['ANNOTATOR_TOKEN_SECRET'], algorithm='HS256') for c in bad_claims]
        invalid_tokens += [jwt.encode(claims, 'different-secret-' * 4, algorithm='HS256'),
                           jwt.encode(claims, '', algorithm='none')]
        with patch.object(service.settings, 'get_client') as client:
            for invalid in invalid_tokens:
                response = self.http.get('/species', headers={'Authorization': 'Bearer ' + invalid})
                self.assertEqual(response.status_code, 401)
        client.assert_not_called()

    def test_missing_or_weak_configuration_fails_closed(self):
        token = self.login().get_json()['access_token']
        for configuration in [{}, {**TEST_ENV, 'ANNOTATOR_TOKEN_SECRET': 'short'},
                              {**TEST_ENV, 'ANNOTATOR_PASSWORD': 'short'}]:
            with self.subTest(configuration_keys=list(configuration)), \
                    patch.dict(os.environ, configuration, clear=True), \
                    patch.object(service.settings, 'get_client') as client:
                self.assertEqual(self.http.post('/auth/login', json={'password': 'anything'}).status_code, 503)
                self.assertEqual(self.http.get('/species', headers={'Authorization': 'Bearer ' + token}).status_code, 503)
                client.assert_not_called()

    def test_preflight_stays_public_and_allows_authorization(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.http.options('/annotations', headers={
                'Origin': 'https://osaconservation.github.io',
                'Access-Control-Request-Headers': 'authorization,content-type',
                'Access-Control-Request-Method': 'POST',
            })
        self.assertEqual(response.status_code, 204)
        self.assertIn('Authorization', response.headers['Access-Control-Allow-Headers'])

    def test_rotating_signing_secret_invalidates_old_tokens(self):
        token = self.login().get_json()['access_token']
        with patch.dict(os.environ, {'ANNOTATOR_TOKEN_SECRET': 'replacement-secret-' * 4}):
            response = self.http.get('/species', headers={'Authorization': 'Bearer ' + token})
        self.assertEqual(response.status_code, 401)
