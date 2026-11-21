"""Shared annotator password and four-hour, stateless access tokens."""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from uuid import uuid4

import jwt
from werkzeug.exceptions import BadRequest


TOKEN_LIFETIME_SECONDS = 4 * 60 * 60
TOKEN_ISSUER = 'osa-annotationservice'
TOKEN_AUDIENCE = 'osa-annotator-api'
MAX_LOGIN_BYTES = 2048


class AuthError(Exception):
    def __init__(self, message, status=401):
        super().__init__(message)
        self.status = status


def _configuration():
    # Secrets are injected by Cloud Run; missing configuration never opens access.
    password = os.environ.get('ANNOTATOR_PASSWORD', '')
    secret = os.environ.get('ANNOTATOR_TOKEN_SECRET', '')
    if not 12 <= len(password) <= 1024 or len(secret.encode('utf-8')) < 32:
        raise AuthError('Annotator access is not configured', 503)
    return password, secret


def _login_password(request):
    if not request.is_json:
        raise AuthError('Use Content-Type: application/json', 415)
    if request.content_length is not None and request.content_length > MAX_LOGIN_BYTES:
        raise AuthError('Login body exceeds 2048 bytes', 413)
    try:
        raw = request.stream.read(MAX_LOGIN_BYTES + 1)
        if len(raw) > MAX_LOGIN_BYTES:
            raise AuthError('Login body exceeds 2048 bytes', 413)
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (BadRequest, ValueError, UnicodeError, RecursionError):
        raise AuthError('Provide a valid JSON password object', 400) from None
    if (not isinstance(payload, dict) or set(payload) != {'password'}
            or not isinstance(payload['password'], str)
            or not 1 <= len(payload['password']) <= 1024):
        raise AuthError('Provide a nonempty password string only', 400)
    return payload['password']


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def login(request):
    supplied = _login_password(request)
    password, secret = _configuration()
    try:
        supplied_digest = hashlib.sha256(supplied.encode('utf-8')).digest()
    except UnicodeError:
        raise AuthError('Invalid password') from None
    # Compare fixed-size digests so neither Unicode nor differing lengths cause
    # compare_digest to raise or expose a prefix-dependent string comparison.
    expected_digest = hashlib.sha256(password.encode('utf-8')).digest()
    if not hmac.compare_digest(supplied_digest, expected_digest):
        raise AuthError('Invalid password')
    now = int(datetime.now(timezone.utc).timestamp())
    expiration = now + TOKEN_LIFETIME_SECONDS
    token = jwt.encode({
        'iss': TOKEN_ISSUER, 'aud': TOKEN_AUDIENCE, 'sub': 'shared-annotator',
        'iat': now, 'nbf': now, 'exp': expiration, 'jti': str(uuid4()),
    }, secret, algorithm='HS256')
    return {
        'access_token': token, 'token_type': 'Bearer',
        'expires_in': TOKEN_LIFETIME_SECONDS,
        'expires_at': datetime.fromtimestamp(expiration, timezone.utc).isoformat(),
    }


def require_access(request):
    parts = request.headers.get('Authorization', '').split()
    if len(parts) != 2 or parts[0].lower() != 'bearer' or len(parts[1]) > 4096:
        raise AuthError('An annotator access token is required')
    _, secret = _configuration()
    try:
        claims = jwt.decode(
            parts[1], secret, algorithms=['HS256'],
            issuer=TOKEN_ISSUER, audience=TOKEN_AUDIENCE,
            options={'require': ['iss', 'aud', 'sub', 'iat', 'nbf', 'exp', 'jti']},
        )
        if (any(type(claims[field]) is not int for field in ('iat', 'nbf', 'exp'))
                or claims['sub'] != 'shared-annotator' or claims['nbf'] != claims['iat']
                or claims['exp'] - claims['iat'] != TOKEN_LIFETIME_SECONDS):
            raise jwt.InvalidTokenError()
    except (jwt.InvalidTokenError, TypeError, ValueError, OverflowError):
        raise AuthError('Annotator access token is invalid or expired') from None
