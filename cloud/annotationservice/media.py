"""Private GCS media delivery through short-lived signed URLs."""

from datetime import datetime, timezone
from functools import lru_cache
import os

import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

from settings import MEDIA_BUCKET, PROJECT_ID, SIGNED_URL_LIFETIME


@lru_cache(maxsize=1)
def get_media_client():
    # Cloud Run provides short-lived credentials, rather than a private key file.
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    return storage.Client(project=PROJECT_ID, credentials=credentials), credentials


def get_signing_context():
    client, credentials = get_media_client()
    if not credentials.valid:
        credentials.refresh(Request())

    # Refresh resolves the attached account's email on Cloud Run. An explicit
    # signer also supports local development using ordinary user credentials.
    email = os.environ.get('SIGNING_SERVICE_ACCOUNT') or getattr(
        credentials, 'service_account_email', None)
    if not email or email == 'default':
        raise RuntimeError('Set SIGNING_SERVICE_ACCOUNT to the signing service account email')
    return client, email, credentials.token


def sign_media_uri(uri, expires_at, signing_context):
    if uri is None:
        return None
    prefix = f'gs://{MEDIA_BUCKET}/'
    if not uri.startswith(prefix) or not uri[len(prefix):]:
        raise ValueError('Media must point to an object in the configured bucket')

    client, email, access_token = signing_context
    blob = client.bucket(MEDIA_BUCKET).blob(uri[len(prefix):])
    # Passing the identity and token uses IAM signBlob remotely. No private key
    # is downloaded, and the URL grants GET access to just this object.
    return blob.generate_signed_url(
        version='v4', method='GET', expiration=expires_at,
        service_account_email=email, access_token=access_token,
    )


class MediaUrls:
    """Sign each distinct object once per response, with a shared expiration."""

    def __init__(self):
        self.expires_at = datetime.now(timezone.utc) + SIGNED_URL_LIFETIME
        self._context = None
        self._urls = {}

    def url(self, uri):
        if uri is None:
            return None
        if uri not in self._urls:
            # Empty batches and absent images never need signing credentials.
            if self._context is None:
                self._context = get_signing_context()
            self._urls[uri] = sign_media_uri(uri, self.expires_at, self._context)
        return self._urls[uri]
