"""Candidate batches with species-bound keyset pagination."""

import base64
import binascii
import json
import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

from google.cloud import bigquery

import media
import settings
from settings import DATASET_ID, MAX_CANDIDATES, PROJECT_ID, SAMPLING_TEMPERATURE


def _score(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Candidate score must be finite and between zero and one')
    return float(value)


def _candidate_id(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or not value.isprintable() or value != value.strip()):
        raise ValueError('Invalid candidate ID')
    return value


def _seed(value):
    """Accept a canonical UUID so request/cursor seed comparisons are unambiguous."""
    if not isinstance(value, str) or len(value) != 36 or str(UUID(value)) != value:
        raise ValueError('seed must be a canonical UUID')
    return value


def _priority(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('Sampling priority must be finite and positive')
    return float(value)


def encode_cursor(species_id, cutoff, row, seed):
    # Continuation data, not an authorization credential. JSON round-trips FLOAT64.
    data = {'v': 3, 'species_id': species_id, 'cutoff': cutoff.isoformat(),
            'seed': _seed(seed), 'temperature': SAMPLING_TEMPERATURE,
            'priority': _priority(row.priority), 'candidate_id': _candidate_id(row.candidate_id)}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip('=')


def decode_cursor(token, species_id):
    try:
        if not token or len(token) > 2048:
            raise ValueError()
        raw = base64.b64decode(token + '=' * (-len(token) % 4), altchars=b'-_', validate=True)
        data = json.loads(raw)
        if (not isinstance(data, dict) or type(data.get('v')) is not int
                or data['v'] != 3 or data['species_id'] != species_id
                or isinstance(data.get('temperature'), bool)
                or data['temperature'] != SAMPLING_TEMPERATURE):
            raise ValueError()
        cutoff = datetime.fromisoformat(data['cutoff'])
        if cutoff.utcoffset() is None:
            raise ValueError()
        cutoff = cutoff.astimezone(timezone.utc)
        if cutoff > datetime.now(timezone.utc):
            raise ValueError()
        return cutoff, _priority(data['priority']), _candidate_id(data['candidate_id']), _seed(data['seed'])
    except (ValueError, KeyError, TypeError, AttributeError, binascii.Error, UnicodeError,
            RecursionError, OverflowError):
        raise ValueError('Invalid or outdated cursor for this species; restart from the first batch') from None


def parse_candidate_request(args):
    for key in ('species_id', 'count', 'cursor', 'seed'):
        if len(args.getlist(key)) > 1:
            raise ValueError(f'Supply {key} only once')
    species_id = args.get('species_id', '').strip()
    if not 1 <= len(species_id) <= 128 or not species_id.isprintable():
        raise ValueError('Provide one nonempty species_id (maximum 128 characters)')
    count_text = args.get('count', '15')
    if (not count_text.isascii() or not count_text.isdecimal()
            or len(count_text) > 2 or not 1 <= int(count_text) <= MAX_CANDIDATES):
        raise ValueError('count must be an integer between 1 and 20')
    if 'cursor' in args:
        cutoff, priority, after_id, seed = decode_cursor(args['cursor'], species_id)
        if 'seed' in args and _seed(args['seed']) != seed:
            raise ValueError('seed does not match cursor; restart from the first batch')
    else:
        cutoff, priority, after_id = datetime.now(timezone.utc), None, None
        seed = _seed(args['seed']) if 'seed' in args else str(uuid4())
    return species_id, int(count_text), cutoff, priority, after_id, seed


def _read_candidate_rows(species_id, count, cutoff, after_priority, after_id, seed):
    # Mask to 52 bits before converting to FLOAT64. Half-step bin centers keep u
    # strictly inside (0, 1), including negative hashes and the largest bin.
    # JSON STRUCT encoding prevents ambiguous concatenations of seed/species/ID.
    query = f"""
        WITH hashed AS (
            SELECT c.candidate_id, c.window_id, c.query_species_id, c.created_at, c.score,
                   c.recording_start_seconds AS start_seconds, c.recording_end_seconds AS end_seconds,
                   c.spectrogram_gcs_uri, c.recording_id, c.audio_gcs_uri,
                   (CAST(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(
                       @seed AS seed, c.query_species_id AS species_id,
                       c.candidate_id AS candidate_id
                   ))) & 4503599627370495 AS FLOAT64) + 0.5) / 4503599627370496.0 AS uniform_value
            FROM `{PROJECT_ID}.{DATASET_ID}.candidates` c
            WHERE c.query_species_id = @species_id AND c.created_at <= @cutoff
        ), ranked AS (
            SELECT hashed.*, -LN(uniform_value) * EXP(score / @temperature) AS priority
            FROM hashed
        )
        SELECT * EXCEPT (uniform_value) FROM ranked
        WHERE @after_priority IS NULL OR priority > @after_priority
           OR (priority = @after_priority AND candidate_id > @after_id)
        ORDER BY priority ASC, candidate_id ASC
        LIMIT @fetch_count
    """
    return settings.read_rows(query, [
        bigquery.ScalarQueryParameter('species_id', 'STRING', species_id),
        bigquery.ScalarQueryParameter('cutoff', 'TIMESTAMP', cutoff),
        bigquery.ScalarQueryParameter('after_priority', 'FLOAT64', after_priority),
        bigquery.ScalarQueryParameter('after_id', 'STRING', after_id),
        bigquery.ScalarQueryParameter('seed', 'STRING', seed),
        bigquery.ScalarQueryParameter('temperature', 'FLOAT64', SAMPLING_TEMPERATURE),
        # Fetch one extra row to detect whether another page exists.
        bigquery.ScalarQueryParameter('fetch_count', 'INT64', count + 1),
    ])


def _recording_bounds(row):
    start, end = row.start_seconds, row.end_seconds
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Candidate recording offsets must be finite numbers')
    if start < 0 or not math.isclose(end - start, 5, rel_tol=0, abs_tol=1e-6):
        raise ValueError('Candidate window must cover exactly five seconds')
    return start, end


def _candidate_payload(row, urls):
    recording_start, recording_end = _recording_bounds(row)
    if not isinstance(row.audio_gcs_uri, str) or not row.audio_gcs_uri.startswith('gs://'):
        raise ValueError('Candidate window requires a GCS audio URI')
    return {
        'candidate_id': _candidate_id(row.candidate_id), 'window_id': row.window_id,
        'query_species_id': row.query_species_id, 'score': _score(row.score),
        'created_at': row.created_at.isoformat(), 'recording_id': row.recording_id,
        'audio_url': urls.url(row.audio_gcs_uri),
        'start_seconds': 0, 'end_seconds': 5,
        'recording_start_seconds': recording_start,
        'recording_end_seconds': recording_end,
        'spectrogram_url': urls.url(row.spectrogram_gcs_uri),
    }


def get_candidates(species_id, count, cutoff, after_priority, after_id, seed):
    rows = _read_candidate_rows(species_id, count, cutoff, after_priority, after_id, seed)
    page = rows[:count]
    next_cursor = encode_cursor(species_id, cutoff, page[-1], seed) if len(rows) > count else None
    urls = media.MediaUrls()
    return {
        'candidates': [_candidate_payload(row, urls) for row in page],
        'next_cursor': next_cursor,
        'urls_expire_at': urls.expires_at.isoformat(),
    }


def parse_media_refresh(args):
    """Refresh exact questions without moving a pagination cursor."""
    if set(args) - {'species_id', 'candidate_id'} or len(args.getlist('species_id')) != 1:
        raise ValueError('Supply species_id and candidate_id only for media refresh')
    species = args.get('species_id', '').strip()
    ids = args.getlist('candidate_id')
    if not 1 <= len(species) <= 128 or not species.isprintable():
        raise ValueError('Provide one nonempty species_id')
    if not 1 <= len(ids) <= MAX_CANDIDATES or len(set(ids)) != len(ids):
        raise ValueError('Provide between 1 and 20 unique candidate IDs')
    return species, [_candidate_id(value) for value in ids]


def refresh_candidate_media(species_id, candidate_ids):
    query = f"""
        SELECT candidate_id, window_id, query_species_id, created_at, score,
               recording_start_seconds AS start_seconds, recording_end_seconds AS end_seconds,
               spectrogram_gcs_uri, recording_id, audio_gcs_uri
        FROM `{PROJECT_ID}.{DATASET_ID}.candidates`
        WHERE query_species_id = @species_id AND candidate_id IN UNNEST(@candidate_ids)
    """
    rows = settings.read_rows(query, [
        bigquery.ScalarQueryParameter('species_id', 'STRING', species_id),
        bigquery.ArrayQueryParameter('candidate_ids', 'STRING', candidate_ids),
    ])
    urls = media.MediaUrls()
    return {'candidates': [_candidate_payload(row, urls) for row in rows],
            'next_cursor': None, 'urls_expire_at': urls.expires_at.isoformat()}
