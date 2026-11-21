"""Check request format and append annotation batches with one BigQuery INSERT."""

import json
import unicodedata
from uuid import UUID

from google.cloud import bigquery
from werkzeug.exceptions import BadRequest



MAX_BODY_BYTES = 16_384
MAX_ANSWERS = 20
RESPONSES = {'present', 'absent', 'unsure'}


class SubmissionError(Exception):
    def __init__(self, message, status=400, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_submission(request):
    if not request.is_json:
        raise SubmissionError('Use Content-Type: application/json', 415)
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        raise SubmissionError('Request body exceeds 16384 bytes', 413)
    try:
        raw = request.stream.read(MAX_BODY_BYTES + 1)
    except BadRequest:
        raise SubmissionError('Request body is incomplete') from None
    if len(raw) > MAX_BODY_BYTES:
        raise SubmissionError('Request body exceeds 16384 bytes', 413)
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (ValueError, UnicodeError, RecursionError):
        raise SubmissionError('Provide a valid JSON object') from None
    return validate_submission(payload)


def validate_submission(payload):
    fields = {'submission_id', 'annotator_id', 'annotator_name', 'annotations'}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise SubmissionError('Provide submission_id, annotator_id, annotator_name, and annotations only')
    return {
        'submission_id': _normalize_uuid(payload['submission_id'], 'submission_id'),
        'annotator_id': _normalize_uuid(payload['annotator_id'], 'annotator_id'),
        'annotator_name': _normalize_name(payload['annotator_name']),
        'annotations': _validate_answers(payload['annotations']),
    }


def _normalize_uuid(value, field):
    try:
        if not isinstance(value, str) or len(value) != 36:
            raise ValueError()
        normalized = str(UUID(value))
        if value.lower() != normalized:
            raise ValueError()
        return normalized
    except ValueError:
        raise SubmissionError(f'{field} must be a UUID') from None


def _normalize_name(name):
    if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 100
            or any(unicodedata.category(char) in ('Cc', 'Cs') for char in name)):
        raise SubmissionError('annotator_name must contain 1 to 100 characters without control characters')
    return name.strip()


def _identifier(value, field, limit=256):
    if (not isinstance(value, str) or not 1 <= len(value) <= limit
            or not value.isprintable() or value != value.strip()):
        raise SubmissionError(f'{field} must contain 1 to {limit} printable characters without surrounding whitespace')
    return value


def _validate_answers(answers):
    if not isinstance(answers, list) or not 1 <= len(answers) <= MAX_ANSWERS:
        raise SubmissionError('Provide between 1 and 20 annotations')
    required = {'window_id', 'query_species_id', 'response'}
    cleaned = []
    for answer in answers:
        if (not isinstance(answer, dict) or not required <= set(answer)
                or set(answer) - required - {'candidate_id'}):
            raise SubmissionError('Each annotation requires window_id, query_species_id, and response; candidate_id is optional')
        response = answer['response']
        if not isinstance(response, str) or response not in RESPONSES:
            raise SubmissionError('response must be present, absent, or unsure')
        candidate_id = answer.get('candidate_id')
        cleaned.append({
            'window_id': _identifier(answer['window_id'], 'window_id'),
            'query_species_id': _identifier(answer['query_species_id'], 'query_species_id', 128),
            'candidate_id': _identifier(candidate_id, 'candidate_id') if candidate_id is not None else None,
            'response': response,
        })
    return cleaned


def submission_query(project_id, dataset_id):
    # All values come from parameters. No existing tables are read or checked.
    return f"""
        INSERT INTO `{project_id}.{dataset_id}.annotations` (
            annotation_id, window_id, candidate_id, query_species_id, response,
            annotator_id, annotator_name, submission_id, created_at
        )
        SELECT GENERATE_UUID(), JSON_VALUE(item, '$.window_id'),
               JSON_VALUE(item, '$.candidate_id'), JSON_VALUE(item, '$.query_species_id'),
               JSON_VALUE(item, '$.response'), @annotator_id, @annotator_name,
               @submission_id, CURRENT_TIMESTAMP()
        FROM UNNEST(JSON_QUERY_ARRAY(@answers)) AS item
    """


def submit_annotations(payload, client, project_id, dataset_id, location, maximum_bytes_billed):
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('submission_id', 'STRING', payload['submission_id']),
            bigquery.ScalarQueryParameter('annotator_id', 'STRING', payload['annotator_id']),
            bigquery.ScalarQueryParameter('annotator_name', 'STRING', payload['annotator_name']),
            bigquery.ScalarQueryParameter('answers', 'STRING', json.dumps(payload['annotations'])),
        ],
        maximum_bytes_billed=maximum_bytes_billed,
    )
    # One INSERT per request. A repeated request intentionally appends more rows.
    job = client.query(submission_query(project_id, dataset_id), job_config=config,
                       location=location, timeout=10, job_retry=None)
    job.result(timeout=25, job_retry=None)
    return {'submission_id': payload['submission_id'],
            'accepted_count': job.num_dml_affected_rows}
