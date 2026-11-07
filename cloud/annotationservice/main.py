"""Cloud Run HTTP entry point: routing, validation, and JSON responses."""

import logging

import functions_framework
from flask import jsonify

import annotate
import auth
import candidates
import catalog
import settings


CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Expose-Headers': 'Retry-After',
}


def json_response(payload, status=200, retry_after=None):
    # Both successes and failures should bypass caches, especially write receipts.
    headers = {**CORS_HEADERS, 'Cache-Control': 'no-store'}
    if retry_after is not None:
        headers['Retry-After'] = str(retry_after)
    return jsonify(payload), status, headers


def handle_submission(request):
    try:
        payload = annotate.parse_submission(request)
        receipt = annotate.submit_annotations(
            payload, settings.get_client(), settings.PROJECT_ID, settings.DATASET_ID,
            settings.LOCATION, settings.MAX_ANNOTATION_BYTES_BILLED,
        )
        return json_response(receipt)
    except annotate.SubmissionError as error:
        return json_response({'error': str(error)}, error.status, error.retry_after)
    except Exception:
        logging.exception('Annotation submission failed')
        return json_response(
            {'error': 'Unable to confirm submission; retrying may create duplicate annotations'},
            503, retry_after=5,
        )


def handle_read(request):
    # Parse before opening any cloud clients. Invalid input must not run queries.
    try:
        if request.path == '/species':
            arguments = catalog.parse_species_request(request.args)
            reader = catalog.get_species
        elif request.path == '/references':
            arguments = catalog.parse_reference_request(request.args)
            reader = catalog.get_references
        else:
            arguments = candidates.parse_candidate_request(request.args)
            reader = candidates.get_candidates
    except ValueError as error:
        return json_response({'error': str(error)}, 400)

    try:
        payload = reader(*arguments)
        return json_response(payload)
    except Exception:
        logging.exception('Catalog query failed for %s', request.path)
        return json_response({'error': 'Unable to fetch catalog data'}, 500)


def auth_response(error):
    response, status, headers = json_response({'error': str(error)}, error.status)
    if status == 401:
        headers['WWW-Authenticate'] = 'Bearer realm="osa-annotators"'
    return response, status, headers


@functions_framework.http
def annotationservice(request):
    if request.method == 'OPTIONS':
        return '', 204, CORS_HEADERS
    if request.method == 'POST' and request.path == '/auth/login':
        try:
            return json_response(auth.login(request))
        except auth.AuthError as error:
            return auth_response(error)

    is_submission = request.method == 'POST' and request.path == '/annotations'
    is_read = request.method == 'GET' and request.path in ('/species', '/references', '/candidates')
    if not (is_submission or is_read):
        return json_response({'error': 'Not found'}, 404)
    try:
        # Authenticate before request parsing, cloud clients, queries, or signing.
        auth.require_access(request)
    except auth.AuthError as error:
        return auth_response(error)
    return handle_submission(request) if is_submission else handle_read(request)
