"""Small immutable-by-construction fixtures shared by the service tests."""


def annotation_payload():
    return {
        'submission_id': 'a8ff083c-d6a1-43f9-bcf2-e52bf19264ba',
        'annotator_id': '0c3ab387-866c-4e9a-b15f-75c21d0d9a76',
        'annotator_name': 'Maria',
        'annotations': [{'candidate_id': 'test_candidate_DENEBR_1',
                         'window_id': 'test_window_DENEBR_1',
                         'query_species_id': 'DENEBR', 'response': 'present'}],
    }
