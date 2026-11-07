"""Species and reference-media catalog reads."""

from google.cloud import bigquery

import media
import settings
from settings import DATASET_ID, MAX_SPECIES_IDS, PROJECT_ID


def parse_species_request(args):
    filters = []
    for name in ('has_audio', 'has_candidates'):
        values = args.getlist(name)
        if values and (len(values) != 1 or values[0] not in ('true', 'false')):
            raise ValueError(f'{name} must be true or false, supplied once')
        filters.append(bool(values and values[0] == 'true'))
    return tuple(filters)


def parse_reference_request(args):
    values = args.getlist('species_id')
    if (not values or len(values) > MAX_SPECIES_IDS
            or any(not value.strip() or len(value.strip()) > 128
                   or not value.strip().isprintable() for value in values)):
        raise ValueError('Provide between 1 and 52 nonempty species_id values (maximum 128 characters each)')
    return (list(dict.fromkeys(value.strip() for value in values)),)


def get_species(has_audio=False, has_candidates=False):
    # EXISTS keeps one species row even when it has multiple reference clips.
    filters = []
    if has_audio:
        filters.append(f'''EXISTS (
            SELECT 1 FROM `{PROJECT_ID}.{DATASET_ID}.reference_media` r
            WHERE r.species_id = s.species_id
              AND NULLIF(TRIM(r.audio_gcs_uri), '') IS NOT NULL
        )''')
    if has_candidates:
        # Match candidate windows that have playable audio; keep one species row.
        filters.append(f'''EXISTS (
            SELECT 1 FROM `{PROJECT_ID}.{DATASET_ID}.candidates` c
            JOIN `{PROJECT_ID}.{DATASET_ID}.windows` w USING (window_id)
            WHERE c.query_species_id = s.species_id
              AND NULLIF(TRIM(w.audio_gcs_uri), '') IS NOT NULL
        )''')
    where = 'WHERE ' + ' AND '.join(filters) if filters else ''
    query = f'''
        WITH counts AS (
            SELECT query_species_id,
                   COUNTIF(response = 'present') AS positive,
                   COUNTIF(response = 'absent') AS negative,
                   COUNTIF(response = 'unsure') AS unsure,
                   COUNT(*) AS total
            FROM `{PROJECT_ID}.{DATASET_ID}.annotations`
            GROUP BY query_species_id
        )
        SELECT s.species_id, s.scientific_name, s.common_name,
               COALESCE(a.positive, 0) AS positive,
               COALESCE(a.negative, 0) AS negative,
               COALESCE(a.unsure, 0) AS unsure,
               COALESCE(a.total, 0) AS total
        FROM `{PROJECT_ID}.{DATASET_ID}.species` s
        LEFT JOIN counts a ON a.query_species_id = s.species_id
        {where}
        ORDER BY s.scientific_name
    '''
    rows = settings.read_rows(query)
    return {'species': [
        {'species_id': row.species_id,
         'scientific_name': row.scientific_name,
         'common_name': row.common_name,
         'annotation_counts': {
             'positive': row.positive, 'negative': row.negative,
             'unsure': row.unsure, 'total': row.total,
         }}
        for row in rows
    ]}


def get_references(species_ids):
    query = f'''
        SELECT reference_id, species_id, audio_gcs_uri, spectrogram_gcs_uri
        FROM `{PROJECT_ID}.{DATASET_ID}.reference_media`
        WHERE species_id IN UNNEST(@species_ids)
        ORDER BY species_id, reference_id
    '''
    rows = settings.read_rows(query, [
        bigquery.ArrayQueryParameter('species_ids', 'STRING', species_ids),
    ])
    urls = media.MediaUrls()

    # Include requested species with no reference media as empty lists.
    references = {species_id: [] for species_id in species_ids}
    for row in rows:
        references[row.species_id].append({
            'reference_id': row.reference_id,
            'audio_url': urls.url(row.audio_gcs_uri),
            'spectrogram_url': urls.url(row.spectrogram_gcs_uri),
        })
    return {'references': references, 'urls_expire_at': urls.expires_at.isoformat()}
