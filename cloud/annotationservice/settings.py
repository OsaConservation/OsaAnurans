"""Shared cloud settings and lazy BigQuery client creation."""

from datetime import timedelta
from functools import lru_cache

from google.cloud import bigquery


PROJECT_ID = 'osaanurans'
DATASET_ID = 'osafrogdataset'
LOCATION = 'us-south1'
MAX_SPECIES_IDS = 52
MAX_CANDIDATES = 20
MAXIMUM_BYTES_BILLED = 100_000_000
# The parameter-only annotation INSERT does not scan existing tables.
MAX_ANNOTATION_BYTES_BILLED = 100_000_000
MEDIA_BUCKET = 'osa-frog-data'
SIGNED_URL_LIFETIME = timedelta(minutes=15)


@lru_cache(maxsize=1)
def get_client():
    # Reuse the client across requests, using the function's service identity.
    return bigquery.Client(project=PROJECT_ID, location=LOCATION)


def read_rows(query, parameters=()):
    """Apply the same billing cap and bounded waits to every catalog query."""
    job = get_client().query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=list(parameters), maximum_bytes_billed=MAXIMUM_BYTES_BILLED,
        ),
        timeout=10,
    )
    return list(job.result(timeout=25))
