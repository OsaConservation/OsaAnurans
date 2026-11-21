"""Read-only usage estimates; no BigQuery SQL, billing changes, or service cutoff."""

from dataclasses import dataclass
from datetime import UTC, datetime

from ..progress import activity
from .bigquery import BigQueryStore
from .storage import StorageStore

GIB = 1024**3


@dataclass(frozen=True)
class UsageRow:
    """One resource's bytes and personal budget; errors must never appear as zero."""

    label: str
    budget_gib: float | None
    used_bytes: int | None = None
    error: str | None = None


def _metadata_pages(session, url, params, key):
    """Follow metadata pagination without issuing data queries."""
    params = dict(params)
    while True:
        response = session.get(url, params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(
                f"Metadata API returned HTTP {response.status_code}; check API access and IAM permissions"
            )
        data = response.json()
        yield from data.get(key, [])
        token = data.get("nextPageToken")
        if not token:
            return
        params["pageToken"] = token


def query_usage(session, project, start, end):
    """Sum completed on-demand jobs created this UTC month, across the project.

    - Top-level scripts already summarize their children; count each job once.
    - Include failed jobs' reported billed bytes; exclude dry runs and cache hits.
    - Require full job statistics rather than silently reporting partial usage.
    """
    params = {
        "allUsers": "true",
        "projection": "full",
        "maxResults": 1000,
        "minCreationTime": int(start.timestamp() * 1000),
        "maxCreationTime": int(end.timestamp() * 1000),
        "fields": "nextPageToken,jobs(jobReference,configuration(jobType,dryRun),status,statistics(parentJobId,query(totalBytesBilled,totalBytesProcessed,cacheHit)))",
    }
    jobs = _metadata_pages(
        session,
        f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/jobs",
        params,
        "jobs",
    )
    total = 0
    seen = set()
    for job in jobs:
        reference = job["jobReference"]
        key = (reference.get("location"), reference["jobId"])
        if key in seen:
            continue
        seen.add(key)
        config = job.get("configuration", {})
        stats = job.get("statistics", {})
        if not config.get("jobType"):
            raise RuntimeError(
                "Some job metadata is redacted; project usage needs full job visibility"
            )
        if config.get("jobType") != "QUERY" or stats.get("parentJobId") or config.get("dryRun"):
            continue
        if job.get("status", {}).get("state") != "DONE":
            continue
        query = stats.get("query", {})
        if query.get("cacheHit"):
            continue
        billed = query.get("totalBytesBilled")
        if billed is None:
            # A rejected query can fail before incurring any query work.
            if job.get("status", {}).get("errorResult") and not int(
                query.get("totalBytesProcessed", 0)
            ):
                continue
            raise RuntimeError(
                "Some completed jobs have no billed-byte statistics; project usage is unavailable"
            )
        total += int(billed)
    return total


def download_usage(session, project, bucket, start, end):
    """Sum bucket network bytes sent; this includes traffic that may not be billed."""
    params = {
        "filter": f'metric.type="storage.googleapis.com/network/sent_bytes_count" AND resource.type="gcs_bucket" AND resource.labels.bucket_name="{bucket}"',
        "interval.startTime": start.isoformat(),
        "interval.endTime": end.isoformat(),
        "aggregation.alignmentPeriod": "3600s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
        "view": "FULL",
    }
    series = _metadata_pages(
        session,
        f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries",
        params,
        "timeSeries",
    )
    total = 0
    found = False
    for item in series:
        for point in item.get("points", []):
            total += int(point["value"]["int64Value"])
            found = True
    if not found:
        raise RuntimeError(
            "Cloud Monitoring returned no traffic samples; usage may be delayed or unavailable"
        )
    return total


def _network_measurement(config, reader, *arguments):
    """Use existing ADC without adding a Monitoring SDK dependency."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=config.project
    )
    with AuthorizedSession(credentials) as session:
        return reader(session, config.project, *arguments)


def _gcs_storage(config):
    sizes = StorageStore(config).storage_usage(include_versions=True)
    return sizes["active_bytes"] + sizes["soft_deleted_bytes"]


def collect_usage(config, *, now=None):
    """Read each category independently so one missing permission does not hide others."""
    now = now or datetime.now(UTC)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    readers = (
        (
            "BQ queries (month, project)",
            config.bq_query_budget_gib,
            lambda: _network_measurement(config, query_usage, start, now),
        ),
        (
            "BQ storage (current dataset)",
            config.bq_storage_budget_gib,
            lambda: BigQueryStore(config).storage_usage()["logical_bytes"],
        ),
        (
            "GCS storage (current bucket)",
            config.gcs_storage_budget_gib,
            lambda: _gcs_storage(config),
        ),
        (
            "GCS network sent (month)",
            config.gcs_download_budget_gib,
            lambda: _network_measurement(config, download_usage, config.bucket, start, now),
        ),
    )
    rows = []
    for label, budget, read in readers:
        try:
            with activity(f"Checking {label}"):
                used = read()
            rows.append(UsageRow(label, budget, used_bytes=used))
        except Exception as error:
            # Reporting is best effort; authentication/API failures remain visible per row.
            rows.append(UsageRow(label, budget, error=str(error)))
    return now, rows


def display_cloud_usage(config):
    """Print remaining personal budgets; negative values show budget overruns."""
    print("\nChecking Google Cloud usage (read-only)...", flush=True)
    now, rows = collect_usage(config)
    print(f"\nGoogle Cloud usage - {now:%Y-%m} UTC (GiB)")
    print(f"{'Resource':<32} {'Used':>12} {'Budget':>12} {'Remaining':>12}")
    for row in rows:
        used = f"{row.used_bytes / GIB:,.3f}" if row.used_bytes is not None else "unavailable"
        budget = f"{row.budget_gib:,.3f}" if row.budget_gib is not None else "not set"
        remaining = (
            f"{row.budget_gib - row.used_bytes / GIB:,.3f}"
            if row.budget_gib is not None and row.used_bytes is not None
            else "-"
        )
        print(f"{row.label:<32} {used:>12} {budget:>12} {remaining:>12}")
        if row.error:
            print(f"  {row.error}")
        if (
            row.used_bytes is not None
            and row.budget_gib is not None
            and row.used_bytes >= row.budget_gib * GIB * 0.9
        ):
            print("  WARNING: at least 90% of the configured budget is used.")
    print("\nBudgets are personal targets, not free-tier guarantees or enforced limits.")
    print(
        "Queries: completed jobs created this UTC month, project-wide; running jobs are excluded."
    )
    print(
        "Storage: current dataset logical bytes; bucket includes versions and soft-deleted objects."
    )
    print("Network sent: delayed Monitoring estimate, not necessarily billable downloads.")
    print(
        "Storage is a current snapshot, not monthly average billed storage. Other projects/buckets are excluded."
    )
