"""Read-only reporting: billed-byte accounting, missing metrics, and personal budgets."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from osa_anurans_app.src import cli, console_ui
from osa_anurans_app.src.cloud import usage
from osa_anurans_app.src.config import AppConfig, load_config, validate_config

START = datetime(2026, 11, 1, tzinfo=UTC)
END = datetime(2026, 11, 10, tzinfo=UTC)


class Session:
    def __init__(self, *pages):
        self.pages = iter(pages)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, dict(params)))
        data = next(self.pages)
        return SimpleNamespace(ok=True, json=lambda: data)


def job(name, billed=None, **statistics):
    query = {} if billed is None else {"totalBytesBilled": str(billed)}
    return {
        "jobReference": {"jobId": name, "location": "us-south1"},
        "configuration": {"jobType": "QUERY"},
        "status": {"state": "DONE"},
        "statistics": {"query": query, **statistics},
    }


def test_monthly_queries_follow_pages_and_count_scripts_once():
    script = job("script", 300)
    failed = job("failed", 50)
    failed["status"]["errorResult"] = {"reason": "invalidQuery"}
    pending = job("running", 1000)
    pending["status"]["state"] = "RUNNING"
    cached = job("cached", 500)
    cached["statistics"]["query"]["cacheHit"] = True
    dry = job("dry", 500)
    dry["configuration"]["dryRun"] = True
    load = job("load", 500)
    load["configuration"]["jobType"] = "LOAD"
    session = Session(
        {"jobs": [job("query", 100), script, pending, cached, dry, load], "nextPageToken": "next"},
        {"jobs": [script, job("child", 300, parentJobId="script"), failed]},
    )
    assert usage.query_usage(session, "project", START, END) == 450
    assert session.calls[0][1]["allUsers"] == "true"
    assert session.calls[0][1]["minCreationTime"] == int(START.timestamp() * 1000)
    assert session.calls[1][1]["pageToken"] == "next"
    assert all("/jobs" in url for url, _ in session.calls)  # No SQL endpoint.


def test_missing_query_statistics_are_not_reported_as_zero():
    with pytest.raises(RuntimeError, match="no billed-byte statistics"):
        usage.query_usage(Session({"jobs": [job("redacted")]}), "project", START, END)
    rejected = job("rejected")
    rejected["status"]["errorResult"] = {"reason": "accessDenied"}
    assert usage.query_usage(Session({"jobs": [rejected]}), "project", START, END) == 0


def test_empty_job_history_is_zero_usage():
    assert usage.query_usage(Session({}), "project", START, END) == 0


def test_redacted_job_metadata_is_unavailable():
    redacted = job("hidden")
    del redacted["configuration"]
    with pytest.raises(RuntimeError, match="redacted"):
        usage.query_usage(Session({"jobs": [redacted]}), "project", START, END)


def test_storage_counts_versions_and_soft_deleted_objects(tmp_path):
    from osa_anurans_app.src.cloud.storage import StorageStore

    calls = []

    def list_blobs(bucket, **options):
        calls.append(options)
        sizes = [10, 20] if options.get("versions") else [30]
        return [SimpleNamespace(size=size) for size in sizes]

    config = AppConfig((tmp_path,), None, tmp_path)
    store = StorageStore(config, client=SimpleNamespace(list_blobs=list_blobs))
    assert store.storage_usage(include_versions=True) == {
        "active_bytes": 30,
        "soft_deleted_bytes": 30,
    }
    assert calls == [{"versions": True}, {"soft_deleted": True}]


def test_downloads_sum_all_pages_and_filter_bucket():
    def series(*values):
        return {"points": [{"value": {"int64Value": str(v)}} for v in values]}

    session = Session(
        {"timeSeries": [series(10, 20)], "nextPageToken": "next"},
        {"timeSeries": [series(30)]},
    )
    assert usage.download_usage(session, "project", "bucket", START, END) == 60
    params = session.calls[0][1]
    assert 'resource.labels.bucket_name="bucket"' in params["filter"]
    assert params["aggregation.crossSeriesReducer"] == "REDUCE_SUM"
    assert session.calls[1][1]["pageToken"] == "next"


def test_missing_download_metrics_are_unavailable_not_zero():
    with pytest.raises(RuntimeError, match="no traffic samples"):
        usage.download_usage(Session({}), "project", "bucket", START, END)


def test_api_failure_does_not_expose_response_body():
    session = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(ok=False, status_code=403))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        usage.query_usage(session, "project", START, END)


def test_categories_fail_independently_and_use_current_month(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path)
    calls = []

    def network(config, reader, *args):
        calls.append((reader, args))
        if reader is usage.download_usage:
            raise RuntimeError("Monitoring permission missing")
        return 100

    monkeypatch.setattr(usage, "_network_measurement", network)
    monkeypatch.setattr(
        usage,
        "BigQueryStore",
        lambda c: SimpleNamespace(storage_usage=lambda: {"logical_bytes": 200}),
    )
    monkeypatch.setattr(
        usage,
        "StorageStore",
        lambda c: SimpleNamespace(
            storage_usage=lambda **k: {"active_bytes": 300, "soft_deleted_bytes": 20}
        ),
    )
    now, rows = usage.collect_usage(config, now=END)
    assert now == END
    assert [r.used_bytes for r in rows] == [100, 200, 320, None]
    assert rows[-1].error == "Monitoring permission missing"
    assert calls[0][1] == (START, END)
    assert calls[1][1] == (config.bucket, START, END)


def test_report_shows_overrun_missing_budget_and_unavailable(tmp_path, monkeypatch, capsys):
    config = AppConfig((tmp_path,), None, tmp_path)
    rows = [
        usage.UsageRow("queries", 100, 101 * usage.GIB),
        usage.UsageRow("network", None, 12 * usage.GIB),
        usage.UsageRow("storage", 10, error="Access denied"),
    ]
    monkeypatch.setattr(usage, "collect_usage", lambda c: (END, rows))
    usage.display_cloud_usage(config)
    output = capsys.readouterr().out
    assert "-1.000" in output and "WARNING" in output
    assert "not set" in output and "unavailable" in output and "Access denied" in output
    assert "not free-tier guarantees" in output


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), "10"])
def test_invalid_budgets_are_rejected(tmp_path, value):
    config = AppConfig((tmp_path,), None, tmp_path)
    with pytest.raises(ValueError, match="bq_query_budget_gib"):
        validate_config(replace(config, bq_query_budget_gib=value))


def test_usage_config_is_optional_and_overridable(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[paths]\naudio_roots=["inbox"]\n[usage]\nbq_query_budget_gib=900\ngcs_download_budget_gib=25\n'
    )
    config = load_config(path)
    assert config.bq_query_budget_gib == 900
    assert config.gcs_download_budget_gib == 25
    assert config.gcs_storage_budget_gib == 10


def test_cli_and_menu_route_cloud_usage_without_workflow(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path)
    calls = []
    monkeypatch.setattr(usage, "display_cloud_usage", lambda c: calls.append(c))
    cli.execute(cli.parser().parse_args(["cloud-usage"]), config)
    monkeypatch.setattr(console_ui, "_workflow_menu", lambda *args: "cloud-usage")
    assert console_ui.configure_workflow(config) == "cloud-usage"
    assert calls == [config]
