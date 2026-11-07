"""Check that annotation reads need only BigQuery and preserve useful errors."""

import json

import pytest
from osa_anurans_app.src import cli
from osa_anurans_app.src.cloud import bigquery, storage
from osa_anurans_app.src.config import AppConfig


@pytest.mark.parametrize("interactive", [False, True])
def test_annotation_refresh_never_creates_or_checks_gcs(tmp_path, monkeypatch, interactive):
    config = AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path / "artifacts")
    checks = []

    class AnnotationStore:
        def __init__(self, config):
            pass

        def check_access(self):
            checks.append("bigquery")

        def fetch_annotations(self):
            return []

    def deny_storage(*args, **kwargs):
        pytest.fail("Annotation refresh must not create a GCS client")

    monkeypatch.setattr(bigquery, "BigQueryStore", AnnotationStore)
    monkeypatch.setattr(storage, "StorageStore", deny_storage)
    args = cli.parser().parse_args(["train" if interactive else "refresh"])
    if interactive:

        def train(config, **kwargs):
            latest = json.loads((config.artifacts / "latest.json").read_text())
            assert latest["snapshot"]
            return tmp_path / "run"

        monkeypatch.setattr(cli, "train_local", train)
    if interactive:
        cli.execute_workflow(cli.Workflow.from_selection(["refresh", "train"]), config)
    else:
        cli.execute(args, config)
    assert checks == ["bigquery"]
    latest = json.loads((config.artifacts / "latest.json").read_text())
    assert latest["snapshot"]


def test_annotation_auth_error_preserves_cause_and_login_advice(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path)
    denied = PermissionError("BigQuery denied")

    def deny_bigquery(*args, **kwargs):
        raise denied

    monkeypatch.setattr(bigquery, "BigQueryStore", deny_bigquery)
    with pytest.raises(RuntimeError, match="application-default login") as caught:
        cli.execute(cli.parser().parse_args(["refresh"]), config)
    assert caught.value.__cause__ is denied


@pytest.mark.parametrize("override", [False, True])
def test_cloud_environment_uses_config_and_preserves_overrides(tmp_path, monkeypatch, override):
    import os
    import warnings

    config = AppConfig((tmp_path,), tmp_path / "annotations.csv", tmp_path)
    for key in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT"):
        if override:
            monkeypatch.setenv(key, "explicit-project")
        else:
            monkeypatch.delenv(key, raising=False)
    with warnings.catch_warnings():
        cli.configure_cloud_environment(config)
    expected = "explicit-project" if override else config.project
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == expected
    assert os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] == expected


@pytest.mark.parametrize(
    "selected",
    [
        ["import"],
        ["refresh"],
        ["train"],
        ["refresh", "train"],
        ["import", "refresh", "train", "publish"],
        ["publish"],
    ],
)
def test_workflow_runs_selected_steps_without_prompting(tmp_path, monkeypatch, selected):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    directory = config.artifacts / "runs" / "new-run"
    calls = []
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("Unexpected prompt"))
    monkeypatch.setattr(cli, "_annotation_store", lambda config: "bq")
    monkeypatch.setattr(cli, "import_local", lambda config: calls.append("import"))
    monkeypatch.setattr(cli, "refresh_annotations", lambda config, store: calls.append("refresh"))

    def train(config):
        calls.append("train")
        return directory

    def upload(args, config):
        assert args.run_directory == directory
        calls.append("publish")

    monkeypatch.setattr(cli, "train_local", train)
    monkeypatch.setattr(cli, "_execute_cloud", upload)
    workflow = cli.Workflow.from_selection(selected)
    cli.execute_workflow(workflow, config)
    assert calls == list(workflow.steps)


def test_failed_training_never_uploads_an_older_run(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")

    def fail(config):
        raise RuntimeError("Training failed")

    monkeypatch.setattr(cli, "train_local", fail)
    monkeypatch.setattr(cli, "_execute_cloud", lambda *args: pytest.fail("Unexpected upload"))
    with pytest.raises(RuntimeError, match="Training failed"):
        cli.execute_workflow(cli.Workflow.from_selection(["publish"]), config)


def test_cancelled_workflow_does_not_process_data(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    monkeypatch.setattr(cli, "configure_workflow", lambda config: None)
    monkeypatch.setattr(cli, "execute_workflow", lambda *args: pytest.fail("Unexpected processing"))
    args = cli.parser().parse_args([])
    args.action = "workflow"
    cli.execute(args, config, interactive=True)


def test_menu_upload_uses_selected_confirmed_run(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    directory = config.artifacts / "runs" / "selected"
    monkeypatch.setattr(cli, "choose_run", lambda config: directory)
    published = []
    monkeypatch.setattr(
        cli, "_execute_cloud", lambda args, config: published.append(args.run_directory)
    )
    cli.execute(cli.parser().parse_args(["publish", "."]), config, interactive=True)
    assert published == [directory]


def test_default_config_prefers_local_and_accepts_explicit_override(tmp_path, monkeypatch):
    from osa_anurans_app.src.config import default_config

    example = tmp_path / "config.example.toml"
    example.touch()
    assert default_config(tmp_path) == example
    local = tmp_path / "config.local.toml"
    local.touch()
    assert default_config(tmp_path) == local
    monkeypatch.setattr(cli, "default_config", lambda: default_config(tmp_path))
    assert cli.parser().parse_args([]).config == local
    assert cli.parser().parse_args(["--config", str(example)]).config == example
