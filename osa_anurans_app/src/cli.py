"""Command-line entry point for local workflows and explicit cloud actions.

Flow:
1. Parse arguments and load configuration.
2. Configure logs and show input/output paths.
3. Use the supplied command, or show the interactive menu.
4. Route the action to pipeline.py or publish.py.

Commands:
- import: Save window WAVs and embeddings; optional CSV for one-time seed labels.
- train: Train a classifier and save scored review candidates locally.
- both: Run import, then train.
- status: Show local catalog counts.
- publish: Upload WAVs to GCS, insert BigQuery metadata, then replace candidates.
- cleanup: Delete GCS window WAVs without current candidates or references.
- refresh: Read BigQuery answers into a local label snapshot.
- add-species: Add reference audio, local vectors, and cloud seed annotations.
- inference: Reserved; not implemented.
"""

import argparse
import json
import logging
import os
import warnings
from dataclasses import replace
from pathlib import Path

from .catalog import Catalog
from .config import default_config, load_config
from .console_ui import STEP_LABELS, Workflow, choose_run, configure_workflow
from .pipeline import import_local, refresh_annotations, train_local

logger = logging.getLogger(__name__)


class _ConsoleFormatter(logging.Formatter):
    """Keep routine output plain; label warnings, errors, and debug messages."""

    def format(self, record):
        message = super().format(record)
        return message if record.levelno == logging.INFO else f"{record.levelname} {message}"


def configure_cloud_environment(config) -> None:
    """Use the configured project for SDK discovery and quota billing.

    Rules:
    - Preserve explicit environment overrides.
    - Hide only Google's Python 3.11 end-of-support notice; errors stay visible.
    """
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", config.project)
    os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", config.project)
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a Python version .*google\.(api_core|cloud\.bigquery).*",
        category=FutureWarning,
        module=r"google\.api_core\._python_version_support",
    )


def parser() -> argparse.ArgumentParser:
    """Define global options and command-specific arguments.

    Rules:
    - Global options precede the command; command-specific options follow it.
    - No command selects interactive mode in main().
    - Defining/parsing commands does not run workflows or contact GCP.
    """
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        type=Path,
        default=default_config(),
        help="Settings file (default: config.local.toml when present, otherwise config.example.toml)",
    )
    result.add_argument("--artifacts", type=Path, help="Override the local output directory")
    result.add_argument("--verbose", action="store_true")
    # Keep the command optional so launching without one opens the menu.
    commands = result.add_subparsers(dest="action")
    for name in ("import", "both"):
        command = commands.add_parser(
            name,
            help="Import local audio/labels"
            if name == "import"
            else "Import, train, and generate local candidates",
        )
        command.add_argument(
            "--limit-recordings",
            type=int,
            help="Small smoke run; samples annotated recordings across species first",
        )
        command.add_argument(
            "--skip-invalid",
            action="store_true",
            help="Skip unavailable/invalid audio and save an exclusion report",
        )
    train = commands.add_parser(
        "train", help="Train and generate local candidates from saved annotations"
    )
    train.add_argument("--snapshot", type=Path)
    train.add_argument("--embedding-config")
    commands.add_parser("status", help="Show local catalog counts")
    commands.add_parser("refresh", help="Fetch BigQuery answers and save a consensus snapshot")
    publish = commands.add_parser("publish", help="Replace cloud candidates with a saved local run")
    publish.add_argument("run_directory", type=Path)
    publish.add_argument(
        "--cleanup", action="store_true", help="Remove unused cloud window clips after publication"
    )
    commands.add_parser(
        "cleanup", help="Remove cloud audio without current candidates or references"
    )
    commands.add_parser("inference", help="Reserved for later implementation")
    add = commands.add_parser(
        "add-species", help="Add reference audio and seed annotations to Google Cloud"
    )
    add.add_argument("--manifest", type=Path, required=True, help="JSON species/reference manifest")
    return result


def display_paths(config) -> None:
    """Print input paths, local output directories, and GCS/BigQuery destinations.

    Notes:
    - Makes the configured read/write locations visible before execution.
    - Displaying a cloud destination does not connect to it.
    """
    print("Recordings READ:", *(str(path) for path in config.audio_roots), sep="\n  ")
    if config.annotations_csv is not None:
        print(f"Initial annotations READ: {config.annotations_csv}")
    else:
        print("Labels: saved snapshot; refresh from BigQuery for new answers")
    print(f"Embeddings/audio/annotation snapshots READ/WRITE: {config.artifacts}")
    print(f"Classifier, metrics, and candidates WRITE: {config.artifacts / 'runs'}")
    print(
        f"Optional cloud destination: {config.project}.{config.dataset}, gs://{config.bucket}/{config.prefix}/"
    )


def _annotation_store(config):
    """Create a BigQuery client and check read access before fetching annotations.

    Steps:
    1. Read Application Default Credentials.
    2. Check access to the configured BigQuery dataset/tables.

    Returns:
    - BigQueryStore; annotation reads do not require GCS access.

    Unhappy paths:
    - Preserve credential/IAM errors and explain how to check local credentials.
    """
    from .cloud.bigquery import BigQueryStore

    try:
        logger.info("Connecting to BigQuery and checking annotation access...")
        bq = BigQueryStore(config)
        bq.check_access()
        logger.info("BigQuery access confirmed")
    except Exception as exc:
        raise RuntimeError(
            "BigQuery access check failed. Run 'gcloud auth application-default login' "
            "if credentials are missing/expired, and check BigQuery IAM permissions."
        ) from exc
    return bq


def _execute_cloud(args, config) -> None:
    """Execute one explicit cloud command.

    Actions:
    - refresh:
        1. Fetch annotator answers from BigQuery.
        2. Count present/absent votes and save a local label snapshot.
    - publish:
        1. Upload only the saved candidates' window WAVs to GCS.
        2. Populate their window metadata in BigQuery.
        3. Replace candidates from a saved run.

    Reuse rules:
    - Check GCS in bulk; restore missing clips even if registered locally.
    - Check existing BigQuery window IDs in batches; insert only missing rows.
    - Reject conflicting window metadata or GCS file contents.
    """
    from .publish import check_cloud_access, cleanup_cloud_audio, publish_candidates

    if args.action == "refresh":
        print("Annotation snapshot:", refresh_annotations(config, _annotation_store(config)))
        return
    bq, storage = check_cloud_access(config)
    with Catalog(config.artifacts / "catalog.sqlite") as catalog:
        if args.action == "cleanup":
            directory = getattr(args, "run_directory", None)
            cleanup_cloud_audio(
                config,
                catalog,
                bq,
                storage,
                publication_manifest=directory / "publish.json" if directory else None,
            )
            return
        directory = args.run_directory.resolve()
        manifest = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
        # Read the saved run's candidate count and JSONL path for BigQuery replacement.
        published = publish_candidates(
            config,
            catalog,
            directory / "candidates.jsonl",
            manifest["count"],
            directory.name,
            bq,
            storage,
            cleanup=getattr(args, "cleanup", False),
        )
        if published:
            print("Published candidates:", manifest["count"])
        else:
            print("Run was already published; current cloud candidate queue preserved.")


def execute(args, config, interactive: bool = False) -> None:
    """Call the function for the selected command.

    Order:
    1. Handle reserved, cloud, and status commands with early returns.
    2. For import/both, save window WAVs, embeddings, and CSV-derived labels.
    3. For train/both, train a classifier and save scored review candidates.

    Interactive mode:
    - Collect all workflow choices before processing starts.
    - No prompts during imports, annotation fetching, training, or uploads.

    Notes:
    - Direct train/both commands use saved labels and keep candidates local.
    """
    if args.action == "exit":
        return
    if args.action == "workflow":
        workflow = configure_workflow(config)
        if isinstance(workflow, str):
            execute(argparse.Namespace(action=workflow), config, interactive=True)
        elif workflow is not None:
            execute_workflow(workflow, config)
        return
    if args.action == "add-species":
        from .species import add_species, load_manifest

        print("Annotation snapshot:", add_species(config, load_manifest(args.manifest)))
        return
    if args.action == "inference":
        print("This action is reserved and has not been implemented yet.")
        return
    if args.action in {"refresh", "publish", "cleanup"}:
        if args.action == "publish" and interactive:
            directory = choose_run(config)
            if directory is not None:
                _execute_cloud(
                    argparse.Namespace(action="publish", run_directory=directory), config
                )
            return
        _execute_cloud(args, config)
        return
    if args.action == "status":
        with Catalog(config.artifacts / "catalog.sqlite") as catalog:
            print(f"Recordings: {len(catalog.recordings())}; windows: {len(catalog.windows())}")
            print("Embedding configurations:", ", ".join(catalog.config_ids()) or "none")
            print("Pending cloud windows:", len(catalog.pending_windows()))
        return
    # For both, import updates latest.json before training selects its inputs.
    if args.action in {"import", "both"}:
        import_local(
            config,
            limit_recordings=getattr(args, "limit_recordings", None),
            skip_invalid=getattr(args, "skip_invalid", False),
        )
    if args.action not in {"train", "both"}:
        return
    directory = train_local(
        config,
        snapshot=getattr(args, "snapshot", None),
        config_id=getattr(args, "embedding_config", None),
    )
    print("Local run saved:", directory)


def execute_workflow(workflow: Workflow, config) -> None:
    """Run the confirmed plan in order; any failure stops subsequent steps.

    Rules:
    - Publish only the run returned by this workflow's training step.
    - Completed local artifacts are retained if a later step fails.
    - Never request more input after processing starts.
    """
    workflow = Workflow.from_selection(list(workflow.steps), workflow.species_manifest)
    refreshed = False
    directory = None
    for index, step in enumerate(workflow.steps, start=1):
        print(flush=True)
        logger.info("Step %s/%s: %s", index, len(workflow.steps), STEP_LABELS[step])
        if step == "import":
            import_local(config)
        elif step == "add-species":
            from .species import add_species

            if workflow.species_manifest is None:
                raise ValueError("Select and review a species manifest before starting")
            add_species(config, workflow.species_manifest)
            refreshed = True
        elif step == "refresh":
            if refreshed:
                logger.info("Using the complete annotation snapshot fetched by Add species")
            else:
                refresh_annotations(config, _annotation_store(config))
        elif step == "train":
            directory = train_local(config)
            print("Local run saved:", directory)
        elif step == "publish":
            _execute_cloud(argparse.Namespace(action="publish", run_directory=directory), config)
        elif step == "cleanup":
            _execute_cloud(
                argparse.Namespace(
                    action="cleanup",
                    run_directory=directory if "publish" in workflow.steps else None,
                ),
                config,
            )
    print("Workflow completed.")


def main(argv=None) -> int:
    """Set up the CLI, execute one action, and return a process exit code.

    Steps:
    1. Parse arguments and configure console logging.
    2. Load configuration, apply the artifact override, and attach a file log.
    3. Display paths and select/execute an action.
    4. Close the file log even if execution fails.

    Exit codes:
    - 0: Action completed, including reserved-action messages.
    - 1: A handled workflow error occurred.
    - 130: Interrupted by the user; completed artifacts are retained.

    Notes:
    - argparse handles --help and invalid arguments before workflow execution.
    - argv=None reads process arguments; tests can pass an explicit list.
    """
    args = parser().parse_args(argv)
    console = logging.StreamHandler()
    console.setFormatter(_ConsoleFormatter("%(message)s"))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, handlers=[console])
    handler = None
    try:
        config = load_config(args.config)
        configure_cloud_environment(config)
        if args.artifacts:
            config = replace(config, artifacts=args.artifacts.resolve())
        config.artifacts.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(config.artifacts / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        # Gather the complete plan before starting any workflow steps.
        interactive = args.action is None
        if not interactive:
            print("Config:", args.config.resolve())
            display_paths(config)
        if interactive:
            args.action = "workflow"
        execute(args, config, interactive)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, TypeError) as error:
        logger.error("%s", error, exc_info=args.verbose)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted. Completed artifacts and pending cloud uploads are retained.")
        return 130
    finally:
        # Avoid leaving an open file/duplicate handler when main() is called again.
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
