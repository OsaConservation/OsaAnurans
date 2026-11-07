"""Keyboard menus for choosing work before any processing starts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

if TYPE_CHECKING:
    from .species import SpeciesManifest

STEP_LABELS = {
    "import": "Import new audio",
    "add-species": "Add species references and seed annotations",
    "refresh": "Fetch latest annotations from BigQuery",
    "train": "Train classifier and generate candidates",
    "publish": "Upload candidates to Google Cloud",
    "cleanup": "Remove unused cloud audio clips",
}


@dataclass(frozen=True)
class Workflow:
    """Selected steps in execution order; uploads always use a newly trained run."""

    steps: tuple[str, ...]
    species_manifest: "SpeciesManifest | None" = None

    @classmethod
    def from_selection(cls, selected: list[str], species_manifest=None) -> "Workflow":
        """Order steps, require a selection, and include training for uploads."""
        steps = set(selected)
        if not steps or steps - STEP_LABELS.keys():
            raise ValueError("Select at least one supported workflow step")
        if "publish" in steps:
            steps.add("train")
        return cls(tuple(step for step in STEP_LABELS if step in steps), species_manifest)


def _ask(question):
    """Escape cancels; Ctrl+C propagates to the CLI's interruption handler."""
    bindings = KeyBindings()

    @bindings.add("escape", eager=True)
    def cancel(event):
        event.app.exit(result=None)

    # Path prompts already have merged bindings; compose instead of mutating them.
    original = question.application.key_bindings
    question.application.key_bindings = (
        merge_key_bindings([original, bindings]) if original is not None else bindings
    )
    return question.unsafe_ask()


def _select(message: str, choices):
    print()
    return _ask(
        questionary.select(
            message,
            choices=choices,
            pointer=">",
            instruction="(Up/Down: move | Enter: select | Esc: cancel)",
        )
    )


def workflow_labels(config) -> dict[str, str]:
    """Keep each step's source or destination beside its label."""
    roots = ", ".join(str(path) for path in config.audio_roots)
    return {
        "import": f"Import new audio (from {roots})",
        "add-species": f"Add species references and seed annotations (from {config.species_manifest_path})",
        "refresh": "Fetch latest annotations Google BigQuery",
        "train": f"Train classifier and generate candidates (save to {config.artifacts / 'runs'})",
        "publish": "Upload candidates to Google Storage and BigQuery",
        "cleanup": "Remove unused cloud audio clips (keep annotated and current candidate clips)",
    }


def _workflow_menu(selected: list[str], config):
    """Show step checkboxes and action buttons together on the opening screen."""
    checked = set(selected)
    rows = list(STEP_LABELS) + ["continue", "exit"]
    labels = {
        **workflow_labels(config),
        "continue": "Continue",
        "exit": "Exit",
    }
    position = 0
    error = ""
    bindings = KeyBindings()

    def render():
        fragments = [("bold", "Choose steps to run\n\n")]
        for index, row in enumerate(rows):
            if row == "continue":
                fragments.append(("", "\n"))
            marker = f"[{'x' if row in checked else ' '}] " if row in STEP_LABELS else ""
            style = "reverse" if index == position else ""
            fragments.append(
                (style, f"{'>' if index == position else ' '} {marker}{labels[row]}\n")
            )
        fragments.append(
            ("", "\nUp/Down: move | Space: toggle | Enter: continue/select | Esc: exit")
        )
        if error:
            fragments.append(("", f"\n{error}"))
        return fragments

    @bindings.add("up")
    def up(event):
        nonlocal position
        position = (position - 1) % len(rows)

    @bindings.add("down")
    def down(event):
        nonlocal position
        position = (position + 1) % len(rows)

    @bindings.add(" ")
    def toggle(event):
        nonlocal error
        row = rows[position]
        if row not in STEP_LABELS:
            return
        if row in checked:
            checked.remove(row)
        else:
            checked.add(row)
        error = ""

    @bindings.add("enter")
    def submit(event):
        nonlocal error
        row = rows[position]
        if row == "continue" or row in STEP_LABELS:
            if not checked:
                error = "Select at least one step to continue."
                return
            event.app.exit(result=[step for step in STEP_LABELS if step in checked])
            return
        event.app.exit(result=row)

    @bindings.add("c-c")
    def interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())

    control = FormattedTextControl(render, focusable=True)
    application = Application(layout=Layout(Window(control)), key_bindings=bindings)
    return _ask(questionary.Question(application))


def configure_workflow(config) -> Workflow | str | None:
    """Choose steps, review the plan, then start or return to editing."""
    selected = ["refresh", "train"]
    manifest = None
    while True:
        selection = _workflow_menu(selected, config)
        if selection is None or isinstance(selection, str):
            return selection
        selected = selection
        if "add-species" in selected:
            from .species import load_manifest

            try:
                manifest = load_manifest(config.species_manifest_path)
            except (ValueError, OSError) as exc:
                print(f"Cannot read species manifest {config.species_manifest_path}: {exc}")
                continue
        workflow = Workflow.from_selection(selected, manifest)
        if "publish" in selected and "train" not in selected:
            print("Training included: upload uses the candidates from this new run.")
        selected = list(workflow.steps)
        display_workflow_review(workflow, config)
        decision = _select("Continue?", ["Start workflow", "Go back", "Cancel"])
        if decision == "Start workflow":
            return workflow
        if decision != "Go back":
            return None


def display_workflow_review(workflow: Workflow, config) -> None:
    """Keep each action's essential effects directly beneath its numbered step."""
    print("\nReady to run:")
    for index, step in enumerate(workflow.steps, start=1):
        title = STEP_LABELS[step]
        if step == "refresh" and "add-species" in workflow.steps:
            title = "Use annotations refreshed when adding species"
        print(f"\n  {index}. {title}")
        if step == "import":
            print(f"     Read audio from: {', '.join(str(path) for path in config.audio_roots)}")
        elif step == "add-species":
            manifest = workflow.species_manifest
            print(f"     Read species from: {manifest.path}")
            for entry in manifest.entries:
                print(f"     {entry.species_id}: {entry.common_name}")
            count = len(manifest.entries)
            print(
                f"     Upload {count} reference clip{'s' if count != 1 else ''} and add presence labels."
            )
        elif step == "publish":
            print("     Uploading replaces the website's candidate queue.")
        elif step == "cleanup":
            print("     Delete clips with no annotations or current candidates.")
    print("\nAll steps run automatically after starting.")


def choose_run(config) -> Path | None:
    """Choose a saved candidate run and confirm before contacting Google Cloud."""
    runs = sorted(
        (
            path.parent
            for path in (config.artifacts / "runs").glob("*/candidates.json")
            if (path.parent / "candidates.jsonl").is_file()
        ),
        reverse=True,
    )
    if not runs:
        print("No saved candidate runs. Train a classifier first.")
        return None
    choices = []
    for directory in runs:
        manifest = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
        choices.append(
            questionary.Choice(
                f"{directory.name} ({manifest['count']} candidates)", value=directory
            )
        )
    directory = _select("Choose a saved run to upload", choices)
    if directory is None:
        return None
    print("Uploading replaces the website's candidate queue.")
    if _select("Upload this run?", ["Cancel", "Upload candidates"]) != "Upload candidates":
        return None
    return directory
