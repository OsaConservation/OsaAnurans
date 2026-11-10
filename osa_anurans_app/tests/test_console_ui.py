"""Exercise menu decisions and real terminal key bindings without cloud access."""

from dataclasses import replace

import pytest
import questionary
from osa_anurans_app.src import console_ui as ui
from osa_anurans_app.src.config import AppConfig
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


@pytest.fixture(autouse=True)
def terminal_session():
    """Build prompts without requiring a real Windows console in pytest."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield pipe


def test_checkbox_supports_arrows_space_and_enter():
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        question = questionary.checkbox(
            "Steps",
            choices=[
                questionary.Choice("Import", value="import"),
                questionary.Choice("Train", value="train"),
            ],
        )
        pipe.send_text(" \x1b[B \r")
        assert ui._ask(question) == ["import", "train"]


def test_escape_cancels_without_starting():
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        question = questionary.checkbox("Steps", choices=["Train"])
        pipe.send_text("\x1b")
        assert ui._ask(question) is None


@pytest.mark.parametrize("keys", ["\r", "\x1b", "\x03"])
def test_path_prompt_supports_enter_escape_and_interrupt(terminal_session, tmp_path, keys):
    default = str(tmp_path / "species.json")
    question = questionary.path("Species manifest", default=default)
    terminal_session.send_text(keys)
    if keys == "\x03":
        with pytest.raises(KeyboardInterrupt):
            ui._ask(question)
        return
    assert ui._ask(question) == (default if keys == "\r" else None)


def test_path_prompt_preserves_text_editing_bindings(terminal_session, tmp_path):
    question = questionary.path("Species manifest", default="old.json")
    replacement = str(tmp_path / "new.json")
    terminal_session.send_text("\x01\x0b" + replacement + "\r")
    assert ui._ask(question) == replacement


@pytest.mark.parametrize("selected", [[], ["unknown"]])
def test_invalid_workflow_is_rejected(selected):
    with pytest.raises(ValueError, match="Select at least one"):
        ui.Workflow.from_selection(selected)


def test_configure_upload_includes_generation_and_go_back_preserves_choices(
    tmp_path, monkeypatch, capsys
):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    # Upload alone becomes generate + upload; training remains an independent choice.
    answers = iter([["publish"], ["refresh", "train"]])
    decisions = iter(["Go back", "Start workflow"])
    questions = []

    def ask(question):
        questions.append(question)
        return next(answers)

    monkeypatch.setattr(ui, "_ask", ask)
    monkeypatch.setattr(ui, "_select", lambda *args: next(decisions))
    workflow = ui.configure_workflow(config)
    assert workflow.steps == ("refresh", "train")
    assert len(questions) == 2
    output = capsys.readouterr().out
    assert "Candidate generation included" in output
    assert "replaces the website's candidate queue" in output


@pytest.mark.parametrize("cancel_at", ["selection", "summary"])
def test_cancel_workflow(tmp_path, monkeypatch, cancel_at):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    monkeypatch.setattr(ui, "_ask", lambda _: None if cancel_at == "selection" else ["train"])
    monkeypatch.setattr(ui, "_select", lambda *args: None)
    assert ui.configure_workflow(config) is None


@pytest.mark.parametrize("confirm", [None, "Cancel", "Upload candidates"])
def test_saved_run_selection_requires_confirmation(tmp_path, monkeypatch, confirm):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    directory = config.artifacts / "runs" / "20261107"
    directory.mkdir(parents=True)
    (directory / "candidates.json").write_text('{"count": 50}')
    (directory / "candidates.jsonl").touch()
    answers = iter([directory, confirm])
    monkeypatch.setattr(ui, "_select", lambda *args: next(answers))
    expected = directory if confirm == "Upload candidates" else None
    assert ui.choose_run(config) == expected


def test_no_saved_runs_does_not_prompt(tmp_path, monkeypatch):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    monkeypatch.setattr(ui, "_select", lambda *args: pytest.fail("Unexpected prompt"))
    assert ui.choose_run(config) is None


def test_opening_screen_shows_checklist_and_toggles_with_space(terminal_session, tmp_path):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    terminal_session.send_text(" \r")
    assert ui._workflow_menu(["refresh", "train"], config) == ["import", "refresh", "train"]


def test_opening_screen_exit_does_not_run_checked_steps(terminal_session, tmp_path):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    terminal_session.send_text("\x1b[B" * (len(ui.STEP_LABELS) + 1) + "\r")
    assert ui._workflow_menu(["refresh", "train"], config) == "exit"


def test_empty_selection_cannot_continue(terminal_session, tmp_path):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    terminal_session.send_text("\r\x1b")
    assert ui._workflow_menu([], config) is None


def test_cleanup_is_an_independent_workflow_step():
    assert ui.Workflow.from_selection(["cleanup"]).steps == ("cleanup",)
    assert ui.Workflow.from_selection(["cleanup", "publish"]).steps == (
        "generate",
        "publish",
        "cleanup",
    )


def test_train_generate_and_upload_have_separate_dependencies():
    assert ui.Workflow.from_selection(["train"]).steps == ("train",)
    assert ui.Workflow.from_selection(["generate"]).steps == ("generate",)
    assert ui.Workflow.from_selection(["publish"]).steps == ("generate", "publish")
    assert ui.Workflow.from_selection(["train", "publish"]).steps == (
        "train",
        "generate",
        "publish",
    )


def test_default_workflow_keeps_training_and_generation_selected(tmp_path, monkeypatch, capsys):
    config = AppConfig((tmp_path,), None, tmp_path / "outputs")
    monkeypatch.setattr(ui, "_workflow_menu", lambda selected, config: selected)
    monkeypatch.setattr(ui, "_select", lambda *args: "Start workflow")
    assert ui.configure_workflow(config).steps == ("refresh", "train", "generate")
    output = capsys.readouterr().out
    assert "no candidate generation" in output
    assert "Use this workflow's training result" in output


def test_opening_menu_can_toggle_cleanup(terminal_session, tmp_path):
    config = AppConfig((tmp_path,), None, tmp_path / "artifacts")
    position = list(ui.STEP_LABELS).index("cleanup")
    terminal_session.send_text("\x1b[B" * position + " \r")
    assert ui._workflow_menu(["refresh", "train"], config) == ["refresh", "train", "cleanup"]


def test_workflow_labels_show_input_and_output_paths(tmp_path):
    config = AppConfig(
        (tmp_path / "inputs" / "audio-inbox",),
        None,
        tmp_path / "outputs",
        species_manifest=tmp_path / "inputs" / "species.json",
    )
    labels = ui.workflow_labels(config)
    assert str(config.audio_roots[0]) in labels["import"]
    assert str(config.species_manifest_path) in labels["add-species"]
    assert str(config.artifacts / "runs") in labels["train"]
    custom = tmp_path / "another.json"
    assert (
        str(custom) in ui.workflow_labels(replace(config, species_manifest=custom))["add-species"]
    )
    assert (
        replace(config, species_manifest=None).species_manifest_path == config.species_manifest_path
    )
