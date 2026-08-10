"""Tests for the `pdf2anki serve` CLI command's watcher-state wiring:
default state_path derivation, --reset-state, and that a failed on_file
re-raises so DirectoryWatcher.handle()'s retry-on-failure logic actually
gets a chance to run for the real service (not just in watcher.py's own
unit tests)."""

from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from pdf2anki.cli import app

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "generate": {"output": {"workspace": str(tmp_path / "workspace")}},
                "service": {
                    "watch_dirs": {
                        "pdfs": str(tmp_path / "pdfs"),
                        "textbooks": str(tmp_path / "textbooks"),
                        "readwise": str(tmp_path / "readwise"),
                    }
                },
            }
        )
    )
    return config_path


def test_serve_defaults_state_path_into_workspace_and_calls_run_forever(tmp_path):
    config_path = _write_config(tmp_path)

    with patch("pdf2anki.service.DirectoryWatcher") as MockWatcher:
        result = runner.invoke(app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert MockWatcher.call_count == 1
    _, kwargs = MockWatcher.call_args
    assert kwargs["state_path"] == tmp_path / "workspace" / "watcher_state.json"
    MockWatcher.return_value.reset_state.assert_not_called()
    MockWatcher.return_value.run_forever.assert_called_once()


def test_serve_reset_state_flag_calls_reset_before_run_forever(tmp_path):
    config_path = _write_config(tmp_path)

    with patch("pdf2anki.service.DirectoryWatcher") as MockWatcher:
        result = runner.invoke(app, ["serve", "--config", str(config_path), "--reset-state"])

    assert result.exit_code == 0, result.output
    MockWatcher.return_value.reset_state.assert_called_once()
    MockWatcher.return_value.run_forever.assert_called_once()


def test_serve_honors_explicit_state_path_from_config(tmp_path):
    config_path = tmp_path / "config.yml"
    explicit_state_path = tmp_path / "custom_state.json"
    config_path.write_text(
        yaml.safe_dump(
            {
                "generate": {"output": {"workspace": str(tmp_path / "workspace")}},
                "service": {
                    "watch_dirs": {
                        "pdfs": str(tmp_path / "pdfs"),
                        "textbooks": str(tmp_path / "textbooks"),
                        "readwise": str(tmp_path / "readwise"),
                    },
                    "state_path": str(explicit_state_path),
                },
            }
        )
    )

    with patch("pdf2anki.service.DirectoryWatcher") as MockWatcher:
        result = runner.invoke(app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    _, kwargs = MockWatcher.call_args
    assert kwargs["state_path"] == explicit_state_path


def test_serve_on_file_reraises_after_reporting_failure(tmp_path):
    """on_file must re-raise (not swallow) so DirectoryWatcher.handle()'s
    retry-on-failure logic can see the failure and skip marking the file
    as seen - see watcher.py::handle()."""
    config_path = _write_config(tmp_path)

    captured_on_file = {}

    class RecordingWatcher:
        def __init__(self, *args, **kwargs):
            captured_on_file["on_file"] = kwargs["on_file"]

        def reset_state(self):
            pass

        def run_forever(self):
            pass

    with patch("pdf2anki.service.DirectoryWatcher", RecordingWatcher), \
         patch("pdf2anki.service.process_new_file", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    on_file = captured_on_file["on_file"]
    try:
        on_file(tmp_path / "some.pdf")
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("on_file() should have re-raised the processing error")
