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


def test_serve_retry_errors_flag_calls_retry_before_run_forever(tmp_path):
    config_path = _write_config(tmp_path)

    with patch("pdf2anki.service.DirectoryWatcher") as MockWatcher:
        MockWatcher.return_value.retry_errors.return_value = 3
        result = runner.invoke(app, ["serve", "--config", str(config_path), "--retry-errors"])

    assert result.exit_code == 0, result.output
    MockWatcher.return_value.retry_errors.assert_called_once_with()
    MockWatcher.return_value.reset_state.assert_not_called()
    MockWatcher.return_value.run_forever.assert_called_once()
    assert "cleared 3" in result.output


def test_serve_reset_state_takes_priority_over_retry_errors(tmp_path):
    config_path = _write_config(tmp_path)

    with patch("pdf2anki.service.DirectoryWatcher") as MockWatcher:
        result = runner.invoke(
            app, ["serve", "--config", str(config_path), "--reset-state", "--retry-errors"]
        )

    assert result.exit_code == 0, result.output
    MockWatcher.return_value.reset_state.assert_called_once()
    MockWatcher.return_value.retry_errors.assert_not_called()


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


def test_watch_status_reports_counts_and_lists_errors(tmp_path):
    from pdf2anki.service.watcher import WatcherState

    config_path = _write_config(tmp_path)
    state_path = tmp_path / "workspace" / "watcher_state.json"
    state = WatcherState(state_path)
    state.record_success(tmp_path / "ok.pdf", 1)
    state.record_error(tmp_path / "bad.pdf", 2, "insufficient_quota: no credits remaining")

    result = runner.invoke(
        app, ["watch-status", "--config", str(config_path)], env={"COLUMNS": "300"}
    )

    assert result.exit_code == 0, result.output
    assert "1 processed successfully" in result.output
    assert "1 errored" in result.output
    assert "bad.pdf" in result.output
    assert "insufficient_quota" in result.output
    assert "ok.pdf" not in result.output  # --processed not passed, shouldn't list successes


def test_watch_status_processed_flag_lists_successes_too(tmp_path):
    from pdf2anki.service.watcher import WatcherState

    config_path = _write_config(tmp_path)
    state_path = tmp_path / "workspace" / "watcher_state.json"
    state = WatcherState(state_path)
    state.record_success(tmp_path / "ok.pdf", 1)

    result = runner.invoke(
        app, ["watch-status", "--config", str(config_path), "--processed"], env={"COLUMNS": "300"}
    )

    assert result.exit_code == 0, result.output
    assert "ok.pdf" in result.output


def test_watch_status_retry_errors_clears_state_without_reprocessing(tmp_path):
    from pdf2anki.service.watcher import WatcherState

    config_path = _write_config(tmp_path)
    state_path = tmp_path / "workspace" / "watcher_state.json"
    state = WatcherState(state_path)
    state.record_success(tmp_path / "ok.pdf", 1)
    state.record_error(tmp_path / "bad.pdf", 2, "boom")

    result = runner.invoke(app, ["watch-status", "--config", str(config_path), "--retry-errors"])

    assert result.exit_code == 0, result.output
    assert "Cleared 1" in result.output

    reloaded = WatcherState(state_path)
    assert reloaded.errors() == {}
    assert len(reloaded.successes()) == 1  # untouched


def test_watch_status_retry_errors_limited_to_path(tmp_path):
    from pdf2anki.service.watcher import WatcherState

    config_path = _write_config(tmp_path)
    state_path = tmp_path / "workspace" / "watcher_state.json"
    state = WatcherState(state_path)
    bad1 = tmp_path / "bad1.pdf"
    bad2 = tmp_path / "bad2.pdf"
    state.record_error(bad1, 1, "boom1")
    state.record_error(bad2, 2, "boom2")

    result = runner.invoke(
        app, ["watch-status", "--config", str(config_path), "--retry-errors", "--path", str(bad1)]
    )

    assert result.exit_code == 0, result.output
    reloaded = WatcherState(state_path)
    remaining = {Path(k).name for k in reloaded.errors()}
    assert remaining == {"bad2.pdf"}
