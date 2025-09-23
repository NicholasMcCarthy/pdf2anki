# tests/test_cli_init.py

import io
from pathlib import Path
from unittest.mock import patch

from pdf2anki.cli import app


def _write(p: Path, txt: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")
    return p


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_init_creates_project_and_copies_all_assets(tmp_path: Path):
    """
    Verify `init`:
      - creates project dirs,
      - writes example configs in examples/ and samples/,
      - copies prompts (*.j2, *.yaml), notes (*.yaml), scripts (*.py),
      - respects non-overwrite on rerun, and overwrites with --force.
    """
    # Simulate package layout and how init resolves assets:
    # init uses Path(__file__).parent.parent.parent / "<assets>"
    # If we set __file__ = tmp_path / "pkgroot/pdf2anki/cli.py"
    # then init will look in (that path).parent.parent.parent == tmp_path
    pkgroot = tmp_path / "pkgroot"
    fake_cli_file = pkgroot / "pdf2anki" / "cli.py"
    _write(fake_cli_file, "# fake cli placeholder\n")

    # Assets must live at the computed base root seen by init:
    # base_root = Path(fake_cli_file).parent.parent.parent == tmp_path
    base_root = tmp_path
    src_prompts = base_root / "prompts"
    src_notes = base_root / "notes"
    src_scripts = base_root / "scripts"

    prompt_j2_src = _write(src_prompts / "default.j2", "{{ front }} :: {{ back }}\n")
    prompt_yaml_src = _write(src_prompts / "prompt.yaml", "name: default\n")
    note_yaml_src = _write(src_notes / "basic.yaml", "note_type: basic\nfields: [front, back]\n")
    script_py_src = _write(src_scripts / "generate_samples.py", "print('samples')\n")

    project_dir = tmp_path / "project"

    # First run: create everything
    with patch("pdf2anki.cli.__file__", str(fake_cli_file)):
        with patch("pdf2anki.cli.console"):  # silence console
            try:
                app(["init", str(project_dir)])
            except SystemExit as e:
                assert e.code == 0

    workspace_dir = project_dir / "workspace"
    prompts_dir = project_dir / "prompts"
    examples_dir = project_dir / "examples"
    notes_dir = project_dir / "notes"
    samples_dir = project_dir / "samples"
    scripts_dir = project_dir / "scripts"

    for d in [workspace_dir, prompts_dir, examples_dir, notes_dir, samples_dir, scripts_dir]:
        assert d.exists() and d.is_dir(), f"Missing directory: {d}"

    examples_cfg = examples_dir / "config.example.yaml"
    samples_cfg = samples_dir / "config.example.yaml"
    assert examples_cfg.exists(), "examples/config.example.yaml not created"
    assert samples_cfg.exists(), "samples/config.example.yaml not created"
    assert _read(examples_cfg).strip(), "examples config is empty"
    assert _read(samples_cfg).strip(), "samples config is empty"

    # Assets copied from base_root
    prompt_j2_dst = prompts_dir / prompt_j2_src.name
    prompt_yaml_dst = prompts_dir / prompt_yaml_src.name
    note_yaml_dst = notes_dir / note_yaml_src.name
    script_py_dst = scripts_dir / script_py_src.name

    assert prompt_j2_dst.exists(), "*.j2 prompt not copied"
    assert prompt_yaml_dst.exists(), "*.yaml prompt not copied"
    assert note_yaml_dst.exists(), "note type yaml not copied"
    assert script_py_dst.exists(), "script not copied"

    assert _read(prompt_j2_dst) == _read(prompt_j2_src)
    assert _read(prompt_yaml_dst) == _read(prompt_yaml_src)
    assert _read(note_yaml_dst) == _read(note_yaml_src)
    assert _read(script_py_dst) == _read(script_py_src)

    # Modify to check non-overwrite on rerun
    prompt_j2_dst.write_text("MODIFIED\n", encoding="utf-8")
    note_yaml_dst.write_text("note_type: basic-mod\n", encoding="utf-8")
    script_py_dst.write_text("print('modified')\n", encoding="utf-8")

    with patch("pdf2anki.cli.__file__", str(fake_cli_file)):
        with patch("pdf2anki.cli.console"):
            try:
                app(["init", str(project_dir)])  # no --force
            except SystemExit as e:
                assert e.code == 0

    assert _read(prompt_j2_dst) == "MODIFIED\n"
    assert _read(note_yaml_dst) == "note_type: basic-mod\n"
    assert _read(script_py_dst) == "print('modified')\n"

    # Now force overwrite
    with patch("pdf2anki.cli.__file__", str(fake_cli_file)):
        with patch("pdf2anki.cli.console"):
            try:
                app(["init", str(project_dir), "--force"])
            except SystemExit as e:
                assert e.code == 0

    assert _read(prompt_j2_dst) == _read(prompt_j2_src)
    assert _read(note_yaml_dst) == _read(note_yaml_src)
    assert _read(script_py_dst) == _read(script_py_src)
    assert examples_cfg.exists() and samples_cfg.exists()
    assert examples_cfg.stat().st_size > 0 and samples_cfg.stat().st_size > 0
