"""Tests for note_types.get_note_type_fields() - the notes/*.yaml loader that
threads per-field LLM instructions into the real prompt-rendering pipeline
(see strategies/base.py::BaseStrategy.generate_cards())."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from pdf2anki import note_types
from pdf2anki.note_types import get_note_type_fields
from pdf2anki.templates import NoteTypeManager


@pytest.fixture(autouse=True)
def _reset_warned_once_cache():
    """get_note_type_fields() warns at most once per note-type name via a
    module-level set (so a long-running production process doesn't re-log
    on every generate_cards() call) - reset it between tests so one test's
    warning doesn't silently suppress another's assertion on the same name."""
    note_types._warned_missing_note_types.clear()
    yield
    note_types._warned_missing_note_types.clear()


def _write_note_type(tmp_path: Path, name: str, fields: dict) -> NoteTypeManager:
    field_yaml = "\n".join(
        f"""  {field_name}:
    description: "{f['description']}"
    llm_instructions: "{f['llm_instructions']}"
    required: {str(f.get('required', True)).lower()}"""
        for field_name, f in fields.items()
    )
    (tmp_path / f"{name}.yaml").write_text(f"""
name: "{name}"
version: "1.0"
description: "Test note type"
fields:
{field_yaml}
""")
    return NoteTypeManager(tmp_path)


def test_get_note_type_fields_reads_real_shipped_yaml():
    """The package's actual notes/basic.yaml and notes/cloze.yaml."""
    basic = get_note_type_fields("basic")
    assert set(basic.keys()) == {"front", "back"}
    assert basic["front"]["required"] is True
    assert "question" in basic["front"]["description"].lower()

    cloze = get_note_type_fields("cloze")
    assert set(cloze.keys()) == {"cloze_text", "extra"}
    assert cloze["extra"]["required"] is False


def test_get_note_type_fields_reflects_custom_yaml(tmp_path):
    """Editing a field's llm_instructions in notes/*.yaml changes what this
    returns - the core mechanism the prompts thread through."""
    manager = _write_note_type(tmp_path, "basic", {
        "front": {"description": "Q", "llm_instructions": "Ask something unusual"},
        "back": {"description": "A", "llm_instructions": "Answer briefly"},
    })

    fields = get_note_type_fields("basic", manager=manager)

    assert fields["front"]["llm_instructions"] == "Ask something unusual"
    assert fields["back"]["llm_instructions"] == "Answer briefly"


def test_get_note_type_fields_falls_back_when_note_type_missing(tmp_path, caplog):
    """A notes dir that exists but doesn't define the requested note type
    (or a manager pointed at an empty/missing dir) must degrade to the
    built-in fallback text, not crash or return nothing usable."""
    empty_manager = NoteTypeManager(tmp_path)  # no yaml files written

    with caplog.at_level("WARNING", logger="pdf2anki.note_types"):
        fields = get_note_type_fields("basic", manager=empty_manager)

    assert set(fields.keys()) == {"front", "back"}
    assert fields["front"]["llm_instructions"]  # non-empty fallback text
    assert any("not found" in r.message for r in caplog.records)


def test_get_note_type_fields_warns_only_once_per_note_type(tmp_path, caplog):
    empty_manager = NoteTypeManager(tmp_path)

    with caplog.at_level("WARNING", logger="pdf2anki.note_types"):
        get_note_type_fields("basic", manager=empty_manager)
        get_note_type_fields("basic", manager=empty_manager)

    warnings = [r for r in caplog.records if "not found" in r.message and "'basic'" in r.message]
    assert len(warnings) <= 1


def test_get_note_type_fields_unknown_type_with_no_fallback_returns_empty(tmp_path):
    empty_manager = NoteTypeManager(tmp_path)
    fields = get_note_type_fields("some_future_note_type", manager=empty_manager)
    assert fields == {}


def test_get_note_type_fields_returns_plain_dicts_not_pydantic_models():
    """Templates use dict .get() chaining (basic_fields.get('front', {})...)
    - the return value must be plain dicts, not pydantic NoteTypeField
    objects, or that chaining breaks."""
    basic = get_note_type_fields("basic")
    assert isinstance(basic["front"], dict)
    assert basic["front"].get("llm_instructions") is not None


def test_generate_cards_threads_custom_note_type_instructions_into_prompt(tmp_path, monkeypatch):
    """End-to-end: BaseStrategy.generate_cards() -> real Jinja render -> the
    actual LLM-bound prompt text reflects notes/basic.yaml's content - not
    just the loader function in isolation. This is the concrete scenario
    the feature exists for: edit notes/basic.yaml, see it in the prompt."""
    manager = _write_note_type(tmp_path, "basic", {
        "front": {"description": "Q", "llm_instructions": "UNIQUE_FRONT_MARKER_XYZ"},
        "back": {"description": "A", "llm_instructions": "UNIQUE_BACK_MARKER_XYZ"},
    })
    monkeypatch.setattr(note_types, "_get_default_manager", lambda: manager)

    from pdf2anki.chunking import TextChunk
    from pdf2anki.config import StrategyConfig
    from pdf2anki.prompts import create_prompt_manager
    from pdf2anki.strategies.key_points import KeyPointsStrategy

    llm_provider = Mock()
    llm_provider.config.model = "gpt-4"
    llm_provider.generate.return_value = Mock(content='{"cards": []}')

    strategy = KeyPointsStrategy(
        llm_provider=llm_provider,
        prompt_manager=create_prompt_manager(),
        strategy_config=StrategyConfig(enabled=True),
        strategy_name="key_points",
    )
    chunk = TextChunk(text="Some content.", start_page=1, end_page=1)
    strategy.generate_cards(chunk, pdf_metadata={"title": "T"}, max_cards=3)

    prompt_arg = llm_provider.generate.call_args.kwargs["prompt"]
    assert "UNIQUE_FRONT_MARKER_XYZ" in prompt_arg
    assert "UNIQUE_BACK_MARKER_XYZ" in prompt_arg
