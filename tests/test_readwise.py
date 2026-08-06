"""Tests for the Readwise markdown ingestion workflow: parsing, the batched
single-chunk construction (all highlights in one LLM call - see
readwise_document_to_single_chunk()), the ReadwiseHighlightStrategy, and the
generate-readwise CLI command. Uses the real sample Readwise/Obsidian export
the user provided as a fixture, plus a synthetic multi-highlight file for
edge cases."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pdf2anki.cli import app
from pdf2anki.io import find_markdown_files
from pdf2anki.llm import LLMResponse
from pdf2anki.prompts import create_prompt_manager
from pdf2anki.readwise import (
    parse_readwise_markdown,
    process_readwise_document,
    readwise_document_to_single_chunk,
)
from pdf2anki.strategies.readwise_highlight import ReadwiseHighlightStrategy

REAL_SAMPLE = Path(__file__).parent / "fixtures" / "readwise_texton_sample.md"

MULTI_HIGHLIGHT_MD = """---
created: 2024-05-01 10:00 AM
author: example.com
url: https://example.com/article
source: reader
category: articles
tags: [science, biology]
---
# Photosynthesis Basics [\U0001F310](https://example.com/article)

## Highlights
> [!info]  [\U0001F506](https://read.readwise.io/read/aaa111)
> Plants convert light energy into chemical energy via photosynthesis.

> [!note]
> This is the key mechanism I want to remember.

> [!info]  [\U0001F506](https://read.readwise.io/read/bbb222)
> The process occurs mainly in the chloroplasts of plant cells.
"""


@pytest.fixture
def multi_highlight_md(tmp_path) -> Path:
    path = tmp_path / "multi.md"
    path.write_text(MULTI_HIGHLIGHT_MD, encoding="utf-8")
    return path


def test_parses_real_sample_file():
    doc = parse_readwise_markdown(REAL_SAMPLE)

    assert doc.title == "Texton"
    assert doc.author == "wikipedia.org"
    assert doc.url == "https://en.wikipedia.org/wiki/Texton"
    assert doc.category == "articles"
    assert len(doc.highlights) == 1
    assert "texton" in doc.highlights[0].text.lower()
    assert doc.highlights[0].source_url == "https://read.readwise.io/read/01hs96ha4yj69beddekeae34aa"


def test_parses_multiple_highlights_with_note(multi_highlight_md):
    doc = parse_readwise_markdown(multi_highlight_md)

    assert doc.title == "Photosynthesis Basics"
    assert doc.tags == ["science", "biology"]
    assert len(doc.highlights) == 2

    first, second = doc.highlights
    assert "chemical energy" in first.text
    assert first.note == "This is the key mechanism I want to remember."
    assert first.source_url == "https://read.readwise.io/read/aaa111"

    assert "chloroplasts" in second.text
    assert second.note is None


def test_parses_file_with_no_highlights_section(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("---\nauthor: x.com\nurl: https://x.com\n---\n# Some Title\n\nJust prose, no highlights.\n")

    doc = parse_readwise_markdown(path)
    assert doc.title == "Some Title"
    assert doc.highlights == []


def test_readwise_document_to_single_chunk_numbers_highlights_and_notes(multi_highlight_md):
    doc = parse_readwise_markdown(multi_highlight_md)
    chunk = readwise_document_to_single_chunk(doc)

    assert chunk.section == "Photosynthesis Basics"
    assert chunk.total_chunks == 1
    assert chunk.chunk_index == 0
    assert "[HIGHLIGHT 1]: Plants convert light energy into chemical energy via photosynthesis." in chunk.text
    assert "[NOTE 1]: This is the key mechanism I want to remember." in chunk.text
    assert "[HIGHLIGHT 2]: The process occurs mainly in the chloroplasts of plant cells." in chunk.text
    assert "[NOTE 2]" not in chunk.text
    # Highlight 1's block (with its note) comes before highlight 2's.
    assert chunk.text.index("[HIGHLIGHT 1]") < chunk.text.index("[HIGHLIGHT 2]")


def test_readwise_document_to_single_chunk_handles_no_highlights():
    from pdf2anki.readwise import ReadwiseDocument

    doc = ReadwiseDocument(title="Empty Doc", highlights=[])
    chunk = readwise_document_to_single_chunk(doc)

    assert chunk.text == ""
    assert chunk.section == "Empty Doc"


def _mock_strategy() -> ReadwiseHighlightStrategy:
    from pdf2anki.config import Strategy as StrategyConfig

    llm_provider = Mock()
    llm_provider.config.model = "gpt-4"
    return ReadwiseHighlightStrategy(
        llm_provider=llm_provider,
        prompt_manager=create_prompt_manager(),
        strategy_config=StrategyConfig(enabled=True, params={}),
        strategy_name="readwise_highlight",
    )


def test_strategy_cites_source_url_not_page_number():
    strategy = _mock_strategy()

    from pdf2anki.chunking import TextChunk
    chunk = TextChunk(text="Plants convert light into chemical energy.", start_page=0, end_page=0)
    pdf_metadata = {"title": "Photosynthesis Basics", "source_url": "https://example.com/article", "category": "articles", "extra_tags": ["science"]}

    response_data = {"cards": [{"front": "Q", "back": "A", "core_concept": "X"}]}
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata)

    assert len(cards) == 1
    assert cards[0].ref_citation == "https://example.com/article"
    assert cards[0].page_citation == "https://example.com/article"
    assert "readwise" in cards[0].tags
    assert "articles" in cards[0].tags
    assert "science" in cards[0].tags
    assert 'href="https://example.com/article"' in cards[0].extra


def test_strategy_validate_response_accepts_cloze_card():
    strategy = _mock_strategy()
    response = {"cards": [{"card_type": "cloze", "cloze_text": "Photosynthesis occurs in the {{c1::chloroplasts}}."}]}
    assert strategy.validate_response(response) is True


def test_strategy_validate_response_rejects_malformed_cloze():
    strategy = _mock_strategy()
    response = {"cards": [{"card_type": "cloze", "cloze_text": "no markers"}]}
    assert strategy.validate_response(response) is False


def test_strategy_parses_mixed_basic_and_cloze_cards():
    strategy = _mock_strategy()

    from pdf2anki.chunking import TextChunk
    chunk = TextChunk(text="Plants convert light into chemical energy.", start_page=0, end_page=0)
    pdf_metadata = {"title": "Photosynthesis Basics", "source_url": "https://example.com/article"}

    response_data = {
        "cards": [
            {"card_type": "basic", "front": "Q1", "back": "A1", "core_concept": "X"},
            {"card_type": "cloze", "cloze_text": "Chlorophyll captures {{c1::light energy}}.",
             "extra": "some context", "core_concept": "Y"},
        ]
    }
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata)

    assert len(cards) == 2
    basic, cloze = cards
    assert basic.note_type == "Basic"
    assert cloze.note_type == "Cloze"
    assert cloze.cloze_text == "Chlorophyll captures {{c1::light energy}}."
    # Both still cite the source URL and get the source-link markup folded into extra.
    assert basic.ref_citation == "https://example.com/article"
    assert cloze.ref_citation == "https://example.com/article"
    assert "some context" in cloze.extra
    assert 'href="https://example.com/article"' in cloze.extra


def test_strategy_defaults_to_basic_without_card_type():
    strategy = _mock_strategy()

    from pdf2anki.chunking import TextChunk
    chunk = TextChunk(text="Some highlight.", start_page=0, end_page=0)

    response_data = {"cards": [{"front": "Q1", "back": "A1", "core_concept": "X"}]}
    cards = strategy.parse_cards(response_data, chunk, pdf_metadata={"title": "T"})

    assert cards[0].note_type == "Basic"


def test_process_readwise_document_makes_exactly_one_call_with_all_highlights(multi_highlight_md):
    """A multi-highlight document should be sent as ONE LLM call containing
    every highlight (numbered [HIGHLIGHT N]/[NOTE N] blocks), not one call
    per highlight."""
    captured_prompts = []

    def fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
        captured_prompts.append(prompt)
        payload = {"cards": [{"front": "Q", "back": "A", "core_concept": "X"}]}
        return LLMResponse(
            content=json.dumps(payload), model=self.config.model, tokens_used=1,
            cost_estimate=0.0, cached=False, response_time=0.0,
        )

    with patch("pdf2anki.llm.LLMProvider.generate", new=fake_generate):
        from pdf2anki.llm import create_llm_provider
        from pdf2anki.config import LLM as LLMConfig

        llm_provider = create_llm_provider(LLMConfig(provider="openai", api_key="dummy"))
        process_readwise_document(multi_highlight_md, llm_provider, create_prompt_manager())

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "chemical energy via photosynthesis" in prompt
    assert "chloroplasts" in prompt
    assert "[HIGHLIGHT 1]" in prompt
    assert "[HIGHLIGHT 2]" in prompt


def test_process_readwise_document_generates_cards(multi_highlight_md):
    def fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
        payload = {"cards": [{"front": "Q", "back": "A", "core_concept": "X"}]}
        return LLMResponse(
            content=json.dumps(payload), model=self.config.model, tokens_used=1,
            cost_estimate=0.0, cached=False, response_time=0.0,
        )

    with patch("pdf2anki.llm.LLMProvider.generate", new=fake_generate):
        from pdf2anki.llm import create_llm_provider
        from pdf2anki.config import LLM as LLMConfig

        llm_provider = create_llm_provider(LLMConfig(provider="openai", api_key="dummy"))
        cards = process_readwise_document(multi_highlight_md, llm_provider, create_prompt_manager())

    # One call, returning one card in the mocked response - a multi-highlight
    # document no longer multiplies the call (and therefore card) count by
    # highlight count.
    assert len(cards) == 1
    assert all(c.strategy == "readwise_highlight" for c in cards)


def test_find_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.markdown").write_text("# B")
    (tmp_path / "c.txt").write_text("not markdown")

    found = find_markdown_files([tmp_path])
    assert {p.name for p in found} == {"a.md", "b.markdown"}


@pytest.mark.integration
def test_generate_readwise_cli_merges_into_existing_csv(tmp_path, multi_highlight_md):
    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "workspace"
    config_path.write_text(f"""
pipeline:
  llm:
    provider: openai
    model: gpt-4-1106-preview
    api_key: dummy
generate:
  output:
    workspace: {workspace}
    csv_path: {workspace}/cards.csv
    media_path: {workspace}/media
    apkg_path: {workspace}/deck.apkg
    manifest_path: {workspace}/manifest.json
""")

    def fake_generate(self, prompt, system_prompt=None, json_mode=False, max_retries=3):
        payload = {"cards": [{
            "front": f"Q-{abs(hash(prompt)) % 10000}", "back": "A", "core_concept": "X",
        }]}
        return LLMResponse(
            content=json.dumps(payload), model=self.config.model, tokens_used=1,
            cost_estimate=0.0, cached=False, response_time=0.0,
        )

    runner = CliRunner()
    with patch("pdf2anki.llm.LLMProvider.generate", new=fake_generate):
        result = runner.invoke(app, [
            "generate-readwise", "--path", str(multi_highlight_md), "--config", str(config_path),
        ])
        assert result.exit_code == 0, result.output

    csv_path = workspace / "cards.csv"
    assert csv_path.exists()
    first_run_content = csv_path.read_text()
    first_run_rows = first_run_content.count("\n")
    assert "readwise_highlight" in first_run_content

    # Re-running against the same file should not duplicate rows (dedup by id).
    with patch("pdf2anki.llm.LLMProvider.generate", new=fake_generate):
        result = runner.invoke(app, [
            "generate-readwise", "--path", str(multi_highlight_md), "--config", str(config_path),
        ])
        assert result.exit_code == 0, result.output

    second_run_rows = csv_path.read_text().count("\n")
    assert second_run_rows == first_run_rows
