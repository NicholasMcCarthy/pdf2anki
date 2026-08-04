"""Regression test: every shipped .j2 prompt template must actually render.

cloze_definitions.j2 (previously named cloze_generation.j2, which the
ClozeDefinitionsStrategy never actually referenced - see strategies/
cloze_definitions.py's get_template_name()) contained literal Anki cloze
syntax like {{c1::term}} in its own example text, which Jinja2 tried to parse
as expressions and failed on. Both bugs meant the cloze strategy could never
actually run - confirmed neither was caught by any existing test."""

import pytest

from pdf2anki.prompts import create_prompt_manager


@pytest.mark.parametrize("template_name", [
    "key_points.j2",
    "cloze_definitions.j2",
    "figure_based.j2",
    "highlight_priority.j2",
    "readwise_highlight.j2",
    "reviewer.j2",
    "reviewer_special.j2",
])
def test_shipped_template_renders_without_error(template_name):
    pm = create_prompt_manager()
    rendered = pm.render_template(
        template_name,
        chunk="Sample chunk text.",
        section="Intro",
        page_start=1,
        page_end=1,
        pdf_title="Test Doc",
        author="",
        max_cards=3,
        card_data={"id": "1", "front": "Q", "back": "A"},
    )
    assert isinstance(rendered, str)
    assert rendered.strip()


def test_cloze_template_preserves_literal_cloze_syntax_as_text():
    pm = create_prompt_manager()
    rendered = pm.render_template(
        "cloze_definitions.j2", chunk="Some text.", page_start=1, page_end=1, pdf_title="T",
    )
    # The example syntax should survive rendering as literal text, not be
    # swallowed or misinterpreted as a Jinja expression.
    assert "{{c1::mitochondria}}" in rendered


def test_cloze_definitions_strategy_points_at_the_renamed_template():
    from pdf2anki.strategies.cloze_definitions import ClozeDefinitionsStrategy
    from unittest.mock import Mock

    strategy = ClozeDefinitionsStrategy.__new__(ClozeDefinitionsStrategy)
    assert strategy.get_template_name() == "cloze_definitions.j2"
    assert "cloze_definitions.j2" in create_prompt_manager().list_templates()
