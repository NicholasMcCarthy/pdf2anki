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


@pytest.mark.parametrize("template_name", [
    "key_points.j2",
    "cloze_definitions.j2",
    "figure_based.j2",
    "highlight_priority.j2",
    "readwise_highlight.j2",
])
def test_shipped_template_threads_real_note_type_instructions(template_name):
    """The Output Format JSON block's front/back/cloze_text/extra placeholder
    text is sourced from notes/basic.yaml and notes/cloze.yaml (see
    note_types.get_note_type_fields()), not hardcoded per-template - render
    with the real loaded fields and confirm the actual yaml content made it
    into the rendered prompt."""
    from pdf2anki.note_types import get_note_type_fields

    pm = create_prompt_manager()
    basic_fields = get_note_type_fields("basic")
    cloze_fields = get_note_type_fields("cloze")

    rendered = pm.render_template(
        template_name,
        chunk="Sample chunk text.",
        section="Intro",
        page_start=1,
        page_end=1,
        pdf_title="Test Doc",
        author="",
        abstract="",
        highlight_count=1,
        max_cards=3,
        basic_fields=basic_fields,
        cloze_fields=cloze_fields,
    )

    # key_points.j2/figure_based.j2 are Basic-only, cloze_definitions.j2 is
    # Cloze-only, highlight_priority.j2/readwise_highlight.j2 emit both -
    # check whichever field pair the template actually renders.
    if '"front"' in rendered:
        assert basic_fields["front"]["llm_instructions"] in rendered
        assert basic_fields["back"]["llm_instructions"] in rendered
    if '"cloze_text"' in rendered:
        assert cloze_fields["cloze_text"]["llm_instructions"] in rendered


def test_highlight_priority_prompt_offers_highlight_index_and_image_on_front():
    pm = create_prompt_manager()
    rendered = pm.render_template(
        "highlight_priority.j2", chunk="[HIGHLIGHT 1]: text", page_start=1, page_end=1, pdf_title="T",
    )
    assert "highlight_index" in rendered
    assert "image_on_front" in rendered
    assert "[HIGHLIGHT N]" in rendered


def test_highlight_priority_prompt_warns_against_unreferenceable_sections():
    pm = create_prompt_manager()
    rendered = pm.render_template(
        "highlight_priority.j2", chunk="[HIGHLIGHT 1]: text", page_start=1, page_end=1, pdf_title="T",
    )
    assert "Section 6.5.3" in rendered or "section number" in rendered.lower()


def test_highlight_priority_prompt_requires_atomic_cards():
    pm = create_prompt_manager()
    rendered = pm.render_template(
        "highlight_priority.j2", chunk="[HIGHLIGHT 1]: text", page_start=1, page_end=1, pdf_title="T",
    )
    assert "atomic" in rendered.lower()


def test_cloze_definitions_strategy_points_at_the_renamed_template():
    from pdf2anki.strategies.cloze_definitions import ClozeDefinitionsStrategy
    from unittest.mock import Mock

    strategy = ClozeDefinitionsStrategy.__new__(ClozeDefinitionsStrategy)
    assert strategy.get_template_name() == "cloze_definitions.j2"
    assert "cloze_definitions.j2" in create_prompt_manager().list_templates()
