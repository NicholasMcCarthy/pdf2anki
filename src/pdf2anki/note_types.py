"""Shared PDF2Anki note-type ("model", in Anki terminology) definitions -
field lists, card templates, and CSS - used by both the local .apkg builder
(build.py, via genanki) and the live AnkiConnect push path
(service/ankiconnect.py).

Kept in one place so a note pushed live via AnkiConnect has the exact same
shape (fields/templates/styling) as one that ends up in the .apkg - see
service/ankiconnect.py:card_row_to_note and :ensure_note_types.

Also the loader for notes/*.yaml's per-field LLM instructions (see
get_note_type_fields() below), which strategies/base.py threads into every
generated prompt - so editing notes/basic.yaml/cloze.yaml changes what the
LLM is told to produce, without touching prompts/*.j2 directly.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .templates import NoteTypeManager

logger = logging.getLogger(__name__)

BASIC_MODEL_NAME = "PDF2Anki Basic"
CLOZE_MODEL_NAME = "PDF2Anki Cloze"

BASIC_FIELDS = ["Front", "Back", "Source", "Page", "Section", "Tags", "Extra"]
CLOZE_FIELDS = ["Text", "Extra", "Source", "Page", "Section", "Tags"]

BASIC_QFMT = '''
    <div class="question">{{Front}}</div>
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

BASIC_AFMT = '''
    <div class="question">{{Front}}</div>
    <hr>
    <div class="answer">{{Back}}</div>
    {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

BASIC_CSS = '''
    .card {
        font-family: Arial, sans-serif;
        font-size: 16px;
        text-align: left;
        color: #333;
        background-color: #fff;
        padding: 20px;
    }

    .question {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #2c3e50;
    }

    .answer {
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 15px;
    }

    .extra {
        font-size: 14px;
        color: #666;
        font-style: italic;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
    }

    .source {
        font-size: 12px;
        color: #888;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }

    .section {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }

    /* MathJax support */
    .MathJax {
        font-size: 1.1em !important;
    }

    /* Code styling */
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }

    pre {
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
'''

CLOZE_QFMT = '''
    <div class="cloze-question">{{cloze:Text}}</div>
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

CLOZE_AFMT = '''
    <div class="cloze-answer">{{cloze:Text}}</div>
    {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

CLOZE_CSS = '''
    .card {
        font-family: Arial, sans-serif;
        font-size: 16px;
        text-align: left;
        color: #333;
        background-color: #fff;
        padding: 20px;
    }

    .cloze-question, .cloze-answer {
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    .cloze {
        background-color: #007bff;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }

    .extra {
        font-size: 14px;
        color: #666;
        font-style: italic;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 3px solid #28a745;
    }

    .source {
        font-size: 12px;
        color: #888;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }

    .section {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }

    /* MathJax support */
    .MathJax {
        font-size: 1.1em !important;
    }
'''


# ---------------------------------------------------------------------------
# notes/*.yaml field-instruction loader
# ---------------------------------------------------------------------------

# Deliberately package-relative (three parents up from this file: pdf2anki/
# -> src -> repo/package root), NOT the CWD-relative default that
# templates.py's own module-level NoteTypeManager singleton uses - that one
# is fine for its only current caller (`--plan-sample-csv`, always invoked
# from a project root) but unsafe here, since prompt rendering must work
# regardless of process CWD (e.g. inside the Docker watcher service).
_DEFAULT_NOTES_DIR = Path(__file__).parent.parent.parent / "notes"

_default_note_type_manager = None  # lazy singleton; see _get_default_manager()

# Fallback field instructions, matching the hand-written prompt text every
# template used before this loader existed - used if notes/*.yaml is
# missing/unreadable, so a broken or absent notes/ dir degrades prompt
# quality rather than crashing card generation.
_FALLBACK_FIELD_INSTRUCTIONS = {
    "basic": {
        "front": "Clear, specific question",
        "back": "Complete answer with explanation",
    },
    "cloze": {
        "cloze_text": "Text with {{c1::a cloze deletion}} in it",
        "extra": "Additional context or explanation",
    },
}

# Note-type names we've already warned about missing/failing to load, so a
# broken notes/ dir logs once (not once per generate_cards() call, which
# would be once per chunk/highlight).
_warned_missing_note_types = set()


def _get_default_manager():
    global _default_note_type_manager
    if _default_note_type_manager is None:
        from .templates import NoteTypeManager
        _default_note_type_manager = NoteTypeManager(_DEFAULT_NOTES_DIR)
    return _default_note_type_manager


def get_note_type_fields(name: str, manager: Optional["NoteTypeManager"] = None) -> Dict[str, Dict[str, Any]]:
    """LLM-facing field definitions (description/llm_instructions/required)
    for the given note type ("basic"/"cloze"), read from notes/<name>.yaml.

    Returns plain nested dicts (not the pydantic NoteTypeField model) so
    templates can safely use dict .get() chaining - see prompts.py's
    render_template() and the "Choosing Basic vs. Cloze" / Output Format
    sections of prompts/*.j2.

    `manager` is an injection point for tests; production callers should
    omit it and get the lazily-constructed package-relative singleton.
    """
    note_type_manager = manager or _get_default_manager()
    note_type = note_type_manager.get_note_type(name)

    if note_type is None:
        if name not in _warned_missing_note_types:
            logger.warning(
                f"Note type '{name}' not found under {note_type_manager.notes_dir} - "
                f"falling back to built-in field instructions"
            )
            _warned_missing_note_types.add(name)
        return {
            field_name: {"description": text, "llm_instructions": text, "required": True}
            for field_name, text in _FALLBACK_FIELD_INSTRUCTIONS.get(name, {}).items()
        }

    return {
        field_name: {
            "description": field.description,
            "llm_instructions": field.llm_instructions,
            "required": field.required,
        }
        for field_name, field in note_type.fields.items()
    }
