"""Note-type definition management (loads notes/*.yaml).

This module used to also hold a second, parallel prompt-rendering system
(PromptTemplate/PromptManager, driving strategy_registry.py) alongside this
one - that system's LLM call was never implemented (strategy_registry.py's
generate_cards() had a literal "# TODO: Call LLM" and returned mock data)
and was never used by the real generation pipeline. It's been removed;
prompts.py::PromptManager + strategies/base.py::BaseStrategy is the only
prompt-rendering path now. NoteTypeManager stays - it's genuinely used, both
by `pdf2anki generate --plan-sample-csv` and (via note_types.py's
get_note_type_fields()) by the real prompt-rendering pipeline itself.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NoteTypeField(BaseModel):
    """Definition of a note type field."""
    description: str
    llm_instructions: str
    required: bool = True


class NoteTypeDefinition(BaseModel):
    """Complete note type definition."""
    name: str
    version: str
    description: str
    fields: Dict[str, NoteTypeField]
    rendering: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)


class NoteTypeManager:
    """Manages note type definitions."""

    def __init__(self, notes_dir: Path = Path("notes")):
        self.notes_dir = notes_dir
        self.note_types: Dict[str, NoteTypeDefinition] = {}
        self._load_note_types()

    def _load_note_types(self):
        """Load all note type definitions from the notes directory."""
        if not self.notes_dir.exists():
            # Debug, not warning: the module-level `note_type_manager` singleton
            # below is constructed with the default "notes" (relative to cwd)
            # at import time just because cli.py imports names from this module
            # - which happens on every CLI invocation, regardless of command -
            # so a missing default notes/ dir is an expected, harmless condition
            # for most callers, not something worth surfacing as a warning.
            logger.debug(f"Notes directory {self.notes_dir} does not exist")
            return

        for yaml_file in self.notes_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                note_type = NoteTypeDefinition(**data)
                self.note_types[note_type.name] = note_type
                logger.info(f"Loaded note type: {note_type.name} v{note_type.version}")

            except Exception as e:
                logger.error(f"Failed to load note type from {yaml_file}: {e}")

    def get_note_type(self, name: str) -> Optional[NoteTypeDefinition]:
        """Get a note type definition by name."""
        return self.note_types.get(name)

    def list_note_types(self) -> List[str]:
        """List all available note type names."""
        return list(self.note_types.keys())

    def get_csv_fields(self, note_type_name: str) -> List[str]:
        """Get CSV field order for a note type, including provenance fields."""
        note_type = self.get_note_type(note_type_name)
        if not note_type:
            return []

        # Start with note type fields
        fields = note_type.rendering.get("csv_fields_order", list(note_type.fields.keys()))

        # Add provenance fields
        provenance_fields = [
            "id", "note_type", "deck", "tags", "media",
            "source_pdf", "page_start", "page_end", "section", "ref_citation",
            "llm_model", "llm_version", "strategy", "template_version",
            "created_at", "updated_at",
            "core_concept", "longtext", "original_text", "my_notes"
        ]

        return fields + provenance_fields


# Global instance (can be overridden for testing)
note_type_manager = NoteTypeManager()


def get_note_type_manager() -> NoteTypeManager:
    """Get the global note type manager instance."""
    return note_type_manager
