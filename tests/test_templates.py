"""Tests for template management system."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.pdf2anki.templates import NoteTypeManager


class TestNoteTypeManager:
    """Test note type loading and management."""
    
    def test_load_note_types(self, tmp_path):
        """Test loading note types from directory."""
        # Create test note type file
        note_file = tmp_path / "test_note.yaml"
        note_file.write_text("""
name: "test_basic"
version: "1.0"
description: "Test note type"
fields:
  front:
    description: "Question"
    llm_instructions: "Create a question"
    required: true
  back:
    description: "Answer"
    llm_instructions: "Provide an answer"
    required: true
rendering:
  csv_fields_order:
    - front
    - back
""")
        
        manager = NoteTypeManager(tmp_path)
        assert "test_basic" in manager.list_note_types()
        
        note_type = manager.get_note_type("test_basic")
        assert note_type is not None
        assert note_type.name == "test_basic"
        assert len(note_type.fields) == 2
    
    def test_get_csv_fields(self, tmp_path):
        """Test CSV field generation including provenance fields."""
        note_file = tmp_path / "basic.yaml"
        note_file.write_text("""
name: "basic"
version: "1.0"
description: "Basic note"
fields:
  front:
    description: "Question"
    llm_instructions: "Create a question"
  back:
    description: "Answer"
    llm_instructions: "Provide an answer"
rendering:
  csv_fields_order:
    - front
    - back
""")
        
        manager = NoteTypeManager(tmp_path)
        fields = manager.get_csv_fields("basic")
        
        # Should include note type fields plus provenance fields
        assert "front" in fields
        assert "back" in fields
        assert "id" in fields
        assert "source_pdf" in fields
        assert "created_at" in fields
        assert len(fields) > 10  # Should have many provenance fields


def test_template_integration():
    """Test loading the actual shipped notes/*.yaml files."""
    # Use the actual notes directory
    note_manager = NoteTypeManager(Path("notes"))

    # Should load our example files
    note_types = note_manager.list_note_types()

    assert "basic" in note_types
    assert "cloze" in note_types

    # Test note type definition
    basic = note_manager.get_note_type("basic")
    assert basic is not None
    assert "front" in basic.fields
    assert "back" in basic.fields

    # Test CSV field generation
    csv_fields = note_manager.get_csv_fields("basic")
    assert len(csv_fields) > 10  # Should include provenance fields