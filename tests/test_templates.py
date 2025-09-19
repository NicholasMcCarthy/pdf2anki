"""Tests for template management system."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.pdf2anki.templates import NoteTypeManager, PromptManager, NoteTypeDefinition, PromptTemplate


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


class TestPromptManager:
    """Test prompt template loading and rendering."""
    
    def test_load_embedded_prompt(self, tmp_path):
        """Test loading prompt with embedded template."""
        prompt_file = tmp_path / "test_prompt.yaml"
        prompt_file.write_text("""
name: "test_key_points"
version: "1.0"
description: "Test prompt"
note_type: "basic"
safety_rule: "Return empty list if not fact-like"
template: |
  Create flashcards from: {{ chunk_text }}
  Max cards: {{ max_cards | default(3) }}
parameters:
  max_cards:
    type: integer
    default: 3
""")
        
        manager = PromptManager(tmp_path)
        assert "test_key_points" in manager.list_prompts()
        
        prompt = manager.get_prompt("test_key_points")
        assert prompt is not None
        assert prompt.name == "test_key_points"
        assert prompt.template is not None
    
    def test_render_prompt(self, tmp_path):
        """Test prompt rendering with Jinja."""
        prompt_file = tmp_path / "render_test.yaml"
        prompt_file.write_text("""
name: "render_test"
version: "1.0"
description: "Test rendering"
note_type: "basic"
safety_rule: "Be safe"
template: |
  Text: {{ chunk_text }}
  Cards: {{ max_cards | default(5) }}
  Rule: {{ safety_rule }}
""")
        
        manager = PromptManager(tmp_path)
        
        rendered = manager.render_prompt("render_test", {
            "chunk_text": "Sample content",
            "max_cards": 3
        })
        
        assert "Text: Sample content" in rendered
        assert "Cards: 3" in rendered
        assert "Rule: Be safe" in rendered
    
    def test_external_template_file(self, tmp_path):
        """Test loading external template file."""
        # Create templates subdirectory
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        
        # Create external template
        template_file = templates_dir / "external.j2"
        template_file.write_text("External template: {{ chunk_text }}")
        
        # Create prompt that references it
        prompt_file = tmp_path / "external_prompt.yaml"
        prompt_file.write_text("""
name: "external_test"
version: "1.0"
description: "External template test"
note_type: "basic"
safety_rule: "Be safe"
template_file: "templates/external.j2"
""")
        
        manager = PromptManager(tmp_path)
        
        rendered = manager.render_prompt("external_test", {
            "chunk_text": "Test content"
        })
        
        assert "External template: Test content" in rendered


def test_template_integration():
    """Test integration between note types and prompts using actual files."""
    # Use the actual notes and prompts directories
    note_manager = NoteTypeManager(Path("notes"))
    prompt_manager = PromptManager(Path("prompts"))
    
    # Should load our example files
    note_types = note_manager.list_note_types()
    prompts = prompt_manager.list_prompts()
    
    assert "basic" in note_types
    assert "cloze" in note_types
    assert len(prompts) >= 2  # Should have key_points and cloze_definitions
    
    # Test note type definition
    basic = note_manager.get_note_type("basic")
    assert basic is not None
    assert "front" in basic.fields
    assert "back" in basic.fields
    
    # Test CSV field generation
    csv_fields = note_manager.get_csv_fields("basic")
    assert len(csv_fields) > 10  # Should include provenance fields