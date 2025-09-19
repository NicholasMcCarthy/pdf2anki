"""Template management for note types and prompts with Jinja rendering."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
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


class PromptTemplate(BaseModel):
    """Prompt template definition."""
    name: str
    version: str
    description: str
    note_type: str
    safety_rule: str
    template: Optional[str] = None  # Embedded template
    template_file: Optional[str] = None  # External template file
    parameters: Dict[str, Any] = Field(default_factory=dict)
    token_limits: Dict[str, Any] = Field(default_factory=dict)


class NoteTypeManager:
    """Manages note type definitions."""
    
    def __init__(self, notes_dir: Path = Path("notes")):
        self.notes_dir = notes_dir
        self.note_types: Dict[str, NoteTypeDefinition] = {}
        self._load_note_types()
    
    def _load_note_types(self):
        """Load all note type definitions from the notes directory."""
        if not self.notes_dir.exists():
            logger.warning(f"Notes directory {self.notes_dir} does not exist")
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


class PromptManager:
    """Manages prompt templates with Jinja rendering."""
    
    def __init__(self, prompts_dir: Path = Path("prompts")):
        self.prompts_dir = prompts_dir
        self.templates_dir = prompts_dir / "templates"
        self.prompts: Dict[str, PromptTemplate] = {}
        
        # Set up Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(self.prompts_dir), str(self.templates_dir)]),
            undefined=StrictUndefined,
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt templates."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory {self.prompts_dir} does not exist")
            return
        
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                prompt = PromptTemplate(**data)
                self.prompts[prompt.name] = prompt
                logger.info(f"Loaded prompt: {prompt.name} v{prompt.version}")
                
            except Exception as e:
                logger.error(f"Failed to load prompt from {yaml_file}: {e}")
    
    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.prompts.get(name)
    
    def render_prompt(
        self,
        prompt_name: str,
        context: Dict[str, Any],
        truncate_tokens: Optional[int] = None
    ) -> str:
        """
        Render a prompt template with the given context.
        
        Args:
            prompt_name: Name of the prompt template
            context: Variables to pass to the template
            truncate_tokens: Maximum tokens for chunk_text (optional)
            
        Returns:
            Rendered prompt string
        """
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt template '{prompt_name}' not found")
        
        # Apply token truncation if specified
        if truncate_tokens and 'chunk_text' in context:
            context = context.copy()
            context['chunk_text'] = self._truncate_text(
                context['chunk_text'],
                truncate_tokens,
                prompt.token_limits.get('truncation_strategy', 'end')
            )
        
        # Add prompt parameters to context
        render_context = {**prompt.parameters, **context}
        render_context['safety_rule'] = prompt.safety_rule
        
        try:
            if prompt.template:
                # Use embedded template
                template = self.jinja_env.from_string(prompt.template)
            elif prompt.template_file:
                # Use external template file
                template = self.jinja_env.get_template(prompt.template_file)
            else:
                raise ValueError(f"Prompt '{prompt_name}' has no template or template_file")
            
            return template.render(**render_context)
            
        except Exception as e:
            logger.error(f"Failed to render prompt '{prompt_name}': {e}")
            raise
    
    def _truncate_text(self, text: str, max_tokens: int, strategy: str) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            strategy: Truncation strategy ('end', 'middle', 'start')
            
        Returns:
            Truncated text
        """
        # TODO: Implement proper token-aware truncation
        # For now, use character-based approximation (4 chars ≈ 1 token)
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        if strategy == 'end':
            return text[:max_chars] + "...[truncated]"
        elif strategy == 'start':
            return "[truncated]..." + text[-max_chars:]
        elif strategy == 'middle':
            half = max_chars // 2
            return text[:half] + "...[middle truncated]..." + text[-half:]
        else:
            return text[:max_chars] + "...[truncated]"
    
    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self.prompts.keys())


# Global instances (can be overridden for testing)
note_type_manager = NoteTypeManager()
prompt_manager = PromptManager()


def get_note_type_manager() -> NoteTypeManager:
    """Get the global note type manager instance."""
    return note_type_manager


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    return prompt_manager