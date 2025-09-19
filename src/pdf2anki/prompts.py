"""Prompt template management using Jinja2."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates and rendering."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize prompt manager with template directory."""
        if templates_dir is None:
            # Default to package prompts directory
            templates_dir = Path(__file__).parent.parent.parent / "prompts"
        
        self.templates_dir = Path(templates_dir)
        
        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # Add custom filters
        self.env.filters['truncate_words'] = self._truncate_words
        self.env.filters['clean_text'] = self._clean_text
    
    def _truncate_words(self, text: str, num_words: int = 100) -> str:
        """Truncate text to specified number of words."""
        words = text.split()
        if len(words) <= num_words:
            return text
        return ' '.join(words[:num_words]) + "..."
    
    def _clean_text(self, text: str) -> str:
        """Clean text for use in prompts."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove special characters that might interfere with JSON
        text = text.replace('"', "'").replace('\n', ' ').replace('\r', '')
        return text
    
    def get_template(self, template_name: str) -> Template:
        """Get a template by name."""
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            raise
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with the given variables."""
        template = self.get_template(template_name)
        return template.render(**kwargs)
    
    def list_templates(self) -> list[str]:
        """List all available templates."""
        if not self.templates_dir.exists():
            return []
        
        templates = []
        for file in self.templates_dir.glob("*.j2"):
            templates.append(file.name)
        
        return sorted(templates)


# Default system prompts for different strategies
DEFAULT_SYSTEM_PROMPTS = {
    "key_points": """You are an expert educational content creator specializing in converting academic content into effective Anki flashcards. Your task is to identify key points from the provided text and create focused, testable flashcards.

Guidelines:
1. Focus on important concepts, definitions, facts, and relationships
2. Create clear, specific questions that test understanding
3. Avoid overly broad or vague questions
4. Include page citations for reference
5. Use active recall principles
6. Generate flashcards that test both recognition and application""",

    "cloze_definitions": """You are an expert at creating cloze deletion flashcards from academic content. Your task is to identify important terms, concepts, and their definitions, then create effective cloze deletions.

Guidelines:
1. Focus on key terms, definitions, formulas, and important numbers
2. Create cloze deletions that test essential knowledge
3. Use {{c1::text}} format for cloze deletions
4. Avoid creating too many cloze deletions in one card
5. Include context to make the cloze meaningful
6. Ensure the cloze tests understanding, not just memorization""",

    "figure_based": """You are an expert at creating flashcards based on figures, diagrams, tables, and visual content from academic materials. Your task is to analyze visual elements and create questions that test understanding of the visual information.

Guidelines:
1. Focus on interpreting charts, graphs, diagrams, and tables
2. Create questions about trends, relationships, and key data points
3. Test understanding of visual concepts and their implications
4. Include references to specific figures or tables
5. Create cards that connect visual information to broader concepts""",

    "reviewer": """You are an expert educational content reviewer. Your task is to evaluate flashcard quality and suggest improvements. Focus on clarity, accuracy, educational value, and adherence to best practices for spaced repetition learning.

Evaluation criteria:
1. Clarity: Is the question clear and unambiguous?
2. Accuracy: Is the information correct and properly sourced?
3. Educational value: Does it test important knowledge?
4. Difficulty: Is it appropriately challenging?
5. Format: Does it follow good flashcard design principles?

Provide a score from 1-10 and specific suggestions for improvement."""
}


# Template variables that are commonly available
COMMON_TEMPLATE_VARS = {
    "chunk": "The text chunk being processed",
    "section": "Section title if available", 
    "page_start": "Starting page number",
    "page_end": "Ending page number",
    "pdf_title": "PDF document title",
    "author": "Document author",
    "strategy": "The generation strategy being used",
    "max_cards": "Maximum number of cards to generate",
    "difficulty_level": "Target difficulty level",
    "language": "Target language for content",
}


def get_default_system_prompt(strategy: str) -> str:
    """Get default system prompt for a strategy."""
    return DEFAULT_SYSTEM_PROMPTS.get(strategy, DEFAULT_SYSTEM_PROMPTS["key_points"])


def create_prompt_manager(templates_dir: Optional[Path] = None) -> PromptManager:
    """Factory function to create a prompt manager."""
    return PromptManager(templates_dir)