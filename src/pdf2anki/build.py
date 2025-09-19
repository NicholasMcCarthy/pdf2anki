"""Anki deck building system using genanki."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import genanki
import pandas as pd

from .config import AnkiConfig, Config, DeckStructure
from .io import load_csv

logger = logging.getLogger(__name__)


class AnkiDeckBuilder:
    """Builds Anki decks from CSV data using genanki."""
    
    def __init__(self, config: AnkiConfig):
        self.config = config
        self.note_types = {}
        self.decks = {}
        
        # Initialize default note types
        self._create_default_note_types()
    
    def _create_default_note_types(self) -> None:
        """Create default Anki note types."""
        
        # Basic note type
        basic_note_type = genanki.Model(
            model_id=random.randrange(1 << 30, 1 << 31),
            name='PDF2Anki Basic',
            fields=[
                {'name': 'Front'},
                {'name': 'Back'},
                {'name': 'Source'},
                {'name': 'Page'},
                {'name': 'Section'},
                {'name': 'Tags'},
                {'name': 'Extra'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '''
                        <div class="question">{{Front}}</div>
                        <div class="source">{{Source}} - {{Page}}</div>
                        {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
                    ''',
                    'afmt': '''
                        <div class="question">{{Front}}</div>
                        <hr>
                        <div class="answer">{{Back}}</div>
                        {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
                        <div class="source">{{Source}} - {{Page}}</div>
                        {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
                    ''',
                },
            ],
            css='''
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
        )
        
        # Cloze note type
        cloze_note_type = genanki.Model(
            model_id=random.randrange(1 << 30, 1 << 31),
            name='PDF2Anki Cloze',
            fields=[
                {'name': 'Text'},
                {'name': 'Extra'},
                {'name': 'Source'},
                {'name': 'Page'},
                {'name': 'Section'},
                {'name': 'Tags'},
            ],
            templates=[
                {
                    'name': 'Cloze',
                    'qfmt': '''
                        <div class="cloze-question">{{cloze:Text}}</div>
                        <div class="source">{{Source}} - {{Page}}</div>
                        {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
                    ''',
                    'afmt': '''
                        <div class="cloze-answer">{{cloze:Text}}</div>
                        {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
                        <div class="source">{{Source}} - {{Page}}</div>
                        {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
                    ''',
                },
            ],
            css='''
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
            ''',
            model_type=genanki.Model.CLOZE
        )
        
        self.note_types['Basic'] = basic_note_type
        self.note_types['Cloze'] = cloze_note_type
    
    def build_deck(self, csv_path: Path, output_path: Path, media_path: Optional[Path] = None) -> Dict[str, Any]:
        """Build Anki deck from CSV data."""
        
        logger.info(f"Building Anki deck from {csv_path}")
        
        # Load CSV data
        df = load_csv(csv_path)
        
        if df.empty:
            raise ValueError("No cards found in CSV file")
        
        # Create deck(s) based on structure configuration
        decks = self._create_decks(df)
        
        # Add notes to appropriate decks
        cards_added = 0
        note_types_used = set()
        
        for _, row in df.iterrows():
            try:
                note = self._create_note(row)
                if note:
                    # Determine which deck this card belongs to
                    deck = self._get_deck_for_card(row, decks)
                    deck.add_note(note)
                    cards_added += 1
                    note_types_used.add(row.get('note_type', 'Basic'))
                
            except Exception as e:
                logger.warning(f"Failed to create note for row {row.get('id', 'unknown')}: {e}")
                continue
        
        # Collect media files
        media_files = []
        if media_path and media_path.exists():
            media_files = list(media_path.glob('*'))
            logger.info(f"Found {len(media_files)} media files")
        
        # Create package
        package = genanki.Package(list(decks.values()))
        if media_files:
            package.media_files = [str(f) for f in media_files]
        
        # Write package
        output_path.parent.mkdir(parents=True, exist_ok=True)
        package.write_to_file(str(output_path))
        
        logger.info(f"Built Anki deck with {cards_added} cards")
        
        result = {
            "apkg_path": output_path,
            "total_cards": cards_added,
            "note_types": list(note_types_used),
            "deck_name": self.config.deck_name,
            "decks_created": len(decks),
            "media_files": len(media_files),
        }
        
        return result
    
    def _create_decks(self, df: pd.DataFrame) -> Dict[str, genanki.Deck]:
        """Create deck(s) based on the configured structure."""
        
        decks = {}
        
        if self.config.deck_structure == DeckStructure.FLAT:
            # Single flat deck
            deck_id = self.config.deck_id or random.randrange(1 << 30, 1 << 31)
            deck = genanki.Deck(deck_id, self.config.deck_name)
            decks['main'] = deck
            
        elif self.config.deck_structure == DeckStructure.BY_CHAPTER:
            # Create subdecks by section/chapter
            sections = df['section'].dropna().unique()
            
            for section in sections:
                if section and str(section).strip():
                    section_name = str(section).strip()
                    deck_name = f"{self.config.deck_name}::{section_name}"
                    deck_id = random.randrange(1 << 30, 1 << 31)
                    deck = genanki.Deck(deck_id, deck_name)
                    decks[section_name] = deck
            
            # Default deck for cards without sections
            if 'main' not in decks:
                deck_id = self.config.deck_id or random.randrange(1 << 30, 1 << 31)
                deck = genanki.Deck(deck_id, self.config.deck_name)
                decks['main'] = deck
                
        elif self.config.deck_structure == DeckStructure.BY_THEME:
            # Create subdecks by strategy/theme
            strategies = df['strategy'].dropna().unique()
            
            for strategy in strategies:
                if strategy and str(strategy).strip():
                    strategy_name = str(strategy).strip().replace('_', ' ').title()
                    deck_name = f"{self.config.deck_name}::{strategy_name}"
                    deck_id = random.randrange(1 << 30, 1 << 31)
                    deck = genanki.Deck(deck_id, deck_name)
                    decks[strategy] = deck
            
            # Default deck
            if 'main' not in decks:
                deck_id = self.config.deck_id or random.randrange(1 << 30, 1 << 31)
                deck = genanki.Deck(deck_id, self.config.deck_name)
                decks['main'] = deck
                
        else:  # PREDEFINED or fallback
            # Single deck with predefined structure
            deck_id = self.config.deck_id or random.randrange(1 << 30, 1 << 31)
            deck = genanki.Deck(deck_id, self.config.deck_name)
            decks['main'] = deck
        
        logger.info(f"Created {len(decks)} deck(s)")
        return decks
    
    def _get_deck_for_card(self, row: pd.Series, decks: Dict[str, genanki.Deck]) -> genanki.Deck:
        """Get the appropriate deck for a card based on deck structure."""
        
        if self.config.deck_structure == DeckStructure.BY_CHAPTER:
            section = row.get('section')
            if section and str(section).strip() in decks:
                return decks[str(section).strip()]
        
        elif self.config.deck_structure == DeckStructure.BY_THEME:
            strategy = row.get('strategy')
            if strategy and str(strategy).strip() in decks:
                return decks[str(strategy).strip()]
        
        # Default to main deck
        return decks.get('main', list(decks.values())[0])
    
    def _create_note(self, row: pd.Series) -> Optional[genanki.Note]:
        """Create an Anki note from a CSV row."""
        
        note_type_name = row.get('note_type', 'Basic')
        note_type = self.note_types.get(note_type_name)
        
        if not note_type:
            logger.warning(f"Unknown note type: {note_type_name}")
            note_type = self.note_types['Basic']  # Fallback
        
        try:
            if note_type_name == 'Basic':
                return self._create_basic_note(row, note_type)
            elif note_type_name == 'Cloze':
                return self._create_cloze_note(row, note_type)
            else:
                logger.warning(f"Unsupported note type for creation: {note_type_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {note_type_name} note: {e}")
            return None
    
    def _create_basic_note(self, row: pd.Series, note_type: genanki.Model) -> genanki.Note:
        """Create a basic note."""
        
        # Process LaTeX math if present
        front = self._process_math_content(str(row.get('front', '')))
        back = self._process_math_content(str(row.get('back', '')))
        
        # Build source information
        source_info = self._build_source_info(row)
        
        # Prepare tags
        tags = self._prepare_tags(row)
        
        note = genanki.Note(
            model=note_type,
            fields=[
                front,  # Front
                back,   # Back
                source_info['source'],  # Source
                source_info['page'],    # Page
                source_info['section'], # Section
                ';'.join(tags),        # Tags (for display)
                str(row.get('extra', '')),  # Extra
            ],
            tags=tags,
            guid=str(row.get('id', ''))
        )
        
        return note
    
    def _create_cloze_note(self, row: pd.Series, note_type: genanki.Model) -> genanki.Note:
        """Create a cloze deletion note."""
        
        # Process LaTeX math in cloze text
        cloze_text = self._process_math_content(str(row.get('cloze_text', '')))
        extra = self._process_math_content(str(row.get('extra', '')))
        
        # Build source information
        source_info = self._build_source_info(row)
        
        # Prepare tags
        tags = self._prepare_tags(row)
        
        note = genanki.Note(
            model=note_type,
            fields=[
                cloze_text,  # Text
                extra,       # Extra
                source_info['source'],  # Source
                source_info['page'],    # Page
                source_info['section'], # Section
                ';'.join(tags),        # Tags (for display)
            ],
            tags=tags,
            guid=str(row.get('id', ''))
        )
        
        return note
    
    def _process_math_content(self, content: str) -> str:
        """Process mathematical content for Anki MathJax support."""
        if not content or not self.config.preserve_latex:
            return content
        
        # Anki uses MathJax, so LaTeX should work as-is
        # Just ensure we don't have conflicting markdown formatting
        return content
    
    def _build_source_info(self, row: pd.Series) -> Dict[str, str]:
        """Build source information for the note."""
        
        source_pdf = Path(str(row.get('source_pdf', ''))).name if row.get('source_pdf') else 'Unknown'
        page_start = row.get('page_start', '')
        page_end = row.get('page_end', '')
        section = str(row.get('section', '')) if row.get('section') else ''
        
        # Format page range
        if page_start and page_end:
            if page_start == page_end:
                page_info = f"p. {page_start}"
            else:
                page_info = f"pp. {page_start}-{page_end}"
        elif page_start:
            page_info = f"p. {page_start}"
        else:
            page_info = ""
        
        return {
            'source': source_pdf,
            'page': page_info,
            'section': section,
        }
    
    def _prepare_tags(self, row: pd.Series) -> List[str]:
        """Prepare tags for the note."""
        
        tags = []
        
        # Add row tags
        row_tags = row.get('tags', [])
        if isinstance(row_tags, str):
            row_tags = [tag.strip() for tag in row_tags.split(';') if tag.strip()]
        elif isinstance(row_tags, list):
            row_tags = [str(tag).strip() for tag in row_tags if str(tag).strip()]
        
        tags.extend(row_tags)
        
        # Add strategy tag
        strategy = row.get('strategy')
        if strategy:
            tags.append(f"strategy:{strategy}")
        
        # Add note type tag
        note_type = row.get('note_type')
        if note_type:
            tags.append(f"type:{note_type.lower()}")
        
        # Clean and deduplicate tags
        clean_tags = []
        for tag in tags:
            # Remove invalid characters and normalize
            clean_tag = str(tag).strip().replace(' ', '_').lower()
            if clean_tag and clean_tag not in clean_tags:
                clean_tags.append(clean_tag)
        
        return clean_tags[:20]  # Limit to 20 tags


def build_anki_deck(config: Config, verbose: bool = False) -> Dict[str, Any]:
    """Build Anki deck from preprocessed data."""
    
    if verbose:
        logging.getLogger("pdf2anki").setLevel(logging.DEBUG)
    
    logger.info("Building Anki deck")
    
    # Initialize builder
    builder = AnkiDeckBuilder(config.anki)
    
    # Build deck
    result = builder.build_deck(
        csv_path=config.output.csv_path,
        output_path=config.output.apkg_path,
        media_path=config.output.media_path
    )
    
    logger.info(f"Anki deck built successfully: {result['apkg_path']}")
    return result