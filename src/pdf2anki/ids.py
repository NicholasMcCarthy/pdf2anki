"""ID generation and management for Anki cards."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

from .config import IdStrategy, IdsConfig
from .strategies.base import FlashcardData

logger = logging.getLogger(__name__)


class IDManager:
    """Manages ID generation for flashcards."""
    
    def __init__(self, config: IdsConfig):
        self.config = config
        self._persistent_ids: Dict[str, str] = {}
        self._id_counter = 0
    
    def generate_id(self, card: FlashcardData) -> str:
        """Generate an ID for a flashcard based on the configured strategy."""
        if self.config.strategy == IdStrategy.CONTENT_HASH:
            return self._generate_content_hash_id(card)
        elif self.config.strategy == IdStrategy.PERSISTENT:
            return self._generate_persistent_id(card)
        else:
            raise ValueError(f"Unknown ID strategy: {self.config.strategy}")
    
    def _generate_content_hash_id(self, card: FlashcardData) -> str:
        """Generate a deterministic ID based on card content."""
        # Create a normalized representation of the card content
        content_parts = [
            self.config.salt,
            card.note_type,
            self._normalize_field_content(card),
            card.source_pdf,
            f"{card.page_start}-{card.page_end}",
        ]
        
        content_string = "|".join(str(part) for part in content_parts)
        
        # Use BLAKE2 for the hash (faster than SHA256, good collision resistance)
        hash_obj = hashlib.blake2b(content_string.encode('utf-8'), digest_size=16)
        return hash_obj.hexdigest()
    
    def _normalize_field_content(self, card: FlashcardData) -> str:
        """Normalize card content for consistent hashing."""
        if card.note_type == "Basic":
            # For basic cards, use front + back
            front = card.front or ""
            back = card.back or ""
            content = f"{front}||{back}"
        elif card.note_type == "Cloze":
            # For cloze cards, use cloze_text + extra
            cloze_text = card.cloze_text or ""
            extra = card.extra or ""
            content = f"{cloze_text}||{extra}"
        else:
            # Fallback: use all available text fields
            content = f"{card.front or ''}||{card.back or ''}||{card.cloze_text or ''}||{card.extra or ''}"
        
        # Normalize whitespace and case for consistency
        content = " ".join(content.split())
        return content.lower()
    
    def _generate_persistent_id(self, card: FlashcardData) -> str:
        """Generate a persistent ID that survives edits."""
        # Create a key for looking up existing IDs
        key = self._create_persistence_key(card)
        
        # Check if we already have an ID for this card
        if key in self._persistent_ids:
            return self._persistent_ids[key]
        
        # Generate a new ID
        self._id_counter += 1
        new_id = f"{self.config.salt}_{self._id_counter:06d}"
        
        # Store the mapping
        self._persistent_ids[key] = new_id
        
        return new_id
    
    def _create_persistence_key(self, card: FlashcardData) -> str:
        """Create a key for persistent ID mapping."""
        # Use a combination of source location and core concept
        key_parts = [
            card.source_pdf,
            str(card.page_start),
            card.core_concept,
            card.note_type,
            card.strategy,
        ]
        
        return "|".join(str(part) for part in key_parts)
    
    def load_persistent_ids(self, csv_path: Optional[Path] = None) -> None:
        """Load existing persistent IDs from a CSV file."""
        if not csv_path or not csv_path.exists():
            return
        
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path)
            
            if 'id' not in df.columns:
                return
            
            # Build mapping from persistence key to ID
            for _, row in df.iterrows():
                try:
                    # Reconstruct the persistence key
                    key_parts = [
                        str(row.get('source_pdf', '')),
                        str(row.get('page_start', '')),
                        str(row.get('core_concept', '')),
                        str(row.get('note_type', '')),
                        str(row.get('strategy', '')),
                    ]
                    key = "|".join(key_parts)
                    
                    self._persistent_ids[key] = str(row['id'])
                    
                except Exception as e:
                    logger.warning(f"Failed to load persistent ID for row: {e}")
                    continue
            
            # Update counter to avoid collisions
            if self._persistent_ids:
                max_counter = 0
                for existing_id in self._persistent_ids.values():
                    if existing_id.startswith(f"{self.config.salt}_"):
                        try:
                            counter = int(existing_id.split("_")[-1])
                            max_counter = max(max_counter, counter)
                        except (ValueError, IndexError):
                            continue
                
                self._id_counter = max_counter
            
            logger.info(f"Loaded {len(self._persistent_ids)} persistent IDs from {csv_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent IDs from {csv_path}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about ID generation."""
        return {
            "persistent_ids_loaded": len(self._persistent_ids),
            "next_id_counter": self._id_counter + 1,
        }


def create_id_manager(config: IdsConfig) -> IDManager:
    """Factory function to create an ID manager."""
    return IDManager(config)