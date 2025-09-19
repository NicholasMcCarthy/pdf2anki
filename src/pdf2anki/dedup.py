"""Deduplication utilities for flashcards."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

from .config import DeduplicationConfig, DeduplicationPolicy
from .strategies.base import FlashcardData

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages deduplication of flashcards using fuzzy matching and embeddings."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.persistent_index: Set[str] = set()
        
        # Load persistent index if configured
        if config.scope_persistent and config.index_path:
            self.load_persistent_index(Path(config.index_path))
    
    def deduplicate_cards(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Remove duplicate cards from the list."""
        if not self.config.enabled or not cards:
            return cards
        
        logger.info(f"Starting deduplication of {len(cards)} cards")
        
        # Within-run deduplication
        if self.config.scope_within_run:
            cards = self._deduplicate_within_run(cards)
        
        # Persistent deduplication
        if self.config.scope_persistent:
            cards = self._deduplicate_against_persistent(cards)
        
        logger.info(f"Deduplication complete: {len(cards)} cards remaining")
        return cards
    
    def _deduplicate_within_run(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Remove duplicates within the current run."""
        if not cards:
            return cards
        
        unique_cards = []
        seen_hashes = set()
        
        for i, card in enumerate(cards):
            is_duplicate = False
            card_text = self._extract_text_for_comparison(card)
            
            # Check against all previous cards
            for j, existing_card in enumerate(unique_cards):
                existing_text = self._extract_text_for_comparison(existing_card)
                
                if self._are_duplicates(card_text, existing_text):
                    logger.debug(f"Found within-run duplicate: card {i} matches card {j}")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_cards.append(card)
                seen_hashes.add(self._get_content_hash(card_text))
        
        removed_count = len(cards) - len(unique_cards)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} within-run duplicates")
        
        return unique_cards
    
    def _deduplicate_against_persistent(self, cards: List[FlashcardData]) -> List[FlashcardData]:
        """Remove duplicates against persistent index."""
        if not self.persistent_index:
            return cards
        
        unique_cards = []
        
        for card in cards:
            card_text = self._extract_text_for_comparison(card)
            content_hash = self._get_content_hash(card_text)
            
            if content_hash not in self.persistent_index:
                unique_cards.append(card)
                self.persistent_index.add(content_hash)
            else:
                logger.debug(f"Found persistent duplicate: {card_text[:50]}...")
        
        removed_count = len(cards) - len(unique_cards)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} persistent duplicates")
        
        return unique_cards
    
    def _are_duplicates(self, text1: str, text2: str) -> bool:
        """Check if two texts are duplicates based on configured thresholds."""
        # Quick exact match check
        if text1 == text2:
            return True
        
        # Fuzzy matching
        fuzzy_similar = False
        if self.config.fuzzy_threshold > 0:
            fuzzy_ratio = fuzz.ratio(text1, text2) / 100.0
            fuzzy_similar = fuzzy_ratio >= self.config.fuzzy_threshold
        
        # Embedding similarity (if available)
        embedding_similar = False
        if self.config.embedding_threshold > 0:
            try:
                similarity = self._compute_embedding_similarity(text1, text2)
                embedding_similar = similarity >= self.config.embedding_threshold
            except Exception as e:
                logger.debug(f"Embedding similarity failed: {e}")
                embedding_similar = False
        
        # Apply policy
        if self.config.policy == DeduplicationPolicy.OR:
            return fuzzy_similar or embedding_similar
        elif self.config.policy == DeduplicationPolicy.AND:
            return fuzzy_similar and embedding_similar
        else:
            return fuzzy_similar  # Default to fuzzy matching
    
    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between text embeddings."""
        embed1 = self._get_embedding(text1)
        embed2 = self._get_embedding(text2)
        
        if embed1 is None or embed2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(embed1, embed2)
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache if available."""
        # Check cache first
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            # This would require OpenAI API - implement as needed
            # For now, return None to disable embedding-based deduplication
            return None
            
            # Placeholder for actual embedding implementation:
            # from openai import OpenAI
            # client = OpenAI()
            # response = client.embeddings.create(
            #     model="text-embedding-3-large",
            #     input=text
            # )
            # embedding = np.array(response.data[0].embedding)
            # self.embeddings_cache[text] = embedding
            # return embedding
            
        except Exception as e:
            logger.debug(f"Failed to get embedding for text: {e}")
            return None
    
    def _extract_text_for_comparison(self, card: FlashcardData) -> str:
        """Extract text content from card for comparison."""
        if card.note_type == "Basic":
            return f"{card.front or ''} | {card.back or ''}"
        elif card.note_type == "Cloze":
            # Remove cloze markers for comparison
            import re
            cloze_text = card.cloze_text or ""
            clean_text = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', cloze_text)
            return f"{clean_text} | {card.extra or ''}"
        else:
            # Fallback: concatenate all text fields
            return f"{card.front or ''} | {card.back or ''} | {card.cloze_text or ''} | {card.extra or ''}"
    
    def _get_content_hash(self, text: str) -> str:
        """Get a hash of normalized text content."""
        import hashlib
        
        # Normalize text
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())  # Normalize whitespace
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def load_persistent_index(self, index_path: Path) -> None:
        """Load persistent deduplication index."""
        if not index_path.exists():
            logger.debug(f"Persistent index not found: {index_path}")
            return
        
        try:
            if index_path.suffix == '.json':
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.persistent_index = set(data.get('content_hashes', []))
            elif index_path.suffix == '.pkl':
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.persistent_index = data.get('content_hashes', set())
            
            logger.info(f"Loaded {len(self.persistent_index)} entries from persistent index")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent index: {e}")
    
    def save_persistent_index(self, index_path: Path) -> None:
        """Save persistent deduplication index."""
        if not self.config.scope_persistent:
            return
        
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            if index_path.suffix == '.json':
                data = {'content_hashes': list(self.persistent_index)}
                with open(index_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif index_path.suffix == '.pkl':
                data = {'content_hashes': self.persistent_index}
                with open(index_path, 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.persistent_index)} entries to persistent index")
            
        except Exception as e:
            logger.warning(f"Failed to save persistent index: {e}")
    
    def add_to_persistent_index(self, cards: List[FlashcardData]) -> None:
        """Add cards to persistent index."""
        if not self.config.scope_persistent:
            return
        
        for card in cards:
            text = self._extract_text_for_comparison(card)
            content_hash = self._get_content_hash(text)
            self.persistent_index.add(content_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "enabled": self.config.enabled,
            "fuzzy_threshold": self.config.fuzzy_threshold,
            "embedding_threshold": self.config.embedding_threshold,
            "policy": self.config.policy.value,
            "persistent_index_size": len(self.persistent_index),
            "embeddings_cached": len(self.embeddings_cache),
        }


def create_deduplication_manager(config: DeduplicationConfig) -> DeduplicationManager:
    """Factory function to create a deduplication manager."""
    return DeduplicationManager(config)