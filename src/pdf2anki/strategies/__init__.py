"""Generation strategies for creating Anki flashcards."""

from .base import BaseStrategy
from .cloze_definitions import ClozeDefinitionsStrategy
from .figure_based import FigureBasedStrategy
from .key_points import KeyPointsStrategy

__all__ = [
    "BaseStrategy",
    "KeyPointsStrategy", 
    "ClozeDefinitionsStrategy",
    "FigureBasedStrategy",
]