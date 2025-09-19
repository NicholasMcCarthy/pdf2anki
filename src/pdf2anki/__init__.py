"""
pdf2anki: Convert PDF documents to Anki flashcards using LLMs.
"""

__version__ = "0.1.0"
__author__ = "Nicholas McCarthy"
__email__ = "nicholas@example.com"

from .config import Config
from .build import build_anki_deck

__all__ = ["Config", "build_anki_deck"]