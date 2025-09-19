"""Validation utilities for CSV data and content."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Any

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class CSVValidator:
    """Validates CSV files containing flashcard data."""
    
    # Required columns for CSV files
    REQUIRED_COLUMNS = {
        "id", "deck", "note_type", "tags", "media"
    }
    
    # Columns that should exist for specific note types
    NOTE_TYPE_COLUMNS = {
        "Basic": {"front", "back"},
        "Cloze": {"cloze_text"},
    }
    
    # Optional but recommended columns
    METADATA_COLUMNS = {
        "source_pdf", "page_start", "page_end", "section", "ref_citation",
        "llm_model", "llm_version", "strategy", "template_version",
        "created_at", "updated_at", "core_concept", "original_text"
    }
    
    def __init__(self, media_path: Path = None):
        self.media_path = media_path
        self.errors = []
        self.warnings = []
    
    def validate_csv(self, csv_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """Validate a CSV file and return validation results."""
        self.errors = []
        self.warnings = []
        
        if not csv_path.exists():
            raise ValidationError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            raise ValidationError(f"Failed to read CSV file: {e}")
        
        # Validate structure
        self._validate_columns(df)
        self._validate_data_types(df)
        self._validate_note_types(df)
        self._validate_ids(df)
        self._validate_media_references(df)
        self._validate_content(df)
        
        # Collect statistics
        stats = self._collect_statistics(df)
        
        result = {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_rows": len(df),
            "note_types": df["note_type"].unique().tolist() if "note_type" in df.columns else [],
            "media_files": len(self._get_all_media_files(df)),
            **stats
        }
        
        if verbose:
            self._log_validation_results(result)
        
        return result
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present."""
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            self.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for empty required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns and df[col].isna().all():
                self.warnings.append(f"Column '{col}' is entirely empty")
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types and formats."""
        # Check ID column
        if "id" in df.columns:
            # IDs should be non-empty strings
            empty_ids = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
            if empty_ids.any():
                self.errors.append(f"Found {empty_ids.sum()} rows with empty IDs")
        
        # Check page numbers
        for col in ["page_start", "page_end"]:
            if col in df.columns:
                # Convert to numeric, errors='coerce' will turn non-numeric to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = numeric_col.isna().sum() - df[col].isna().sum()
                if non_numeric_count > 0:
                    self.warnings.append(f"Found {non_numeric_count} non-numeric values in '{col}'")
    
    def _validate_note_types(self, df: pd.DataFrame) -> None:
        """Validate note types and their required fields."""
        if "note_type" not in df.columns:
            return
        
        for note_type in df["note_type"].unique():
            if pd.isna(note_type):
                self.errors.append("Found rows with missing note_type")
                continue
            
            note_type = str(note_type)
            if note_type not in self.NOTE_TYPE_COLUMNS:
                self.warnings.append(f"Unknown note type: '{note_type}'")
                continue
            
            # Check required fields for this note type
            required_fields = self.NOTE_TYPE_COLUMNS[note_type]
            note_type_rows = df[df["note_type"] == note_type]
            
            for field in required_fields:
                if field not in df.columns:
                    self.errors.append(f"Missing required field '{field}' for note type '{note_type}'")
                else:
                    empty_count = (note_type_rows[field].isna() | 
                                   (note_type_rows[field].astype(str).str.strip() == "")).sum()
                    if empty_count > 0:
                        self.errors.append(f"Found {empty_count} '{note_type}' cards with empty '{field}' field")
    
    def _validate_ids(self, df: pd.DataFrame) -> None:
        """Validate ID uniqueness and format."""
        if "id" not in df.columns:
            return
        
        # Check for duplicate IDs
        duplicates = df["id"].duplicated()
        if duplicates.any():
            duplicate_ids = df[duplicates]["id"].tolist()
            self.errors.append(f"Found {len(duplicate_ids)} duplicate IDs: {duplicate_ids[:5]}")
        
        # Check ID format (should be reasonable length and format)
        for idx, id_val in df["id"].items():
            if pd.isna(id_val):
                continue
            
            id_str = str(id_val).strip()
            if len(id_str) < 4:
                self.warnings.append(f"Very short ID at row {idx}: '{id_str}'")
            elif len(id_str) > 100:
                self.warnings.append(f"Very long ID at row {idx}: '{id_str[:20]}...'")
    
    def _validate_media_references(self, df: pd.DataFrame) -> None:
        """Validate media file references."""
        if "media" not in df.columns or not self.media_path:
            return
        
        all_media_files = self._get_all_media_files(df)
        missing_files = []
        
        for media_file in all_media_files:
            media_file_path = self.media_path / media_file
            if not media_file_path.exists():
                missing_files.append(media_file)
        
        if missing_files:
            self.warnings.append(f"Found {len(missing_files)} missing media files: {missing_files[:5]}")
    
    def _validate_content(self, df: pd.DataFrame) -> None:
        """Validate card content for common issues."""
        # Check for overly long content
        content_columns = ["front", "back", "cloze_text", "extra"]
        
        for col in content_columns:
            if col not in df.columns:
                continue
            
            # Check for extremely long content
            long_content = df[col].astype(str).str.len() > 10000
            if long_content.any():
                self.warnings.append(f"Found {long_content.sum()} cards with very long '{col}' content (>10k chars)")
            
            # Check for suspiciously short content
            if col in ["front", "back", "cloze_text"]:
                short_content = (df[col].astype(str).str.len() < 5) & (df[col].notna())
                if short_content.any():
                    self.warnings.append(f"Found {short_content.sum()} cards with very short '{col}' content (<5 chars)")
        
        # Validate cloze format
        if "cloze_text" in df.columns:
            self._validate_cloze_format(df)
    
    def _validate_cloze_format(self, df: pd.DataFrame) -> None:
        """Validate cloze deletion format."""
        cloze_pattern = re.compile(r'\{\{c\d+::[^}]+\}\}')
        
        cloze_rows = df[df["note_type"] == "Cloze"]["cloze_text"].dropna()
        
        invalid_cloze_count = 0
        for idx, cloze_text in cloze_rows.items():
            if not isinstance(cloze_text, str):
                continue
            
            matches = cloze_pattern.findall(cloze_text)
            if not matches:
                invalid_cloze_count += 1
            elif len(matches) > 5:  # Too many cloze deletions
                self.warnings.append(f"Row {idx} has {len(matches)} cloze deletions (consider splitting)")
        
        if invalid_cloze_count > 0:
            self.errors.append(f"Found {invalid_cloze_count} cloze cards with invalid cloze format")
    
    def _get_all_media_files(self, df: pd.DataFrame) -> Set[str]:
        """Extract all unique media file references from the DataFrame."""
        media_files = set()
        
        if "media" not in df.columns:
            return media_files
        
        for media_value in df["media"].dropna():
            if isinstance(media_value, str) and media_value.strip():
                # Split on semicolon and add each file
                files = [f.strip() for f in media_value.split(";") if f.strip()]
                media_files.update(files)
        
        return media_files
    
    def _collect_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Collect statistics about the CSV data."""
        stats = {}
        
        if "note_type" in df.columns:
            stats["note_type_counts"] = df["note_type"].value_counts().to_dict()
        
        if "strategy" in df.columns:
            stats["strategy_counts"] = df["strategy"].value_counts().to_dict()
        
        if "difficulty" in df.columns:
            stats["difficulty_counts"] = df["difficulty"].value_counts().to_dict()
        
        # Calculate completeness scores
        completeness = {}
        for col in self.METADATA_COLUMNS:
            if col in df.columns:
                non_empty = (~df[col].isna()) & (df[col].astype(str).str.strip() != "")
                completeness[col] = non_empty.sum() / len(df) * 100
        
        stats["metadata_completeness"] = completeness
        
        return stats
    
    def _log_validation_results(self, result: Dict[str, Any]) -> None:
        """Log validation results."""
        if result["valid"]:
            logger.info("✅ CSV validation passed")
        else:
            logger.error("❌ CSV validation failed")
        
        for error in result["errors"]:
            logger.error(f"  ERROR: {error}")
        
        for warning in result["warnings"]:
            logger.warning(f"  WARNING: {warning}")
        
        logger.info(f"Total cards: {result['total_rows']}")
        logger.info(f"Note types: {', '.join(result['note_types'])}")


def validate_csv(csv_path: Path, media_path: Path = None, verbose: bool = False) -> Dict[str, Any]:
    """Validate a CSV file containing flashcard data."""
    validator = CSVValidator(media_path)
    return validator.validate_csv(csv_path, verbose)


def validate_anki_deck_name(deck_name: str) -> bool:
    """Validate Anki deck name according to Anki's rules."""
    if not deck_name or not deck_name.strip():
        return False
    
    # Anki deck names can't contain certain characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
    if any(char in deck_name for char in invalid_chars):
        return False
    
    # Can't start or end with whitespace
    if deck_name != deck_name.strip():
        return False
    
    return True


def validate_note_type_config(note_type_config: Dict[str, Any]) -> List[str]:
    """Validate note type configuration."""
    errors = []
    
    required_keys = ["fields", "templates"]
    for key in required_keys:
        if key not in note_type_config:
            errors.append(f"Missing required key: {key}")
    
    # Validate fields
    fields = note_type_config.get("fields", [])
    if not isinstance(fields, list) or not fields:
        errors.append("Fields must be a non-empty list")
    
    # Validate templates
    templates = note_type_config.get("templates", [])
    if not isinstance(templates, list) or not templates:
        errors.append("Templates must be a non-empty list")
    
    return errors