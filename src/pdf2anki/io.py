"""I/O utilities for file operations and data handling."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image
from rich.console import Console
from rich.table import Table

from .llm import clear_llm_cache

logger = logging.getLogger(__name__)


def save_csv(
    cards_data: List[Dict[str, Any]], 
    csv_path: Path,
    ensure_columns: bool = True
) -> None:
    """Save flashcard data to CSV file."""
    if not cards_data:
        logger.warning("No cards data to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(cards_data)
    
    # Ensure required columns exist
    if ensure_columns:
        required_columns = [
            "id", "deck", "note_type", "tags", "media",
            "front", "back", "cloze_text", "extra",
            "source_pdf", "page_start", "page_end", "section", "ref_citation",
            "llm_model", "llm_version", "strategy", "template_version",
            "created_at", "updated_at", "core_concept", "longtext", 
            "original_text", "my_notes"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""
    
    # Ensure proper data types
    df = df.fillna("")
    
    # Convert tags to semicolon-delimited string if it's a list
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(
            lambda x: ";".join(x) if isinstance(x, list) else str(x)
        )
    
    # Convert media to semicolon-delimited string if it's a list
    if "media" in df.columns:
        df["media"] = df["media"].apply(
            lambda x: ";".join(x) if isinstance(x, list) else str(x)
        )
    
    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved {len(df)} cards to {csv_path}")


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load flashcard data from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Convert semicolon-delimited strings back to lists for certain columns
    list_columns = ["tags", "media"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: [item.strip() for item in str(x).split(";") if item.strip()] if pd.notna(x) else []
            )
    
    logger.info(f"Loaded {len(df)} cards from {csv_path}")
    return df


def save_images(images: List[Dict], media_path: Path) -> List[str]:
    """Save extracted images to media directory."""
    media_path.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    for image_info in images:
        try:
            filename = image_info["filename"]
            pil_image = image_info["pil_image"]
            
            file_path = media_path / filename
            pil_image.save(file_path, "PNG", optimize=True)
            
            saved_files.append(filename)
            logger.debug(f"Saved image: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save image {image_info.get('filename', 'unknown')}: {e}")
            continue
    
    logger.info(f"Saved {len(saved_files)} images to {media_path}")
    return saved_files


def save_manifest(
    manifest_data: Dict[str, Any], 
    manifest_path: Path
) -> None:
    """Save processing manifest with metadata and telemetry."""
    # Add timestamp
    manifest_data["generated_at"] = datetime.now().isoformat()
    
    # Ensure parent directory exists
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved manifest to {manifest_path}")


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load processing manifest."""
    if not manifest_path.exists():
        return {}
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load manifest from {manifest_path}: {e}")
        return {}


def preview_cards(df: pd.DataFrame, console: Console, max_cards: int = 10) -> None:
    """Preview cards in a formatted table."""
    if df.empty:
        console.print("No cards to preview", style="yellow")
        return
    
    # Limit number of cards
    preview_df = df.head(max_cards)
    
    # Create table
    table = Table(title=f"Card Preview ({len(preview_df)} of {len(df)} cards)")
    
    # Add columns
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Front/Cloze", style="green", max_width=40)
    table.add_column("Back/Extra", style="blue", max_width=40)
    table.add_column("Tags", style="yellow")
    table.add_column("Pages", style="red")
    
    # Add rows
    for _, row in preview_df.iterrows():
        # Determine content based on note type
        if row.get("note_type") == "Cloze":
            front_content = str(row.get("cloze_text", ""))[:80] + "..." if len(str(row.get("cloze_text", ""))) > 80 else str(row.get("cloze_text", ""))
            back_content = str(row.get("extra", ""))[:80] + "..." if len(str(row.get("extra", ""))) > 80 else str(row.get("extra", ""))
        else:
            front_content = str(row.get("front", ""))[:80] + "..." if len(str(row.get("front", ""))) > 80 else str(row.get("front", ""))
            back_content = str(row.get("back", ""))[:80] + "..." if len(str(row.get("back", ""))) > 80 else str(row.get("back", ""))
        
        # Format tags
        tags = row.get("tags", [])
        if isinstance(tags, list):
            tags_str = ", ".join(tags[:3])  # Show first 3 tags
            if len(tags) > 3:
                tags_str += f" (+{len(tags)-3})"
        else:
            tags_str = str(tags)
        
        # Page range
        page_range = f"{row.get('page_start', '?')}-{row.get('page_end', '?')}"
        
        table.add_row(
            str(row.get("id", ""))[:12],
            str(row.get("note_type", "")),
            front_content,
            back_content,
            tags_str,
            page_range
        )
    
    console.print(table)
    
    # Show summary
    note_types = df["note_type"].value_counts()
    console.print(f"\nSummary: {len(df)} total cards")
    for note_type, count in note_types.items():
        console.print(f"  - {note_type}: {count} cards")


def find_pdf_files(paths: List[Path], patterns: List[str], recursive: bool = True) -> List[Path]:
    """Find PDF files matching the given patterns."""
    pdf_files = []
    
    for path in paths:
        path = Path(path)
        
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files.append(path)
        elif path.is_dir():
            for pattern in patterns:
                if recursive:
                    pdf_files.extend(path.rglob(pattern))
                else:
                    pdf_files.extend(path.glob(pattern))
    
    # Remove duplicates and sort
    pdf_files = sorted(list(set(pdf_files)))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def clear_cache() -> int:
    """Clear all caches and return number of entries cleared."""
    cleared = clear_llm_cache()
    
    # Could add other cache clearing here (embeddings, etc.)
    
    return cleared


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: Path) -> str:
    """Get SHA256 hash of a file."""
    import hashlib
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Optional[Path]:
    """Create a backup of a file."""
    if not file_path.exists():
        return None
    
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup of {file_path}: {e}")
        return None