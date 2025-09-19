"""Command-line interface for pdf2anki."""

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .build import build_anki_deck
from .config import Config
from .io import clear_cache, load_csv, preview_cards
from .preprocess import preprocess_pdf
from .validate import validate_csv

app = typer.Typer(
    name="pdf2anki",
    help="Convert PDF documents to Anki flashcards using LLMs",
    add_completion=False,
)
console = Console()

@app.command()
def init(
    target_dir: Path = typer.Argument(Path.cwd(), help="Target directory for initialization"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """Initialize a new pdf2anki project with example configuration and prompts."""
    console.print("🚀 Initializing pdf2anki project...", style="bold blue")
    
    # Create directories
    workspace_dir = target_dir / "workspace"
    prompts_dir = target_dir / "prompts"
    examples_dir = target_dir / "examples"
    
    for directory in [workspace_dir, prompts_dir, examples_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"📁 Created directory: {directory}")
    
    # Copy example config
    config_example_path = target_dir / "examples" / "config.example.yaml"
    if config_example_path.exists() and not force:
        console.print(f"⚠️  {config_example_path} already exists. Use --force to overwrite.")
    else:
        # Create example config
        default_config = Config()
        default_config.to_yaml(config_example_path)
        console.print(f"📝 Created example configuration: {config_example_path}")
    
    # Copy default prompts
    package_prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if package_prompts_dir.exists():
        for prompt_file in package_prompts_dir.glob("*.j2"):
            target_file = prompts_dir / prompt_file.name
            if target_file.exists() and not force:
                console.print(f"⚠️  {target_file} already exists. Use --force to overwrite.")
            else:
                shutil.copy2(prompt_file, target_file)
                console.print(f"📄 Copied prompt template: {target_file}")
    
    console.print(Panel.fit(
        "✅ Initialization complete!\n\n"
        "Next steps:\n"
        "1. Edit examples/config.example.yaml to configure your project\n"
        "2. Place PDF files in the configured input paths\n"
        "3. Run: pdf2anki preprocess --config examples/config.example.yaml",
        title="Success",
        style="green"
    ))


@app.command()
def preprocess(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    pdf_path: Optional[Path] = typer.Option(None, "--pdf", help="Specific PDF file to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Preprocess PDF files into CSV format with flashcard data."""
    console.print("🔄 Starting PDF preprocessing...", style="bold blue")
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Override input paths if specific PDF provided
        if pdf_path:
            config.inputs.paths = [pdf_path]
        
        # Run preprocessing
        result = preprocess_pdf(config, verbose=verbose)
        
        console.print(Panel.fit(
            f"✅ Preprocessing complete!\n\n"
            f"Generated {result['total_cards']} cards from {result['processed_pdfs']} PDFs\n"
            f"Output: {result['csv_path']}\n"
            f"Media: {result['media_path']}\n"
            f"Manifest: {result['manifest_path']}",
            title="Success",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error during preprocessing: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def validate(
    csv_path: Path = typer.Option(..., "--csv", help="Path to CSV file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Validate CSV file schema and content."""
    console.print("🔍 Validating CSV file...", style="bold blue")
    
    try:
        result = validate_csv(csv_path, verbose=verbose)
        
        if result["valid"]:
            console.print(Panel.fit(
                f"✅ Validation successful!\n\n"
                f"Total rows: {result['total_rows']}\n"
                f"Note types: {', '.join(result['note_types'])}\n"
                f"Media files: {result['media_files']} found",
                title="Valid",
                style="green"
            ))
        else:
            console.print("❌ Validation failed:", style="bold red")
            for error in result["errors"]:
                console.print(f"  • {error}", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Error during validation: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def build(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="Path to CSV file (overrides config)"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output .apkg path (overrides config)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Build Anki deck from preprocessed CSV data."""
    console.print("🔨 Building Anki deck...", style="bold blue")
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Override paths if provided
        if csv_path:
            config.output.csv_path = csv_path
        if output_path:
            config.output.apkg_path = output_path
        
        # Build deck
        result = build_anki_deck(config, verbose=verbose)
        
        console.print(Panel.fit(
            f"✅ Deck built successfully!\n\n"
            f"Output: {result['apkg_path']}\n"
            f"Cards: {result['total_cards']}\n"
            f"Note types: {', '.join(result['note_types'])}\n"
            f"Deck: {result['deck_name']}",
            title="Success",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error during build: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def preview(
    csv_path: Path = typer.Option(..., "--csv", help="Path to CSV file to preview"),
    n: int = typer.Option(10, "--n", help="Number of cards to preview"),
    note_type: Optional[str] = typer.Option(None, "--type", help="Filter by note type"),
) -> None:
    """Preview cards from CSV file."""
    console.print(f"👀 Previewing {n} cards from CSV...", style="bold blue")
    
    try:
        df = load_csv(csv_path)
        
        if note_type:
            df = df[df["note_type"] == note_type]
        
        sample = df.head(n)
        preview_cards(sample, console)
        
    except Exception as e:
        console.print(f"❌ Error during preview: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"pdf2anki version {__version__}")


if __name__ == "__main__":
    app()
