"""Command-line interface for pdf2anki."""

import shutil
from pathlib import Path
from typing import Optional, List
import glob
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .build import build_anki_deck
from .config import Config, DocumentsConfig
from .heuristics import DocumentAnalyzer
from .io import clear_cache, load_csv, preview_cards
from .validate import validate_csv

# Import functions used by tests
from .pdf import PDFProcessor
from .chunking import TextChunker
from .llm import create_llm_provider
from .templates import get_note_type_manager

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
    notes_dir = target_dir / "notes"
    samples_dir = target_dir / "samples"
    scripts_dir = target_dir / "scripts"
    
    for directory in [workspace_dir, prompts_dir, examples_dir, notes_dir, samples_dir, scripts_dir]:
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
    
    # Copy note type definitions
    package_notes_dir = Path(__file__).parent.parent.parent / "notes"
    if package_notes_dir.exists():
        for note_file in package_notes_dir.glob("*.yaml"):
            target_file = notes_dir / note_file.name
            if target_file.exists() and not force:
                console.print(f"⚠️  {target_file} already exists. Use --force to overwrite.")
            else:
                shutil.copy2(note_file, target_file)
                console.print(f"📝 Copied note type: {target_file}")
    
    # Copy sample generation script
    package_scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    if package_scripts_dir.exists():
        for script_file in package_scripts_dir.glob("*.py"):
            target_file = scripts_dir / script_file.name
            if target_file.exists() and not force:
                console.print(f"⚠️  {target_file} already exists. Use --force to overwrite.")
            else:
                shutil.copy2(script_file, target_file)
                console.print(f"🔧 Copied script: {target_file}")
    
    console.print(Panel.fit(
        "✅ Initialization complete!\n\n"
        "Next steps:\n"
        "1. Edit examples/config.example.yaml to configure your project\n"
        "2. Place PDF files in the configured input paths\n"
        "3. Run: pdf2anki scan-docs --config examples/config.example.yaml\n"
        "4. Run: pdf2anki generate --config examples/config.example.yaml\n\n"
        "Optional:\n"
        "- Generate sample PDFs: python scripts/generate_samples.py\n"
        "- Customize note types in notes/\n"
        "- Modify prompt templates in prompts/",
        title="Success",
        style="green"
    ))


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
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"❌ Error during validation: {e}", style="bold red")
        raise typer.Exit(code=1)


@app.command()
def build(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="Path to CSV file (overrides config)"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output .apkg path (overrides config)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Build Anki deck from CSV data."""
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
        raise typer.Exit(code=1)


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
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"pdf2anki version {__version__}")


@app.command(name="scan-docs")
def scan_docs(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    documents_file: Path = typer.Option(Path("documents.yaml"), "--documents", help="Path to documents.yaml file"),
    update: bool = typer.Option(True, "--update/--no-update", help="Update existing documents.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Discover PDFs and extract metadata, creating or updating documents.yaml."""
    console.print("🔍 Scanning documents and extracting metadata...", style="bold blue")
    
    try:
        # Load base configuration for input paths
        if config_path and config_path.exists():
            base_config = Config.from_yaml(config_path)
        else:
            base_config = Config()
            if not base_config.inputs.paths:
                console.print("⚠️  No input paths configured. Please set paths in config or add PDFs to current directory.", style="yellow")
                base_config.inputs.paths = ["."]
        
        # Load or create documents configuration
        documents_config = DocumentsConfig.from_yaml(documents_file)
        
        # Discover PDF files
        discovered_pdfs = []
        for path_pattern in base_config.inputs.paths:
            path = Path(path_pattern)
            
            if path.is_file() and path.suffix.lower() == '.pdf':
                discovered_pdfs.append(str(path))
            elif path.is_dir():
                for pattern in base_config.inputs.patterns:
                    if base_config.inputs.recursive:
                        discovered_pdfs.extend(glob.glob(str(path / "**" / pattern), recursive=True))
                    else:
                        discovered_pdfs.extend(glob.glob(str(path / pattern)))
        
        if not discovered_pdfs:
            console.print("❌ No PDF files found in configured paths.", style="red")
            raise typer.Exit(code=1)
        
        console.print(f"📄 Found {len(discovered_pdfs)} PDF files")
        
        # Initialize analyzer
        analyzer = DocumentAnalyzer()
        
        # Analyze documents
        results_table = Table(title="Document Analysis Results")
        results_table.add_column("File", style="cyan")
        results_table.add_column("Pages", justify="right")
        results_table.add_column("Type", style="green")
        results_table.add_column("TOC", justify="center")
        results_table.add_column("Chapters", justify="center")
        results_table.add_column("2-col", justify="center")
        results_table.add_column("DOI", justify="center")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing documents...", total=len(discovered_pdfs))
            
            for pdf_path in discovered_pdfs:
                progress.update(task, description=f"Analyzing {Path(pdf_path).name}")
                
                # Analyze document
                metadata = analyzer.analyze_document(pdf_path)
                
                # Add to documents config
                documents_config.add_or_update_document(pdf_path, metadata)
                
                # Add to results table
                results_table.add_row(
                    Path(pdf_path).name,
                    str(metadata.page_count),
                    metadata.doc_type.value,
                    "✓" if metadata.toc_present else "✗",
                    "✓" if metadata.chapters_detected else "✗",
                    "✓" if metadata.two_column_layout else "✗",
                    "✓" if metadata.has_doi else "✗",
                )
                
                progress.advance(task)
        
        # Save documents configuration
        documents_config.to_yaml(documents_file)
        
        # Display results
        console.print("\n")
        console.print(results_table)
        
        # Summary
        doc_types = {}
        for doc_config in documents_config.documents.values():
            if doc_config.metadata:
                doc_type = doc_config.metadata.doc_type.value
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        summary_lines = [
            f"✅ Analysis complete!",
            f"",
            f"Documents analyzed: {len(discovered_pdfs)}",
            f"Configuration saved: {documents_file}",
            f"",
            "Document types:",
        ]
        
        for doc_type, count in doc_types.items():
            summary_lines.append(f"  • {doc_type}: {count}")
        
        console.print(Panel.fit(
            "\n".join(summary_lines),
            title="Scan Results",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error during document scanning: {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def generate(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    documents_file: Path = typer.Option(Path("documents.yaml"), "--documents", help="Path to documents.yaml file"),
    plan: bool = typer.Option(False, "--plan", help="Show generation plan without LLM calls"),
    sample: bool = typer.Option(False, "--sample", help="Generate sample cards from first chunk only"),
    plan_sample_csv: bool = typer.Option(False, "--plan-sample-csv", help="Generate sample CSV schema"),
    pdf_override: Optional[Path] = typer.Option(None, "--pdf", help="Process specific PDF only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Generate flashcards using documents.yaml configuration."""
    console.print("🚀 Starting flashcard generation...", style="bold blue")
    
    try:
        # Load base configuration
        if config_path and config_path.exists():
            base_config = Config.from_yaml(config_path)
        else:
            base_config = Config()
        
        # Check if documents.yaml exists
        if not documents_file.exists():
            console.print(f"📄 {documents_file} not found. Running scan-docs first...", style="yellow")
            # TODO: Call scan_docs automatically
            console.print("❌ Please run 'pdf2anki scan-docs' first to create documents.yaml", style="red")
            raise typer.Exit(code=1)
        
        # Load documents configuration
        documents_config = DocumentsConfig.from_yaml(documents_file)
        
        if not documents_config.documents:
            console.print("❌ No documents found in documents.yaml", style="red")
            raise typer.Exit(code=1)
        
        # Filter documents if PDF override specified
        if pdf_override:
            key = pdf_override.name
            if key not in documents_config.documents:
                console.print(f"❌ PDF {key} not found in documents.yaml", style="red")
                raise typer.Exit(code=1)
            process_documents = {key: documents_config.documents[key]}
        else:
            process_documents = {k: v for k, v in documents_config.documents.items() if v.enabled}
        
        if plan:
            _show_generation_plan(process_documents, base_config, documents_config)
        elif sample:
            _generate_samples(process_documents, base_config, documents_config)
        elif plan_sample_csv:
            _generate_sample_csv_schema(base_config)
        else:
            _run_full_generation(process_documents, base_config, documents_config, verbose)
        
    except Exception as e:
        console.print(f"❌ Error during generation: {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def _show_generation_plan(documents: dict, base_config: Config, documents_config: DocumentsConfig) -> None:
    """Show generation plan for documents."""
    console.print("📋 Generation Plan", style="bold green")
    
    for doc_key, doc_config in documents.items():
        if not doc_config.metadata:
            continue
            
        # Get effective configuration
        effective_config = documents_config.get_effective_config(doc_key, base_config)
        
        console.print(f"\n📄 {doc_key}", style="bold cyan")
        
        # Show metadata and heuristics
        metadata_table = Table(show_header=False, box=None)
        metadata_table.add_column("Field", style="yellow")
        metadata_table.add_column("Value")
        
        metadata_table.add_row("File Path", doc_config.file_path)
        metadata_table.add_row("Pages", str(doc_config.metadata.page_count))
        metadata_table.add_row("Document Type", doc_config.metadata.doc_type.value)
        metadata_table.add_row("Has TOC", "✓" if doc_config.metadata.toc_present else "✗")
        metadata_table.add_row("Has DOI", "✓" if doc_config.metadata.has_doi else "✗")
        
        console.print(metadata_table)
        
        # Show effective configuration
        console.print("⚙️  Effective Configuration:", style="yellow")
        console.print(f"  Chunking: {effective_config.ingestion.chunking.mode.value}")
        console.print(f"  Tokens per chunk: {effective_config.ingestion.chunking.tokens_per_chunk}")
        console.print(f"  Enabled strategies: {list(effective_config.strategies.__dict__.keys())[:3]}...")  # TODO: Show actual enabled strategies
        
        # Show sample prompt (first chunk preview)
        console.print("🎯 First Chunk Prompt Preview:", style="yellow")
        
        try:
            # Load and chunk the first few pages to get first chunk
            from .pdf import PDFProcessor
            from .chunking import TextChunker
            from .prompts import create_prompt_manager
            
            pdf_processor = PDFProcessor()
            text_chunker = TextChunker(effective_config.ingestion.chunking)
            prompt_manager = create_prompt_manager()
            
            # Extract text from first few pages
            pdf_text = pdf_processor.extract_text(doc_config.file_path, max_pages=3)
            
            # Get first chunk
            chunks = text_chunker.chunk_text(pdf_text, start_page=1)
            if chunks:
                first_chunk = chunks[0]
                
                # Try to get first enabled strategy and render preview
                enabled_strategies = [name for name, config in effective_config.strategies.__dict__.items() 
                                    if hasattr(config, 'enabled') and config.enabled]
                
                if enabled_strategies:
                    strategy_name = enabled_strategies[0]
                    
                    # Get strategy template (simplified)
                    template_content = f"Strategy: {strategy_name}\nChunk text preview:\n"
                    
                    # Show truncated chunk content
                    chunk_preview = first_chunk.text[:200] + "..." if len(first_chunk.text) > 200 else first_chunk.text
                    template_content += f"\n{chunk_preview}\n\n[Prompt would continue with strategy-specific instructions...]"
                    
                    console.print(f"  📄 Pages {first_chunk.start_page}-{first_chunk.end_page} | {first_chunk.token_count} tokens")
                    console.print(f"  🔧 Strategy: {strategy_name}")
                    console.print("  📝 Template preview:")
                    console.print(f"     {chunk_preview}")
                else:
                    console.print("  ⚠️  No enabled strategies found")
            else:
                console.print("  ⚠️  No chunks generated (PDF may be empty or unreadable)")
                
        except Exception as e:
            console.print(f"  ⚠️  Could not generate prompt preview: {e}")
            console.print("  📝 [First chunk preview would be rendered here]")
            console.print("  🔧 [Strategy-specific prompt template would be shown]")


def _generate_samples(documents: dict, base_config: Config, documents_config: DocumentsConfig) -> None:
    """Generate sample cards from first chunk of each document."""
    console.print("🔬 Generating samples from first chunk of each document...", style="bold green")
    
    for doc_key, doc_config in documents.items():
        console.print(f"\n📄 {doc_key}", style="bold cyan")
        
        if not doc_config.metadata:
            console.print("  ⚠️  No metadata available - skip")
            continue
            
        try:
            # Get effective configuration
            effective_config = documents_config.get_effective_config(doc_key, base_config)
            
            # Load and chunk the document to get first chunk
            from .pdf import PDFProcessor
            from .chunking import TextChunker
            from .llm import create_llm_provider
            
            pdf_processor = PDFProcessor()
            text_chunker = TextChunker(effective_config.ingestion.chunking)
            
            # Extract text from first few pages
            pdf_text = pdf_processor.extract_text(doc_config.file_path, max_pages=3)
            
            # Get first chunk
            chunks = text_chunker.chunk_text(pdf_text, start_page=1)
            if not chunks:
                console.print("  ⚠️  No chunks generated (PDF may be empty or unreadable)")
                continue
                
            first_chunk = chunks[0]
            console.print(f"  📄 Processing chunk: pages {first_chunk.start_page}-{first_chunk.end_page}, {first_chunk.token_count} tokens")
            
            # Get enabled strategies
            enabled_strategies = [name for name, config in effective_config.strategies.__dict__.items() 
                                if hasattr(config, 'enabled') and config.enabled]
            
            if not enabled_strategies:
                console.print("  ⚠️  No enabled strategies found")
                continue
                
            # Mock LLM call for sample generation (no real LLM call yet)
            console.print(f"  🤖 Mock LLM generation with strategy: {enabled_strategies[0]}")
            
            # Mock generated cards
            sample_cards = [
                {
                    "id": f"sample_{doc_key}_001",
                    "note_type": "basic",
                    "front": f"Sample question from {Path(doc_config.file_path).stem}",
                    "back": f"Sample answer based on first chunk content",
                    "source_pdf": Path(doc_config.file_path).name,
                    "page_start": first_chunk.start_page,
                    "page_end": first_chunk.end_page,
                    "strategy": enabled_strategies[0]
                },
                {
                    "id": f"sample_{doc_key}_002", 
                    "note_type": "cloze",
                    "cloze_text": f"The main concept from this document is {{{{c1::sample concept}}}}",
                    "extra": "Additional context",
                    "source_pdf": Path(doc_config.file_path).name,
                    "page_start": first_chunk.start_page,
                    "page_end": first_chunk.end_page,
                    "strategy": enabled_strategies[0]
                }
            ]
            
            # Display generated cards
            console.print(f"  ✅ Generated {len(sample_cards)} sample cards:")
            
            for i, card in enumerate(sample_cards, 1):
                console.print(f"    {i}. [{card['note_type'].upper()}] {card.get('front', card.get('cloze_text', 'N/A'))}")
                if 'back' in card:
                    console.print(f"       → {card['back']}")
                if 'extra' in card:
                    console.print(f"       + {card['extra']}")
                    
            console.print("  📊 Sample cards generated (display-only, not saved)")
            
        except Exception as e:
            console.print(f"  ❌ Error generating samples: {e}")
            console.print("  🔧 [Mock sample generation would be shown here]")


def _generate_sample_csv_schema(base_config: Config) -> None:
    """Generate sample CSV schema from note type templates."""
    console.print("📊 Generating sample CSV schema...", style="bold green")
    
    from .templates import get_note_type_manager
    import csv
    from io import StringIO
    
    note_manager = get_note_type_manager()
    all_note_types = note_manager.list_note_types()
    
    if not all_note_types:
        console.print("❌ No note types found. Please check notes/ directory.", style="red")
        return
    
    # Collect all unique fields across note types
    all_fields = set()
    note_type_fields = {}
    
    for note_type in all_note_types:
        fields = note_manager.get_csv_fields(note_type)
        note_type_fields[note_type] = fields
        all_fields.update(fields)
    
    # Convert to sorted list for consistent output
    all_fields = sorted(all_fields)
    
    # Display summary
    console.print(f"Found {len(all_note_types)} note types:")
    for note_type in all_note_types:
        console.print(f"  • {note_type}: {len(note_type_fields[note_type])} fields")
    
    console.print(f"\nUnion of all fields: {len(all_fields)} columns")
    
    # Generate sample CSV content
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(all_fields)
    
    # Write one sample row per note type
    for note_type in all_note_types:
        row = []
        for field in all_fields:
            if field in note_type_fields[note_type]:
                # Generate sample data based on field name
                if field == "id":
                    row.append(f"card_{note_type}_001")
                elif field == "note_type":
                    row.append(note_type)
                elif field == "front":
                    row.append(f"Sample question for {note_type}")
                elif field == "back":
                    row.append(f"Sample answer for {note_type}")
                elif field == "cloze_text":
                    row.append(f"Sample {{{{c1::cloze deletion}}}} for {note_type}")
                elif field == "extra":
                    row.append(f"Extra context for {note_type}")
                elif field == "deck":
                    row.append("PDF2Anki Generated")
                elif field == "tags":
                    row.append(f"pdf2anki;{note_type}")
                elif field == "source_pdf":
                    row.append("sample.pdf")
                elif field in ["page_start", "page_end"]:
                    row.append("1")
                elif field in ["created_at", "updated_at"]:
                    row.append("2024-01-15T10:00:00Z")
                else:
                    row.append(f"sample_{field}")
            else:
                row.append("")  # Empty for fields not in this note type
        writer.writerow(row)
    
    # Save to file
    output_file = Path("sample_schema.csv")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    console.print(f"\n✅ Sample CSV schema saved to: {output_file}")
    console.print(f"Preview (first 5 columns):")
    
    # Show preview
    lines = output.getvalue().strip().split('\n')
    preview_table = Table()
    headers = lines[0].split(',')[:5]
    for header in headers:
        preview_table.add_column(header.strip('"'), style="cyan")
    
    for line in lines[1:]:
        cols = line.split(',')[:5]
        preview_table.add_row(*[col.strip('"') for col in cols])
    
    console.print(preview_table)


def _run_full_generation(documents: dict, base_config: Config, documents_config: DocumentsConfig, verbose: bool) -> None:
    """Run full generation process."""
    console.print("⚡ Running full generation...", style="bold green")
    
    # TODO: Implement full generation orchestration
    # This should:
    # 1. Use documents.yaml layering for each PDF
    # 2. Process by chunk and apply strategies in order
    # 3. Persist CSV/media/manifest to workspace/
    
    console.print("  [Full generation not yet implemented]")
    console.print("  [Would orchestrate generation using documents.yaml configuration]")


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"pdf2anki version {__version__}")


if __name__ == "__main__":
    app()
