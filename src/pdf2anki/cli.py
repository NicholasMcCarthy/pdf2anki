"""Command-line interface for pdf2anki."""

import shutil
from dataclasses import asdict, replace
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
from .config import Chunking, Config, DocumentsConfig, DocumentType
from .heuristics import DocumentAnalyzer, get_heuristic_defaults
from .ids import create_id_manager
from .io import clear_cache, find_markdown_files, load_csv, merge_cards_into_csv, preview_cards, save_csv
from .readwise import process_readwise_document
from .validate import validate_csv
from .workflow_router import WORKFLOW_TO_DOCUMENT_TYPE, select_workflow

# Import functions used by tests
from .pdf import PDFProcessor
from .chunking import TextChunker
from .llm import create_llm_provider
from .preprocess import process_single_pdf
from .templates import get_note_type_manager, NoteTypeManager, PromptManager

app = typer.Typer(
    name="pdf2anki",
    help="Convert PDF documents to Anki flashcards using LLMs",
    add_completion=False,
)
console = Console()


def _get_template_managers(current_dir: Path = Path.cwd()):
    """Get template managers, using init directories if they exist."""
    init_prompts_dir = current_dir / "prompts"
    init_notes_dir = current_dir / "notes"
    
    # Check if we're in an initialized directory with custom templates
    if init_prompts_dir.exists() and init_notes_dir.exists():
        console.print(f"📁 Using templates from init directory: {current_dir}", style="yellow")
        
        # Create custom managers using init directories
        from .prompts import create_prompt_manager
        prompt_manager = create_prompt_manager(init_prompts_dir)
        note_type_manager = NoteTypeManager(init_notes_dir)
        template_prompt_manager = PromptManager(init_prompts_dir)
        
        return prompt_manager, note_type_manager, template_prompt_manager
    else:
        # Use default managers
        from .prompts import create_prompt_manager
        prompt_manager = create_prompt_manager()
        note_type_manager = get_note_type_manager()
        template_prompt_manager = PromptManager()
        
        return prompt_manager, note_type_manager, template_prompt_manager


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
    pdfs_dir = target_dir / "pdfs"

    for directory in [workspace_dir, prompts_dir, examples_dir, notes_dir, samples_dir, scripts_dir, pdfs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"📁 Created directory: {directory}")
    
    # Copy example config to examples directory
    config_example_path = target_dir / "examples" / "config.example.yaml"
    if config_example_path.exists() and not force:
        console.print(f"⚠️  {config_example_path} already exists. Use --force to overwrite.")
    else:
        # Create example config
        default_config = Config()
        default_config.to_yaml(config_example_path)
        console.print(f"📝 Created example configuration: {config_example_path}")

    # Copy default prompts (.j2 templates and .yaml configs)
    package_prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if package_prompts_dir.exists():
        # Copy .j2 template files
        for prompt_file in package_prompts_dir.glob("*.j2"):
            target_file = prompts_dir / prompt_file.name
            if target_file.exists() and not force:
                console.print(f"⚠️  {target_file} already exists. Use --force to overwrite.")
            else:
                shutil.copy2(prompt_file, target_file)
                console.print(f"📄 Copied prompt template: {target_file}")
        
        # Copy .yaml configuration files
        for prompt_file in package_prompts_dir.glob("*.yaml"):
            target_file = prompts_dir / prompt_file.name
            if target_file.exists() and not force:
                console.print(f"⚠️  {target_file} already exists. Use --force to overwrite.")
            else:
                shutil.copy2(prompt_file, target_file)
                console.print(f"📄 Copied prompt config: {target_file}")
    
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
        results_table.add_column("Workflow", style="magenta")
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

                # Resolve the workflow (honoring any manual override already set
                # on this document) and apply heuristic chunking/strategy/
                # annotation-extraction defaults for it. Only do this when the
                # resolved workflow isn't GENERIC: get_effective_config() treats
                # heuristic_* as taking precedence over the user's global
                # config.yaml, so writing a heuristic suggestion for every
                # document - including a bland generic one - would silently
                # override explicit user config on any document that isn't
                # confidently classified (or manually routed) as a paper or textbook.
                doc_config = documents_config.documents[Path(pdf_path).name]
                workflow = select_workflow(Path(pdf_path), metadata, override=doc_config.workflow)
                doc_config.workflow = workflow.value

                effective_doc_type = WORKFLOW_TO_DOCUMENT_TYPE.get(workflow, DocumentType.UNKNOWN)
                if effective_doc_type != DocumentType.UNKNOWN:
                    effective_metadata = replace(metadata, doc_type=effective_doc_type)
                    defaults = get_heuristic_defaults(effective_metadata)
                    if "chunking_mode" in defaults:
                        doc_config.heuristic_chunking = Chunking(
                            mode=defaults["chunking_mode"],
                            tokens_per_chunk=defaults.get("tokens_per_chunk", 2000),
                        )
                    if "strategies" in defaults:
                        doc_config.heuristic_strategies = defaults["strategies"]
                    if "extract_annotations" in defaults:
                        doc_config.heuristic_extract_annotations = defaults["extract_annotations"]

                # Add to results table
                results_table.add_row(
                    Path(pdf_path).name,
                    str(metadata.page_count),
                    metadata.doc_type,
                    workflow.value,
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
                doc_type = doc_config.metadata.doc_type
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
        
        # Get template managers (using init directories if available)
        prompt_manager, note_type_manager, template_prompt_manager = _get_template_managers()
        
        # Handle plan-sample-csv first as it doesn't need documents.yaml
        if plan_sample_csv:
            _generate_sample_csv_schema(base_config, note_type_manager)
            return
        
        # Check if documents.yaml exists for other operations
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
            _show_generation_plan(process_documents, base_config, documents_config, prompt_manager)
        elif sample:
            _generate_samples(process_documents, base_config, documents_config, prompt_manager)
        else:
            _run_full_generation(process_documents, base_config, documents_config, verbose, prompt_manager, note_type_manager)
        
    except Exception as e:
        console.print(f"❌ Error during generation: {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def _show_generation_plan(documents: dict, base_config: Config, documents_config: DocumentsConfig, prompt_manager=None) -> None:
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
        metadata_table.add_row("Document Type", doc_config.metadata.doc_type)
        metadata_table.add_row("Has TOC", "✓" if doc_config.metadata.toc_present else "✗")
        metadata_table.add_row("Has DOI", "✓" if doc_config.metadata.has_doi else "✗")
        
        console.print(metadata_table)
        
        # Show effective configuration
        console.print("⚙️  Effective Configuration:", style="yellow")
        console.print(f"  Chunking: {effective_config.ingestion.chunking.mode}")
        console.print(f"  Tokens per chunk: {effective_config.ingestion.chunking.tokens_per_chunk}")
        console.print(f"  Enabled strategies: {list(effective_config.strategies.__dict__.keys())[:3]}...")  # TODO: Show actual enabled strategies
        
        # Show sample prompt (first chunk preview)
        console.print("🎯 First Chunk Prompt Preview:", style="yellow")
        
        try:
            # Load and chunk the first few pages to get first chunk
            from .pdf import PDFProcessor
            from .chunking import TextChunker
            
            if prompt_manager is None:
                from .prompts import create_prompt_manager
                prompt_manager = create_prompt_manager()
            
            pdf_processor = PDFProcessor()
            text_chunker = TextChunker(effective_config.ingestion.chunking)
            
            # Extract text from first few pages
            pdf_content = pdf_processor.extract_text(doc_config.file_path, max_pages=3)
            
            # Get first chunk
            chunks = text_chunker.chunk_document(pdf_content) #, start_page=1)
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


def _generate_samples(documents: dict, base_config: Config, documents_config: DocumentsConfig, prompt_manager=None) -> None:
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
            chunks = text_chunker.chunk_document(pdf_text)
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
                    "front": f"Sample question from {Path(doc_config.file_path)}",
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


def _generate_sample_csv_schema(base_config: Config, note_type_manager=None) -> None:
    """Generate sample CSV schema from note type templates."""
    console.print("📊 Generating sample CSV schema...", style="bold green")
    
    from .templates import get_note_type_manager
    import csv
    from io import StringIO
    
    if note_type_manager is None:
        note_type_manager = get_note_type_manager()
    
    all_note_types = note_type_manager.list_note_types()
    
    if not all_note_types:
        console.print("❌ No note types found. Please check notes/ directory.", style="red")
        return
    
    # Collect all unique fields across note types
    all_fields = set()
    note_type_fields = {}
    
    for note_type in all_note_types:
        fields = note_type_manager.get_csv_fields(note_type)
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


@app.command(name="generate-readwise")
def generate_readwise(
    path: Path = typer.Option(..., "--path", "-p", help="Markdown file or directory of Readwise/Obsidian exports"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    max_cards: int = typer.Option(2, "--max-cards", help="Max cards to generate per highlight"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Generate flashcards from Readwise/Obsidian markdown highlight exports.

    Cards are merged into the same CSV output as `generate` (by content-hash
    id, so re-running is safe) rather than overwriting it, so PDF- and
    Readwise-derived cards can accumulate into one deck.
    """
    console.print("📚 Generating flashcards from Readwise highlights...", style="bold blue")

    try:
        base_config = Config.from_yaml(config_path) if config_path and config_path.exists() else Config()
        base_config.create_workspace()

        md_files = find_markdown_files([path]) if path.is_dir() else [path]
        if not md_files:
            console.print(f"❌ No markdown files found at {path}", style="red")
            raise typer.Exit(code=1)

        console.print(f"📄 Found {len(md_files)} markdown files")

        prompt_manager, _, _ = _get_template_managers()
        llm_provider = create_llm_provider(base_config.llm)
        id_manager = create_id_manager(base_config.ids)

        all_cards = []
        for md_path in md_files:
            console.print(f"  📄 Processing {md_path.name}...")
            try:
                cards = process_readwise_document(
                    md_path, llm_provider, prompt_manager, max_cards_per_highlight=max_cards
                )
                all_cards.extend(cards)
                console.print(f"     ✅ {len(cards)} cards generated")
            except Exception as e:
                console.print(f"     ❌ Failed to process {md_path.name}: {e}", style="red")
                if verbose:
                    console.print_exception()
                continue

        if not all_cards:
            console.print("⚠️  No cards generated from any Readwise file.", style="yellow")
            return

        now = datetime.now().isoformat()
        new_rows = []
        for card in all_cards:
            card_dict = asdict(card)
            card_dict["id"] = id_manager.generate_id(card)
            card_dict["created_at"] = now
            card_dict["updated_at"] = now
            card_dict["deck"] = base_config.anki.deck_name
            card_dict["longtext"] = ""
            card_dict["my_notes"] = ""
            new_rows.append(card_dict)

        merge_result = merge_cards_into_csv(new_rows, base_config.output.csv_path)

        console.print(Panel.fit(
            f"✅ Readwise generation complete!\n\n"
            f"Files processed: {len(md_files)}\n"
            f"New cards: {merge_result['added']}\n"
            f"Total cards in CSV: {merge_result['total']}\n"
            f"CSV: {base_config.output.csv_path}",
            title="Success",
            style="green"
        ))

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"❌ Error during Readwise generation: {e}", style="bold red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def serve(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run the watcher service: watch configured directories for new PDFs/
    textbooks/Readwise markdown files, classify and process each one, keep
    the Anki deck up to date, and optionally push to AnkiConnect/Slack.

    This is the entrypoint the Docker service container runs.
    """
    import logging as _logging

    from .service import DirectoryWatcher, process_new_file
    from .service.notify import notify_error, notify_file_processed, notify_startup

    if verbose:
        _logging.getLogger("pdf2anki").setLevel(_logging.DEBUG)
    else:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = Config.from_yaml(config_path)
    webhook_url = config.service.notifications.slack_webhook_url

    console.print("🔭 Starting pdf2anki watcher service...", style="bold blue")
    watch_dirs = {
        "pdfs": config.service.watch_dirs.pdfs,
        "textbooks": config.service.watch_dirs.textbooks,
        "readwise": config.service.watch_dirs.readwise,
    }
    for name, path in watch_dirs.items():
        console.print(f"  📁 {name}: {path}")

    if config.service.notifications.notify_on_processed:
        notify_startup(webhook_url, watch_dirs)

    def on_file(path: Path) -> None:
        console.print(f"📄 New file detected: {path}")
        try:
            result = process_new_file(path, config)
            console.print(
                f"   ✅ {result['workflow']}: {result['cards_generated']} cards "
                f"(+{result['cards_added']} new)"
            )
            if config.service.notifications.notify_on_processed:
                notify_file_processed(webhook_url, result)
        except Exception as e:
            console.print(f"   ❌ Failed to process {path}: {e}", style="red")
            if verbose:
                console.print_exception()
            if config.service.notifications.notify_on_error:
                notify_error(webhook_url, str(path), str(e))

    watcher = DirectoryWatcher(
        directories=[watch_dirs["pdfs"], watch_dirs["textbooks"], watch_dirs["readwise"]],
        on_file=on_file,
        debounce_seconds=config.service.debounce_seconds,
        poll_interval_seconds=config.service.poll_interval_seconds,
    )
    watcher.run_forever()


def _run_full_generation(documents: dict, base_config: Config, documents_config: DocumentsConfig, verbose: bool, prompt_manager=None, note_type_manager=None) -> None:
    """Run full generation process.

    Mirrors preprocess.preprocess_pdf()'s orchestration, but drives it per-document
    using documents.yaml's layered effective config (base -> heuristic -> override)
    instead of a single global config, since each document may have its own chunking
    mode/strategy list.
    """
    console.print("⚡ Running full generation...", style="bold green")

    from .dedup import create_deduplication_manager
    from .ids import create_id_manager
    from .llm import create_llm_provider
    from .preprocess import finalize_generation
    from .rag import create_rag_manager
    from .telemetry import create_telemetry_collector

    base_config.create_workspace()

    telemetry = create_telemetry_collector(base_config.telemetry)
    telemetry.start_phase("initialization")

    llm_provider = create_llm_provider(base_config.llm)
    if prompt_manager is None:
        from .prompts import create_prompt_manager
        prompt_manager = create_prompt_manager()
    id_manager = create_id_manager(base_config.ids)
    dedup_manager = create_deduplication_manager(base_config.deduplication)
    rag_manager = create_rag_manager(base_config.rag)

    if base_config.ids.strategy == "persistent":
        id_manager.load_persistent_ids(base_config.output.csv_path)

    telemetry.end_phase()

    all_cards = []
    all_images = []
    processed_files: List[Path] = []

    for doc_key, doc_config in documents.items():
        pdf_path = Path(doc_config.file_path)
        console.print(f"  📄 Processing {doc_key}...")

        try:
            effective_config = documents_config.get_effective_config(doc_key, base_config)
            text_chunker = TextChunker(effective_config.ingestion.chunking, effective_config.llm.model)

            cards, images = process_single_pdf(
                pdf_path=pdf_path,
                config=effective_config,
                llm_provider=llm_provider,
                prompt_manager=prompt_manager,
                text_chunker=text_chunker,
                id_manager=id_manager,
                dedup_manager=dedup_manager,
                rag_manager=rag_manager,
                telemetry=telemetry,
            )

            all_cards.extend(cards)
            all_images.extend(images)
            processed_files.append(pdf_path)
            telemetry.record_pdf_processed()
            console.print(f"     ✅ {len(cards)} cards generated")

        except Exception as e:
            console.print(f"     ❌ Failed to process {doc_key}: {e}", style="red")
            if verbose:
                console.print_exception()
            telemetry.record_error("pdf_processing_error")
            continue

    if not processed_files:
        console.print("❌ No documents were successfully processed.", style="red")
        raise typer.Exit(code=1)

    result = finalize_generation(
        config=base_config,
        all_cards=all_cards,
        all_images=all_images,
        dedup_manager=dedup_manager,
        id_manager=id_manager,
        rag_manager=rag_manager,
        telemetry=telemetry,
        source_files=processed_files,
    )

    console.print(Panel.fit(
        f"✅ Generation complete!\n\n"
        f"Documents processed: {result['processed_pdfs']}\n"
        f"Cards generated: {result['total_cards']}\n"
        f"Images saved: {result['images_saved']}\n"
        f"CSV: {result['csv_path']}\n"
        f"Manifest: {result['manifest_path']}",
        title="Success",
        style="green"
    ))


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"pdf2anki version {__version__}")


if __name__ == "__main__":
    app()
