# pdf2anki

pdf2anki is a Python library and CLI that converts technical PDFs (research papers, textbooks) into high‑quality Anki flashcards. It uses a two‑stage workflow:

1) Preprocess: Ingest PDFs, chunk text intelligently, generate candidate cards via LLM strategies, and write an editable CSV plus media folder.
2) Build: Validate the CSV and produce an .apkg deck using configurable note types and deck structure.

Lean defaults make it easy to get started, while configuration, prompt templates, and plugin‑style strategies let you extend and control behavior.

- Default LLM: OpenAI gpt‑4.1 (configurable, including optional gpt‑5 and other LangChain‑supported models).
- Optional OCR, tables extraction, RAG‑lite indexing, duplicate detection, reviewer quality gates, cost telemetry, hallucination mitigation, curriculum tags, and deterministic seeds.
- Deck automation via GitHub Actions: rebuild deck when CSV or media change (no PDFs required in the repo), and publish releases to PyPI and GHCR.

Links:
- PR: Scaffold pdf2anki: CLI, config, prompts, preprocess/build pipeline, CI/CD
- License: MIT

--------------------------------------------------------------------------------

Features

- Two-stage pipeline
  - Preprocess: PDF ingestion, heading‑aware chunking, strategy‑based card generation (Basic & Cloze), optional reviewer pass, CSV + media output with manifest/telemetry.
  - Build: Generate .apkg using genanki, configurable note types, deck structure (flat, by chapter, by theme, predefined), and tag policies.
- Math, tables, images
  - Math: preserve LaTeX ($...$, $$...$$); Anki’s MathJax renders by default.
  - Tables: heuristic text→HTML table rendering; optional Camelot/Tabula integration.
  - Images: extract figure/table images; include when relevant.
- Quality, reliability, and cost controls
  - Hallucination mitigation: require page citations; verify quoted substrings against source; drop non‑verifiable cards if configured.
  - Reviewer chain: optional second LLM to score/edit cards; configurable thresholds.
  - Duplicate detection: fuzzy + embeddings to avoid redundant cards across PDFs/runs.
  - RAG‑lite indexing: FAISS/Chroma focus to reduce cost and improve relevance.
  - Determinism: temperature=0.0, caching, and deterministic seeds where supported.
  - Telemetry: token usage, costs, cache hits, elapsed time, and counts in manifest.
- IDs and updates
  - content_hash: deterministic content‑based IDs; edits create new cards.
  - persistent: keep stable IDs in CSV; edits update existing cards.
- Configurability and extensibility
  - YAML config with Pydantic validation; environment variable support.
  - Prompt templates (Jinja2) and pluggable strategies.
  - Curriculum tags and tag generation with sensible caps.

--------------------------------------------------------------------------------

Installation

- From PyPI (once released):
  - pip install pdf2anki
- From source:
  - pip install -U pip
  - pip install .        # core
  - pip install .[ocr]   # adds OCR (tesseract/ocrmypdf) support
  - pip install .[tables]# adds table extraction (camelot) support
  - pip install .[dev]   # dev tools (pytest, ruff, black, mypy)

Environment
- Set your OpenAI key (example): export OPENAI_API_KEY=sk-...

--------------------------------------------------------------------------------

Quick start

1) Scaffold a project
- Create a workspace and copy example config and prompts to your local project:
  - pdf2anki init
  - This will create:
    - examples/config.example.yaml
    - prompts/ (default templates)
    - workspace/ (for CSV, media, manifest, outputs)

2) Preprocess PDFs into CSV and media
- Edit the config (see Configuration below).
- Run:
  - pdf2anki preprocess --config examples/config.example.yaml
- What happens:
  - PDFs are discovered (glob or explicit).
  - Text is extracted and chunked near headings with token limits.
  - Strategies (e.g., key_points, cloze_definitions) generate candidate cards using LLM templates; JSON schema enforced.
  - Optional reviewer LLM scores/edits cards; low‑scoring cards are dropped.
  - Hallucination guard: require page citations; verify quotes.
  - Duplicate detection skips redundant cards.
  - cards.csv, media/, and manifest.json written to workspace/.

3) Review or edit the CSV
- Open workspace/cards.csv in your editor.
- You can add/remove/edit rows, fix fields, or attach media filenames.

4) Validate and build the Anki deck
- Validate:
  - pdf2anki validate --csv workspace/cards.csv
- Build:
  - pdf2anki build --config examples/config.example.yaml
- Output:
  - workspace/MyDeck.apkg

5) Preview sample rows (optional)
- pdf2anki preview --csv workspace/cards.csv --n 10

6) Clear cache (optional)
- pdf2anki cache clear

--------------------------------------------------------------------------------

CLI reference

- pdf2anki init
  - Scaffolds prompts/ and examples/config.example.yaml; creates workspace/.
- pdf2anki preprocess --config config.yaml [--pdf path.pdf]
  - Generates workspace/cards.csv, media/, manifest.json.
- pdf2anki validate --csv workspace/cards.csv
  - Checks required columns, media references, note_type correctness, and unique IDs.
- pdf2anki build --config config.yaml
  - Builds .apkg using CSV + media. Honors deck structure, note types, tags, and ID strategy.
- pdf2anki preview --csv cards.csv --n 10
  - Prints sample rows to terminal.
- pdf2anki cache clear
  - Clears LLM response cache.

--------------------------------------------------------------------------------

Configuration

Place your YAML config at examples/config.example.yaml or a custom path. The Pydantic schema validates on load. Env vars are supported for secrets (e.g., api_key_env).

Example
```yaml
project:
  name: "MyDeck"
  workspace_dir: "./workspace"
inputs:
  pdf_glob: "./pdfs/*.pdf"
  include_images: true
  ocr:
    enabled: false
    language: "eng"
ingestion:
  mode: "auto"            # auto | pages | chapters | entire
  chunking:
    tokens_per_chunk: 800
    overlap_tokens: 120
    preserve_headings: true
    max_pages_per_chunk: 4
llm:
  provider: "openai"
  model: "gpt-4.1"
  temperature: 0.0
  max_output_tokens: 900
  cache: true
  seed: 12345
  api_key_env: "OPENAI_API_KEY"
  model_registry:
    allow: ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5"]
strategies:
  - name: "key_points"
    note_type: "Basic"
    params:
      max_cards_per_chunk: 4
      include_citations: true
      include_images_if_relevant: true
      fields:
        core_concept: true
        longtext: true
        original_text: true
        my_notes: true
  - name: "cloze_definitions"
    note_type: "Cloze"
    params:
      max_clozes_per_chunk: 3
      max_total_cloze_blanks: 5
      include_citations: true
rag:
  enabled: true
  k: 8
  embedding_model: "text-embedding-3-large"
deduplication:
  enabled: true
  methods:
    embeddings: { enabled: true, threshold: 0.86 }
    fuzzy: { enabled: true, threshold: 90 }
  policy: "OR"            # OR | AND
  scope: "persistent"     # run | persistent
hallucination:
  require_citations: true
  verify_quotes: true
  drop_on_mismatch: true
review:
  enabled: true
  model: "gpt-4.1"
  min_score: 4
  allow_edits: true
tags:
  required: ["pdf2anki"]
  llm_select_from_required: true
  llm_generated: { enabled: true, max: 3 }
taxonomy:
  file: "./taxonomy/curriculum.yaml"
  enforce: false
ids:
  strategy: "content_hash"     # content_hash | persistent
  salt: "change-me"
language:
  default: "en"
  multilingual:
    enabled: false
anki:
  deck_name: "MyDeck"
  deck_id: 2059400110
  deck_structure:
    mode: "by_chapter"    # flat | by_chapter | by_theme | predefined
    theme_map: {}
  default_note_types:
    Basic:
      fields: ["Front", "Back", "core_concept", "longtext", "original_text", "my_notes", "source_pdf", "page_start", "page_end", "ref_citation"]
      templates:
        - name: "Card 1"
          qfmt: "{{Front}}"
          afmt: "{{Front}}<hr id=answer>{{Back}}"
    Cloze:
      fields: ["Text", "Extra", "core_concept", "longtext", "original_text", "my_notes", "source_pdf", "page_start", "page_end", "ref_citation"]
      templates:
        - name: "Cloze"
          qfmt: "{{cloze:Text}}"
          afmt: "{{cloze:Text}}<br>{{Extra}}"
output:
  csv_path: "./workspace/cards.csv"
  media_dir: "./workspace/media"
  apkg_path: "./workspace/MyDeck.apkg"
telemetry:
  enabled: true
  write_manifest: true
  manifest_path: "./workspace/manifest.json"
```

Notes
- llm.model default is gpt‑4.1; you can allow other models via model_registry.
- OCR is off by default. Enable it for scanned PDFs.
- Deck structure: choose flat, by_chapter (Deck::Chapter), by_theme (taxonomy/required tags), or predefined mappings.

--------------------------------------------------------------------------------

CSV schema

Every row is a card (note). Columns:
- Core
  - id: unique note ID (strategy dependent)
  - deck: deck or subdeck (e.g., MyDeck::Chapter 3)
  - note_type: Basic or Cloze
  - tags: semicolon‑delimited (e.g., pdf2anki;topicX)
  - media: semicolon‑delimited filenames; referenced from fields
- Fields by type
  - Basic: front, back
  - Cloze: cloze_text, extra
- Common metadata (on all cards)
  - source_pdf, page_start, page_end, section, ref_citation
  - llm_model, llm_version, strategy, template_version
  - created_at, updated_at
  - core_concept, longtext, original_text, my_notes

Validation
- pdf2anki validate checks presence and correctness by note_type, that media files exist, HTML is safe, and ids are unique.

--------------------------------------------------------------------------------

Strategies and prompts

Strategies determine how cards are generated. Initial set:
- key_points (Basic): concise Q/A for core concepts and findings.
- cloze_definitions (Cloze): definitions, theorems, and key sentences with cloze syntax.
- figure_based (Basic): Q/A referencing useful figures/tables (with images when relevant).

Prompt templates live under prompts/ and are Jinja2‑rendered. You can copy and customize them per project. Templates enforce strict JSON outputs; schema failures trigger retries.

--------------------------------------------------------------------------------

Math, tables, and images

- Math: Keep LaTeX inline $...$ or block $$...$$. Anki’s MathJax renders them in templates—no extra setup.
- Tables: A default heuristic converts simple structured text into minimal HTML tables. For complex PDFs, enable the [tables] extra to use Camelot/Tabula.
- Images: The preprocessor saves figure crops into workspace/media with deterministic filenames. Fields can include them via <img src="...">. The LLM and/or heuristics decide inclusion based on relevance.

--------------------------------------------------------------------------------

Duplicate detection and RAG‑lite

- Duplicates: Enable deduplication to skip near‑identical cards across PDFs or runs using:
  - Embeddings (cosine similarity) with a configurable threshold (e.g., 0.86).
  - Fuzzy text similarity (e.g., RapidFuzz ratio ≥ 90).
  - Policy can be OR (default) or AND; scope can be within a run or persistent (simple index file).
- RAG‑lite: For very long PDFs, build a lightweight FAISS/Chroma index to focus generation on the top‑k relevant chunks (e.g., k=8). Reduces cost and increases relevance.

--------------------------------------------------------------------------------

Hallucination mitigation and reviewer

- Hallucination mitigations:
  - Require page citations (page_start/page_end).
  - Verify any quoted strings in fields exist in the source chunk; drop candidates that fail (configurable).
- Reviewer:
  - Optional second LLM pass with a scoring rubric. Cards below min_score are dropped; if allow_edits is true, reviewer can propose minimal field edits that are applied automatically.

--------------------------------------------------------------------------------

IDs: content_hash vs persistent

- content_hash:
  - id = BLAKE2(salt + deck + note_type + normalized(fields) + source_pdf + page span)
  - Fully deterministic. Editing content yields new IDs (new cards).
- persistent:
  - The CSV’s id field is authoritative. Editing fields does not change id (updates existing card).
- Choose via ids.strategy in config.

--------------------------------------------------------------------------------

Deck building

- genanki is used to construct .apkg.
- Note types:
  - Basic: at least Front/Back fields.
  - Cloze: at least Text/Extra fields.
  - Extended fields available: core_concept, longtext, original_text, my_notes, and metadata fields are included to facilitate advanced templates.
- Deck structure:
  - flat: all notes under a single deck.
  - by_chapter: subdecks per chapter/section (e.g., MyDeck::Chapter 3).
  - by_theme: organize via tags/taxonomy.
  - predefined: fully explicit mapping in config.
- Tags: combine required tags and optional LLM‑generated tags (bounded by max).

--------------------------------------------------------------------------------

Docker

Build
- Default (no OCR):
  - docker build -t pdf2anki:local .
- With OCR dependencies:
  - docker build --build-arg INSTALL_OCR=true -t pdf2anki:ocr .

Run
- Using environment variable and mounting a project directory:
  - docker run --rm -it \
      -e OPENAI_API_KEY=$OPENAI_API_KEY \
      -v "$(pwd)":/work \
      -w /work \
      pdf2anki:local \
      pdf2anki preprocess --config examples/config.example.yaml

- Build the deck from an existing CSV (no PDFs required):
  - docker run --rm -it \
      -v "$(pwd)":/work -w /work \
      pdf2anki:local \
      pdf2anki validate --csv workspace/cards.csv && \
      pdf2anki build --config examples/config.example.yaml

--------------------------------------------------------------------------------

Build and push container to GHCR

Prerequisites
- A GitHub account with permission to push to ghcr.io/<OWNER>.
- A Personal Access Token (classic) or fine‑grained token with write:packages scope, or use the repo GITHUB_TOKEN in GitHub Actions.
- Docker installed locally.

Login to GHCR (local)
- export CR_PAT=ghp_...   # or use GH_TOKEN
- echo $CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

Build and tag
- OWNER=YourGitHubUserOrOrg
- VERSION=0.1.0
- docker build -t ghcr.io/$OWNER/pdf2anki:$VERSION .
- Optionally include OCR:
  - docker build --build-arg INSTALL_OCR=true -t ghcr.io/$OWNER/pdf2anki:$VERSION .

Push
- docker push ghcr.io/$OWNER/pdf2anki:$VERSION

Latest tag (optional)
- docker tag ghcr.io/$OWNER/pdf2anki:$VERSION ghcr.io/$OWNER/pdf2anki:latest
- docker push ghcr.io/$OWNER/pdf2anki:latest

Via GitHub Actions (recommended)
- This repo includes a release workflow that builds and pushes to GHCR on tags v*.*.* automatically (using GITHUB_TOKEN). See .github/workflows/release.yml. Create a tag:
  - git tag v0.1.0
  - git push origin v0.1.0

--------------------------------------------------------------------------------

GitHub Actions

1) Build deck on CSV/media changes
- Workflow: .github/workflows/build-deck-on-csv-change.yml
- Triggers:
  - push to main on changes to workspace/cards.csv or workspace/media/**
  - manual dispatch
- Requires:
  - Repo secret OPENAI_API_KEY
- Output:
  - Builds deck via examples/config.example.yaml and uploads .apkg as an artifact.

2) Release to PyPI and GHCR on tag
- Workflow: .github/workflows/release.yml
- Triggers:
  - push tags like v*.*.*
- Requires:
  - PYPI_API_TOKEN secret (optional, for PyPI)
  - GITHUB_TOKEN (provided automatically) for GHCR
- Output:
  - Builds wheel/sdist, publishes to PyPI (if token present), and builds/pushes Docker image to GHCR.

--------------------------------------------------------------------------------

Developer guide

Project structure (src layout)
- src/pdf2anki/
  - __init__.py
  - cli.py
  - config.py
  - pdf.py
  - chunking.py
  - llm.py
  - prompts.py
  - preprocess.py
  - build.py
  - ids.py
  - io.py
  - validate.py
  - telemetry.py
  - dedup.py
  - rag.py
  - strategies/
    - base.py
    - key_points.py
    - cloze_definitions.py
    - figure_based.py
- prompts/
  - key_points.j2
  - cloze_definitions.j2
  - reviewer.j2
- examples/
  - config.example.yaml
- docs/
  - IMPLEMENTATION_PLAN.md
- tests/
  - unit/
  - integration/

Coding standards
- Formatting: black
- Linting: ruff
- Types: mypy (optional relaxed)
- Tests: pytest (+ pytest‑cov)

Common dev tasks
- Install dev deps:
  - pip install .[dev]
- Run linters/formatters:
  - ruff check .
  - black --check .
- Run tests:
  - pytest -q
- Run a fast E2E with mocked LLM (recommended):
  - pytest tests/integration -k e2e_mock -q

Contributing
- Create small, focused PRs.
- Add or update tests for new behavior.
- Keep prompts and strategies versioned; bump template_version in prompts on semantic changes.

--------------------------------------------------------------------------------

Best practices

- Keep temperature=0.0 for deterministic behavior.
- Use the reviewer gate when generating at scale to maintain quality.
- Start with small k in RAG‑lite (e.g., k=5–8) and increase only if needed.
- Avoid enabling OCR or Camelot unless required; they add dependencies and time.
- Prefer content_hash IDs during large/batch generation; switch to persistent for curated, stable decks you hand‑edit over time.

--------------------------------------------------------------------------------

FAQ

- Do I need MathJax? No; Anki includes MathJax. Just keep LaTeX inside $...$ or $$...$$.
- How do I avoid duplicates? Enable deduplication; tune thresholds and policy in config.
- Can I run without PDFs? Yes. If you have cards.csv and media/, run validate then build.
- How do I add a new strategy? Add a module under strategies/ and a prompt template; register it in config.strategies.

--------------------------------------------------------------------------------

License

MIT © Nicholas McCarthy

--------------------------------------------------------------------------------

Acknowledgements

- genanki for Anki packaging
- PyMuPDF for fast PDF parsing
- LangChain for LLM orchestration
- The Anki community for templates and MathJax guidance
