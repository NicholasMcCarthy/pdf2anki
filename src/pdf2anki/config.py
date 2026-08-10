# config.py
"""
Lightweight, permissive config using dataclasses (no Pydantic).
- Sensible defaults
- Tolerant YAML loading (unknown keys ignored)
- Strategies are a LIST in base config
- DocumentConfig can override/compose ingestion + strategies
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_origin, get_args, get_type_hints
import copy
import yaml


# --------------------------- helpers ---------------------------------

def _resolve_env_value(v: Optional[str]) -> Optional[str]:
    """Allow ${ENV_VAR} in YAML to resolve from environment."""
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return os.getenv(v[2:-1])
    return v


def _deep_merge(a: Any, b: Any) -> Any:
    """
    Recursively merge b into a and return a new object.
    - dicts: deep merge
    - lists: override (b wins) – avoids accidental duplication
    - scalars: override (b wins)
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    # For lists and scalars, prefer b if provided (explicit override)
    return copy.deepcopy(b)


def _select_dict_keys(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def _dataclass_from_dict(dc_type, data: Dict[str, Any]):
    """Create dataclass instance from dict, ignoring unknown keys, recursing into nested dataclasses."""
    if not is_dataclass(dc_type):
        return data  # not a dataclass type
    
    # Get resolved type hints to handle forward references
    try:
        type_hints = get_type_hints(dc_type)
    except (NameError, AttributeError):
        # Fallback to field types if type hints can't be resolved
        type_hints = {f.name: f.type for f in fields(dc_type)}
    
    kwargs = {}
    for f in fields(dc_type):
        key = f.name
        if key not in data:
            continue
        raw = data[key]
        
        # Use resolved type hint instead of field type
        field_type = type_hints.get(key, f.type)
        
        # handle Optional[T], List[T], Dict[K,V], nested dataclasses
        origin = get_origin(field_type)
        args = get_args(field_type)

        if is_dataclass(field_type) and isinstance(raw, dict):
            kwargs[key] = _dataclass_from_dict(field_type, raw)

        elif field_type is Path:
            kwargs[key] = Path(raw) if isinstance(raw, str) else raw

        elif origin is Union and len(args) == 2 and type(None) in args:
            inner = args[0] if args[1] is type(None) else args[1]
            if is_dataclass(inner) and isinstance(raw, dict):
                kwargs[key] = _dataclass_from_dict(inner, raw)
            elif inner is Path:
                kwargs[key] = Path(raw) if isinstance(raw, str) else raw
            else:
                kwargs[key] = raw

        elif origin in (list, List) and isinstance(raw, list):
            inner = args[0] if args else Any
            if is_dataclass(inner):
                kwargs[key] = [_dataclass_from_dict(inner, x) if isinstance(x, dict) else x for x in raw]
            else:
                kwargs[key] = raw

        elif origin in (dict, Dict) and isinstance(raw, dict):
            k_t, v_t = (args + (Any, Any))[:2]
            if is_dataclass(v_t):
                kwargs[key] = {k: _dataclass_from_dict(v_t, v) if isinstance(v, dict) else v for k, v in raw.items()}
            else:
                kwargs[key] = raw
        else:
            kwargs[key] = raw
    return dc_type(**kwargs)


def _coerce_to_dc(dc_obj, patch: Dict[str, Any]):
    """Merge dict patch into dataclass by converting to dict, deep-merge, then rebuild."""
    merged = _deep_merge(asdict(dc_obj), patch or {})
    return _dataclass_from_dict(type(dc_obj), merged)


def _as_plain(o: Any) -> Any:
    """asdict() with Path/env/enum-safe scalars for YAML dump."""
    if is_dataclass(o):
        return {k: _as_plain(v) for k, v in asdict(o).items()}
    if isinstance(o, dict):
        return {k: _as_plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_as_plain(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return o


def _unique_by_name(strategies: List["Strategy"]) -> List["Strategy"]:
    """Keep last occurrence by name to allow overrides to win."""
    seen = {}
    for s in strategies:
        if s.name:  # safe guard
            seen[s.name] = s
    return list(seen.values())


# --------------------------- dataclasses ------------------------------

@dataclass
class Project:
    name: str = "pdf2anki"
    version: str = "1.0"
    author: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Inputs:
    paths: List[Union[str, Path]] = field(default_factory=lambda: ["pdfs/"])
    patterns: List[str] = field(default_factory=lambda: ["*.pdf"])
    recursive: bool = True


@dataclass
class Chunking:
    mode: str = "smart"                 # pages|sections|paragraphs|smart|figures|highlights|entire|outline
    tokens_per_chunk: int = 2000
    overlap_tokens: int = 200
    respect_page_bounds: bool = True
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 4000
    enable_trimming: bool = True        # entire-mode helpers
    token_budget: int = 8000
    single_call_max_pages: int = 12     # highlights-mode: papers this short or shorter go in one LLM call, not chunked


@dataclass
class Ingestion:
    mode: str = "extract_text"
    chunking: Chunking = field(default_factory=Chunking)
    extract_images: bool = True
    extract_tables: bool = False
    extract_annotations: bool = False
    ocr_fallback: bool = False


@dataclass
class LLM:
    provider: str = "openai"
    model: str = "gpt-4-1106-preview"
    temperature: float = 0.0
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None  # supports ${ENV_VAR}
    base_url: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3

    def resolve_env(self) -> None:
        self.api_key = _resolve_env_value(self.api_key)


@dataclass
class Strategy:
    """Single generation strategy (list item)."""
    name: str = "basic_key_points"
    prompt: str = "key_points"
    note: str = "basic"
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    min_score: Optional[float] = None
    template_version: str = "1.0"


@dataclass
class Strategies:
    """Collection of strategy configurations."""
    key_points: Strategy = field(default_factory=lambda: Strategy(
        name="key_points", 
        prompt="key_points", 
        note="basic", 
        enabled=True
    ))
    cloze_definitions: Strategy = field(default_factory=lambda: Strategy(
        name="cloze_definitions", 
        prompt="cloze_definitions", 
        note="cloze", 
        enabled=True
    ))
    figure_based: Strategy = field(default_factory=lambda: Strategy(
        name="figure_based",
        prompt="figure_based",
        note="image_occlusion",
        enabled=True
    ))
    highlight_priority: Strategy = field(default_factory=lambda: Strategy(
        name="highlight_priority",
        prompt="highlight_priority",
        note="basic",
        enabled=False  # opt-in: only meaningful when ingestion.extract_annotations is on
    ))

    def items(self):
        """Support for dict-like .items() iteration."""
        return [
            ('key_points', self.key_points),
            ('cloze_definitions', self.cloze_definitions),
            ('figure_based', self.figure_based),
            ('highlight_priority', self.highlight_priority),
        ]

    def keys(self):
        """Support for dict-like .keys() iteration."""
        return ['key_points', 'cloze_definitions', 'figure_based', 'highlight_priority']

    def __getitem__(self, key):
        """Support for dict-like access."""
        if key == 'key_points':
            return self.key_points
        elif key == 'cloze_definitions':
            return self.cloze_definitions
        elif key == 'figure_based':
            return self.figure_based
        elif key == 'highlight_priority':
            return self.highlight_priority
        else:
            raise KeyError(key)

    def __deepcopy__(self, memo):
        """Custom deepcopy to avoid __dict__ issues."""
        return Strategies(
            key_points=copy.deepcopy(self.key_points, memo),
            cloze_definitions=copy.deepcopy(self.cloze_definitions, memo),
            figure_based=copy.deepcopy(self.figure_based, memo),
            highlight_priority=copy.deepcopy(self.highlight_priority, memo),
        )

    def __getattribute__(self, name):
        if name == '__dict__':
            # Return a dict-like view for legacy compatibility
            # Use object.__getattribute__ to avoid recursion
            try:
                key_points = object.__getattribute__(self, 'key_points')
                cloze_definitions = object.__getattribute__(self, 'cloze_definitions')
                figure_based = object.__getattribute__(self, 'figure_based')
                highlight_priority = object.__getattribute__(self, 'highlight_priority')
                return {
                    'key_points': key_points,
                    'cloze_definitions': cloze_definitions,
                    'figure_based': figure_based,
                    'highlight_priority': highlight_priority,
                }
            except AttributeError:
                # Fallback to regular __dict__ during object construction
                return object.__getattribute__(self, '__dict__')
        return super().__getattribute__(name)


@dataclass
class RAG:
    enabled: bool = False
    provider: str = "faiss"
    embedding_model: str = "text-embedding-3-large"
    k: int = 5
    index_path: Optional[str] = None


@dataclass
class Deduplication:
    enabled: bool = True
    fuzzy_threshold: float = 0.85
    embedding_threshold: float = 0.9
    policy: str = "or"      # or|and
    scope_within_run: bool = True
    scope_persistent: bool = False
    index_path: Optional[str] = None


@dataclass
class Hallucination:
    require_citations: bool = True
    verify_quotes: bool = True
    drop_on_failure: bool = True


@dataclass
class Review:
    enabled: bool = False
    min_score: float = 7.0
    allow_edits: bool = True
    template_version: str = "1.0"


@dataclass
class Tags:
    default_tags: List[str] = field(default_factory=list)
    auto_generate: bool = True
    include_source: bool = True
    include_strategy: bool = True


@dataclass
class Taxonomy:
    auto_detect: bool = True
    hierarchical: bool = True
    max_depth: int = 3


@dataclass
class Ids:
    strategy: str = "content_hash"   # content_hash|persistent
    salt: str = "pdf2anki"


@dataclass
class Anki:
    deck_name: str = "PDF2Anki"
    deck_id: Optional[int] = None
    # workflow|flat|chapter|theme - "workflow" (the default) puts cards from
    # each workflow into their own subdeck under deck_name (e.g.
    # "PDF2Anki::Readwise", "PDF2Anki::Textbooks::SomeBook",
    # "PDF2Anki::Articles") - see workflow_router.deck_subdeck_for_workflow()
    # and build.py::AnkiDeckBuilder._create_decks().
    deck_structure: str = "workflow"
    note_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preserve_latex: bool = True


@dataclass
class Output:
    workspace: Path = Path("workspace")
    csv_path: Path = Path("workspace/cards.csv")
    media_path: Path = Path("workspace/media")
    apkg_path: Path = Path("workspace/deck.apkg")
    manifest_path: Path = Path("workspace/manifest.json")


@dataclass
class Telemetry:
    enabled: bool = True
    track_tokens: bool = True
    track_costs: bool = True
    track_timing: bool = True
    track_cache_hits: bool = True


@dataclass
class WatchDirs:
    """Directories the watcher service polls for new input files."""
    pdfs: Optional[str] = "/data/pdfs"
    textbooks: Optional[str] = "/data/textbooks"
    readwise: Optional[str] = "/data/readwise"


@dataclass
class AnkiConnectSettings:
    """AnkiConnect is the only realistic way to push new cards into AnkiWeb
    automatically - AnkiWeb has no public upload API. Requires a real Anki
    desktop instance (with the AnkiConnect add-on) reachable at `url`,
    already logged into AnkiWeb. Failures here are always non-fatal: the
    service still writes the local .apkg regardless of whether AnkiConnect
    is reachable."""
    enabled: bool = False
    url: str = "http://host.docker.internal:8765"
    sync_after_update: bool = True
    timeout: int = 30


@dataclass
class NotificationSettings:
    slack_webhook_url: Optional[str] = None  # supports ${ENV_VAR}
    notify_on_processed: bool = True
    notify_on_error: bool = True

    def resolve_env(self) -> None:
        self.slack_webhook_url = _resolve_env_value(self.slack_webhook_url)


@dataclass
class Service:
    """Config for the `pdf2anki serve` watcher service (see src/pdf2anki/service/)."""
    watch_dirs: WatchDirs = field(default_factory=WatchDirs)
    poll_interval_seconds: int = 300  # periodic reconciliation fallback
    debounce_seconds: float = 5.0
    # Persists which files have already been processed - and which errored,
    # with why - across restarts/rebuilds, so the startup reconciliation scan
    # doesn't re-run the full (LLM-calling) pipeline on files it already
    # handled, and doesn't keep re-attempting (and re-spending API credits
    # on) files that already failed. Only genuinely new or modified (e.g.
    # explicitly `touch`ed) files get (re)processed. See
    # service/watcher.py::WatcherState - `pdf2anki watch-status` inspects
    # this file; `pdf2anki serve --retry-errors`/`--reset-state` clear it.
    # Defaults to a file inside Output.workspace (already a persistent
    # volume in the Docker deployment) - see cli.py's `serve` command. Set
    # to null/empty to disable persistence (in-memory only, the old
    # behavior - every restart reprocesses everything found).
    state_path: Optional[str] = None
    ankiconnect: AnkiConnectSettings = field(default_factory=AnkiConnectSettings)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)


@dataclass
class Pipeline:
    ingestion: Ingestion = field(default_factory=Ingestion)
    llm: LLM = field(default_factory=LLM)
    rag: RAG = field(default_factory=RAG)
    deduplication: Deduplication = field(default_factory=Deduplication)
    hallucination: Hallucination = field(default_factory=Hallucination)
    review: Review = field(default_factory=Review)
    telemetry: Telemetry = field(default_factory=Telemetry)


@dataclass
class Generate:
    strategies: Strategies = field(default_factory=Strategies)
    tags: Tags = field(default_factory=Tags)
    taxonomy: Taxonomy = field(default_factory=Taxonomy)
    ids: Ids = field(default_factory=Ids)
    language: str = "en"
    anki: Anki = field(default_factory=Anki)
    output: Output = field(default_factory=Output)


@dataclass
class Config:
    """Primary config (permissive)."""
    project: Project = field(default_factory=Project)
    inputs: Inputs = field(default_factory=Inputs)
    pipeline: Pipeline = field(default_factory=Pipeline)
    generate: Generate = field(default_factory=Generate)
    service: Service = field(default_factory=Service)

    # ---------- I/O ----------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        if not Path(path).exists():
            cfg = cls()
            cfg.pipeline.llm.resolve_env()
            cfg.service.notifications.resolve_env()
            return cfg

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle legacy format - move top-level keys to correct nested locations
        legacy_keys = ["llm", "ingestion", "rag", "deduplication", "hallucination", "review", "telemetry"]
        has_legacy = any(key in data for key in legacy_keys)
        
        if has_legacy:
            warnings.warn(
                "Legacy configuration format detected. Please update to the new nested format. "
                "See documentation for migration guide.",
                DeprecationWarning,
                stacklevel=2
            )
            
            # Convert legacy flat structure to nested structure
            if "pipeline" not in data:
                data["pipeline"] = {}
            
            for key in legacy_keys:
                if key in data:
                    data["pipeline"][key] = data.pop(key)
            
            # Handle legacy strategy structure
            if "strategies" in data:
                if "generate" not in data:
                    data["generate"] = {}
                data["generate"]["strategies"] = data.pop("strategies")
            
            # Handle other legacy generate-level keys
            legacy_generate_keys = ["anki", "output", "tags", "taxonomy", "ids", "language"]
            for key in legacy_generate_keys:
                if key in data:
                    if "generate" not in data:
                        data["generate"] = {}
                    data["generate"][key] = data.pop(key)

        # Merge raw YAML into defaults as dicts (permits unknown keys)
        merged = _deep_merge(_as_plain(cls()), data)

        # Build strongly-typed dataclasses from merged dict (ignore unknowns)
        cfg = _dataclass_from_dict(cls, merged)

        # Resolve ${ENV} just-in-time
        cfg.pipeline.llm.resolve_env()
        cfg.service.notifications.resolve_env()
        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_as_plain(self), f, sort_keys=False, indent=2)

    # ---------- utils ----------
    def create_workspace(self) -> None:
        self.generate.output.workspace.mkdir(parents=True, exist_ok=True)
        self.generate.output.media_path.mkdir(parents=True, exist_ok=True)

    # Legacy convenience accessors (optional)
    @property
    def ingestion(self) -> Ingestion: return self.pipeline.ingestion
    @property
    def llm(self) -> LLM: return self.pipeline.llm
    @property
    def strategies(self) -> Strategies: return self.generate.strategies
    @property
    def rag(self) -> RAG: return self.pipeline.rag
    @property
    def deduplication(self) -> Deduplication: return self.pipeline.deduplication
    @property
    def hallucination(self) -> Hallucination: return self.pipeline.hallucination
    @property
    def review(self) -> Review: return self.pipeline.review
    @property
    def telemetry(self) -> Telemetry: return self.pipeline.telemetry
    @property
    def tags(self) -> Tags: return self.generate.tags
    @property
    def taxonomy(self) -> Taxonomy: return self.generate.taxonomy
    @property
    def ids(self) -> Ids: return self.generate.ids
    @property
    def language(self) -> str: return self.generate.language
    @property
    def anki(self) -> Anki: return self.generate.anki
    @property
    def output(self) -> Output: return self.generate.output


# ----------------------- documents.yaml structures --------------------

@dataclass
class DocumentMetadata:
    page_count: int = 0
    toc_present: bool = False
    chapters_detected: bool = False
    abstract_present: bool = False
    references_present: bool = False
    two_column_layout: bool = False
    has_doi: bool = False
    file_size: Optional[int] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    doc_type: str = "unknown"  # research_paper|textbook|unknown


@dataclass
class DocumentConfig:
    # File information
    file_path: str = ""
    file_hash: Optional[str] = None

    # Extracted metadata
    metadata: Optional[DocumentMetadata] = None

    # Heuristic suggestions (from scan)
    heuristic_chunking: Optional[Chunking] = None
    heuristic_strategies: Optional[List[Union[str, Dict[str, Any]]]] = None
    heuristic_extract_annotations: Optional[bool] = None

    # Explicit overrides (user-defined)
    override_chunking: Optional[Chunking] = None
    override_strategies: Optional[List[Union[str, Dict[str, Any]]]] = None
    override_ingestion: Optional[Ingestion] = None
    override_extract_annotations: Optional[bool] = None

    # Resolved/overridden workflow (textbook|academic_paper|readwise|generic).
    # Set by the user to force classification of a specific document; scan-docs
    # also writes back its resolved value (auto-detected or override-honoring)
    # here for visibility - see workflow_router.select_workflow().
    workflow: Optional[str] = None

    # Processing flags
    enabled: bool = True
    last_scanned: Optional[str] = None


@dataclass
class Documents:
    version: str = "1.0"
    documents: Dict[str, DocumentConfig] = field(default_factory=dict)
    global_overrides: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Documents":
        if not Path(path).exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # normalize 'documents'
        docs = data.get("documents") or {}
        data["documents"] = {
            k: _dataclass_from_dict(DocumentConfig, v if isinstance(v, dict) else {})
            for k, v in docs.items()
        }
        return _dataclass_from_dict(cls, data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_as_plain(self), f, sort_keys=False, indent=2)

    # ---------------- merge logic with base Config ---------------------

    def get_effective_config(self, document_key: str, base_config: Config) -> Config:
        """
        Precedence:
          1) base_config (defaults + config.yaml)
          2) heuristic_* (from documents.yaml)
          3) override_*  (from documents.yaml)
        """
        doc = self.documents.get(document_key)
        if not doc:
            return copy.deepcopy(base_config)

        eff = copy.deepcopy(base_config)

        # ---- Ingestion & Chunking ----
        # Start from base ingestion, merge heuristic chunking, then override chunking.
        if doc.heuristic_chunking:
            eff.pipeline.ingestion.chunking = _coerce_to_dc(
                eff.pipeline.ingestion.chunking,
                _as_plain(doc.heuristic_chunking)
            )
        if doc.override_chunking:
            eff.pipeline.ingestion.chunking = _coerce_to_dc(
                eff.pipeline.ingestion.chunking,
                _as_plain(doc.override_chunking)
            )
        # Full ingestion override (merge-by-field to keep other flags)
        if doc.override_ingestion:
            eff.pipeline.ingestion = _coerce_to_dc(
                eff.pipeline.ingestion,
                _as_plain(doc.override_ingestion)
            )

        # extract_annotations is applied as a single explicit flag rather than
        # via override_ingestion, since override_ingestion's whole-object merge
        # would otherwise clobber heuristic_chunking with Ingestion()'s defaults.
        if doc.heuristic_extract_annotations is not None:
            eff.pipeline.ingestion.extract_annotations = doc.heuristic_extract_annotations
        if doc.override_extract_annotations is not None:
            eff.pipeline.ingestion.extract_annotations = doc.override_extract_annotations

        # ---- Strategies ----
        # Convert Strategies object to list for processing, then back to object
        def strategies_to_list(strategies: Strategies) -> List[Strategy]:
            """Convert Strategies dataclass to list for processing."""
            return [
                strategies.key_points,
                strategies.cloze_definitions,
                strategies.figure_based,
                strategies.highlight_priority,
            ]

        def list_to_strategies(strategy_list: List[Strategy]) -> Strategies:
            """Convert list back to Strategies dataclass."""
            result = Strategies()
            for strategy in strategy_list:
                if strategy.name == "key_points":
                    result.key_points = strategy
                elif strategy.name == "cloze_definitions":
                    result.cloze_definitions = strategy
                elif strategy.name == "figure_based":
                    result.figure_based = strategy
                elif strategy.name == "highlight_priority":
                    result.highlight_priority = strategy
            return result
        
        base_list = strategies_to_list(eff.generate.strategies)

        def _normalize_strategy_item(x: Union[str, Dict[str, Any], Strategy]) -> Strategy:
            if isinstance(x, Strategy):
                return x
            if isinstance(x, str):
                # If only a name is given, adopt base item if present; else create a simple Strategy.
                ref = next((s for s in base_list if s.name == x), None)
                return copy.deepcopy(ref) if ref else Strategy(name=x, prompt=x, note="basic")
            if isinstance(x, dict):
                # If dict has name: merge onto base same-name, else create fresh
                name = x.get("name", "")
                ref = next((s for s in base_list if s.name == name), None)
                merged = _as_plain(ref) if ref else _as_plain(Strategy(name=name))
                merged = _deep_merge(merged, x)
                return _dataclass_from_dict(Strategy, merged)
            return Strategy()  # fallback harmless default

        # Compose order: base -> heuristic (append/override) -> override (append/override).
        composed = base_list[:]
        if doc.heuristic_strategies:
            composed.extend(_normalize_strategy_item(x) for x in doc.heuristic_strategies)
        if doc.override_strategies:
            composed.extend(_normalize_strategy_item(x) for x in doc.override_strategies)

        eff.generate.strategies = list_to_strategies(_unique_by_name(composed))

        return eff

    def add_or_update_document(self, file_path: str, metadata: DocumentMetadata) -> None:
        key = Path(file_path).name
        existing = self.documents.get(key)
        if existing:
            existing.metadata = metadata
        else:
            self.documents[key] = DocumentConfig(
                file_path=file_path,
                metadata=metadata,
            )


# ------------------------ type aliases and compatibility ---------------------

# Type aliases for backward compatibility with existing imports
ChunkingConfig = Chunking
LLMConfig = LLM  
StrategyConfig = Strategy
RAGConfig = RAG
DeduplicationConfig = Deduplication
TelemetryConfig = Telemetry
IdsConfig = Ids
GenerateConfig = Generate
PipelineConfig = Pipeline
DocumentsConfig = Documents
ReviewConfig = Review

# Enums converted to string constants
class IdStrategy:
    CONTENT_HASH = "content_hash"
    PERSISTENT = "persistent"

class DeduplicationPolicy:
    OR = "or"
    AND = "and"

class DocumentType:
    RESEARCH_PAPER = "research_paper"
    TEXTBOOK = "textbook"
    UNKNOWN = "unknown"

class ChunkingMode:
    PAGES = "pages"
    SECTIONS = "sections"
    PARAGRAPHS = "paragraphs"
    SMART = "smart"
    FIGURES = "figures"
    HIGHLIGHTS = "highlights"
    ENTIRE = "entire"
    OUTLINE = "outline"

# --------------------------- usage example ---------------------------
# cfg = Config.from_yaml("config.yaml")
# docs = Documents.from_yaml("documents.yaml")
# eff = docs.get_effective_config("example.pdf", cfg)
# eff.to_yaml("effective.yaml")
# eff.create_workspace()
