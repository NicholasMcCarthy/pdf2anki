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
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_origin, get_args
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
    kwargs = {}
    for f in fields(dc_type):
        key = f.name
        if key not in data:
            continue
        raw = data[key]
        # handle Optional[T], List[T], Dict[K,V], nested dataclasses
        origin = get_origin(f.type)
        args = get_args(f.type)

        if is_dataclass(f.type) and isinstance(raw, dict):
            kwargs[key] = _dataclass_from_dict(f.type, raw)

        elif origin is Union and len(args) == 2 and type(None) in args:
            inner = args[0] if args[1] is type(None) else args[1]
            if is_dataclass(inner) and isinstance(raw, dict):
                kwargs[key] = _dataclass_from_dict(inner, raw)
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
    mode: str = "smart"                 # pages|sections|paragraphs|smart|figures|highlights|entire
    tokens_per_chunk: int = 2000
    overlap_tokens: int = 200
    respect_page_bounds: bool = True
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 4000
    enable_trimming: bool = True        # entire-mode helpers
    token_budget: int = 8000


@dataclass
class Ingestion:
    mode: str = "extract_text"
    chunking: Chunking = field(default_factory=Chunking)
    extract_images: bool = True
    extract_tables: bool = False
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
    deck_structure: str = "flat"     # flat|by_chapter|by_theme|predefined
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
    strategies: List[Strategy] = field(default_factory=lambda: [Strategy()])
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

    # ---------- I/O ----------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        if not Path(path).exists():
            cfg = cls()
            cfg.pipeline.llm.resolve_env()
            return cfg

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Merge raw YAML into defaults as dicts (permits unknown keys)
        merged = _deep_merge(_as_plain(cls()), data)

        # Build strongly-typed dataclasses from merged dict (ignore unknowns)
        cfg = _dataclass_from_dict(cls, merged)

        # Resolve ${ENV} just-in-time
        cfg.pipeline.llm.resolve_env()
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
    def strategies(self) -> List[Strategy]: return self.generate.strategies
    @property
    def rag(self) -> RAG: return self.pipeline.rag
    @property
    def deduplication(self) -> Deduplication: return self.pipeline.deduplication
    @property
    def hallucination(self) -> Hallucination: return self.pipeline.hallucination
    @property
    def review(self) -> Review: return self.pipeline.review
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

    # Explicit overrides (user-defined)
    override_chunking: Optional[Chunking] = None
    override_strategies: Optional[List[Union[str, Dict[str, Any]]]] = None
    override_ingestion: Optional[Ingestion] = None

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

        # ---- Strategies (list) ----
        # Base list -> optionally filter/compose by heuristic/override lists.
        base_list = copy.deepcopy(eff.generate.strategies)

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

        eff.generate.strategies = _unique_by_name(composed)

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


# --------------------------- usage example ---------------------------
# cfg = Config.from_yaml("config.yaml")
# docs = Documents.from_yaml("documents.yaml")
# eff = docs.get_effective_config("example.pdf", cfg)
# eff.to_yaml("effective.yaml")
# eff.create_workspace()
