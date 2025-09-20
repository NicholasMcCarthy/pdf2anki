"""Configuration management using Pydantic."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChunkingMode(str, Enum):
    """PDF chunking strategies."""
    PAGES = "pages"
    SECTIONS = "sections"
    PARAGRAPHS = "paragraphs"
    SMART = "smart"
    FIGURES = "figures"  # New: Extract and chunk based on figures
    HIGHLIGHTS = "highlights"  # New: Extract and chunk based on highlights
    ENTIRE = "entire"  # New: Prefer single chunk, auto-split if needed


class DeckStructure(str, Enum):
    """Anki deck organization strategies."""
    FLAT = "flat"
    BY_CHAPTER = "by_chapter"
    BY_THEME = "by_theme"
    PREDEFINED = "predefined"


class IdStrategy(str, Enum):
    """ID generation strategies."""
    CONTENT_HASH = "content_hash"
    PERSISTENT = "persistent"


class DeduplicationPolicy(str, Enum):
    """Deduplication merge policies."""
    OR = "or"
    AND = "and"


class DocumentType(str, Enum):
    """Detected document types."""
    RESEARCH_PAPER = "research_paper"
    TEXTBOOK = "textbook"
    UNKNOWN = "unknown"


class ProjectConfig(BaseModel):
    """Project-level configuration."""
    name: str = "pdf2anki"
    version: str = "1.0"
    author: Optional[str] = None
    description: Optional[str] = None


class InputConfig(BaseModel):
    """Input PDF configuration."""
    paths: List[Union[str, Path]] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=lambda: ["*.pdf"])
    recursive: bool = True


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    mode: ChunkingMode = ChunkingMode.SMART
    tokens_per_chunk: int = 2000
    overlap_tokens: int = 200
    respect_page_bounds: bool = True
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 4000
    # Entire mode specific settings
    enable_trimming: bool = True  # Enable/disable terminal section trimming
    token_budget: int = 8000  # Max tokens before auto-splitting in entire mode


class IngestionConfig(BaseModel):
    """PDF ingestion configuration."""
    mode: str = "extract_text"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    extract_images: bool = True
    extract_tables: bool = False
    ocr_fallback: bool = False


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: ProviderType = ProviderType.OPENAI
    model: str = "gpt-4-1106-preview"
    temperature: float = 0.0
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3

    @validator("api_key", pre=True)
    def resolve_api_key(cls, v):
        if v and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var)
        return v


class StrategyConfig(BaseModel):
    """Individual strategy configuration."""
    enabled: bool = True
    template_version: str = "1.0"
    params: Dict[str, Any] = Field(default_factory=dict)
    min_score: Optional[float] = None


class StrategiesConfig(BaseModel):
    """All generation strategies configuration."""
    key_points: StrategyConfig = Field(default_factory=StrategyConfig)
    cloze_definitions: StrategyConfig = Field(default_factory=StrategyConfig)
    figure_based: StrategyConfig = Field(default_factory=StrategyConfig)


class RAGConfig(BaseModel):
    """RAG-lite configuration."""
    enabled: bool = False
    provider: str = "faiss"
    embedding_model: str = "text-embedding-3-large"
    k: int = 5
    index_path: Optional[str] = None


class DeduplicationConfig(BaseModel):
    """Deduplication configuration."""
    enabled: bool = True
    fuzzy_threshold: float = 0.85
    embedding_threshold: float = 0.9
    policy: DeduplicationPolicy = DeduplicationPolicy.OR
    scope_within_run: bool = True
    scope_persistent: bool = False
    index_path: Optional[str] = None


class HallucinationConfig(BaseModel):
    """Hallucination mitigation configuration."""
    require_citations: bool = True
    verify_quotes: bool = True
    drop_on_failure: bool = True


class ReviewConfig(BaseModel):
    """Review step configuration."""
    enabled: bool = False
    min_score: float = 7.0
    allow_edits: bool = True
    template_version: str = "1.0"


class TagsConfig(BaseModel):
    """Tagging configuration."""
    default_tags: List[str] = Field(default_factory=list)
    auto_generate: bool = True
    include_source: bool = True
    include_strategy: bool = True


class TaxonomyConfig(BaseModel):
    """Content taxonomy configuration."""
    auto_detect: bool = True
    hierarchical: bool = True
    max_depth: int = 3


class IdsConfig(BaseModel):
    """ID generation configuration."""
    strategy: IdStrategy = IdStrategy.CONTENT_HASH
    salt: str = "pdf2anki"


class AnkiConfig(BaseModel):
    """Anki deck configuration."""
    deck_name: str = "PDF2Anki"
    deck_id: Optional[int] = None
    deck_structure: DeckStructure = DeckStructure.FLAT
    note_types: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    preserve_latex: bool = True


class OutputConfig(BaseModel):
    """Output paths configuration."""
    workspace: Path = Path("workspace")
    csv_path: Path = Path("workspace/cards.csv")
    media_path: Path = Path("workspace/media")
    apkg_path: Path = Path("workspace/deck.apkg")
    manifest_path: Path = Path("workspace/manifest.json")


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""
    enabled: bool = True
    track_tokens: bool = True
    track_costs: bool = True
    track_timing: bool = True
    track_cache_hits: bool = True


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


class GenerateConfig(BaseModel):
    """Generation-specific configuration."""
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    tags: TagsConfig = Field(default_factory=TagsConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    ids: IdsConfig = Field(default_factory=IdsConfig)
    language: str = "en"
    anki: AnkiConfig = Field(default_factory=AnkiConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class DocumentMetadata(BaseModel):
    """Metadata extracted from PDF document."""
    page_count: int
    toc_present: bool = False
    chapters_detected: bool = False
    abstract_present: bool = False
    references_present: bool = False
    two_column_layout: bool = False
    has_doi: bool = False
    doc_type: DocumentType = DocumentType.UNKNOWN
    file_size: Optional[int] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None


class DocumentConfig(BaseModel):
    """Per-document configuration with metadata and overrides."""
    # File information
    file_path: str
    file_hash: Optional[str] = None
    
    # Extracted metadata (populated by scan-docs)
    metadata: Optional[DocumentMetadata] = None
    
    # Heuristic defaults (populated by scan-docs)
    heuristic_chunking: Optional[ChunkingConfig] = None
    heuristic_strategies: Optional[List[str]] = None
    
    # Explicit overrides (user-defined)
    override_chunking: Optional[ChunkingConfig] = None
    override_strategies: Optional[List[str]] = None
    override_ingestion: Optional[IngestionConfig] = None
    
    # Processing flags
    enabled: bool = True
    last_scanned: Optional[str] = None


class DocumentsConfig(BaseModel):
    """Root documents.yaml configuration."""
    version: str = "1.0"
    documents: Dict[str, DocumentConfig] = Field(default_factory=dict)
    global_overrides: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DocumentsConfig":
        """Load documents configuration from YAML file."""
        if not Path(path).exists():
            return cls()
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save documents configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)
    
    def get_effective_config(self, document_key: str, base_config: "Config") -> "Config":
        """
        Compute effective configuration for a document using layering precedence:
        1. Global defaults from base_config
        2. Heuristic defaults per-PDF
        3. Explicit per-PDF overrides
        4. CLI overrides (handled by caller)
        """
        # TODO: Implement configuration layering logic
        # For now, return base config as-is
        return base_config
    
    def add_or_update_document(self, file_path: str, metadata: DocumentMetadata) -> None:
        """Add or update a document in the configuration."""
        key = str(Path(file_path).name)  # Use filename as key
        
        if key in self.documents:
            # Update existing document
            self.documents[key].metadata = metadata
            self.documents[key].last_scanned = str(Path().cwd())  # TODO: Use proper timestamp
        else:
            # Add new document
            self.documents[key] = DocumentConfig(
                file_path=file_path,
                metadata=metadata,
                last_scanned=str(Path().cwd())  # TODO: Use proper timestamp
            )


class Config(BaseSettings):
    """Complete pdf2anki configuration."""
    
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    inputs: InputConfig = Field(default_factory=InputConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file with legacy support."""
        import warnings
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Check if this is a legacy config (has top-level keys like llm, strategies, etc.)
        legacy_keys = {
            'ingestion', 'llm', 'strategies', 'rag', 'deduplication', 
            'hallucination', 'review', 'tags', 'taxonomy', 'ids', 'anki', 
            'output', 'telemetry', 'language'
        }
        
        found_legacy_keys = set(data.keys()) & legacy_keys
        
        if found_legacy_keys:
            warnings.warn(
                "Legacy configuration format detected. Please migrate to the new "
                "generate + pipeline schema. Legacy support will be removed in the next minor version.",
                DeprecationWarning,
                stacklevel=2
            )
            
            # Auto-translate legacy config
            translated_data = {
                'project': data.get('project', {}),
                'inputs': data.get('inputs', {}),
                'pipeline': {
                    'ingestion': data.get('ingestion', {}),
                    'llm': data.get('llm', {}),
                    'rag': data.get('rag', {}),
                    'deduplication': data.get('deduplication', {}),
                    'hallucination': data.get('hallucination', {}),
                    'review': data.get('review', {}),
                    'telemetry': data.get('telemetry', {}),
                },
                'generate': {
                    'strategies': data.get('strategies', {}),
                    'tags': data.get('tags', {}),
                    'taxonomy': data.get('taxonomy', {}),
                    'ids': data.get('ids', {}),
                    'language': data.get('language', 'en'),
                    'anki': data.get('anki', {}),
                    'output': data.get('output', {}),
                }
            }
            data = translated_data
        
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file with proper scalar serialization."""
        def represent_enum(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data.value)
        
        def represent_path(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
        
        # Register custom representers for enums and paths
        yaml.add_representer(ProviderType, represent_enum)
        yaml.add_representer(ChunkingMode, represent_enum)
        yaml.add_representer(DeckStructure, represent_enum)
        yaml.add_representer(IdStrategy, represent_enum)
        yaml.add_representer(DeduplicationPolicy, represent_enum)
        yaml.add_representer(DocumentType, represent_enum)
        yaml.add_representer(Path, represent_path)
        # Support for different pathlib types
        import pathlib
        yaml.add_representer(pathlib.PosixPath, represent_path)
        yaml.add_representer(pathlib.WindowsPath, represent_path)
        yaml.add_representer(pathlib.PurePath, represent_path)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)

    def create_workspace(self) -> None:
        """Create workspace directories."""
        self.generate.output.workspace.mkdir(parents=True, exist_ok=True)
        self.generate.output.media_path.mkdir(parents=True, exist_ok=True)
    
    # Legacy property access for backward compatibility
    @property
    def ingestion(self):
        """Legacy access to pipeline.ingestion."""
        return self.pipeline.ingestion
    
    @property 
    def llm(self):
        """Legacy access to pipeline.llm."""
        return self.pipeline.llm
    
    @property
    def strategies(self):
        """Legacy access to generate.strategies."""
        return self.generate.strategies
    
    @property
    def rag(self):
        """Legacy access to pipeline.rag."""
        return self.pipeline.rag
    
    @property
    def deduplication(self):
        """Legacy access to pipeline.deduplication."""
        return self.pipeline.deduplication
    
    @property
    def hallucination(self):
        """Legacy access to pipeline.hallucination."""
        return self.pipeline.hallucination
    
    @property
    def review(self):
        """Legacy access to pipeline.review.""" 
        return self.pipeline.review
    
    @property
    def tags(self):
        """Legacy access to generate.tags."""
        return self.generate.tags
    
    @property
    def taxonomy(self):
        """Legacy access to generate.taxonomy."""
        return self.generate.taxonomy
    
    @property
    def ids(self):
        """Legacy access to generate.ids."""
        return self.generate.ids
    
    @property
    def language(self):
        """Legacy access to generate.language."""
        return self.generate.language
    
    @property
    def anki(self):
        """Legacy access to generate.anki."""
        return self.generate.anki
    
    @property
    def output(self):
        """Legacy access to generate.output."""
        return self.generate.output
    
    @property 
    def telemetry(self):
        """Legacy access to pipeline.telemetry."""
        return self.pipeline.telemetry