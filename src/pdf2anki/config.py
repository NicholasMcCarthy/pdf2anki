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


class Config(BaseSettings):
    """Complete pdf2anki configuration."""
    
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    inputs: InputConfig = Field(default_factory=InputConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    tags: TagsConfig = Field(default_factory=TagsConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)
    ids: IdsConfig = Field(default_factory=IdsConfig)
    language: str = "en"
    anki: AnkiConfig = Field(default_factory=AnkiConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    def create_workspace(self) -> None:
        """Create workspace directories."""
        self.output.workspace.mkdir(parents=True, exist_ok=True)
        self.output.media_path.mkdir(parents=True, exist_ok=True)