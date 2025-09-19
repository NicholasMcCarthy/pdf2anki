"""LLM integration with caching and retry logic."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Represents an LLM response with metadata."""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    cached: bool = False
    response_time: float = 0.0
    seed: Optional[int] = None


class TokenUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMProvider:
    """Base LLM provider interface."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.api_calls = 0
        
        # Set up caching
        cache_path = ".llm_cache/langchain.db"
        set_llm_cache(SQLiteCache(database_path=cache_path))
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM instance."""
        if self.config.provider.value == "openai":
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.api_key,
                openai_api_base=self.config.base_url,
                request_timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                model_kwargs={"seed": self.config.seed} if self.config.seed else {},
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        max_retries: int = 3
    ) -> LLMResponse:
        """Generate response from LLM with retry logic."""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        
        # Configure for JSON mode if requested
        model_kwargs = {}
        if json_mode and "gpt" in self.config.model.lower():
            model_kwargs["response_format"] = {"type": "json_object"}
        
        for attempt in range(max_retries + 1):
            try:
                with get_openai_callback() as cb:
                    # Check if this is a cache hit by calling without callback first
                    test_response = self.llm.invoke(messages, **model_kwargs)
                    was_cached = cb.total_tokens == 0
                    
                    if not was_cached:
                        # Real API call
                        response = self.llm.invoke(messages, **model_kwargs)
                        tokens_used = cb.total_tokens
                        cost = cb.total_cost
                    else:
                        response = test_response
                        tokens_used = 0
                        cost = 0.0
                        self.cache_hits += 1
                    
                    self.api_calls += 1
                    self.total_tokens += tokens_used
                    self.total_cost += cost
                
                # Validate JSON if requested
                if json_mode:
                    try:
                        json.loads(response.content)
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            logger.warning(f"Invalid JSON response, retrying (attempt {attempt + 1}): {e}")
                            continue
                        else:
                            raise ValueError(f"Failed to get valid JSON after {max_retries} retries: {e}")
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response.content,
                    model=self.config.model,
                    tokens_used=tokens_used,
                    cost_estimate=cost,
                    cached=was_cached,
                    response_time=response_time,
                    seed=self.config.seed,
                )
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"LLM call failed, retrying in {wait_time}s (attempt {attempt + 1}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {max_retries} retries: {e}")
                    raise
        
        raise RuntimeError("Should never reach here")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cache_hits": self.cache_hits,
            "api_calls": self.api_calls,
            "cache_hit_rate": self.cache_hits / max(1, self.api_calls),
        }


class ModelRegistry:
    """Registry of supported models with metadata."""
    
    MODELS = {
        "gpt-4-1106-preview": {
            "provider": "openai",
            "context_length": 128000,
            "input_cost_per_1k": 0.01,
            "output_cost_per_1k": 0.03,
            "supports_json": True,
            "supports_seed": True,
        },
        "gpt-4": {
            "provider": "openai",
            "context_length": 8192,
            "input_cost_per_1k": 0.03,
            "output_cost_per_1k": 0.06,
            "supports_json": False,
            "supports_seed": False,
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "context_length": 16385,
            "input_cost_per_1k": 0.001,
            "output_cost_per_1k": 0.002,
            "supports_json": True,
            "supports_seed": False,
        },
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        return cls.MODELS.get(model_name, {})
    
    @classmethod
    def supports_json_mode(cls, model_name: str) -> bool:
        """Check if model supports JSON mode."""
        return cls.MODELS.get(model_name, {}).get("supports_json", False)
    
    @classmethod
    def supports_seed(cls, model_name: str) -> bool:
        """Check if model supports seed parameter."""
        return cls.MODELS.get(model_name, {}).get("supports_seed", False)
    
    @classmethod
    def estimate_cost(cls, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        model_info = cls.MODELS.get(model_name, {})
        input_cost = model_info.get("input_cost_per_1k", 0.01) * input_tokens / 1000
        output_cost = model_info.get("output_cost_per_1k", 0.03) * output_tokens / 1000
        return input_cost + output_cost


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create LLM provider."""
    return LLMProvider(config)


def clear_llm_cache() -> int:
    """Clear the LLM cache and return number of entries cleared."""
    try:
        import sqlite3
        from pathlib import Path
        
        cache_path = Path(".llm_cache/langchain.db")
        if not cache_path.exists():
            return 0
        
        conn = sqlite3.connect(str(cache_path))
        cursor = conn.cursor()
        
        # Count entries before clearing
        cursor.execute("SELECT COUNT(*) FROM full_llm_cache")
        count = cursor.fetchone()[0]
        
        # Clear cache
        cursor.execute("DELETE FROM full_llm_cache")
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {count} entries from LLM cache")
        return count
        
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")
        return 0
