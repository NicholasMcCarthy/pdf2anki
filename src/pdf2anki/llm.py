"""LLM integration with caching and retry logic."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from .config import Config, LLMConfig

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


def _extract_text_content(content: Any) -> str:
    """Normalize a LangChain AIMessage.content value to a plain string.

    Models capable of structured/multi-block responses (e.g. extended
    thinking) can return `content` as a list of content blocks - typically
    dicts like {"type": "text", "text": "..."} interleaved with non-text
    blocks (thinking, redacted_thinking, tool_use, etc.) - instead of a plain
    string. json.loads() and downstream code expect a string; concatenate
    just the text blocks, in order, ignoring everything else.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        result = "".join(parts)
        if not result and content:
            # A successful (200 OK) call that nonetheless yields no usable text -
            # e.g. every block was "thinking"/reasoning with none marked "text".
            # Surface exactly what came back rather than let the caller see only
            # a downstream "Expecting value" JSON error with no clue why.
            block_types = [b.get("type") if isinstance(b, dict) else type(b).__name__ for b in content]
            logger.warning(
                f"LLM response had no text content blocks (block types: {block_types}) - "
                f"returning empty string. If this recurs, the model may be exhausting "
                f"max_tokens on internal reasoning before emitting an answer."
            )
        return result
    return str(content)


def _rejects_temperature(error: Exception) -> bool:
    """Some newer models reject the `temperature` param outright (e.g.
    Anthropic returning a 400 "`temperature` is deprecated for this model").
    Detected by message rather than a hardcoded model-name list, since which
    models this applies to changes over time and isn't documented per-model
    anywhere the config can check in advance."""
    msg = str(error).lower()
    return "temperature" in msg and any(
        phrase in msg for phrase in ("deprecated", "not supported", "unsupported", "not allowed")
    )


class LLMProvider:
    """Base LLM provider interface."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.api_calls = 0
        self._omit_temperature = False

        # Set up caching
        cache_path = ".llm_cache/langchain.db"
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=cache_path))

        # Initialize the LLM
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM instance."""
        temperature_kwargs = {} if self._omit_temperature else {"temperature": self.config.temperature}

        if self.config.provider == "openai":
            return ChatOpenAI(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.api_key,
                openai_api_base=self.config.base_url,
                request_timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                model_kwargs={"seed": self.config.seed} if self.config.seed else {},
                **temperature_kwargs,
            )
        elif self.config.provider == "anthropic":
            # Anthropic requires an explicit max_tokens (unlike OpenAI, None isn't
            # accepted). Default raised from 4096: models that reject `temperature`
            # (see _rejects_temperature() above) are consistent with reasoning-first
            # models whose internal reasoning counts against the same output token
            # budget as the visible answer - a small max_tokens risks the budget
            # being fully consumed before any answer text is emitted, producing an
            # empty response that still parses as a successful (200 OK) API call.
            return ChatAnthropic(
                model=self.config.model,
                max_tokens=self.config.max_tokens or 8192,
                anthropic_api_key=self.config.api_key,
                anthropic_api_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                cache=False,  # Avoid caching empty/failed Anthropic responses
                **temperature_kwargs,
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
        
        # Configure for JSON mode if requested. response_format={"type": "json_object"}
        # is an OpenAI-specific feature; other providers rely on prompt instructions
        # plus the parse-and-retry loop below.
        model_kwargs = {}
        if json_mode and self.config.provider == "openai" and "gpt" in self.config.model.lower():
            model_kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries + 1):
            try:
                if self.config.provider == "openai":
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
                else:
                    # get_openai_callback only instruments OpenAI calls, so for other
                    # providers (e.g. Anthropic) skip the cache-hit heuristic rather than
                    # report misleading zero-token/zero-cost "cache hits" for real calls.
                    response = self.llm.invoke(messages, **model_kwargs)
                    was_cached = False
                    tokens_used = 0
                    cost = 0.0
                    self.api_calls += 1
                
                # response.content is usually a plain string, but models capable of
                # structured/multi-block responses (e.g. extended thinking) can return
                # a list of content blocks instead - normalize before any string use.
                response_text = _extract_text_content(response.content)

                if not response_text.strip():
                    # Empty response text with no diagnostic already logged by
                    # _extract_text_content (that only fires for a non-empty list
                    # with no text blocks) - most likely response.content was
                    # already an empty string. Log everything the AIMessage
                    # carries about *why* - stop_reason/usage in response_metadata
                    # is the key signal (e.g. "max_tokens" = truncated before any
                    # output, vs "end_turn" = the model deliberately said nothing).
                    logger.warning(
                        f"LLM returned an empty response body on a successful call. "
                        f"content={response.content!r} "
                        f"response_metadata={getattr(response, 'response_metadata', None)!r} "
                        f"usage_metadata={getattr(response, 'usage_metadata', None)!r}"
                    )

                # Validate JSON if requested
                if json_mode:
                    try:
                        json.loads(response_text)
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            logger.warning(f"Invalid JSON response, retrying (attempt {attempt + 1}): {e}")
                            continue
                        else:
                            raise ValueError(f"Failed to get valid JSON after {max_retries} retries: {e}")

                response_time = time.time() - start_time

                return LLMResponse(
                    content=response_text,
                    model=self.config.model,
                    tokens_used=tokens_used,
                    cost_estimate=cost,
                    cached=was_cached,
                    response_time=response_time,
                    seed=self.config.seed,
                )
                
            except Exception as e:
                # This is a deterministic rejection of a request parameter, not a
                # transient failure - retrying the identical request would just
                # fail the same way every time. Rebuild the client without
                # temperature and retry immediately, without consuming the
                # normal exponential-backoff budget below.
                if not self._omit_temperature and _rejects_temperature(e):
                    logger.warning(
                        f"Model {self.config.model} rejected the 'temperature' parameter - "
                        f"retrying without it: {e}"
                    )
                    self._omit_temperature = True
                    self.llm = self._initialize_llm()
                    continue

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
    
    def review_card(self, card_data: Dict[str, Any], context: Dict[str, Any], prompt_manager, template_name: str = "reviewer_special.j2") -> Dict[str, Any]:
        """Review a single flashcard and return assessment with strict JSON output.
        
        Args:
            card_data: The flashcard data to review
            context: Source context (pdf_title, page_start, page_end, strategy, section)
            prompt_manager: The prompt manager instance
            template_name: Name of the template to use for review
            
        Returns:
            Dict containing: id, score, issues, edited (optional)
        """
        import json
        
        # Render the review prompt using the prompt manager
        try:
            prompt = prompt_manager.render_template(template_name, 
                                                  card_data=card_data,
                                                  **context)
        except Exception as e:
            logger.error(f"Failed to render review template: {e}")
            return {
                "id": card_data.get("id", "unknown"),
                "score": 5.0,
                "issues": [f"Template rendering failed: {str(e)}"],
                "edited": None
            }
        
        # Generate review with JSON mode
        response = self.generate(
            prompt=prompt,
            json_mode=True,
            max_retries=3
        )
        
        try:
            # Parse the JSON response
            review_data = json.loads(response.content)
            
            # Validate required fields
            if "id" not in review_data:
                review_data["id"] = card_data.get("id", "unknown")
            if "score" not in review_data:
                raise ValueError("Review response missing required 'score' field")
            if "issues" not in review_data:
                review_data["issues"] = []
                
            # Ensure score is numeric
            review_data["score"] = float(review_data["score"])
            
            return review_data
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse review response, using fallback: {e}")
            # Return a fallback review indicating parsing issues
            return {
                "id": card_data.get("id", "unknown"),
                "score": 5.0,  # Below threshold to be safe
                "issues": [f"Review parsing failed: {str(e)}"],
                "edited": None
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
        "claude-opus-5": {
            "provider": "anthropic",
            "context_length": 200000,
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
            "supports_json": False,
            "supports_seed": False,
        },
        "claude-sonnet-5": {
            "provider": "anthropic",
            "context_length": 200000,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
            "supports_json": False,
            "supports_seed": False,
        },
        "claude-haiku-4-5-20251001": {
            "provider": "anthropic",
            "context_length": 200000,
            "input_cost_per_1k": 0.001,
            "output_cost_per_1k": 0.005,
            "supports_json": False,
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
