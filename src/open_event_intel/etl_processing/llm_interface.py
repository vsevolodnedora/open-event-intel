"""
LLM Interface for Energy Events Intelligence System.

This module provides a type-safe interface to LLM providers (OpenAI, local models)
with automatic cost tracking, caching, and budget management. It enforces validation
of all external inputs and maintains deterministic behavior for testability.

Invariants:
- All costs are in USD
- Token estimates use 4 chars/token heuristic
- Cache keys are deterministic SHA-256 hashes
- Budget checks happen before non-free API calls
- Entity masking uses length-descending, stable-ordered replacement
"""

import hashlib
import json
import math
import os
import re
import time
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, Type, TypeVar
from uuid import uuid4

from config_interface import LLMConfig, ModelDefinition, ModelProvider, PromptDefinition
from database_interface import DatabaseInterface, LLMCacheRow, LLMUsageLogRow, compute_sha256_id
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from open_event_intel.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

ENTITY_PLACEHOLDER_PREFIX = "__ENT_"
ENTITY_PLACEHOLDER_SUFFIX = "__"
_PLACEHOLDER_RE = re.compile(
    re.escape(ENTITY_PLACEHOLDER_PREFIX) + r"(\d+)" + re.escape(ENTITY_PLACEHOLDER_SUFFIX)
)


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMConnectionError(LLMError):
    """Connection to LLM endpoint failed."""


class LLMBudgetExceededError(LLMError):
    """LLM budget exceeded."""


class LLMExtractionError(LLMError):
    """Structured extraction failed."""


class LLMModelNotFoundError(LLMError):
    """Model not found in config."""


class LLMTranslationError(LLMError):
    """Translation operation failed."""


class MessageRole(str, Enum):
    """Valid roles for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single chat message with validated role and content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Message content cannot be empty or whitespace-only")
        return v

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI API format."""
        return {"role": self.role.value, "content": self.content}


class EndpointHealth(BaseModel):
    """Health status of an LLM endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    base_url: str
    ok: bool
    available_models: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class CompletionResult(BaseModel):
    """Result of a completion call."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    cached: bool
    latency_ms: int = Field(ge=0)


class EmbeddingResult(BaseModel):
    """Result of an embedding call."""

    model_config = ConfigDict(extra="forbid")

    embeddings: list[list[float]]
    model: str
    tokens_in: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    embedding_dim: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class EntityMaskEntry(BaseModel):
    """Single entity-to-placeholder mapping used during masked translation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    placeholder: str
    original: str


class EntityMaskMap(BaseModel):
    """
    Complete mask map for an entity-masking operation.

    Entries are ordered longest-original-first to guarantee greedy,
    non-overlapping replacement.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[EntityMaskEntry, ...] = Field(default_factory=tuple)

    @property
    def placeholder_to_original(self) -> dict[str, str]:
        """Return mapping from placeholder to original text."""
        return {e.placeholder: e.original for e in self.entries}


class TranslationResult(BaseModel):
    """Result of a translation call."""

    model_config = ConfigDict(extra="forbid")

    translated_text: str
    source_language: str
    target_language: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    cached: bool
    latency_ms: int = Field(ge=0)
    entity_masked: bool = False
    mask_map: Optional[EntityMaskMap] = None


def compute_cache_key(
    model: str, messages: list[ChatMessage], temperature: float, max_tokens: int
) -> str:
    """Compute deterministic cache key for a request."""
    data = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_cost(model_def: ModelDefinition, tokens_in: int, tokens_out: int) -> float:
    """Compute cost in USD for a request."""
    cost_in = (tokens_in / 1_000_000) * model_def.cost_per_million_tokens_input
    cost_out = (tokens_out / 1_000_000) * model_def.cost_per_million_tokens_output
    return cost_in + cost_out


def estimate_tokens(text: str) -> int:
    """Estimate token count using 4 chars/token heuristic."""
    return max(1, len(text) // 4)


def normalize_embedding(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        return [v / norm for v in vector]
    return vector


def mask_entities(text: str, entities: list[str]) -> tuple[str, EntityMaskMap]:
    """
    Replace entity mentions in *text* with deterministic placeholders.

    Entities are sorted longest-first so that ``"Amprion GmbH"`` is masked
    before ``"Amprion"``.  Only non-empty entities that actually appear in the
    text produce entries.

    :param text: Source text to mask.
    :param entities: Entity surface forms to replace.
    :return: ``(masked_text, mask_map)``
    """
    unique_entities = sorted(
        {e for e in entities if e and e in text},
        key=lambda e: (-len(e), e),
    )

    entries: list[EntityMaskEntry] = []
    masked = text
    for idx, entity in enumerate(unique_entities):
        placeholder = f"{ENTITY_PLACEHOLDER_PREFIX}{idx}{ENTITY_PLACEHOLDER_SUFFIX}"
        masked = masked.replace(entity, placeholder)
        entries.append(EntityMaskEntry(placeholder=placeholder, original=entity))

    return masked, EntityMaskMap(entries=tuple(entries))


def unmask_entities(text: str, mask_map: EntityMaskMap) -> str:
    """
    Restore original entities from placeholders produced by :func:`mask_entities`.

    :param text: Translated (or otherwise processed) text with placeholders.
    :param mask_map: Mapping produced by a prior :func:`mask_entities` call.
    :return: Text with placeholders replaced by originals.
    """
    result = text
    for entry in mask_map.entries:
        result = result.replace(entry.placeholder, entry.original)
    return result


def format_prompt_template(template: str, **kwargs: str) -> str:
    """
    Substitute ``{key}`` placeholders in a prompt template.

    :param template: Prompt string with ``{key}`` placeholders.
    :param kwargs: Substitution values.
    :return: Formatted prompt string.
    :raises LLMTranslationError: When a required placeholder is missing.
    """
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise LLMTranslationError(
            f"Missing prompt template variable: {exc}"
        ) from exc


class ClientManager:
    """Manages OpenAI client instances for different providers/endpoints."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._clients: dict[str, OpenAI] = {}

    def get_client(self, model_def: ModelDefinition) -> OpenAI:
        """Get or create OpenAI client for a model."""
        cache_key = f"{model_def.provider.value}:{model_def.base_url or 'default'}"

        if cache_key in self._clients:
            return self._clients[cache_key]

        timeout = self._config.defaults["timeout"]

        if model_def.provider == ModelProvider.OPENAI:
            client = self._create_openai_client(timeout)
        else:
            client = self._create_local_client(model_def, timeout)

        self._clients[cache_key] = client
        return client

    def _create_openai_client(self, timeout: float) -> OpenAI:
        """Create OpenAI API client."""
        api_key_env = self._config.env["openai_api_key"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise LLMError(f"API key not found in environment: {api_key_env}")
        return OpenAI(api_key=api_key, timeout=timeout)

    def _create_local_client(self, model_def: ModelDefinition, timeout: float) -> OpenAI:
        """Create client for local model endpoint."""
        if not model_def.base_url:
            raise LLMError(f"base_url required for local model: {model_def.name}")
        return OpenAI(base_url=model_def.base_url, api_key="not-used", timeout=timeout)


class CacheStoreInterface:
    def __init__(self, db: DatabaseInterface):
        """Initialize cache store."""
        self._db = db

    def get(self, cache_key: str) -> Optional[str]:
        """Get cached value from cache."""
        cached = self._db.get_llm_cache(cache_key)
        return cached.response if cached else None

    def set(self, cache_key: str, model: str, prompt_hash: str, response: str, ttl: timedelta) -> None:
        """Set cached value from cache."""
        self._db.set_llm_cache(cache_key, model, prompt_hash, response, ttl)


class BudgetTracker:
    """Tracks LLM spending against configured budget limits."""

    def __init__(self, config: LLMConfig, db: DatabaseInterface):
        """Initialize budget tracker."""
        self._config = config
        self._db = db
        self._run_id: Optional[str] = None

    def set_run_id(self, run_id: str) -> None:
        """Set the current pipeline run ID."""
        self._run_id = run_id

    def check_budget(self, estimated_additional_cost: float = 0.0) -> tuple[bool, float, float]:
        """Check if budget allows an operation."""
        spent = self._get_spent()
        limit = self._get_limit()
        within_budget = (spent + estimated_additional_cost) <= limit
        return within_budget, spent, limit

    def update_budget(self, cost: float) -> None:
        """Update budget with actual cost incurred."""
        if self._run_id and cost > 0:
            current_spent = self._get_spent()
            new_spent = current_spent + cost
            self._db.update_pipeline_run_counters(self._run_id, budget_spent=new_spent)

    def _get_spent(self) -> float:
        """Get amount spent in current run."""
        if not self._run_id:
            return 0.0
        run = self._db.get_pipeline_run(self._run_id)
        return run.budget_spent if run else 0.0

    def _get_limit(self) -> float:
        """Get budget limit for current run."""
        return self._config.budget_limit if hasattr(self._config, "budget_limit") else float("inf")


class LLMInterface:
    """LLM operations interface with cost tracking and caching."""

    def __init__(
        self,
        config: LLMConfig,
        db: DatabaseInterface,
        stage_name: str,
        run_id: Optional[str],
        cache_ttl_hours: int,
        allow_external_calls: bool = True,
    ) -> None:
        """Initialize LLM interface."""
        self._config = config
        self._db = db
        self._stage = stage_name
        self._client_manager = ClientManager(config)
        self._cache = CacheStoreInterface(db)
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._budget = BudgetTracker(config, db)
        self._allow_external_calls = allow_external_calls
        if run_id:
            self._budget.set_run_id(run_id)

    def check_health(self, model_name: str) -> EndpointHealth:
        """Check if a model endpoint is healthy."""
        model_def = self._config.get_model(model_name)
        if not model_def:
            return EndpointHealth(
                model_name=model_name,
                base_url="unknown",
                ok=False,
                error=f"Model not in config: {model_name}",
            )

        base_url = model_def.base_url or "https://api.openai.com/v1"
        start = time.perf_counter()

        try:
            client = self._client_manager.get_client(model_def)
            response = client.models.list()
            latency = int((time.perf_counter() - start) * 1000)

            model_ids = [
                m.id for m in getattr(response, "data", []) if getattr(m, "id", None)
            ]

            return EndpointHealth(
                model_name=model_name,
                base_url=base_url,
                ok=True,
                available_models=model_ids,
                latency_ms=latency,
            )
        except Exception as e:
            latency = int((time.perf_counter() - start) * 1000)
            return EndpointHealth(
                model_name=model_name,
                base_url=base_url,
                ok=False,
                error=str(e),
                latency_ms=latency,
            )

    def complete(
        self,
        messages: list[ChatMessage],
        purpose: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> CompletionResult:
        """
        Execute a chat completion with cost tracking.

        :param messages: Chat messages
        :param purpose: Purpose of this completion (for logging)
        :param model_name: Model to use (defaults to routing for purpose)
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        :param max_tokens: Max tokens to generate
        :param json_mode: Force JSON output
        :param use_cache: Whether to use caching
        :return: Completion result with metadata
        """
        if not model_name:
            routing = self._config.get_routing(purpose)
            if not routing:
                raise LLMError(f"No routing configured for purpose: {purpose}")
            model_name = routing.primary

        model_def = self._config.get_model(model_name)
        if not model_def:
            raise LLMModelNotFoundError(f"Model not found: {model_name}")
        if not self._allow_external_calls and model_def.provider != ModelProvider.LOCAL:
            raise LLMError("Call to external model is not allowed (set 'allow_external_calls' to True to change)")

        temp, tp, mt = self._resolve_params(temperature, top_p, max_tokens, purpose)
        cache_key = compute_cache_key(model_name, messages, temp, mt)

        if use_cache:
            cached_result = self._try_get_cached(cache_key, model_name, messages, purpose)
            if cached_result:
                return cached_result

        if model_def.cost_per_million_tokens_input > 0:
            estimated_cost = self._estimate_cost(model_def, messages, mt)
            within_budget, spent, limit = self._budget.check_budget(estimated_cost)
            if not within_budget:
                raise LLMBudgetExceededError(
                    f"Budget limit reached: ${spent:.4f} of ${limit:.4f} spent"
                )

        result = self._execute_completion(
            model_def, messages, temp, tp, mt, json_mode, purpose, cache_key, use_cache
        )

        self._budget.update_budget(result.cost)

        return result

    def embed(
        self,
        texts: list[str],
        model_def: ModelDefinition,
        normalize: bool = False,
        purpose: str = "embedding",
    ) -> EmbeddingResult:
        """
        Generate embeddings for texts.

        :param texts: Texts to embed
        :param model_name: Model to use (defaults to embedding routing)
        :param normalize: Whether to normalize vectors to unit length
        :param purpose: Purpose of embedding (for logging)
        :return: Embedding result with metadata
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        if not model_def:
            raise LLMModelNotFoundError("Model not found provided")
        if not self._allow_external_calls and model_def.provider != ModelProvider.LOCAL:
            raise LLMError("Call to external model is not allowed (set 'allow_external_calls' to True to change)")

        if not model_def.has_capability("embedding"):
            raise LLMError(f"Model {model_def.name} does not support embeddings")

        tokens_in = sum(estimate_tokens(t) for t in texts)
        estimated_cost = (tokens_in / 1_000_000) * model_def.cost_per_million_tokens_input

        if estimated_cost > 0:
            within_budget, spent, limit = self._budget.check_budget(estimated_cost)
            if not within_budget:
                raise LLMBudgetExceededError(
                    f"Budget limit reached: ${spent:.4f} of ${limit:.4f} spent"
                )

        client = self._client_manager.get_client(model_def)

        start = time.perf_counter()
        try:
            response = client.embeddings.create(model=model_def.model_id, input=texts)
        except OpenAIError as e:
            raise LLMConnectionError(f"API error for {model_def.name}: {e}") from e

        latency = int((time.perf_counter() - start) * 1000)

        embeddings = [d.embedding for d in response.data]
        if normalize:
            embeddings = [normalize_embedding(e) for e in embeddings]

        usage = response.usage
        tokens = usage.total_tokens if usage else tokens_in
        cost = (tokens / 1_000_000) * model_def.cost_per_million_tokens_input

        self._log_usage(purpose, model_def.name, tokens, 0, cost, False, None, latency)
        self._budget.update_budget(cost)

        return EmbeddingResult(
            embeddings=embeddings,
            model=model_def.name,
            tokens_in=tokens,
            cost=cost,
            embedding_dim=len(embeddings[0]) if embeddings else 0,
            latency_ms=latency,
        )

    def extract_structured(
        self,
        messages: list[ChatMessage],
        schema: Type[T],
        purpose: str,
        model_name: Optional[str] = None,
        max_retries: int = 2,
    ) -> T:
        """
        Extract structured data from LLM response.

        :param messages: Chat messages
        :param schema: Pydantic model class for validation
        :param purpose: Purpose of extraction (for logging)
        :param model_name: Model to use
        :param max_retries: Max retry attempts on validation failure
        :return: Validated instance of schema
        """
        for attempt in range(max_retries + 1):
            result = self.complete(
                messages=messages,
                purpose=purpose,
                model_name=model_name,
                json_mode=True,
                use_cache=False,
            )

            try:
                parsed = json.loads(result.content)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries:
                    raise LLMExtractionError(
                        f"Failed to extract {schema.__name__} after {max_retries + 1} attempts: {e}"
                    ) from e
                logger.warning(
                    f"Extraction attempt {attempt + 1} failed for {schema.__name__}: {e}"
                )

        raise LLMExtractionError("Unreachable")

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        purpose: str = "translation",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        entity_masking: bool = False,
        entities: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> TranslationResult:
        """
        Translate text between languages, optionally masking named entities.

        When *entity_masking* is ``True``, recognised *entities* are replaced
        with deterministic placeholders before translation and restored
        afterwards, preventing the model from transliterating proper nouns.

        :param text: Source text to translate.
        :param source_language: ISO-639 language code of the source text.
        :param target_language: ISO-639 language code for the output.
        :param purpose: Logging / routing purpose key (also selects prompt
            template from config — ``"translation"`` or
            ``"translation_masked"``).
        :param model_name: Override model (defaults to task routing).
        :param temperature: Sampling temperature override.
        :param max_tokens: Max output tokens override.
        :param entity_masking: If ``True``, apply entity masking/unmasking.
        :param entities: Entity surface forms to mask.  Required when
            *entity_masking* is ``True``.
        :param use_cache: Whether to use response caching.
        :return: :class:`TranslationResult` with translated text and metadata.
        :raises LLMTranslationError: On invalid arguments or failed unmasking.
        :raises LLMError: On routing/model resolution failures.
        """
        if not text or not text.strip():
            raise LLMTranslationError("Translation text cannot be empty")
        if source_language == target_language:
            raise LLMTranslationError(
                f"Source and target language are identical: {source_language}"
            )

        mask_map: Optional[EntityMaskMap] = None
        input_text = text

        if entity_masking:
            if not entities:
                raise LLMTranslationError(
                    "entities must be provided when entity_masking is True"
                )
            input_text, mask_map = mask_entities(text, entities)
            logger.info(
                "Masked %d entities for translation (%s -> %s)",
                len(mask_map.entries),
                source_language,
                target_language,
            )

        prompt_key = "translation_masked" if entity_masking else "translation"
        messages = self._build_translation_messages(
            input_text, source_language, target_language, prompt_key
        )

        if not model_name:
            model_name = self._resolve_translation_model(purpose)

        logger.info(
            "Translating %d chars (%s -> %s) with model=%s, entity_masking=%s",
            len(text),
            source_language,
            target_language,
            model_name,
            entity_masking,
        )

        completion = self.complete(
            messages=messages,
            purpose=purpose,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )

        translated = completion.content.strip()

        if entity_masking and mask_map:
            translated = unmask_entities(translated, mask_map)
            remaining = _PLACEHOLDER_RE.search(translated)
            if remaining:
                logger.warning(
                    "Unresolved entity placeholder after unmasking: %s",
                    remaining.group(),
                )

        logger.info(
            "Translation complete: %d->%d chars, cost=$%.6f, cached=%s",
            len(text),
            len(translated),
            completion.cost,
            completion.cached,
        )

        return TranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            model=completion.model,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost=completion.cost,
            cached=completion.cached,
            latency_ms=completion.latency_ms,
            entity_masked=entity_masking,
            mask_map=mask_map,
        )

    def _build_translation_messages(
        self,
        text: str,
        source_language: str,
        target_language: str,
        prompt_key: str,
    ) -> list[ChatMessage]:
        """
        Build chat messages for translation using config prompt templates.

        Falls back to a minimal built-in prompt when no config template exists
        for *prompt_key*.
        """
        prompt_def = self._config.get_prompt(prompt_key)
        if prompt_def:
            user_content = format_prompt_template(
                prompt_def.user,
                text=text,
                source_language=source_language,
                target_language=target_language,
            )
            system_content = None
            if prompt_def.system:
                system_content = format_prompt_template(
                    prompt_def.system,
                    source_language=source_language,
                    target_language=target_language,
                )
            return self._build_messages(system_content, user_content)

        logger.warning("No prompt template found for '%s', using fallback", prompt_key)
        system = (
            f"You are a professional translator. "
            f"Translate the following text from {source_language} to {target_language}. "
            f"Preserve formatting and do not add explanations."
        )
        return self._build_messages(system, text)

    def _resolve_translation_model(self, purpose: str) -> str:
        """
        Resolve model name for translation via task routing.

        Tries *purpose* first, then falls back to ``"translation"``, then
        scans models for the ``"translation"`` capability.
        """
        routing = self._config.get_routing(purpose)
        if routing:
            return routing.primary

        if purpose != "translation":
            routing = self._config.get_routing("translation")
            if routing:
                return routing.primary

        for name, model in self._config.models.items():
            if model.has_capability("translation"):
                return name

        raise LLMError(
            f"No translation model configured (tried routing for "
            f"'{purpose}' and 'translation', then capability scan)"
        )

    def _estimate_cost(
        self, model_def: ModelDefinition, messages: list[ChatMessage], max_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        tokens_in = sum(estimate_tokens(m.content) for m in messages)
        return compute_cost(model_def, tokens_in, max_tokens)

    def _resolve_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        purpose: str,
    ) -> tuple[float, float, int]:
        """Resolve completion parameters using config defaults."""
        temp = temperature if temperature is not None else self._config.defaults["temperature"]
        tp = top_p if top_p is not None else self._config.defaults["top_p"]

        if max_tokens is not None:
            mt = max_tokens
        else:
            mt = self._config.defaults["max_tokens"].get(
                purpose, self._config.defaults["max_tokens"]["general"]
            )

        return temp, tp, mt

    def _try_get_cached(
        self,
        cache_key: str,
        model_name: str,
        messages: list[ChatMessage],
        purpose: str,
    ) -> Optional[CompletionResult]:
        """Attempt to retrieve cached response."""
        cached_content = self._cache.get(cache_key)
        if not cached_content:
            return None

        tokens_in = sum(estimate_tokens(m.content) for m in messages)
        tokens_out = estimate_tokens(cached_content)

        self._log_usage(purpose, model_name, tokens_in, tokens_out, 0.0, True, cache_key, 0)

        return CompletionResult(
            content=cached_content,
            model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=0.0,
            cached=True,
            latency_ms=0,
        )

    def _execute_completion(
        self,
        model_def: ModelDefinition,
        messages: list[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool,
        purpose: str,
        cache_key: str,
        use_cache: bool,
    ) -> CompletionResult:
        """Execute the actual API call for completion."""
        client = self._client_manager.get_client(model_def)

        request_kwargs: dict[str, Any] = {
            "model": model_def.model_id,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_mode and model_def.provider == ModelProvider.OPENAI:
            request_kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**request_kwargs)
        except OpenAIError as e:
            raise LLMConnectionError(f"API error for {model_def.name}: {e}") from e

        latency = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content or ""

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        cost = compute_cost(model_def, tokens_in, tokens_out)

        if use_cache:
            messages_json = json.dumps([msg.to_dict() for msg in messages], sort_keys=True)
            prompt_hash = hashlib.sha256(messages_json.encode()).hexdigest()[:32]
            self._cache.set(cache_key, model_def.name, prompt_hash, content, self._cache_ttl)

        self._log_usage(
            purpose, model_def.name, tokens_in, tokens_out, cost, False, cache_key, latency
        )

        return CompletionResult(
            content=content,
            model=model_def.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cached=False,
            latency_ms=latency,
        )

    def _log_usage(
        self,
        purpose: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        cached: bool,
        cache_key: Optional[str],
        latency_ms: Optional[int],
    ) -> str:
        """
        Log LLM usage to database.

        :return: Log entry ID
        """
        log_id = str(uuid4())

        row = LLMUsageLogRow(
            log_id=log_id,
            run_id=self._budget._run_id,
            stage=self._stage,
            purpose=purpose,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cached=1 if cached else 0,
            cache_key=cache_key,
            latency_ms=latency_ms,
        )

        self._db.insert_llm_usage_log(row)
        return log_id

    def _get_embedding_model_name(self) -> Optional[str]:
        """Get configured embedding model name from routing or first capable model."""
        routing = self._config.get_routing("embedding")
        if routing:
            return routing.primary

        for name, model in self._config.models.items():
            if model.has_capability("embedding"):
                return name

        return None

    def _build_messages(
        self, system_prompt: Optional[str], user_content: str
    ) -> list[ChatMessage]:
        """Build message list with optional system prompt."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(ChatMessage(role=MessageRole.USER, content=user_content))
        return messages