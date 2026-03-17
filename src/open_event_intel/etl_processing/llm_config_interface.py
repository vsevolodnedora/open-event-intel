from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelProvider(StrEnum):
    """LLM provider types."""

    LOCAL = "local"
    OPENAI = "openai"


class ModelDefinition(BaseModel):
    """Definition of a single LLM model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model_id: str
    source: str
    provider: ModelProvider
    base_url: Optional[str] = None  # None for OpenAI (uses default), required for local
    capabilities: list[str]
    max_context_length: int = Field(gt=0) # 1024 (default for embedding local model)
    cost_per_million_tokens_input: float = Field(ge=0)
    cost_per_million_tokens_output: float = Field(ge=0)
    embedding_dim: Optional[int] = None # 1024 (default for the local model)
    license: str

    @field_validator("capabilities", mode="before")
    @classmethod
    def normalize_capabilities(cls, v: list[str]) -> list[str]:
        """Normalize capabilities."""
        return [c.lower() for c in v]

    def has_capability(self, cap: str) -> bool:
        """Check capability."""
        return cap.lower() in self.capabilities

    def is_local(self) -> bool:
        """Check if model is local."""
        return self.provider == ModelProvider.LOCAL

    def is_free(self) -> bool:
        """Check if model is free."""
        return self.cost_per_million_tokens_input == 0 and self.cost_per_million_tokens_output == 0


class TaskRouting(BaseModel):
    """Task routing with primary model and fallbacks."""

    model_config = ConfigDict(extra="forbid")  # noqa: F821
    primary: str
    fallback: list[str]


class PromptDefinition(BaseModel):
    """System and user prompt templates."""

    model_config = ConfigDict(extra="forbid")
    system: Optional[str] = None
    user: str


class LLMConfig(BaseModel):
    """
    Complete LLM configuration.

    All fields with actual values must come from config.yaml.
    Only structural defaults (empty dicts/lists) are provided.
    """

    model_config = ConfigDict(extra="forbid")

    # Core configuration - these should come from config.yaml
    version: Optional[str] = None
    defaults: dict[str, Any] = Field(default_factory=dict, description="Default LLM parameters: temperature, top_p, timeout, max_retries, max_tokens")
    budget: dict[str, float] = Field(default_factory=dict, description="Budget constraints: incremental_run_budget_usd, backfill_run_budget_usd")
    budget_warning_percentage: Optional[int] = Field(default=None, ge=0, le=100, description="Percentage of budget at which to trigger warnings")
    local_models_only_if_budget_exceeded: Optional[bool] = Field(default=None, description="Whether to restrict to local models when budget is exceeded")

    # Model definitions and routing
    models: dict[str, ModelDefinition] = Field(default_factory=dict, description="Available LLM model definitions")
    task_routing: dict[str, TaskRouting] = Field(default_factory=dict, description="Task-to-model routing configuration")

    # Additional configuration
    embedding: dict[str, Any] = Field(default_factory=dict, description="Embedding configuration: batch_size_local, batch_size_remote, normalize")
    prompts: dict[str, PromptDefinition] = Field(default_factory=dict, description="Prompt templates for various tasks")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variable names: openai_api_key, etc.")

    def get_model(self, name: str) -> Optional[ModelDefinition]:
        """Get a model definition by name."""
        return self.models.get(name)

    def get_routing(self, task: str) -> Optional[TaskRouting]:
        """Get routing configuration for a task."""
        return self.task_routing.get(task)

    def get_prompt(self, task: str) -> Optional[PromptDefinition]:
        """Get prompt template for a task."""
        return self.prompts.get(task)

    def get_models_for_task(self, task: str) -> list[ModelDefinition]:
        """Get all models (primary + fallbacks) configured for a task."""
        routing = self.get_routing(task)
        if not routing:
            return []
        models = []
        if routing.primary in self.models:
            models.append(self.models[routing.primary])
        for fb in routing.fallback:
            if fb in self.models:
                models.append(self.models[fb])
        return models

    def get_local_models(self) -> dict[str, ModelDefinition]:
        """Get all local model definitions."""
        return {k: v for k, v in self.models.items() if v.is_local()}

    def get_remote_models(self) -> dict[str, ModelDefinition]:
        """Get all remote model definitions."""
        return {k: v for k, v in self.models.items() if not v.is_local()}

    def get_budget_limit(self, mode: str) -> float:
        """
        Get budget limit for a given mode.

        Args:
            mode: Either "incremental" or "backfill"

        Returns:
            Budget limit in USD, or 0.0 if not configured
        """
        if mode == "incremental":
            return self.budget.get("incremental_run_budget_usd", 0.0)
        return self.budget.get("backfill_run_budget_usd", 0.0)