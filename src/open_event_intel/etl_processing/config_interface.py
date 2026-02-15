"""
Configuration Interface for Energy Events Intelligence System.

This module provides Pydantic models for loading, validating, and accessing
the configuration defined in config.yaml. All models use strict validation
(forbid unknown keys) and provide sensible defaults where appropriate.

Key features:
- Strict type validation with Literal types for publisher names, mention patterns, and event types
- No default values for fields that should be read from config.yaml
- Full validation of configuration structure at load time

Usage:
    from config_interface import load_config, Config

    config = load_config("config/processing/config.yaml")

    # Access configuration sections
    publishers = config.publishers
    entities = config.entities
    extraction = config.extraction
    taxonomy = config.taxonomy
    alerts = config.alerts
    llm_config = config.llm_config
    pii_masking = config.pii_masking
    units = config.units
    global_settings = config.global_settings

"""
import hashlib
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Publisher names as defined in config.yaml SECTION 1: PUBLISHERS
PublisherName = Literal[
    "SMARD",
    "EEX",
    "ENTSOE",
    "ACER",
    "EC",
    "BNETZA",
    "TRANSNETBW",
    "TENNET",
    "FIFTY_HERTZ",
    "AMPRION",
    "ICIS",
    "AGORA",
    "ENERGY_WIRE",
]

# All valid publisher names as a tuple for validation
VALID_PUBLISHER_NAMES: tuple[str, ...] = (
    "SMARD",
    "EEX",
    "ENTSOE",
    "ACER",
    "EC",
    "BNETZA",
    "TRANSNETBW",
    "TENNET",
    "FIFTY_HERTZ",
    "AMPRION",
    "ICIS",
    "AGORA",
    "ENERGY_WIRE",
)

# Mention pattern names as defined in config.yaml SECTION 3: EXTRACTION PATTERNS
MentionPatternName = Literal[
    "DEADLINE",
    "EFFECTIVE_DATE",
    "LEGAL_REF",
    "QUANTITY",
    "PROJECT_SECTION",
    "CONTACT_INFO",
    "GEO_COUNTRY",
    "GEO_REGION",
    "BIDDING_ZONE",
    "SPEAKER_QUOTE",
    "EXTERNAL_URL",
]

# All valid mention pattern names as a tuple for validation
VALID_MENTION_PATTERN_NAMES: tuple[str, ...] = (
    "DEADLINE",
    "EFFECTIVE_DATE",
    "LEGAL_REF",
    "QUANTITY",
    "PROJECT_SECTION",
    "CONTACT_INFO",
    "GEO_COUNTRY",
    "GEO_REGION",
    "BIDDING_ZONE",
    "SPEAKER_QUOTE",
    "EXTERNAL_URL",
)

# Event type names as defined in config.yaml extraction.event_types
EventTypeName = Literal[
    "consultation_opened",
    "project_milestone",
    "regulatory_decision",
    "network_code_amendment",
    "price_movement",
    "generation_record",
]

# All valid event type names as a tuple for validation
VALID_EVENT_TYPE_NAMES: tuple[str, ...] = (
    "consultation_opened",
    "project_milestone",
    "regulatory_decision",
    "network_code_amendment",
    "price_movement",
    "generation_record",
)

# Processing tier types
ProcessingTier = Literal["data_heavy", "regulatory", "infrastructure", "narrative"]

# Boilerplate mode types
BoilerplateMode = Literal["aggressive", "moderate"]

# BASE CONFIGURATION CLASSES

class StrictModel(BaseModel):
    """Base model with strict validation - forbids unknown keys."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class FlexibleModel(BaseModel):
    """Model that allows extra fields for complex nested structures."""

    model_config = ConfigDict(extra="allow", validate_default=True)

# SECTION 1: PUBLISHERS

class MetadataAnchor(StrictModel):
    """Metadata anchor configuration for extracting title/date."""

    method: str
    pattern: Optional[str] = None
    format: Optional[str] = None


class UrlNormalization(StrictModel):
    """URL normalization rules for a publisher."""

    canonical_host: str
    strip_params: list[str] = Field(default_factory=list)
    preserve_params: list[str] = Field(default_factory=list)


class DateFormats(StrictModel):
    """Date format configuration for a publisher."""

    primary: str
    secondary: list[str] = Field(default_factory=list)
    locale: Optional[str] = None


class BoilerplatePatterns(StrictModel):
    """Boilerplate detection patterns."""

    footer: list[str] = Field(default_factory=list)
    navigation: list[str] = Field(default_factory=list)
    contact: list[str] = Field(default_factory=list)
    interactive: list[str] = Field(default_factory=list)


class TableExtraction(StrictModel):
    """Table extraction configuration."""

    method: Optional[str] = None
    header_heuristics: list[str] = Field(default_factory=list)
    numeric_column_threshold: Optional[float] = None


class Publisher(StrictModel):
    """Publisher profile configuration."""

    full_name: str
    processing_tier: ProcessingTier
    language_default: str
    language_variants: list[str] = Field(default_factory=list)
    boilerplate_mode: BoilerplateMode
    url_normalization: Optional[UrlNormalization] = None
    metadata_anchors: Optional[dict[str, list[MetadataAnchor]]] = None
    date_formats: Optional[DateFormats] = None
    boilerplate_patterns: Optional[BoilerplatePatterns] = None
    table_extraction: Optional[TableExtraction] = None

# SECTION 2: ENTITIES

class DisambiguationHints(FlexibleModel):
    """Hints for disambiguating entities."""

    country: Optional[str] = None
    control_area: Optional[str] = None
    region: Optional[str] = None
    jurisdiction: Optional[str] = None
    headquarters: Optional[str] = None
    note: Optional[str] = None
    type: Optional[str] = None
    parent: Optional[str] = None
    markets: Optional[list[str]] = None
    sector: Optional[str] = None
    product: Optional[str] = None
    countries: Optional[list[str]] = None


class ProjectNumbers(StrictModel):
    """Project number references."""

    bbplg: list[int] = Field(default_factory=list)


class Entity(FlexibleModel):
    """Entity definition in the registry."""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    abbreviations: list[str] = Field(default_factory=list)
    compound_forms: list[str] = Field(default_factory=list)
    disambiguation_hints: Optional[DisambiguationHints] = None
    identifier_patterns: list[str] = Field(default_factory=list)
    project_type: Optional[str] = None
    responsible_tsos: list[str] = Field(default_factory=list)
    geography: list[str] = Field(default_factory=list)
    project_numbers: Optional[ProjectNumbers] = None
    capacity_mw: Optional[int] = None
    length_km: Optional[int] = None
    route: Optional[str] = None
    voltage_kv: Optional[int] = None
    parent_entity_id: Optional[str] = None
    name_variants_de: list[str] = Field(default_factory=list)
    name_variants_en: list[str] = Field(default_factory=list)


class EntityTypeDefinition(StrictModel):
    """Definition for an entity type in the hierarchy."""

    parent: Optional[str] = None
    description: str
    subtypes: list[str] = Field(default_factory=list)


class DisambiguationAlternative(StrictModel):
    """Alternative entity for disambiguation."""

    entity: str
    keywords: list[str] = Field(default_factory=list)


class DisambiguationRule(StrictModel):
    """Rule for disambiguating entity mentions."""

    surface_patterns: list[str]
    default_entity: str
    context_keywords: list[str] = Field(default_factory=list)
    alternatives: list[DisambiguationAlternative] = Field(default_factory=list)

# SECTION 3: EXTRACTION PATTERNS

class PatternComponents(StrictModel):
    """
    Shared regex pattern components.

    All fields are required and must be provided from config.yaml.
    These patterns are foundational for extraction and cannot have defaults.
    """

    german_date_dmy: str
    german_date_numeric: str
    english_date_mdy: str
    english_date_dmy: str
    iso_date: str
    time_hhmm: str
    number_de: str
    number_en: str
    power_unit: str
    energy_unit: str
    bbplg_vorhaben: str
    section_identifier: str


class ConfidenceModifiers(StrictModel):
    """
    Confidence modifiers for mention extraction.

    All fields are optional since different mention types use different modifiers.
    """

    explicit_deadline_word: Optional[float] = None
    consultation_context: Optional[float] = None
    authority_nearby: Optional[float] = None
    legal_context: Optional[float] = None
    explicit_effective_word: Optional[float] = None


class MentionPattern(FlexibleModel):
    """
    Pattern configuration for mention extraction.

    The confidence_base field is optional - some patterns define confidence
    at the sub-pattern level instead of at the top level.
    """

    description: str
    markers: Optional[dict[str, list[str]]] = None
    date_proximity_chars: Optional[int] = None
    confidence_base: Optional[float] = None  # Optional - some patterns use sub-pattern confidence
    confidence_modifiers: Optional[ConfidenceModifiers] = None
    patterns: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None
    known_german_laws: Optional[list[dict[str, str]]] = None
    magnitude_words: Optional[dict[str, dict[str, float]]] = None
    unit_categories: Optional[dict[str, Any]] = None
    requires_project_context: bool = False
    project_context_keywords: list[str] = Field(default_factory=list)
    block_markers: Optional[dict[str, list[str]]] = None
    extraction_fields: Optional[dict[str, Any]] = None
    block_boundary_chars: Optional[int] = None
    gazetteer: Optional[list[dict[str, Any]]] = None
    context_keywords: list[str] = Field(default_factory=list)
    quote_patterns: Optional[list[dict[str, Any]]] = None
    max_quote_length_chars: Optional[int] = None
    exclude_domains: list[str] = Field(default_factory=list)
    highlight_domains: list[str] = Field(default_factory=list)


class EventSlot(FlexibleModel):
    """Slot definition for event extraction."""

    name: str
    description: Optional[str] = None
    selector: Optional[dict[str, Any]] = None
    extraction_method: Optional[str] = None
    proximity_chars: Optional[int] = None
    required_confidence: Optional[float] = None
    patterns: Optional[list[str]] = None
    enum_values: Optional[list[str]] = None
    unit_filter: Optional[list[str]] = None
    domain_hints: Optional[list[str]] = None
    fallback: Optional[str] = None


class EventType(FlexibleModel):
    """
    Event type definition for extraction.

    Required fields (priority, confidence_threshold) must come from config.yaml.
    """

    description: str
    priority: int  # Required - no default
    required_slots: list[EventSlot] = Field(default_factory=list)
    optional_slots: list[EventSlot] = Field(default_factory=list)
    trigger_keywords: Optional[dict[str, Any]] = None
    canonical_key_template: Optional[str] = None
    confidence_threshold: float  # Required - no default


class ExtractionTier(StrictModel):
    """Extraction tier configuration."""

    description: str
    methods: list[str]
    min_confidence: float
    llm_validation: Union[bool, str] = False
    budget_constrained: bool = False


class ExtractionPipeline(StrictModel):
    """Pipeline configuration for extraction."""

    mention_extraction_order: list[str] = Field(default_factory=list)
    event_extraction_tiers: dict[str, ExtractionTier] = Field(default_factory=dict)
    context_windows: dict[str, int] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)


class CompoundWordSuffix(StrictModel):
    """Compound word suffix pattern."""

    suffix: str
    type: str


class CompoundWordConfig(StrictModel):
    """Configuration for German compound word handling."""

    enabled: bool
    min_component_length: int
    max_components: int
    suffix_patterns: list[CompoundWordSuffix] = Field(default_factory=list)
    decomposition_dictionary: dict[str, list[str]] = Field(default_factory=dict)


class Extraction(StrictModel):
    """
    Complete extraction configuration.

    Uses typed dictionaries with MentionPatternName and EventTypeName
    to enforce that only valid pattern/event names are used.
    """

    pattern_components: PatternComponents
    mention_patterns: dict[MentionPatternName, MentionPattern]
    event_types: dict[EventTypeName, EventType]
    pipeline: ExtractionPipeline = Field(default_factory=ExtractionPipeline)
    compound_word_config: Optional[CompoundWordConfig] = None

    @field_validator("mention_patterns", mode="before")
    @classmethod
    def validate_mention_pattern_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that all mention pattern names are in the allowed set."""
        invalid_names = set(v.keys()) - set(VALID_MENTION_PATTERN_NAMES)
        if invalid_names:
            raise ValueError(
                f"Invalid mention pattern names: {invalid_names}. "
                f"Valid names are: {VALID_MENTION_PATTERN_NAMES}"
            )
        return v

    @field_validator("event_types", mode="before")
    @classmethod
    def validate_event_type_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that all event type names are in the allowed set."""
        invalid_names = set(v.keys()) - set(VALID_EVENT_TYPE_NAMES)
        if invalid_names:
            raise ValueError(
                f"Invalid event type names: {invalid_names}. "
                f"Valid names are: {VALID_EVENT_TYPE_NAMES}"
            )
        return v

# SECTION 4: TAXONOMY

class TopicKeywords(StrictModel):
    """Keywords for topic classification."""

    high_signal: list[str] = Field(default_factory=list)
    medium_signal: list[str] = Field(default_factory=list)


class Topic(FlexibleModel):
    """Topic definition for document classification."""

    description: str
    keywords: dict[str, TopicKeywords] = Field(default_factory=dict)
    mention_signals: dict[str, float] = Field(default_factory=dict)
    structural_signals: dict[str, float] = Field(default_factory=dict)
    threshold: float
    min_confidence: Optional[float] = None


class DocumentType(FlexibleModel):
    """Document type definition."""

    description: str
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    mention_signals: dict[str, float] = Field(default_factory=dict)
    structural_signals: dict[str, float] = Field(default_factory=dict)
    threshold: float


class MultiLabelSettings(StrictModel):
    """Multi-label classification settings."""

    enabled: bool
    max_topics: int
    min_secondary_confidence: float
    common_combinations: list[list[str]] = Field(default_factory=list)


class KeywordWeights(StrictModel):
    """Weights for keyword scoring."""

    high_signal: float
    medium_signal: float
    compound_match_bonus: float


class ConfidenceSettings(StrictModel):
    """Confidence calculation settings."""

    base: float
    keyword_cap: float
    mention_cap: float
    structural_cap: float
    prior_cap: float


class ClassificationThresholds(StrictModel):
    """Classification threshold settings."""

    topic_assignment: float
    document_type_assignment: float
    confident_assignment: float
    low_confidence_flag: float


class AmbiguitySettings(StrictModel):
    """Settings for handling ambiguous classifications."""

    confidence_gap_threshold: float
    max_before_unfocused_flag: int


class ClassificationSettings(StrictModel):
    """Complete classification settings."""

    multi_label: MultiLabelSettings
    keyword_weights: KeywordWeights
    confidence: ConfidenceSettings
    thresholds: ClassificationThresholds
    ambiguity: AmbiguitySettings


class UrgencyLevel(FlexibleModel):
    """Urgency level definition."""

    id: str
    description: str
    signals: list[dict[str, Any]] = Field(default_factory=list)
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    priority: int


class UrgencySettings(StrictModel):
    """Urgency classification settings."""

    levels: list[UrgencyLevel] = Field(default_factory=list)
    default_level: str
    deadline_urgency_mapping: dict[str, str] = Field(default_factory=dict)


class Taxonomy(StrictModel):
    """Complete taxonomy configuration."""

    topics: dict[str, Topic] = Field(default_factory=dict)
    document_types: dict[str, DocumentType] = Field(default_factory=dict)
    classification_settings: Optional[ClassificationSettings] = None
    publisher_priors: dict[str, dict[str, float]] = Field(default_factory=dict)
    urgency: Optional[UrgencySettings] = None

# SECTION 5: ALERTS

class AlertChannel(StrictModel):
    """Alert channel configuration."""

    type: str
    schedule: Optional[str] = None
    include_urgency: list[str] = Field(default_factory=list)
    cooldown_minutes: Optional[int] = None


class AlertCondition(FlexibleModel):
    """Condition for triggering an alert."""

    field: str
    operator: str
    value: Optional[Any] = None
    values: Optional[list[Any]] = None


class AlertTrigger(StrictModel):
    """Alert trigger configuration."""

    event_types: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    conditions: list[AlertCondition] = Field(default_factory=list)
    condition_logic: str = "AND"


class AlertTemplate(StrictModel):
    """Template for alert messages."""

    title: str
    body: str


class AlertRule(StrictModel):
    """Alert rule definition."""

    id: str
    name: str
    name_de: Optional[str] = None
    enabled: bool = True
    urgency: str = "normal"
    triggers: AlertTrigger = Field(default_factory=AlertTrigger)
    channels: list[str] = Field(default_factory=list)
    suppression_window_hours: int = 24
    template: Optional[AlertTemplate] = None


class Watchlist(StrictModel):
    """Watchlist definition for entity monitoring."""

    name: str
    name_de: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    entities: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    urgency_override: Optional[str] = None
    channels: list[str] = Field(default_factory=list)


class DeduplicationSettings(StrictModel):
    """Alert deduplication settings."""

    enabled: bool
    time_window_hours: int
    similarity_fields: list[str] = Field(default_factory=list)
    similarity_threshold: float
    strategy: str


class AlertSettings(StrictModel):
    """Global alert settings."""

    enabled: bool
    default_channel: str
    timezone: str
    channels: dict[str, AlertChannel] = Field(default_factory=dict)


class Alerts(StrictModel):
    """Complete alerts configuration."""

    settings: Optional[AlertSettings] = None
    rules: list[AlertRule] = Field(default_factory=list)
    watchlists: dict[str, Watchlist] = Field(default_factory=dict)
    deduplication: Optional[DeduplicationSettings] = None

# SECTION 6: UNITS

class NumberFormat(StrictModel):
    """Number format configuration."""

    thousands_separator: str
    decimal_separator: str


class UnitVariant(StrictModel):
    """Unit variant with conversion multiplier."""

    multiplier: float
    patterns: list[str]


class UnitValidation(StrictModel):
    """Validation ranges for unit values."""

    min_reasonable: float
    max_reasonable: float


class UnitCategory(StrictModel):
    """Unit category with variants and validation."""

    canonical_unit: str
    variants: dict[str, UnitVariant] = Field(default_factory=dict)
    validation: Optional[UnitValidation] = None


class SanityCheck(StrictModel):
    """Sanity check bounds."""

    extreme_low: Optional[float] = None
    typical_low: Optional[float] = None
    typical_high: Optional[float] = None
    extreme_high: Optional[float] = None
    single_plant_max: Optional[float] = None
    country_total_max: Optional[float] = None
    annual_country_max: Optional[float] = None
    single_project_max: Optional[float] = None


class Units(StrictModel):
    """Complete units configuration."""

    number_formats: dict[str, NumberFormat] = Field(default_factory=dict)
    energy: Optional[UnitCategory] = None
    power: Optional[UnitCategory] = None
    currency: Optional[UnitCategory] = None
    electricity_price: Optional[UnitCategory] = None
    carbon_price: Optional[UnitCategory] = None
    length: Optional[UnitCategory] = None
    percentage: Optional[UnitCategory] = None
    gas_volume: Optional[UnitCategory] = None
    frequency: Optional[UnitCategory] = None
    sanity_checks: dict[str, SanityCheck] = Field(default_factory=dict)

# SECTION 7: LLM BUDGET

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

    model_config = ConfigDict(extra="forbid")
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

# SECTION 8: PII MASKING

class MaskFormat(StrictModel):
    """PII mask format templates."""

    email: str
    phone: str
    person: str
    address: str


class PIISettings(StrictModel):
    """PII masking settings."""

    enabled: bool
    mode: str
    mask_format: MaskFormat


class PIIPatternDef(StrictModel):
    """Individual PII pattern definition."""

    pattern: str
    confidence: float
    label: Optional[str] = None


class PIIPatternConfig(StrictModel):
    """Configuration for a PII pattern type."""

    enabled: bool
    patterns: list[PIIPatternDef] = Field(default_factory=list)
    whitelist: Optional[dict[str, list[str]]] = None
    context_indicators: list[str] = Field(default_factory=list)
    context_required: bool = False
    context_patterns: list[str] = Field(default_factory=list)


class PublicOfficial(StrictModel):
    """Public official whitelist entry."""

    name: str
    role: str


class PIIWhitelist(StrictModel):
    """PII whitelist configuration."""

    public_officials: list[PublicOfficial] = Field(default_factory=list)
    public_title_prefixes: list[str] = Field(default_factory=list)


class ContextExclusionSection(StrictModel):
    """Context exclusion section definition."""

    name: str
    indicators: list[str]
    scope_lines: int


class ContextExclusions(StrictModel):
    """Context exclusion settings."""

    sections: list[ContextExclusionSection] = Field(default_factory=list)
    registered_entities: bool


class MaskingSettings(StrictModel):
    """Masking behavior settings."""

    consistent_placeholders: bool
    counter_scope: str
    include_mapping: bool


class PIIMasking(StrictModel):
    """Complete PII masking configuration."""

    settings: Optional[PIISettings] = None
    patterns: dict[str, PIIPatternConfig] = Field(default_factory=dict)
    whitelist: Optional[PIIWhitelist] = None
    context_exclusions: Optional[ContextExclusions] = None
    masking: Optional[MaskingSettings] = None

# SECTION 9: GLOBAL SETTINGS

class ChunkingLanguageHint(StrictModel):
    """Language-specific chunking hints."""

    compound_word_split: bool
    min_word_length: int


class ChunkingSettings(StrictModel):
    """Chunking configuration."""

    target_tokens: int # 512
    overlap_tokens: int # 64
    min_tokens: int # 128
    max_tokens: int # 1024
    respect_block_boundaries: bool # true
    table_handling: str # dedicated_chunk_type
    sentence_boundary_chars: list[str] # [".", "!", "?", "\n\n"]
    language_hints: dict[str, ChunkingLanguageHint] = Field(default_factory=dict)
    #       de: {compound_word_split: false, min_word_length: 2}
    #       en: {compound_word_split: false, min_word_length: 2}


class LanguageDetection(StrictModel):
    """Language detection settings."""

    confidence_threshold: float
    fallback_to_publisher_default: bool
    indicators: dict[str, list[str]] = Field(default_factory=dict)


class QualityThresholdsGlobal(StrictModel):
    """Global quality thresholds."""

    min_content_ratio: float
    skip_below_quality_score: float
    min_meaningful_text_length: int
    max_link_density: float
    max_navigation_ratio: float


class DateValidation(StrictModel):
    """Date validation rules."""

    min_year: int
    max_year: int
    reject_future_effective_dates_beyond_years: int
    reject_past_deadlines_beyond_days: int


class QuantityValidationRange(StrictModel):
    """Quantity validation range."""

    min: float
    max: float


class CrossReferenceValidation(StrictModel):
    """Cross-reference validation settings."""

    verify_entity_links: bool
    verify_project_sections: bool


class ValidationSettings(StrictModel):
    """Validation settings."""

    date_validation: DateValidation
    quantity_validation: dict[str, QuantityValidationRange] = Field(default_factory=dict)
    cross_reference: CrossReferenceValidation


class GlobalSettings(StrictModel):
    """Global configuration settings."""

    chunking: ChunkingSettings
    language_detection: LanguageDetection
    quality_thresholds: QualityThresholdsGlobal
    validation: ValidationSettings
    abbreviations: dict[str, dict[str, str]] = Field(default_factory=dict)

# ROOT CONFIGURATION

class Config(BaseModel):
    """
    Root configuration for the Energy Events Intelligence System.

    This model represents the complete configuration loaded from config.yaml.
    All sections use strict validation to ensure configuration integrity.

    Publisher names are validated against the VALID_PUBLISHER_NAMES set.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    publishers: dict[PublisherName, Publisher]
    entities: list[Entity] = Field(default_factory=list)
    entity_type_hierarchy: dict[str, EntityTypeDefinition] = Field(default_factory=dict)
    disambiguation_rules: list[DisambiguationRule] = Field(default_factory=list)
    extraction: Extraction
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)
    alerts: Alerts = Field(default_factory=Alerts)
    units: Units = Field(default_factory=Units)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    pii_masking: PIIMasking = Field(default_factory=PIIMasking)
    global_settings: GlobalSettings = Field(validation_alias="global")

    @field_validator("publishers", mode="before")
    @classmethod
    def validate_publisher_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that all publisher names are in the allowed set."""
        invalid_names = set(v) - set(VALID_PUBLISHER_NAMES)
        if invalid_names:
            raise ValueError(
                f"Invalid publisher names: {invalid_names}. "
                f"Valid names are: {VALID_PUBLISHER_NAMES}"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def handle_global_alias(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Handle 'global' key which is a Python reserved word."""
        if isinstance(data, dict):
            # Convert 'global' to 'global_settings' before validation
            if "global" in data:
                data["global_settings"] = data.pop("global")
        return data

    def get_publisher(self, publisher_id: str) -> Optional[Publisher]:
        """Get a publisher by ID (case-insensitive)."""
        # Try exact match first
        if publisher_id in self.publishers:
            return self.publishers[publisher_id]  # type: ignore
        # Try case-insensitive match
        for key, publisher in self.publishers.items():
            if key.upper() == publisher_id.upper():
                return publisher
        return None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        """Get a topic by ID."""
        return self.taxonomy.topics.get(topic_id)

    def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        for rule in self.alerts.rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_event_type(self, event_type: str) -> Optional[EventType]:
        """Get an event type definition."""
        return self.extraction.event_types.get(event_type)  # type: ignore

    def get_mention_pattern(self, pattern_name: str) -> Optional[MentionPattern]:
        """Get a mention pattern definition."""
        return self.extraction.mention_patterns.get(pattern_name)  # type: ignore

def load_config(config_path: Union[str, Path]) -> Config:
    """Load and validate configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return Config.model_validate(raw_config)

def get_config_version(config: Config) -> str:
    """Generate a hash-based version string for the configuration."""
    # Serialize config to JSON for hashing
    config_json = config.model_dump_json(exclude_none=True)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]

if __name__ == "__main__":
    config = load_config(Path("../../../config/config.yaml"))
    logging.info(get_config_version(config))