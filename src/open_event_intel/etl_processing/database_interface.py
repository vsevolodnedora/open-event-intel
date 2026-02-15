"""
Database interface for the document processing pipeline.

Provides Pydantic models for all tables and a DatabaseInterface base class
for stage-specific adapters with stage isolation, transaction management,
and deterministic ID computation.
"""
import hashlib
import json
import sqlite3
from abc import ABC
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Iterator, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from open_event_intel.logger import get_logger
from open_event_intel.publications_database import decompress_publication_text

SCHEMA_PATH = Path(__file__).resolve().parent / "database_schema.sql"

logger = get_logger(__name__)

def _validate_sha256_hex(value: str) -> str:
    """Validate that a string is a valid SHA256 hex digest."""
    if len(value) != 64:
        raise ValueError(f"SHA256 hex must be 64 characters, got {len(value)}")
    if value != value.lower():
        raise ValueError("SHA256 hex must be lowercase")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError("SHA256 hex must contain only hex characters")  # noqa: B904
    return value

# ==== DATA MODELS ====

class _BaseRowModel(BaseModel):
    """Base model for database rows with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        extra="forbid",
    )


class SourcePublicationRow(_BaseRowModel):
    """Row from any publisher table in the source DB (Stage 01)."""

    id: str
    published_on: datetime
    added_on: datetime
    url: str
    title: str
    content: str
    language: str | None = None


class PipelineRunRow(_BaseRowModel):
    """Row from pipeline_run table."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    config_version: str
    budget_spent: float = 0.0
    doc_count_processed: int = 0
    doc_count_skipped: int = 0
    doc_count_failed: int = 0
    status: Literal["running", "completed", "failed", "aborted"] = "running"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        return _validate_sha256_hex(v)


class ScrapeRecordRow(_BaseRowModel):
    """Row from scrape_record table."""

    scrape_id: str
    publisher_id: str
    source_id: str
    url_raw: str
    url_normalized: str
    scraped_at: datetime
    source_published_at: datetime
    source_title: str
    source_language: str | None = None
    raw_content: bytes
    raw_encoding_detected: str | None = None
    scrape_kind: Literal["page"] = "page"
    ingest_run_id: str
    processing_status: str = "pending"
    created_at: datetime | None = None

    @field_validator("scrape_id")
    @classmethod
    def validate_scrape_id(cls, v: str) -> str:
        """Validate that a scrape_id is a valid ScrapeRecord ID."""
        return _validate_sha256_hex(v)


class DocumentRow(_BaseRowModel):
    """Row from document table."""

    document_id: str
    publisher_id: str
    url_normalized: str
    source_published_at: datetime
    url_raw_first_seen: str
    document_class: str | None = None
    is_attachment: int = 0
    created_at: datetime | None = None

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate that a document_id is a valid Document ID."""
        return _validate_sha256_hex(v)


class DocumentVersionRow(_BaseRowModel):
    """Row from document_version table."""

    doc_version_id: str
    document_id: str
    scrape_id: str
    content_hash_raw: str
    encoding_repairs_applied: list | None = None
    cleaning_spec_version: str
    normalization_spec: dict
    pii_masking_enabled: int = 0
    scrape_kind: Literal["page"] = "page"
    pii_mask_log: dict | None = None
    content_hash_clean: str
    clean_content: str
    span_indexing: str = "unicode_codepoint"
    content_length_raw: int | None = None
    content_length_clean: int | None = None
    boilerplate_ratio: float | None = None
    content_quality_score: float | None = None
    primary_language: str | None = None
    secondary_languages: list | None = None
    language_detection_confidence: float | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("doc_version_id", "document_id", "scrape_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate that a doc_version_id is a valid Document ID."""
        return _validate_sha256_hex(v)

    @field_validator("normalization_spec", mode="before")
    @classmethod
    def parse_normalization_spec(cls, v: Any) -> dict:
        """Parse the normalization spec."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("encoding_repairs_applied", "secondary_languages", mode="before")
    @classmethod
    def parse_json_arrays(cls, v: Any) -> list | None:
        """Parse JSON arrays into a list."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("pii_mask_log", mode="before")
    @classmethod
    def parse_pii_mask_log(cls, v: Any) -> dict | None:
        """Parse the PII mask log."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class EvidenceSpanRow(_BaseRowModel):
    """Row from evidence_span table."""

    evidence_id: str
    doc_version_id: str
    span_start: int
    span_end: int
    text: str
    purpose: str | None = None
    created_at: datetime | None = None
    created_in_run_id: str

    @field_validator("evidence_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)

    @model_validator(mode="after")
    def validate_span_bounds(self) -> "EvidenceSpanRow":
        """Validate that a span bounds is valid."""
        if self.span_start < 0:
            raise ValueError("span_start must be >= 0")
        if self.span_end <= self.span_start:
            raise ValueError("span_end must be > span_start")
        return self


class DocStageStatusRow(_BaseRowModel):
    """Row from doc_stage_status table."""

    doc_version_id: str
    stage: str
    run_id: str
    attempt: int = 1
    status: Literal["pending", "ok", "failed", "blocked", "skipped"]
    processed_at: datetime | None = None
    config_hash: str
    error_message: str | None = None
    details: str | None = None


class RunStageStatusRow(_BaseRowModel):
    """Row from run_stage_status table."""

    run_id: str
    stage: str
    attempt: int = 1
    status: Literal["pending", "ok", "failed"]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    config_hash: str
    error_message: str | None = None
    details: str | None = None


class LLMUsageLogRow(_BaseRowModel):
    """Row from llm_usage_log table."""

    log_id: str
    run_id: str | None = None
    stage: str
    purpose: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    cached: int = 0
    cache_key: str | None = None
    latency_ms: int | None = None
    created_at: datetime | None = None


class LLMCacheRow(_BaseRowModel):
    """Row from llm_cache table."""

    cache_key: str
    model: str
    prompt_hash: str
    response: str
    expires_at: datetime
    hit_count: int = 0
    created_at: datetime | None = None


class ValidationFailureRow(_BaseRowModel):
    """Row from validation_failure table."""

    failure_id: str
    run_id: str | None = None
    stage: str
    doc_version_id: str | None = None
    check_name: str
    details: str | None = None
    severity: str | None = None
    auto_repaired: int = 0
    created_at: datetime | None = None


class BlockRow(_BaseRowModel):
    """Row from block table."""

    block_id: str
    doc_version_id: str
    block_type: str
    block_level: int | None = None
    span_start: int
    span_end: int
    parse_confidence: float | None = None
    boilerplate_flag: str | None = None
    boilerplate_reason: str | None = None
    parent_block_id: str | None = None
    language_hint: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("block_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)


class ChunkRow(_BaseRowModel):
    """Row from chunk table."""

    chunk_rowid: int | None = None
    chunk_id: str
    doc_version_id: str
    span_start: int
    span_end: int
    evidence_id: str
    chunk_type: str
    block_ids: list[str]
    chunk_text: str
    heading_context: str | None = None
    retrieval_exclude: int = 0
    mention_boundary_safe: int = 1
    token_count_approx: int | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("chunk_id", "doc_version_id", "evidence_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)

    @field_validator("block_ids", mode="before")
    @classmethod
    def parse_block_ids(cls, v: Any) -> list[str]:
        """Parse block_ids."""
        if isinstance(v, str):
            return json.loads(v)
        return v


class TableExtractRow(_BaseRowModel):
    """Row from table_extract table."""

    table_id: str
    block_id: str
    doc_version_id: str
    row_count: int | None = None
    col_count: int | None = None
    headers_json: list | None = None
    header_row_index: int | None = None
    parse_quality: float | None = None
    parse_method: str | None = None
    table_class: str | None = None
    period_granularity: str | None = None
    units_detected: list | None = None
    raw_table_text: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("table_id", "block_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)

    @field_validator("headers_json", "units_detected", mode="before")
    @classmethod
    def parse_json_arrays(cls, v: Any) -> list | None:
        """Parse json arrays."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class DocMetadataRow(_BaseRowModel):
    """Row from doc_metadata table."""

    doc_version_id: str
    title: str | None = None
    title_span_start: int | None = None
    title_span_end: int | None = None
    title_source: str
    title_confidence: float | None = None
    title_evidence_id: str | None = None
    published_at: datetime | None = None
    published_at_raw: str | None = None
    published_at_format: str | None = None
    published_at_span_start: int | None = None
    published_at_span_end: int | None = None
    published_at_source: str
    published_at_confidence: float | None = None
    published_at_evidence_id: str | None = None
    detected_document_class: str | None = None
    document_class_confidence: float | None = None
    metadata_extraction_log: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None


class EntityRegistryRow(_BaseRowModel):
    """Row from entity_registry table."""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list | None = None
    name_variants_de: str | None = None
    name_variants_en: str | None = None
    abbreviations: str | None = None
    compound_forms: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    source_authority: str | None = None
    disambiguation_hints: dict | None = None
    parent_entity_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("aliases", mode="before")
    @classmethod
    def parse_aliases(cls, v: Any) -> list | None:
        """Parse aliases."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("disambiguation_hints", mode="before")
    @classmethod
    def parse_disambiguation_hints(cls, v: Any) -> dict | None:
        """Parse disambiguation hints."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class EntityRegistryAuditRow(_BaseRowModel):
    """Row from entity_registry_audit table."""

    audit_id: str
    entity_id: str
    change_type: str
    old_value_json: dict | None = None
    new_value_json: dict
    changed_at: datetime | None = None
    changed_by: str | None = None
    run_id: str | None = None
    reason: str | None = None

    @field_validator("old_value_json", "new_value_json", mode="before")
    @classmethod
    def parse_json(cls, v: Any) -> dict | None:
        """Parse json."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class MentionRow(_BaseRowModel):
    """Row from mention table."""

    mention_id: str
    doc_version_id: str
    chunk_ids: list | None = None
    mention_type: str
    surface_form: str
    normalized_value: str | None = None
    span_start: int
    span_end: int
    confidence: float
    extraction_method: str | None = None
    context_window_start: int | None = None
    context_window_end: int | None = None
    rejection_reason: str | None = None
    metadata: dict | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("mention_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def parse_chunk_ids(cls, v: Any) -> list | None:
        """Parse chunk_ids."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> dict | None:
        """Parse metadata."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class MentionLinkRow(_BaseRowModel):
    """Row from mention_link table."""

    link_id: str
    mention_id: str
    entity_id: str | None = None
    link_confidence: float | None = None
    link_method: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None


class RegistryUpdateProposalRow(_BaseRowModel):
    """Row from registry_update_proposal table."""

    proposal_id: str
    surface_form: str
    proposal_type: str
    target_entity_id: str | None = None
    inferred_type: str | None = None
    evidence_doc_ids: list[str]
    occurrence_count: int = 1
    status: str = "pending"
    review_notes: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("evidence_doc_ids", mode="before")
    @classmethod
    def parse_evidence_doc_ids(cls, v: Any) -> list[str]:
        """Parse evidence_doc_ids."""
        if isinstance(v, str):
            return json.loads(v)
        return v


class ChunkEmbeddingRow(_BaseRowModel):
    """Row from chunk_embedding table."""

    chunk_id: str
    embedding_vector: bytes
    embedding_dim: int
    model_version: str
    language_used: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None


class EmbeddingIndexRow(_BaseRowModel):
    """Row from embedding_index table."""

    index_id: str
    run_id: str
    model_version: str
    embedding_dim: int
    method: str
    index_path: str
    built_at: datetime
    chunk_count: int
    build_params_json: dict | None = None
    checksum: str | None = None
    created_at: datetime | None = None

    @field_validator("build_params_json", mode="before")
    @classmethod
    def parse_build_params(cls, v: Any) -> dict | None:
        """Parse build_params."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class FacetAssignmentRow(_BaseRowModel):
    """Row from facet_assignment table."""

    facet_id: str
    doc_version_id: str
    facet_type: str
    facet_value: str
    confidence: float
    signals_json: dict | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("facet_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ids."""
        return _validate_sha256_hex(v)

    @field_validator("signals_json", mode="before")
    @classmethod
    def parse_signals(cls, v: Any) -> dict | None:
        """Parse signals."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class FacetAssignmentEvidenceRow(_BaseRowModel):
    """Row from facet_assignment_evidence table."""

    facet_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None


class NoveltyLabelRow(_BaseRowModel):
    """Row from novelty_label table."""

    doc_version_id: str
    label: str
    neighbor_doc_version_ids: list | None = None
    similarity_score: float | None = None
    shared_mentions: dict | None = None
    linking_window_days: int | None = None
    confidence: float
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("neighbor_doc_version_ids", mode="before")
    @classmethod
    def parse_neighbors(cls, v: Any) -> list | None:
        """Parse neighbors."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("shared_mentions", mode="before")
    @classmethod
    def parse_shared_mentions(cls, v: Any) -> dict | None:
        """Parse shared_mentions."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class NoveltyLabelEvidenceRow(_BaseRowModel):
    """Row from novelty_label_evidence table."""

    doc_version_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None


class ChunkNoveltyRow(_BaseRowModel):
    """Row from chunk_novelty table."""

    chunk_id: str
    novelty_label: str
    source_chunk_ids: list | None = None
    similarity_scores: list | None = None
    created_at: datetime | None = None

    @field_validator("source_chunk_ids", "similarity_scores", mode="before")
    @classmethod
    def parse_json_arrays(cls, v: Any) -> list | None:
        """Parse json_arrays."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class ChunkNoveltyScoreRow(_BaseRowModel):
    """Row from chunk_novelty_score table."""

    chunk_id: str
    novelty_score: float
    method: str
    created_in_run_id: str
    created_at: datetime | None = None


class DocumentFingerprintRow(_BaseRowModel):
    """Row from document_fingerprint table."""

    doc_version_id: str
    minhash_signature: bytes
    simhash_signature: bytes | None = None
    created_in_run_id: str
    created_at: datetime | None = None


class StoryClusterRow(_BaseRowModel):
    """Row from story_cluster table."""

    run_id: str
    story_id: str
    created_at: datetime | None = None
    cluster_method: str | None = None
    seed_doc_version_id: str | None = None
    summary_text: str | None = None


class StoryClusterMemberRow(_BaseRowModel):
    """Row from story_cluster_member table."""

    run_id: str
    story_id: str
    doc_version_id: str
    score: float | None = None
    role: str | None = None


class EventRow(_BaseRowModel):
    """Row from event table."""

    event_id: str
    event_type: str
    canonical_key: str
    current_revision_id: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        """Validate event_id."""
        return _validate_sha256_hex(v)


class EventRevisionRow(_BaseRowModel):
    """Row from event_revision table."""

    revision_id: str
    event_id: str
    revision_no: int
    slots_json: dict
    doc_version_ids: list[str]
    confidence: float
    extraction_method: str | None = None
    extraction_tier: int | None = None
    supersedes_revision_id: str | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("revision_id", "event_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate revision_id."""
        return _validate_sha256_hex(v)

    @field_validator("slots_json", mode="before")
    @classmethod
    def parse_slots(cls, v: Any) -> dict:
        """Parse slots_json."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("doc_version_ids", mode="before")
    @classmethod
    def parse_doc_ids(cls, v: Any) -> list[str]:
        """Parse doc_version_ids."""
        if isinstance(v, str):
            return json.loads(v)
        return v


class EventRevisionEvidenceRow(_BaseRowModel):
    """Row from event_revision_evidence table."""

    revision_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None


class EventEntityLinkRow(_BaseRowModel):
    """Row from event_entity_link table."""

    revision_id: str
    entity_id: str
    role: str
    confidence: float
    created_in_run_id: str
    created_at: datetime | None = None


class MetricObservationRow(_BaseRowModel):
    """Row from metric_observation table."""

    metric_id: str
    doc_version_id: str
    table_id: str | None = None
    metric_name: str
    value_raw: str
    unit_raw: str | None = None
    value_norm: float | None = None
    unit_norm: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    period_granularity: str | None = None
    geography: str | None = None
    table_row_index: int | None = None
    table_col_index: int | None = None
    evidence_id: str | None = None
    parse_quality: float | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("metric_id", "doc_version_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate metric_id."""
        return _validate_sha256_hex(v)


class EventCandidateRow(_BaseRowModel):
    """Row from event_candidate table."""

    candidate_id: str
    doc_version_id: str
    event_type: str
    partial_slots: dict | None = None
    confidence: float | None = None
    status: Literal["candidate", "rejected", "promoted"] = "candidate"
    rejection_reason: str | None = None
    extraction_tier: int | None = None
    created_in_run_id: str
    created_at: datetime | None = None

    @field_validator("partial_slots", mode="before")
    @classmethod
    def parse_partial_slots(cls, v: Any) -> dict | None:
        """Parse partial_slots."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class MetricSeriesRow(_BaseRowModel):
    """Row from metric_series table."""

    run_id: str
    series_id: str
    metric_name: str
    geography: str | None = None
    period_granularity: str
    unit_norm: str | None = None
    created_at: datetime | None = None


class MetricSeriesPointRow(_BaseRowModel):
    """Row from metric_series_point table."""

    run_id: str
    series_id: str
    period_start: date
    period_end: date | None = None
    value_norm: float
    source_doc_version_id: str
    evidence_id: str
    created_at: datetime | None = None


class WatchlistRow(_BaseRowModel):
    """Row from watchlist table."""

    watchlist_id: str
    name: str
    entity_type: str
    entity_values: list | str
    track_events: str | None = None
    track_topics: str | None = None
    alert_severity: str = "info"
    active: int = 1
    created_at: datetime | None = None

    @field_validator("entity_values", mode="before")
    @classmethod
    def parse_entity_values(cls, v: Any) -> list | str:
        """Parse entity_values."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


class AlertRuleRow(_BaseRowModel):
    """Row from alert_rule table."""

    rule_id: str
    name: str
    conditions_json: dict | list
    severity: str
    suppression_window_hours: int = 24
    active: int = 1
    created_at: datetime | None = None

    @field_validator("conditions_json", mode="before")
    @classmethod
    def parse_conditions(cls, v: Any) -> dict | list:
        """Parse conditions."""
        if isinstance(v, str):
            return json.loads(v)
        return v


class AlertRow(_BaseRowModel):
    """Row from alert table."""

    alert_id: str
    run_id: str
    rule_id: str | None = None
    triggered_at: datetime
    trigger_payload_json: dict
    doc_version_ids: list[str]
    event_ids: list | None = None
    acknowledged: int = 0
    created_at: datetime | None = None

    @field_validator("trigger_payload_json", mode="before")
    @classmethod
    def parse_trigger_payload(cls, v: Any) -> dict:
        """Parse trigger_payload."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("doc_version_ids", mode="before")
    @classmethod
    def parse_doc_ids(cls, v: Any) -> list[str]:
        """Parse doc_version_ids."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("event_ids", mode="before")
    @classmethod
    def parse_event_ids(cls, v: Any) -> list | None:
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class AlertEvidenceRow(_BaseRowModel):
    """Row from alert_evidence table."""

    alert_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None


class DigestItemRow(_BaseRowModel):
    """Row from digest_item table."""

    item_id: str
    run_id: str
    digest_date: date
    section: str
    item_type: str
    doc_version_ids: list[str]
    payload_json: dict
    event_ids: list | None = None
    novelty_label: str | None = None
    language_original: str | None = None
    language_presented: str | None = None
    translation_status: str | None = None
    translation_text: str | None = None
    created_at: datetime | None = None

    @field_validator("doc_version_ids", mode="before")
    @classmethod
    def parse_doc_ids(cls, v: Any) -> list[str]:
        """Parse doc_version_ids."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("payload_json", mode="before")
    @classmethod
    def parse_payload(cls, v: Any) -> dict:
        """Parse payload."""
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("event_ids", mode="before")
    @classmethod
    def parse_event_ids(cls, v: Any) -> list | None:
        """Parse event_ids."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class DigestItemEvidenceRow(_BaseRowModel):
    """Row from digest_item_evidence table."""

    item_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None


class EntityTimelineItemRow(_BaseRowModel):
    """Row from entity_timeline_item table."""

    item_id: str
    run_id: str
    entity_id: str
    item_type: str
    category: str
    ref_revision_id: str | None = None
    ref_mention_id: str | None = None
    ref_doc_version_id: str | None = None
    event_time: datetime | None = None
    time_source: str | None = None
    summary_text: str | None = None
    context_json: dict | None = None
    created_at: datetime | None = None

    @field_validator("context_json", mode="before")
    @classmethod
    def parse_context(cls, v: Any) -> dict | None:
        """Parse context."""
        if v is None:
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class EntityTimelineItemEvidenceRow(_BaseRowModel):
    """Row from entity_timeline_item_evidence table."""

    item_id: str
    evidence_id: str
    purpose: str
    created_at: datetime | None = None

# --- DATABASE ERRORS

class DBError(Exception):
    """Base exception for database operations."""


class DBConstraintError(DBError):
    """Raised when a database constraint is violated."""


class DBSchemaError(DBError):
    """Raised when schema validation fails."""


class DBDeterminismError(DBError):
    """Raised when deterministic ID verification fails."""


class StageAccessError(DBError):
    """Raised when a stage attempts unauthorized table access."""


class AnotherRunActiveError(Exception):
    """Raised when another pipeline run is already active."""

    def __init__(self, run_id: str) -> None:
        """Initialize the exception."""
        self.run_id = run_id
        super().__init__(
            f"Cannot acquire run lock for '{run_id}'. "
            "Another pipeline run is already active."
        )


def compute_sha256_id(*components: str | int | None) -> str:
    """
    Compute a deterministic SHA256 hex ID from components.

    :param components: Values to join with '|' separator.
    :return: Lowercase 64-character hex string.
    """
    joined = "|".join(str(c) if c is not None else "" for c in components)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest().lower()
    if len(digest) != 64:
        raise ValueError(f"SHA256 digest has unexpected length: {len(digest)}")
    return digest


def _serialize_json(value: dict | list | None) -> str | None:
    """Serialize a value to canonical JSON text."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _parse_json(text: str | None) -> dict | list | None:
    """Parse JSON text to Python object."""
    if text is None:
        return None
    return json.loads(text)

EXPECTED_TABLES = frozenset([
    "pipeline_run", "scrape_record", "document", "document_version",
    "evidence_span", "doc_stage_status", "run_stage_status", "llm_usage_log",
    "llm_cache", "validation_failure", "block", "chunk", "chunk_fts",
    "table_extract", "doc_metadata", "entity_registry", "entity_registry_audit",
    "mention", "mention_link", "registry_update_proposal", "chunk_embedding",
    "embedding_index", "facet_assignment", "facet_assignment_evidence",
    "novelty_label", "novelty_label_evidence", "chunk_novelty",
    "chunk_novelty_score", "document_fingerprint", "story_cluster",
    "story_cluster_member", "event", "event_revision", "event_revision_evidence",
    "event_entity_link", "metric_observation", "event_candidate", "metric_series",
    "metric_series_point", "watchlist", "alert_rule", "alert", "alert_evidence",
    "digest_item", "digest_item_evidence", "entity_timeline_item",
    "entity_timeline_item_evidence",
])

# DATABASE INTERFACE

class DatabaseInterface(ABC):
    """
    Base class for stage-specific database adapters.

    Manages connections to the source DB (read-only) and working DB (read/write),
    enforces stage isolation via READS/WRITES sets, and provides transaction helpers
    and per-table CRUD methods.
    """

    READS: ClassVar[set[str]] = set()
    WRITES: ClassVar[set[str]] = set()

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
        stage_name: str = "unknown",
    ) -> None:
        """Initialize the database adapter."""
        self._working_db_path = working_db_path
        self._source_db_path = source_db_path
        self._stage_name = stage_name
        self._working_conn: sqlite3.Connection | None = None
        self._source_conn: sqlite3.Connection | None = None
        self._in_transaction = False

    def open(self) -> None:
        """Open database connections and ensure schema."""
        self._working_conn = sqlite3.connect(
            str(self._working_db_path), timeout=30.0, isolation_level=None,
        )
        self._working_conn.row_factory = sqlite3.Row
        self._working_conn.execute("PRAGMA foreign_keys = ON")
        self._working_conn.execute("PRAGMA busy_timeout = 30000")
        self._validate_sqlite_features()
        if self._is_fresh_db():
            self._create_schema()
        else:
            self._validate_schema()
        if self._source_db_path is not None:
            uri = f"file:{self._source_db_path}?mode=ro"
            self._source_conn = sqlite3.connect(uri, uri=True)
            self._source_conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close all database connections."""
        if self._working_conn is not None:
            self._working_conn.close()
            self._working_conn = None
        if self._source_conn is not None:
            self._source_conn.close()
            self._source_conn = None

    def __enter__(self) -> "DatabaseInterface":
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()

    def _validate_sqlite_features(self) -> None:
        assert self._working_conn is not None
        try:
            self._working_conn.execute("SELECT json_valid('[]')")
        except sqlite3.OperationalError as e:
            raise DBSchemaError(f"JSON1 extension not available: {e}") from e
        compile_opts = {
            row[0] for row in self._working_conn.execute("PRAGMA compile_options")
        }
        if "ENABLE_FTS5" not in compile_opts:
            raise DBSchemaError("FTS5 extension not available")

    def _is_fresh_db(self) -> bool:
        assert self._working_conn is not None
        result = self._working_conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        return result[0] == 0

    def _create_schema(self) -> None:
        assert self._working_conn is not None
        schema_sql = Path(SCHEMA_PATH).read_text(encoding="utf-8")
        self._working_conn.executescript(schema_sql)

    def _validate_schema(self) -> None:
        assert self._working_conn is not None
        existing = {
            row[0] for row in self._working_conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        # missing = EXPECTED_TABLES - existing
        # if missing:
        #     raise DBSchemaError(
        #         f"Schema validation failed. Missing tables: {sorted(missing)}. Rebuild required."
        #     )
        required_tables = self.READS | self.WRITES
        missing = required_tables - existing

        if missing:
            raise DBSchemaError(f"Schema validation failed for stage '{self._stage_name}'. Missing tables: {sorted(missing)}. This stage requires: {sorted(required_tables)}")

    def _check_read_access(self, table: str) -> None:
        if table not in self.READS and table not in self.WRITES:
            raise StageAccessError(
                f"Stage '{self._stage_name}' cannot READ from '{table}'."
            )

    def _check_write_access(self, table: str) -> None:
        if table not in self.WRITES:
            raise StageAccessError(
                f"Stage '{self._stage_name}' cannot WRITE to '{table}'."
            )

    @contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[None]:
        """Context manager for a database transaction."""
        assert self._working_conn is not None
        if self._in_transaction:
            yield
            return
        begin_stmt = "BEGIN IMMEDIATE" if immediate else "BEGIN"
        self._working_conn.execute(begin_stmt)
        self._in_transaction = True
        try:
            yield
            self._working_conn.execute("COMMIT")
        except Exception:
            self._working_conn.execute("ROLLBACK")
            raise
        finally:
            self._in_transaction = False

    def _execute(self, sql: str, params: tuple | dict | None = None) -> sqlite3.Cursor:
        assert self._working_conn is not None
        try:
            return self._working_conn.execute(sql, params or ())
        except sqlite3.IntegrityError as e:
            raise DBConstraintError(str(e)) from e

    def _fetchone(self, sql: str, params: tuple | dict | None = None) -> sqlite3.Row | None:
        return self._execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: tuple | dict | None = None) -> list[sqlite3.Row]:
        return self._execute(sql, params).fetchall()

    def get_latest_completed_run(self) -> PipelineRunRow | None:
        """
        Get the most recent completed pipeline run.

        :return: Latest completed run or None if no completed runs exist.
        """
        self._check_read_access("pipeline_run")
        row = self._fetchone(
            """SELECT * FROM pipeline_run
               WHERE status = 'completed'
               ORDER BY completed_at DESC
               LIMIT 1"""
        )
        return PipelineRunRow.model_validate(dict(row)) if row else None

    def get_any_running_run(self) -> PipelineRunRow | None:
        """
        Get any currently running pipeline run.

        :return: Running pipeline run or None if none active.
        """
        self._check_read_access("pipeline_run")
        row = self._fetchone(
            "SELECT * FROM pipeline_run WHERE status = 'running' LIMIT 1"
        )
        return PipelineRunRow.model_validate(dict(row)) if row else None

    def resume_pipeline_run(self, run_id: str, config_version: str) -> None:
        """
        Resume a failed/aborted pipeline run.

        Updates status to 'running' and clears completed_at.
        Config version is updated to current (caller must validate drift).

        :param run_id: Run ID to resume.
        :param config_version: Current config version hash.
        :raises DBConstraintError: If another run is already running.
        """
        self._check_write_access("pipeline_run")
        try:
            self._execute(
                """UPDATE pipeline_run
                   SET status = 'running',
                       completed_at = NULL,
                       config_version = ?,
                       started_at = ?
                   WHERE run_id = ?
                     AND status IN ('failed', 'aborted')""",
                (config_version, datetime.now().isoformat(), run_id),
            )
        except DBConstraintError as e:
            raise AnotherRunActiveError(run_id) from e

    def get_or_create_evidence_span(
        self, doc_version_id: str, span_start: int, span_end: int, run_id: str,
        purpose: str | None = None, clean_content: str | None = None,
    ) -> EvidenceSpanRow:
        """Get or create an evidence span with deterministic ID verification."""
        self._check_write_access("evidence_span")
        if clean_content is None:
            row = self._fetchone(
                "SELECT clean_content FROM document_version WHERE doc_version_id = ?",
                (doc_version_id,),
            )
            if row is None:
                raise DBError(f"document_version not found: {doc_version_id}")
            clean_content = row["clean_content"]
        text = clean_content[span_start:span_end]
        expected_id = compute_sha256_id(doc_version_id, span_start, span_end)
        self._execute(
            """INSERT OR IGNORE INTO evidence_span
            (evidence_id, doc_version_id, span_start, span_end, text, purpose, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (expected_id, doc_version_id, span_start, span_end, text, purpose, run_id),
        )
        row = self._fetchone(
            "SELECT * FROM evidence_span WHERE doc_version_id = ? AND span_start = ? AND span_end = ?",
            (doc_version_id, span_start, span_end),
        )
        if row is None:
            raise DBError("Failed to retrieve evidence_span after insert")
        if row["evidence_id"] != expected_id:
            raise DBDeterminismError(
                f"Evidence span ID mismatch. Expected {expected_id}, found {row['evidence_id']}."
            )
        if row["text"] != text:
            raise DBDeterminismError(f"Evidence span text mismatch for {row['evidence_id']}.")
        return EvidenceSpanRow.model_validate(dict(row))

    def get_doc_stage_status(self, doc_version_id: str, stage: str) -> DocStageStatusRow | None:
        """Get the status for a document at a specific stage."""
        self._check_read_access("doc_stage_status")
        row = self._fetchone(
            "SELECT * FROM doc_stage_status WHERE doc_version_id = ? AND stage = ?",
            (doc_version_id, stage),
        )
        return DocStageStatusRow.model_validate(dict(row)) if row else None

    def upsert_doc_stage_status(
        self, doc_version_id: str, stage: str, run_id: str, config_hash: str,
        status: str, error_message: str | None = None, details: str | None = None,
    ) -> DocStageStatusRow:
        """Upsert document stage status respecting immutability rules."""
        self._check_write_access("doc_stage_status")
        existing = self.get_doc_stage_status(doc_version_id, stage)
        if existing is not None and existing.status in ("ok", "skipped"):
            return existing
        processed_at = datetime.now().isoformat() if status in ("ok", "failed", "blocked", "skipped") else None
        attempt = 1 if existing is None else existing.attempt
        if existing and existing.status in ("failed", "blocked", "pending") and status in ("ok", "failed", "blocked", "skipped"):
            attempt = existing.attempt + 1
        self._execute(
            """INSERT INTO doc_stage_status
            (doc_version_id, stage, run_id, attempt, status, processed_at, config_hash, error_message, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_version_id, stage) DO UPDATE SET
                run_id = excluded.run_id, attempt = excluded.attempt, status = excluded.status,
                processed_at = excluded.processed_at, config_hash = excluded.config_hash,
                error_message = excluded.error_message, details = excluded.details""",
            (doc_version_id, stage, run_id, attempt, status, processed_at, config_hash, error_message, details),
        )
        result = self.get_doc_stage_status(doc_version_id, stage)
        assert result is not None
        return result

    def get_run_stage_status(self, run_id: str, stage: str) -> RunStageStatusRow | None:
        """Get the status for a run at a specific stage."""
        self._check_read_access("run_stage_status")
        row = self._fetchone(
            "SELECT * FROM run_stage_status WHERE run_id = ? AND stage = ?", (run_id, stage),
        )
        return RunStageStatusRow.model_validate(dict(row)) if row else None

    def upsert_run_stage_status(
        self, run_id: str, stage: str, config_hash: str, status: str,
        error_message: str | None = None, details: str | None = None,
    ) -> RunStageStatusRow:
        """Upsert run stage status."""
        self._check_write_access("run_stage_status")
        existing = self.get_run_stage_status(run_id, stage)
        started_at = datetime.now().isoformat() if existing is None else None
        completed_at = datetime.now().isoformat() if status in ("ok", "failed") else None
        attempt = 1 if existing is None else existing.attempt
        if existing and existing.status == "failed" and status in ("ok", "failed"):
            attempt = existing.attempt + 1
        self._execute(
            """INSERT INTO run_stage_status
            (run_id, stage, attempt, status, started_at, completed_at, config_hash, error_message, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, stage) DO UPDATE SET
                attempt = excluded.attempt, status = excluded.status,
                completed_at = COALESCE(excluded.completed_at, run_stage_status.completed_at),
                config_hash = excluded.config_hash, error_message = excluded.error_message, details = excluded.details""",
            (run_id, stage, attempt, status, started_at, completed_at, config_hash, error_message, details),
        )
        result = self.get_run_stage_status(run_id, stage)
        assert result is not None
        return result

    def insert_pipeline_run(self, row: PipelineRunRow) -> None:
        """Insert a new pipeline run."""
        self._check_write_access("pipeline_run")
        self._execute(
            """INSERT INTO pipeline_run (run_id, started_at, completed_at, config_version,
                budget_spent, doc_count_processed, doc_count_skipped, doc_count_failed, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.run_id, row.started_at.isoformat(), row.completed_at.isoformat() if row.completed_at else None,
             row.config_version, row.budget_spent, row.doc_count_processed, row.doc_count_skipped,
             row.doc_count_failed, row.status),
        )

    def get_pipeline_run(self, run_id: str) -> PipelineRunRow | None:
        """Get a pipeline run by ID."""
        self._check_read_access("pipeline_run")
        row = self._fetchone("SELECT * FROM pipeline_run WHERE run_id = ?", (run_id,))
        return PipelineRunRow.model_validate(dict(row)) if row else None

    def update_pipeline_run_status(self, run_id: str, status: str, completed_at: datetime | None = None) -> None:
        """Update pipeline run status."""
        self._check_write_access("pipeline_run")
        self._execute(
            "UPDATE pipeline_run SET status = ?, completed_at = ? WHERE run_id = ?",
            (status, completed_at.isoformat() if completed_at else None, run_id),
        )

    def update_pipeline_run_counters(
        self, run_id: str, doc_count_processed: int | None = None,
        doc_count_skipped: int | None = None, doc_count_failed: int | None = None,  # noqa: S608
        budget_spent: float | None = None,
    ) -> None:
        """Update pipeline run counters."""
        self._check_write_access("pipeline_run")
        updates, params = [], []
        if doc_count_processed is not None:
            updates.append("doc_count_processed = ?")
            params.append(doc_count_processed)
        if doc_count_skipped is not None:
            updates.append("doc_count_skipped = ?")
            params.append(doc_count_skipped)
        if doc_count_failed is not None:
            updates.append("doc_count_failed = ?")
            params.append(doc_count_failed)
        if budget_spent is not None:
            updates.append("budget_spent = ?")
            params.append(budget_spent)
        if updates:
            params.append(run_id)
            self._execute(f"UPDATE pipeline_run SET {', '.join(updates)} WHERE run_id = ?", tuple(params)) # noqa: S608

    def insert_scrape_record(self, row: ScrapeRecordRow) -> None:
        """Insert a scrape record."""
        self._check_write_access("scrape_record")
        self._execute(
            """INSERT INTO scrape_record (scrape_id, publisher_id, source_id, url_raw, url_normalized,
                scraped_at, source_published_at, source_title, source_language, raw_content,
                raw_encoding_detected, scrape_kind, ingest_run_id, processing_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.scrape_id, row.publisher_id, row.source_id, row.url_raw, row.url_normalized,
             row.scraped_at.isoformat(), row.source_published_at.isoformat(), row.source_title,
             row.source_language, row.raw_content, row.raw_encoding_detected, row.scrape_kind,
             row.ingest_run_id, row.processing_status),
        )

    def get_scrape_record(self, scrape_id: str) -> ScrapeRecordRow | None:
        """Get a scrape record by ID."""
        self._check_read_access("scrape_record")
        row = self._fetchone("SELECT * FROM scrape_record WHERE scrape_id = ?", (scrape_id,))
        return ScrapeRecordRow.model_validate(dict(row)) if row else None

    def update_scrape_record_processing_status(self, scrape_id: str, processing_status: str) -> None:
        """Update scrape record processing status (only allowed update)."""
        self._check_write_access("scrape_record")
        self._execute("UPDATE scrape_record SET processing_status = ? WHERE scrape_id = ?", (processing_status, scrape_id))

    def insert_watchlist(self, row: WatchlistRow) -> None:
        """Insert a watchlist entry."""
        self._check_write_access("watchlist")
        entity_values = json.dumps(row.entity_values, ensure_ascii=False) if isinstance(row.entity_values, list) else row.entity_values
        self._execute(
            """INSERT INTO watchlist (watchlist_id, name, entity_type, entity_values,
                                      track_events, track_topics, alert_severity, active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.watchlist_id, row.name, row.entity_type, entity_values, row.track_events, row.track_topics, row.alert_severity, row.active),
        )

    def insert_alert_rule(self, row: AlertRuleRow) -> None:
        """Insert an alert rule."""
        self._check_write_access("alert_rule")
        conditions = json.dumps(row.conditions_json, ensure_ascii=False) if isinstance(row.conditions_json, (dict, list)) else row.conditions_json
        self._execute(
            """INSERT INTO alert_rule (rule_id, name, conditions_json, severity,
                                       suppression_window_hours, active)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row.rule_id, row.name, conditions, row.severity, row.suppression_window_hours, row.active),
        )

    def insert_document(self, row: DocumentRow) -> None:
        """Insert a document."""
        self._check_write_access("document")
        self._execute(
            """INSERT INTO document (document_id, publisher_id, url_normalized, source_published_at,
                url_raw_first_seen, document_class, is_attachment) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row.document_id, row.publisher_id, row.url_normalized, row.source_published_at.isoformat(),
             row.url_raw_first_seen, row.document_class, row.is_attachment),
        )

    def get_document(self, document_id: str) -> DocumentRow | None:
        """Get a document by ID."""
        self._check_read_access("document")
        row = self._fetchone("SELECT * FROM document WHERE document_id = ?", (document_id,))
        return DocumentRow.model_validate(dict(row)) if row else None

    def get_or_create_document(self, row: DocumentRow) -> DocumentRow:
        """Get or create a document using its natural key."""
        self._check_write_access("document")
        existing = self._fetchone(
            "SELECT * FROM document WHERE publisher_id = ? AND url_normalized = ? AND source_published_at = ?",
            (row.publisher_id, row.url_normalized, row.source_published_at.isoformat()),
        )
        if existing:
            return DocumentRow.model_validate(dict(existing))
        self.insert_document(row)
        return row

    def insert_document_version(self, row: DocumentVersionRow) -> None:
        """Insert a document version."""
        self._check_write_access("document_version")
        self._execute(
            """INSERT INTO document_version (doc_version_id, document_id, scrape_id, content_hash_raw,
                encoding_repairs_applied, cleaning_spec_version, normalization_spec, pii_masking_enabled,
                scrape_kind, pii_mask_log, content_hash_clean, clean_content, span_indexing,
                content_length_raw, content_length_clean, boilerplate_ratio, content_quality_score,
                primary_language, secondary_languages, language_detection_confidence, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.doc_version_id, row.document_id, row.scrape_id, row.content_hash_raw,
             _serialize_json(row.encoding_repairs_applied), row.cleaning_spec_version,
             _serialize_json(row.normalization_spec), row.pii_masking_enabled, row.scrape_kind,
             _serialize_json(row.pii_mask_log), row.content_hash_clean, row.clean_content,
             row.span_indexing, row.content_length_raw, row.content_length_clean,
             row.boilerplate_ratio, row.content_quality_score, row.primary_language,
             _serialize_json(row.secondary_languages), row.language_detection_confidence, row.created_in_run_id),
        )

    def get_document_version(self, doc_version_id: str) -> DocumentVersionRow | None:
        """Get a document version by ID."""
        self._check_read_access("document_version")
        row = self._fetchone("SELECT * FROM document_version WHERE doc_version_id = ?", (doc_version_id,))
        return DocumentVersionRow.model_validate(dict(row)) if row else None

    def list_doc_version_ids(self) -> list[str]:
        """List all document version IDs in deterministic order."""
        self._check_read_access("document_version")
        rows = self._fetchall("SELECT doc_version_id FROM document_version ORDER BY doc_version_id")
        return [row["doc_version_id"] for row in rows]

    def insert_block(self, row: BlockRow) -> None:
        """Insert a block."""
        self._check_write_access("block")
        self._execute(
            """INSERT INTO block (block_id, doc_version_id, block_type, block_level, span_start, span_end,
                parse_confidence, boilerplate_flag, boilerplate_reason, parent_block_id, language_hint, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.block_id, row.doc_version_id, row.block_type, row.block_level, row.span_start, row.span_end,
             row.parse_confidence, row.boilerplate_flag, row.boilerplate_reason, row.parent_block_id,
             row.language_hint, row.created_in_run_id),
        )

    def insert_blocks(self, rows: Sequence[BlockRow]) -> None:
        """Insert multiple blocks."""
        for row in rows:
            self.insert_block(row)

    def get_blocks_by_doc_version_id(self, doc_version_id: str) -> list[BlockRow]:
        """Get all blocks for a document version."""
        self._check_read_access("block")
        rows = self._fetchall("SELECT * FROM block WHERE doc_version_id = ? ORDER BY span_start", (doc_version_id,))
        return [BlockRow.model_validate(dict(row)) for row in rows]

    def insert_chunk(self, row: ChunkRow) -> None:
        """Insert a chunk."""
        self._check_write_access("chunk")
        self._execute(
            """INSERT INTO chunk (chunk_id, doc_version_id, span_start, span_end, evidence_id, chunk_type,
                block_ids, chunk_text, heading_context, retrieval_exclude, mention_boundary_safe,
                token_count_approx, created_in_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.chunk_id, row.doc_version_id, row.span_start, row.span_end, row.evidence_id, row.chunk_type,
             _serialize_json(row.block_ids), row.chunk_text, row.heading_context, row.retrieval_exclude,
             row.mention_boundary_safe, row.token_count_approx, row.created_in_run_id),
        )

    def insert_chunks(self, rows: Sequence[ChunkRow]) -> None:
        """Insert multiple chunks."""
        for row in rows:
            self.insert_chunk(row)

    def get_chunk(self, chunk_id: str) -> ChunkRow | None:
        """Get a chunk by ID."""
        self._check_read_access("chunk")
        row = self._fetchone("SELECT * FROM chunk WHERE chunk_id = ?", (chunk_id,))
        return ChunkRow.model_validate(dict(row)) if row else None

    def get_chunks_by_doc_version_id(self, doc_version_id: str) -> list[ChunkRow]:
        """Get all chunks for a document version."""
        self._check_read_access("chunk")
        rows = self._fetchall("SELECT * FROM chunk WHERE doc_version_id = ? ORDER BY span_start", (doc_version_id,))
        return [ChunkRow.model_validate(dict(row)) for row in rows]

    def search_chunks_fts(self, query: str, limit: int = 10) -> list[tuple[ChunkRow, float]]:
        """Search chunks using FTS5."""
        self._check_read_access("chunk")
        rows = self._fetchall(
            """SELECT c.*, bm25(chunk_fts) as rank FROM chunk_fts
            JOIN chunk c ON chunk_fts.chunk_id = c.chunk_id
            WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?""", (query, limit),
        )
        return [(ChunkRow.model_validate({k: r[k] for k in r.keys() if k != "rank"}), r["rank"]) for r in rows]

    def insert_table_extract(self, row: TableExtractRow) -> None:
        """Insert a table extract."""
        self._check_write_access("table_extract")
        self._execute(
            """INSERT INTO table_extract (table_id, block_id, doc_version_id, row_count, col_count,
                headers_json, header_row_index, parse_quality, parse_method, table_class,
                period_granularity, units_detected, raw_table_text, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.table_id, row.block_id, row.doc_version_id, row.row_count, row.col_count,
             _serialize_json(row.headers_json), row.header_row_index, row.parse_quality, row.parse_method,
             row.table_class, row.period_granularity, _serialize_json(row.units_detected),
             row.raw_table_text, row.created_in_run_id),
        )

    def get_table_extracts_by_doc_version_id(self, doc_version_id: str) -> list[TableExtractRow]:
        """Get all table extracts for a document version."""
        self._check_read_access("table_extract")
        rows = self._fetchall("SELECT * FROM table_extract WHERE doc_version_id = ?", (doc_version_id,))
        return [TableExtractRow.model_validate(dict(row)) for row in rows]

    def insert_doc_metadata(self, row: DocMetadataRow) -> None:
        """Insert document metadata."""
        self._check_write_access("doc_metadata")
        self._execute(
            """INSERT INTO doc_metadata (doc_version_id, title, title_span_start, title_span_end,
                title_source, title_confidence, title_evidence_id, published_at, published_at_raw,
                published_at_format, published_at_span_start, published_at_span_end, published_at_source,
                published_at_confidence, published_at_evidence_id, detected_document_class,
                document_class_confidence, metadata_extraction_log, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.doc_version_id, row.title, row.title_span_start, row.title_span_end, row.title_source,
             row.title_confidence, row.title_evidence_id, row.published_at.isoformat() if row.published_at else None,
             row.published_at_raw, row.published_at_format, row.published_at_span_start, row.published_at_span_end,
             row.published_at_source, row.published_at_confidence, row.published_at_evidence_id,
             row.detected_document_class, row.document_class_confidence, row.metadata_extraction_log, row.created_in_run_id),
        )

    def get_doc_metadata(self, doc_version_id: str) -> DocMetadataRow | None:
        """Get document metadata by document version ID."""
        self._check_read_access("doc_metadata")
        row = self._fetchone("SELECT * FROM doc_metadata WHERE doc_version_id = ?", (doc_version_id,))
        return DocMetadataRow.model_validate(dict(row)) if row else None

    def insert_entity_registry(self, row: EntityRegistryRow) -> None:
        """Insert an entity registry entry."""
        self._check_write_access("entity_registry")
        self._execute(
            """INSERT INTO entity_registry (entity_id, entity_type, canonical_name, aliases,
                name_variants_de, name_variants_en, abbreviations, compound_forms, valid_from,
                valid_to, source_authority, disambiguation_hints, parent_entity_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.entity_id, row.entity_type, row.canonical_name, _serialize_json(row.aliases),
             row.name_variants_de, row.name_variants_en, row.abbreviations, row.compound_forms,
             row.valid_from.isoformat() if row.valid_from else None,
             row.valid_to.isoformat() if row.valid_to else None, row.source_authority,
             _serialize_json(row.disambiguation_hints), row.parent_entity_id),
        )

    def get_entity_registry(self, entity_id: str) -> EntityRegistryRow | None:
        """Get an entity by ID."""
        self._check_read_access("entity_registry")
        row = self._fetchone("SELECT * FROM entity_registry WHERE entity_id = ?", (entity_id,))
        return EntityRegistryRow.model_validate(dict(row)) if row else None

    def list_entity_registry(self) -> list[EntityRegistryRow]:
        """List all entities in the registry."""
        self._check_read_access("entity_registry")
        rows = self._fetchall("SELECT * FROM entity_registry ORDER BY entity_id")
        return [EntityRegistryRow.model_validate(dict(row)) for row in rows]

    def insert_mention(self, row: MentionRow) -> None:
        """Insert a mention."""
        self._check_write_access("mention")
        self._execute(
            """INSERT INTO mention (mention_id, doc_version_id, chunk_ids, mention_type, surface_form,
                normalized_value, span_start, span_end, confidence, extraction_method,
                context_window_start, context_window_end, rejection_reason, metadata, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.mention_id, row.doc_version_id, _serialize_json(row.chunk_ids), row.mention_type,
             row.surface_form, row.normalized_value, row.span_start, row.span_end, row.confidence,
             row.extraction_method, row.context_window_start, row.context_window_end, row.rejection_reason,
             _serialize_json(row.metadata), row.created_in_run_id),
        )

    def insert_mentions(self, rows: Sequence[MentionRow]) -> None:
        """Insert multiple mentions."""
        for row in rows:
            self.insert_mention(row)

    def get_mentions_by_doc_version_id(self, doc_version_id: str) -> list[MentionRow]:
        """Get all mentions for a document version."""
        self._check_read_access("mention")
        rows = self._fetchall("SELECT * FROM mention WHERE doc_version_id = ? ORDER BY span_start", (doc_version_id,))
        return [MentionRow.model_validate(dict(row)) for row in rows]

    def insert_mention_link(self, row: MentionLinkRow) -> None:
        """Insert a mention link."""
        self._check_write_access("mention_link")
        self._execute(
            """INSERT INTO mention_link (link_id, mention_id, entity_id, link_confidence, link_method, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (row.link_id, row.mention_id, row.entity_id, row.link_confidence, row.link_method, row.created_in_run_id),
        )

    def insert_validation_failure(self, row: ValidationFailureRow) -> None:
        """Insert a validation failure."""
        self._check_write_access("validation_failure")
        self._execute(
            """INSERT INTO validation_failure (failure_id, run_id, stage, doc_version_id, check_name, details, severity, auto_repaired)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.failure_id, row.run_id, row.stage, row.doc_version_id, row.check_name, row.details, row.severity, row.auto_repaired),
        )

    def insert_llm_usage_log(self, row: LLMUsageLogRow) -> None:
        """Insert an LLM usage log entry."""
        self._check_write_access("llm_usage_log")
        self._execute(
            """INSERT INTO llm_usage_log (log_id, run_id, stage, purpose, model, tokens_in, tokens_out, cost, cached, cache_key, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.log_id, row.run_id, row.stage, row.purpose, row.model, row.tokens_in, row.tokens_out, row.cost, row.cached, row.cache_key, row.latency_ms),
        )

    def get_llm_cache(self, cache_key: str) -> LLMCacheRow | None:
        """Retrieve a cached LLM response if it exists and hasn't expired. Automatically increments hit_count when a valid cache entry is found."""
        self._check_read_access("llm_cache")
        row = self._fetchone(
            """SELECT * FROM llm_cache
               WHERE cache_key = ? AND expires_at > ?""",
            (cache_key, datetime.now().isoformat()),
        )
        if row:
            # Update hit count and re-fetch to get updated value
            self._check_write_access("llm_cache")
            self._execute("UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (cache_key,))
            updated_row = self._fetchone("SELECT * FROM llm_cache WHERE cache_key = ?", (cache_key,))
            return LLMCacheRow.model_validate(dict(updated_row))
        return None

    def set_llm_cache(self, cache_key: str, model: str, prompt_hash: str, response: str, ttl: timedelta) -> None:
        """Store an LLM response in the cache with a time-to-live."""
        self._check_write_access("llm_cache")
        expires_at = datetime.now() + ttl
        self._execute(
            """INSERT OR REPLACE INTO llm_cache
               (cache_key, model, prompt_hash, response, expires_at, hit_count)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (cache_key, model, prompt_hash, response, expires_at.isoformat()),
        )

    def insert_alert(self, row: AlertRow) -> None:
        """Insert an alert."""
        self._check_write_access("alert")
        self._execute(
            """INSERT INTO alert (alert_id, run_id, rule_id, triggered_at, trigger_payload_json, doc_version_ids, event_ids, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.alert_id, row.run_id, row.rule_id, row.triggered_at.isoformat(),
             _serialize_json(row.trigger_payload_json), _serialize_json(row.doc_version_ids),
             _serialize_json(row.event_ids), row.acknowledged),
        )

    def insert_alert_evidence(self, row: AlertEvidenceRow) -> None:
        """Insert alert evidence."""
        self._check_write_access("alert_evidence")
        self._execute(
            "INSERT INTO alert_evidence (alert_id, evidence_id, purpose) VALUES (?, ?, ?)",
            (row.alert_id, row.evidence_id, row.purpose),
        )

    def insert_alert_with_evidence(self, alert: AlertRow, evidence: Sequence[AlertEvidenceRow]) -> None:
        """Insert an alert with its evidence atomically."""
        self._check_write_access("alert")
        self._check_write_access("alert_evidence")
        self.insert_alert(alert)
        for ev in evidence:
            self.insert_alert_evidence(ev)

    def delete_alerts_for_run(self, run_id: str) -> int:
        """Delete alerts for a run (FK-safe order)."""
        self._check_write_access("alert_evidence")
        self._check_write_access("alert")
        self._execute("DELETE FROM alert_evidence WHERE alert_id IN (SELECT alert_id FROM alert WHERE run_id = ?)", (run_id,))
        cursor = self._execute("DELETE FROM alert WHERE run_id = ?", (run_id,))
        return cursor.rowcount

    def insert_digest_item(self, row: DigestItemRow) -> None:
        """Insert a digest item."""
        self._check_write_access("digest_item")
        self._execute(
            """INSERT INTO digest_item (item_id, run_id, digest_date, section, item_type, doc_version_ids,
                payload_json, event_ids, novelty_label, language_original, language_presented, translation_status, translation_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.item_id, row.run_id, row.digest_date.isoformat(), row.section, row.item_type,
             _serialize_json(row.doc_version_ids), _serialize_json(row.payload_json),
             _serialize_json(row.event_ids), row.novelty_label, row.language_original,
             row.language_presented, row.translation_status, row.translation_text),
        )

    def insert_digest_item_evidence(self, row: DigestItemEvidenceRow) -> None:
        """Insert digest item evidence."""
        self._check_write_access("digest_item_evidence")
        self._execute(
            "INSERT INTO digest_item_evidence (item_id, evidence_id, purpose) VALUES (?, ?, ?)",
            (row.item_id, row.evidence_id, row.purpose),
        )

    def insert_digest_item_with_evidence(self, item: DigestItemRow, evidence: Sequence[DigestItemEvidenceRow]) -> None:
        """Insert a digest item with its evidence atomically."""
        self._check_write_access("digest_item")
        self._check_write_access("digest_item_evidence")
        self.insert_digest_item(item)
        for ev in evidence:
            self.insert_digest_item_evidence(ev)

    def delete_digest_items_for_run(self, run_id: str) -> int:
        """Delete digest items for a run (FK-safe order)."""
        self._check_write_access("digest_item_evidence")
        self._check_write_access("digest_item")
        self._execute("DELETE FROM digest_item_evidence WHERE item_id IN (SELECT item_id FROM digest_item WHERE run_id = ?)", (run_id,))
        cursor = self._execute("DELETE FROM digest_item WHERE run_id = ?", (run_id,))
        return cursor.rowcount

    def insert_entity_timeline_item(self, row: EntityTimelineItemRow) -> None:
        """Insert an entity timeline item."""
        self._check_write_access("entity_timeline_item")
        self._execute(
            """INSERT INTO entity_timeline_item (item_id, run_id, entity_id, item_type, category,
                ref_revision_id, ref_mention_id, ref_doc_version_id, event_time, time_source, summary_text, context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.item_id, row.run_id, row.entity_id, row.item_type, row.category, row.ref_revision_id,
             row.ref_mention_id, row.ref_doc_version_id, row.event_time.isoformat() if row.event_time else None,
             row.time_source, row.summary_text, _serialize_json(row.context_json)),
        )

    def insert_entity_timeline_item_evidence(self, row: EntityTimelineItemEvidenceRow) -> None:
        """Insert entity timeline item evidence."""
        self._check_write_access("entity_timeline_item_evidence")
        self._execute(
            "INSERT INTO entity_timeline_item_evidence (item_id, evidence_id, purpose) VALUES (?, ?, ?)",
            (row.item_id, row.evidence_id, row.purpose),
        )

    def insert_timeline_item_with_evidence(self, item: EntityTimelineItemRow, evidence: Sequence[EntityTimelineItemEvidenceRow]) -> None:
        """Insert a timeline item with its evidence atomically."""
        self._check_write_access("entity_timeline_item")
        self._check_write_access("entity_timeline_item_evidence")
        self.insert_entity_timeline_item(item)
        for ev in evidence:
            self.insert_entity_timeline_item_evidence(ev)

    def delete_timeline_items_for_run(self, run_id: str) -> int:
        """Delete timeline items for a run (FK-safe order)."""
        self._check_write_access("entity_timeline_item_evidence")
        self._check_write_access("entity_timeline_item")
        self._execute("DELETE FROM entity_timeline_item_evidence WHERE item_id IN (SELECT item_id FROM entity_timeline_item WHERE run_id = ?)", (run_id,))
        cursor = self._execute("DELETE FROM entity_timeline_item WHERE run_id = ?", (run_id,))
        return cursor.rowcount

    def read_source_publications(self, publisher_table: str, sort_date:bool=True) -> list[SourcePublicationRow]:
        """Read publications from a source DB publisher table."""
        if self._source_conn is None:
            raise DBError("Source database not connected")
        # Source table schema
        sql = f'SELECT ID, published_on, title, added_on, url, language, post FROM "{publisher_table}"' # noqa S608
        if sort_date:
            sql += " ORDER BY published_on DESC"
        sql += ";"

        publications: List[SourcePublicationRow] = []

        cursor = self._source_conn.execute(sql)
        for pid, pub_dt, title, add_dt, url, language, blob in cursor.fetchall():
            text = decompress_publication_text(pid, blob) # source table contains compressed text
            try:
                publications.append(
                    SourcePublicationRow(
                        id = str(pid),
                        published_on = pub_dt,
                        added_on = add_dt,
                        url = url,
                        title = title,
                        content = text,
                        language = language,
                    )
                )
            except ValidationError as e:
                logger.error(f"Failed to extract publication from source table {pub_dt}_{publisher_table}_{title} with: {e}")
                continue
        return publications

    def get_source_table_names(self) -> list[str]:
        """Get list of publisher tables in source DB."""
        if self._source_conn is None:
            raise DBError("Source database not connected")
        cursor = self._source_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]