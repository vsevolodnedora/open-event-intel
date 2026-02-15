"""
Stage 06 — Taxonomy classification.

Assigns topic and document-type facets to each document using keyword scoring,
mention signals, structural signals, and publisher priors from ``config.yaml``.
Every facet is provenance-locked via ``evidence_span`` / ``facet_assignment_evidence``.

:Reads: ``document_version``, ``document``, ``block``, ``mention``,
        ``doc_stage_status``, ``pipeline_run``
:Writes: ``facet_assignment``, ``facet_assignment_evidence``, ``evidence_span``,
         ``doc_stage_status``
:Prerequisites: ``stage_03_metadata`` AND ``stage_04_mentions`` (both ``ok``)

**Responsibility**:
  * Normalize heterogeneous documents into a consistent taxonomy (e.g., "grid outage", "TSO plan", "EU regulation", "market update", etc.—the concrete labels come from config.yaml).
  * Produce explainable, provenance-locked classifications: every facet must be justifiable by exact spans in document_version.clean_content.
  * Provide stable, queryable signals for (with `facet_id` being deterministic, e.g., hash):
  * Employs: Candidate generation, Feature scoring → confidence, Evidence anchoring algorithms
"""

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config_interface import (
    ClassificationSettings,
    Config,
    DocumentType,
    KeywordWeights,
    Taxonomy,
    Topic,
    TopicKeywords,
    get_config_version,
    load_config,
)
from database_interface import (
    BlockRow,
    DatabaseInterface,
    DBError,
    DocStageStatusRow,
    EvidenceSpanRow,
    FacetAssignmentEvidenceRow,
    FacetAssignmentRow,
    MentionRow,
    _serialize_json,
    compute_sha256_id,
)

from open_event_intel.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "stage_06_taxonomy"
PREREQUISITE_STAGES: tuple[str, ...] = ("stage_03_metadata", "stage_04_mentions")

# ── Module-level constants (not sourced from config) ──────────────────────
# Number of characters of context to include on each side of the best keyword
# hit when selecting an evidence span.  This is a presentation concern (how
# much surrounding text to store as evidence), not a classification parameter.
EVIDENCE_CONTEXT_CHARS: int = 80

# Minimum keyword length below which a high_signal keyword is flagged as
# a false-positive risk during config validation.  Short words like "Gas"
# (3 chars) tend to match ubiquitously in energy-sector text.
SHORT_KEYWORD_WARN_THRESHOLD: int = 4

# Maximum number of document types that can be assigned before we cap and
# flag the classification.  Unlike topics, document types lack a config-level
# multi_label.max_topics, so we enforce a sensible ceiling in code.
# A document is rarely more than two types simultaneously.
MAX_DOCUMENT_TYPES: int = 2


class Stage06DatabaseInterface(DatabaseInterface):
    """
    Database adapter for Stage 06 (taxonomy classification).

    All SQL lives in :class:`DatabaseInterface`; this subclass only declares
    table ownership and adds the few missing CRUD helpers for
    ``facet_assignment`` / ``facet_assignment_evidence``.
    """

    READS = {
        "pipeline_run",
        "doc_stage_status",
        "document_version",
        "block",
        "mention",
        "document",
    }
    WRITES = {
        "doc_stage_status",
        "facet_assignment",
        "facet_assignment_evidence",
        "evidence_span",
    }

    def __init__(
        self,
        working_db_path: Path,
        source_db_path: Path | None = None,
    ) -> None:
        """Initialize a Stage06DatabaseInterface."""
        super().__init__(working_db_path, source_db_path, STAGE_NAME)

    def get_iteration_set(self) -> list[str]:
        """
        Return ``doc_version_id``s that need (re-)evaluation.

        Per §6.3.0 the iteration set comprises documents with:
        * no ``doc_stage_status`` row for this stage, OR
        * ``status = 'failed'``, OR
        * ``status = 'blocked'`` **and** all prerequisites are now ``ok``.
        """
        self._check_read_access("doc_stage_status")
        self._check_read_access("document_version")
        rows = self._fetchall(
            """
            SELECT dv.doc_version_id
            FROM document_version dv
            JOIN document d ON dv.document_id = d.document_id
            WHERE dv.doc_version_id NOT IN (
                SELECT doc_version_id FROM doc_stage_status
                WHERE stage = ? AND status IN ('ok', 'skipped')
            )
            ORDER BY d.publisher_id, d.url_normalized, dv.doc_version_id
            """,
            (STAGE_NAME,),
        )
        return [r["doc_version_id"] for r in rows]

    def check_prerequisites(self, doc_version_id: str) -> tuple[bool, str | None]:
        """
        Check whether all prerequisite stages are ``ok``.

        :returns: ``(True, None)`` if all ok, else ``(False, error_message)``.
        """
        for prereq in PREREQUISITE_STAGES:
            status_row = self.get_doc_stage_status(doc_version_id, prereq)
            if status_row is None or status_row.status != "ok":
                blocking_status = status_row.status if status_row else "missing"
                return False, f"prerequisite_not_ok:{prereq}:{blocking_status}"
        return True, None

    def insert_facet_assignment(self, row: FacetAssignmentRow) -> None:
        """Insert a facet assignment row."""
        self._check_write_access("facet_assignment")
        self._execute(
            """INSERT INTO facet_assignment
            (facet_id, doc_version_id, facet_type, facet_value, confidence,
             signals_json, created_in_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row.facet_id,
                row.doc_version_id,
                row.facet_type,
                row.facet_value,
                row.confidence,
                _serialize_json(row.signals_json),
                row.created_in_run_id,
            ),
        )

    def insert_facet_assignment_evidence(
        self, row: FacetAssignmentEvidenceRow
    ) -> None:
        """Insert a facet-assignment ↔ evidence link."""
        self._check_write_access("facet_assignment_evidence")
        self._execute(
            """INSERT INTO facet_assignment_evidence
            (facet_id, evidence_id, purpose) VALUES (?, ?, ?)""",
            (row.facet_id, row.evidence_id, row.purpose),
        )

    def get_facet_assignments_for_doc(
        self, doc_version_id: str
    ) -> list[FacetAssignmentRow]:
        """Return all facet assignments for a document version."""
        self._check_read_access("facet_assignment")
        rows = self._fetchall(
            "SELECT * FROM facet_assignment WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        return [FacetAssignmentRow.model_validate(dict(r)) for r in rows]

    def get_document_publisher_id(self, doc_version_id: str) -> str | None:
        """Look up the publisher_id for a given doc_version_id."""
        self._check_read_access("document_version")
        self._check_read_access("document")
        row = self._fetchone(
            """SELECT d.publisher_id
            FROM document_version dv
            JOIN document d ON dv.document_id = d.document_id
            WHERE dv.doc_version_id = ?""",
            (doc_version_id,),
        )
        return row["publisher_id"] if row else None

    def delete_facets_for_doc(self, doc_version_id: str) -> int:
        """
        Delete all facet assignments and their evidence links for a document.

        Used for idempotent re-processing: cleans up partial results from a
        previously failed run before re-classifying.

        :returns: Number of facet_assignment rows deleted.
        """
        self._check_write_access("facet_assignment_evidence")
        self._check_write_access("facet_assignment")
        self._execute(
            """DELETE FROM facet_assignment_evidence
            WHERE facet_id IN (
                SELECT facet_id FROM facet_assignment WHERE doc_version_id = ?
            )""",
            (doc_version_id,),
        )
        cursor = self._execute(
            "DELETE FROM facet_assignment WHERE doc_version_id = ?",
            (doc_version_id,),
        )
        return cursor.rowcount


@dataclass(frozen=True)
class KeywordHit:
    """A single keyword match found in the document text."""

    keyword: str
    start: int
    end: int
    weight: float


@dataclass
class FacetScore:
    """Aggregated score for a single (facet_type, facet_value) candidate."""

    facet_type: str
    facet_value: str
    keyword_score: float = 0.0
    mention_score: float = 0.0
    structural_score: float = 0.0
    prior_score: float = 0.0
    confidence: float = 0.0
    keyword_hits: list[KeywordHit] = field(default_factory=list)
    # NOTE: `signals` is mutated after initial scoring to attach confidence
    # level flags and ambiguity/unfocused markers.  The final dict is what
    # gets persisted in facet_assignment.signals_json.
    signals: dict = field(default_factory=dict)


def _build_word_pattern(keyword: str) -> re.Pattern[str]:
    """
    Compile a case-insensitive whole-word regex for *keyword*.

    Multi-word keywords are matched literally.
    """
    escaped = re.escape(keyword)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def score_keywords(
    clean_content: str,
    keywords_by_lang: dict[str, "TopicKeywords"],
    weights: KeywordWeights,
) -> tuple[float, list[KeywordHit]]:
    """
    Score keyword presence in *clean_content*.

    Each **distinct** keyword contributes its weight once to the score,
    regardless of how many times it occurs.  Occurrence counts are recorded
    in hits (for evidence selection) and in debug logging, but do not
    inflate the score.

    Applies compound_match_bonus when >=2 distinct keywords co-occur.

    :returns: ``(score, hits)`` where *hits* carry span offsets for all
        occurrences (used for evidence span selection).
    """
    hits: list[KeywordHit] = []

    # Track distinct keyword matches per signal level
    high_distinct: set[str] = set()
    medium_distinct: set[str] = set()
    occurrence_counts: Counter = Counter()

    for _lang, kw_group in sorted(keywords_by_lang.items()):
        for kw in kw_group.high_signal:
            for m in _build_word_pattern(kw).finditer(clean_content):
                w = weights.high_signal
                hits.append(KeywordHit(kw, m.start(), m.end(), w))
                occurrence_counts[kw.lower()] += 1
                high_distinct.add(kw.lower())
        for kw in kw_group.medium_signal:
            for m in _build_word_pattern(kw).finditer(clean_content):
                w = weights.medium_signal
                hits.append(KeywordHit(kw, m.start(), m.end(), w))
                occurrence_counts[kw.lower()] += 1
                medium_distinct.add(kw.lower())

    # Score based on distinct keywords, not total occurrences.
    # Each distinct high_signal keyword contributes its weight once;
    # same for medium_signal.
    raw_score = (
        len(high_distinct) * weights.high_signal
        + len(medium_distinct) * weights.medium_signal
    )

    # Apply compound match bonus when >=2 distinct keywords found
    total_distinct = len(high_distinct) + len(medium_distinct)
    if total_distinct >= 2:
        raw_score += weights.compound_match_bonus

    if hits:
        # Log top repeated keywords so false-positive drivers are visible
        top_repeats = occurrence_counts.most_common(3)
        logger.debug(
            "Keyword scoring: %d hit(s) from %d distinct keyword(s) "
            "(high=%s, medium=%s), top_repeats=%s, "
            "distinct_score=%.4f%s",
            len(hits), total_distinct,
            sorted(high_distinct) if high_distinct else "[]",
            sorted(medium_distinct) if medium_distinct else "[]",
            top_repeats,
            raw_score,
            f" (includes compound_bonus={weights.compound_match_bonus})"
            if total_distinct >= 2 else "",
        )

    return raw_score, hits


def score_mentions(
    mentions: Sequence[MentionRow],
    mention_signals: dict[str, float],
) -> float:
    """Sum signal weights for mention types present in the document."""
    seen_types: set[str] = {m.mention_type for m in mentions}
    matched = {
        mtype: weight
        for mtype, weight in mention_signals.items()
        if mtype in seen_types
    }
    total = sum(matched.values())
    if matched:
        logger.debug(
            "Mention scoring: %d mention(s), %d type(s) seen=%s, "
            "matched signals=%s, score=%.4f",
            len(mentions), len(seen_types), sorted(seen_types),
            matched, total,
        )
    return total


def score_structural(
    blocks: Sequence[BlockRow],
    structural_signals: dict[str, float],
) -> float:
    """
    Sum signal weights for structural features present in the document.

    Structural signals from config (e.g. ``large_table_present``,
    ``multiple_tables``, ``contact_section``) are inferred from block metadata,
    not matched literally against ``block_type``.
    """
    if not structural_signals:
        return 0.0

    # Derive structural features from blocks
    derived_features: set[str] = set()

    table_blocks = [b for b in blocks if b.block_type == "table"]
    if len(table_blocks) >= 1:
        derived_features.add("large_table_present")
    if len(table_blocks) >= 2:
        derived_features.add("multiple_tables")

    # Check for contact section (blocks that look like contact info)
    for b in blocks:
        if b.block_type in ("contact", "footer"):
            derived_features.add("contact_section")
        if b.block_type in ("boilerplate",) or b.boilerplate_flag == "boilerplate":
            derived_features.add("boilerplate_company_info")

    matched_score = sum(
        weight
        for feature, weight in structural_signals.items()
        if feature in derived_features
    )
    if derived_features:
        matched_features = {
            f: structural_signals[f]
            for f in derived_features
            if f in structural_signals
        }
        logger.debug(
            "Structural scoring: %d block(s), %d table(s), "
            "derived_features=%s, matched=%s, score=%.4f",
            len(blocks), len(table_blocks), sorted(derived_features),
            matched_features, matched_score,
        )
    return matched_score


def compute_confidence(
    keyword_score: float,
    mention_score: float,
    structural_score: float,
    prior_score: float,
    settings: ClassificationSettings,
) -> float:
    """
    Combine component scores into a single confidence in [0, 1].

    Each component is capped per ``settings.confidence`` before summing with
    the ``base`` value.  The result is clamped to [0, 1].
    """
    conf = settings.confidence
    total = conf.base
    kw_contrib = min(keyword_score, conf.keyword_cap)
    m_contrib = min(mention_score, conf.mention_cap)
    s_contrib = min(structural_score, conf.structural_cap)
    p_contrib = min(prior_score, conf.prior_cap)
    total += kw_contrib + m_contrib + s_contrib + p_contrib
    result = max(0.0, min(1.0, total))
    logger.debug(
        "Confidence calc: base=%.3f + kw=%.4f(cap=%.2f) + mention=%.4f(cap=%.2f) "
        "+ struct=%.4f(cap=%.2f) + prior=%.4f(cap=%.2f) = %.4f -> clamped=%.4f",
        conf.base,
        kw_contrib, conf.keyword_cap,
        m_contrib, conf.mention_cap,
        s_contrib, conf.structural_cap,
        p_contrib, conf.prior_cap,
        total, result,
    )
    return result


def _resolve_processing_tier(
    publisher_id: str | None,
    config: Config,
) -> str | None:
    """Map a publisher_id to its processing_tier via config."""
    if publisher_id is None:
        return None
    publisher = config.get_publisher(publisher_id)
    if publisher is None:
        logger.debug(
            "Publisher '%s' not found in config; no processing tier",
            publisher_id,
        )
        return None
    return publisher.processing_tier


def validate_taxonomy_config(taxonomy: Taxonomy) -> None:  # noqa: C901
    """
    Emit warnings for taxonomy configuration patterns likely to cause false positives, over-classification, or undiagnosable rejections.

    Called once at stage startup, before any documents are processed.
    All issues are logged as warnings -- nothing is blocked.
    """
    settings = taxonomy.classification_settings
    if settings is None:
        return

    conf = settings.confidence
    thresholds = settings.thresholds
    warnings_emitted = 0

    # -- Check 1: Short high_signal keywords -----------------------------------
    # A short keyword like "Gas" (3 chars) will match ubiquitously in
    # energy-sector text, inflating scores for unrelated topics.
    for topic_id, topic in sorted(taxonomy.topics.items()):
        for lang, kw_group in sorted(topic.keywords.items()):
            for kw in kw_group.high_signal:
                if len(kw) <= SHORT_KEYWORD_WARN_THRESHOLD:
                    logger.warning(
                        "CONFIG: topic '%s' has short high_signal keyword "
                        "'%s' (%s, %d chars) -- likely to cause false "
                        "positives; consider moving to medium_signal or "
                        "requiring compound context",
                        topic_id, kw, lang, len(kw),
                    )
                    warnings_emitted += 1

    # -- Check 2: Publisher priors that auto-pass threshold ---------------------
    # If base + capped_prior >= topic_assignment threshold, the topic is
    # guaranteed to pass for that tier with zero keyword evidence.
    for tier_name, priors in sorted(taxonomy.publisher_priors.items()):
        for topic_id, prior_val in sorted(priors.items()):
            capped_prior = min(prior_val, conf.prior_cap)
            effective_threshold = thresholds.topic_assignment
            topic = taxonomy.topics.get(topic_id)
            if topic is not None:
                effective_threshold = max(
                    effective_threshold, topic.threshold
                )
                if topic.min_confidence is not None:
                    effective_threshold = max(
                        effective_threshold, topic.min_confidence
                    )
            floor = conf.base + capped_prior
            if floor >= effective_threshold:
                logger.warning(
                    "CONFIG: tier '%s' + topic '%s': base(%.2f) + "
                    "prior(%.2f, capped=%.2f) = %.3f >= threshold(%.3f) -- "
                    "topic will pass without any keyword evidence",
                    tier_name, topic_id,
                    conf.base, prior_val, capped_prior,
                    floor, effective_threshold,
                )
                warnings_emitted += 1
            elif floor + settings.keyword_weights.medium_signal >= effective_threshold:
                logger.info(
                    "CONFIG NOTE: tier '%s' + topic '%s': base(%.2f) + "
                    "prior(%.2f) = %.3f -- a single medium_signal keyword "
                    "match (%.3f) reaches threshold(%.3f)",
                    tier_name, topic_id,
                    conf.base, capped_prior, floor,
                    settings.keyword_weights.medium_signal,
                    effective_threshold,
                )

    # -- Check 3: Large gap between threshold and min_confidence ---------------
    # When min_confidence >> global threshold, operators may not realize the
    # effective threshold is much higher than the global setting.
    for topic_id, topic in sorted(taxonomy.topics.items()):
        effective = max(topic.threshold, thresholds.topic_assignment)
        if topic.min_confidence is not None:
            final_effective = max(effective, topic.min_confidence)
            gap = final_effective - thresholds.topic_assignment
            if gap >= 0.15:
                logger.info(
                    "CONFIG NOTE: topic '%s' effective_threshold=%.3f "
                    "(global=%.3f + min_confidence=%.3f, gap=+%.3f) -- "
                    "significantly stricter than global setting",
                    topic_id, final_effective,
                    thresholds.topic_assignment,
                    topic.min_confidence, gap,
                )

    # -- Check 4: No multi-label constraint on document types ------------------
    # The config has multi_label for topics but nothing for document types.
    # Log the code-level ceiling we apply.
    logger.info(
        "Document type limit: MAX_DOCUMENT_TYPES=%d "
        "(code-level ceiling; no config equivalent of multi_label.max_topics)",
        MAX_DOCUMENT_TYPES,
    )

    if warnings_emitted > 0:
        logger.warning(
            "Config validation complete: %d warning(s) emitted -- "
            "review config.yaml taxonomy section",
            warnings_emitted,
        )
    else:
        logger.info("Config validation complete: no issues detected")


def classify_document(
    clean_content: str,
    mentions: Sequence[MentionRow],
    blocks: Sequence[BlockRow],
    publisher_id: str | None,
    taxonomy: Taxonomy,
    processing_tier: str | None = None,
) -> list[FacetScore]:
    """
    Run the full classification pipeline, returning scored facets.

    Only facets whose confidence meets the relevant threshold are returned.
    Multi-label constraints are enforced if enabled.
    """
    settings = taxonomy.classification_settings
    if settings is None:
        logger.warning("No classification_settings in taxonomy; skipping classification")
        return []

    weights = settings.keyword_weights
    results: list[FacetScore] = []

    # Log mention type distribution for input inspection
    mention_type_counts = Counter(m.mention_type for m in mentions)
    logger.debug(
        "classify_document: content_len=%d, mentions=%d (types=%s), blocks=%d, "
        "publisher_id=%s, processing_tier=%s",
        len(clean_content), len(mentions),
        dict(mention_type_counts) if mention_type_counts else "{}",
        len(blocks),
        publisher_id, processing_tier,
    )
    logger.debug(
        "Taxonomy config: %d topic(s), %d document_type(s), "
        "weights=(high=%.3f, medium=%.3f, compound=%.3f), "
        "thresholds=(topic=%.3f, doctype=%.3f, confident=%.3f, low=%.3f)",
        len(taxonomy.topics), len(taxonomy.document_types),
        weights.high_signal, weights.medium_signal, weights.compound_match_bonus,
        settings.thresholds.topic_assignment,
        settings.thresholds.document_type_assignment,
        settings.thresholds.confident_assignment,
        settings.thresholds.low_confidence_flag,
    )

    rejected_topics: list[tuple[str, float, float]] = []
    for topic_id, topic in sorted(taxonomy.topics.items()):
        kw_score, kw_hits = score_keywords(clean_content, topic.keywords, weights)
        m_score = score_mentions(mentions, topic.mention_signals)
        s_score = score_structural(blocks, topic.structural_signals)
        p_score = _publisher_prior(processing_tier, topic_id, taxonomy)

        conf = compute_confidence(kw_score, m_score, s_score, p_score, settings)

        # Use the stricter of per-topic threshold and global threshold
        effective_threshold = max(topic.threshold, settings.thresholds.topic_assignment)

        # Also enforce per-topic min_confidence if set
        if topic.min_confidence is not None:
            effective_threshold = max(effective_threshold, topic.min_confidence)

        if conf >= effective_threshold:
            # Log the distinct keywords that drove the acceptance at INFO
            distinct_kws = sorted({h.keyword.lower() for h in kw_hits})
            logger.info(
                "Topic ACCEPTED: %s confidence=%.4f >= threshold=%.4f "
                "(kw=%.4f[%s], mention=%.4f, struct=%.4f, prior=%.4f)",
                topic_id, conf, effective_threshold,
                kw_score, ", ".join(distinct_kws) if distinct_kws else "none",
                m_score, s_score, p_score,
            )
            fs = FacetScore(
                facet_type="topic",
                facet_value=topic_id,
                keyword_score=kw_score,
                mention_score=m_score,
                structural_score=s_score,
                prior_score=p_score,
                confidence=round(conf, 4),
                keyword_hits=kw_hits,
                signals={
                    "keyword": round(kw_score, 4),
                    "mention": round(m_score, 4),
                    "structural": round(s_score, 4),
                    "prior": round(p_score, 4),
                    "distinct_keywords": distinct_kws,
                },
            )
            results.append(fs)
        else:
            rejected_topics.append((topic_id, conf, effective_threshold))
            logger.debug(
                "Topic REJECTED: %s confidence=%.4f < threshold=%.4f "
                "(kw=%.4f, mention=%.4f, struct=%.4f, prior=%.4f, hits=%d)",
                topic_id, conf, effective_threshold,
                kw_score, m_score, s_score, p_score, len(kw_hits),
            )

    # Log summary of rejected topics at INFO for easier visual inspection
    if rejected_topics:
        # Show the closest misses (top 3 by confidence) for diagnostic value
        closest_misses = sorted(rejected_topics, key=lambda x: x[1], reverse=True)[:3]
        logger.info(
            "Topics rejected: %d/%d -- closest misses: %s",
            len(rejected_topics), len(taxonomy.topics),
            ", ".join(
                f"{tid}({conf:.3f}<{thr:.3f})"
                for tid, conf, thr in closest_misses
            ),
        )

    rejected_doctypes: list[tuple[str, float, float]] = []
    for dtype_id, dtype in sorted(taxonomy.document_types.items()):
        kw_score, kw_hits = _score_doctype_keywords(clean_content, dtype, weights)
        m_score = score_mentions(mentions, dtype.mention_signals)
        s_score = score_structural(blocks, dtype.structural_signals)

        conf = compute_confidence(kw_score, m_score, s_score, 0.0, settings)
        effective_threshold = max(
            dtype.threshold, settings.thresholds.document_type_assignment
        )

        if conf >= effective_threshold:
            distinct_kws = sorted({h.keyword.lower() for h in kw_hits})
            logger.info(
                "DocType ACCEPTED: %s confidence=%.4f >= threshold=%.4f "
                "(kw=%.4f[%s], mention=%.4f, struct=%.4f)",
                dtype_id, conf, effective_threshold,
                kw_score, ", ".join(distinct_kws) if distinct_kws else "none",
                m_score, s_score,
            )
            fs = FacetScore(
                facet_type="document_type",
                facet_value=dtype_id,
                keyword_score=kw_score,
                mention_score=m_score,
                structural_score=s_score,
                prior_score=0.0,
                confidence=round(conf, 4),
                keyword_hits=kw_hits,
                signals={
                    "keyword": round(kw_score, 4),
                    "mention": round(m_score, 4),
                    "structural": round(s_score, 4),
                    "distinct_keywords": distinct_kws,
                },
            )
            results.append(fs)
        else:
            rejected_doctypes.append((dtype_id, conf, effective_threshold))
            logger.debug(
                "DocType REJECTED: %s confidence=%.4f < threshold=%.4f "
                "(kw=%.4f, mention=%.4f, struct=%.4f, hits=%d)",
                dtype_id, conf, effective_threshold,
                kw_score, m_score, s_score, len(kw_hits),
            )

    if rejected_doctypes:
        closest_misses = sorted(rejected_doctypes, key=lambda x: x[1], reverse=True)[:3]
        logger.info(
            "DocTypes rejected: %d/%d -- closest misses: %s",
            len(rejected_doctypes), len(taxonomy.document_types),
            ", ".join(
                f"{did}({conf:.3f}<{thr:.3f})"
                for did, conf, thr in closest_misses
            ),
        )

    logger.debug(
        "Pre-constraint results: %d facet(s) -- topics=%s, doctypes=%s",
        len(results),
        [f.facet_value for f in results if f.facet_type == "topic"],
        [f.facet_value for f in results if f.facet_type == "document_type"],
    )
    results = _apply_multi_label_constraints(results, settings)
    logger.debug(
        "Post-constraint results: %d facet(s) -- topics=%s, doctypes=%s",
        len(results),
        [f.facet_value for f in results if f.facet_type == "topic"],
        [f.facet_value for f in results if f.facet_type == "document_type"],
    )
    return results


def _publisher_prior(
    processing_tier: str | None,
    topic_id: str,
    taxonomy: Taxonomy,
) -> float:
    """
    Look up the publisher prior for a topic based on the publisher's processing tier.

    The ``publisher_priors`` config keys (``data_heavy``, ``regulatory``,
    ``infrastructure``, ``narrative``) correspond to processing tiers, not
    publisher IDs.
    """
    if processing_tier is None:
        return 0.0
    priors = taxonomy.publisher_priors.get(processing_tier, {})
    prior_val = priors.get(topic_id, 0.0)
    if prior_val > 0.0:
        logger.debug(
            "Publisher prior: tier=%s, topic=%s -> %.4f",
            processing_tier, topic_id, prior_val,
        )
    return prior_val


def _score_doctype_keywords(
    clean_content: str,
    dtype: DocumentType,
    weights: KeywordWeights,
) -> tuple[float, list[KeywordHit]]:
    """
    Score document-type keywords.

    ``DocumentType.keywords`` maps language codes to flat lists of keywords.
    We treat all of them at the *high_signal* weight.  Scoring is by
    **distinct** keyword count, consistent with ``score_keywords``.
    """
    hits: list[KeywordHit] = []
    distinct_kws: set[str] = set()
    for _lang, kw_list in sorted(dtype.keywords.items()):
        for kw in kw_list:
            for m in _build_word_pattern(kw).finditer(clean_content):
                w = weights.high_signal
                hits.append(KeywordHit(kw, m.start(), m.end(), w))
                distinct_kws.add(kw.lower())

    raw_score = len(distinct_kws) * weights.high_signal
    return raw_score, hits


def _apply_multi_label_constraints(  # noqa: C901
    facets: list[FacetScore],
    settings: ClassificationSettings,
) -> list[FacetScore]:
    """Enforce multi-label topic limits, document-type ceiling,secondary-confidence floor, and ambiguity flags."""
    ml = settings.multi_label

    # -- Topic constraints -----------------------------------------------------
    if not ml.enabled:
        topics = [f for f in facets if f.facet_type == "topic"]
        if topics:
            best = max(topics, key=lambda f: f.confidence)
            if len(topics) > 1:
                logger.debug(
                    "Multi-label disabled: keeping best topic '%s' (%.4f), "
                    "dropping %d other topic(s): %s",
                    best.facet_value, best.confidence, len(topics) - 1,
                    [(t.facet_value, t.confidence) for t in topics if t is not best],
                )
            facets = [f for f in facets if f.facet_type != "topic"] + [best]
    else:
        topics = sorted(
            [f for f in facets if f.facet_type == "topic"],
            key=lambda f: f.confidence,
            reverse=True,
        )

        kept: list[FacetScore] = []
        for i, t in enumerate(topics):
            if i == 0:
                kept.append(t)
            elif i < ml.max_topics and t.confidence >= ml.min_secondary_confidence:
                kept.append(t)
            else:
                logger.debug(
                    "Multi-label dropped topic '%s' (conf=%.4f): %s",
                    t.facet_value, t.confidence,
                    "exceeded max_topics" if i >= ml.max_topics
                    else f"below min_secondary_confidence={ml.min_secondary_confidence}",
                )

        if len(topics) > len(kept):
            logger.debug(
                "Multi-label constraint: %d -> %d topic(s), kept=%s",
                len(topics), len(kept),
                [(t.facet_value, t.confidence) for t in kept],
            )

        # Flag ambiguity: if top two topics are within confidence_gap_threshold
        ambiguity = settings.ambiguity
        if len(topics) >= 2:
            gap = topics[0].confidence - topics[1].confidence
            if gap < ambiguity.confidence_gap_threshold:
                logger.info(
                    "Ambiguity detected: top topics '%s' (%.4f) and '%s' (%.4f), "
                    "gap=%.4f < threshold=%.4f",
                    topics[0].facet_value, topics[0].confidence,
                    topics[1].facet_value, topics[1].confidence,
                    gap, ambiguity.confidence_gap_threshold,
                )
                for t in kept:
                    t.signals["ambiguous"] = True
                    t.signals["confidence_gap"] = round(gap, 4)

        # Flag unfocused: too many topics assigned
        if len(kept) > ambiguity.max_before_unfocused_flag:
            for t in kept:
                t.signals["unfocused"] = True
            logger.info(
                "Unfocused classification: %d topics assigned (max recommended %d)",
                len(kept),
                ambiguity.max_before_unfocused_flag,
            )

        non_topics = [f for f in facets if f.facet_type != "topic"]
        facets = kept + non_topics

    # -- Document type constraint ----------------------------------------------
    # There is no config-level multi_label equivalent for document types.
    # Apply a code-level ceiling (MAX_DOCUMENT_TYPES) and keep the highest
    # confidence selections.
    doctypes = sorted(
        [f for f in facets if f.facet_type == "document_type"],
        key=lambda f: f.confidence,
        reverse=True,
    )
    if len(doctypes) > MAX_DOCUMENT_TYPES:
        dropped = doctypes[MAX_DOCUMENT_TYPES:]
        kept_dtypes = doctypes[:MAX_DOCUMENT_TYPES]
        logger.warning(
            "Document type over-assignment: %d doctypes passed threshold, "
            "capped to %d -- kept=%s, dropped=%s",
            len(doctypes), MAX_DOCUMENT_TYPES,
            [(d.facet_value, d.confidence) for d in kept_dtypes],
            [(d.facet_value, d.confidence) for d in dropped],
        )
        for d in kept_dtypes:
            d.signals["doctype_capped"] = True
        non_doctypes = [f for f in facets if f.facet_type != "document_type"]
        facets = non_doctypes + kept_dtypes

    return facets


def select_evidence_span(
    keyword_hits: list[KeywordHit],
    clean_content: str,
    context_chars: int = EVIDENCE_CONTEXT_CHARS,
) -> tuple[int, int] | None:
    """
    Pick the best evidence span from keyword hits.

    Selects the hit with the highest weight (first by weight desc, then
    position asc for determinism) and expands it with *context_chars* on each
    side, clamped to word boundaries where feasible.

    :param context_chars: Characters of context around the best hit.
        Defaults to module-level ``EVIDENCE_CONTEXT_CHARS``.
    :returns: ``(span_start, span_end)`` or ``None`` if no hits.
    """
    if not keyword_hits:
        return None

    best = sorted(keyword_hits, key=lambda h: (-h.weight, h.start))[0]
    raw_start = max(0, best.start - context_chars)
    raw_end = min(len(clean_content), best.end + context_chars)

    span_start = _snap_to_boundary(clean_content, raw_start, direction="left")
    span_end = _snap_to_boundary(clean_content, raw_end, direction="right")

    if span_end <= span_start:
        span_start, span_end = raw_start, raw_end

    logger.debug(
        "Evidence span selected: best_keyword='%s' (weight=%.3f, pos=%d), "
        "context_chars=%d, span=[%d:%d] (%d chars)",
        best.keyword, best.weight, best.start,
        context_chars, span_start, span_end, span_end - span_start,
    )

    return span_start, span_end


def _snap_to_boundary(text: str, pos: int, direction: str) -> int:
    """
    Snap *pos* outward to the nearest whitespace boundary.

    For ``direction="left"``, move *pos* leftward until the character at
    ``pos`` is the start of a word (i.e., preceded by whitespace or at
    position 0).

    For ``direction="right"``, move *pos* rightward until the character at
    ``pos`` is at the end of a word (i.e., at whitespace or end of text).
    """
    if direction == "left":
        while pos > 0 and not text[pos - 1].isspace():
            pos -= 1
        return pos
    else:
        while pos < len(text) and not text[pos].isspace():
            pos += 1
        return pos


def process_document(
    doc_version_id: str,
    db: Stage06DatabaseInterface,
    taxonomy: Taxonomy,
    run_id: str,
    config_hash: str,
    config: Config,
) -> None:
    """
    Classify a single document and persist results.

    This is the per-document transaction body.  The caller is responsible
    for wrapping it in ``db.transaction()``.

    :param config: Required -- used to resolve publisher processing tier for
        publisher-prior lookups.
    """
    # Idempotency: clean up any partial results from a prior failed attempt
    deleted = db.delete_facets_for_doc(doc_version_id)
    if deleted > 0:
        logger.debug(
            "Cleaned up %d stale facet(s) for %s before re-processing",
            deleted,
            doc_version_id[:16],
        )

    dv = db.get_document_version(doc_version_id)
    if dv is None:
        raise DBError(f"document_version not found: {doc_version_id}")

    clean_content = dv.clean_content
    publisher_id = db.get_document_publisher_id(doc_version_id)
    blocks = db.get_blocks_by_doc_version_id(doc_version_id)
    mentions = db.get_mentions_by_doc_version_id(doc_version_id)

    # Resolve processing tier for publisher prior lookup
    processing_tier = _resolve_processing_tier(publisher_id, config)

    logger.info(
        "Processing doc %s: content_len=%d, publisher=%s, tier=%s, "
        "blocks=%d, mentions=%d",
        doc_version_id[:16],
        len(clean_content) if clean_content else 0,
        publisher_id, processing_tier,
        len(blocks), len(mentions),
    )
    if not clean_content:
        logger.warning(
            "Document %s has empty clean_content; recording as skipped",
            doc_version_id[:16],
        )
        db.upsert_doc_stage_status(
            doc_version_id=doc_version_id,
            stage=STAGE_NAME,
            run_id=run_id,
            config_hash=config_hash,
            status="skipped",
            error_message="empty_clean_content",
        )
        return

    facets = classify_document(
        clean_content, mentions, blocks, publisher_id, taxonomy,
        processing_tier=processing_tier,
    )

    if not facets:
        logger.info("No facets assigned for %s", doc_version_id[:16])

    evidence_count = 0
    for fs in facets:
        facet_id = compute_sha256_id(doc_version_id, fs.facet_type, fs.facet_value)

        # Add confidence-level flags to signals
        if fs.confidence >= taxonomy.classification_settings.thresholds.confident_assignment:
            fs.signals["confidence_level"] = "confident"
        elif fs.confidence < taxonomy.classification_settings.thresholds.low_confidence_flag:
            fs.signals["confidence_level"] = "low"

        fa_row = FacetAssignmentRow(
            facet_id=facet_id,
            doc_version_id=doc_version_id,
            facet_type=fs.facet_type,
            facet_value=fs.facet_value,
            confidence=fs.confidence,
            signals_json=fs.signals,
            created_in_run_id=run_id,
        )
        db.insert_facet_assignment(fa_row)
        logger.debug(
            "Persisted facet: %s:%s confidence=%.4f, confidence_level=%s, "
            "facet_id=%s",
            fs.facet_type, fs.facet_value, fs.confidence,
            fs.signals.get("confidence_level", "normal"),
            facet_id[:16],
        )

        span = select_evidence_span(fs.keyword_hits, clean_content)
        if span is not None:
            span_start, span_end = span
            span_text = clean_content[span_start:span_end]
            logger.debug(
                "Evidence span for %s:%s -- [%d:%d] (%d chars): '%.80s%s'",
                fs.facet_type, fs.facet_value,
                span_start, span_end, span_end - span_start,
                span_text, "..." if len(span_text) > 80 else "",
            )
            ev = db.get_or_create_evidence_span(
                doc_version_id=doc_version_id,
                span_start=span_start,
                span_end=span_end,
                run_id=run_id,
                purpose=f"taxonomy:{fs.facet_type}:{fs.facet_value}",
                clean_content=clean_content,
            )
            db.insert_facet_assignment_evidence(
                FacetAssignmentEvidenceRow(
                    facet_id=facet_id,
                    evidence_id=ev.evidence_id,
                    purpose=f"keyword_evidence:{fs.facet_value}",
                )
            )
            evidence_count += 1
        else:
            logger.debug(
                "No evidence span for %s:%s (no keyword hits)",
                fs.facet_type, fs.facet_value,
            )

    logger.info(
        "Doc %s classification complete: %d facet(s) assigned "
        "(%d topic, %d doctype), %d evidence span(s), "
        "assignments=[%s]",
        doc_version_id[:16], len(facets),
        sum(1 for f in facets if f.facet_type == "topic"),
        sum(1 for f in facets if f.facet_type == "document_type"),
        evidence_count,
        ", ".join(
            f"{f.facet_type}:{f.facet_value}({f.confidence:.3f})"
            for f in facets
        ) or "none",
    )

    db.upsert_doc_stage_status(
        doc_version_id=doc_version_id,
        stage=STAGE_NAME,
        run_id=run_id,
        config_hash=config_hash,
        status="ok",
        details=f"facets_assigned={len(facets)}",
    )


def run_stage(  # noqa: C901
    db: Stage06DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> int:
    """
    Execute Stage 06 over all eligible documents.

    :returns: 0 on success, 1 on fatal / systemic failure.
    """
    taxonomy = config.taxonomy
    if taxonomy.classification_settings is None:
        logger.warning(
            "taxonomy.classification_settings is not configured; "
            "all documents will receive status='skipped'"
        )
    else:
        cs = taxonomy.classification_settings
        logger.info(
            "Classification config: topics=%d, document_types=%d, "
            "multi_label=%s (max_topics=%d), publisher_prior_tiers=%s",
            len(taxonomy.topics), len(taxonomy.document_types),
            cs.multi_label.enabled, cs.multi_label.max_topics,
            sorted(taxonomy.publisher_priors.keys()) if taxonomy.publisher_priors else "[]",
        )
        logger.info(
            "Thresholds: topic_assignment=%.3f, doctype_assignment=%.3f, "
            "confident=%.3f, low_flag=%.3f",
            cs.thresholds.topic_assignment,
            cs.thresholds.document_type_assignment,
            cs.thresholds.confident_assignment,
            cs.thresholds.low_confidence_flag,
        )
        logger.info(
            "Constants: EVIDENCE_CONTEXT_CHARS=%d, "
            "SHORT_KEYWORD_WARN_THRESHOLD=%d, MAX_DOCUMENT_TYPES=%d",
            EVIDENCE_CONTEXT_CHARS,
            SHORT_KEYWORD_WARN_THRESHOLD,
            MAX_DOCUMENT_TYPES,
        )
        # Run startup config validation to surface likely scoring problems
        validate_taxonomy_config(taxonomy)

    iteration_set = db.get_iteration_set()
    logger.info(
        "Stage 06 iteration set: %d document(s) to evaluate", len(iteration_set)
    )

    attempted = 0
    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0

    # Track facet frequencies across the entire run for the summary
    topic_counter: Counter = Counter()
    doctype_counter: Counter = Counter()

    for dvid in iteration_set:
        prereqs_ok, block_msg = db.check_prerequisites(dvid)

        if not prereqs_ok:
            logger.debug(
                "Doc %s blocked: %s", dvid[:16], block_msg,
            )
            existing = db.get_doc_stage_status(dvid, STAGE_NAME)
            if (
                existing is not None
                and existing.status == "blocked"
                and existing.error_message == block_msg
            ):
                blocked_count += 1
                continue
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=dvid,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="blocked",
                    error_message=block_msg,
                )
            blocked_count += 1
            continue

        if taxonomy.classification_settings is None:
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=dvid,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="skipped",
                    error_message="no_classification_settings",
                )
            skipped_count += 1
            continue

        attempted += 1
        try:
            with db.transaction():
                process_document(dvid, db, taxonomy, run_id, config_hash, config=config)
            ok_count += 1

            # Collect facet assignments for this doc for the run summary
            doc_facets = db.get_facet_assignments_for_doc(dvid)
            for fa in doc_facets:
                if fa.facet_type == "topic":
                    topic_counter[fa.facet_value] += 1
                elif fa.facet_type == "document_type":
                    doctype_counter[fa.facet_value] += 1

        except Exception:
            logger.exception("Failed to classify doc %s", dvid[:16])
            try:
                with db.transaction():
                    db.upsert_doc_stage_status(
                        doc_version_id=dvid,
                        stage=STAGE_NAME,
                        run_id=run_id,
                        config_hash=config_hash,
                        status="failed",
                        error_message="classification_error",
                    )
            except Exception:
                logger.exception(
                    "Could not write failed status for %s", dvid[:16]
                )
            failed_count += 1

    logger.info(
        "Stage 06 complete: ok=%d failed=%d blocked=%d skipped=%d "
        "(total_evaluated=%d, success_rate=%.1f%%)",
        ok_count,
        failed_count,
        blocked_count,
        skipped_count,
        len(iteration_set),
        (ok_count / attempted * 100) if attempted > 0 else 0.0,
    )

    # -- Run-level distribution summary ----------------------------------------
    # Exposes systematic over-assignment: if a topic appears on >50% of docs
    # it's likely a false-positive problem in the keywords or priors.
    if ok_count > 0:
        logger.info(
            "Topic distribution (across %d ok docs): %s",
            ok_count,
            ", ".join(
                f"{tid}={cnt}({cnt/ok_count*100:.0f}%%)"
                for tid, cnt in topic_counter.most_common()
            ) or "none",
        )
        logger.info(
            "DocType distribution (across %d ok docs): %s",
            ok_count,
            ", ".join(
                f"{did}={cnt}({cnt/ok_count*100:.0f}%%)"
                for did, cnt in doctype_counter.most_common()
            ) or "none",
        )

        # Flag topics that appear on a suspicious proportion of documents
        for tid, cnt in topic_counter.items():
            pct = cnt / ok_count
            if pct > 0.50 and ok_count >= 10:
                logger.warning(
                    "DISTRIBUTION ANOMALY: topic '%s' assigned to %d/%d "
                    "(%.0f%%) documents -- likely false-positive driver; "
                    "review keywords and publisher priors",
                    tid, cnt, ok_count, pct * 100,
                )

    if attempted > 0 and failed_count == attempted and ok_count == 0 and skipped_count == 0:
        logger.error("Systemic failure: all %d attempted documents failed", attempted)
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 06: Taxonomy Classification")
    parser.add_argument(
        "--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", help="Pipeline run ID"
    )
    parser.add_argument("--config-dir", type=Path, default=Path("../../../config/"))
    parser.add_argument(
        "--source-db", type=Path, default=Path("../../../database/preprocessed_posts.db")
    )
    parser.add_argument(
        "--working-db", type=Path, default=Path("../../../database/processed_posts.db")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("../../../output/processed/logs/")
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main_stage_06_taxonomy() -> int:
    """
    Entry point for Stage 06.

    :returns: 0 on success, 1 on fatal error.
    """
    args = parse_args()

    config_path = args.config_dir / "config.yaml"
    config = load_config(config_path)
    config_hash = get_config_version(config)
    run_id = args.run_id

    logger.info("Stage 06 starting: run_id=%s", run_id)
    logger.info(
        "Config: path=%s, config_hash=%s", config_path, config_hash[:16],
    )
    logger.info(
        "Database: working_db=%s, source_db=%s",
        args.working_db, args.source_db,
    )

    db = Stage06DatabaseInterface(
        working_db_path=args.working_db,
        source_db_path=args.source_db if args.source_db.exists() else None,
    )
    try:
        db.open()
        run_row = db.get_pipeline_run(run_id)
        if run_row is None:
            logger.error("pipeline_run row not found for run_id=%s", run_id)
            return 1
        if run_row.status != "running":
            logger.error("pipeline_run status is '%s', expected 'running'", run_row.status)
            return 1

        return run_stage(db, config, run_id, config_hash)
    except Exception:
        logger.exception("Stage 06 fatal error")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main_stage_06_taxonomy())