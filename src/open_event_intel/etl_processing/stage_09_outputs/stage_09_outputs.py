"""
stage_09_outputs.py — Run-scoped materialization of surfaced outputs.

Turns internal pipeline artifacts (events, metric observations, facet
assignments, novelty labels) into the analyst-facing surfaced tables:
``metric_series`` / ``metric_series_point``, ``alert`` / ``alert_evidence``,
and ``digest_item`` / ``digest_item_evidence``.

**Reads:** `event`, `event_revision`, `novelty_label`, `mention`, `facet_assignment`, `metric_observation`, config tables.
**Writes (run-scoped):** `metric_series`, `metric_series_point`, `alert`, `alert_evidence`, `digest_item`, `digest_item_evidence`, `evidence_span`, `run_stage_status(stage_09_outputs)`.
**Constraints:** Uses only `current_revision_id`; restricts to `eligible_docs` and `eligible_events`.
**Metric point rule (normative):** Stage 9 MAY emit multiple `metric_series_point` rows for the same `(series_id, period_start)`
as long as `source_doc_version_id` differs and `evidence_id` is present for each row (no silent overwrites).

**Responsibility**:
  * run-scoped "materialization" stage that turns the pipeline's internal, provenance-anchored understanding (events + metric observations + classifications) into the analyst-facing surfaced tables: metric time series, alerts, and digest items—all restricted to eligible documents/events and evidence-linked.
  * Produce surfaced, query-ready outputs; Enforce "current truth" semantics; Fail-closed provenance; Run-scoped rebuild
  * Core processing: Eligibility gating; Run-scoped rebuild mechanics (deterministic, atomic); Metric series materialization (structured → time series); Alert generation (rules over structured state, not free-form text)

"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from open_event_intel.etl_processing.config_interface import (
    VALID_EVENT_TYPE_NAMES,
    Config,
    get_config_version,
    load_config,
)
from open_event_intel.etl_processing.database_interface import (
    AlertEvidenceRow,
    AlertRow,
    AlertRuleRow,
    DigestItemEvidenceRow,
    DigestItemRow,
    EventRevisionRow,
    EventRow,
    MetricObservationRow,
    MetricSeriesPointRow,
    MetricSeriesRow,
    WatchlistRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_09_outputs.database_stage_09_outputs import STAGE_NAME, EligibleEvent, Stage09DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Module-level constants — no silently set values in business logic

# Purpose strings for evidence linkage (used in join tables).
PURPOSE_ALERT_TRIGGER = "alert_trigger"
PURPOSE_WATCHLIST_TRIGGER = "watchlist_trigger"
PURPOSE_DIGEST_CONTENT = "digest_content"

# Default digest section for event types that don't match any mapping.
DEFAULT_DIGEST_SECTION = "general"

# Digest section labels.
DIGEST_SECTION_REGULATORY = "regulatory"
DIGEST_SECTION_INFRASTRUCTURE = "infrastructure"
DIGEST_SECTION_MARKETS = "markets"

# Keywords used to classify event types into digest sections from config descriptions.
# Extracted here so they are visible and auditable.
DIGEST_SECTION_KEYWORDS: dict[str, list[str]] = {
    DIGEST_SECTION_REGULATORY: ["regulat", "consultation", "network code"],
    DIGEST_SECTION_INFRASTRUCTURE: ["project", "infrastructure", "milestone", "grid"],
    DIGEST_SECTION_MARKETS: ["price", "market", "generation", "energy"],
}

# Fallback digest section mapping when config is unavailable.
# Kept only as safety net; normal operation uses _build_digest_section_mapping().
_DIGEST_SECTION_FALLBACK: dict[str, str] = {
    "regulatory_decision": DIGEST_SECTION_REGULATORY,
    "consultation_opened": DIGEST_SECTION_REGULATORY,
    "project_milestone": DIGEST_SECTION_INFRASTRUCTURE,
    "network_code_amendment": DIGEST_SECTION_REGULATORY,
    "price_movement": DIGEST_SECTION_MARKETS,
    "generation_record": DIGEST_SECTION_MARKETS,
}


def _build_event_type_alias_map() -> dict[str, str]:
    """
    Build event-type alias map from VALID_EVENT_TYPE_NAMES.

    Config alert rules use tokens like ``CONSULTATION`` or ``REGULATORY_DECISION``
    while the events table uses canonical names like ``consultation_opened`` and
    ``regulatory_decision``.  This map normalizes both directions.

    The map is built from the canonical names in config_interface plus known
    short-form aliases used in alert rule configs.
    """
    alias_map: dict[str, str] = {}
    # Identity mappings for all canonical event type names
    for name in VALID_EVENT_TYPE_NAMES:
        alias_map[name] = name

    # Short-form aliases used in config alert rules (Section 5)
    # These map the config's trigger event_type tokens to canonical names.
    _config_aliases: dict[str, str] = {
        "consultation": "consultation_opened",
    }
    alias_map.update(_config_aliases)
    return alias_map


# Built once at module load from config_interface's VALID_EVENT_TYPE_NAMES.
_EVENT_TYPE_ALIAS: dict[str, str] = _build_event_type_alias_map()


# Pure business logic (no I/O — testable in isolation)
class _SeriesKey(BaseModel):
    """Natural key for a metric series (used for grouping observations)."""

    model_config = ConfigDict(frozen=True)

    metric_name: str
    geography: str | None
    period_granularity: str
    unit_norm: str | None


def _compute_series_id(key: _SeriesKey) -> str:
    return compute_sha256_id(
        key.metric_name,
        key.geography or "",
        key.period_granularity,
        key.unit_norm or "",
    )


def materialize_metric_series(  # noqa: C901
    observations: Sequence[MetricObservationRow],
    run_id: str,
) -> tuple[list[MetricSeriesRow], list[MetricSeriesPointRow]]:
    """
    Build metric_series and metric_series_point rows from eligible observations.

    Observations without ``period_start``, ``value_norm``, or
    ``period_granularity`` are skipped.
    Multiple points for the same ``(series_id, period_start)`` are allowed
    as long as ``source_doc_version_id`` differs (spec §6.6 Stage 9 metric rule).

    :param observations: Eligible metric observations (already filtered for evidence).
    :param run_id: Current pipeline run ID.
    :return: Tuple of (series rows, point rows).
    """
    # Log sample of first few observations for debugging
    if observations:
        logger.info("materialize_metric_series: inspecting first 3 observations:")
        for i, obs in enumerate(observations[:3]):
            logger.info(
                "  obs[%d]: metric_id=%s..%s metric_name=%s period_start=%s "
                "period_end=%s period_granularity=%s value_norm=%s unit_norm=%s geography=%s",
                i, obs.metric_id[:8], obs.metric_id[-4:], obs.metric_name,
                obs.period_start, obs.period_end, obs.period_granularity,
                obs.value_norm, obs.unit_norm, obs.geography,
            )

    grouped: dict[_SeriesKey, list[MetricObservationRow]] = defaultdict(list)
    skipped = 0
    skip_reasons: dict[str, int] = defaultdict(int)
    for obs in observations:
        if obs.period_start is None or obs.value_norm is None:
            skipped += 1
            if obs.period_start is None and obs.value_norm is None:
                skip_reasons["missing_both_period_start_and_value_norm"] += 1
            elif obs.period_start is None:
                skip_reasons["missing_period_start"] += 1
            else:
                skip_reasons["missing_value_norm"] += 1
            logger.debug(
                "materialize: skip obs %s — period_start=%s value_norm=%s",
                obs.metric_id, obs.period_start, obs.value_norm,
            )
            continue
        if obs.period_granularity is None:
            skipped += 1
            skip_reasons["missing_period_granularity"] += 1
            logger.debug(
                "materialize: skip obs %s — missing period_granularity", obs.metric_id,
            )
            continue
        key = _SeriesKey(
            metric_name=obs.metric_name,
            geography=obs.geography,
            period_granularity=obs.period_granularity,
            unit_norm=obs.unit_norm,
        )
        grouped[key].append(obs)

    if skipped:
        logger.info(
            "materialize_metric_series: skipped %d/%d observations "
            "(missing period_start/value_norm/granularity), breakdown: %s",
            skipped, len(observations), dict(skip_reasons),
        )

    series_rows: list[MetricSeriesRow] = []
    point_rows: list[MetricSeriesPointRow] = []
    seen_point_keys: set[tuple[str, str, str]] = set()
    duplicate_count = 0

    for key in sorted(grouped, key=lambda k: (k.metric_name, k.geography or "", k.period_granularity)):
        series_id = _compute_series_id(key)
        series_rows.append(
            MetricSeriesRow(
                run_id=run_id,
                series_id=series_id,
                metric_name=key.metric_name,
                geography=key.geography,
                period_granularity=key.period_granularity,
                unit_norm=key.unit_norm,
            )
        )
        for obs in sorted(grouped[key], key=lambda o: (o.period_start, o.doc_version_id)):  # type: ignore[arg-type]
            assert obs.period_start is not None
            assert obs.value_norm is not None
            assert obs.evidence_id is not None
            point_key = (series_id, obs.period_start.isoformat(), obs.doc_version_id)
            if point_key in seen_point_keys:
                duplicate_count += 1
                logger.debug(
                    "Duplicate (series, period, doc) skipped: series=%s..%s period=%s doc=%s..%s",
                    series_id[:8], series_id[-4:],
                    obs.period_start,
                    obs.doc_version_id[:8], obs.doc_version_id[-4:],
                )
                continue
            seen_point_keys.add(point_key)
            point_rows.append(
                MetricSeriesPointRow(
                    run_id=run_id,
                    series_id=series_id,
                    period_start=obs.period_start,
                    period_end=obs.period_end,
                    value_norm=obs.value_norm,
                    source_doc_version_id=obs.doc_version_id,
                    evidence_id=obs.evidence_id,
                )
            )

    logger.info(
        "materialize_metric_series: %d series, %d points from %d observations "
        "(%d unique series keys, %d duplicate points skipped)",
        len(series_rows), len(point_rows), len(observations), len(grouped),
        duplicate_count,
    )
    return series_rows, point_rows


# ------------------------------------------------------------------ alerts

def _normalize_event_type(raw: str) -> str:
    """
    Normalize event type for case-insensitive comparison.

    Config alert rules may use UPPER_CASE (``REGULATORY_DECISION``) while
    ``EventTypeName`` uses lower_case (``regulatory_decision``).  This
    function normalises to lowercase so both conventions match.
    """
    return raw.strip().lower()


def _event_type_matches(event_type: str, rule_types: list) -> bool:
    """
    Return True if *event_type* matches any entry in *rule_types*.

    Handles both case differences (``REGULATORY_DECISION`` vs
    ``regulatory_decision``) and alias differences (``CONSULTATION`` vs
    ``consultation_opened``) that exist between config alert rules and the
    canonical ``EventTypeName`` values.
    """
    norm_event = _normalize_event_type(event_type)
    for rt in rule_types:
        norm_rule = _normalize_event_type(rt)
        # Direct match (after lowering)
        if norm_event == norm_rule:
            return True
        # Alias-resolved match
        canonical = _EVENT_TYPE_ALIAS.get(norm_rule)
        if canonical and canonical == norm_event:
            return True
    return False


def _get_config_alert_rule_metadata(
    config: Config, rule_id: str,
) -> tuple[str, list[str], int]:
    """
    Retrieve urgency, channels, and suppression_window from config alert rule.

    :return: (urgency, channels, suppression_window_hours)
    """
    cfg_rule = config.get_alert_rule(rule_id)
    if cfg_rule is not None:
        return (
            cfg_rule.urgency,
            cfg_rule.channels,
            cfg_rule.suppression_window_hours,
        )
    # Fallback defaults from AlertRule model defaults in config_interface
    return ("normal", [], 24)


def _event_matches_alert_rule(  # noqa: C901
    event: EligibleEvent,
    rule: AlertRuleRow,
    doc_topics: set[str] | None = None,
) -> bool:
    """
    Check whether an eligible event matches a DB alert rule's conditions.

    The ``conditions_json`` stored in the DB by Stage 01 (seeded from config)
    is a list of condition dicts with ``field``, ``operator``, ``value``/``values``
    keys, plus an optional ``event_types`` list, ``topics`` list, ``keywords``
    list, and ``condition_logic`` flag.

    **FIX (v3):** now handles:
    * case-insensitive + alias-aware event-type matching
    * ``topics`` trigger (matched against document facet assignments)
    * ``keywords`` trigger (matched against event slots text)
    """
    conds = rule.conditions_json
    if isinstance(conds, dict):
        conds = [conds]
    if not isinstance(conds, list):
        return False

    event_type = event.event.event_type
    slots = event.revision.slots_json

    for cond_block in conds:
        if not isinstance(cond_block, dict):
            continue

        # ---- event_types filter (case-insensitive + alias-aware) ----
        evt_types = cond_block.get("event_types")
        if evt_types and not _event_type_matches(event_type, evt_types):
            continue

        # ---- topics filter ----
        rule_topics = cond_block.get("topics")
        if rule_topics:
            if doc_topics is None or not doc_topics:
                continue  # rule requires topics but none available
            if not (doc_topics & {t.lower() for t in rule_topics}):
                continue

        # ---- keywords filter (matched against slot values) ----
        rule_keywords = cond_block.get("keywords")
        if rule_keywords:
            slots_text = " ".join(
                str(v) for v in slots.values() if v is not None
            ).lower()
            if not any(kw.lower() in slots_text for kw in rule_keywords):
                continue

        conditions = cond_block.get("conditions", [])
        logic = cond_block.get("condition_logic", "AND")

        if not conditions:
            # No sub-conditions; passing filters above suffices
            # (at least one of event_types / topics / keywords matched)
            if evt_types or rule_topics or rule_keywords:
                return True
            continue

        results = []
        for c in conditions:
            field = c.get("field", "")
            op = c.get("operator", "")
            val = c.get("value")
            vals = c.get("values", [])
            slot_val = slots.get(field)
            if slot_val is None:
                results.append(False)
                continue
            if op == "in":
                results.append(slot_val in vals)
            elif op == "eq":
                results.append(slot_val == val)
            elif op == "gt":
                try:
                    results.append(float(slot_val) > float(val))
                except (TypeError, ValueError):
                    results.append(False)
            elif op == "lt":
                try:
                    results.append(float(slot_val) < float(val))
                except (TypeError, ValueError):
                    results.append(False)
            elif op == "within_days":
                try:
                    target = date.fromisoformat(str(slot_val))
                    delta = (target - date.today()).days
                    results.append(0 <= delta <= int(val))
                except (TypeError, ValueError):
                    results.append(False)
            else:
                logger.debug(
                    "Unknown operator '%s' in alert rule %s condition, treating as False",
                    op, rule.rule_id[:16],
                )
                results.append(False)

        if logic == "OR":
            if any(results):
                return True
        else:
            if results and all(results):
                return True

    return False


def generate_alerts(  # noqa: C901
    eligible_events: Sequence[EligibleEvent],
    alert_rules: Sequence[AlertRuleRow],
    run_id: str,
    db: Stage09DatabaseInterface,
    config: Config,
    facets_by_doc: dict[str, list[str]] | None = None,
) -> tuple[list[AlertRow], list[AlertEvidenceRow]]:
    """
    Match eligible events against alert rules, producing alert + evidence rows.

    Events that match a rule but whose revision has no evidence are skipped
    (fail-closed provenance).  Urgency and channels from config are propagated
    into the alert's trigger_payload_json.

    **FIX (v3):** accepts ``facets_by_doc`` so that topic-based and keyword-
    based rules (e.g. ``grid_emergency``) can be evaluated.

    **FIX (v4):** propagates urgency and channels from config; logs per-rule
    match counts.

    :param eligible_events: Events passing the eligibility gate.
    :param alert_rules: Active alert rules from DB.
    :param run_id: Pipeline run id.
    :param db: Database interface (used to resolve evidence spans).
    :param config: Pipeline configuration (used for urgency/channels/suppression).
    :param facets_by_doc: Mapping of doc_version_id → topic facet values
        (used for topic-based alert rules).
    :return: Tuple of (alert rows, alert evidence rows).
    """
    alerts: list[AlertRow] = []
    evidence_rows: list[AlertEvidenceRow] = []
    matched = 0
    skipped_no_evidence = 0
    suppressed = 0
    per_rule_counts: dict[str, int] = defaultdict(int)

    for ee in eligible_events:
        # Collect topics across the event's source documents once per event
        doc_topics: set[str] = set()
        if facets_by_doc is not None:
            for dvid in ee.revision.doc_version_ids:
                doc_topics.update(facets_by_doc.get(dvid, []))

        for rule in alert_rules:
            if not _event_matches_alert_rule(ee, rule, doc_topics=doc_topics):
                continue

            matched += 1

            # Get config metadata for this rule (urgency, channels, suppression)
            cfg_urgency, cfg_channels, cfg_suppression_hours = (
                _get_config_alert_rule_metadata(config, rule.rule_id)
            )

            # Suppression window check: use config value, fall back to DB value
            suppression_hours = cfg_suppression_hours or rule.suppression_window_hours
            if suppression_hours > 0:
                recent = db.get_recent_alert_ids_for_rule(
                    rule.rule_id, ee.event.event_id, suppression_hours,
                )
                if recent:
                    suppressed += 1
                    logger.debug(
                        "Alert suppressed (rule=%s, event=%s): %d existing alert(s) "
                        "within %dh suppression window",
                        rule.rule_id[:16], ee.event.event_id[:16],
                        len(recent), suppression_hours,
                    )
                    continue

            # Fail-closed: evidence must exist BEFORE the alert is created
            rev_evidence_ids = db.get_revision_evidence_ids(ee.revision.revision_id)
            if not rev_evidence_ids:
                skipped_no_evidence += 1
                logger.warning(
                    "Alert match (rule=%s, event=%s) skipped: "
                    "revision %s has no evidence (fail-closed)",
                    rule.rule_id[:16], ee.event.event_id[:16],
                    ee.revision.revision_id[:16],
                )
                continue

            alert_id = compute_sha256_id(run_id, rule.rule_id, ee.event.event_id)
            alert = AlertRow(
                alert_id=alert_id,
                run_id=run_id,
                rule_id=rule.rule_id,
                triggered_at=datetime.now(timezone.utc),
                trigger_payload_json={
                    "event_type": ee.event.event_type,
                    "canonical_key": ee.event.canonical_key,
                    "slots": ee.revision.slots_json,
                    "urgency": cfg_urgency,
                    "channels": cfg_channels,
                    "severity": rule.severity,
                },
                doc_version_ids=ee.revision.doc_version_ids,
                event_ids=[ee.event.event_id],
                acknowledged=0,
            )
            alerts.append(alert)
            per_rule_counts[rule.rule_id] += 1

            logger.debug(
                "Alert generated: rule=%s event=%s..%s urgency=%s channels=%s",
                rule.rule_id, ee.event.event_id[:8], ee.event.event_id[-4:],
                cfg_urgency, cfg_channels,
            )

            for eid in dict.fromkeys(rev_evidence_ids):  # dedupe, preserving order
                evidence_rows.append(
                    AlertEvidenceRow(
                        alert_id=alert_id,
                        evidence_id=eid,
                        purpose=PURPOSE_ALERT_TRIGGER,
                    )
                )

    logger.info(
        "generate_alerts: %d rule matches → %d alerts emitted, "
        "%d skipped (no evidence), %d suppressed",
        matched, len(alerts), skipped_no_evidence, suppressed,
    )
    if per_rule_counts:
        logger.info("generate_alerts: per-rule breakdown: %s", dict(per_rule_counts))
    return alerts, evidence_rows


# --------------------------------------------------------------- watchlists

def _resolve_watchlist_entity_ids(
    watchlist: WatchlistRow,
    name_to_id: dict[str, str],
) -> frozenset[str]:
    """
    Resolve watchlist entity_values (names) to entity_ids via registry.

    Watchlist config stores human-readable names (e.g. "SuedLink", "50Hertz")
    while ``mention_link`` stores ``entity_id`` (e.g. "proj_suedlink",
    "tso_50hz").  This function bridges the gap by looking up names in the
    ``entity_registry`` (canonical_name + aliases).

    As a fallback, values that already look like entity_ids (i.e. not found
    in the name→id map) are passed through unchanged.
    """
    entity_values = watchlist.entity_values if isinstance(watchlist.entity_values, list) else []
    resolved: set[str] = set()
    unresolved: list[str] = []
    for name in entity_values:
        eid = name_to_id.get(name)
        if eid:
            resolved.add(eid)
        else:
            # Fallback: maybe the value IS already an entity_id
            resolved.add(name)
            unresolved.append(name)
    if unresolved:
        logger.debug(
            "Watchlist %s (%s): %d/%d entity names not in registry "
            "(passed through as-is): %s",
            watchlist.watchlist_id, watchlist.name,
            len(unresolved), len(entity_values),
            unresolved[:5],
        )
    return frozenset(resolved)


def _get_config_watchlist_metadata(
    config: Config, watchlist_id: str,
) -> tuple[str | None, list[str]]:
    """
    Retrieve urgency_override and channels from config watchlist.

    :return: (urgency_override or None, channels list)
    """
    if config.alerts and config.alerts.watchlists:
        wl_cfg = config.alerts.watchlists.get(watchlist_id)
        if wl_cfg is not None:
            return wl_cfg.urgency_override, wl_cfg.channels
    return None, []


def _event_matches_watchlist(  # noqa: C901
    ee: EligibleEvent,
    wl: WatchlistRow,
    resolved_entity_ids: frozenset[str],
    entity_ids_by_doc: dict[str, list[str]],
    facets_by_doc: dict[str, list[str]],
    publisher_by_doc: dict[str, str | None],
    config_publishers: list[str] | None = None,
) -> bool:
    """
    Check whether an eligible event matches a watchlist entry.

    **FIX (v3):** Uses OR logic across the watchlist's filter dimensions
    (entities, publishers, topics) so that publisher-only or topic-only
    watchlists can fire.  Entity matching now uses ``resolved_entity_ids``
    rather than raw watchlist names.  Publisher matching uses the config
    watchlist's ``publishers`` list (since ``WatchlistRow`` does not carry
    publisher info).
    """
    # --- Determine which filter dimensions are configured ---
    has_entity_filter = bool(resolved_entity_ids)
    has_publisher_filter = bool(config_publishers)
    has_no_primary_filter = not has_entity_filter and not has_publisher_filter

    # If the watchlist has no entity or publisher filter, it cannot match
    # an event (safety: avoid matching everything)
    if has_no_primary_filter:
        return False

    # --- Entity match ---
    entity_match = False
    if has_entity_filter:
        all_doc_entities: set[str] = set()
        for dvid in ee.revision.doc_version_ids:
            all_doc_entities.update(entity_ids_by_doc.get(dvid, []))
        entity_match = bool(all_doc_entities & resolved_entity_ids)

    # --- Publisher match ---
    publisher_match = False
    if has_publisher_filter:
        for dvid in ee.revision.doc_version_ids:
            pub = publisher_by_doc.get(dvid)
            if pub and pub in config_publishers:
                publisher_match = True
                break

    # At least one primary filter (entity OR publisher) must match
    if not entity_match and not publisher_match:
        return False

    # --- Event-type filter (secondary, narrows down) ---
    track_events_str = wl.track_events
    if track_events_str:
        try:
            track_events = json.loads(track_events_str) if isinstance(track_events_str, str) else track_events_str
        except (json.JSONDecodeError, TypeError):
            track_events = []
        if track_events and not _event_type_matches(ee.event.event_type, track_events):
            return False

    # --- Topic filter (secondary, narrows down) ---
    track_topics_str = wl.track_topics
    if track_topics_str:
        try:
            track_topics = json.loads(track_topics_str) if isinstance(track_topics_str, str) else track_topics_str
        except (json.JSONDecodeError, TypeError):
            track_topics = []
        if track_topics:
            all_topics: set[str] = set()
            for dvid in ee.revision.doc_version_ids:
                all_topics.update(facets_by_doc.get(dvid, []))
            if not (all_topics & set(track_topics)):
                return False

    return True


def generate_watchlist_alerts(
    eligible_events: Sequence[EligibleEvent],
    watchlists: Sequence[WatchlistRow],
    run_id: str,
    db: Stage09DatabaseInterface,
    config: Config,
    name_to_id: dict[str, str],
    entity_ids_by_doc: dict[str, list[str]],
    facets_by_doc: dict[str, list[str]],
    publisher_by_doc: dict[str, str | None],
    config_watchlist_publishers: dict[str, list[str]] | None = None,
) -> tuple[list[AlertRow], list[AlertEvidenceRow]]:
    """
    Generate alerts from watchlist matches.

    :param name_to_id: Entity-name→entity_id mapping from registry (used
        to resolve the human-readable watchlist names to IDs that
        ``mention_link`` stores).
    :param config: Pipeline configuration (used for urgency/channels).
    :param config_watchlist_publishers: Mapping of watchlist key → list of
        publisher IDs from config, since ``WatchlistRow`` does not carry
        publisher info.
    """
    alerts: list[AlertRow] = []
    evidence_rows: list[AlertEvidenceRow] = []
    matched = 0
    skipped_no_evidence = 0
    _config_pubs = config_watchlist_publishers or {}
    per_watchlist_counts: dict[str, int] = defaultdict(int)

    # Pre-resolve each watchlist's entity names → entity_ids once
    resolved_cache: dict[str, frozenset[str]] = {}
    for wl in watchlists:
        resolved = _resolve_watchlist_entity_ids(wl, name_to_id)
        resolved_cache[wl.watchlist_id] = resolved
        logger.debug(
            "Watchlist %s (%s): %d resolved entity_ids",
            wl.watchlist_id, wl.name, len(resolved),
        )

    for ee in eligible_events:
        for wl in watchlists:
            resolved_ids = resolved_cache.get(wl.watchlist_id, frozenset())
            wl_publishers = _config_pubs.get(wl.watchlist_id)
            if not _event_matches_watchlist(
                ee, wl, resolved_ids,
                entity_ids_by_doc, facets_by_doc, publisher_by_doc,
                config_publishers=wl_publishers,
            ):
                continue

            matched += 1

            # Fail-closed: evidence must exist
            rev_evidence_ids = db.get_revision_evidence_ids(ee.revision.revision_id)
            if not rev_evidence_ids:
                skipped_no_evidence += 1
                logger.debug(
                    "Watchlist alert (wl=%s, event=%s) skipped: no evidence",
                    wl.watchlist_id, ee.event.event_id[:16],
                )
                continue

            # Get config metadata for urgency/channels
            cfg_urgency_override, cfg_channels = _get_config_watchlist_metadata(
                config, wl.watchlist_id,
            )

            alert_id = compute_sha256_id(run_id, "watchlist", wl.watchlist_id, ee.event.event_id)
            alert = AlertRow(
                alert_id=alert_id,
                run_id=run_id,
                rule_id=None,
                triggered_at=datetime.now(timezone.utc),
                trigger_payload_json={
                    "source": "watchlist",
                    "watchlist_id": wl.watchlist_id,
                    "watchlist": wl.name,
                    "event_type": ee.event.event_type,
                    "canonical_key": ee.event.canonical_key,
                    "urgency": cfg_urgency_override or wl.alert_severity,
                    "channels": cfg_channels,
                },
                doc_version_ids=ee.revision.doc_version_ids,
                event_ids=[ee.event.event_id],
                acknowledged=0,
            )
            alerts.append(alert)
            per_watchlist_counts[wl.watchlist_id] += 1

            logger.debug(
                "Watchlist alert generated: wl=%s event=%s..%s urgency=%s",
                wl.watchlist_id, ee.event.event_id[:8], ee.event.event_id[-4:],
                cfg_urgency_override or wl.alert_severity,
            )

            for eid in dict.fromkeys(rev_evidence_ids):  # dedupe, preserving order
                evidence_rows.append(
                    AlertEvidenceRow(
                        alert_id=alert_id, evidence_id=eid, purpose=PURPOSE_WATCHLIST_TRIGGER,
                    )
                )

    logger.info(
        "generate_watchlist_alerts: %d matches → %d alerts emitted, "
        "%d skipped (no evidence)",
        matched, len(alerts), skipped_no_evidence,
    )
    if per_watchlist_counts:
        logger.info(
            "generate_watchlist_alerts: per-watchlist breakdown: %s",
            dict(per_watchlist_counts),
        )
    return alerts, evidence_rows


# ------------------------------------------------------------ digest items

def generate_digest_items(
    eligible_events: Sequence[EligibleEvent],
    run_id: str,
    db: Stage09DatabaseInterface,
    section_mapping: dict[str, str] | None = None,
) -> tuple[list[DigestItemRow], list[DigestItemEvidenceRow]]:
    """
    Create digest items from eligible events.

    Groups events by date for digest consumption. Each event with
    evidence becomes one digest item.

    :param section_mapping: Config-derived event_type → section label mapping.
    """
    items: list[DigestItemRow] = []
    evidence_rows: list[DigestItemEvidenceRow] = []
    skipped_no_evidence = 0
    section_counts: dict[str, int] = defaultdict(int)

    for ee in eligible_events:
        rev_evidence_ids = db.get_revision_evidence_ids(ee.revision.revision_id)
        if not rev_evidence_ids:
            skipped_no_evidence += 1
            logger.debug(
                "Digest item skipped for event %s: revision %s has no evidence",
                ee.event.event_id[:16], ee.revision.revision_id[:16],
            )
            continue

        # Determine digest date from the revision's creation time
        created_at = ee.revision.created_at
        if isinstance(created_at, datetime):
            digest_date = created_at.date()
        else:
            digest_date = date.today()
            logger.debug(
                "Digest item for event %s: created_at is not datetime (%s), "
                "falling back to date.today()",
                ee.event.event_id[:16], type(created_at).__name__,
            )

        # Determine novelty label from the first doc in the revision
        novelty = None
        if ee.revision.doc_version_ids:
            novelty = db.get_novelty_label(ee.revision.doc_version_ids[0])

        item_id = compute_sha256_id(run_id, "digest", ee.event.event_id)
        section = _classify_digest_section(ee.event.event_type, section_mapping)
        section_counts[section] += 1

        item = DigestItemRow(
            item_id=item_id,
            run_id=run_id,
            digest_date=digest_date,
            section=section,
            item_type=ee.event.event_type,
            doc_version_ids=ee.revision.doc_version_ids,
            payload_json={
                "canonical_key": ee.event.canonical_key,
                "slots": ee.revision.slots_json,
                "confidence": ee.revision.confidence,
            },
            event_ids=[ee.event.event_id],
            novelty_label=novelty,
        )
        items.append(item)

        for eid in dict.fromkeys(rev_evidence_ids):  # dedupe, preserving order
            evidence_rows.append(
                DigestItemEvidenceRow(
                    item_id=item_id, evidence_id=eid, purpose=PURPOSE_DIGEST_CONTENT,
                )
            )

    logger.info(
        "generate_digest_items: %d items from %d events, "
        "%d skipped (no evidence), %d evidence rows total",
        len(items), len(eligible_events),
        skipped_no_evidence, len(evidence_rows),
    )
    if section_counts:
        logger.info("generate_digest_items: section distribution: %s", dict(section_counts))
    return items, evidence_rows


def _build_digest_section_mapping(config: Config) -> dict[str, str]:
    """
    Build event_type → digest section mapping from config event types.

    Maps event types to sections based on their config descriptions and
    the keywords defined in DIGEST_SECTION_KEYWORDS.  Falls back to
    DEFAULT_DIGEST_SECTION for unknown types.

    This avoids hardcoding the mapping while still producing meaningful
    section groupings for the analyst-facing digest.
    """
    mapping: dict[str, str] = {}
    for et_name, et_def in config.extraction.event_types.items():
        desc_lower = et_def.description.lower()
        matched_section = DEFAULT_DIGEST_SECTION
        for section, keywords in DIGEST_SECTION_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                matched_section = section
                break
        mapping[et_name] = matched_section
    logger.info(
        "_build_digest_section_mapping: built mapping from %d event types: %s",
        len(mapping), mapping,
    )
    return mapping


def _classify_digest_section(
    event_type: str,
    section_mapping: dict[str, str] | None = None,
) -> str:
    """
    Map event type to a digest section label.

    Uses the config-derived *section_mapping* when available, falling back
    to a static mapping only as a safety net.
    """
    if section_mapping is not None:
        return section_mapping.get(event_type, DEFAULT_DIGEST_SECTION)
    logger.debug(
        "_classify_digest_section: no section_mapping provided, using fallback "
        "for event_type=%s", event_type,
    )
    return _DIGEST_SECTION_FALLBACK.get(event_type, DEFAULT_DIGEST_SECTION)



# Config state logging

def _log_config_state(config: Config) -> None:
    """
    Log the relevant config state at the start of stage 09.

    Provides visibility into what the pipeline will use for alert generation
    and digest construction so operators can validate input configuration.
    """
    # Alerts config
    alerts_enabled = (
        config.alerts.settings.enabled if config.alerts and config.alerts.settings else False
    )
    num_config_rules = len(config.alerts.rules) if config.alerts else 0
    num_config_watchlists = len(config.alerts.watchlists) if config.alerts else 0
    logger.info(
        "Stage 09 config state: alerts.settings.enabled=%s, "
        "%d alert rules in config, %d watchlists in config",
        alerts_enabled, num_config_rules, num_config_watchlists,
    )

    # Deduplication config
    if config.alerts and config.alerts.deduplication:
        dedup = config.alerts.deduplication
        logger.info(
            "Stage 09 config state: deduplication enabled=%s, "
            "time_window_hours=%d, similarity_threshold=%.2f, strategy=%s",
            dedup.enabled, dedup.time_window_hours,
            dedup.similarity_threshold, dedup.strategy,
        )
    else:
        logger.info("Stage 09 config state: deduplication not configured")

    # Alert channels
    if config.alerts and config.alerts.settings:
        logger.info(
            "Stage 09 config state: alert timezone=%s, default_channel=%s, "
            "channels=%s",
            config.alerts.settings.timezone,
            config.alerts.settings.default_channel,
            list(config.alerts.settings.channels.keys()),
        )

    # Event types (for digest mapping)
    logger.info(
        "Stage 09 config state: %d event types defined: %s",
        len(config.extraction.event_types),
        list(config.extraction.event_types.keys()),
    )

    # Log individual alert rule summaries
    if config.alerts and config.alerts.rules:
        for rule in config.alerts.rules:
            logger.info(
                "  Alert rule: id=%s name=%s enabled=%s urgency=%s "
                "channels=%s suppression_hours=%d",
                rule.id, rule.name, rule.enabled, rule.urgency,
                rule.channels, rule.suppression_window_hours,
            )

    # Log individual watchlist summaries
    if config.alerts and config.alerts.watchlists:
        for wl_key, wl_cfg in config.alerts.watchlists.items():
            logger.info(
                "  Watchlist: key=%s name=%s enabled=%s entities=%d "
                "publishers=%s event_types=%s urgency_override=%s",
                wl_key, wl_cfg.name, wl_cfg.enabled, len(wl_cfg.entities),
                wl_cfg.publishers, wl_cfg.event_types, wl_cfg.urgency_override,
            )

# Orchestration
def run_stage_09(  # noqa: C901
    db: Stage09DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> dict[str, int]:
    """
    Execute the stage 09 outputs pipeline inside a caller-managed transaction.

    :param db: Open database interface.
    :param config: Validated pipeline configuration.
    :param run_id: Pipeline run ID.
    :param config_hash: Config version hash for audit.
    :return: Summary counts of generated outputs.
    :raises DBError: On database-level failures.
    """
    # ---- 0. Log config state for visibility ----
    _log_config_state(config)

    # Check alerts.settings.enabled from config
    alerts_enabled = (
        config.alerts.settings.enabled if config.alerts and config.alerts.settings else False
    )
    if not alerts_enabled:
        logger.warning(
            "Stage 09: alerts.settings.enabled is False in config; "
            "alert and watchlist generation will be skipped",
        )

    # ---- 1. Eligibility gating ----
    logger.info("Stage 09: computing eligible documents")
    eligible_doc_ids = frozenset(db.get_eligible_doc_version_ids())
    logger.info("Stage 09: %d eligible documents", len(eligible_doc_ids))

    logger.info("Stage 09: computing eligible events")
    eligible_events = db.get_eligible_events(eligible_doc_ids)
    logger.info("Stage 09: %d eligible events", len(eligible_events))
    if eligible_events:
        event_type_counts: dict[str, int] = defaultdict(int)
        for ee in eligible_events:
            event_type_counts[ee.event.event_type] += 1
        logger.info("Stage 09: event type breakdown: %s", dict(event_type_counts))

    logger.info("Stage 09: loading eligible metric observations")
    observations = db.get_eligible_metric_observations(eligible_doc_ids)
    logger.info("Stage 09: %d eligible observations (with evidence)", len(observations))

    # Log sample of observations for debugging
    if observations:
        logger.info("Stage 09: sample of first 3 observations:")
        for i, obs in enumerate(observations[:3]):
            logger.info(
                "  obs[%d]: metric_name=%s period_start=%s period_end=%s "
                "period_granularity=%s value_norm=%s unit_norm=%s geography=%s",
                i, obs.metric_name, obs.period_start, obs.period_end,
                obs.period_granularity, obs.value_norm, obs.unit_norm, obs.geography,
            )

    # ---- 2. Delete prior outputs for this run (idempotent rebuild) ----
    logger.info("Stage 09: deleting prior outputs for run_id=%s", run_id[:16])
    deleted = db.delete_stage09_outputs_for_run(run_id)
    logger.info("Stage 09: deleted prior outputs: %s", deleted)

    # ---- 3. Metric series materialization ----
    logger.info("Stage 09: materializing metric series")
    series_rows, point_rows = materialize_metric_series(observations, run_id)
    for sr in series_rows:
        db.insert_metric_series(sr)
    for pr in point_rows:
        db.insert_metric_series_point(pr)
    logger.info(
        "Stage 09: materialized %d series, %d points", len(series_rows), len(point_rows),
    )

    # ---- 3b. Precompute per-doc lookups (shared by steps 4, 5, 6) ----
    all_doc_ids_in_events: set[str] = set()
    for ee in eligible_events:
        all_doc_ids_in_events.update(ee.revision.doc_version_ids)
    logger.info(
        "Stage 09: precomputing per-doc lookups for %d unique docs across events",
        len(all_doc_ids_in_events),
    )

    entity_ids_by_doc: dict[str, list[str]] = {}
    facets_by_doc: dict[str, list[str]] = {}
    publisher_by_doc: dict[str, str | None] = {}
    for dvid in sorted(all_doc_ids_in_events):
        entity_ids_by_doc[dvid] = db.get_mention_entity_ids(dvid)
        facets_by_doc[dvid] = db.get_facet_values(dvid, "topic")
        publisher_by_doc[dvid] = db.get_document_publisher(dvid)

    # Initialize counters for the case where alerts are disabled
    rule_alerts: list[AlertRow] = []
    rule_alert_ev: list[AlertEvidenceRow] = []
    wl_alerts: list[AlertRow] = []
    wl_alert_ev: list[AlertEvidenceRow] = []

    if alerts_enabled:
        # ---- 4. Alert generation (rules) ----
        logger.info("Stage 09: generating rule-based alerts")
        alert_rules = db.list_active_alert_rules()
        logger.info("Stage 09: %d active alert rules loaded from DB", len(alert_rules))

        # Log reconciliation between config rules and DB rules
        config_rule_ids = {r.id for r in config.alerts.rules} if config.alerts else set()
        db_rule_ids = {r.rule_id for r in alert_rules}
        if config_rule_ids != db_rule_ids:
            only_config = config_rule_ids - db_rule_ids
            only_db = db_rule_ids - config_rule_ids
            if only_config:
                logger.warning(
                    "Stage 09: %d alert rule(s) in config but not in DB: %s",
                    len(only_config), sorted(only_config),
                )
            if only_db:
                logger.warning(
                    "Stage 09: %d alert rule(s) in DB but not in config: %s",
                    len(only_db), sorted(only_db),
                )

        rule_alerts, rule_alert_ev = generate_alerts(
            eligible_events, alert_rules, run_id, db, config,
            facets_by_doc=facets_by_doc,
        )
        for a in rule_alerts:
            db.insert_alert(a)
        for ae in rule_alert_ev:
            db.insert_alert_evidence(ae)

        # ---- 5. Alert generation (watchlists) ----
        logger.info("Stage 09: generating watchlist alerts")

        # Build entity name→id map once for all watchlist matching
        name_to_id = db.build_entity_name_to_id_map()
        logger.info("Stage 09: entity name→id map has %d entries", len(name_to_id))

        # Build config-sourced publisher lists for each watchlist (since
        # WatchlistRow does not carry publisher info from config).
        config_watchlist_publishers: dict[str, list[str]] = {}
        if config.alerts and config.alerts.watchlists:
            for wl_key, wl_cfg in config.alerts.watchlists.items():
                if wl_cfg.publishers:
                    config_watchlist_publishers[wl_key] = wl_cfg.publishers
        if config_watchlist_publishers:
            logger.info(
                "Stage 09: loaded publisher filters for %d watchlists from config: %s",
                len(config_watchlist_publishers),
                {k: v for k, v in config_watchlist_publishers.items()},
            )

        watchlists = db.list_active_watchlists()
        logger.info("Stage 09: %d active watchlists loaded from DB", len(watchlists))

        # Log reconciliation between config watchlists and DB watchlists
        config_wl_keys = set(config.alerts.watchlists.keys()) if config.alerts else set()
        db_wl_ids = {w.watchlist_id for w in watchlists}
        if config_wl_keys != db_wl_ids:
            only_config_wl = config_wl_keys - db_wl_ids
            only_db_wl = db_wl_ids - config_wl_keys
            if only_config_wl:
                logger.warning(
                    "Stage 09: %d watchlist(s) in config but not in DB: %s",
                    len(only_config_wl), sorted(only_config_wl),
                )
            if only_db_wl:
                logger.warning(
                    "Stage 09: %d watchlist(s) in DB but not in config: %s",
                    len(only_db_wl), sorted(only_db_wl),
                )

        wl_alerts, wl_alert_ev = generate_watchlist_alerts(
            eligible_events, watchlists, run_id, db, config,
            name_to_id, entity_ids_by_doc, facets_by_doc, publisher_by_doc,
            config_watchlist_publishers=config_watchlist_publishers,
        )
        for a in wl_alerts:
            db.insert_alert(a)
        for ae in wl_alert_ev:
            db.insert_alert_evidence(ae)
    else:
        logger.info(
            "Stage 09: alert generation skipped (alerts.settings.enabled=False)"
        )

    # ---- 6. Digest items ----
    logger.info("Stage 09: generating digest items")
    section_mapping = _build_digest_section_mapping(config)
    digest_items, digest_ev = generate_digest_items(
        eligible_events, run_id, db, section_mapping=section_mapping,
    )
    for di in digest_items:
        db.insert_digest_item(di)
    for de in digest_ev:
        db.insert_digest_item_evidence(de)

    # ---- 7. Run stage status ----
    total_alerts = len(rule_alerts) + len(wl_alerts)
    summary = {
        "eligible_docs": len(eligible_doc_ids),
        "eligible_events": len(eligible_events),
        "metric_series": len(series_rows),
        "metric_points": len(point_rows),
        "alerts_rule": len(rule_alerts),
        "alerts_watchlist": len(wl_alerts),
        "digest_items": len(digest_items),
        "alerts_enabled": alerts_enabled,
    }
    details_json = json.dumps(summary)
    logger.info("Stage 09: output summary: %s", summary)

    db.upsert_run_stage_status(
        run_id=run_id,
        stage=STAGE_NAME,
        config_hash=config_hash,
        status="ok",
        details=details_json,
    )

    return {
        "eligible_docs": len(eligible_doc_ids),
        "eligible_events": len(eligible_events),
        "metric_series": len(series_rows),
        "metric_points": len(point_rows),
        "alerts": total_alerts,
        "digest_items": len(digest_items),
    }

# CLI
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 09: Outputs")
    parser.add_argument(
        "--run-id", type=str,
        default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (64-char SHA256 hex)",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("../../../../config/etl_config/"))
    parser.add_argument("--source-db", type=Path, default=Path("../../../../database/preprocessed_posts.db"))
    parser.add_argument("--working-db", type=Path, default=Path("../../../../database/processed_posts.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../../../output/processed/"))
    parser.add_argument("--log-dir", type=Path, default=Path("../../../../output/processed/logs/"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main_stage_09_outputs() -> int:
    """
    Entry point for stage 09.

    :return: 0 on success, 1 on fatal error.
    """
    args = parse_args()
    run_id = args.run_id
    config_path = args.config_dir / "config.yaml"

    logger.info("Stage 09 starting: run_id=%s", run_id)
    logger.info("Stage 09: config_path=%s working_db=%s", config_path, args.working_db)

    try:
        config = load_config(config_path)
    except Exception:
        logger.exception("Failed to load config from %s", config_path)
        return 1

    config_hash = get_config_version(config)
    logger.info("Stage 09: config loaded, config_hash=%s", config_hash)

    db = Stage09DatabaseInterface(args.working_db)

    try:
        db.open()
        logger.info("Stage 09: database opened at %s", args.working_db)
        with db.transaction():
            result = run_stage_09(db, config, run_id, config_hash)
        logger.info("Stage 09 completed successfully: %s", result)
        return 0
    except Exception:
        logger.exception("Stage 09 fatal error")
        try:
            db.upsert_run_stage_status(
                run_id=run_id, stage=STAGE_NAME, config_hash=config_hash,
                status="failed", error_message="fatal_error",
            )
        except Exception:
            logger.exception("Failed to record stage failure status")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main_stage_09_outputs())