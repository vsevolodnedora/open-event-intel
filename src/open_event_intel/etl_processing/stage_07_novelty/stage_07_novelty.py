"""
Stage 07 — Novelty Detection and Story Clustering.

**Reads:** ``document_version``, ``chunk``, ``chunk_embedding``, ``mention``, ``facet_assignment``.
**Per-doc writes:** ``novelty_label``, ``novelty_label_evidence``, ``document_fingerprint``, ``chunk_novelty``, ``chunk_novelty_score``, ``evidence_span``, ``doc_stage_status(stage_07_novelty)``.
**Run-scoped writes:** ``story_cluster``, ``story_cluster_member``, ``run_stage_status(stage_07_story_cluster)``.
**Execution model:** Same two-phase pattern as Stage 5. Run-scoped clustering executes after the per-doc novelty labeling
has written status rows for all clustering-input documents (no missing rows). Clustering is built over
documents with ``doc_stage_status(stage_07_novelty).status='ok'`` only. Record coverage in ``run_stage_status.details``.
If run-scoped clustering fails, its transaction rolls back and the pipeline aborts; per-doc novelty outputs remain and
will be reused on rerun.
Exit codes: Per §6.5. 0 = per-doc phase reached terminal status for all inputs AND run-scoped phase completed; 1 = fatal error or run-scoped failure (run-scoped writes rolled back; per-doc writes remain).
  * Example algorithms: Candidate neighbor retrieval (doc-level), Similarity scoring + doc label, Chunk alignment + chunk labels, Evidence-backed explanations

"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from open_event_intel.etl_processing.config_interface import Config, get_config_version, load_config
from open_event_intel.etl_processing.database_interface import (
    ChunkEmbeddingRow,
    ChunkNoveltyRow,
    ChunkNoveltyScoreRow,
    ChunkRow,
    DBError,
    DocumentFingerprintRow,
    DocumentVersionRow,
    EvidenceSpanRow,
    MentionRow,
    NoveltyLabelEvidenceRow,
    NoveltyLabelRow,
    StoryClusterMemberRow,
    StoryClusterRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_07_novelty.database_stage_07_novelty import CLUSTER_STAGE_NAME, STAGE_NAME, Stage07DatabaseInterface, _bytes_to_vector
from open_event_intel.logger import get_logger

logger = get_logger(__name__)



# Module-level defaults for novelty detection parameters.
# These are used when no config section or CLI override is provided.
# Every operational parameter is traced to one of:
#   1. a DEFAULT_* constant below,
#   2. a value read from config.yaml, or
#   3. a CLI --override.
DEFAULT_MINHASH_NUM_PERM = 128
DEFAULT_SHINGLE_SIZE = 3
DEFAULT_NEIGHBOUR_LIMIT = 10
DEFAULT_NEW_THRESHOLD = 0.30
DEFAULT_UPDATE_THRESHOLD = 0.70
DEFAULT_CLUSTER_THRESHOLD = 0.40
DEFAULT_LINKING_WINDOW_DAYS = 90
DEFAULT_TOP_K_NEIGHBOURS_FOR_DETAIL = 3
DEFAULT_CHUNK_NOVELTY_NEW_THRESHOLD = 0.70
DEFAULT_CHUNK_NOVELTY_OVERLAP_THRESHOLD = 0.30
DEFAULT_EVIDENCE_SNIPPET_LENGTH = 200
DEFAULT_PROGRESS_LOG_INTERVAL = 100

# Confidence formula tuning constants for classify_novelty().
# Extracted here to avoid silent magic numbers in the scoring logic.

# For "update" labels: confidence = CONF_UPDATE_BASE + CONF_UPDATE_SCALE * interpolation
CONF_UPDATE_BASE = 0.5
CONF_UPDATE_SCALE = 0.5
# For "new" labels with shared mentions: penalty per shared mention, up to max count
CONF_MENTION_PENALTY = 0.1
CONF_MENTION_MAX_COUNT = 3
CONF_MENTION_FLOOR = 0.5


# Semantic string constants for DB columns / method fields.
CHUNK_NOVELTY_METHOD = "cosine_top3"
CLUSTER_METHOD = "cosine_cc"
EVIDENCE_PURPOSE_NOVELTY_ANCHOR = "novelty_anchor"


class NoveltyConfig(BaseModel):
    """
    Operational parameters for novelty detection.

    Extracted once from ``Config`` + CLI overrides so pure functions receive
    only typed values.  No silent defaults — every value is traceable to a
    module-level constant, a config file, or a CLI argument.
    """

    model_config = ConfigDict(frozen=True)

    minhash_num_perm: int = DEFAULT_MINHASH_NUM_PERM
    shingle_size: int = DEFAULT_SHINGLE_SIZE
    neighbour_limit: int = DEFAULT_NEIGHBOUR_LIMIT
    new_threshold: float = DEFAULT_NEW_THRESHOLD
    update_threshold: float = DEFAULT_UPDATE_THRESHOLD
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    linking_window_days: int = DEFAULT_LINKING_WINDOW_DAYS
    top_k_neighbours_for_detail: int = DEFAULT_TOP_K_NEIGHBOURS_FOR_DETAIL
    chunk_novelty_new_threshold: float = DEFAULT_CHUNK_NOVELTY_NEW_THRESHOLD
    chunk_novelty_overlap_threshold: float = DEFAULT_CHUNK_NOVELTY_OVERLAP_THRESHOLD
    evidence_snippet_length: int = DEFAULT_EVIDENCE_SNIPPET_LENGTH
    progress_log_interval: int = DEFAULT_PROGRESS_LOG_INTERVAL


def build_novelty_config(config: Config, cli_overrides: dict | None = None) -> NoveltyConfig:
    """
    Derive :class:`NoveltyConfig` from the project root config and CLI overrides.

    Currently the root ``Config`` does not expose a dedicated novelty section.
    If one is added in the future, values should be read here.  For now,
    module-level defaults are used and any CLI overrides are applied on top.

    :param config: The loaded project configuration (consulted for future
        novelty-section fields).
    :param cli_overrides: Dict of field-name → value from argparse; ``None``
        values are skipped.
    """
    params: dict = {}

    # ---- Read from config if a novelty section is ever added ----
    # (placeholder for forward compatibility)
    # e.g.:
    # if hasattr(config, 'novelty') and config.novelty is not None:
    #     params['minhash_num_perm'] = config.novelty.minhash_num_perm
    #     ...

    # Use alert deduplication similarity_threshold as cluster_threshold hint
    # if no explicit CLI override is given (soft coupling — logged below).
    if config.alerts and config.alerts.deduplication:
        dedup = config.alerts.deduplication
        if dedup.enabled and dedup.similarity_threshold is not None:
            logger.info("build_novelty_config: alerts.deduplication.similarity_threshold=%.4f available "
                        "(not auto-applied to cluster_threshold; use --cluster-threshold to set explicitly)",
                        dedup.similarity_threshold)

    # ---- Apply CLI overrides (non-None values) ----
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                params[key] = value

    ncfg = NoveltyConfig(**params)
    logger.info("build_novelty_config: constructed NoveltyConfig from %d explicit overrides "
                "(remaining fields use module-level defaults)", len(params))
    return ncfg

# Pure helper functions


def _vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialise a numpy array to a little-endian float32 blob."""
    return vec.astype("<f4").tobytes()


def _text_to_shingles(text: str, k: int) -> set[str]:
    """
    Split *text* into character-level k-shingles.

    :param k: Shingle size — passed explicitly, never defaulted silently.
    """
    text = text.lower()
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def compute_minhash_signature(
    shingles: set[str], num_perm: int,
) -> bytes:
    """
    Compute a deterministic MinHash signature.

    Uses the *hash-trick* with ``num_perm`` independent hash functions
    (seeded via index) to produce a ``num_perm * 4`` byte signature
    (uint32 little-endian).

    :param shingles: Token/shingle set.
    :param num_perm: Number of permutations (hash functions).  Must be
        passed explicitly by caller.
    :return: Packed bytes of length ``num_perm * 4``.
    """
    max_hash = (1 << 32) - 1
    sig = np.full(num_perm, max_hash, dtype=np.uint32)
    for s in shingles:
        s_bytes = s.encode("utf-8")
        for i in range(num_perm):
            h = int(
                hashlib.md5(s_bytes + i.to_bytes(4, "little")).hexdigest()[:8], 16
            ) & max_hash
            if h < sig[i]:
                sig[i] = h
    return sig.astype("<u4").tobytes()


def estimate_jaccard_from_minhash(sig_a: bytes, sig_b: bytes) -> float:
    """Estimate Jaccard similarity between two MinHash signatures."""
    a = np.frombuffer(sig_a, dtype="<u4")
    b = np.frombuffer(sig_b, dtype="<u4")
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return float(np.sum(a == b)) / len(a)


def compute_simhash_signature(text: str) -> bytes:
    """
    Compute a 64-bit SimHash for *text*.

    :return: 8 bytes (uint64 LE).
    """
    tokens = text.lower().split()
    v = np.zeros(64, dtype=np.float64)
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            if (h >> i) & 1:
                v[i] += 1.0
            else:
                v[i] -= 1.0
    fingerprint = np.uint64(0)
    for i in range(64):
        if v[i] > 0:
            fingerprint |= np.uint64(1) << np.uint64(i)
    return fingerprint.tobytes()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_candidate_neighbours(
    target_centroid: np.ndarray,
    target_doc_id: str,
    all_centroids: list[tuple[str, np.ndarray]],
    limit: int,
) -> list[tuple[str, float]]:
    """
    Return up to *limit* nearest documents by cosine similarity.

    Excludes the target document itself. Results are sorted descending by
    similarity.

    .. note::
       Temporal windowing (``linking_window_days``) is **not** applied here
       because centroids do not carry publication-date metadata. The parameter
       is stored in ``novelty_label.linking_window_days`` for traceability
       and future use when date-aware filtering is implemented.

    :param limit: Maximum neighbours to return — must be passed explicitly.
    """
    scored: list[tuple[str, float]] = []
    for dvid, centroid in all_centroids:
        if dvid == target_doc_id:
            continue
        sim = cosine_similarity(target_centroid, centroid)
        scored.append((dvid, sim))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:limit]


def classify_novelty(
    max_similarity: float,
    shared_mention_count: int,
    ncfg: NoveltyConfig,
) -> tuple[str, float]:
    """
    Assign a novelty label and confidence.

    Confidence scoring logic:

    * **re_report** (``max_similarity >= update_threshold``):
      confidence = clamp(max_similarity, 0, 1).
    * **update** (``new_threshold <= max_similarity < update_threshold``):
      confidence linearly interpolates from ``CONF_UPDATE_BASE`` to
      ``CONF_UPDATE_BASE + CONF_UPDATE_SCALE`` across the threshold range.
    * **new** (``max_similarity < new_threshold``):
      confidence = 1 − max_similarity, with a per-shared-mention penalty
      of ``CONF_MENTION_PENALTY`` (up to ``CONF_MENTION_MAX_COUNT``
      mentions), floored at ``CONF_MENTION_FLOOR``.

    :return: ``(label, confidence)`` where label is one of
        ``"new"``, ``"update"``, ``"re_report"``.
    """
    if max_similarity >= ncfg.update_threshold:
        label = "re_report"
        confidence = min(1.0, max_similarity)
    elif max_similarity >= ncfg.new_threshold:
        label = "update"
        confidence = CONF_UPDATE_BASE + CONF_UPDATE_SCALE * (
            (max_similarity - ncfg.new_threshold)
            / max(ncfg.update_threshold - ncfg.new_threshold, 1e-9)
        )
    else:
        label = "new"
        confidence = 1.0 - max_similarity
    if shared_mention_count > 0 and label == "new":
        penalty = CONF_MENTION_PENALTY * min(shared_mention_count, CONF_MENTION_MAX_COUNT)
        confidence = max(CONF_MENTION_FLOOR, confidence - penalty)
    return label, round(confidence, 4)


def compute_shared_mentions(
    target_mentions: list[MentionRow],
    neighbour_mentions: list[MentionRow],
) -> dict[str, list[str]]:
    """
    Find mentions shared between two documents.

    :return: Mapping of shared ``normalized_value`` → list of
        ``mention_type`` values.
    """
    target_set: dict[str, set[str]] = {}
    for m in target_mentions:
        key = (m.normalized_value or m.surface_form).lower()
        target_set.setdefault(key, set()).add(m.mention_type)
    shared: dict[str, list[str]] = {}
    for m in neighbour_mentions:
        key = (m.normalized_value or m.surface_form).lower()
        if key in target_set:
            types = sorted(target_set[key] | {m.mention_type})
            shared[key] = types
    return dict(sorted(shared.items()))


def score_chunk_novelty(
    chunk_embedding: np.ndarray,
    neighbour_chunk_embeddings: list[tuple[str, np.ndarray]],
    top_k: int,
) -> tuple[float, list[str], list[float]]:
    """
    Score how novel a single chunk is relative to neighbour chunks.

    :param top_k: Number of top-scoring neighbours to consider.  Passed
        explicitly from ``ncfg.top_k_neighbours_for_detail``.
    :return: ``(novelty_score, source_chunk_ids, similarity_scores)``
        where ``novelty_score`` ∈ [0, 1], 1 = completely novel.
    """
    if not neighbour_chunk_embeddings:
        return 1.0, [], []
    sims: list[tuple[str, float]] = []
    for cid, emb in neighbour_chunk_embeddings:
        sims.append((cid, cosine_similarity(chunk_embedding, emb)))
    sims.sort(key=lambda x: -x[1])
    top = sims[:top_k]
    max_sim = top[0][1] if top else 0.0
    novelty = 1.0 - max(0.0, min(1.0, max_sim))
    return round(novelty, 4), [c for c, _ in top], [round(s, 4) for _, s in top]


def chunk_novelty_label(
    novelty_score: float,
    new_threshold: float,
    overlap_threshold: float,
) -> str:
    """
    Map a novelty score to a categorical label.

    :param new_threshold: Score at or above which the chunk is labelled ``"new"``.
    :param overlap_threshold: Score at or above which the chunk is labelled
        ``"partial_overlap"``; below is ``"duplicate"``.
    """
    if novelty_score >= new_threshold:
        return "new"
    if novelty_score >= overlap_threshold:
        return "partial_overlap"
    return "duplicate"


def _build_connected_components(
    edges: list[tuple[str, str]],
    nodes: set[str],
) -> list[set[str]]:
    """Build connected components from an undirected edge list via union-find."""
    parent: dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    components: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        components[find(n)].add(n)
    return sorted(components.values(), key=lambda s: sorted(s)[0])


def _log_similarity_distribution(
    similarities: list[float],
    threshold: float,
    n_docs: int,
) -> None:
    """
    Log detailed statistics about pairwise similarity distribution.

    This is the key diagnostic: if the median similarity is above threshold,
    the graph is near-complete and connected components will collapse into
    one mega-cluster.
    """
    if not similarities:
        logger.warning("CLUSTER-DIAG: No pairwise similarities to analyze")
        return
    arr = np.array(similarities)
    total_pairs = len(arr)
    above = int(np.sum(arr >= threshold))
    max_possible_edges = n_docs * (n_docs - 1) // 2

    logger.info("=" * 72)
    logger.info("CLUSTER-DIAG: Pairwise cosine similarity distribution")
    logger.info("=" * 72)
    logger.info("  Documents in clustering:    %d", n_docs)
    logger.info("  Total pairwise comparisons: %d", total_pairs)
    logger.info("  Min similarity:             %.4f", float(np.min(arr)))
    logger.info("  Max similarity:             %.4f", float(np.max(arr)))
    logger.info("  Mean similarity:            %.4f", float(np.mean(arr)))
    logger.info("  Median similarity:          %.4f", float(np.median(arr)))
    logger.info("  Std deviation:              %.4f", float(np.std(arr)))

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pvals = np.percentile(arr, percentiles)
    logger.info("  Percentiles:")
    for p, v in zip(percentiles, pvals):
        marker = " *** ABOVE THRESHOLD" if v >= threshold else ""
        logger.info("    P%02d: %.4f%s", p, v, marker)

    logger.info("  Current threshold:          %.4f", threshold)
    logger.info("  Edges above threshold:      %d / %d (%.1f%%)",
                above, total_pairs, 100.0 * above / max(total_pairs, 1))
    logger.info("  Edge density:               %.4f (1.0 = complete graph)",
                above / max(max_possible_edges, 1))

    # Threshold sensitivity analysis
    logger.info("  --- Threshold sensitivity (edge count at various thresholds) ---")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        count = int(np.sum(arr >= t))
        pct = 100.0 * count / max(total_pairs, 1)
        marker = " <-- CURRENT" if abs(t - threshold) < 0.001 else ""
        logger.info("    threshold=%.2f → %6d edges (%5.1f%%)%s", t, count, pct, marker)
    logger.info("=" * 72)


def _log_centroid_diagnostics(
    filtered: list[tuple[str, np.ndarray]],
) -> None:
    """Log centroid vector statistics to detect degenerate embeddings."""
    if not filtered:
        return
    norms = [float(np.linalg.norm(c)) for _, c in filtered]
    dims = [c.shape[0] for _, c in filtered]
    unique_dims = set(dims)

    logger.info("CLUSTER-DIAG: Centroid vector diagnostics")
    logger.info("  Number of centroids:  %d", len(filtered))
    logger.info("  Embedding dimensions: %s", sorted(unique_dims))
    logger.info("  Norm min/mean/max:    %.4f / %.4f / %.4f",
                min(norms), sum(norms) / len(norms), max(norms))
    zero_norm = sum(1 for n in norms if n < 1e-8)
    if zero_norm:
        logger.warning("  Zero-norm centroids:  %d (these will have similarity 0.0 with everything)", zero_norm)

    # Check for near-identical centroids (a sign of degenerate averaging)
    if len(filtered) >= 2:
        sample_size = min(50, len(filtered))
        sample_indices = np.random.default_rng(42).choice(len(filtered), sample_size, replace=False)
        sample_sims = []
        for ii in range(len(sample_indices)):
            for jj in range(ii + 1, len(sample_indices)):
                a_idx, b_idx = sample_indices[ii], sample_indices[jj]
                s = cosine_similarity(filtered[a_idx][1], filtered[b_idx][1])
                sample_sims.append(s)
        if sample_sims:
            logger.info("  Sample pairwise (n=%d pairs from %d random docs):", len(sample_sims), sample_size)
            logger.info("    min=%.4f  mean=%.4f  median=%.4f  max=%.4f",
                        min(sample_sims), sum(sample_sims) / len(sample_sims),
                        float(np.median(sample_sims)), max(sample_sims))


def _log_component_diagnostics(
    components: list[set[str]],
    n_docs: int,
) -> None:
    """Log connected component size distribution."""
    if not components:
        logger.info("CLUSTER-DIAG: No connected components")
        return
    sizes = sorted([len(c) for c in components], reverse=True)
    logger.info("CLUSTER-DIAG: Connected component analysis")
    logger.info("  Total components:  %d", len(sizes))
    logger.info("  Total nodes:       %d", sum(sizes))
    logger.info("  Largest component: %d nodes (%.1f%% of all docs)",
                sizes[0], 100.0 * sizes[0] / max(n_docs, 1))
    if len(sizes) > 1:
        logger.info("  2nd largest:       %d nodes", sizes[1])
    logger.info("  Singletons:        %d", sum(1 for s in sizes if s == 1))
    logger.info("  Size >= 2:         %d components", sum(1 for s in sizes if s >= 2))

    # Show top-10 component sizes
    top = sizes[:min(20, len(sizes))]
    logger.info("  Top component sizes: %s", top)


def build_story_clusters(
    centroids: list[tuple[str, np.ndarray]],
    eligible_ids: set[str],
    threshold: float,
) -> list[set[str]]:
    """
    Cluster documents into story clusters using cosine similarity.

    Two documents are linked if their centroid cosine similarity exceeds
    *threshold*. Connected components form the clusters.

    :param threshold: Cosine similarity threshold — passed explicitly from
        ``ncfg.cluster_threshold``.
    """
    logger.info("build_story_clusters: input centroids=%d, eligible_ids=%d, threshold=%.4f",
                len(centroids), len(eligible_ids), threshold)

    filtered = [(dvid, c) for dvid, c in centroids if dvid in eligible_ids]
    not_in_centroids = eligible_ids - {dvid for dvid, _ in centroids}
    if not_in_centroids:
        logger.warning("build_story_clusters: %d eligible docs have NO centroid (missing embeddings): %s",
                        len(not_in_centroids),
                        list(not_in_centroids)[:5])

    logger.info("build_story_clusters: %d documents after filtering to eligible set", len(filtered))
    if not filtered:
        return []

    # Log centroid diagnostics
    _log_centroid_diagnostics(filtered)

    # Compute ALL pairwise similarities and collect diagnostics
    all_similarities: list[float] = []
    edges: list[tuple[str, str]] = []
    n = len(filtered)
    log_interval = max(1, n // 10)
    for i in range(n):
        if i % log_interval == 0:
            logger.info("build_story_clusters: computing similarities for doc %d/%d ...", i, n)
        for j in range(i + 1, n):
            sim = cosine_similarity(filtered[i][1], filtered[j][1])
            all_similarities.append(sim)
            if sim >= threshold:
                edges.append((filtered[i][0], filtered[j][0]))

    # Log the similarity distribution — this is the key diagnostic
    _log_similarity_distribution(all_similarities, threshold, n)

    logger.info("build_story_clusters: %d edges created (threshold=%.4f)", len(edges), threshold)

    nodes = {dvid for dvid, _ in filtered}
    components = _build_connected_components(edges, nodes)

    # Log component diagnostics
    _log_component_diagnostics(components, n)

    return components


class DocumentNoveltyResult(BaseModel):
    """Intermediate result of per-document novelty processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_version_id: str
    label: str
    confidence: float
    neighbours: list[tuple[str, float]]
    shared_mentions: dict[str, list[str]]
    fingerprint: DocumentFingerprintRow
    chunk_novelties: list[ChunkNoveltyRow]
    chunk_scores: list[ChunkNoveltyScoreRow]
    evidence_spans: list[EvidenceSpanRow]
    linking_window_days: int


def process_document_novelty(
    doc_version: DocumentVersionRow,
    chunks: list[ChunkRow],
    chunk_embeddings: list[ChunkEmbeddingRow],
    mentions: list[MentionRow],
    all_centroids: list[tuple[str, np.ndarray]],
    neighbour_mentions_fn: object,
    neighbour_chunk_embeddings_fn: object,
    ncfg: NoveltyConfig,
    run_id: str,
) -> DocumentNoveltyResult:
    """
    Compute novelty for a single document (pure logic, no DB writes).

    :param neighbour_mentions_fn:
        ``Callable[[str], list[MentionRow]]`` — fetches mentions for a
        neighbour ``doc_version_id``.
    :param neighbour_chunk_embeddings_fn:
        ``Callable[[str], list[tuple[str, np.ndarray]]]`` — fetches
        ``(chunk_id, embedding)`` pairs for a neighbour document.
    """
    dvid = doc_version.doc_version_id
    clean = doc_version.clean_content
    top_k = ncfg.top_k_neighbours_for_detail

    logger.debug("process_document_novelty: doc=%s content_len=%d chunks=%d embeddings=%d mentions=%d top_k=%d",
                 dvid[:12], len(clean) if clean else 0,
                 len(chunks), len(chunk_embeddings), len(mentions), top_k)

    # --- Fingerprinting ---
    shingles = _text_to_shingles(clean, ncfg.shingle_size)
    minhash = compute_minhash_signature(shingles, ncfg.minhash_num_perm)
    simhash = compute_simhash_signature(clean)
    fingerprint = DocumentFingerprintRow(
        doc_version_id=dvid,
        minhash_signature=minhash,
        simhash_signature=simhash,
        created_in_run_id=run_id,
    )
    logger.debug("process_document_novelty: doc=%s shingles=%d minhash_bytes=%d",
                 dvid[:12], len(shingles), len(minhash))

    # --- Centroid computation ---
    if chunk_embeddings:
        vecs = [_bytes_to_vector(ce.embedding_vector, ce.embedding_dim) for ce in chunk_embeddings]
        centroid = np.mean(vecs, axis=0)
    else:
        centroid = np.zeros(1)
        logger.warning("process_document_novelty: doc=%s has NO chunk embeddings; centroid is zero-vector",
                        dvid[:12])

    # --- Neighbour search ---
    # NOTE: linking_window_days is recorded but NOT used for temporal filtering
    # here — centroids lack publication-date metadata.  See find_candidate_neighbours docstring.
    neighbours = find_candidate_neighbours(centroid, dvid, all_centroids, ncfg.neighbour_limit)
    max_sim = neighbours[0][1] if neighbours else 0.0
    logger.debug("process_document_novelty: doc=%s neighbours=%d max_sim=%.4f top3_sims=%s",
                 dvid[:12], len(neighbours), max_sim,
                 [round(s, 4) for _, s in neighbours[:3]])

    # --- Shared mention analysis (top-k neighbours) ---
    shared: dict[str, list[str]] = {}
    for n_dvid, _ in neighbours[:top_k]:
        n_mentions = neighbour_mentions_fn(n_dvid)  # type: ignore[operator]
        shared.update(compute_shared_mentions(mentions, n_mentions))

    # --- Document-level novelty label ---
    label, confidence = classify_novelty(max_sim, len(shared), ncfg)
    logger.info("process_document_novelty: doc=%s → label=%s confidence=%.4f "
                "max_sim=%.4f shared_mentions=%d neighbours_considered=%d",
                dvid[:12], label, confidence, max_sim, len(shared), len(neighbours))

    # --- Evidence span (leading snippet) ---
    evidence_spans: list[EvidenceSpanRow] = []
    snippet_len = min(ncfg.evidence_snippet_length, len(clean))
    if snippet_len > 0:
        ev_id = compute_sha256_id(dvid, 0, snippet_len)
        evidence_spans.append(
            EvidenceSpanRow(
                evidence_id=ev_id,
                doc_version_id=dvid,
                span_start=0,
                span_end=snippet_len,
                text=clean[:snippet_len],
                purpose=EVIDENCE_PURPOSE_NOVELTY_ANCHOR,
                created_in_run_id=run_id,
            )
        )

    # --- Chunk-level novelty ---
    chunk_novelties: list[ChunkNoveltyRow] = []
    chunk_scores: list[ChunkNoveltyScoreRow] = []

    neighbour_embs_cache: dict[str, list[tuple[str, np.ndarray]]] = {}
    chunk_score_values: list[float] = []
    for ce in chunk_embeddings:
        chunk_vec = _bytes_to_vector(ce.embedding_vector, ce.embedding_dim)
        all_neighbour_embs: list[tuple[str, np.ndarray]] = []
        for n_dvid, _ in neighbours[:top_k]:
            if n_dvid not in neighbour_embs_cache:
                neighbour_embs_cache[n_dvid] = neighbour_chunk_embeddings_fn(n_dvid)  # type: ignore[operator]
            all_neighbour_embs.extend(neighbour_embs_cache[n_dvid])

        nscore, src_ids, src_sims = score_chunk_novelty(chunk_vec, all_neighbour_embs, top_k)
        cn_label = chunk_novelty_label(
            nscore,
            ncfg.chunk_novelty_new_threshold,
            ncfg.chunk_novelty_overlap_threshold,
        )
        chunk_novelties.append(
            ChunkNoveltyRow(
                chunk_id=ce.chunk_id,
                novelty_label=cn_label,
                source_chunk_ids=src_ids if src_ids else None,
                similarity_scores=src_sims if src_sims else None,
            )
        )
        chunk_scores.append(
            ChunkNoveltyScoreRow(
                chunk_id=ce.chunk_id,
                novelty_score=nscore,
                method=CHUNK_NOVELTY_METHOD,
                created_in_run_id=run_id,
            )
        )
        chunk_score_values.append(nscore)

    # Log chunk novelty score distribution for this document
    if chunk_score_values:
        arr = np.array(chunk_score_values)
        new_count = sum(1 for v in chunk_score_values if v >= ncfg.chunk_novelty_new_threshold)
        dup_count = sum(1 for v in chunk_score_values if v < ncfg.chunk_novelty_overlap_threshold)
        logger.debug("process_document_novelty: doc=%s chunk_novelty_scores: "
                     "min=%.4f mean=%.4f max=%.4f new=%d partial=%d dup=%d (n=%d)",
                     dvid[:12], float(arr.min()), float(arr.mean()), float(arr.max()),
                     new_count, len(chunk_score_values) - new_count - dup_count, dup_count,
                     len(chunk_score_values))

    return DocumentNoveltyResult(
        doc_version_id=dvid,
        label=label,
        confidence=confidence,
        neighbours=neighbours,
        shared_mentions=shared,
        fingerprint=fingerprint,
        chunk_novelties=chunk_novelties,
        chunk_scores=chunk_scores,
        evidence_spans=evidence_spans,
        linking_window_days=ncfg.linking_window_days,
    )


def _write_per_doc_results(
    db: Stage07DatabaseInterface,
    result: DocumentNoveltyResult,
    run_id: str,
    config_hash: str,
) -> None:
    """
    Persist per-document novelty results inside an open transaction.

    Cleans up any partial results from a prior failed attempt before
    inserting new rows, preventing UNIQUE constraint violations on retry.
    """
    dvid = result.doc_version_id

    # --- Clean up partial results from prior failed attempts ---
    db.delete_prior_doc_results(dvid)

    db.insert_document_fingerprint(result.fingerprint)

    neighbour_ids = (
        [n[0] for n in result.neighbours] if result.neighbours else None
    )
    db.insert_novelty_label(
        NoveltyLabelRow(
            doc_version_id=dvid,
            label=result.label,
            neighbor_doc_version_ids=neighbour_ids,
            similarity_score=result.neighbours[0][1] if result.neighbours else None,
            shared_mentions=result.shared_mentions if result.shared_mentions else None,
            linking_window_days=result.linking_window_days,
            confidence=result.confidence,
            created_in_run_id=run_id,
        )
    )

    for ev in result.evidence_spans:
        db.get_or_create_evidence_span(
            doc_version_id=ev.doc_version_id,
            span_start=ev.span_start,
            span_end=ev.span_end,
            run_id=run_id,
            purpose=ev.purpose,
            clean_content=ev.text,
        )
        db.insert_novelty_label_evidence(
            NoveltyLabelEvidenceRow(
                doc_version_id=dvid,
                evidence_id=ev.evidence_id,
                purpose=ev.purpose,
            )
        )

    for cn in result.chunk_novelties:
        db.insert_chunk_novelty(cn)
    for cs in result.chunk_scores:
        db.insert_chunk_novelty_score(cs)

    db.upsert_doc_stage_status(
        doc_version_id=dvid,
        stage=STAGE_NAME,
        run_id=run_id,
        config_hash=config_hash,
        status="ok",
    )


def _get_neighbour_chunk_embeddings(
    db: Stage07DatabaseInterface, doc_version_id: str
) -> list[tuple[str, np.ndarray]]:
    """Fetch ``(chunk_id, vector)`` pairs for a document."""
    embs = db.get_chunk_embeddings_by_doc(doc_version_id)
    return [
        (ce.chunk_id, _bytes_to_vector(ce.embedding_vector, ce.embedding_dim))
        for ce in embs
    ]


def run_per_doc_phase(  # noqa: C901
    db: Stage07DatabaseInterface,
    run_id: str,
    config_hash: str,
    ncfg: NoveltyConfig,
) -> tuple[int, int, int, int]:
    """
    Execute the per-document novelty phase.

    :return: ``(ok_count, failed_count, blocked_count, skipped_count)``
    """
    phase_start = time.monotonic()

    iteration_set = db.get_iteration_set()
    logger.info("=" * 72)
    logger.info("Stage 07 per-doc phase: %d documents in iteration set", len(iteration_set))
    logger.info("=" * 72)

    all_centroids = db.get_all_doc_embeddings_centroids()
    logger.info("Loaded %d document centroids for neighbour search", len(all_centroids))

    # Log centroid coverage vs iteration set
    centroid_ids = {dvid for dvid, _ in all_centroids}
    iteration_with_centroid = set(iteration_set) & centroid_ids
    iteration_without_centroid = set(iteration_set) - centroid_ids
    logger.info("Per-doc: %d/%d iteration docs have existing centroids",
                len(iteration_with_centroid), len(iteration_set))
    if iteration_without_centroid:
        logger.info("Per-doc: %d iteration docs have no centroid yet (will be computed)",
                     len(iteration_without_centroid))

    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0
    progress_interval = ncfg.progress_log_interval

    for idx, dvid in enumerate(iteration_set):
        if idx > 0 and idx % progress_interval == 0:
            elapsed = time.monotonic() - phase_start
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info("Per-doc progress: %d/%d processed (ok=%d failed=%d blocked=%d skipped=%d) "
                        "[%.1f docs/s, elapsed %.1fs]",
                        idx, len(iteration_set), ok_count, failed_count, blocked_count, skipped_count,
                        rate, elapsed)
        prereqs_ok, block_msg = db.check_prerequisites(dvid)
        if not prereqs_ok:
            assert block_msg is not None
            logger.debug("Per-doc: doc=%s blocked: %s", dvid[:12], block_msg)
            existing = db.get_doc_stage_status(dvid, STAGE_NAME)
            if existing and existing.status == "blocked" and existing.error_message == block_msg:
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

        try:
            doc_version = db.get_document_version(dvid)
            if doc_version is None:
                raise DBError(f"document_version not found: {dvid}")

            chunks = db.get_chunks_by_doc(dvid)
            chunk_embeddings = db.get_chunk_embeddings_by_doc(dvid)
            mentions = db.get_mentions_by_doc(dvid)

            logger.debug("Per-doc: doc=%s loaded %d chunks, %d embeddings, %d mentions",
                         dvid[:12], len(chunks), len(chunk_embeddings), len(mentions))

            result = process_document_novelty(
                doc_version=doc_version,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                mentions=mentions,
                all_centroids=all_centroids,
                neighbour_mentions_fn=db.get_mentions_by_doc,
                neighbour_chunk_embeddings_fn=lambda did: _get_neighbour_chunk_embeddings(db, did),
                ncfg=ncfg,
                run_id=run_id,
            )

            with db.transaction():
                _write_per_doc_results(db, result, run_id, config_hash)
            ok_count += 1
            logger.debug("Per-doc: doc=%s → %s (confidence=%.3f, chunks=%d)",
                         dvid[:12], result.label, result.confidence, len(result.chunk_novelties))

        except Exception:
            logger.exception("Failed processing doc %s", dvid[:12])
            try:
                with db.transaction():
                    db.upsert_doc_stage_status(
                        doc_version_id=dvid,
                        stage=STAGE_NAME,
                        run_id=run_id,
                        config_hash=config_hash,
                        status="failed",
                        error_message=f"exception:{sys.exc_info()[1]}",
                    )
            except Exception:
                logger.exception("Could not write failed status for %s", dvid[:12])
            failed_count += 1

    phase_elapsed = time.monotonic() - phase_start
    total_processed = ok_count + failed_count + blocked_count + skipped_count
    throughput = total_processed / phase_elapsed if phase_elapsed > 0 else 0
    logger.info(
        "Per-doc phase complete: ok=%d failed=%d blocked=%d skipped=%d "
        "(total=%d, elapsed=%.1fs, throughput=%.1f docs/s)",
        ok_count, failed_count, blocked_count, skipped_count,
        total_processed, phase_elapsed, throughput,
    )

    # Log novelty label distribution for diagnostics
    try:
        label_rows = db.get_novelty_label_distribution()
        logger.info("Per-doc novelty label distribution:")
        for lr in label_rows:
            logger.info("  %s: %d documents", lr["label"], lr["cnt"])
    except Exception:
        logger.debug("Could not query novelty label distribution (non-fatal)")

    return ok_count, failed_count, blocked_count, skipped_count


def run_cluster_phase(
    db: Stage07DatabaseInterface,
    run_id: str,
    config_hash: str,
    ncfg: NoveltyConfig,
) -> None:
    """
    Execute the run-scoped story-clustering phase.

    Runs in a single transaction. On failure the transaction rolls back and
    the function re-raises.
    """
    cluster_start = time.monotonic()
    logger.info("=" * 72)
    logger.info("Stage 07 run-scoped clustering phase starting")
    logger.info("  cluster_threshold: %.4f", ncfg.cluster_threshold)
    logger.info("  cluster_method:    %s", CLUSTER_METHOD)
    logger.info("=" * 72)
    eligible_ids = set(db.get_novelty_ok_doc_ids())
    logger.info("Clustering: %d documents with stage_07_novelty='ok'", len(eligible_ids))
    if not eligible_ids:
        logger.warning("Clustering: NO eligible documents — nothing to cluster")

    all_centroids = db.get_all_doc_embeddings_centroids()
    logger.info("Clustering: loaded %d total centroids from all documents", len(all_centroids))

    centroid_ids = {dvid for dvid, _ in all_centroids}
    eligible_with_centroid = eligible_ids & centroid_ids
    eligible_without_centroid = eligible_ids - centroid_ids
    logger.info("Clustering: %d eligible docs have centroids, %d do NOT",
                len(eligible_with_centroid), len(eligible_without_centroid))
    if eligible_without_centroid:
        logger.warning("Clustering: eligible docs missing centroids (first 10): %s",
                        sorted(eligible_without_centroid)[:10])

    clusters = build_story_clusters(all_centroids, eligible_ids, ncfg.cluster_threshold)

    # Build centroid map ONCE outside the cluster loop
    centroid_map: dict[str, np.ndarray] = {dvid: c for dvid, c in all_centroids}

    with db.transaction():
        deleted = db.delete_story_clusters_for_run(run_id)
        if deleted:
            logger.info("Deleted %d prior story_cluster rows for run %s", deleted, run_id[:12])

        cluster_count = 0
        member_count = 0
        singleton_count = 0
        cluster_sizes: list[int] = []
        for cluster_nodes in clusters:
            if len(cluster_nodes) < 2:
                singleton_count += 1
                continue
            cluster_sizes.append(len(cluster_nodes))
            sorted_nodes = sorted(cluster_nodes)
            story_id = compute_sha256_id(run_id, *sorted_nodes)
            seed = sorted_nodes[0]

            db.insert_story_cluster(
                StoryClusterRow(
                    run_id=run_id,
                    story_id=story_id,
                    cluster_method=CLUSTER_METHOD,
                    seed_doc_version_id=seed,
                    summary_text=None,
                )
            )
            cluster_count += 1

            seed_centroid = centroid_map.get(seed)
            for dvid in sorted_nodes:
                dvid_centroid = centroid_map.get(dvid)
                score = (
                    cosine_similarity(seed_centroid, dvid_centroid)
                    if seed_centroid is not None and dvid_centroid is not None
                    else None
                )
                role = "seed" if dvid == seed else "member"
                db.insert_story_cluster_member(
                    StoryClusterMemberRow(
                        run_id=run_id,
                        story_id=story_id,
                        doc_version_id=dvid,
                        score=round(score, 4) if score is not None else None,
                        role=role,
                    )
                )
                member_count += 1

        details = json.dumps({
            "clusters": cluster_count,
            "members": member_count,
            "eligible_docs": len(eligible_ids),
            "singletons_skipped": singleton_count,
            "cluster_sizes": sorted(cluster_sizes, reverse=True)[:20],
        })
        db.upsert_run_stage_status(
            run_id=run_id,
            stage=CLUSTER_STAGE_NAME,
            config_hash=config_hash,
            status="ok",
            details=details,
        )

    cluster_elapsed = time.monotonic() - cluster_start
    logger.info("=" * 72)
    logger.info("Cluster phase complete (elapsed=%.1fs):", cluster_elapsed)
    logger.info("  Clusters (size >= 2): %d", cluster_count)
    logger.info("  Total members:        %d", member_count)
    logger.info("  Singletons dropped:   %d", singleton_count)
    logger.info("  Eligible docs:        %d", len(eligible_ids))
    logger.info("  Unclustered docs:     %d", len(eligible_ids) - member_count)
    if cluster_sizes:
        logger.info("  Cluster sizes (top 20): %s", sorted(cluster_sizes, reverse=True)[:20])
    logger.info("=" * 72)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    All novelty parameters can be overridden from the command line.
    Defaults are taken from the module-level ``DEFAULT_*`` constants.
    """
    parser = argparse.ArgumentParser(description="Stage 07: Novelty Detection")
    parser.add_argument("--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                        help="Pipeline run ID (SHA256 hex)")
    parser.add_argument("--config-dir", type=Path, default=Path("../../../config/"),
                        help="Directory containing config.yaml (default: ../../../config/)")
    parser.add_argument("--working-db", type=Path, default=Path("../../../database/processed_posts.db"),
                        help="Path to working database (default: ../../../database/processed_posts.db)")
    parser.add_argument("--output-dir", type=Path, default=Path("../../../output/processed/"),
                        help="Output directory (default: ../../../output/processed/)")
    parser.add_argument("--log-dir", type=Path, default=Path("../../../output/processed/logs/"),
                        help="Log directory (default: ../../../output/processed/logs/)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")

    # --- Novelty detection parameters (override module defaults) ---
    novelty_group = parser.add_argument_group("Novelty Detection Parameters")
    novelty_group.add_argument("--minhash-num-perm", type=int, default=None,
                               help=f"MinHash permutation count (default: {DEFAULT_MINHASH_NUM_PERM})")
    novelty_group.add_argument("--shingle-size", type=int, default=None,
                               help=f"Character shingle size (default: {DEFAULT_SHINGLE_SIZE})")
    novelty_group.add_argument("--neighbour-limit", type=int, default=None,
                               help=f"Max candidate neighbours per document (default: {DEFAULT_NEIGHBOUR_LIMIT})")
    novelty_group.add_argument("--new-threshold", type=float, default=None,
                               help=f"Similarity below which doc is 'new' (default: {DEFAULT_NEW_THRESHOLD})")
    novelty_group.add_argument("--update-threshold", type=float, default=None,
                               help=f"Similarity above which doc is 're_report' (default: {DEFAULT_UPDATE_THRESHOLD})")
    novelty_group.add_argument("--cluster-threshold", type=float, default=None,
                               help=f"Cosine threshold for story clustering (default: {DEFAULT_CLUSTER_THRESHOLD})")
    novelty_group.add_argument("--linking-window-days", type=int, default=None,
                               help=f"Temporal window for linking (default: {DEFAULT_LINKING_WINDOW_DAYS})")
    novelty_group.add_argument("--top-k-neighbours", type=int, default=None,
                               help=f"Top-k neighbours for detailed analysis (default: {DEFAULT_TOP_K_NEIGHBOURS_FOR_DETAIL})")
    novelty_group.add_argument("--chunk-novelty-new-threshold", type=float, default=None,
                               help=f"Chunk novelty score >= this → 'new' (default: {DEFAULT_CHUNK_NOVELTY_NEW_THRESHOLD})")
    novelty_group.add_argument("--chunk-novelty-overlap-threshold", type=float, default=None,
                               help=f"Chunk novelty score >= this → 'partial_overlap' (default: {DEFAULT_CHUNK_NOVELTY_OVERLAP_THRESHOLD})")
    novelty_group.add_argument("--evidence-snippet-length", type=int, default=None,
                               help=f"Max chars for evidence anchor snippet (default: {DEFAULT_EVIDENCE_SNIPPET_LENGTH})")
    novelty_group.add_argument("--progress-log-interval", type=int, default=None,
                               help=f"Log progress every N documents (default: {DEFAULT_PROGRESS_LOG_INTERVAL})")
    return parser.parse_args()


def _extract_cli_novelty_overrides(args: argparse.Namespace) -> dict:
    """Extract novelty parameter CLI overrides as a dict (None values included)."""
    return {
        "minhash_num_perm": args.minhash_num_perm,
        "shingle_size": args.shingle_size,
        "neighbour_limit": args.neighbour_limit,
        "new_threshold": args.new_threshold,
        "update_threshold": args.update_threshold,
        "cluster_threshold": args.cluster_threshold,
        "linking_window_days": args.linking_window_days,
        "top_k_neighbours_for_detail": args.top_k_neighbours,
        "chunk_novelty_new_threshold": args.chunk_novelty_new_threshold,
        "chunk_novelty_overlap_threshold": args.chunk_novelty_overlap_threshold,
        "evidence_snippet_length": args.evidence_snippet_length,
        "progress_log_interval": args.progress_log_interval,
    }


def main_stage_07_novelty() -> int:
    """
    Set main entry point for Stage 07 — Novelty Detection.

    :return: 0 on success, 1 on fatal error.
    """
    stage_start = time.monotonic()
    args = parse_args()
    if args.verbose:
        import logging as _logging
        _logging.getLogger().setLevel(_logging.DEBUG)

    logger.info("=" * 72)
    logger.info("Stage 07 starting")
    logger.info("  run_id:      %s", args.run_id)
    logger.info("  config_dir:  %s", args.config_dir.resolve())
    logger.info("  working_db:  %s", args.working_db.resolve())
    logger.info("  output_dir:  %s", args.output_dir.resolve())
    logger.info("  log_dir:     %s", args.log_dir.resolve())
    logger.info("  verbose:     %s", args.verbose)
    logger.info("=" * 72)

    # --- Load config ---
    config_path = args.config_dir / "config.yaml"
    logger.info("Loading config from: %s", config_path.resolve())
    config = load_config(config_path)
    config_hash = get_config_version(config)
    logger.info("Config loaded successfully (config_hash=%s)", config_hash)

    # --- Build novelty config from config + CLI overrides ---
    cli_overrides = _extract_cli_novelty_overrides(args)
    active_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    if active_overrides:
        logger.info("CLI overrides for novelty config: %s", active_overrides)
    else:
        logger.info("No CLI overrides for novelty config; using module defaults")
    ncfg = build_novelty_config(config, cli_overrides)

    logger.info("=" * 72)
    logger.info("Stage 07 Effective Configuration:")
    logger.info("  minhash_num_perm:                %d", ncfg.minhash_num_perm)
    logger.info("  shingle_size:                    %d", ncfg.shingle_size)
    logger.info("  neighbour_limit:                 %d", ncfg.neighbour_limit)
    logger.info("  new_threshold:                   %.4f", ncfg.new_threshold)
    logger.info("  update_threshold:                %.4f", ncfg.update_threshold)
    logger.info("  cluster_threshold:               %.4f  ← controls story clustering", ncfg.cluster_threshold)
    logger.info("  linking_window_days:             %d  (recorded; temporal filtering not yet applied)", ncfg.linking_window_days)
    logger.info("  top_k_neighbours_for_detail:     %d", ncfg.top_k_neighbours_for_detail)
    logger.info("  chunk_novelty_new_threshold:     %.4f", ncfg.chunk_novelty_new_threshold)
    logger.info("  chunk_novelty_overlap_threshold: %.4f", ncfg.chunk_novelty_overlap_threshold)
    logger.info("  evidence_snippet_length:         %d", ncfg.evidence_snippet_length)
    logger.info("  progress_log_interval:           %d", ncfg.progress_log_interval)
    logger.info("  --- Confidence formula constants ---")
    logger.info("  CONF_UPDATE_BASE:                %.2f", CONF_UPDATE_BASE)
    logger.info("  CONF_UPDATE_SCALE:               %.2f", CONF_UPDATE_SCALE)
    logger.info("  CONF_MENTION_PENALTY:            %.2f", CONF_MENTION_PENALTY)
    logger.info("  CONF_MENTION_MAX_COUNT:          %d", CONF_MENTION_MAX_COUNT)
    logger.info("  CONF_MENTION_FLOOR:              %.2f", CONF_MENTION_FLOOR)
    logger.info("  --- Semantic constants ---")
    logger.info("  CHUNK_NOVELTY_METHOD:            %s", CHUNK_NOVELTY_METHOD)
    logger.info("  CLUSTER_METHOD:                  %s", CLUSTER_METHOD)
    logger.info("  EVIDENCE_PURPOSE:                %s", EVIDENCE_PURPOSE_NOVELTY_ANCHOR)
    logger.info("=" * 72)

    # --- Open database ---
    logger.info("Opening database: %s", args.working_db.resolve())
    db = Stage07DatabaseInterface(args.working_db)
    try:
        db.open()
        logger.info("Database opened successfully")

        run_row = db.get_pipeline_run(args.run_id)
        if run_row is None:
            logger.error("pipeline_run not found for %s", args.run_id)
            return 1
        if run_row.status != "running":
            logger.error("pipeline_run %s has status '%s', expected 'running'", args.run_id, run_row.status)
            return 1
        logger.info("Pipeline run validated: run_id=%s status=%s config_version=%s",
                     args.run_id[:12], run_row.status, run_row.config_version)

        # --- Per-document phase ---
        ok, failed, blocked, skipped = run_per_doc_phase(db, args.run_id, config_hash, ncfg)

        attempted = ok + failed
        if attempted > 0 and ok == 0 and skipped == 0:
            logger.error("Systemic failure: all %d attempted documents failed", attempted)
            return 1

        # --- Run-scoped clustering phase ---
        try:
            run_cluster_phase(db, args.run_id, config_hash, ncfg)
        except Exception:
            logger.exception("Run-scoped clustering phase failed")
            try:
                with db.transaction():
                    db.upsert_run_stage_status(
                        run_id=args.run_id,
                        stage=CLUSTER_STAGE_NAME,
                        config_hash=config_hash,
                        status="failed",
                        error_message=f"exception:{sys.exc_info()[1]}",
                    )
            except Exception:
                logger.exception("Could not write failed run_stage_status")
            return 1

        stage_elapsed = time.monotonic() - stage_start
        logger.info("=" * 72)
        logger.info("Stage 07 completed successfully (total elapsed=%.1fs)", stage_elapsed)
        logger.info("  Per-doc results: ok=%d failed=%d blocked=%d skipped=%d", ok, failed, blocked, skipped)
        logger.info("=" * 72)
        return 0

    finally:
        db.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(main_stage_07_novelty())