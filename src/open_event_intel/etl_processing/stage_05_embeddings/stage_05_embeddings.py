"""
Stage 05 — Embedding computation and ANN index construction.

Two-phase stage:

1. **Per-doc phase** — iterates ``embedding_input_docs`` (``stage_02_parse='ok'``),
   computes chunk embeddings via :class:`LLMInterface`, and writes
   ``chunk_embedding`` rows plus ``doc_stage_status(stage_05_embeddings)``.

2. **Run-scoped phase** — builds an ANN index (hnswlib HNSW) over all
   ``eligible_embedding_docs`` (``stage_05_embeddings='ok'``), writes the
   ``embedding_index`` row and ``run_stage_status(stage_05_embeddings_index)``.

Exit codes per §6.5: 0 = both phases complete; 1 = fatal / run-scoped failure.
"""

import argparse
import array as _array
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from open_event_intel.etl_processing.config_interface import LLMConfig, ModelDefinition, get_config_version, load_config
from open_event_intel.etl_processing.database_interface import (
    ChunkEmbeddingRow,
    ChunkRow,
    EmbeddingIndexRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.llm_interface import EmbeddingResult, LLMError, LLMInterface
from open_event_intel.etl_processing.stage_05_embeddings.database_stage_05_embeddings import PREREQUISITE_STAGE, STAGE_NAME, STAGE_NAME_INDEX, Stage05DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults – used only when the config key is genuinely absent.
# All of these should normally come from config.yaml llm_config.embedding.
# ---------------------------------------------------------------------------
DEFAULT_CACHE_TTL_HOURS: int = 24

# ANN (HNSW) build parameters – declared here so they are visible/auditable.
# These can be promoted to config.yaml if tuning is needed.
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION_MIN: int = 48
HNSW_EF_CONSTRUCTION_MAX: int = 200
HNSW_SPACE: str = "cosine"


def serialize_embedding(vector: list[float]) -> bytes:
    """
    Pack a float vector into little-endian binary.

    :param vector: Embedding vector as a list of floats.
    :return: Packed bytes (``len(vector) * 4`` bytes, ``<f`` format).
    """
    return struct.pack(f"<{len(vector)}f", *vector)


def deserialize_embedding(data: bytes, dim: int) -> list[float]:
    """
    Unpack little-endian binary into a float vector.

    :param data: Packed bytes produced by :func:`serialize_embedding`.
    :param dim: Expected dimensionality.
    :return: Embedding vector as a list of floats.
    :raises ValueError: If byte length does not match *dim*.
    """
    expected = dim * 4
    if len(data) != expected:
        raise ValueError(f"Expected {expected} bytes for dim={dim}, got {len(data)}")
    return list(struct.unpack(f"<{dim}f", data))


def _resolve_embedding_model(llm_config: LLMConfig) -> ModelDefinition:
    """
    Resolve the single embedding model from config.

    :raises RuntimeError: If no embedding model is configured.
    """
    routing = llm_config.get_routing("embedding")
    model_name = routing.primary if routing else None

    if not model_name:
        for name, m in llm_config.models.items():
            if m.has_capability("embedding"):
                model_name = name
                break

    if not model_name:
        raise RuntimeError("No embedding model configured in llm_config")

    model_def = llm_config.get_model(model_name)
    if model_def is None:
        raise RuntimeError(f"Embedding model '{model_name}' not found in llm_config.models")

    logger.info(
        "Resolved embedding model: name=%s model_id=%s provider=%s dim=%s local=%s",
        model_def.name,
        model_def.model_id,
        model_def.provider.value,
        model_def.embedding_dim,
        model_def.is_local(),
    )
    return model_def


def _get_batch_size(llm_config: LLMConfig, model_def: ModelDefinition) -> int:
    """
    Return the configured batch size for embedding requests.

    Config values: ``llm_config.embedding.batch_size_local`` (default 8)
    and ``llm_config.embedding.batch_size_remote`` (default 64).
    """
    embedding_cfg = llm_config.embedding
    if model_def.is_local():
        batch_size = int(embedding_cfg.get("batch_size_local", 8))
    else:
        batch_size = int(embedding_cfg.get("batch_size_remote", 64))
    logger.info("Embedding batch size: %d (local=%s)", batch_size, model_def.is_local())
    return batch_size


def _should_normalize(llm_config: LLMConfig) -> bool:
    """
    Return whether embeddings should be L2-normalized.

    Config value: ``llm_config.embedding.normalize`` (default True).
    """
    normalize = bool(llm_config.embedding.get("normalize", True))
    logger.info("Embedding normalization: %s", normalize)
    return normalize


def _embed_chunks_for_doc(
    chunks: list[ChunkRow],
    llm: LLMInterface,
    model_def: ModelDefinition,
    batch_size: int,
    normalize: bool,
    run_id: str,
) -> list[ChunkEmbeddingRow]:
    """
    Compute embeddings for a document's chunks and return rows.

    :param chunks: Chunks to embed (non-empty).
    :param llm: LLM interface for API calls.
    :param model_def: Resolved embedding model definition.
    :param batch_size: Number of texts per API call.
    :param normalize: Whether to L2-normalize vectors.
    :param run_id: Current pipeline run ID.
    :return: One :class:`ChunkEmbeddingRow` per input chunk.
    :raises LLMError: On embedding API failure or dimension mismatch.
    """
    if not chunks:
        return []

    texts = [c.chunk_text for c in chunks]
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(
        "Embedding %d chunks in %d batches (batch_size=%d, normalize=%s)",
        len(chunks), total_batches, batch_size, normalize,
    )

    for batch_idx, start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start : start + batch_size]
        logger.debug(
            "  Batch %d/%d: %d texts, first text (truncated): %.120s…",
            batch_idx + 1, total_batches, len(batch), batch[0],
        )
        result: EmbeddingResult = llm.embed(
            texts=batch,
            model_def=model_def,
            normalize=normalize,
            purpose="embedding",
        )
        if model_def.embedding_dim and result.embedding_dim != model_def.embedding_dim:
            raise LLMError(
                f"Dimension mismatch: config says {model_def.embedding_dim}, "
                f"API returned {result.embedding_dim}"
            )
        all_embeddings.extend(result.embeddings)
        logger.debug(
            "  Batch %d/%d complete: dim=%d, tokens=%d, cost=$%.6f, latency=%dms",
            batch_idx + 1, total_batches,
            result.embedding_dim, result.tokens_in, result.cost, result.latency_ms,
        )

    if len(all_embeddings) != len(chunks):
        raise LLMError(
            f"Embedding count mismatch: {len(chunks)} chunks, "
            f"{len(all_embeddings)} embeddings returned"
        )

    dim = model_def.embedding_dim or len(all_embeddings[0])
    rows: list[ChunkEmbeddingRow] = []
    for chunk, vec in zip(chunks, all_embeddings):
        rows.append(
            ChunkEmbeddingRow(
                chunk_id=chunk.chunk_id,
                embedding_vector=serialize_embedding(vec),
                embedding_dim=dim,
                model_version=model_def.model_id,
                language_used=None,
                created_in_run_id=run_id,
            )
        )
    return rows


def run_per_doc_phase(
    db: Stage05DatabaseInterface,
    llm: LLMInterface,
    llm_config: LLMConfig,
    run_id: str,
    config_hash: str,
) -> tuple[int, int, int, int]:
    """
    Execute the per-document embedding phase.

    :return: ``(ok_count, failed_count, blocked_count, skipped_existing_count)``
    :raises SystemExit: If all attempted documents fail (systemic failure guard).
    """
    model_def = _resolve_embedding_model(llm_config)
    batch_size = _get_batch_size(llm_config, model_def)
    normalize = _should_normalize(llm_config)

    iteration_set = db.get_iteration_set()
    logger.info(
        "Per-doc phase: %d documents in iteration set, model=%s dim=%s "
        "batch_size=%d normalize=%s",
        len(iteration_set), model_def.name, model_def.embedding_dim,
        batch_size, normalize,
    )

    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_existing = 0

    for doc_idx, doc_version_id in enumerate(iteration_set, 1):
        prereq = db.check_prerequisite(doc_version_id)
        if prereq is None or prereq.status != "ok":
            blocking_status = prereq.status if prereq else "missing"
            logger.info(
                "[%d/%d] doc=%s BLOCKED (prerequisite %s=%s)",
                doc_idx, len(iteration_set), doc_version_id[:16],
                PREREQUISITE_STAGE, blocking_status,
            )
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="blocked",
                    error_message=f"prerequisite_not_ok:{PREREQUISITE_STAGE}:{blocking_status}",
                )
            blocked_count += 1
            continue

        existing = db.get_doc_stage_status(doc_version_id, STAGE_NAME)
        if existing and existing.status in ("ok", "skipped"):
            logger.debug(
                "[%d/%d] doc=%s SKIP (already status=%s)",
                doc_idx, len(iteration_set), doc_version_id[:16], existing.status,
            )
            skipped_existing += 1
            continue

        try:
            chunks = db.get_chunks_by_doc_version_id(doc_version_id)
            logger.info(
                "[%d/%d] doc=%s — %d chunks to embed",
                doc_idx, len(iteration_set), doc_version_id[:16], len(chunks),
            )

            if not chunks:
                logger.info(
                    "[%d/%d] doc=%s SKIPPED (no chunks)",
                    doc_idx, len(iteration_set), doc_version_id[:16],
                )
                with db.transaction():
                    db.upsert_doc_stage_status(
                        doc_version_id=doc_version_id,
                        stage=STAGE_NAME,
                        run_id=run_id,
                        config_hash=config_hash,
                        status="skipped",
                        error_message="no_chunks_for_document",
                    )
                ok_count += 1
                continue

            embedding_rows = _embed_chunks_for_doc(
                chunks, llm, model_def, batch_size, normalize, run_id,
            )

            with db.transaction():
                db.insert_chunk_embeddings(embedding_rows)
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="ok",
                )
            ok_count += 1
            logger.info(
                "[%d/%d] doc=%s OK — embedded %d chunks (dim=%d)",
                doc_idx, len(iteration_set), doc_version_id[:16],
                len(embedding_rows),
                embedding_rows[0].embedding_dim if embedding_rows else 0,
            )

        except Exception:
            logger.exception(
                "[%d/%d] doc=%s FAILED — embedding computation error",
                doc_idx, len(iteration_set), doc_version_id[:16],
            )
            try:
                with db.transaction():
                    db.upsert_doc_stage_status(
                        doc_version_id=doc_version_id,
                        stage=STAGE_NAME,
                        run_id=run_id,
                        config_hash=config_hash,
                        status="failed",
                        error_message="embedding_computation_error",
                    )
            except Exception:
                logger.exception(
                    "Failed to write failure status for doc %s", doc_version_id[:16],
                )
            failed_count += 1

    logger.info(
        "Per-doc phase complete: ok=%d failed=%d blocked=%d skipped_existing=%d "
        "(total_iteration_set=%d)",
        ok_count, failed_count, blocked_count, skipped_existing, len(iteration_set),
    )

    attempted = ok_count + failed_count
    if attempted > 0 and ok_count == 0 and failed_count == attempted:
        logger.error("Systemic failure: all %d attempted documents failed", attempted)
        raise SystemExit(1)

    return ok_count, failed_count, blocked_count, skipped_existing


def _check_all_input_docs_have_status(db: Stage05DatabaseInterface) -> bool:
    """Verify every ``embedding_input_doc`` has a stage 05 status row."""
    input_docs = set(db.get_embedding_input_docs())
    if not input_docs:
        logger.info("No embedding input docs found; nothing to verify")
        return True

    with_status = db.get_stage_doc_version_ids(STAGE_NAME)
    missing = input_docs - with_status
    if missing:
        logger.error(
            "%d embedding_input_docs lack a stage_05 status row (sample: %s)",
            len(missing),
            list(missing)[:5],
        )
        return False
    logger.info(
        "All %d embedding_input_docs have a stage_05 status row", len(input_docs),
    )
    return True


def build_ann_index(
    db: Stage05DatabaseInterface,
    run_id: str,
    config_hash: str,
    model_def: ModelDefinition,
    output_dir: Path,
) -> None:
    """
    Build the ANN index (run-scoped phase).

    Writes the index file first, then records metadata in ``embedding_index``
    and ``run_stage_status`` within a single transaction.

    :param db: Stage database interface.
    :param run_id: Current pipeline run ID.
    :param config_hash: Config version hash for audit.
    :param model_def: Embedding model definition.
    :param output_dir: Root output directory (``ann_indexes/`` subdirectory used).
    """
    logger.info("Run-scoped ANN index build starting: run_id=%s", run_id[:16])

    try:
        import hnswlib  # type: ignore[import-untyped]
    except ImportError:
        hnswlib = None
        logger.warning("hnswlib not available; will fall back to flat_numpy index")

    if np is None:
        raise RuntimeError("numpy is required for ANN index construction")

    eligible_embeddings = db.get_all_eligible_embeddings()
    chunk_count = len(eligible_embeddings)

    eligible_ok_docs = db.get_eligible_embedding_docs()
    status_counts = db.count_doc_stage_statuses(STAGE_NAME)
    details_json = json.dumps(
        {
            "embedded_ok": len(eligible_ok_docs),
            "indexed": chunk_count,
            "status_counts": status_counts,
        },
        sort_keys=True,
    )

    logger.info(
        "ANN index input: %d eligible docs, %d total embeddings, status_counts=%s",
        len(eligible_ok_docs), chunk_count, status_counts,
    )

    if chunk_count == 0:
        logger.info("No eligible embeddings to index; recording empty run-scoped status")
        with db.transaction():
            db.delete_embedding_indexes_for_run(run_id)
            db.upsert_run_stage_status(
                run_id=run_id,
                stage=STAGE_NAME_INDEX,
                config_hash=config_hash,
                status="ok",
                details=details_json,
            )
        return

    dim = eligible_embeddings[0][2]
    chunk_ids: list[str] = []
    vectors_flat = _array.array("f")
    for chunk_id, emb_bytes, emb_dim in eligible_embeddings:
        if emb_dim != dim:
            raise RuntimeError(
                f"Inconsistent embedding dim: expected {dim}, got {emb_dim} "
                f"for chunk {chunk_id}"
            )
        chunk_ids.append(chunk_id)
        vectors_flat.extend(struct.unpack(f"<{dim}f", emb_bytes))

    ann_dir = output_dir / "ann_indexes"
    ann_dir.mkdir(parents=True, exist_ok=True)

    method: str
    build_params: dict
    index_file: Path

    if hnswlib is not None:
        method = "hnsw"
        ef_construction = min(HNSW_EF_CONSTRUCTION_MAX, max(HNSW_EF_CONSTRUCTION_MIN, chunk_count))
        build_params = {
            "M": HNSW_M,
            "ef_construction": ef_construction,
            "space": HNSW_SPACE,
        }

        logger.info(
            "Building HNSW index: dim=%d chunks=%d M=%d ef_construction=%d space=%s",
            dim, chunk_count, HNSW_M, ef_construction, HNSW_SPACE,
        )

        index = hnswlib.Index(space=HNSW_SPACE, dim=dim)
        index.init_index(
            max_elements=chunk_count, ef_construction=ef_construction, M=HNSW_M,
        )

        data = np.frombuffer(vectors_flat, dtype=np.float32).reshape(chunk_count, dim)
        index.add_items(data, list(range(chunk_count)))
        ef_search = min(HNSW_EF_CONSTRUCTION_MAX, max(HNSW_EF_CONSTRUCTION_MIN, chunk_count))
        index.set_ef(ef_search)

        index_file = ann_dir / f"{run_id}_hnsw.bin"
        index.save_index(str(index_file))
        logger.info("HNSW index saved: %s (ef_search=%d)", index_file, ef_search)
    else:
        method = "flat_numpy"
        build_params = {"note": "hnswlib unavailable; flat fallback"}

        logger.info(
            "Building flat numpy index: dim=%d chunks=%d", dim, chunk_count,
        )

        data = np.frombuffer(vectors_flat, dtype=np.float32).reshape(chunk_count, dim)
        index_file = ann_dir / f"{run_id}_flat.npz"
        np.savez_compressed(str(index_file), vectors=data)
        logger.info("Flat numpy index saved: %s", index_file)

    id_map_file = index_file.with_suffix(".ids.json")
    id_map_file.write_text(
        json.dumps(chunk_ids, separators=(",", ":")), encoding="utf-8",
    )
    logger.info("ID map written: %s (%d entries)", id_map_file, len(chunk_ids))

    file_hash = hashlib.sha256(index_file.read_bytes()).hexdigest()
    index_id = compute_sha256_id(run_id, model_def.model_id, str(dim), method)

    index_row = EmbeddingIndexRow(
        index_id=index_id,
        run_id=run_id,
        model_version=model_def.model_id,
        embedding_dim=dim,
        method=method,
        index_path=str(index_file),
        built_at=datetime.now(tz=timezone.utc),
        chunk_count=chunk_count,
        build_params_json=build_params,
        checksum=file_hash,
    )

    with db.transaction():
        db.delete_embedding_indexes_for_run(run_id)
        db.insert_embedding_index(index_row)
        db.upsert_run_stage_status(
            run_id=run_id,
            stage=STAGE_NAME_INDEX,
            config_hash=config_hash,
            status="ok",
            details=details_json,
        )

    logger.info(
        "ANN index built: method=%s chunks=%d dim=%d path=%s checksum=%s",
        method, chunk_count, dim, index_file, file_hash[:16],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Stage 05."""
    parser = argparse.ArgumentParser(description="Stage 05: Embeddings")
    parser.add_argument(
        "--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        help="Pipeline run ID (SHA-256 hex, required)",
    )
    parser.add_argument("--config-dir", type=Path, default=Path("../../../config/"))
    parser.add_argument(
        "--source-db", type=Path,
        default=Path("../../../database/preprocessed_posts.db"),
    )
    parser.add_argument(
        "--working-db", type=Path,
        default=Path("../../../database/processed_posts.db"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../../output/processed/"),
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("../../../output/processed/logs/"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main_stage_05_embeddings() -> int:
    """
    Set main entry point for Stage 05 Embeddings.

    :return: 0 on success, 1 on fatal error.
    """
    args = parse_args()

    config = load_config(args.config_dir / "config.yaml")
    config_hash = get_config_version(config)

    model_def = _resolve_embedding_model(config.llm_config)

    # Log all effective configuration for visual validation
    embedding_cfg = config.llm_config.embedding
    cache_ttl_hours = int(config.llm_config.defaults.get("cache_ttl_hours", DEFAULT_CACHE_TTL_HOURS))
    logger.info(
        "Stage 05 starting: run_id=%s config_hash=%s model=%s dim=%s "
        "provider=%s batch_local=%s batch_remote=%s normalize=%s "
        "cache_ttl_hours=%d",
        args.run_id[:16],
        config_hash,
        model_def.name,
        model_def.embedding_dim,
        model_def.provider.value,
        embedding_cfg.get("batch_size_local", 8),
        embedding_cfg.get("batch_size_remote", 64),
        embedding_cfg.get("normalize", True),
        cache_ttl_hours,
    )
    logger.info(
        "Stage 05 paths: working_db=%s output_dir=%s",
        args.working_db, args.output_dir,
    )
    logger.info(
        "Stage 05 ANN constants: HNSW_M=%d ef_construction_range=[%d,%d] space=%s",
        HNSW_M, HNSW_EF_CONSTRUCTION_MIN, HNSW_EF_CONSTRUCTION_MAX, HNSW_SPACE,
    )

    # Determine whether external (non-local) model calls are allowed.
    # External calls are disallowed when PII masking is enabled (content is sensitive).
    pii_enabled = (
        config.pii_masking.settings is not None
        and config.pii_masking.settings.enabled
    )
    allow_external = model_def.is_local() or not pii_enabled
    logger.info(
        "External calls allowed: %s (model_is_local=%s, pii_enabled=%s)",
        allow_external, model_def.is_local(), pii_enabled,
    )

    db = Stage05DatabaseInterface(args.working_db)
    try:
        db.open()

        run = db.get_pipeline_run(args.run_id)
        if run is None:
            logger.error("Pipeline run %s not found", args.run_id)
            return 1
        if run.status != "running":
            logger.error(
                "Pipeline run %s status is '%s', expected 'running'",
                args.run_id, run.status,
            )
            return 1
        logger.info(
            "Pipeline run validated: status=%s budget_spent=$%.4f",
            run.status, run.budget_spent,
        )

        llm = LLMInterface(
            config=config.llm_config,
            db=db,
            stage_name=STAGE_NAME,
            run_id=args.run_id,
            cache_ttl_hours=cache_ttl_hours,
            allow_external_calls=allow_external,
        )

        ok, failed, blocked, skipped = run_per_doc_phase(
            db, llm, config.llm_config, args.run_id, config_hash,
        )

        if not _check_all_input_docs_have_status(db):
            logger.warning(
                "Not all embedding_input_docs have a status row; "
                "proceeding with run-scoped phase",
            )

        try:
            build_ann_index(db, args.run_id, config_hash, model_def, args.output_dir)
        except Exception:
            logger.exception("Run-scoped ANN index build failed")
            try:
                with db.transaction():
                    db.upsert_run_stage_status(
                        run_id=args.run_id,
                        stage=STAGE_NAME_INDEX,
                        config_hash=config_hash,
                        status="failed",
                        error_message="ann_index_build_error",
                    )
            except Exception:
                logger.exception("Failed to record run-scoped failure status")
            return 1

        logger.info(
            "Stage 05 complete: per_doc(ok=%d fail=%d blocked=%d skip=%d) "
            "ann_index=ok",
            ok, failed, blocked, skipped,
        )
        return 0

    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception:
        logger.exception("Stage 05 fatal error")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main_stage_05_embeddings())