"""Evaluation module for temporal knowledge graph statement extraction."""

import json
import os
import re
import string
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openai import AsyncOpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz
from scipy.spatial.distance import cosine

from open_event_intel.data_models import Publication
from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase
from src.tkg.config import Config, LlmOptions
from src.tkg.data_models import (
    RawEntity,
    RawExtraction,
    RawStatement,
    RawStatementList,
    RawTemporalRange,
    RawTriplet,
    parse_date_str,
)
from src.tkg.extraction_agent import TemporalAgent, get_publication_metadata
from src.tkg.prompt_registry import PromptRegistry
from src.tkg.utils import create_file_name

logger = get_logger(__name__)


class StatementMatch(BaseModel):
    """Represents a match between golden and production statements."""

    golden_idx: int
    production_idx: int
    similarity_score: float
    exact_match: bool
    temporal_type_match: bool
    statement_type_match: bool
    temporal_confidence_match: bool  # Added field


class EvaluationMetrics(BaseModel):
    """Stores evaluation metrics for statement extraction."""

    precision: float
    recall: float
    f1_score: float
    exact_match_rate: float
    temporal_type_accuracy: float
    statement_type_accuracy: float
    temporal_confidence_accuracy: float  # Added field
    total_golden: int
    total_production: int
    matched_statements: int


class SampleOutputGeneration:
    """Use various agents to generate samples of extracted statements with all their properties."""

    def __init__(self, root_output_path: str, output_dir_name: str):
        """Initialize the sample generator."""
        self.output_path = root_output_path
        self.output_dir_name = output_dir_name

    async def generate_and_save_statements(
            self, agent: TemporalAgent, publication: Publication, limit_statements: int|None = None
    ) -> None:
        """Generate and save sample with temporal validity, triplets, and entities."""
        doc_summary = get_publication_metadata(publication)
        logger.info(
            f"Processing publication {publication.published_on} {publication.title}..."
        )

        outfpath = (
                self.output_path
                + f"{publication.publisher}/{self.output_dir_name}"
                + "/"
        )
        os.makedirs(outfpath, exist_ok=True)
        fname = create_file_name(publication) + ".txt"
        full_path = os.path.join(outfpath, fname)

        if os.path.isfile(full_path):
            logger.info(f"File {fname} already exists. Skipping...")
            return None

        # Call Extraction Pipeline
        statements_list: RawStatementList = await agent.extract_statements(
            publication, doc_summary
        )

        # Collect all statement data including temporal validity, triplets, and entities
        statements_data = []
        if limit_statements is not None:
            logger.info(f"Limiting processing of statements to {limit_statements} out of {len(statements_list.statements)} statements...")
            statements_to_process = statements_list.statements[:limit_statements]
        else:
            statements_to_process = statements_list.statements

        for i_stmt, stmt in enumerate(statements_to_process):
            logger.info(
                f"Processing {i_stmt}/{len(statements_to_process)} statement"
            )
            raw_validity, _, triplets, entities = await agent._process_statement(
                publication, stmt, doc_summary
            )

            # Format temporal validity dates
            valid_at_str = None
            invalid_at_str = None
            if raw_validity:
                if raw_validity.valid_at:
                    valid_at_str = raw_validity.valid_at
                if raw_validity.invalid_at:
                    invalid_at_str = raw_validity.invalid_at

            # Format triplets as human-readable strings
            triplets_list = [
                f"{t.subject_name} || {t.predicate} || {t.object_name}"
                for t in triplets
            ]

            # Format entities as human-readable strings
            entities_list = [f"{e.name} : {e.type}" for e in entities]

            # Build statement dictionary with all data
            stmt_dict = stmt.model_dump()
            stmt_dict["id"] = str(stmt_dict["id"])  # Convert UUID to string
            stmt_dict["valid_at"] = valid_at_str
            stmt_dict["invalid_at"] = invalid_at_str
            stmt_dict["valid_at_confidence"] = raw_validity.valid_at_confidence.value
            stmt_dict["invalid_at_confidence"] = raw_validity.invalid_at_confidence.value
            stmt_dict["rationale"] = raw_validity.rationale
            stmt_dict["triplets"] = triplets_list
            stmt_dict["entities"] = entities_list

            statements_data.append(stmt_dict)

        # Save text and extracted statements with extended data
        logger.info(f"Saving statements into {full_path}...")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write('"""\n')
            f.write(publication.text)
            f.write('\n"""\n\n')

            statements_dict = {"statements": statements_data}
            f.write(json.dumps(statements_dict, indent=2, ensure_ascii=False))
            f.write("\n")

        logger.info(
            f"Finished saving sample statements. Extracted {len(statements_to_process)} statements."
        )

    @staticmethod
    def load_sample_statements(
            golden_example_path: str, output_dir_name: str, publication: Publication
    ) -> list:
        """Load sample statements from files including temporal validity, triplets, and entities."""
        example_statements = []

        logger.info(
            f"Loading publication {publication.published_on}__{publication.title}..."
        )

        outfpath = (
                golden_example_path
                + f"{publication.publisher}/{output_dir_name}"
                + "/"
        )
        fname = create_file_name(publication) + ".txt"
        full_path = os.path.join(outfpath, fname)

        if not os.path.exists(full_path):
            logger.warning(f"Sample statements file not found: {full_path}")
            return []

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file: extract text between """ and JSON after
            parts = content.split('"""')
            if len(parts) >= 3:
                text_ = parts[1].strip()  # unused
                json_content = parts[2].strip()

                # Parse JSON
                statements_dict = json.loads(json_content)

                for stmt_data in statements_dict.get("statements", []):
                    # Extract statement with all fields
                    raw_statement = RawStatement(
                        id=stmt_data["id"],
                        statement=stmt_data["statement"],
                        temporal_type=stmt_data["temporal_type"],
                        statement_type=stmt_data["statement_type"],
                        publication_id=stmt_data["publication_id"],
                        temporal_confidence=stmt_data.get(
                            "temporal_confidence", "MEDIUM"
                        ),
                    )

                    # Extract temporal range with confidence levels
                    raw_temporal_range = RawTemporalRange(
                        valid_at=stmt_data.get("valid_at"),
                        invalid_at=stmt_data.get("invalid_at"),
                        valid_at_confidence=stmt_data.get(
                            "valid_at_confidence", "LOW"
                        ),
                        invalid_at_confidence=stmt_data.get(
                            "invalid_at_confidence", "LOW"
                        ),
                        rationale=stmt_data.get("rationale", ""),
                    )

                    # Extract triplets
                    triplets = []
                    for triplet_str in stmt_data.get("triplets", []):
                        # Parse "subject_name || predicate || object_name"
                        parts = [p.strip() for p in triplet_str.split("||")]
                        if len(parts) == 3:
                            triplets.append(
                                RawTriplet(
                                    subject_name=parts[0],
                                    subject_id=0,  # Placeholder ID
                                    predicate=parts[1],
                                    object_name=parts[2],
                                    object_id=0,  # Placeholder ID
                                    value=None,
                                )
                            )

                    # Extract entities
                    entities = []
                    for entity_str in stmt_data.get("entities", []):
                        # Parse "name : type"
                        parts = [p.strip() for p in entity_str.split(":", 1)]
                        if len(parts) == 2:
                            entities.append(
                                RawEntity(
                                    entity_idx=len(entities),
                                    name=parts[0],
                                    type=parts[1],
                                    description="",
                                )
                            )

                    # Create RawExtraction
                    raw_extraction = RawExtraction(
                        triplets=triplets, entities=entities
                    )

                    # Append to example samples
                    example_statements.append(
                        {
                            "statement": raw_statement,
                            "temporal_range": raw_temporal_range,
                            "extraction": raw_extraction,
                        }
                    )

                logger.info(
                    f"Loaded {len(statements_dict.get('statements', []))} statements from {fname}"
                )
            else:
                logger.error(f"Invalid file format in {full_path}")
        except Exception as e:
            logger.error(f"Error loading example statements from {full_path}: {e}")

        return example_statements


class EvaluateStatementExtraction:
    """Evaluator class that analyses how well the production LLM model extracts statements."""

    def __init__(
            self,
            root_data_path: str,
            golden_output_dir_name: str,
            production_output_dir_name: str,
            evaluation_dir_name: str,
    ) -> None:
        """Initialize the class."""
        self.root_data_path = root_data_path
        self.golden_output_dir_name = golden_output_dir_name
        self.production_output_dir_name = production_output_dir_name
        self.evaluation_dir_name = evaluation_dir_name
        self._client = AsyncOpenAI()
        self._embedding_model = Config().statement_embedding_model
        self._embedding_size = Config().statement_embedding_size

    @staticmethod
    def normalize_statement(text: str) -> str:
        """Normalize statement text for comparison."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Convert to lowercase for comparison
        text = text.lower()
        # Remove punctuation from the end
        text = text.rstrip(string.punctuation)
        return text

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
            dimensions=self._embedding_size,
        )
        return response.data[0].embedding

    def compute_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        return 1 - cosine(emb1, emb2)

    async def match_statements(
            self,
            golden_statements: list[RawStatement],
            production_statements: list[RawStatement],
            fuzzy_threshold: float = 70.0,
            semantic_threshold: float = 0.75,
    ) -> list[StatementMatch]:
        """
        Match golden and production statements using fuzzy matching and semantic similarity.

        Returns list of StatementMatch objects with matched pairs.
        """
        matches = []
        used_production_idx = set()

        # Get embeddings for all statements
        logger.info("Computing embeddings for statement matching...")
        golden_embeddings = []
        for stmt in golden_statements:
            emb = await self.get_embedding(stmt.statement)
            golden_embeddings.append(emb)

        production_embeddings = []
        for stmt in production_statements:
            emb = await self.get_embedding(stmt.statement)
            production_embeddings.append(emb)

        # Match each golden statement with best production statement
        for golden_idx, golden_stmt in enumerate(golden_statements):
            golden_norm = self.normalize_statement(golden_stmt.statement)
            best_match_idx = None
            best_score = 0.0
            best_fuzzy_score = 0.0
            best_semantic_score = 0.0

            for prod_idx, prod_stmt in enumerate(production_statements):
                if prod_idx in used_production_idx:
                    continue

                prod_norm = self.normalize_statement(prod_stmt.statement)

                # Compute fuzzy string matching score
                fuzzy_score = fuzz.ratio(golden_norm, prod_norm)

                # Compute semantic similarity
                semantic_score = (
                        self.compute_similarity(
                            golden_embeddings[golden_idx],
                            production_embeddings[prod_idx],
                        )
                        * 100
                )  # Scale to 0-100

                # Combined score (weighted average)
                combined_score = 0.4 * fuzzy_score + 0.6 * semantic_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = prod_idx
                    best_fuzzy_score = fuzzy_score
                    best_semantic_score = semantic_score

            # Only accept match if it exceeds thresholds
            if best_match_idx is not None and (
                    best_fuzzy_score >= fuzzy_threshold
                    or best_semantic_score >= semantic_threshold
            ):
                used_production_idx.add(best_match_idx)

                golden_norm = self.normalize_statement(golden_stmt.statement)
                prod_norm = self.normalize_statement(
                    production_statements[best_match_idx].statement
                )
                exact_match = golden_norm == prod_norm

                temporal_type_match = (
                        golden_stmt.temporal_type
                        == production_statements[best_match_idx].temporal_type
                )
                statement_type_match = (
                        golden_stmt.statement_type
                        == production_statements[best_match_idx].statement_type
                )
                temporal_confidence_match = (
                        golden_stmt.temporal_confidence
                        == production_statements[best_match_idx].temporal_confidence
                )

                matches.append(
                    StatementMatch(
                        golden_idx=golden_idx,
                        production_idx=best_match_idx,
                        similarity_score=best_score,
                        exact_match=exact_match,
                        temporal_type_match=temporal_type_match,
                        statement_type_match=statement_type_match,
                        temporal_confidence_match=temporal_confidence_match,
                    )
                )

        logger.info(
            f"Matched {len(matches)} statement pairs out of {len(golden_statements)} golden statements"
        )
        return matches

    def evaluate_temporal_ranges(
            self,
            golden_samples: list[dict],
            production_samples: list[dict],
            matches: list[StatementMatch],
            tolerance_days: int = 1,
    ) -> dict[str, Any]:
        """Evaluate temporal range extraction accuracy including confidence levels."""
        valid_at_exact = 0
        valid_at_tolerant = 0
        invalid_at_exact = 0
        invalid_at_tolerant = 0
        valid_at_total = 0
        invalid_at_total = 0

        # New: confidence metrics
        valid_at_confidence_match = 0
        invalid_at_confidence_match = 0

        for match in matches:
            golden_range = golden_samples[match.golden_idx]["temporal_range"]
            prod_range = production_samples[match.production_idx]["temporal_range"]

            # Evaluate valid_at
            if golden_range.valid_at is not None:
                valid_at_total += 1
                golden_valid = parse_date_str(golden_range.valid_at)
                prod_valid = parse_date_str(prod_range.valid_at)

                if golden_valid and prod_valid:
                    if golden_valid == prod_valid:
                        valid_at_exact += 1
                        valid_at_tolerant += 1
                    elif abs((golden_valid - prod_valid).days) <= tolerance_days:
                        valid_at_tolerant += 1

                # Check confidence match
                if golden_range.valid_at_confidence == prod_range.valid_at_confidence:
                    valid_at_confidence_match += 1

            # Evaluate invalid_at
            if golden_range.invalid_at is not None:
                invalid_at_total += 1
                golden_invalid = parse_date_str(golden_range.invalid_at)
                prod_invalid = parse_date_str(prod_range.invalid_at)

                if golden_invalid and prod_invalid:
                    if golden_invalid == prod_invalid:
                        invalid_at_exact += 1
                        invalid_at_tolerant += 1
                    elif abs((golden_invalid - prod_invalid).days) <= tolerance_days:
                        invalid_at_tolerant += 1

                # Check confidence match
                if (
                        golden_range.invalid_at_confidence
                        == prod_range.invalid_at_confidence
                ):
                    invalid_at_confidence_match += 1

        return {
            "valid_at_exact_match": (
                valid_at_exact / valid_at_total if valid_at_total > 0 else 0
            ),
            "valid_at_tolerant_match": (
                valid_at_tolerant / valid_at_total if valid_at_total > 0 else 0
            ),
            "invalid_at_exact_match": (
                invalid_at_exact / invalid_at_total if invalid_at_total > 0 else 0
            ),
            "invalid_at_tolerant_match": (
                invalid_at_tolerant / invalid_at_total if invalid_at_total > 0 else 0
            ),
            "valid_at_confidence_accuracy": (
                valid_at_confidence_match / valid_at_total if valid_at_total > 0 else 0
            ),
            "invalid_at_confidence_accuracy": (
                invalid_at_confidence_match / invalid_at_total
                if invalid_at_total > 0
                else 0
            ),
            "valid_at_count": valid_at_total,
            "invalid_at_count": invalid_at_total,
        }

    def evaluate_triplets(
            self,
            golden_samples: list[dict],
            production_samples: list[dict],
            matches: list[StatementMatch],
    ) -> dict[str, Any]:
        """Evaluate triplet extraction using precision, recall, and F1."""
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        predicate_correct = 0
        predicate_total = 0

        for match in matches:
            golden_triplets = golden_samples[match.golden_idx][
                "extraction"
            ].triplets
            prod_triplets = production_samples[match.production_idx][
                "extraction"
            ].triplets

            if not golden_triplets:
                continue

            # Create normalized representations
            golden_set = set()
            for t in golden_triplets:
                key = f"{t.subject_name.lower()}|{t.predicate}|{t.object_name.lower()}"
                golden_set.add(key)

            prod_set = set()
            for t in prod_triplets:
                key = f"{t.subject_name.lower()}|{t.predicate}|{t.object_name.lower()}"
                prod_set.add(key)

            # Calculate metrics
            true_positives = len(golden_set & prod_set)
            false_positives = len(prod_set - golden_set)
            false_negatives = len(golden_set - prod_set)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Check predicate accuracy
            for g_trip in golden_triplets:
                predicate_total += 1
                for p_trip in prod_triplets:
                    if (
                            g_trip.subject_name.lower() == p_trip.subject_name.lower()
                            and g_trip.object_name.lower() == p_trip.object_name.lower()
                    ):
                        if g_trip.predicate == p_trip.predicate:
                            predicate_correct += 1
                        break

        num_matches = len(matches)
        return {
            "precision": total_precision / num_matches if num_matches > 0 else 0,
            "recall": total_recall / num_matches if num_matches > 0 else 0,
            "f1_score": total_f1 / num_matches if num_matches > 0 else 0,
            "predicate_accuracy": (
                predicate_correct / predicate_total if predicate_total > 0 else 0
            ),
            "predicate_correct": predicate_correct,
            "predicate_total": predicate_total,
        }

    def evaluate_entities(
            self,
            golden_samples: list[dict],
            production_samples: list[dict],
            matches: list[StatementMatch],
    ) -> dict[str, Any]:
        """Evaluate entity extraction using precision, recall, and F1."""
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        type_correct = 0
        type_total = 0

        for match in matches:
            golden_entities = golden_samples[match.golden_idx][
                "extraction"
            ].entities
            prod_entities = production_samples[match.production_idx][
                "extraction"
            ].entities

            if not golden_entities:
                continue

            # Create normalized name sets
            golden_names = {e.name.lower() for e in golden_entities}
            prod_names = {e.name.lower() for e in prod_entities}

            # Calculate metrics
            true_positives = len(golden_names & prod_names)
            false_positives = len(prod_names - golden_names)
            false_negatives = len(golden_names - prod_names)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Check type accuracy for matched entities
            for g_ent in golden_entities:
                for p_ent in prod_entities:
                    if g_ent.name.lower() == p_ent.name.lower():
                        type_total += 1
                        if g_ent.type.lower() == p_ent.type.lower():
                            type_correct += 1
                        break

        num_matches = len(matches)
        return {
            "precision": total_precision / num_matches if num_matches > 0 else 0,
            "recall": total_recall / num_matches if num_matches > 0 else 0,
            "f1_score": total_f1 / num_matches if num_matches > 0 else 0,
            "type_accuracy": type_correct / type_total if type_total > 0 else 0,
            "type_correct": type_correct,
            "type_total": type_total,
        }

    async def evaluate(self, publication: Publication) -> None:
        """Evaluate the production sample statement against golden statement."""
        logger.info(f"Starting evaluation for publication: {publication.title}")

        eval_path = (
                self.root_data_path
                + publication.publisher
                + "/"
                + self.evaluation_dir_name
                + "/"
        )
        os.makedirs(eval_path, exist_ok=True)

        # Load golden and production samples
        logger.info("Loading golden and production samples...")
        golden_samples = SampleOutputGeneration.load_sample_statements(
            self.root_data_path, self.golden_output_dir_name, publication
        )
        production_samples = SampleOutputGeneration.load_sample_statements(
            self.root_data_path, self.production_output_dir_name, publication
        )

        if not golden_samples or not production_samples:
            logger.warning(
                f"No samples found for evaluation. Golden: {len(golden_samples)}, Production: {len(production_samples)}"
            )
            return

        # Extract statements for matching
        golden_statements = [s["statement"] for s in golden_samples]
        production_statements = [s["statement"] for s in production_samples]

        # Match statements
        logger.info("Matching statements...")
        matches = await self.match_statements(
            golden_statements, production_statements
        )

        # Calculate statement-level metrics
        matched_count = len(matches)
        exact_matches = sum(1 for m in matches if m.exact_match)
        temporal_type_correct = sum(1 for m in matches if m.temporal_type_match)
        statement_type_correct = sum(1 for m in matches if m.statement_type_match)
        temporal_confidence_correct = sum(
            1 for m in matches if m.temporal_confidence_match
        )

        statement_metrics = EvaluationMetrics(
            precision=(
                matched_count / len(production_statements)
                if production_statements
                else 0
            ),
            recall=(
                matched_count / len(golden_statements) if golden_statements else 0
            ),
            f1_score=(
                2
                * matched_count
                / (len(golden_statements) + len(production_statements))
                if (len(golden_statements) + len(production_statements)) > 0
                else 0
            ),
            exact_match_rate=(
                exact_matches / matched_count if matched_count > 0 else 0
            ),
            temporal_type_accuracy=(
                temporal_type_correct / matched_count if matched_count > 0 else 0
            ),
            statement_type_accuracy=(
                statement_type_correct / matched_count if matched_count > 0 else 0
            ),
            temporal_confidence_accuracy=(
                temporal_confidence_correct / matched_count
                if matched_count > 0
                else 0
            ),
            total_golden=len(golden_statements),
            total_production=len(production_statements),
            matched_statements=matched_count,
        )

        # Evaluate temporal ranges
        logger.info("Evaluating temporal ranges...")
        temporal_metrics = self.evaluate_temporal_ranges(
            golden_samples, production_samples, matches
        )

        # Evaluate triplets
        logger.info("Evaluating triplets...")
        triplet_metrics = self.evaluate_triplets(
            golden_samples, production_samples, matches
        )

        # Evaluate entities
        logger.info("Evaluating entities...")
        entity_metrics = self.evaluate_entities(
            golden_samples, production_samples, matches
        )

        # Compile results
        results = {
            "publication_metadata": {
                "id": publication.id,
                "title": publication.title,
                "publisher": publication.publisher,
                "published_on": publication.published_on.isoformat(),
                "url": publication.url,
            },
            "statement_extraction": {
                "precision": statement_metrics.precision,
                "recall": statement_metrics.recall,
                "f1_score": statement_metrics.f1_score,
                "exact_match_rate": statement_metrics.exact_match_rate,
                "temporal_type_accuracy": statement_metrics.temporal_type_accuracy,
                "statement_type_accuracy": statement_metrics.statement_type_accuracy,
                "temporal_confidence_accuracy": statement_metrics.temporal_confidence_accuracy,
                "total_golden": statement_metrics.total_golden,
                "total_production": statement_metrics.total_production,
                "matched_statements": statement_metrics.matched_statements,
            },
            "temporal_range_extraction": temporal_metrics,
            "triplet_extraction": triplet_metrics,
            "entity_extraction": entity_metrics,
            "matches": [
                {
                    "golden_idx": m.golden_idx,
                    "production_idx": m.production_idx,
                    "similarity_score": m.similarity_score,
                    "exact_match": m.exact_match,
                    "temporal_type_match": m.temporal_type_match,
                    "statement_type_match": m.statement_type_match,
                    "temporal_confidence_match": m.temporal_confidence_match,
                    "golden_statement": golden_statements[m.golden_idx].statement,
                    "production_statement": production_statements[
                        m.production_idx
                    ].statement,
                }
                for m in matches
            ],
        }

        # Save results
        fname = os.path.join(
            eval_path, create_file_name(publication) + "_evaluation.json"
        )
        logger.info(f"Saving evaluation results to {fname}")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Also save as CSV for easy analysis
        csv_fname = os.path.join(
            eval_path, create_file_name(publication) + "_summary.csv"
        )
        summary_data = {"Metric": [], "Value": []}

        # Add all metrics to CSV
        summary_data["Metric"].extend(
            [
                "Statement Precision",
                "Statement Recall",
                "Statement F1",
                "Exact Match Rate",
                "Temporal Type Accuracy",
                "Statement Type Accuracy",
                "Temporal Confidence Accuracy",
                "Valid_At Exact Match",
                "Valid_At Tolerant Match",
                "Valid_At Confidence Accuracy",
                "Invalid_At Exact Match",
                "Invalid_At Tolerant Match",
                "Invalid_At Confidence Accuracy",
                "Triplet Precision",
                "Triplet Recall",
                "Triplet F1",
                "Predicate Accuracy",
                "Entity Precision",
                "Entity Recall",
                "Entity F1",
                "Entity Type Accuracy",
            ]
        )

        summary_data["Value"].extend(
            [
                f"{statement_metrics.precision:.3f}",
                f"{statement_metrics.recall:.3f}",
                f"{statement_metrics.f1_score:.3f}",
                f"{statement_metrics.exact_match_rate:.3f}",
                f"{statement_metrics.temporal_type_accuracy:.3f}",
                f"{statement_metrics.statement_type_accuracy:.3f}",
                f"{statement_metrics.temporal_confidence_accuracy:.3f}",
                f"{temporal_metrics['valid_at_exact_match']:.3f}",
                f"{temporal_metrics['valid_at_tolerant_match']:.3f}",
                f"{temporal_metrics['valid_at_confidence_accuracy']:.3f}",
                f"{temporal_metrics['invalid_at_exact_match']:.3f}",
                f"{temporal_metrics['invalid_at_tolerant_match']:.3f}",
                f"{temporal_metrics['invalid_at_confidence_accuracy']:.3f}",
                f"{triplet_metrics['precision']:.3f}",
                f"{triplet_metrics['recall']:.3f}",
                f"{triplet_metrics['f1_score']:.3f}",
                f"{triplet_metrics['predicate_accuracy']:.3f}",
                f"{entity_metrics['precision']:.3f}",
                f"{entity_metrics['recall']:.3f}",
                f"{entity_metrics['f1_score']:.3f}",
                f"{entity_metrics['type_accuracy']:.3f}",
            ]
        )

        pd.DataFrame(summary_data).to_csv(csv_fname, index=False)
        logger.info(f"Saved summary to {csv_fname}")

        logger.info("Evaluation complete!")

    def visualize_results_for_one_publication(
            self, publication: Publication
    ) -> None:
        """Visualizes the previously saved results of the analysis."""
        eval_path = (
                self.root_data_path
                + publication.publisher
                + "/"
                + self.evaluation_dir_name
                + "/"
        )
        fname = os.path.join(
            eval_path, create_file_name(publication) + "_evaluation.json"
        )

        if not os.path.exists(fname):
            logger.error(f"Evaluation results not found: {fname}")
            return

        # Load results
        with open(fname, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. Statement Extraction Metrics (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        stmt_metrics = results["statement_extraction"]
        metrics_names = ["Precision", "Recall", "F1 Score"]
        metrics_values = [
            stmt_metrics["precision"],
            stmt_metrics["recall"],
            stmt_metrics["f1_score"],
        ]
        bars1 = ax1.bar(
            metrics_names, metrics_values, color=["#3498db", "#2ecc71", "#e74c3c"]
        )
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Score")
        ax1.set_title("Statement Extraction Performance")
        ax1.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars1, metrics_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                )

        # 2. Classification Accuracy (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        class_names = [
            "Exact\nMatch",
            "Temporal\nType",
            "Statement\nType",
            "Temporal\nConfidence",
        ]
        class_values = [
            stmt_metrics["exact_match_rate"],
            stmt_metrics["temporal_type_accuracy"],
            stmt_metrics["statement_type_accuracy"],
            stmt_metrics["temporal_confidence_accuracy"],
        ]
        bars2 = ax2.bar(
            class_names,
            class_values,
            color=["#9b59b6", "#f39c12", "#1abc9c", "#e91e63"],
        )
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Classification Accuracy")
        ax2.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars2, class_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                )

        # 3. Temporal Range Extraction (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        temp_metrics = results["temporal_range_extraction"]
        temp_names = [
            "Valid_At\nExact",
            "Valid_At\nTolerant",
            "Invalid_At\nExact",
            "Invalid_At\nTolerant",
        ]
        temp_values = [
            temp_metrics["valid_at_exact_match"],
            temp_metrics["valid_at_tolerant_match"],
            temp_metrics["invalid_at_exact_match"],
            temp_metrics["invalid_at_tolerant_match"],
        ]
        bars3 = ax3.bar(
            temp_names,
            temp_values,
            color=["#e67e22", "#f39c12", "#d35400", "#e74c3c"],
        )
        ax3.set_ylim(0, 1.0)
        ax3.set_ylabel("Match Rate")
        ax3.set_title("Temporal Range Accuracy")
        ax3.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars3, temp_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                )

        # 4. Confidence Level Accuracy (Second Row Left)
        ax4 = fig.add_subplot(gs[1, 0])
        conf_names = ["Valid_At\nConfidence", "Invalid_At\nConfidence"]
        conf_values = [
            temp_metrics.get("valid_at_confidence_accuracy", 0),
            temp_metrics.get("invalid_at_confidence_accuracy", 0),
        ]
        bars4 = ax4.bar(conf_names, conf_values, color=["#3498db", "#9b59b6"])
        ax4.set_ylim(0, 1.0)
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Temporal Confidence Accuracy")
        ax4.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars4, conf_values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                )

        # 5. Triplet Extraction Metrics (Second Row Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        trip_metrics = results["triplet_extraction"]
        trip_names = ["Precision", "Recall", "F1 Score", "Predicate\nAccuracy"]
        trip_values = [
            trip_metrics["precision"],
            trip_metrics["recall"],
            trip_metrics["f1_score"],
            trip_metrics["predicate_accuracy"],
        ]
        bars5 = ax5.bar(
            trip_names,
            trip_values,
            color=["#16a085", "#27ae60", "#2ecc71", "#229954"],
        )
        ax5.set_ylim(0, 1.0)
        ax5.set_ylabel("Score")
        ax5.set_title("Triplet Extraction Performance")
        ax5.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars5, trip_values):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                )

        # 6. Entity Extraction Metrics (Second Row Right)
        ax6 = fig.add_subplot(gs[1, 2])
        ent_metrics = results["entity_extraction"]
        ent_names = ["Precision", "Recall", "F1 Score", "Type\nAccuracy"]
        ent_values = [
            ent_metrics["precision"],
            ent_metrics["recall"],
            ent_metrics["f1_score"],
            ent_metrics["type_accuracy"],
        ]
        bars6 = ax6.bar(
            ent_names,
            ent_values,
            color=["#8e44ad", "#9b59b6", "#a569bd", "#bb8fce"],
        )
        ax6.set_ylim(0, 1.0)
        ax6.set_ylabel("Score")
        ax6.set_title("Entity Extraction Performance")
        ax6.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars6, ent_values):
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                )

        # 7. Coverage Overview (Third Row Left)
        ax7 = fig.add_subplot(gs[2, 0])
        total_golden = stmt_metrics["total_golden"]
        total_production = stmt_metrics["total_production"]
        matched = stmt_metrics["matched_statements"]

        categories = ["Golden\nStatements", "Production\nStatements", "Matched\nPairs"]
        values = [total_golden, total_production, matched]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        bars7 = ax7.bar(categories, values, color=colors)
        ax7.set_ylabel("Count")
        ax7.set_title("Statement Coverage")
        ax7.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars7, values):
            ax7.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontweight="bold",
                )

        # 8. Similarity Score Distribution (Third Row Middle)
        ax8 = fig.add_subplot(gs[2, 1])
        similarity_scores = [m["similarity_score"] for m in results["matches"]]
        if similarity_scores:
            ax8.hist(
                similarity_scores,
                bins=20,
                color="#3498db",
                alpha=0.7,
                edgecolor="black",
            )
            ax8.axvline(
                np.mean(similarity_scores),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(similarity_scores):.2f}",
            )
            ax8.set_xlabel("Similarity Score")
            ax8.set_ylabel("Frequency")
            ax8.set_title("Statement Similarity Distribution")
            ax8.legend()
            ax8.grid(axis="y", alpha=0.3)

        # 9. Overall Performance Radar Chart (Third Row Right)
        ax9 = fig.add_subplot(gs[2, 2], projection="polar")
        categories_radar = [
            "Statement\nExtraction",
            "Temporal\nType",
            "Statement\nType",
            "Temporal\nRange",
            "Triplet\nExtraction",
            "Entity\nExtraction",
        ]
        values_radar = [
            stmt_metrics["f1_score"],
            stmt_metrics["temporal_type_accuracy"],
            stmt_metrics["statement_type_accuracy"],
            (
                    temp_metrics["valid_at_tolerant_match"]
                    + temp_metrics["invalid_at_tolerant_match"]
            )
            / 2,
            trip_metrics["f1_score"],
            ent_metrics["f1_score"],
            ]

        # Close the plot
        values_radar = values_radar + values_radar[:1]
        angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()
        angles += angles[:1]

        ax9.plot(angles, values_radar, "o-", linewidth=2, color="#e74c3c")
        ax9.fill(angles, values_radar, alpha=0.25, color="#e74c3c")
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories_radar, size=8)
        ax9.set_ylim(0, 1)
        ax9.set_title("Overall Performance Overview", pad=20)
        ax9.grid(True)

        # 10. Summary Statistics (Bottom Span)
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis("off")

        summary_text = f"""
PUBLICATION DETAILS:
  • Title: {publication.title[:60]}...
  • Publisher: {publication.publisher}
  • Published: {publication.published_on.strftime('%Y-%m-%d')}

STATEMENT EXTRACTION:                    TEMPORAL RANGE EXTRACTION:              TRIPLET & ENTITY EXTRACTION:
  • Precision: {stmt_metrics["precision"]:.3f}               • Valid_At Exact: {temp_metrics["valid_at_exact_match"]:.3f}             • Triplet F1: {trip_metrics["f1_score"]:.3f}
  • Recall: {stmt_metrics["recall"]:.3f}                  • Valid_At Tolerant: {temp_metrics["valid_at_tolerant_match"]:.3f}          • Predicate Acc: {trip_metrics["predicate_accuracy"]:.3f}
  • F1 Score: {stmt_metrics["f1_score"]:.3f}                 • Invalid_At Exact: {temp_metrics["invalid_at_exact_match"]:.3f}           • Entity F1: {ent_metrics["f1_score"]:.3f}
  • Temporal Type Acc: {stmt_metrics["temporal_type_accuracy"]:.3f}       • Invalid_At Tolerant: {temp_metrics["invalid_at_tolerant_match"]:.3f}        • Entity Type Acc: {ent_metrics["type_accuracy"]:.3f}
  • Statement Type Acc: {stmt_metrics["statement_type_accuracy"]:.3f}      • Valid_At Conf Acc: {temp_metrics.get("valid_at_confidence_accuracy", 0):.3f}
  • Temporal Conf Acc: {stmt_metrics["temporal_confidence_accuracy"]:.3f}   • Invalid_At Conf Acc: {temp_metrics.get("invalid_at_confidence_accuracy", 0):.3f}

COVERAGE: Golden={total_golden} | Production={total_production} | Matched={matched} ({matched/total_golden*100:.1f}% recall)
        """

        ax10.text(
            0.05,
            0.95,
            summary_text,
            transform=ax10.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        # Main title
        fig.suptitle(
            f"Statement Extraction Evaluation Report\n{publication.publisher.upper()}",
            fontsize=16,
            fontweight="bold",
        )

        # Save figure
        plot_fname = os.path.join(
            eval_path, create_file_name(publication) + "_visualization.png"
        )
        plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {plot_fname}")
        plt.close()


async def main_evaluate_statement_extraction_pipeline(
        db_fpath: str,
        publisher: str | None,
        limit_publications: int | None,
        limit_statements: int | None,
        reference_model: LlmOptions,
        production_model: LlmOptions,
        suffix: str,
) -> None:
    """Evaluate statement generation."""
    # Get All Publications for one publisher
    if publisher is None:
        publisher = "entsoe"

    if not os.path.isfile(db_fpath):
        raise FileNotFoundError(
            f"Database with preprocessed publications is not found: {db_fpath}"
        )

    source_db = PostsDatabase(db_fpath)

    all_publications = source_db.list_publications(
        table_name=publisher, sort_date=True
    )
    if limit_publications is not None:
        selected_publications: list[Publication] = all_publications[:limit_publications]
    else:
        selected_publications: list[Publication] = all_publications

    source_db.close()

    # Initialize prompt registry for the agent
    prompt_registry = PromptRegistry(prompts_path=Config().prompts_path)
    prompt_registry.validate_files(filenames=Config().required_prompts_and_definitions)

    # Initialize generator class that will create golden sample of extracted statements
    eval_statements_root_path = Config().eval_statements_example_path

    production_sample_generation_config = Config()
    production_sample_generation_config.statement_extraction_model = production_model
    production_sample_generation_config.temporal_range_extraction_model = production_model
    production_sample_generation_config.triple_extraction_model = production_model
    production_generation_agent = TemporalAgent(production_sample_generation_config, prompt_registry)

    golden_sample_generation_config = Config()
    golden_sample_generation_config.statement_extraction_model = reference_model
    golden_sample_generation_config.temporal_range_extraction_model = reference_model

    golden_sample_generation_config.triple_extraction_model = reference_model
    golden_sample_generation_agent = TemporalAgent(golden_sample_generation_config, prompt_registry)

    golden_sample_generator = SampleOutputGeneration(
        eval_statements_root_path, output_dir_name="golden_sample_statements"
    )
    production_generator = SampleOutputGeneration(
        eval_statements_root_path, output_dir_name=f"{suffix}_sample_statements"
    )

    evaluator = EvaluateStatementExtraction(
        eval_statements_root_path,
        "golden_sample_statements",
        f"{suffix}_sample_statements",
        f"statement_extraction_eval_{suffix}",
    )

    # Create golden samples of extracted statements for human analysis
    for i_publication, publication in enumerate(selected_publications):
        logger.info(
            f"Generating samples for {i_publication}/{len(selected_publications)} publication"
        )

        # Generate golden sample of extracted samples (if they do not exist)
        await golden_sample_generator.generate_and_save_statements(
            golden_sample_generation_agent, publication, limit_statements=limit_statements
        )

        # Generate production sample of extractions (using production LLM) (if does not exist)
        await production_generator.generate_and_save_statements(
            production_generation_agent, publication, limit_statements=limit_statements
        )

        # Evaluate how well the statements were extracted with production model against golden sample
        await evaluator.evaluate(publication)

        # Visualize the result of evaluation
        evaluator.visualize_results_for_one_publication(publication)

    logger.info("Finished generating and evaluating statements for all publications.")

    return None
