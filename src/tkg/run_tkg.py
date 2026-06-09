#!/usr/bin/env python3
"""
Runner for the Temporal Knowledge Graph (TKG) extraction + invalidation pipeline.

Reads preprocessed publications for a single publisher from
``preprocessed_posts.db``, extracts statements / events / triplets / entities
via the temporal agent, resolves entities, runs invalidation, and writes the
results into the TKG database plus human-readable / public-view exports.

Prompts and label/predicate definitions are NOT part of the public repository
(see README). Link them into ``src/tkg/prompts_and_definitions/`` first, e.g.::

    bash src/tkg/link_prompts_and_definitions.sh

and ensure ``OPENAI_API_KEY`` is set for the LLM-backed agents.

Examples::

    # Process all ENTSO-E publications
    python -m src.tkg.run_tkg --publisher entsoe

    # Smoke test: 3 publications, 5 statements each, rebuild the TKG DB
    python -m src.tkg.run_tkg --publisher eex --limit-publications 3 \\
        --limit-statements 5 --refresh-database
"""
import argparse
import asyncio
import sys
from pathlib import Path

from open_event_intel.logger import get_logger
from src.tkg.config import Config
from src.tkg.extraction_invalidation_pipeline import main_tkg_pipeline
from src.tkg.prompt_registry import PromptRegistry

logger = get_logger(__name__)


def build_config(args: argparse.Namespace) -> Config:
    """Build a TKG ``Config``, applying any path overrides from the CLI."""
    config = Config()
    if args.preprocessed_db:
        config.preprocessed_db_fpath = args.preprocessed_db
    if args.tkg_db:
        config.tkg_db_fpath = args.tkg_db
    if args.prompts_path:
        config.prompts_path = args.prompts_path
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the TKG extraction + invalidation pipeline for one publisher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--publisher",
        "-p",
        required=True,
        help="Publisher table name in preprocessed_posts.db (e.g. entsoe, eex, acer)",
    )
    parser.add_argument(
        "--limit-publications",
        type=int,
        default=None,
        help="Process at most N publications (default: all)",
    )
    parser.add_argument(
        "--limit-statements",
        type=int,
        default=None,
        help="Extract at most N statements per publication (default: all)",
    )
    parser.add_argument(
        "--refresh-database",
        action="store_true",
        help="Rebuild the TKG database from scratch instead of appending",
    )
    parser.add_argument(
        "--preprocessed-db",
        default=None,
        help="Override path to preprocessed_posts.db (source database)",
    )
    parser.add_argument(
        "--tkg-db",
        default=None,
        help="Override path to the TKG database",
    )
    parser.add_argument(
        "--prompts-path",
        default=None,
        help="Override path to the prompts_and_definitions directory",
    )
    return parser.parse_args()


def main() -> int:
    """Set main entry point for the TKG pipeline."""
    args = parse_arguments()
    config = build_config(args)

    # Fail early with an actionable message if prompts/definitions are missing.
    if not Path(config.prompts_path).is_dir():
        logger.error(
            "Prompts directory not found: %s\n"
            "Prompts and definitions are private (see README). Link them with:\n"
            "  bash src/tkg/link_prompts_and_definitions.sh",
            config.prompts_path,
        )
        return 1

    prompt_registry = PromptRegistry(config.prompts_path)
    try:
        prompt_registry.validate_files(config.required_prompts_and_definitions)
    except FileNotFoundError as exc:
        logger.error("Missing required prompt/definition files: %s", exc)
        return 1

    if not Path(config.preprocessed_db_fpath).is_file():
        logger.error("Source database not found: %s", config.preprocessed_db_fpath)
        return 1

    logger.info(
        "Starting TKG pipeline: publisher=%s limit_publications=%s limit_statements=%s refresh=%s",
        args.publisher,
        args.limit_publications,
        args.limit_statements,
        args.refresh_database,
    )

    try:
        asyncio.run(
            main_tkg_pipeline(
                config=config,
                prompt_registry=prompt_registry,
                publisher=args.publisher,
                limit_publications=args.limit_publications,
                limit_n_statements=args.limit_statements,
                refresh_database=args.refresh_database,
            )
        )
    except Exception:
        logger.exception("TKG pipeline failed")
        return 2

    logger.info("TKG pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
