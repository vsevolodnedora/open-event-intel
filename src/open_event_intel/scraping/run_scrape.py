#!/usr/bin/env python3
"""
Web scraper runner for energy market news sources.

This script orchestrates the scraping of various energy market news sources,
stores them in a SQLite database, and exports the results as markdown files
and metadata JSON.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.scrape_acer_posts import main_scrape_acer_posts
from src.open_event_intel.scraping.scrapers.scrape_agora_posts import main_scrape_agora_posts
from src.open_event_intel.scraping.scrapers.scrape_amprion_posts import main_scrape_amprion_posts
from src.open_event_intel.scraping.scrapers.scrape_bnetza_posts import main_scrape_bnetza_posts
from src.open_event_intel.scraping.scrapers.scrape_ec_posts import main_scrape_ec_posts
from src.open_event_intel.scraping.scrapers.scrape_eex_posts import main_scrape_eex_posts
from src.open_event_intel.scraping.scrapers.scrape_energy_wire_posts import main_scrape_energy_wire_posts
from src.open_event_intel.scraping.scrapers.scrape_entsoe_posts import main_scrape_entsoe_posts
from src.open_event_intel.scraping.scrapers.scrape_fifty_hertz_posts import main_scrape_fifty_hertz_posts
from src.open_event_intel.scraping.scrapers.scrape_icis_posts import main_scrape_icis_posts
from src.open_event_intel.scraping.scrapers.scrape_smard_posts import main_scrape_smard_posts
from src.open_event_intel.scraping.scrapers.scrape_tennet_posts import main_scrape_tennet_posts
from src.open_event_intel.scraping.scrapers.scrape_transnetbw_posts import main_scrape_transnetbw_posts

logger = get_logger(__name__)


# Source configurations
SCRAPER_CONFIGS = {
    "entsoe": {
        "root_url": "https://www.entsoe.eu/news-events/",
        "scraper_func": main_scrape_entsoe_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "eex": {
        "root_url": "https://www.eex.com/en/newsroom/",
        "scraper_func": main_scrape_eex_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "acer": {
        "root_url": "https://www.acer.europa.eu/news-and-events/news",
        "scraper_func": main_scrape_acer_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "ec": {
        "root_url": "https://energy.ec.europa.eu/news_en",
        "scraper_func": main_scrape_ec_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "icis": {
        "root_url": "https://www.icis.com/explore/resources/news/",
        "scraper_func": main_scrape_icis_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "bnetza": {
        "root_url": "https://www.bundesnetzagentur.de/DE/Allgemeines/Aktuelles/start.html",
        "scraper_func": main_scrape_bnetza_posts,
        "params": {"language":"de", "overwrite":False},
    },
    "smard": {
        "root_url": "https://www.smard.de/home/energiemarkt-aktuell/energiemarkt-aktuell",
        "scraper_func": main_scrape_smard_posts,
        "params": {"language":"de", "overwrite":False},
    },
    "agora": {
        "root_url": "https://www.agora-energiewende.org/news-events",
        "scraper_func": main_scrape_agora_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "energy_wire": {
        "root_url": "https://www.cleanenergywire.org/news/",
        "scraper_func": main_scrape_energy_wire_posts,
        "params": {"language":"en", "overwrite":False},
    },
    "transnetbw": {
        "root_url": "https://www.transnetbw.de/de/newsroom/",
        "scraper_func": main_scrape_transnetbw_posts,
        "params": {"language":"de", "overwrite":False},
    },
    "tennet": {
        "root_url": "https://www.tennet.eu/de/news-de",
        "scraper_func": main_scrape_tennet_posts,
        "params": {"language":"de", "overwrite":False},
    },
    "fifty_hertz": {
        "root_url": "https://www.50hertz.com/de/Medien/",
        "scraper_func": main_scrape_fifty_hertz_posts,
        "params": {"default_date": "1990-01-01", "language":"de", "overwrite":False},
    },
    "amprion": {
        "root_url": "https://www.amprion.net/",
        "scraper_func": main_scrape_amprion_posts,
        "params": {"language":"de", "overwrite":False},
    },
}


def scrape_single_source(
        source_name: str,
        config: Dict[str, Any],
        db_path: Path,
        output_base_dir: Path,
        max_runtime: int,
) -> None:
    """
    Scrape a single news source and save results to database and markdown files.

    :param source_name: Name/identifier of the source
    :param config: Configuration dict containing root_url, scraper_func, and params
    :param db_path: Path to SQLite database
    :param output_base_dir: Base directory for output markdown files
    :param max_runtime: Maximum runtime in seconds before timeout
    """
    logger.info(f"Starting scraper for '{source_name}'...")

    # Prepare output directory
    source_output_dir = output_base_dir / source_name
    source_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database
    news_db = PostsDatabase(db_path=str(db_path))
    news_db.check_create_table(source_name)

    async def runner():
        """Async wrapper for the scraper function."""
        await asyncio.wait_for(
            config["scraper_func"](
                root_url=config["root_url"],
                table_name=source_name,
                database=news_db,
                params=config["params"],
            ),
            timeout=max_runtime,
        )

    try:
        asyncio.run(runner())
        logger.info(f"Successfully scraped '{source_name}'")
    except asyncio.TimeoutError:
        logger.error(
            f"Scraper for '{source_name}' timed out after {max_runtime} seconds"
        )
    except Exception as e:
        logger.error(
            f"Failed to run scraper for '{source_name}': {e}",
            exc_info=True,
        )
    finally:
        # Save scraped posts as markdown files
        news_db.dump_publications_as_markdown(
            table_name=source_name,
            out_dir=str(source_output_dir),
        )
        news_db.close()


def export_metadata(db_path: Path, output_dir: Path, filename: str = "scraped_publications_metadata") -> None:
    """
    Export metadata for all scraped publications.

    :param db_path: Path to SQLite database
    :param output_dir: Directory to save metadata file
    :param filename: Base filename (without extension) for metadata

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    news_db = PostsDatabase(db_path=str(db_path))
    news_db.export_all_publications_metadata(
        out_dir=str(output_dir),
        format="json",
        filename=filename,
    )
    news_db.close()

    logger.info(f"Exported metadata to {output_dir / f'{filename}.json'}")


def scrape_sources(
        sources: list[str],
        db_path: Path,
        output_base_dir: Path,
        metadata_output_dir: Path,
        max_runtime: int,
) -> None:
    """
    Scrape one or more news sources.

    :param sources: List of source names to scrape
    :param db_path: Path to SQLite database
    :param output_base_dir: Base directory for raw markdown outputs
    :param metadata_output_dir: Directory for metadata export
    :param max_runtime: Maximum runtime per scraper in seconds
    """
    # Validate sources
    invalid_sources = [s for s in sources if s not in SCRAPER_CONFIGS]
    if invalid_sources:
        raise ValueError(
            f"Unknown sources: {', '.join(invalid_sources)}. "
            f"Valid options: {', '.join(SCRAPER_CONFIGS.keys())}"
        )

    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Scrape each source
    for source_name in sources:
        config = SCRAPER_CONFIGS[source_name]
        scrape_single_source(
            source_name=source_name,
            config=config,
            db_path=db_path,
            output_base_dir=output_base_dir,
            max_runtime=max_runtime,
        )

        # Export updated metadata after each source
        export_metadata(db_path, metadata_output_dir)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape energy market news sources and store in database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available sources:
  {', '.join(SCRAPER_CONFIGS.keys())}

Examples:
  # Scrape a single source
  python run_scrape.py --source icis

  # Scrape all sources
  python run_scrape.py --source all

  # Scrape multiple specific sources
  python run_scrape.py --source entsoe eex acer

  # Custom database and output paths
  python run_scrape.py --source all --db-path /path/to/db.sqlite \\
      --output-dir /path/to/raw --metadata-dir /path/to/metadata
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        nargs="+",
        default=["smard"],
        metavar="SOURCE",
        help='Source(s) to scrape or "all" for all sources',
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("../../../database/scraped_posts.db"),
        help="Path to SQLite database (default: ../../../database/scraped_posts.db)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../../output/posts_raw"),
        help="Base directory for raw markdown outputs (default: ../../../output/posts_raw)",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("../../../output/public_view"),
        help="Directory for metadata JSON export (default: ../../../output/public_view)",
    )

    parser.add_argument(
        "--max-runtime",
        type=int,
        default=18000,
        help="Maximum runtime per scraper in seconds (default: 1800)",
    )

    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available sources and exit",
    )

    return parser.parse_args()


def main() -> int:
    """Set main entry point for the scraper."""
    args = parse_arguments()

    # Handle --list-sources
    if args.list_sources:
        logger.info("Available sources:")
        for source in sorted(SCRAPER_CONFIGS.keys()):
            config = SCRAPER_CONFIGS[source]
            logger.info(f"  {source:15s} - {config['root_url']}")
        return 0

    # Expand "all" to all sources
    if "all" in args.source:
        sources = list(SCRAPER_CONFIGS.keys())
        logger.info("Scraping all sources")
    else:
        sources = args.source
        logger.info(f"Scraping sources: {', '.join(sources)}")

    try:
        scrape_sources(
            sources=sources,
            db_path=args.db_path,
            output_base_dir=args.output_dir,
            metadata_output_dir=args.metadata_dir,
            max_runtime=args.max_runtime,
        )
        logger.info("All scraping tasks completed successfully")
        return 0
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
