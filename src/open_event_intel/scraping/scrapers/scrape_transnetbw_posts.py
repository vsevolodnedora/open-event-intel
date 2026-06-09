import asyncio
import re
from datetime import datetime

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

GERMAN_MONTHS = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4,
    "Mai": 5, "Juni": 6, "Juli": 7, "August": 8,
    "September": 9, "Oktober": 10, "November": 11, "Dezember": 12,
}

GERMAN_DATE_RE = re.compile(
    r"\b(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|"
    r"Oktober|November|Dezember)\s+(\d{4})\b"
)

# Individual press releases live at /de/newsroom/pressemitteilungen/<slug>.
# The trailing segment (slug) is what distinguishes an article from the listing hub.
ARTICLE_RE = re.compile(r"/de/newsroom/pressemitteilungen/[^/?#]+$", re.IGNORECASE)


def _extract_article_links(result) -> list[str]:
    """Return de-duplicated individual press-release URLs from a listing crawl."""
    seen: set[str] = set()
    links: list[str] = []
    for link in (getattr(result, "links", None) or {}).get("internal", []):
        href = (link.get("href") or "").split("?", 1)[0].split("#", 1)[0].rstrip("/")
        if href and ARTICLE_RE.search(href) and href not in seen:
            seen.add(href)
            links.append(href)
    return links


async def main_scrape_transnetbw_posts(root_url: str, table_name: str, database: PostsDatabase, params: dict) -> None:
    """Scrape press releases from the TransnetBW newsroom.

    The newsroom landing page surfaces the most recent press releases as links
    to ``/de/newsroom/pressemitteilungen/<slug>``. We extract those links
    directly (the paginated listing page is JS-driven and does not expose the
    individual links server-side), then fetch and store each article.
    """
    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        listing = await crawler.arun(url=root_url, config=config)
        listing_result = listing[0] if isinstance(listing, list) else listing
        article_urls = _extract_article_links(listing_result)
        logger.info(f"Discovered {len(article_urls)} press-release links from {root_url}")

        new_articles: list[str] = []
        for url in article_urls:
            # check for post in the database before pulling it as it is long
            if (
                database is not None
                and database.is_table(table_name=table_name)
                and database.is_publication(
                    table_name=table_name,
                    publication_id=database.create_publication_id(post_url=url),
                )
                and not params["overwrite"]
            ):
                logger.info(f"Post already exists in the database for {table_name}. Skipping: {url} (overwrite={params['overwrite']})")
                continue

            results = await crawler.arun(url=url, config=config)
            result = results[0] if isinstance(results, list) else results
            if not result or not result.markdown:
                logger.warning(f"No crawl result for {url}")
                continue

            raw_md = result.markdown.raw_markdown

            # Extract the publication date (German long form) from the article body.
            match = GERMAN_DATE_RE.search(raw_md)
            if not match:
                logger.warning(f"Could not extract date in article in {url}")
                continue

            day, month_str, year = match.groups()
            date_iso = datetime(int(year), GERMAN_MONTHS[month_str], int(day)).strftime("%Y-%m-%d")

            # Replace hyphens with underscores in the title for readability
            title = url.split("/")[-1].replace("-", "_")

            # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
            published_on = format_date_to_datetime(date_iso)

            database.add_publication(
                table_name=table_name,
                published_on=published_on,
                title=title,
                post_url=url,
                language=params["language"],
                post=raw_md,
                overwrite=params["overwrite"],
            )
            new_articles.append(url)
            logger.info(f"Saved article {url}")

            await asyncio.sleep(5)  # to avoid hitting IP limits

    logger.info(f"Finished saving {len(new_articles)} new articles out of {len(article_urls)} discovered")
