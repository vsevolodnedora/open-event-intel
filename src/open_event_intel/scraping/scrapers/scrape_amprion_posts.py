import asyncio
import fnmatch
import re
from datetime import datetime

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
)

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

def find_and_format_numeric_date(text:str)->str|None:
    """Extract date from markdown."""
    pattern = r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"
    match = re.search(pattern, text)

    if not match:
        return None

    day, month, year = match.groups()
    return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")

async def main_scrape_amprion_posts(root_url:str, table_name:str, database: PostsDatabase, params: dict) -> None:
    """Scrape posts from amprion news page."""
    async with AsyncWebCrawler() as crawler:

        # Create a filter that only allows URLs with 'guide' in them
        url_filter_news = URLPatternFilter(patterns=["*Presse*"])

        # Chain them so all must pass (AND logic)
        filter_chain = FilterChain([
            url_filter_news,
        ])

        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                filter_chain=filter_chain,  # Single filter
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            verbose=True,
        )

        # collect all results from the webpage
        results = await crawler.arun(url=root_url, config=config)
        if len(results) == 1:
            logger.warning(f"Only one result found for {root_url}. Suspected limit.")
        # date_pattern = re.compile(r"https?://[^ ]*/\d{4}-\d{2}-\d{2}[^ ]*") # to remove non-articles entries

        logger.info(f"Crawled {len(results)} pages matching '*news*'")
        new_articles = []
        for result in results:  # Show first 3 results
            url = result.url

            # check for post in the database before trying to pull it as it is long
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

            if fnmatch.fnmatch(url, "*Presse*") and not fnmatch.fnmatch(url, "*Pressemitteilungen-[0-9][0-9][0-9][0-9]*") and not url.endswith(".jpg") and not url.endswith(".pdf"):

                date_iso = find_and_format_numeric_date(text=result.markdown.raw_markdown)
                if date_iso is None:
                    logger.warning(f"No date found. Skipping: {url}")
                    continue

                # Replace hyphens with underscores in the title for readability
                title = url.split("/")[-1].replace("-", "_")
                title = title.replace(".html", "")

                # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
                published_on = format_date_to_datetime(date_iso)

                # store full article in the database
                database.add_publication(
                    table_name=table_name,
                    published_on=published_on,
                    title=title,
                    post_url=url,
                    language=params["language"],
                    post=result.markdown.raw_markdown,
                    overwrite=params["overwrite"]
                )

        await asyncio.sleep(5) # to avoid IP blocking

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")