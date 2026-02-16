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

async def main_scrape_transnetbw_posts(root_url:str, table_name:str, database: PostsDatabase, params: dict) -> None:
    """Scrape posts from transnetbw news page."""
    async with AsyncWebCrawler() as crawler:

        # Create a filter that only allows URLs with 'guide' in them
        url_filter_news = URLPatternFilter(patterns=["*pressemitteilungen*"])

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

            min_url = "https://www.transnetbw.de/de/newsroom/pressemitteilungen"
            if fnmatch.fnmatch(url, "*pressemitteilungen*") and len(result.url) > len(min_url):
                # Extract the title and date from the URL
                german_months = {
                    "Januar": 1, "Februar": 2, "März": 3, "April": 4,
                    "Mai": 5, "Juni": 6, "Juli": 7, "August": 8,
                    "September": 9, "Oktober": 10, "November": 11, "Dezember": 12
                }

                pattern = r"\b(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+(\d{4})\b"
                match = re.search(pattern, result.markdown.raw_markdown)

                if not match:
                    logger.warning(f"Could not extract date in article in {url}")
                    continue

                # build date
                day, month_str, year = match.groups()
                month = german_months[month_str]
                date_iso = datetime(int(year), month, int(day)).strftime("%Y-%m-%d")

                # Replace hyphens with underscores in the title for readability
                title = url.split("/")[-1].replace("-", "_")

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
                    overwrite=params["overwrite"],
                )

        await asyncio.sleep(5) # to avoid hitting IP limits

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")