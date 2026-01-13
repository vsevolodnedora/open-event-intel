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
from open_event_intel.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

def extract_and_format_date(markdown: str) -> str | None:
    """Extract date from markdown"""

    # Define regex pattern to match dates like "11 June 2024"
    date_pattern = r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})\b"

    # Find all matching dates
    matches = re.findall(date_pattern, markdown)

    if matches:
        # Take the last matched date
        day, month, year = matches[-1]
        # Convert to YYYY-MM-DD format
        date_obj = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
        return date_obj.strftime("%Y-%m-%d")
    else:
        return None

async def main_scrape_agora_posts(root_url:str, database: PostsDatabase, table_name:str, params: dict) -> None:
    """Scrape agora news articles from agora webpage"""

    async with AsyncWebCrawler() as crawler:

        url_filter_news = URLPatternFilter(patterns=["*/news-events/*"])

        # Chain them so all must pass (AND logic)
        filter_chain = FilterChain([
            url_filter_news,
        ])

        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=3,
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

        logger.info(f"Crawled {len(results)} pages matching '*news-events*'")
        new_articles = []
        for result in results:  # Show first 3 results
            url = result.url
            if fnmatch.fnmatch(url, "*news-events*") \
                    and url.replace(root_url, "") != "" \
                    and "/filter/" not in url \
                    and "/page/" not in url \
                    and "pdf" not in url:

                if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=url)):
                    logger.info(f"Post already exists in the database. Skipping: {url}")
                    continue

                logger.info(f"Processing {url}")
                date_iso = extract_and_format_date(result.markdown.raw_markdown) # Date in YYYY-MM-DD
                if date_iso is None:
                    logger.warning(f"Could not extract date for {url}")
                    continue
                url = url.split("?")[0]
                title_part = url.split("/")[-1].replace("-", "_")

                # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
                published_on = format_date_to_datetime(date_iso)

                # store full article in the database
                database.add_publication(
                    table_name="agora",
                    published_on=published_on,
                    title=title_part,
                    post_url=url,
                    language=params["language"],
                    post=result.markdown.raw_markdown,
                )

        await asyncio.sleep(5) # to avoid IP blocking

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")