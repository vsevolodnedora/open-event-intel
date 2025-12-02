import asyncio
import fnmatch
import re

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
)

from src.database import PostsDatabase
from src.logger import get_logger
from src.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

async def main_scrape_ec_posts(root_url:str, table_name:str, database: PostsDatabase) -> None:
    """Scrape posts from ec news page."""
    async with AsyncWebCrawler() as crawler:

        # Create a filter that only allows URLs with 'guide' in them
        url_filter_news = URLPatternFilter(patterns=["*/news/*"])
        url_filter_en = URLPatternFilter(patterns=["*_en"])

        # Chain them so all must pass (AND logic)
        filter_chain = FilterChain([
            url_filter_news,
            url_filter_en,
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
        # date_pattern = re.compile(r"https?://[^ ]*/\d{4}-\d{2}-\d{2}[^ ]*") # to remove non-articles entries

        logger.info(f"Crawled {len(results)} pages matching '*news*'")
        new_articles = []
        for result in results:  # Show first 3 results
            url = result.url

            if database.is_table(table_name=table_name) and database.is_post(table_name=table_name, post_id=database.create_post_id(post_url=url)):
                logger.info(f"Post already exists in the database. Skipping: {url}")
                continue

            if fnmatch.fnmatch(url, "*news*") and "news_en" not in url:
                # Extract the title and date from the URL
                match = re.match(r"(.+)-(\d{4}-\d{2}-\d{2})_en", url.split("/")[-1])
                if not match:
                    raise ValueError(f"URL format is unexpected. No match for date is found. URL: {url}")

                title = match.group(1)
                date_iso = match.group(2)

                # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
                published_on = format_date_to_datetime(date_iso)

                # Replace hyphens with underscores in the title for readability
                title = title.replace("-", "_")

                # store full article in the database
                database.add_post(
                    table_name=table_name,
                    published_on=published_on,
                    title=title,
                    post_url=url,
                    post=result.markdown.raw_markdown,
                )

        await asyncio.sleep(5) # to avoid IP blocking

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")