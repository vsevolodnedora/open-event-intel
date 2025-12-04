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

from src.logger import get_logger
from src.publications_database import PostsDatabase
from src.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

async def main_scrape_icis_posts(root_url:str, database: PostsDatabase, table_name:str) -> None:
    """Scrape ICIS news posts from webpage."""
    async with AsyncWebCrawler() as crawler:

        url_filter_news = URLPatternFilter(patterns=["*/news/*"])

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

        # date_pattern = re.compile(r"https?://[^ ]*/\d{4}-\d{2}-\d{2}[^ ]*") # to remove non-articles entries

        year = 2025

        logger.info(f"Crawled {len(results)} pages matching '*news*'")
        new_articles = []
        for result in results:  # Show first 3 results
            url = result.url
            if fnmatch.fnmatch(result.url, "*news*") and str(year) in result.url and "news_id" not in result.url:

                # Extract the title and date from the URL
                match = re.match(r".*/news/(\d{4})/(\d{2})/(\d{2})/\d+/([\w-]+)", url)
                if not match:
                    raise ValueError("URL format is unexpected.")

                year, month, day = match.group(1), match.group(2), match.group(3)
                date_iso = f"{year}-{month}-{day}"  # e.g., 2025-07-07

                # Replace hyphens with underscores in the title for readability
                title_part = url.split("/")[-1]
                title = title_part.replace("-", "_")

                if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=url)):
                    logger.info(f"Post already exists in the database. Skipping: {url}")
                    continue

                # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
                published_on = format_date_to_datetime(date_iso)

                # store full article in the database
                database.add_publication(
                    table_name=table_name,
                    published_on=published_on,
                    title=title,
                    post_url=url,
                    post=result.markdown.raw_markdown,
                )
                new_articles.append(url)

        await asyncio.sleep(5) # to avoid IP blocking

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")