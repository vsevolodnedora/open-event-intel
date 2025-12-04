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

async def main_scrape_acer_posts(
    root_url: str,
    database: PostsDatabase,
    table_name:str
) -> None:
    """Scrape acer news posts from news-and-engagement database."""
    async with AsyncWebCrawler() as crawler:
        url_filter = URLPatternFilter(patterns=["*news*"])
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                filter_chain=FilterChain([url_filter]),
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            verbose=True,
        )

        results = await crawler.arun(url=root_url, config=config)
        if len(results) == 1:
            logger.warning(f"Only one result found for {root_url}. Suspected limit.")

        logger.info(f"Crawled {len(results)} pages matching '*news*'")
        for result in results:
            logger.debug(f"\tCrawled {result.url}")

        new_articles = []
        for result in results:
            url = result.url
            if (
                fnmatch.fnmatch(url, "*news*")
                and "news-and-events" not in url
                and "news-and-engagement" not in url
                and "events-and-engagement" not in url
            ):
                url = url.split("?")[0]

                match = re.search(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", result.markdown)
                if not match:
                    logger.warning(f"No date found in {url}; skipping.")
                    continue

                if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=url)):
                    logger.info(f"Post already exists in the database. Skipping: {url}")
                    continue

                # parse date DD.MM.YYYY -> YYYY-MM-DD
                day, month, year = match.group().split(".")
                date_iso = f"{year}-{int(month):02d}-{int(day):02d} 12:00" # YYYY-MM-DD HH:MM
                title = url.rstrip("/").split("/")[-1].replace("-", "_")

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
        logger.info(
            f"Finished: {len(new_articles)} new articles out of {len(results)} crawled."
        )