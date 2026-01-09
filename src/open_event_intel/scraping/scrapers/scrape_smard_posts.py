"""
Scraping SMARD news.

I had to fall back on processing each HTML manually since
information is commonly spread our between charts which themselves are not loaded.
In order to prevent overloading LLMs with useless chart technical messeges I remove them manually
using two lists of blacklisted strings as well as manually removing whole blocks of text that contain
references to "Highcharts"
"""

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

async def main_scrape_smard_posts(root_url:str, database: PostsDatabase, table_name:str, params: dict) -> None:

    known_bad_links = [
        "https://www.smard.de/page/home/topic-article/211972/214452/energietraegerscharfe-exporte-nach-laendern" # no text
    ]

    """Scrape acer news posts from news-and-engagement database."""
    async with AsyncWebCrawler() as crawler:

        # Create a filter that only allows URLs with 'guide' in them
        # Create one filter for each required pattern
        url_filter_news = URLPatternFilter(patterns=["*smard*"])
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

        # check if it exists in the database
        logger.info(f"Crawled {len(results)} pages matching '*news*'")

        # inform about which to process and which to skip
        count = 0
        for result in results:
            if "topic-article" in result.url:
                logger.debug(f"Found article {result.url}")
                count += 1
            else:
                logger.debug(f"Rejecting article {result.url}")
        logger.info(f"Crawled {count} pages matching '*news*' and selected {count} articles with 'topic-article' in url")

        new_articles = []
        for result in results:  # Show first 3 results

            url = result.url
            if url in known_bad_links:
                continue

            if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=url)):
                logger.info(f"Post already exists in the database '{url}' Skipping...")
                continue

            if fnmatch.fnmatch(result.url, "*topic-article*"):
                # Try to match "YYYY.MM.DD" format first
                full_date_match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", result.markdown)

                date_iso = "1990-01-01" # very far in the past; no current scrapes are expected to reach it
                if full_date_match:
                    day, month, year = full_date_match.groups()
                    date_iso = f"{year}-{month}-{day}"
                else:
                    # Try to match "DD Month YYYY" format
                    date_match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", result.markdown)
                    if date_match:
                        try:
                            parsed_date = datetime.strptime(f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}", "%d %B %Y")
                            date_iso = parsed_date.strftime("%Y-%m-%d")
                        except ValueError:
                            logger.warning(f"Date found but invalid date format in markdown for URL: {url}. Using dummy date.")
                    else:
                        logger.warning(f"Date not found in markdown for URL: {url}. Using dummy date.")

                # Extract the last segment of the URL for the title part
                title = url.split("/")[-1].replace("-", "_")

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