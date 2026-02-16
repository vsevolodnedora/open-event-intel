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

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

def extract_date_from_markdown(markdown_text:str):
    """Extract date from markdown text."""

    # Split text into lines
    lines = markdown_text.splitlines()

    # Pattern to match the date format DD/MM/YYYY
    date_pattern = r"\b(\d{2}/\d{2}/\d{4})\b"

    date_str = ""
    for line in lines:
        if "EEX Press Release" in line or "Volume Report" in line:
            # Search for the date pattern in the line
            match = re.search(date_pattern, line)
            if match:
                date_str = match.group(1).replace("/", "-")

    if date_str == "":
        return None

    month, day, year = date_str.split("-")
    # Rearrange and return in YYYY-MM-DD format
    return f"{year}-{month}-{day}"

def invert_date_format(date_str:str):
    """Invert date format."""

    # Split the string by the dash
    year, month, day = date_str.split("-")
    # Rearrange and return in MM-DD-YYYY format
    return f"{month}-{day}-{year}"

async def main_scrape_eex_posts(root_url:str, table_name:str, database: PostsDatabase, params: dict) -> None:
    """
    Scrape EEX news posts.

    https://www.eex.com/en/newsroom/news?tx_news_pi1%5Bcontroller%5D=News&tx_news_pi1%5BcurrentPage%5D=2&tx_news_pi1%5Bsearch%5D%5BfilteredCategories%5D=&tx_news_pi1%5Bsearch%5D%5BfromDate%5D=&tx_news_pi1%5Bsearch%5D%5Bsubject%5D=&tx_news_pi1%5Bsearch%5D%5BtoDate%5D=&cHash=83e307337837c6f5bd5e40a530acad7a
    :param date_int:
    :param output_dir:
    :param clean_output_dir:
    :return:
    """

    async with (AsyncWebCrawler() as crawler):

        # Create a filter that only allows URLs with 'guide' in them
        url_filter = URLPatternFilter(patterns=["*_news_*"])

        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                filter_chain=FilterChain([url_filter]),  # Single filter
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            verbose=True,
        )

        # collect all results from the webpage
        results = await crawler.arun(url=root_url, config=config)
        if len(results) == 1:
            logger.warning(f"Only one result found for {root_url}. Suspected limit.")

        logger.info(f"Crawled {len(results)} pages matching '*_news_*'")
        new_articles = []
        for result in results:  # Show first 3 results
            url = result.url

            if fnmatch.fnmatch(url, "*_news_*") \
                    and ("EEX Press Release" in result.markdown.raw_markdown or "Volume Report" in result.markdown.raw_markdown) \
                    and "_news_" in url:

                # extract date from markdown
                date_iso = extract_date_from_markdown(result.markdown.raw_markdown) # YYYY-MM-DD

                if date_iso is None:
                    logger.debug(f"Skipping scraped markdown from {url}. Could not extract date from markdown.")
                    continue

                # select title based on the contenct
                if ("EEX Press Release" in result.markdown.raw_markdown):
                    title="eex_press_release"
                elif ("Volume Report" in result.markdown.raw_markdown):
                    title="volume_report"
                else:
                    title="unknown"

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

                # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
                published_on = format_date_to_datetime(date_iso)

                database.add_publication(
                    table_name=table_name,
                    published_on=published_on,
                    title=title,
                    post_url=url,
                    language=params["language"],
                    post=result.markdown.raw_markdown,
                    overwrite=params["overwrite"]
                )
                new_articles.append(url)

        await asyncio.sleep(5) # to avoid IP blocking

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(results)} articles")