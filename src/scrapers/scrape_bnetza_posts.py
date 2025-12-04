import asyncio
import re
import urllib.parse

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
)

from src.logger import get_logger
from src.publications_database import PostsDatabase
from src.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

async def main_scrape_bnetza_posts(
    root_url: str,
    database: PostsDatabase,
    table_name:str
) -> None:
    """
    Fetch the root page HTML, extract all links to BNetzA with news_href_part (press-releases) then scrape each article as Markdown using crawl4AI and save new ones.

    Special approach is needed due to old HTML page technology.
    """
    # Step 1: Download and parse root HTML
    async with aiohttp.ClientSession() as session:
        async with session.get(root_url) as response:
            response.raise_for_status()
            html = await response.text()

    soup = BeautifulSoup(html, "lxml")

    # Step 2: Collect all hrefs matching news_href_part
    links = set()
    root_url_=root_url.replace("DE/Allgemeines/Aktuelles/", "")  # remove part not in the news article link...
    news_href_part: str = "SharedDocs/Pressemitteilungen/DE/2025"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if news_href_part in href:
            # create full news artcile url
            full_url = urllib.parse.urljoin(root_url_, href)
            links.add(full_url)

    if not links:
        logger.error(f"No press-release links found at {root_url}")
        return

    logger.info(f"Found {len(links)} candidate press-release URLs")

    # Step 3: check if this URL already in the database and then skip this URL
    if (not database.is_table(table_name="bnetza")):
        new_links = links
    else:
        new_links = []
        for link in links:
            article_url = link.split("?", 1)[0]
            if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=article_url)):
                logger.info(f"Post already exists in the database. Skipping: {article_url}")
                continue
            new_links.append(article_url)

    logger.info(f"Selected {len(new_links)} out of {len(links)} new links")

    # Step 4: Crawl each article and save as Markdown into the database
    new_articles = []
    async with AsyncWebCrawler() as crawler:
        for url in new_links:
            # Configure crawler to fetch only the target page (no deep crawl)
            config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=0,
                    include_external=False,
                    filter_chain=FilterChain([]),
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                cache_mode=CacheMode.BYPASS,
                verbose=True,
            )

            results = await crawler.arun(url=url, config=config)
            if not results:
                logger.warning(f"No crawl result for {url}")
                continue

            # Expect one result per URL
            result = results[0]

            url = url.split("?", 1)[0]
            date_title = url.split("/")[-1]
            # Match the pattern: date (YYYYMMDD) + underscore + title + optional extension
            match = re.match(r"(\d{4})(\d{2})(\d{2})_([^\.]+)", date_title)
            if not match:
                raise ValueError("URL format is unexpected: {}".format(url))
            year, month, day, title_part = match.groups()

            # Format the date as YYYY-MM-DD
            date_iso = f"{year}-{month}-{day}"

            # Replace hyphens with underscores in the title for readability
            title = title_part.replace("-", "_")

            # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
            published_on = format_date_to_datetime(date_iso)

            # store full article in the database
            database.add_publication(
                table_name="bnetza",
                published_on=published_on,
                title=title,
                post_url=url,
                post=result.markdown.raw_markdown,
            )

            await asyncio.sleep(5) # to avoid IP blocking

    logger.info(f"Finished saving {len(new_articles)} new articles out of {len(links)} links")