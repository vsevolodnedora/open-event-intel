import asyncio
import gc
import urllib.parse
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
)

from src.database import PostsDatabase
from src.logger import get_logger
from src.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

async def fetch_articles(page_url: str):
    """Fetch list of articles from page_url."""
    async with aiohttp.ClientSession() as session:
        async with session.get(page_url) as resp:
            resp.raise_for_status()
            html = await resp.text()

    soup = BeautifulSoup(html, "lxml")

    articles = []
    # Loop through each teaser article
    for art in soup.select("article.m-node--list--teaser"):
        # Title & URL
        a = art.select_one("h3.m-node--list--teaser__title > a")
        if not a:
            continue
        href  = a["href"]
        title = a.get_text(strip=True)
        url   = urllib.parse.urljoin(page_url, href)

        # Date
        date_tag = art.select_one("span.date-display-single")
        date_str = date_tag.get_text(strip=True) if date_tag else None

        articles.append({
            "url":   url,
            "title": title,
            "date":  date_str,
        })

    return articles

async def main_scrape_energy_wire_posts(root_url: str, database: PostsDatabase, table_name: str) -> None:
    """Scrape news posts from clean energy wire website one by one to avoid IP blocking."""
    articles = await fetch_articles(page_url=root_url)
    for article_ in articles:
        logger.debug(f"Found: {article_['date']} {article_['url']}")

    new_articles = []
    async with AsyncWebCrawler() as crawler:
        for article in articles:

            article_url = article["url"].split("?", 1)[0]
            # sometimes page returns invalid core URL
            article_url = "https://www.cleanenergywire.org/news/" + article_url.split("/")[-1]

            article_title = article_url.split("/")[-1]
            article_title = article_title.replace("-","_")

            dt = datetime.strptime(article["date"], "%d %b %Y - %H:%M")
            formatted_datetime = dt.strftime("%Y-%m-%d %H:%M")

            # check if file exists and if so, skip
            if database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=article_url)):
                logger.info(f"Post already exists in the database. Skipping: {article_url}")
                continue

            logger.info(f"Processing {article['date']} {article_url}")

            rate_limiter = RateLimiter(
                base_delay = (10.0, 40.0),
                max_delay = 100.0,
                max_retries = 3,
                # rate_limit_codes: List[int] = None,
            )

            dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=80.0,
                check_interval=1.0,
                max_session_permit=1,
                monitor=CrawlerMonitor(),
                rate_limiter=rate_limiter
            )

            config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=0,
                    include_external=False,
                    filter_chain=FilterChain([]),
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                cache_mode=CacheMode.BYPASS,
                verbose=True,
                page_timeout=100_000
            )

            article_url:str = article["url"]

            results = await crawler.arun(url=article_url, config=config, dispatcher=dispatcher)
            if not results:
                logger.warning(f"No crawl result for {article_url}")
                continue
            # Expect one result per URL

            result = results[0]
            raw_md = result.markdown.raw_markdown

            # convert date "YYYY-MM-DD:HH:MM:SS" to datetime
            published_on = format_date_to_datetime(formatted_datetime)

            database.add_publication(
                table_name=table_name,
                published_on=published_on,
                title=article_title,
                post_url=article_url,
                post=raw_md,
            )

            new_articles.append(article_url)
            logger.info(f"Saved article {article_url} (size: {len(raw_md)} chars)")

            await asyncio.sleep(10) # to avoid IP blocking
            gc.collect() # clean memory

    logger.info(f"Finished saving {len(new_articles)} new articles out of {len(articles)} links")