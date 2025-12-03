import asyncio
import copy
import gc
import random
import re
from datetime import datetime
from typing import List
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
)
from markdownify import markdownify as md
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.database import PostsDatabase
from src.logger import get_logger
from src.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

async def fetch_news_links_with_playwright_async(
    url: str = "https://www.50hertz.com/de/Medien/",
    headless: bool = True,
    timeout_ms: int = 45000,
    wait_after_network_idle: float = 5.0,
) -> List[str]:
    """ Async: load the page with JS executed and extract absolute links containing 'News/Details/' (case-insensitive)."""
    logger.info(f"Fetching 50Hertz news links with playwright async for {url}")
    found = set()
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    )
    page = await context.new_page()
    try:
        # Try waiting for true network idle; may timeout on slow CI
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    except PlaywrightTimeoutError as e:
        logger.error("PlaywrightTimeoutError raised for '{}' with {}".format(url, e))
        # Fallback: proceed even if networkidle didn’t happen
        pass

    # Give any late-rendered JS a moment
    logger.debug(f"Waiting {int(wait_after_network_idle * 1000)} for JS to be loaded for {url}")
    await page.wait_for_timeout(int(wait_after_network_idle * 1000))

    # Best‐effort: wait for at least one relevant link
    try:
        logger.debug(f"Waiting {15000} for" + 'a[href*="News/Details"]' + f" to be loaded for {url}")
        await page.wait_for_selector('a[href*="News/Details"]', timeout=15000)
    except PlaywrightTimeoutError as e:
        logger.error("PlaywrightTimeoutError raised for '{}' with {}".format(url, e))
        pass

    # Scrape all anchors
    anchors = await page.query_selector_all("a[href]")
    for a in anchors:
        href = (await a.get_attribute("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "#", "mailto:")):
            continue
        if re.search(r"news/details", href, re.IGNORECASE):
            found.add(urljoin(url, href))

    # Fallback: regex scan of the fully rendered HTML
    logger.debug(f"Processing {len(found)} news articles")
    content = await page.content()
    for m in re.findall(
        r"https?://[^\s\"'>]+/News/Details/[^\s\"'>]+", content, re.IGNORECASE
    ):
        found.add(m)

    # Clean up
    await context.close()
    await browser.close()
    await playwright.stop()

    return sorted(found) # select unique links

def find_and_format_numeric_date(text:str)->str|None:
    """Extract date from markdown."""
    pattern = r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"
    match = re.search(pattern, text)

    if not match:
        return None

    day, month, year = match.groups()
    return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")

def is_challenge_page(markdown: str) -> bool:
    """Heuristic for detecting Cloudflare / human-verification interstitials."""
    if markdown is None:
        return True

    lowered = markdown.lower()
    return (
        "verifying you are human" in lowered
        or "ray id" in lowered and "cloudflare" in lowered
        or "please enable javascript" in lowered and "security check" in lowered
    )

async def scrape_page_with_crawl4ai(link: str, default_date:str) -> str|None:
    """Scrape page using Crawl4AI."""
    # Shared dispatcher and rate limiter (reuse for all crawls)
    rate_limiter = RateLimiter(
        base_delay=(20.0, 110.0),
        max_delay=400.0,
        max_retries=10,
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=85.0,
        check_interval=3.0,
        max_session_permit=1,
        monitor=CrawlerMonitor(),
        rate_limiter=rate_limiter,
    )

    # Base run config (can be mutated per attempt if needed)
    base_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=0,
            include_external=False,
            filter_chain=FilterChain([]),
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        page_timeout=600_000,
    )

    plausable_user_agents = [
        # A few realistic modern user agents; rotate if challenged.
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # retry logic
    attempt = 0
    max_attempts = 4
    last_markdown = None
    success = False

    # Create a crawler with initial BrowserConfig (will be reused unless we need to rotate UA)
    async with AsyncWebCrawler(
        config=BrowserConfig(
            user_agent=plausable_user_agents[0],
            headless=True,
            use_persistent_context=True,  # to keep cookies between requests if beneficial
        )
    ) as crawler:
        while (attempt < max_attempts) and not success:
            attempt += 1
            # Rotate user-agent on retry
            user_agent = random.choice(plausable_user_agents) if attempt > 1 else plausable_user_agents[0]
            if attempt > 1:
                # Reconfigure crawler's user-agent by rebuilding browser context.
                # Lightweight since it's only on failure.
                await crawler.close()  # close previous context
                crawler = AsyncWebCrawler(
                    config=BrowserConfig(
                        user_agent=user_agent,
                        headless=True,
                        use_persistent_context=True,
                    )
                )
                await crawler.__aenter__()  # re-enter context manually for retries

            # Small jitter to reduce fingerprinting
            await asyncio.sleep(2 + random.random() * 2)

            try:
                config = copy.deepcopy(base_config)
                results = await crawler.arun(url=link, config=config, dispatcher=dispatcher)
            except Exception as e:
                logger.warning(f"Exception while crawling {link} on attempt {attempt}: {e}")
                await asyncio.sleep(2**attempt)
                continue

            if results is None or not results:
                logger.warning(f"No crawl result for {link} on attempt {attempt}")
                await asyncio.sleep(2**attempt)
                continue

            # accessing the article page (should be only one in the list)
            result = results[0]
            raw_md = result.markdown
            last_markdown = raw_md

            if is_challenge_page(raw_md):
                logger.warning(f"Detected challenge page for {link} on attempt {attempt}; retrying with different UA/backoff")
                await asyncio.sleep(2**attempt)
                continue  # retry

            # Extract date
            date = find_and_format_numeric_date(raw_md)
            if date is None:
                logger.warning(f"Could not locate date in the page scraped with Crawl4AI after {attempt} attempts. Link: {link}")
                logger.debug(last_markdown)
                continue

            # stop the retry
            success = True

        # check if at the and the download was successfull
        if not success:
            logger.warning(f"Failed to scrape challenge {link} after {attempt} attempts.")
            logger.debug(last_markdown)
            await asyncio.sleep(5)

    if success:
        return raw_md
    else:
        return None

async def scrape_page_with_playwright(url: str, content_selector: str = ".n-grid.xsp-news-item", timeout: int = 30) -> str:
    """
    Scrape the JS-rendered content at `url`, extract the element matching `content_selector`,
    strip scripts/styles, and convert to Markdown.

    Args:
        url: The page URL to scrape.
        content_selector: CSS selector for the main article container.
        timeout: Seconds to wait for the element to appear.
        chrome_driver_path: Path to your chromedriver executable.

    Returns:
        A Markdown string of the article.
    """
    logger.info(f"Scraping with secondary (playwrite) scraper since main (crawl4a) has failed for {url}")
    # 1. Launch headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome( options=options)

    try:
        # 2. Load page and wait for the article container
        driver.get(url)
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, content_selector)))

        # 3. Grab the outer HTML of that container
        elem = driver.find_element(By.CSS_SELECTOR, content_selector)
        raw_html = elem.get_attribute("outerHTML")
    finally:
        driver.quit()

    # 4. Clean with BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # 5. Convert to Markdown
    markdown = md(str(soup), heading_style="ATX")
    return markdown

async def main_scrape_50hz_posts(root_url: str, table_name: str, database: PostsDatabase|None) -> None:
    """Scrape 50hz news pages."""
    links = await fetch_news_links_with_playwright_async(url=root_url) # get list of unique links in the page
    if len(links) > 20:
        logger.warning(f"Too many links to scrape for {table_name} ({len(links)} links)")
    for link in links:
        logger.debug(f"Found: {link}")
    if len(links) == 0:
        raise Exception("No links found")

    default_date = "1990-01-01"

    new_articles = []
    for link in links:
        logger.info(f"Processing {link}")

        # check for post in the database before trying to pull it as it is long
        if database is not None and database.is_table(table_name=table_name) and database.is_publication(table_name=table_name, publication_id=database.create_publication_id(post_url=link)):
            logger.info(f"Post already exists in the database. Skipping: {link}")
            continue

        # attempt scraping with crawl4ai
        raw_md = await scrape_page_with_crawl4ai(link=link, default_date=default_date)

        if raw_md is None:
            raw_md = await scrape_page_with_playwright(url=link)

        if raw_md is None:
            logger.error(f"All attempts to scrape the page failed. Url: {link}")
            logger.debug(raw_md)
            continue

        # locate date
        date = find_and_format_numeric_date(raw_md)
        if date is None:
            logger.error(f"After scraping the page, could not locate date in {link}")
            logger.debug(raw_md)
            continue

        # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
        published_on = format_date_to_datetime(date)

        # add post to the database
        article_title = link.rstrip("/").split("/")[-1].replace("-", "_")
        if database is not None:
            database.add_publication(
                table_name=table_name,
                published_on=published_on,
                title=article_title,
                post_url=link,
                post=raw_md,
            )

        new_articles.append(link)
        logger.info(f"Saved article {link} (size: {len(raw_md)} chars)")

        await asyncio.sleep(5)  # to avoid IP blocking
        gc.collect()  # clean memory

    logger.info(f"Saving {len(new_articles)} new posts out of {len(links)} total links")