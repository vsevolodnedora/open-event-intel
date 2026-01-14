import asyncio
import gc
import random
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import httpx  # async HTTP client
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
)

from open_event_intel.logger import get_logger
from open_event_intel.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)

# Playwright globals so we reuse browser across calls in the process
_playwright = None
_browser = None

async def _get_playwright_browser():
    global _playwright, _browser
    if _browser:
        return _browser
    from playwright.async_api import async_playwright

    _playwright = await async_playwright().start()
    # launch in headless mode; on GitHub Actions you want no-sandbox
    _browser = await _playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-setuid-sandbox"],
    )
    return _browser

async def fetch_html(
    url: str,
    timeout: int = 10,
    headers: Optional[dict] = None,
    max_retries: int = 3,
) -> str:
    """
    Fetch HTML from a URL with fallback to headless browser if blocked.
    Handles TenneT redirect loops by setting German headers and locale.
    """
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9",  # force German version
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    merged_headers = {**default_headers, **(headers or {})}

    # normalize to avoid redirect loop (trailing slash is required by TenneT)
    target = url if url.endswith("/") else url + "/"

    async with httpx.AsyncClient(timeout=timeout, headers=merged_headers, follow_redirects=True) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.get(target)
                status = resp.status_code
                if status == 403:
                    logger.warning("Received 403 for %s on HTTP fetch; will attempt browser fallback.", target)
                    break
                if status in (429, 500, 502, 503, 504):
                    backoff = 2 ** (attempt - 1)
                    logger.info("Transient status %s for %s, retrying after %s seconds (attempt %s).", status, target, backoff, attempt)
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning("HTTPStatusError 403 for %s; falling back to browser.", target)
                    break
                if attempt == max_retries:
                    logger.error("Max retries hit fetching %s via HTTP: %s", target, e)
                    raise
                await asyncio.sleep(2 ** (attempt - 1))
            except (httpx.TransportError, httpx.ReadTimeout) as e:
                if attempt == max_retries:
                    logger.error("Network error fetching %s via HTTP: %s", target, e)
                    raise
                backoff = 2 ** (attempt - 1)
                logger.info("Network error %s for %s, retrying after %s seconds (attempt %s).", e, target, backoff, attempt)
                await asyncio.sleep(backoff)

    # Fallback to Playwright
    try:
        browser = await _get_playwright_browser()
        context = await browser.new_context(
            user_agent=merged_headers["User-Agent"],
            locale="de-DE",  # force browser locale to German
            bypass_csp=True,
        )
        page = await context.new_page()
        await page.set_extra_http_headers({
            "Accept": merged_headers["Accept"],
            "Accept-Language": merged_headers["Accept-Language"],
        })
        await page.goto(target, wait_until="networkidle", timeout=timeout * 1000)
        content = await page.content()
        await context.close()
        return content
    except Exception as e:
        logger.error("Playwright fallback failed for %s: %s", target, e)
        raise

def extract_news_links_from_html(html: str, base_url: str) -> list[str]:
    """Parse HTML and return a sorted list of absolute links whose href contains '/de/news/'."""
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith("javascript:") or href.startswith("#") or href.startswith("mailto:"):
            continue
        if "/de/news/" in href:
            full = urljoin(base_url, href)
            found.add(full)
    return sorted(found)

async def get_tennet_news_links_async(url: str = "https://www.tennet.eu/de/news-de") -> list[str]:
    """Async extraction using the async fetch_html."""
    html = await fetch_html(url)
    return extract_news_links_from_html(html, url)

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
    if markdown is None: return True

    lowered = markdown.lower()
    return (
        "verifying you are human" in lowered
        or "ray id" in lowered and "cloudflare" in lowered
        or "please enable javascript" in lowered and "security check" in lowered
    )

async def main_scrape_tennet_posts(root_url: str, table_name: str, database: PostsDatabase|None, params: dict) -> None:
    """Scrape tennet news pages."""
    html = await fetch_html(url=root_url)
    links = extract_news_links_from_html(html=html, base_url=root_url)

    # links that are know to fail due to language/other issues
    known_exceptions = [
        "https://www.tennet.eu/de/news-de/",
        "https://www.tennet.eu/de/news/tennet-holding-kuendigt-neue-finanzierungsstruktur-mit-niederlaendischer-staatsgarantie-und-startet-den-prozess-zur-zustimmung-der",
    ]

    # select links to process
    selected_links = []
    for link in links:
        if link not in known_exceptions:
            logger.debug(f"Found: {link}")
            selected_links.append(link)

    # Shared dispatcher and rate limiter (reuse for all crawls)
    rate_limiter = RateLimiter(
        base_delay=(20.0, 90.0),
        max_delay=200.0,
        max_retries=5,
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        check_interval=2.0,
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
        page_timeout=200_000,
    )

    new_articles = []

    plausable_user_agents = [
        # A few realistic modern user agents; rotate if challenged.
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # Create a crawler with initial BrowserConfig (will be reused unless we need to rotate UA)
    async with AsyncWebCrawler(
        config=BrowserConfig(
            user_agent=plausable_user_agents[0],
            headless=True,
            use_persistent_context=True,  # to keep cookies between requests if beneficial
        )
    ) as crawler:

        # Process links sequentially with retry/backoff. Could be batched with arun_many for higher throughput.
        for link in selected_links:
            logger.info(f"Processing {link}")

            # check for post in the database before trying to pull it as it is long
            if (
                database is not None
                and database.is_table(table_name=table_name)
                and database.is_publication(
                    table_name=table_name,
                    publication_id=database.create_publication_id(post_url=link),
                )
                and not params["overwrite"]
            ):
                logger.info(f"Post already exists in the database for {table_name}. Skipping: {link} (overwrite={params['overwrite']})")
                continue

            attempt = 0
            max_attempts = 3
            last_markdown = None
            success = False

            while attempt < max_attempts and not success:
                # Rotate user-agent on retry
                ua = random.choice(plausable_user_agents) if attempt > 0 else plausable_user_agents[0]
                if attempt > 0:
                    # Reconfigure crawler's user-agent by rebuilding browser context.
                    # Lightweight since it's only on failure.
                    await crawler.close()  # close previous context
                    crawler = AsyncWebCrawler(
                        config=BrowserConfig(
                            user_agent=ua,
                            headless=True,
                            use_persistent_context=True,
                        )
                    )
                    await crawler.__aenter__()  # re-enter context manually for retries

                # Small jitter to reduce fingerprinting
                await asyncio.sleep(1 + random.random() * 2)

                try:
                    config = base_config  # could deep-copy and modify if needed
                    results = await crawler.arun(url=link, config=config, dispatcher=dispatcher)
                except Exception as e:
                    logger.warning(f"Exception while crawling {link} on attempt {attempt+1}: {e}")
                    attempt += 1
                    await asyncio.sleep(2 ** attempt)
                    continue

                if not results:
                    logger.warning(f"No crawl result for {link} on attempt {attempt+1}")
                    attempt += 1
                    await asyncio.sleep(2 ** attempt)
                    continue

                result = results[0]
                raw_md = result.markdown
                last_markdown = raw_md

                if is_challenge_page(raw_md):
                    logger.warning(f"Detected challenge page for {link} on attempt {attempt+1}; retrying with different UA/backoff.")
                    attempt += 1
                    await asyncio.sleep(2 ** attempt)
                    continue  # retry

                # Extract date
                date = find_and_format_numeric_date(raw_md)
                if date is None:
                    logger.warning(f"Could not locate date. Skipping: {link}")
                    logger.debug(f"Raw markdown for {link}:\n{raw_md[:1000]}")
                    break  # give up on this link

                article_title = link.rstrip("/").split("/")[-1].replace("-", "_")
                success = True

            # check if the overall scrape was successfull
            if not success:
                logger.error(f"Failed to scrape challenge {link} after {max_attempts} attempts. Last markdown snippet:\n{(last_markdown or '')[:500]}")
                await asyncio.sleep(10)
                continue

            # convert date "YYYY-MM-DD" to datetime as "YYYY-MM-DD:12:00:00" for uniformity
            published_on = format_date_to_datetime(date)

            # addd to the database
            if database is not None:
                database.add_publication(
                    table_name=table_name,
                    published_on=published_on,
                    title=article_title,
                    post_url=link,
                    language=params["language"],
                    post=raw_md,
                    overwrite=params["overwrite"],
                )

            new_articles.append(link)
            logger.info(f"Saved article {link} (size: {len(raw_md)} chars)")

            await asyncio.sleep(5) # to avoid IP blocking
            gc.collect() # clean memory

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(selected_links)} articles")