"""
Scraping SMARD news + extracting Highcharts “table view” data reliably.

Problem fixed:
- Crawl4AI's generic table extraction can miss the first column when Highcharts
  renders row labels as <th scope="row"> in <tbody>. That produces rows like:
    | 8,21 |   |
  We fix this by parsing the Highcharts HTML table ourselves and explicitly
  including tbody <th> cells.
- We also add a stable table name (caption) from Highcharts data-options.title.text
  (or <caption> if present).

Approach:
1) Deep-crawl to discover SMARD topic-article URLs.
2) For each new article:
   - Execute JS to click all "Tabelle anzeigen" toggle buttons.
   - Wait up to ~2.5s (or until tables appear) and snapshot HTML.
   - Extract tables using a Highcharts-aware parser (preferred).
   - Keep Crawl4AI result.tables as secondary, but drop "broken" ones.
3) Append extracted tables (as Markdown) to the stored article Markdown.

Notes:

- Requires lxml installed (already used in your project via LXML strategy).
- Keeps your database logic and date parsing.
"""

import asyncio
import fnmatch
import html as html_lib
import json
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from lxml import html as lxml_html

from open_event_intel.logger import get_logger
from open_event_intel.publications_database import PostsDatabase
from src.open_event_intel.scraping.scrapers.utils_scrape import format_date_to_datetime

logger = get_logger(__name__)


# -----------------------------
# Helpers: dates / cleaning
# -----------------------------
def _extract_date_iso_from_markdown(md: str, url: str) -> str:
    """Try to extract date from markdown. Falls back to dummy old date."""
    full_date_match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", md)
    if full_date_match:
        day, month, year = full_date_match.groups()
        return f"{year}-{month}-{day}"

    date_match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", md)
    if date_match:
        try:
            parsed_date = datetime.strptime(
                f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
                "%d %B %Y",
            )
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Date found but invalid format in markdown for URL: {url}")

    logger.warning(f"Date not found in markdown for URL: {url}. Using dummy date.")
    return "1990-01-01"


def _strip_highcharts_noise(markdown: str) -> str:
    """Remove common Highcharts/controls chatter while keeping actual article text. Conservative line-based removal."""
    blacklist_substrings = [
        "Created with Highcharts",
        "Highcharts",
        "Show table",
        "Show chart",
        "Tabelle anzeigen",
        "Diagramm anzeigen",
        "Export diagram",
        "Export table",
        "Grafik exportieren",
        "Tabelle exportieren",
        # typical export format labels that often appear as junk lines in markdown
        "PDF",
        "SVG",
        "PNG",
        "JPEG",
        "CSV",
        "XLS",
        "Mehr",
    ]

    cleaned_lines: List[str] = []
    for line in markdown.splitlines():
        s = line.strip()
        if not s:
            cleaned_lines.append(line)
            continue
        if any(b in s for b in blacklist_substrings):
            continue
        cleaned_lines.append(line)

    out = "\n".join(cleaned_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# -----------------------------
# Highcharts-aware extraction
# -----------------------------
def _norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def _parse_highcharts_options(raw_data_options: str) -> Optional[dict]:
    """data-options is usually HTML-escaped JSON."""
    if not raw_data_options:
        return None
    try:
        return json.loads(html_lib.unescape(raw_data_options))
    except Exception:
        return None


def _get_xaxis_title(options: Optional[dict]) -> str:
    if not options:
        return ""
    xaxis = options.get("xAxis")
    title = ""
    if isinstance(xaxis, list) and xaxis:
        title = ((xaxis[0].get("title") or {}).get("text")) or ""
    elif isinstance(xaxis, dict):
        title = ((xaxis.get("title") or {}).get("text")) or ""
    return _norm_text(title)


def _get_chart_title(options: Optional[dict]) -> str:
    if not options:
        return ""
    title = ((options.get("title") or {}).get("text")) or ""
    return _norm_text(title)


def _highcharts_options_to_table(options: dict) -> Optional[Dict[str, Any]]:
    """
    Build a table from Highcharts options when an HTML <table> is not present.
    Handles:
      - xAxis categories + multiple series
      - pie-like series with point name/y
      - [name, value] pairs
    """
    title = _get_chart_title(options)
    x_title = _get_xaxis_title(options) or "Category"
    series = options.get("series") or []
    if not series:
        return None

    # categories case
    xaxis = options.get("xAxis")
    categories: List[str] = []
    if isinstance(xaxis, list) and xaxis:
        categories = xaxis[0].get("categories") or []
    elif isinstance(xaxis, dict):
        categories = xaxis.get("categories") or []

    if categories:
        headers = [x_title] + [
            _norm_text(s.get("name") or "") or f"series_{i+1}"
            for i, s in enumerate(series)
        ]
        rows: List[List[str]] = []
        for idx, cat in enumerate(categories):
            row = [str(cat)]
            for s in series:
                data = s.get("data") or []
                cell = data[idx] if idx < len(data) else ""
                if isinstance(cell, dict) and "y" in cell:
                    row.append(str(cell.get("y", "")))
                else:
                    row.append(str(cell))
            rows.append(row)

        return {
            "caption": title,
            "headers": headers,
            "rows": rows,
            "metadata": {"source": "highcharts_data_options_categories"},
        }

    # named points / pairs (typically single series)
    if len(series) == 1:
        s0 = series[0]
        s_name = _norm_text(s0.get("name") or "") or "Value"
        data = s0.get("data") or []

        # dict points with name/y
        if data and isinstance(data[0], dict) and ("name" in data[0] or "y" in data[0]):
            headers = [x_title, s_name]
            rows = []
            for p in data:
                if not isinstance(p, dict):
                    continue
                rows.append([str(p.get("name", "")), str(p.get("y", p.get("value", "")))])
            return {
                "caption": title,
                "headers": headers,
                "rows": rows,
                "metadata": {"source": "highcharts_data_options_named_points"},
            }

        # pair points like ["Wind", 8.21]
        if data and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
            headers = [x_title, s_name]
            rows = [
                [str(p[0]), str(p[1])]
                for p in data
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            return {
                "caption": title,
                "headers": headers,
                "rows": rows,
                "metadata": {"source": "highcharts_data_options_pairs"},
            }

    return None


def extract_smard_highcharts_tables(raw_html: str) -> List[Dict[str, Any]]:
    """Extract Highcharts tables correctly, including row labels stored in <th scope="row">,and attach a human-readable caption from Highcharts title (data-options) if needed."""
    if not raw_html:
        return []

    try:
        doc = lxml_html.fromstring(raw_html)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []

    # SMARD charts: div with class js-chart and data-options JSON
    chart_nodes = doc.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' js-chart ') and @data-options]"
    )

    for chart in chart_nodes:
        options = _parse_highcharts_options(chart.get("data-options", ""))
        title = _get_chart_title(options)
        x_title = _get_xaxis_title(options)

        # Prefer actual Highcharts-generated HTML table if present.
        # Sometimes it is inside the chart node; sometimes inside nearest container.
        table_nodes = chart.xpath(
            ".//div[contains(concat(' ', normalize-space(@class), ' '), ' highcharts-data-table ')]//table"
        )

        if not table_nodes:
            # Try nearest standalone chart container (more robust)
            container = chart.xpath(
                "ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' c-standalone-chart-container ')][1]"
            )
            if container:
                table_nodes = container[0].xpath(
                    ".//div[contains(concat(' ', normalize-space(@class), ' '), ' highcharts-data-table ')]//table"
                )

        if table_nodes:
            for tnode in table_nodes:
                cap_nodes = tnode.xpath(".//caption")
                caption = _norm_text(cap_nodes[0].text_content()) if cap_nodes else title

                # headers
                ths = tnode.xpath(".//thead//th")
                headers = [_norm_text(th.text_content()) for th in ths] if ths else []

                # Fix empty first header (Highcharts often leaves it blank)
                if headers:
                    if not headers[0] or headers[0].lower() in {"", "category", "kategorie"}:
                        headers[0] = x_title or "Category"

                # rows: include tbody th + tds  (CRITICAL FIX)
                rows: List[List[str]] = []
                for tr in tnode.xpath(".//tbody/tr"):
                    cells = tr.xpath("./th|./td")
                    row = [_norm_text(c.text_content()) for c in cells]
                    if row:
                        rows.append(row)

                # If headers missing, create them from row widths
                if not headers and rows:
                    max_cols = max(len(r) for r in rows)
                    headers = [f"col_{i+1}" for i in range(max_cols)]

                # normalize row lengths
                ncols = len(headers)
                if ncols and rows:
                    rows = [r[:ncols] + [""] * max(0, ncols - len(r)) for r in rows]

                if headers and rows:
                    out.append(
                        {
                            "caption": caption,
                            "headers": headers,
                            "rows": rows,
                            "metadata": {"source": "highcharts_dom_table"},
                        }
                    )
            continue

        # No HTML table found -> build from options
        if options:
            built = _highcharts_options_to_table(options)
            if built and built.get("rows"):
                out.append(built)

    return out


# -----------------------------
# Crawl4AI table helpers
# -----------------------------
def _looks_like_broken_highcharts_table(t: Dict[str, Any]) -> bool:
    """
    Detect the specific broken shape you reported.

    - headers show 2+ cols
    - most rows have numeric-ish first cell
    - second cell is empty for most rows
    """
    headers = t.get("headers") or []
    rows = t.get("rows") or []
    if len(headers) < 2 or not rows:
        return False

    empty_second = 0
    numeric_first = 0
    checked = 0

    for r in rows[:50]:
        if not r:
            continue
        checked += 1
        first = (r[0] if len(r) > 0 else "").strip()
        second = (r[1] if len(r) > 1 else "").strip()
        if second == "":
            empty_second += 1
        if re.fullmatch(r"[0-9\.,]+", first or ""):
            numeric_first += 1

    if checked == 0:
        return False

    return (empty_second / checked) > 0.85 and (numeric_first / checked) > 0.70


def _merge_tables(*table_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge tables while avoiding obvious duplicates. Heuristic key: caption + row_count + col_count."""
    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, int]] = set()

    for tables in table_lists:
        for t in tables or []:
            cap = _norm_text(t.get("caption") or "")
            rows = t.get("rows") or []
            headers = t.get("headers") or []
            cols = len(headers) if headers else max((len(r) for r in rows), default=0)
            key = (cap, len(rows), cols)
            if key in seen:
                continue
            seen.add(key)
            merged.append(t)

    return merged


def tables_to_markdown(tables: List[Dict[str, Any]]) -> str:
    """Convert table dicts to Markdown. Uses caption as the heading."""
    chunks: List[str] = []
    for i, t in enumerate(tables, start=1):
        caption = _norm_text(t.get("caption") or "")
        headers = t.get("headers") or []
        rows = t.get("rows") or []

        if not headers or not rows:
            continue

        heading = f"### {caption}" if caption else f"### Table {i}"
        chunks.append(heading)
        chunks.append("| " + " | ".join(headers) + " |")
        chunks.append("|" + "|".join("---" for _ in headers) + "|")

        ncols = len(headers)
        for r in rows:
            rr = r[:ncols] + [""] * max(0, ncols - len(r))
            chunks.append("| " + " | ".join(rr) + " |")

        chunks.append("")

    return "\n".join(chunks).strip()


# -----------------------------
# Main scrape
# -----------------------------
async def main_scrape_smard_posts(root_url: str, database: PostsDatabase, table_name: str, params: dict) -> None:
    known_bad_links = [
        "https://www.smard.de/page/home/topic-article/211972/214452/energietraegerscharfe-exporte-nach-laendern"
    ]

    # Global browser config (JS is required for SMARD charts/tables)
    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        java_script_enabled=True,
        viewport_width=1400,
        viewport_height=900,
        # enable_stealth=True,  # enable if you see bot-detection issues
        # verbose=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        # -----------------------------
        # Phase 1: discover URLs (deep crawl)
        # -----------------------------
        url_filter_smard = URLPatternFilter(patterns=["*smard*"])
        filter_chain = FilterChain([url_filter_smard])

        discovery_config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                filter_chain=filter_chain,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            verbose=True,
            wait_until="domcontentloaded",
            page_timeout=60000,
            excluded_tags=["script", "style", "noscript"],
        )

        results = await crawler.arun(url=root_url, config=discovery_config)
        if isinstance(results, list) and len(results) == 1:
            logger.warning(f"Only one result found for {root_url}. Suspected limit.")

        if not isinstance(results, list):
            # Some configurations can return a single result; normalize to list
            results = [results]

        logger.info(f"Crawled {len(results)} pages under root.")

        # Select only topic-article pages
        article_urls: List[str] = []
        for r in results:
            try:
                if "topic-article" in r.url:
                    article_urls.append(r.url)
            except Exception:
                logger.warning(f"Expected 'topic-article' is not found in {r.url}. Skipping.")
                continue

        article_urls = sorted(set(article_urls))
        logger.info(f"Selected {len(article_urls)} unique topic-article URLs.")

        # -----------------------------
        # Phase 2: fetch each article with JS interactions + table extraction
        # -----------------------------
        # JS that:
        # - clicks all table toggle buttons
        # - sets a timer fallback so wait_for can complete even if no table appears
        js_click_tables_and_timer = r"""
        (() => {
            try { window.__c4a_table_wait_done = false; } catch(e) {}
            try {
                setTimeout(() => { window.__c4a_table_wait_done = true; }, 2500);
            } catch(e) {}

            const buttons = Array.from(
                document.querySelectorAll('button.js-toggle-table')
            ).filter(btn => (btn.textContent || '').toLowerCase().includes('tabelle'));

            buttons.forEach(btn => { try { btn.click(); } catch(e) {} });
        })();
        """

        wait_for_tables_or_timer = """js:() => {
            const hasTables = document.querySelectorAll('.highcharts-data-table table').length > 0;
            const done = (window.__c4a_table_wait_done === true);
            return hasTables || done;
        }"""

        # Important bits:
        # - table_score_threshold lower helps Crawl4AI include more tables
        # - keep_data_attributes allows data-options to survive into cleaned_html (useful if you parse that)
        # - delay_before_return_html gives Highcharts time to inject the HTML table
        article_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=True,
            wait_until="networkidle",
            page_timeout=90000,
            js_code=[js_click_tables_and_timer],
            wait_for=wait_for_tables_or_timer,
            wait_for_timeout=90000,
            delay_before_return_html=0.8,
            scan_full_page=True,
            max_scroll_steps=6,
            scroll_delay=0.25,
            keep_data_attributes=True,
            table_score_threshold=5,
            excluded_tags=["script", "style", "noscript"],
            # reduce chart UI noise in markdown; we still parse raw HTML for tables
            excluded_selector=",".join(
                [
                    ".c-standalone-chart-container__buttons",
                    ".js-article-menu",
                    ".c-article-menu__list",
                    ".c-chart-export",
                    ".highcharts-container",
                    ".highcharts-a11y-proxy-container",
                    ".highcharts-credits",
                ]
            ),
        )

        new_articles: List[str] = []

        for url in article_urls:
            if url in known_bad_links:
                continue

            publication_id = database.create_publication_id(post_url=url)
            post_exists = database.is_table(table_name=table_name) and database.is_publication(
                table_name=table_name,
                publication_id=publication_id,
            )
            if post_exists and not params["overwrite"]:
                logger.info(f"Post already exists in database: {url} (skipping)")
                continue

            if not fnmatch.fnmatch(url, "*topic-article*"):
                continue

            logger.info(f"Fetching article (JS enabled): {url}")
            article_result = await crawler.arun(url=url, config=article_config)

            # Crawl4AI returns a single CrawlResult here
            if not getattr(article_result, "success", True):
                logger.warning(
                    f"Failed to crawl article: {url} :: {getattr(article_result, 'error_message', '')}"
                )
                continue

            raw_md = getattr(getattr(article_result, "markdown", None), "raw_markdown", "") or ""
            cleaned_md = _strip_highcharts_noise(raw_md)

            date_iso = _extract_date_iso_from_markdown(cleaned_md, url)
            published_on = format_date_to_datetime(date_iso)
            title = url.split("/")[-1].replace("-", "_")

            # Prefer parsing from the *post-interaction* raw HTML snapshot.
            # CrawlResult commonly exposes `html` (raw) and `cleaned_html`.
            raw_html = getattr(article_result, "html", "") or ""
            if not raw_html:
                # fall back to cleaned_html if raw isn't available
                raw_html = getattr(article_result, "cleaned_html", "") or ""

            highcharts_tables = extract_smard_highcharts_tables(raw_html)

            # Keep Crawl4AI tables, but drop the broken Highcharts-shaped ones
            c4a_tables = getattr(article_result, "tables", []) or []
            c4a_tables = [t for t in c4a_tables if not _looks_like_broken_highcharts_table(t)]

            all_tables = _merge_tables(highcharts_tables, c4a_tables)

            tables_md = ""
            if all_tables:
                tables_md = "\n\n## Extracted data tables\n\n" + tables_to_markdown(all_tables)

            final_post = (cleaned_md + (tables_md or "")).strip()

            database.add_publication(
                table_name=table_name,
                published_on=published_on,
                title=title,
                post_url=url,
                language=params["language"],
                post=final_post,
                overwrite=params["overwrite"],
            )

            new_articles.append(url)
            await asyncio.sleep(1.0 + random.random() * 1.5)

        logger.info(f"Finished saving {len(new_articles)} new articles out of {len(article_urls)} discovered.")
