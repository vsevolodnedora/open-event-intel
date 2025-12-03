import copy
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from langid import langid

from src.database import PostsDatabase, Publication
from src.logger import get_logger

logger = get_logger(__name__)


def process_one_article_text(  # noqa: C901
        id:str,
        publisher:str,
        text:str,
        date:datetime,
        title:str,
        url:str,
        start_markers: List[str],
        end_markers: List[str],
        start_marker_constructs:Dict|None,
        skip_start_lines:int|None, max_lines:int|None,
        custom_black_list_starters:List,
        black_list_single_word_lines:List,
        black_list_blocks:List|None,
        remove_image_links:bool,
        strip_links:bool,
        remove_empty_links:bool,
        strip_generic_page_links:bool,
) -> str:
    """Process one article text and return a snippet."""
    # Find start point from which to cut the article
    start_idx = None
    if start_marker_constructs is not None:
        for start_marker_name, start_marker_func in start_marker_constructs.items():
            if start_marker_name == "date":
                # start marker is a date. Function expects date as a string in YYYY-MM-DD format
                start_markers.append(start_marker_func(date))
            else:
                raise NotImplementedError(f"Start marker {start_marker_name} not implemented.")

    for start_marker in start_markers:
        start_idx_ = text.find(start_marker)
        if start_idx_ == -1 or not start_idx_:
            continue
        else:
            start_idx = start_idx_ + len(start_marker)
            break

    if len(start_markers)>0 and len(end_markers) > 0:
        if not start_idx or start_idx == -1 or start_idx == len(text)-1:
            raise ValueError(f"Start marker {start_markers} not found in Publication from {publisher}: {date} {title} | ID='{id}' | {url}")


    # find end point up to which to cut the article
    end_idx = None
    for end_marker in end_markers:
        end_idx_ = text.find(end_marker)
        if end_idx_ == -1 or not end_idx_:
            continue
        else:
            end_idx = end_idx_
            break
    if len(start_markers)>0 and len(end_markers) > 0:
        if not end_idx or end_idx == -1 or end_idx == len(text)-1:
            raise ValueError(f"End marker not found in {title}, skipping. Publication from {publisher}: {date} {title} | ID='{id}' | {url}")

        # sanity check
        if start_idx > end_idx:
            raise ValueError(f"Start marker {start_idx} is before end marker {end_idx}.")

        # Extract and clean up the snippet
        snippet = text[start_idx:end_idx].strip()
        if len(snippet) == 0:
            raise ValueError("Snippet is empty, skipping.")
    else:
        snippet = copy.deepcopy(text)
        logger.debug(f"No start or end marker provided for {title}, Processing the entire text.")

    # check fot file that might be too big
    num_lines = snippet.count("\n") + 1  # Count lines in the markdown
    if max_lines is not None and num_lines > max_lines:
        logger.warn(f"Post '{title}' has {num_lines} lines, which exceeds the max_lines limit of {max_lines}.")

    # remove lines from the file that start with an unwanted element
    if custom_black_list_starters is not None:
        current_lines = snippet.split("\n")
        selected_lines = []
        for line in current_lines:
            if any([line.startswith(element) for element in custom_black_list_starters]):
                continue
            selected_lines.append(line)
        snippet = "\n".join(selected_lines)

    # remove lines from the file that are an unwanted element (1to1)
    if black_list_single_word_lines is not None:
        selected_lines = []
        for line in snippet.split("\n"):
            found = False
            if any(line == element for element in black_list_single_word_lines):
                found = True
            if found:
                continue
            selected_lines.append(line)
        snippet = "\n".join(selected_lines)

    if skip_start_lines > 0:
        lines = snippet.split("\n")
        selected_lines = lines[skip_start_lines:]
        snippet = "\n".join(selected_lines)

    # remove the whole block of text if needed
    if black_list_blocks is not None:
        blocks = re.split(r"(\n\s*\n)", snippet)  # split while preserving newlines
        cleaned_blocks = []

        for block in blocks:
            skip_block = False
            for black_list_block_component in black_list_blocks:
                if black_list_block_component in block:
                    logger.debug(f"removing {black_list_block_component} from block")
                    skip_block = True
                    continue
            if skip_block:
                continue

            cleaned_blocks.append(block)

        snippet = "".join(cleaned_blocks)

    if len(snippet.split("\n")) <= 1:
        logger.warning(
            f"Only one line in snippet, nothing to write after skipping first line. Date:{date} title:{title}\n{snippet}"
        )

    # remove links
    if remove_image_links:
        pattern = r"!\[.*?\]\(([^)]+?\.(?:png|jpg|jpeg))(?:\?.*?)?\)"
        snippet = re.sub(pattern, "", snippet, flags=re.IGNORECASE)

    # strip page links
    if strip_links:
        pattern = r"\[([^\]]+)\]\(([^)]+?\.(?:html|aspx|pdf|doc)(?:\?.*?)?)\)"
        snippet = re.sub(pattern, r"\1 ", snippet, flags=re.IGNORECASE)

    if remove_empty_links:
        pattern = r"\[\]\((https?://[^)]+)\)"
        snippet = re.sub(pattern, "", snippet, flags=re.IGNORECASE)

    if strip_generic_page_links:
        pattern = r"\[([^\]]+)\]\(https://www\.[^)]+\)"
        snippet = re.sub(pattern, r"\1 ", snippet)

        pattern = r"\[([^\]]+)\]\((https?://[^)]+)\)"
        snippet = re.sub(pattern, r"\1 ", snippet)

    return snippet

def filter_german_posts(publications: List[Publication]) -> List[Publication]:
    """
    From a list of publications, group by calendar date (publication.published_on.date()). If a date has exactly two posts and one of them is German ("de"), keep the German one.

    Otherwise (no German counterpart), keep the English ("en") ones.

    Returns a list of selected Publication objects.
    """
    if not publications:
        return []

    # Classify languages once and cache results
    lang_by_id: Dict[str, Tuple[str, float]] = {}
    for p in publications:
        try:
            lang_by_id[p.id] = langid.classify(p.text)  # (lang, confidence)
        except Exception as e:
            logger.warning("Language detection failed for ID %s (%s). Assuming English.", p.id, e)
            lang_by_id[p.id] = ("en", 0.0)

    # Group by calendar date
    groups: Dict[str, List[Publication]] = defaultdict(list)
    for p in publications:
        date_key = p.published_on.date().isoformat()
        groups[date_key].append(p)

    selected: List[Publication] = []

    for date_key, items in groups.items():
        # Sort items deterministically (e.g., by time then id) for stable output
        items.sort(key=lambda x: (x.published_on, x.id))

        if len(items) == 2:
            # Check for a German counterpart
            german_items = [p for p in items if lang_by_id[p.id][0] == "de"]
            english_items = [p for p in items if lang_by_id[p.id][0] == "en"]

            if german_items:
                # Prefer (first) German item when a pair exists
                chosen = german_items[0]
                logger.info(
                    "Date %s: Found EN/DE pair; selecting German post ID %s (conf=%.3f).",
                    date_key, chosen.id, lang_by_id[chosen.id][1]
                )
                selected.append(chosen)
            else:
                # No German in the pair -> keep all English ones (usually both or one)
                ids = ", ".join(p.id for p in english_items) or "none"
                logger.info(
                    "Date %s: No German counterpart; keeping English post(s): %s.",
                    date_key, ids
                )
                selected.extend(english_items)
        else:
            # Not a pair -> keep English-only items
            english_items = [p for p in items if lang_by_id[p.id][0] == "en"]
            ids = ", ".join(p.id for p in english_items) or "none"
            logger.info(
                "Date %s: %d post(s) (not a pair). Keeping English post(s): %s.",
                date_key, len(items), ids
            )
            selected.extend(english_items)

    # Sort chronological ascending for reproducibility.
    selected.sort(key=lambda x: (x.published_on, x.id))
    return selected

def preprocess_posts_for_a_table(
    source_db: PostsDatabase,
    target_db: PostsDatabase,
    table_name: str,
    start_markers: List[str],
    end_markers: List[str],
    start_marker_constructs: Dict[str, Callable] | None = None,
    skip_start_lines: int = 0,
    max_lines: Optional[int] = None,
    custom_black_list_starters: Optional[List[str]] = None,
    black_list_single_word_lines: Optional[List[str]] = None,
    black_list_blocks: Optional[List[str]] = None,
    prefer_german: bool = False,
    title_blacklist: list = [],
    allow_failed:bool = False,
    overwrite: bool = False,
) -> None:
    """Process publications for one publisher and add them to the new database."""
    # 1) Get all article metadata (ID, date, title, url)
    publications:List[Publication] = source_db.list_publications(table_name=table_name, sort_date=True)
    logger.info(f"Found {len(publications)} publications in table '{table_name}'.")

    # check if the table exists in the target database
    target_db.check_create_table(table_name=table_name)

    if prefer_german:
        publications = filter_german_posts(publications=publications)

    # 2) Iterate and process
    for publication in publications:
        pub_id      = publication.id
        published_on= publication.published_on
        title       = publication.title
        url         = publication.url
        post        = publication.text

        if not post:
            logger.warning(f"No post for url={url}; skipping.")
            continue

        if len(post) < 5:
            logger.warning(f"Post for url={url}; was not scraped correctly. Got: '{post}' Skipping.")
            continue

        if title in title_blacklist:
            logger.warning(f"Post for url={url}; is blacklisted. Skipping.")
            continue

        if target_db.is_table(table_name=table_name) and target_db.is_publication(table_name=table_name, publication_id=pub_id) and not overwrite:
            logger.info(f"Post for url={url}; is already a the preprocessed database; skipping.")
            continue

        # 4) Clean/process the text
        try:
            cleaned = process_one_article_text(
                id=pub_id,
                publisher=table_name,
                text=post,
                date=published_on,
                title=title,
                url=url,
                start_markers=start_markers,
                start_marker_constructs=start_marker_constructs,
                end_markers=end_markers,
                custom_black_list_starters=custom_black_list_starters,
                black_list_single_word_lines=black_list_single_word_lines,
                skip_start_lines=skip_start_lines,
                black_list_blocks=black_list_blocks,
                max_lines=max_lines,
                remove_image_links=True,
                strip_links=True,
                remove_empty_links = True,
                strip_generic_page_links = True,
            )

            # 5) Store compressed cleaned text back into target DB
            target_db.add_publication(
                table_name=table_name,
                published_on=published_on,
                title=title,
                post_url=url,
                post=cleaned,
                overwrite=overwrite,
            )

        except Exception as e:
            logger.error(f"Failed to process post for Publication from {publication.publisher}: {published_on} {title} | ID='{pub_id}' | {url}")
            if allow_failed:
                continue
            else:
                raise e

    logger.info(f"Completed cleaning for table '{table_name}'.")


class Preprocessor:
    """Process raw posts and remove markdown bloat: links, HTML leftovers."""

    def __init__(self, config: dict) -> None:
        """Initialize the Preprocessor."""
        self.config = config

    @staticmethod
    def date_to_dd_mm_yyyy(input_datetime: datetime) -> str:
        """Convert datetime to DD.MM.YYYY format."""
        day = input_datetime.day
        month = input_datetime.month
        year = input_datetime.year
        # Format with zero-padding for day and month
        return f"{int(day)}.{int(month)}.{year}"

    @staticmethod
    def date_to_yyyy_mm_dd(input_datetime: datetime) -> str:
        """Convert datetime to DD-MM-YYYY format (without using string splitting)."""
        year = input_datetime.year
        month = input_datetime.month
        day = input_datetime.day
        # Format as MM/DD/YYYY to match original intent
        formatted_date = f"{month:02d}/{day:02d}/{year}"
        return formatted_date

    def __call__(self, source_db_path: str, target_db_path: str, table_name:str, out_dir:str, allow_failed:bool, overwrite:bool) -> None:
        """Process agora raw posts."""
        if not os.path.isfile(source_db_path):
            raise FileNotFoundError(f"source_db not found: {source_db_path}")

        source_db = PostsDatabase(source_db_path)

        if not os.path.isfile(target_db_path):
            logger.info(f"Target database is not found: {target_db_path}. Creating target database with table {table_name}")
            target_db = PostsDatabase(target_db_path)
            target_db.check_create_table(table_name)
        else:
            target_db = PostsDatabase(target_db_path)

        preprocess_posts_for_a_table(
            source_db=source_db,
            target_db=target_db,
            table_name=table_name,
            start_markers=self.config["start_markers"],
            end_markers=self.config["end_markers"],
            start_marker_constructs=self.config.get("start_marker_constructs", None),
            custom_black_list_starters=self.config.get("custom_black_list_starters", None),
            black_list_single_word_lines=self.config.get("black_list_single_word_lines", None),
            black_list_blocks=self.config.get("black_list_blocks", None),
            skip_start_lines=self.config.get("skip_start_lines", 0),
            max_lines=self.config.get("max_lines", None),
            prefer_german=self.config.get("prefer_german", False),
            title_blacklist=self.config.get("title_blacklist", []),
            allow_failed=allow_failed,
            overwrite=overwrite
        )
        # save scraped posts as raw .md files for analysis
        target_db.dump_publications_as_markdown(table_name=table_name, out_dir=out_dir)

        source_db.close()
        target_db.close()
        logger.info(f"Finished preprocessing raw posts for {table_name}.")
