"""
Stage 02: Parse - Structural parsing, chunking, and table extraction.

Transforms document_version.clean_content into a deterministic structural
representation: blocks, chunks with evidence anchoring, and table extracts.

**Reads:** `document_version`.
**Writes:** `block`, `evidence_span` (one per chunk span), `chunk` (with `chunk.evidence_id`), `table_extract`,
`doc_stage_status(stage_02_parse)`.
**Responsibility**:
  * turn each document's immutable document_version.clean_content into a deterministic structural representation and retrieval units that downstream stages can reliably reference with exact character spans (provenance locking).
  * Structural parsing → `block` via a deterministic, source-position-preserving parser over clean_content
  * Chunking → `chunk` (+ evidence anchoring) via block-aware and span-preserving process
  * Table extraction → `table_extract`: parse the raw table region (still anchored to the original text) into: row/col counts, header row index, headers JSON (best-effort), parse_method (e.g., markdown_pipe, html_table if any survived cleaning)
"""
import argparse
import bisect
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from open_event_intel.etl_processing.config_interface import (
    Config,
    get_config_version,
    load_config,
)
from open_event_intel.etl_processing.database_interface import (
    BlockRow,
    ChunkRow,
    DBError,
    TableExtractRow,
    compute_sha256_id,
)
from open_event_intel.etl_processing.stage_02_parse.database_stage_02_parse import PREREQUISITE_STAGE, STAGE_NAME, Stage02DatabaseInterface
from open_event_intel.logger import get_logger

logger = get_logger(__name__)

# Diagnostic helpers (logging-only, no functional impact)
_TEMPORAL_HINT_RE = re.compile(
    r"""(?ix)
      \b(?:
        \d{4}                         # 4-digit year
        |Q[1-4]                       # quarter labels
        |(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*  # month names
        |(?:fy|h[12]|cy)\s*\d         # fiscal/half/calendar year
        |year|month|quarter|date|period|week|annual|semi[-\s]?annual
      )\b
    """
)

def _has_temporal_hint(headers: list[str] | None) -> tuple[bool, list[str]]:
    """Check headers for temporal patterns. Returns (has_hint, matching_headers)."""
    if not headers:
        return False, []
    matches = [h for h in headers if _TEMPORAL_HINT_RE.search(h)]
    return bool(matches), matches

@dataclass(frozen=True)
class ParsedBlock:
    """Intermediate representation of a parsed block."""

    block_type: str
    span_start: int
    span_end: int
    block_level: int | None = None
    parent_index: int | None = None
    boilerplate_flag: str | None = None
    boilerplate_reason: str | None = None
    parse_confidence: float | None = None
    language_hint: str | None = None


@dataclass(frozen=True)
class ParsedTable:
    """Intermediate representation of a parsed table."""

    block_index: int
    row_count: int | None
    col_count: int | None
    headers: list[str] | None
    header_row_index: int | None
    parse_quality: float | None
    parse_method: str
    raw_table_text: str


class DocumentParser:
    """
    Deterministic parser for document structure extraction.

    Parses clean_content into blocks (headings, paragraphs, lists, tables)
    using position-preserving rules.
    """

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    LIST_ITEM_PATTERN = re.compile(r"^(\s*)[-*+]\s+(.+)$", re.MULTILINE)
    NUMBERED_LIST_PATTERN = re.compile(r"^(\s*)\d+[.)]\s+(.+)$", re.MULTILINE)
    TABLE_ROW_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)
    TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)

    def __init__(self, clean_content: str, primary_language: str | None = None) -> None:
        self._content = clean_content
        self._language = primary_language
        self._content_length = len(clean_content)

    def parse(self) -> tuple[list[ParsedBlock], list[ParsedTable]]:
        """
        Parse content into blocks and tables.

        :returns: Tuple of (blocks, tables) in span order.
        """
        blocks: list[ParsedBlock] = []
        tables: list[ParsedTable] = []

        table_regions = self._find_table_regions()
        consumed_ranges: list[tuple[int, int]] = [(s, e) for s, e, *_ in table_regions]

        for start, end, rows, headers, hdr_idx in table_regions:
            blocks.append(
                ParsedBlock(
                    block_type="TABLE",
                    span_start=start,
                    span_end=end,
                    parse_confidence=0.9,
                    language_hint=self._language,
                )
            )
            tables.append(
                ParsedTable(
                    block_index=len(blocks) - 1,
                    row_count=len(rows),
                    col_count=max((len(r) for r in rows), default=0),
                    headers=headers,
                    header_row_index=hdr_idx,
                    parse_quality=0.8,
                    parse_method="markdown_pipe",
                    raw_table_text=self._content[start:end],
                )
            )

        headings = list(self.HEADING_PATTERN.finditer(self._content))
        for m in headings:
            if not self._in_consumed_range(m.start(), m.end(), consumed_ranges):
                level = len(m.group(1))
                blocks.append(
                    ParsedBlock(
                        block_type="HEADING",
                        span_start=m.start(),
                        span_end=m.end(),
                        block_level=level,
                        parse_confidence=1.0,
                        language_hint=self._language,
                    )
                )
                consumed_ranges.append((m.start(), m.end()))

        list_regions = self._find_list_regions(consumed_ranges)
        for start, end in list_regions:
            blocks.append(
                ParsedBlock(
                    block_type="LIST",
                    span_start=start,
                    span_end=end,
                    parse_confidence=0.85,
                    language_hint=self._language,
                )
            )
            consumed_ranges.append((start, end))

        paragraph_regions = self._find_paragraph_regions(consumed_ranges)
        for start, end in paragraph_regions:
            text = self._content[start:end].strip()
            if len(text) >= 10:
                blocks.append(
                    ParsedBlock(
                        block_type="PARAGRAPH",
                        span_start=start,
                        span_end=end,
                        parse_confidence=0.95,
                        language_hint=self._language,
                    )
                )
                consumed_ranges.append((start, end))

        # ----------------------------------------------------------------
        # FIX: Detect uncovered content gaps and add as PARAGRAPH blocks.
        #
        # Previous code relied solely on blank-line splitting for paragraph
        # detection, which fails when:
        #   (a) content between headings uses single newlines (no blank
        #       lines) → the whole text becomes one "paragraph" starting
        #       with '#' → skipped by the startswith('#') filter,
        #   (b) position tracking in _find_paragraph_regions drifts due
        #       to variable-width separators (regex \n\s*\n can match >2
        #       chars but code always adds +2).
        #
        # This pass scans for any content region ≥10 meaningful characters
        # that is not covered by any existing block and adds it as a
        # PARAGRAPH.  This guarantees no content is silently lost.
        # ----------------------------------------------------------------
        gap_blocks = self._find_uncovered_regions(consumed_ranges)
        if gap_blocks:
            logger.info(
                "[parse:gap_recovery] Found %d uncovered content regions "
                "(%d chars total) — adding as PARAGRAPH blocks",
                len(gap_blocks),
                sum(b.span_end - b.span_start for b in gap_blocks),
            )
            blocks.extend(gap_blocks)

        blocks.sort(key=lambda b: (b.span_start, b.span_end))
        return blocks, tables

    def _find_uncovered_regions(
        self, consumed_ranges: list[tuple[int, int]]
    ) -> list[ParsedBlock]:
        """
        Find content regions not covered by any existing block.

        Scans the entire content for gaps between/around consumed ranges,
        then splits those gaps on blank lines to produce individual
        PARAGRAPH blocks.

        :param consumed_ranges: List of (start, end) spans already assigned
            to blocks.
        :returns: List of ParsedBlock(PARAGRAPH) for uncovered regions.
        """
        if self._content_length == 0:
            return []

        # Build sorted, merged consumed intervals
        merged = self._merge_ranges(consumed_ranges)

        # Find gaps
        gaps: list[tuple[int, int]] = []
        prev_end = 0
        for cs, ce in merged:
            if cs > prev_end:
                gaps.append((prev_end, cs))
            prev_end = max(prev_end, ce)
        if prev_end < self._content_length:
            gaps.append((prev_end, self._content_length))

        # Split each gap on blank lines and emit PARAGRAPH blocks
        result: list[ParsedBlock] = []
        for gap_start, gap_end in gaps:
            gap_text = self._content[gap_start:gap_end]
            # Split on blank lines within the gap
            sub_paragraphs = re.split(r"\n\s*\n", gap_text)
            sub_pos = 0
            for sub in sub_paragraphs:
                if not sub.strip():
                    # Advance past the empty segment + separator
                    sub_pos += len(sub)
                    # Skip the separator characters
                    while sub_pos < len(gap_text) and gap_text[sub_pos] in " \t\n\r":
                        sub_pos += 1
                    continue

                # Find exact position within the gap
                sub_start_in_gap = gap_text.find(sub, sub_pos)
                if sub_start_in_gap == -1:
                    # Fallback: use current position
                    sub_start_in_gap = sub_pos

                abs_start = gap_start + sub_start_in_gap
                abs_end = abs_start + len(sub)

                text = sub.strip()
                # Accept if text has ≥10 meaningful characters (after
                # stripping any leading '#' lines that the heading regex
                # may have already consumed — we re-check consumed_ranges
                # below to avoid double-counting).
                if len(text) >= 10:
                    # Verify this specific sub-region isn't already consumed
                    # (could partially overlap a heading at the start of a gap)
                    if not self._in_consumed_range(abs_start, abs_end, consumed_ranges):
                        result.append(
                            ParsedBlock(
                                block_type="PARAGRAPH",
                                span_start=abs_start,
                                span_end=abs_end,
                                parse_confidence=0.7,  # lower confidence: gap-recovered
                                language_hint=self._language,
                            )
                        )

                sub_pos = sub_start_in_gap + len(sub)

        return result

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping/adjacent (start, end) intervals."""
        if not ranges:
            return []
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        for cs, ce in sorted_ranges[1:]:
            prev_s, prev_e = merged[-1]
            if cs <= prev_e:
                merged[-1] = (prev_s, max(prev_e, ce))
            else:
                merged.append((cs, ce))
        return merged

    def _find_table_regions(
        self,
    ) -> list[tuple[int, int, list[list[str]], list[str] | None, int | None]]:
        """Identify markdown table regions with parsed row data."""
        regions: list[tuple[int, int, list[list[str]], list[str] | None, int | None]] = []
        lines = self._content.split("\n")
        i = 0
        pos = 0

        while i < len(lines):
            line = lines[i]
            if self.TABLE_ROW_PATTERN.match(line):
                start = pos
                rows: list[list[str]] = []
                headers: list[str] | None = None
                header_row_index: int | None = None

                while i < len(lines) and (
                    self.TABLE_ROW_PATTERN.match(lines[i])
                    or self.TABLE_SEPARATOR_PATTERN.match(lines[i])
                ):
                    current_line = lines[i]
                    if self.TABLE_SEPARATOR_PATTERN.match(current_line):
                        if rows and header_row_index is None:
                            header_row_index = len(rows) - 1
                            headers = rows[-1] if rows else None
                    else:
                        cells = [
                            c.strip()
                            for c in current_line.strip("|").split("|")
                        ]
                        rows.append(cells)
                    pos += len(current_line) + 1
                    i += 1

                end = pos - 1 if pos > start else pos
                if end > start and rows:
                    regions.append((start, end, rows, headers, header_row_index))
            else:
                pos += len(line) + 1
                i += 1

        return regions

    def _find_list_regions(
        self, consumed: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Find contiguous list item regions."""
        regions: list[tuple[int, int]] = []
        lines = self._content.split("\n")
        i = 0
        pos = 0

        while i < len(lines):
            line = lines[i]
            is_list = self.LIST_ITEM_PATTERN.match(line) or self.NUMBERED_LIST_PATTERN.match(line)

            if is_list and not self._in_consumed_range(pos, pos + len(line), consumed):
                start = pos
                while i < len(lines):
                    current = lines[i]
                    is_current_list = (
                        self.LIST_ITEM_PATTERN.match(current)
                        or self.NUMBERED_LIST_PATTERN.match(current)
                    )
                    if not is_current_list:
                        break
                    pos += len(current) + 1
                    i += 1
                end = pos - 1 if pos > start else pos
                if end > start:
                    regions.append((start, end))
            else:
                pos += len(line) + 1
                i += 1

        return regions

    def _find_paragraph_regions(
        self, consumed: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        r"""
        Find paragraph regions (text separated by blank lines).

        FIX: Uses regex finditer for separator positions instead of
        assuming a fixed +2 offset.  This correctly handles separators
        like '\\n   \\n' that are longer than 2 characters.
        """
        regions: list[tuple[int, int]] = []

        # Use finditer to get actual separator positions
        separator_pattern = re.compile(r"\n\s*\n")
        separators = list(separator_pattern.finditer(self._content))

        # Build paragraph boundaries from separator positions
        boundaries: list[tuple[int, int]] = []
        prev_end = 0
        for sep in separators:
            if sep.start() > prev_end:
                boundaries.append((prev_end, sep.start()))
            prev_end = sep.end()
        # Add final segment after last separator
        if prev_end < self._content_length:
            boundaries.append((prev_end, self._content_length))

        # If no separators found, the whole content is one paragraph
        if not separators and self._content_length > 0:
            boundaries.append((0, self._content_length))

        for start, end in boundaries:
            para_text = self._content[start:end]
            text = para_text.strip()
            if not text:
                continue

            if not self._in_consumed_range(start, end, consumed):
                # NOTE: We no longer filter out text.startswith("#") here.
                # Heading lines are already in consumed_ranges, so the
                # _in_consumed_range check handles them.  Filtering by '#'
                # was the root cause of BUG: when content between headings
                # had no blank-line separators, the entire content was one
                # "paragraph" starting with '#' and got silently dropped.
                if len(text) >= 10:
                    regions.append((start, end))

        return regions

    def _in_consumed_range(
        self, start: int, end: int, consumed: list[tuple[int, int]]
    ) -> bool:
        """
        Check if a span overlaps with any consumed range.

        Uses binary search on merged intervals when the list is large
        (>20 entries), falling back to linear scan for small lists.
        """
        if not consumed:
            return False
        # For small lists, linear scan is faster than sorting overhead
        if len(consumed) <= 20:
            for cs, ce in consumed:
                if start < ce and end > cs:
                    return True
            return False
        # For larger lists, build sorted merged intervals and bisect
        merged = self._merge_ranges(consumed)
        ends = [ce for _, ce in merged]
        idx = bisect.bisect_right(ends, start)
        if idx < len(merged):
            cs, ce = merged[idx]
            if start < ce and end > cs:
                return True
        return False


class DocumentChunker:
    """
    Block-aware chunker that respects structure and creates evidence-anchored chunks.

    Uses config from ChunkingSettings to control:
    - target_tokens / max_tokens / min_tokens: chunk size boundaries
    - overlap_tokens: overlap between consecutive chunks
    - respect_block_boundaries: whether to honour block structure
    - table_handling: "dedicated_chunk_type" emits tables as their own chunks
    - sentence_boundary_chars: characters used to find sentence breaks
    """

    # Recognised values for table_handling config field.
    _TABLE_HANDLING_DEDICATED = "dedicated_chunk_type"

    # Safety margin to absorb divergence between _estimate_tokens() heuristic
    # and the real embedding-model tokenizer.  The heuristic can undercount by
    # up to ~10 % on dense/structured content (pipes, numbers, non-ASCII).
    # Applying this margin means we target 90 % of the configured max_tokens,
    # giving ~10 % headroom so the real tokenizer never exceeds the hard limit.
    _TOKEN_SAFETY_MARGIN: float = 0.85

    def __init__(self, settings) -> None:  # ChunkingSettings type
        self._target_tokens = settings.target_tokens
        self._overlap_tokens = settings.overlap_tokens
        self._min_tokens = settings.min_tokens
        self._max_tokens = int(settings.max_tokens * self._TOKEN_SAFETY_MARGIN)
        self._config_max_tokens = settings.max_tokens  # original value for logging
        self._respect_boundaries = settings.respect_block_boundaries
        self._table_handling = settings.table_handling
        self._sentence_chars = settings.sentence_boundary_chars

    def chunk(self, clean_content: str, blocks: list) -> list[tuple[int, int, str, list[int]]]:
        """
        Generate chunks from content using block structure.

        :param clean_content: The document's clean content.
        :param blocks: Parsed blocks in span order.
        :returns: List of (span_start, span_end, chunk_type, block_indices).
        """
        if not blocks or not self._respect_boundaries:
            chunks = self._chunk_without_blocks(clean_content)
            return self._validate_and_split_oversized(chunks, clean_content)

        chunks: list[tuple[int, int, str, list[int]]] = []
        current_start: int | None = None
        current_blocks: list[int] = []
        current_text = ""

        # -----------------------------------------------------------
        # Track whether we have any chunkable (non-HEADING) blocks so
        # we can fall back to block-less chunking if all blocks are
        # headings or otherwise unchunkable.
        # -----------------------------------------------------------
        has_chunkable_blocks = any(
            b.block_type not in ("HEADING",) for b in blocks
        )

        if not has_chunkable_blocks:
            logger.warning(
                "[chunker:no_chunkable_blocks] All %d blocks are HEADING-type. "
                "Falling back to block-less chunking (content_len=%d)",
                len(blocks),
                len(clean_content),
            )
            chunks = self._chunk_without_blocks(clean_content)
            return self._validate_and_split_oversized(chunks, clean_content)

        for i, block in enumerate(blocks):
            block_text = clean_content[block.span_start : block.span_end]
            block_tokens = self._estimate_tokens(block_text)

            # Handle headings - they break chunks but aren't included
            if block.block_type == "HEADING":
                if current_start is not None and current_text.strip():
                    chunk_i = self._finalize_chunk(
                        current_start,
                        blocks[current_blocks[-1]].span_end if current_blocks else current_start + len(current_text),
                        "semantic",
                        current_blocks[:],
                    )
                    chunks.append(chunk_i)
                    current_start = None
                    current_blocks = []
                    current_text = ""
                continue

            # Handle tables
            if block.block_type == "TABLE":
                # Finalize any pending chunk
                if current_start is not None and current_text.strip():
                    chunks.append(
                        self._finalize_chunk(
                            current_start,
                            blocks[current_blocks[-1]].span_end if current_blocks else current_start + len(current_text),
                            "semantic",
                            current_blocks[:],
                        )
                    )
                    current_start = None
                    current_blocks = []
                    current_text = ""

                if self._table_handling == self._TABLE_HANDLING_DEDICATED:
                    # Check if table exceeds max_tokens
                    if block_tokens > self._max_tokens:
                        logger.warning(f"Table block exceeds max_tokens: {block_tokens} tokens > {self._max_tokens} max. Splitting table (span: {block.span_start}-{block.span_end})")
                        table_chunks = self._split_oversized_chunk(clean_content, block.span_start, block.span_end, "table_summary", [i])
                        chunks.extend(table_chunks)
                    else:
                        chunks.append((block.span_start, block.span_end, "table_summary", [i]))
                continue

            # Check if single block exceeds max_tokens
            if block_tokens > self._max_tokens:
                logger.warning(
                    f"Single block exceeds max_tokens: {block_tokens} tokens > {self._max_tokens} max. Splitting block (type: {block.block_type}, span: {block.span_start}-{block.span_end})"
                )

                # Finalize any pending chunk first
                if current_start is not None and current_text.strip():
                    chunks.append(
                        self._finalize_chunk(
                            current_start,
                            blocks[current_blocks[-1]].span_end,
                            "semantic",
                            current_blocks[:],
                        )
                    )
                    current_start = None
                    current_blocks = []
                    current_text = ""

                # Split the oversized block
                block_chunks = self._split_oversized_chunk(clean_content, block.span_start, block.span_end, "semantic", [i])
                chunks.extend(block_chunks)
                continue

            # Normal block accumulation logic
            if current_start is None:
                current_start = block.span_start
                current_blocks = [i]
                current_text = block_text
            else:
                combined_tokens = self._estimate_tokens(current_text + " " + block_text)
                if combined_tokens > self._max_tokens:
                    final_chunk = self._finalize_chunk(
                        current_start,
                        blocks[current_blocks[-1]].span_end,
                        "semantic",
                        current_blocks[:],
                    )
                    chunks.append(final_chunk)
                    current_start = block.span_start
                    current_blocks = [i]
                    current_text = block_text
                else:
                    current_blocks.append(i)
                    current_text += " " + block_text

                    if combined_tokens >= self._target_tokens:
                        chunk_end = block.span_end
                        final_chunk = self._finalize_chunk(
                            current_start,
                            chunk_end,
                            "semantic",
                            current_blocks[:],
                        )
                        chunks.append(final_chunk)

                        overlap_chars = self._overlap_tokens * 4
                        if overlap_chars > 0 and chunk_end - block.span_start <= overlap_chars:
                            # Overlap starts at the current block
                            current_start = block.span_start
                            current_blocks = [i]
                            current_text = block_text
                        elif overlap_chars > 0:
                            overlap_start = max(current_start, chunk_end - overlap_chars)
                            current_start = overlap_start
                            # FIX: Collect all block indices whose spans overlap
                            # [overlap_start, chunk_end), not just block [i].
                            current_blocks = [
                                j for j, b in enumerate(blocks)
                                if b.span_start < chunk_end
                                and b.span_end > overlap_start
                                and b.block_type != "HEADING"
                            ]
                            if not current_blocks:
                                current_blocks = [i]
                            current_text = clean_content[overlap_start:chunk_end]
                        else:
                            current_start = None
                            current_blocks = []
                            current_text = ""

        # Finalize any remaining chunk
        if current_start is not None and current_text.strip():
            end_pos = blocks[current_blocks[-1]].span_end if current_blocks else current_start + len(current_text)
            final_chunk = self._finalize_chunk(current_start, end_pos, "semantic", current_blocks[:])
            chunks.append(final_chunk)

        # Enforce minimum token requirement
        chunks = self._enforce_min_tokens(chunks, clean_content)

        # Final validation pass to catch any oversized chunks
        chunks = self._validate_and_split_oversized(chunks, clean_content)

        # -----------------------------------------------------------
        # FIX: Fallback — if block-aware chunking produced no chunks
        # despite non-trivial content, fall back to block-less chunking.
        # This catches edge cases where all chunkable blocks were too
        # small (below min_tokens) and couldn't be merged, or where
        # block boundaries caused all content to be filtered out.
        # -----------------------------------------------------------
        if not chunks and clean_content.strip():
            content_tokens = self._estimate_tokens(clean_content)
            logger.warning(
                "[chunker:block_aware_fallback] Block-aware chunking produced "
                "0 chunks despite content_len=%d (~%d tokens). Falling back to "
                "block-less chunking.",
                len(clean_content),
                content_tokens,
            )
            chunks = self._chunk_without_blocks(clean_content)
            chunks = self._validate_and_split_oversized(chunks, clean_content)

        return chunks

    def _split_oversized_chunk(
        self,
        clean_content: str,
        start: int,
        end: int,
        chunk_type: str,
        block_indices: list[int],
    ) -> list[tuple[int, int, str, list[int]]]:
        """Split an oversized chunk into smaller chunks that respect max_tokens."""
        text = clean_content[start:end]
        text_len = len(text)

        # Calculate maximum character length (using token estimation)
        max_chars = self._max_tokens * 4
        overlap_chars = self._overlap_tokens * 4

        sub_chunks: list[tuple[int, int, str, list[int]]] = []
        pos = 0

        while pos < text_len:
            chunk_end = min(pos + max_chars, text_len)

            # Try to find a sentence boundary if not at the end
            if chunk_end < text_len:
                boundary = self._find_sentence_boundary(text, pos, chunk_end)
                if boundary > pos:
                    chunk_end = boundary

            # Create sub-chunk
            sub_start = start + pos
            sub_end = start + chunk_end
            sub_text = clean_content[sub_start:sub_end]
            sub_tokens = self._estimate_tokens(sub_text)

            # Hard enforcement: shrink until _estimate_tokens is within budget
            if sub_tokens > self._max_tokens:
                shrunk_end = self._shrink_to_fit(clean_content, sub_start, sub_end)
                if shrunk_end > sub_start:
                    chunk_end = shrunk_end - start
                    sub_end = shrunk_end
                    sub_text = clean_content[sub_start:sub_end]
                    sub_tokens = self._estimate_tokens(sub_text)
                    logger.debug(
                        "Shrunk oversized sub-chunk to %d tokens (span: %d-%d)",
                        sub_tokens, sub_start, sub_end,
                    )

            sub_chunks.append((sub_start, sub_end, chunk_type, block_indices[:]))

            # Move position forward with overlap
            if chunk_end >= text_len:
                break
            pos = max(pos + 1, chunk_end - overlap_chars)

        logger.info(f"Split oversized chunk (span: {start}-{end}, {self._estimate_tokens(text)} tokens) into {len(sub_chunks)} smaller chunks")

        return sub_chunks

    def _find_word_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the nearest word boundary (whitespace) before end position.

        :param text: Text to search in.
        :param start: Start position of search region.
        :param end: End position of search region.
        :returns: Position of word boundary, or end if none found.
        """
        search_region = text[start:end]

        # Look for whitespace characters in reverse
        for i in range(len(search_region) - 1, -1, -1):
            if search_region[i] in " \t\n\r":
                return start + i + 1

        # No word boundary found, return end
        return end

    def _shrink_to_fit(self, text: str, start: int, end: int) -> int:
        """
        Set Binary-search for the largest sub-span [start, result) whose estimate_tokens() <= max_tokens.

        Tries sentence then word boundaries at the found length so the
        cut is human-readable.  Falls back to a hard character cut as a
        last resort (guaranteed to terminate).

        :param text: Full document text.
        :param start: Span start (inclusive).
        :param end: Span end (exclusive) — known to be too large.
        :returns: Adjusted end position guaranteed <= max_tokens.
        """
        lo, hi = start + 1, end
        best = start + 1  # absolute minimum: one character

        while lo <= hi:
            mid = (lo + hi) // 2
            est = self._estimate_tokens(text[start:mid])
            if est <= self._max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Try to land on a sentence boundary for readability
        sent = self._find_sentence_boundary(text, start, best)
        if start < sent <= best and self._estimate_tokens(text[start:sent]) <= self._max_tokens:
            return sent

        # Try word boundary
        word = self._find_word_boundary(text, start, best)
        if start < word <= best and self._estimate_tokens(text[start:word]) <= self._max_tokens:
            return word

        return best

    def _validate_and_split_oversized(
        self,
        chunks: list[tuple[int, int, str, list[int]]],
        clean_content: str,
    ) -> list[tuple[int, int, str, list[int]]]:
        """Make Final pass to split any chunks that exceed max_tokens."""
        validated_chunks: list[tuple[int, int, str, list[int]]] = []
        split_count = 0

        for chunk in chunks:
            start, end, chunk_type, block_indices = chunk
            chunk_text = clean_content[start:end]
            token_est = self._estimate_tokens(chunk_text)

            if token_est > self._max_tokens:
                logger.warning(f"Validation found oversized chunk: {token_est} tokens > {self._max_tokens} max. Splitting now (span: {start}-{end}, type: {chunk_type})")
                split_chunks = self._split_oversized_chunk(clean_content, start, end, chunk_type, block_indices)
                validated_chunks.extend(split_chunks)
                split_count += 1
            else:
                validated_chunks.append(chunk)

        if split_count > 0:
            logger.info(f"Validation split {split_count} oversized chunks")

        return validated_chunks

    def _enforce_min_tokens(
        self,
        chunks: list[tuple[int, int, str, list[int]]],
        clean_content: str,
    ) -> list[tuple[int, int, str, list[int]]]:
        """
        Remove or merge chunks that fall below min_tokens.

        Table chunks are exempt (they have their own sizing semantics).
        """
        if not chunks or self._min_tokens <= 0:
            return chunks

        result: list[tuple[int, int, str, list[int]]] = []

        for chunk in chunks:
            start, end, ctype, block_indices = chunk
            chunk_text = clean_content[start:end]
            token_est = self._estimate_tokens(chunk_text)

            # Always keep table chunks regardless of size.
            if ctype == "table_summary":
                result.append(chunk)
                continue

            if token_est >= self._min_tokens:
                result.append(chunk)
            elif result and result[-1][2] != "table_summary":
                prev_start, _prev_end, prev_type, prev_blocks = result[-1]
                merged_text = clean_content[prev_start:end]
                merged_tokens = self._estimate_tokens(merged_text)

                if merged_tokens <= self._max_tokens:
                    # Safe to merge
                    merged_blocks = list(dict.fromkeys(prev_blocks + block_indices))
                    result[-1] = (prev_start, end, prev_type, merged_blocks)
                    logger.debug(f"Merged small chunk ({token_est} tokens) into previous chunk (resulting size: {merged_tokens} tokens)")
                else:
                    # Merging would exceed max_tokens, keep chunk separate
                    logger.warning(
                        f"Cannot merge small chunk ({token_est} tokens < {self._min_tokens} min) "
                        f"because merge would exceed max_tokens ({merged_tokens} > {self._max_tokens}). "
                        f"Keeping as separate chunk (span: {start}-{end})."
                    )
                    result.append(chunk)
            else:
                # No predecessor to merge with – keep the small chunk
                # rather than silently dropping content.
                logger.warning(f"Chunk below min_tokens and cannot be merged: {token_est} tokens < {self._min_tokens} min (span: {start}-{end}, length: {end - start} chars, blocks: {block_indices})")
                result.append(chunk)

        return result

    def _chunk_without_blocks(self, clean_content: str) -> list[tuple[int, int, str, list[int]]]:
        """
        Fallback chunking when no blocks are detected.

        Uses max_tokens (not target_tokens) as the hard limit.
        """
        chunks: list[tuple[int, int, str, list[int]]] = []
        content_len = len(clean_content)

        if content_len == 0:
            return chunks

        max_chars = self._max_tokens * 4
        overlap_chars = self._overlap_tokens * 4
        pos = 0

        while pos < content_len:
            end = min(pos + max_chars, content_len)

            # Try to find a sentence boundary if not at the end
            if end < content_len:
                break_point = self._find_sentence_boundary(clean_content, pos, end)
                if break_point > pos:
                    end = break_point

            # Verify the chunk doesn't exceed max_tokens
            chunk_text = clean_content[pos:end]
            token_est = self._estimate_tokens(chunk_text)

            # If chunk is too large, hard-shrink to fit
            if token_est > self._max_tokens:
                shrunk_end = self._shrink_to_fit(clean_content, pos, end)
                if shrunk_end > pos:
                    end = shrunk_end
                    chunk_text = clean_content[pos:end]
                    token_est = self._estimate_tokens(chunk_text)
                    logger.debug(
                        "Shrunk text-level chunk to %d tokens (span: %d-%d)",
                        token_est, pos, end,
                    )

            if end > pos:
                chunks.append((pos, end, "semantic", []))

            if end >= content_len:
                break

            pos = max(pos + 1, end - overlap_chars)

        return chunks

    def _finalize_chunk(self, start: int, end: int, chunk_type: str, block_indices: list[int]) -> tuple[int, int, str, list[int]]:
        """Finalize a chunk with validated bounds."""
        return (start, end, chunk_type, block_indices)

    def _estimate_tokens(self, text: str) -> int:
        """
        Conservative token estimate for typical LLM BPE tokenizers.

        Goals:
        - Better than naive len(text)//4 for English + European languages.
        - Handles tables/CSV/markdown reasonably.
        - Avoids underestimation by combining heuristics and adding a buffer.
        """
        _WORD_RE = re.compile(r"\S+", re.UNICODE)
        if not text:
            return 0

        # Normalize newlines
        t = text.replace("\r\n", "\n").replace("\r", "\n")

        n_chars = len(t)
        n_newlines = t.count("\n")
        n_tabs = t.count("\t")
        n_pipes = t.count("|")
        n_commas = t.count(",")
        n_semicolons = t.count(";")

        # Rough signals that usually increase tokenization density
        n_non_ascii = sum(1 for c in t if ord(c) > 127)
        n_punct = sum(1 for c in t if not c.isalnum() and not c.isspace())
        n_digits = sum(1 for c in t if c.isdigit())

        # 1) Character-based estimate with adaptive chars/token (lower => more conservative).
        non_ascii_ratio = n_non_ascii / max(1, n_chars)
        punct_ratio = n_punct / max(1, n_chars)

        # Default: English-ish prose
        chars_per_token = 3.6

        # More punctuation/symbols (tables, code-ish text, lots of separators) => denser tokenization
        if punct_ratio > 0.08:
            chars_per_token = 3.3

        # Many non-ascii letters (diacritics etc.) or very symbol-heavy text => even denser
        if non_ascii_ratio > 0.05 or punct_ratio > 0.12:
            chars_per_token = 3.0

        est_by_chars = math.ceil(n_chars / chars_per_token)

        # 2) Chunk-based estimate (splitting on whitespace), helpful for tables and mixed content.
        chunks = _WORD_RE.findall(t)
        est_by_chunks = 0
        for ch in chunks:
            ch_len = len(ch)
            has_non_ascii = any(ord(c) > 127 for c in ch)
            is_all_punct = all((not c.isalnum()) for c in ch)

            if is_all_punct:
                # Sequences like "----", "||", "..." often tokenize into multiple pieces
                est_by_chunks += math.ceil(ch_len / 2)
                continue

            if has_non_ascii:
                # European languages can tokenize slightly denser than English on average
                est_by_chunks += max(1, math.ceil(ch_len / 3))
            elif ch.isdigit():
                # Numbers can tokenize into smaller pieces depending on length/patterns
                est_by_chunks += max(1, math.ceil(ch_len / 2.5))
            else:
                # English-ish word/mixed token
                est_by_chunks += max(1, math.ceil(ch_len / 4))

                # If chunk has some punctuation (e.g. "foo-bar", "a/b", "x.y"), add a bit
                extra_punct = sum(1 for c in ch if not c.isalnum())
                est_by_chunks += math.ceil(extra_punct / 3)

        # 3) Structural overhead: newlines + table delimiters often add tokens.
        structural = (
            n_newlines  # conservative: 1 token per line break
            + math.ceil(n_tabs / 2)  # tabs tend to tokenize as their own/with neighbors
            + math.ceil(n_pipes / 6)  # markdown tables / ASCII grids
            + math.ceil((n_commas + n_semicolons) / 20)  # CSV-ish separators
        )

        # 4) Combine + safety buffer to avoid underestimation.
        base = max(est_by_chars, est_by_chunks + structural)

        # Extra small buffer that scales with "hard" text (digits/punct), but stays modest.
        hardness = (n_digits + n_punct) / max(1, n_chars)
        buffer = 1.08 if hardness < 0.15 else 1.12

        return max(1, math.ceil(base * buffer))

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        overlap_chars = self._overlap_tokens * 4
        if len(text) <= overlap_chars:
            return text
        return text[-overlap_chars:]

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the nearest sentence boundary before end position."""
        search_region = text[start:end]
        last_boundary = -1

        for char in self._sentence_chars:
            pos = search_region.rfind(char)
            if pos > last_boundary:
                last_boundary = pos

        if last_boundary > 0:
            return start + last_boundary + 1
        return end


class ParseQualityCalculator:
    """Calculate parse quality score for a document."""

    def calculate(
        self, blocks: list[ParsedBlock], tables: list[ParsedTable], content_length: int
    ) -> float:
        """
        Calculate overall parse quality score.

        :returns: Score between 0.0 and 1.0.
        """
        if content_length == 0:
            return 0.0

        coverage = sum(b.span_end - b.span_start for b in blocks) / content_length
        coverage = min(coverage, 1.0)

        avg_confidence = 0.0
        if blocks:
            confidences = [b.parse_confidence for b in blocks if b.parse_confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        structure_bonus = 0.0
        has_headings = any(b.block_type == "HEADING" for b in blocks)
        has_paragraphs = any(b.block_type == "PARAGRAPH" for b in blocks)
        if has_headings:
            structure_bonus += 0.1
        if has_paragraphs:
            structure_bonus += 0.05

        score = (coverage * 0.4) + (avg_confidence * 0.4) + structure_bonus
        return min(score, 1.0)


def _compute_block_coverage(blocks: list[ParsedBlock], content_length: int) -> tuple[float, int]:
    """
    Compute what fraction of content is covered by blocks.

    :returns: (coverage_ratio, uncovered_chars)
    """
    if content_length == 0:
        return 1.0, 0

    # Build merged intervals to avoid double-counting overlaps
    if not blocks:
        return 0.0, content_length

    intervals = sorted((b.span_start, b.span_end) for b in blocks)
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    covered = sum(e - s for s, e in merged)
    uncovered = content_length - covered
    ratio = covered / content_length
    return min(ratio, 1.0), max(uncovered, 0)


def process_document(  # noqa: C901
    db: Stage02DatabaseInterface,
    doc_version_id: str,
    config: Config,
    run_id: str,
    config_hash: str,
) -> tuple[bool, str | None]:
    """
    Process a single document through parsing, chunking, and table extraction.

    :param db: Database interface.
    :param doc_version_id: Document version to process.
    :param config: Pipeline configuration.
    :param run_id: Current pipeline run ID.
    :param config_hash: Configuration hash for status records.
    :returns: Tuple of (success, error_message).
    """
    doc_version = db.get_document_version(doc_version_id)
    if doc_version is None:
        return False, f"document_version not found: {doc_version_id}"

    clean_content = doc_version.clean_content
    primary_language = doc_version.primary_language

    # -----------------------------------------------------------
    # FIX: Early exit for empty or trivially small content.
    # Avoids running the full parse/chunk pipeline on content that
    # can never produce meaningful output, and prevents the hard
    # failure at the chunk validation gate below.
    # -----------------------------------------------------------
    content_stripped = clean_content.strip() if clean_content else ""
    if not content_stripped:
        logger.info(
            "[skip:empty] doc=%s content_len=%d — empty or whitespace-only content",
            doc_version_id[:12],
            len(clean_content) if clean_content else 0,
        )
        return True, None  # nothing to parse; success with 0 artifacts

    content_length = len(clean_content)
    min_tokens = config.global_settings.chunking.min_tokens

    # Estimate tokens for the entire content once up-front
    _est_tokens_upfront = DocumentChunker(config.global_settings.chunking)._estimate_tokens(content_stripped)
    if _est_tokens_upfront < min_tokens and content_length < 200:
        logger.info(
            "[skip:trivial] doc=%s content_len=%d est_tokens=%d < min_tokens=%d "
            "— content too small for meaningful chunks",
            doc_version_id[:12],
            content_length,
            _est_tokens_upfront,
            min_tokens,
        )
        return True, None  # too small to chunk; success with 0 artifacts

    logger.info(
        "[diag:input] doc=%s content_len=%d language=%s",
        doc_version_id[:12],
        len(clean_content) if clean_content else 0,
        primary_language,
    )

    parser = DocumentParser(clean_content, primary_language)
    parsed_blocks, parsed_tables = parser.parse()

    # -- diagnostic: block type distribution --
    _block_types = Counter(b.block_type for b in parsed_blocks)
    logger.info(
        "[diag:parse] doc=%s block_types=%s total_blocks=%d total_tables=%d",
        doc_version_id[:12],
        dict(_block_types),
        len(parsed_blocks),
        len(parsed_tables),
    )

    # -- diagnostic: block coverage validation --
    _coverage_ratio, _uncovered_chars = _compute_block_coverage(parsed_blocks, len(clean_content))
    if _coverage_ratio < 0.5 and len(clean_content) > 100:
        logger.warning(
            "[diag:low_coverage] doc=%s block_coverage=%.2f uncovered_chars=%d "
            "content_len=%d blocks=%d — significant content may be invisible to chunker",
            doc_version_id[:12],
            _coverage_ratio,
            _uncovered_chars,
            len(clean_content),
            len(parsed_blocks),
        )
    else:
        logger.debug(
            "[diag:coverage] doc=%s block_coverage=%.2f uncovered_chars=%d",
            doc_version_id[:12],
            _coverage_ratio,
            _uncovered_chars,
        )

    # -- diagnostic: table header temporal analysis (stage-9 relevance) --
    _tables_with_temporal = 0
    for _ti, _pt in enumerate(parsed_tables):
        _has_temp, _temp_hdrs = _has_temporal_hint(_pt.headers)
        if _has_temp:
            _tables_with_temporal += 1
        logger.info(
            "[diag:table] doc=%s table_idx=%d rows=%s cols=%s "
            "headers=%s header_row_idx=%s has_temporal_headers=%s "
            "temporal_matches=%s parse_method=%s",
            doc_version_id[:12],
            _ti,
            _pt.row_count,
            _pt.col_count,
            _pt.headers,
            _pt.header_row_index,
            _has_temp,
            _temp_hdrs,
            _pt.parse_method,
        )
    if parsed_tables:
        logger.info(
            "[diag:table_summary] doc=%s tables_total=%d tables_with_temporal_headers=%d "
            "NOTE: table_extract.period_granularity and .units_detected are always NULL at stage_02",
            doc_version_id[:12],
            len(parsed_tables),
            _tables_with_temporal,
        )

    if not parsed_blocks and len(clean_content) > 0:
        parsed_blocks.append(
            ParsedBlock(
                block_type="PARAGRAPH",
                span_start=0,
                span_end=len(clean_content),
                parse_confidence=0.5,
                language_hint=primary_language,
            )
        )

    # -----------------------------------------------------------
    # FIX: Deduplicate parsed_blocks BEFORE chunking.
    #
    # Gap recovery and paragraph detection can produce blocks with
    # identical (span_start, span_end, block_type, block_level).
    # If these duplicates reach the chunker, it generates chunks
    # with identical natural keys (doc_version_id, span_start,
    # span_end, chunk_type), which violate the UNIQUE index on
    # the chunk table and roll back the entire document.
    #
    # Previously, dedup only happened on block_rows (after
    # chunking).  Moving it here eliminates the root cause.
    # -----------------------------------------------------------
    _seen_parsed_keys: set[tuple] = set()
    deduped_parsed_blocks: list[ParsedBlock] = []
    for pb in parsed_blocks:
        key = (pb.span_start, pb.span_end, pb.block_type,
               pb.block_level if pb.block_level is not None else -1)
        if key not in _seen_parsed_keys:
            _seen_parsed_keys.add(key)
            deduped_parsed_blocks.append(pb)
        else:
            logger.debug(
                "[dedup:parsed_block] Skipping duplicate parsed block: "
                "type=%s span=%d-%d level=%s",
                pb.block_type, pb.span_start, pb.span_end, pb.block_level,
            )
    if len(deduped_parsed_blocks) < len(parsed_blocks):
        logger.info(
            "[dedup:parsed_block] Removed %d duplicate parsed block(s) for "
            "doc=%s BEFORE chunking (prevents chunk natural-key collisions)",
            len(parsed_blocks) - len(deduped_parsed_blocks),
            doc_version_id[:12],
        )
    parsed_blocks = deduped_parsed_blocks

    block_rows: list[BlockRow] = []
    block_id_map: dict[int, str] = {}

    for i, pb in enumerate(parsed_blocks):
        block_id = compute_sha256_id(
            doc_version_id,
            pb.span_start,
            pb.span_end,
            pb.block_type,
            pb.block_level if pb.block_level is not None else "",
        )

        parent_block_id = None
        if pb.parent_index is not None and pb.parent_index in block_id_map:
            parent_block_id = block_id_map[pb.parent_index]

        block_rows.append(
            BlockRow(
                block_id=block_id,
                doc_version_id=doc_version_id,
                block_type=pb.block_type,
                block_level=pb.block_level,
                span_start=pb.span_start,
                span_end=pb.span_end,
                parse_confidence=pb.parse_confidence,
                boilerplate_flag=pb.boilerplate_flag,
                boilerplate_reason=pb.boilerplate_reason,
                parent_block_id=parent_block_id,
                language_hint=pb.language_hint,
                created_in_run_id=run_id,
            )
        )
        block_id_map[i] = block_id

    chunker = DocumentChunker(config.global_settings.chunking)
    raw_chunks = chunker.chunk(clean_content, parsed_blocks)
    if not raw_chunks:
        logger.warning(
            "[diag:chunking_failed] doc=%s content_len=%d blocks=%d reason=no_chunks_generated min_tokens=%d",
            doc_version_id[:12],
            len(clean_content),
            len(parsed_blocks),
            config.global_settings.chunking.min_tokens,
        )
    # -- diagnostic: chunk type distribution --
    _chunk_types = Counter(ct for _, _, ct, _ in raw_chunks)
    logger.info(
        "[diag:chunks] doc=%s chunk_types=%s total_chunks=%d",
        doc_version_id[:12],
        dict(_chunk_types),
        len(raw_chunks),
    )

    chunk_rows: list[ChunkRow] = []
    evidence_spans: list[tuple[int, int]] = []

    # Pre-compute heading context index for O(log n) lookups per chunk
    # (sorted list of (span_end, heading_text) for headings).
    _heading_index: list[tuple[int, str]] = []
    for br in block_rows:
        if br.block_type == "HEADING":
            _heading_index.append(
                (br.span_end, clean_content[br.span_start : br.span_end].lstrip("#").strip())
            )
    _heading_index.sort(key=lambda h: h[0])

    # Reuse the chunker's _estimate_tokens for consistent token counting
    _token_estimator = chunker._estimate_tokens

    for span_start, span_end, chunk_type, block_indices in raw_chunks:
        if span_end <= span_start:
            continue

        chunk_text = clean_content[span_start:span_end]
        if not chunk_text.strip():
            continue

        evidence_spans.append((span_start, span_end))

        block_ids_for_chunk = [block_id_map[i] for i in block_indices if i in block_id_map]
        if not block_ids_for_chunk and block_rows:
            overlapping = [
                br.block_id
                for br in block_rows
                if br.span_start < span_end and br.span_end > span_start
            ]
            block_ids_for_chunk = overlapping[:1] if overlapping else [block_rows[0].block_id]

        if not block_ids_for_chunk:
            continue

        chunk_id = compute_sha256_id(doc_version_id, span_start, span_end, chunk_type)
        evidence_id = compute_sha256_id(doc_version_id, span_start, span_end)

        # O(log n) heading context lookup via bisect on pre-sorted index
        heading_ctx = None
        if _heading_index:
            idx = bisect.bisect_right([h[0] for h in _heading_index], span_start)
            if idx > 0:
                heading_ctx = _heading_index[idx - 1][1]

        chunk_rows.append(
            ChunkRow(
                chunk_id=chunk_id,
                doc_version_id=doc_version_id,
                span_start=span_start,
                span_end=span_end,
                evidence_id=evidence_id,
                chunk_type=chunk_type,
                block_ids=block_ids_for_chunk,
                chunk_text=chunk_text,
                heading_context=heading_ctx,
                retrieval_exclude=0,
                mention_boundary_safe=1,
                token_count_approx=_token_estimator(chunk_text),
                created_in_run_id=run_id,
            )
        )

    table_rows: list[TableExtractRow] = []
    for pt in parsed_tables:
        if pt.block_index not in block_id_map:
            logger.debug(
                "[diag:table_skip] doc=%s table block_index=%d not in block_id_map",
                doc_version_id[:12],
                pt.block_index,
            )
            continue

        block_id = block_id_map[pt.block_index]
        table_id = compute_sha256_id(doc_version_id, block_id, "table_extract")

        table_rows.append(
            TableExtractRow(
                table_id=table_id,
                block_id=block_id,
                doc_version_id=doc_version_id,
                row_count=pt.row_count,
                col_count=pt.col_count,
                headers_json=pt.headers,
                header_row_index=pt.header_row_index,
                parse_quality=pt.parse_quality,
                parse_method=pt.parse_method,
                table_class=None,
                period_granularity=None,
                units_detected=None,
                raw_table_text=pt.raw_table_text,
                created_in_run_id=run_id,
            )
        )

    # -----------------------------------------------------------
    # FIX: Deduplicate blocks by their natural key before insertion.
    # Gap recovery can (rarely) produce a PARAGRAPH block whose span
    # coincides with an existing block due to position-tracking edge
    # cases in _find_uncovered_regions.  A UNIQUE constraint failure
    # would roll back the entire document, so deduplicate up front.
    # -----------------------------------------------------------
    _seen_block_keys: set[tuple] = set()
    deduped_block_rows: list[BlockRow] = []
    for br in block_rows:
        key = (br.doc_version_id, br.span_start, br.span_end, br.block_type,
               br.block_level if br.block_level is not None else -1)
        if key not in _seen_block_keys:
            _seen_block_keys.add(key)
            deduped_block_rows.append(br)
        else:
            logger.debug(
                "[dedup:block] Skipping duplicate block: type=%s span=%d-%d",
                br.block_type, br.span_start, br.span_end,
            )
    if len(deduped_block_rows) < len(block_rows):
        logger.info(
            "[dedup:block] Removed %d duplicate block(s) for doc=%s",
            len(block_rows) - len(deduped_block_rows),
            doc_version_id[:12],
        )
    block_rows = deduped_block_rows

    # -----------------------------------------------------------
    # FIX: Deduplicate chunks by their natural key before insertion.
    #
    # The UNIQUE index on chunk is:
    #   (doc_version_id, span_start, span_end, chunk_type)
    #
    # Duplicate chunks can arise when:
    #   (a) duplicate parsed_blocks survived to chunking (now
    #       prevented by the earlier parsed_block dedup), or
    #   (b) overlap logic in _split_oversized_chunk produces
    #       sub-chunks with identical span boundaries, or
    #   (c) block-aware chunking + validation split produce
    #       coincidental identical spans.
    #
    # This is a defence-in-depth safety net.
    # -----------------------------------------------------------
    _seen_chunk_keys: set[tuple] = set()
    deduped_chunk_rows: list[ChunkRow] = []
    for cr in chunk_rows:
        key = (cr.doc_version_id, cr.span_start, cr.span_end, cr.chunk_type)
        if key not in _seen_chunk_keys:
            _seen_chunk_keys.add(key)
            deduped_chunk_rows.append(cr)
        else:
            logger.warning(
                "[dedup:chunk] Skipping duplicate chunk: doc=%s type=%s "
                "span=%d-%d chunk_id=%s (would violate UNIQUE index "
                "idx_chunk_natural_key)",
                cr.doc_version_id[:12], cr.chunk_type,
                cr.span_start, cr.span_end, cr.chunk_id[:12],
            )
    if len(deduped_chunk_rows) < len(chunk_rows):
        logger.info(
            "[dedup:chunk] Removed %d duplicate chunk(s) for doc=%s "
            "(defence-in-depth; investigate parsed_block dedup if this "
            "occurs frequently)",
            len(chunk_rows) - len(deduped_chunk_rows),
            doc_version_id[:12],
        )
    chunk_rows = deduped_chunk_rows

    # Also deduplicate evidence_spans by (span_start, span_end)
    # since get_or_create_evidence_span handles DB-level dedup,
    # but avoiding redundant calls is cleaner.
    _seen_evidence_keys: set[tuple] = set()
    deduped_evidence_spans: list[tuple[int, int]] = []
    for span_start_e, span_end_e in evidence_spans:
        key = (span_start_e, span_end_e)
        if key not in _seen_evidence_keys:
            _seen_evidence_keys.add(key)
            deduped_evidence_spans.append((span_start_e, span_end_e))
    evidence_spans = deduped_evidence_spans

    for br in block_rows:
        db.insert_block(br)

    for span_start, span_end in evidence_spans:
        db.get_or_create_evidence_span(
            doc_version_id=doc_version_id,
            span_start=span_start,
            span_end=span_end,
            run_id=run_id,
            purpose="chunk",
            clean_content=clean_content,
        )

    for cr in chunk_rows:
        db.insert_chunk(cr)

    for tr in table_rows:
        db.insert_table_extract(tr)

    # -----------------------------------------------------------
    # Validation: ensure chunking produced results for non-trivial
    # documents.  This is a hard assertion that will surface any
    # remaining edge cases as explicit failures rather than silent
    # data loss.
    # -----------------------------------------------------------
    if not chunk_rows and len(clean_content) >= 40:
        # 40 chars is ~10 tokens — below any reasonable min_tokens,
        # so no chunks is acceptable.  But for anything larger, this
        # is suspicious.
        _est_tokens = DocumentChunker(config.global_settings.chunking)._estimate_tokens(clean_content)
        if _est_tokens >= config.global_settings.chunking.min_tokens:
            logger.error(
                "[VALIDATION:no_chunks] doc=%s content_len=%d est_tokens=%d "
                "blocks=%d tables=%d raw_chunks=%d — document has enough "
                "content for at least one chunk but none were produced. "
                "This is a chunking bug.",
                doc_version_id[:12],
                len(clean_content),
                _est_tokens,
                len(block_rows),
                len(table_rows),
                len(raw_chunks),
            )
            return False, (
                f"chunking_validation_failed: content_len={len(clean_content)} "
                f"est_tokens={_est_tokens} but 0 chunks produced"
            )
        else:
            logger.warning(
                "[diag:no_chunks] doc=%s content_len=%d est_tokens=%d blocks=%d "
                "tables=%d reason=document_too_small_for_chunking min_tokens=%d "
                "target_tokens=%d raw_chunks_produced=%d",
                doc_version_id[:12],
                len(clean_content),
                _est_tokens,
                len(block_rows),
                len(table_rows),
                config.global_settings.chunking.min_tokens,
                config.global_settings.chunking.target_tokens,
                len(raw_chunks),
            )
    elif not chunk_rows:
        logger.info(
            "[diag:no_chunks_tiny] doc=%s content_len=%d — too small for chunking",
            doc_version_id[:12],
            len(clean_content),
        )

    quality_calc = ParseQualityCalculator()
    parse_quality = quality_calc.calculate(parsed_blocks, parsed_tables, len(clean_content))

    logger.info(
        "Processed doc %s: %d blocks, %d chunks, %d tables, %d evidence_spans, quality=%.3f",
        doc_version_id[:12],
        len(block_rows),
        len(chunk_rows),
        len(table_rows),
        len(evidence_spans),
        parse_quality,
    )
    # -- diagnostic: output summary for downstream tracing --
    _table_chunks = sum(1 for cr in chunk_rows if cr.chunk_type == "table_summary")
    _tables_with_headers = sum(1 for tr in table_rows if tr.headers_json)
    _tables_pg_set = sum(1 for tr in table_rows if tr.period_granularity is not None)
    _tables_units_set = sum(1 for tr in table_rows if tr.units_detected is not None)
    logger.info(
        "[diag:output] doc=%s blocks=%d chunks=%d (table_chunks=%d) "
        "table_extracts=%d (with_headers=%d period_granularity_set=%d units_detected_set=%d) "
        "evidence_spans=%d quality=%.3f",
        doc_version_id[:12],
        len(block_rows),
        len(chunk_rows),
        _table_chunks,
        len(table_rows),
        _tables_with_headers,
        _tables_pg_set,
        _tables_units_set,
        len(evidence_spans),
        parse_quality,
    )

    return True, None


def run_stage(
    db: Stage02DatabaseInterface,
    config: Config,
    run_id: str,
    config_hash: str,
) -> tuple[int, int, int, int]:
    """
    Execute stage 02 parse for all eligible documents.

    :returns: Tuple of (ok_count, failed_count, blocked_count, skipped_count).
    """
    iteration_set = db.get_iteration_set(run_id)
    logger.info("Stage 02 iteration set: %d documents", len(iteration_set))

    # -----------------------------------------------------------
    # Pre-flight diagnostics: if iteration set is empty, log why.
    # This distinguishes "no documents in DB" from "all documents
    # already processed" from "all docs blocked on stage_01".
    # -----------------------------------------------------------
    if not iteration_set:
        _diag_rows = db._fetchall(
            "SELECT COUNT(*) AS cnt FROM document_version"
        )
        _total_docvers = _diag_rows[0]["cnt"] if _diag_rows else 0

        _diag_rows = db._fetchall(
            "SELECT COUNT(*) AS cnt FROM document"
        )
        _total_docs = _diag_rows[0]["cnt"] if _diag_rows else 0

        if _total_docvers == 0:
            logger.warning(
                "[preflight:empty_db] document_version table has 0 rows. "
                "Has stage_01_ingest been run for this database? "
                "(document table has %d rows, run_id=%s)",
                _total_docs, run_id,
            )
        else:
            # Documents exist but none are eligible — check status breakdown
            _diag_rows = db._fetchall(
                """
                SELECT dss.status, COUNT(*) AS cnt
                FROM doc_stage_status dss
                WHERE dss.stage = ?
                GROUP BY dss.status
                """,
                (STAGE_NAME,),
            )
            _status_dist = {r["status"]: r["cnt"] for r in _diag_rows} if _diag_rows else {}

            _diag_rows = db._fetchall(
                """
                SELECT dss.status, COUNT(*) AS cnt
                FROM doc_stage_status dss
                WHERE dss.stage = ?
                GROUP BY dss.status
                """,
                (PREREQUISITE_STAGE,),
            )
            _prereq_dist = {r["status"]: r["cnt"] for r in _diag_rows} if _diag_rows else {}

            # Count docs with NO stage_02 status at all
            _diag_rows = db._fetchall(
                """
                SELECT COUNT(*) AS cnt
                FROM document_version dv
                LEFT JOIN doc_stage_status dss
                    ON dss.doc_version_id = dv.doc_version_id
                    AND dss.stage = ?
                WHERE dss.status IS NULL
                """,
                (STAGE_NAME,),
            )
            _no_status = _diag_rows[0]["cnt"] if _diag_rows else 0

            logger.warning(
                "[preflight:no_eligible_docs] %d document_version(s) exist "
                "but 0 are eligible for stage_02. Breakdown: "
                "stage_02 statuses=%s | stage_01 statuses=%s | "
                "docs with no stage_02 status=%d | "
                "HINT: if all are 'ok', stage_02 already completed for all docs. "
                "If all are 'failed', check that upsert_doc_stage_status wrote "
                "'failed' (not 'ok') in the prior run. "
                "If stage_01 has no 'ok' entries, run stage_01 first.",
                _total_docvers,
                _status_dist,
                _prereq_dist,
                _no_status,
            )

    ok_count = 0
    failed_count = 0
    blocked_count = 0
    skipped_count = 0
    # -- diagnostic accumulators (stage-9 root cause investigation) --
    _diag_docs_with_tables = 0
    _diag_total_tables = 0
    _diag_tables_with_headers = 0
    _diag_tables_with_temporal_headers = 0

    for publisher_id, url_normalized, doc_version_id in iteration_set:
        prereq_status = db.get_prerequisite_status(doc_version_id)

        if prereq_status is None or prereq_status.status != "ok":
            blocking_status = prereq_status.status if prereq_status else "missing"
            error_msg = f"prerequisite_not_ok:{PREREQUISITE_STAGE}:{blocking_status}"

            existing = db.get_doc_stage_status(doc_version_id, STAGE_NAME)
            if existing and existing.status == "blocked" and existing.error_message == error_msg:
                blocked_count += 1
                continue

            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="blocked",
                    error_message=error_msg,
                )
            blocked_count += 1
            logger.debug("Blocked doc %s: %s", doc_version_id[:12], error_msg)
            continue

        try:
            with db.transaction():
                success, error_msg = process_document(
                    db, doc_version_id, config, run_id, config_hash
                )

                if success:
                    db.upsert_doc_stage_status(
                        doc_version_id=doc_version_id,
                        stage=STAGE_NAME,
                        run_id=run_id,
                        config_hash=config_hash,
                        status="ok",
                    )
                    ok_count += 1
                else:
                    raise DBError(error_msg or "Unknown processing error")

        except Exception as e:
            error_str = str(e)
            # Enhanced logging for UNIQUE constraint failures to aid debugging
            if "UNIQUE constraint" in error_str:
                logger.warning(
                    "Failed to process doc %s: %s | "
                    "DIAGNOSTIC: This indicates duplicate rows were generated "
                    "despite deduplication. Check parsed_blocks and chunks for "
                    "identical natural keys. Error detail: %s",
                    doc_version_id[:12], error_str, error_str,
                )
            else:
                logger.warning(
                    "Failed to process doc %s: %s",
                    doc_version_id[:12], error_str,
                )
            with db.transaction():
                db.upsert_doc_stage_status(
                    doc_version_id=doc_version_id,
                    stage=STAGE_NAME,
                    run_id=run_id,
                    config_hash=config_hash,
                    status="failed",
                    error_message=str(e)[:500],
                )
            failed_count += 1

    # -- diagnostic: aggregate stage-02 output summary for stage-9 investigation --
    try:
        _te_summary = db.get_table_extract_run_summary(run_id)
        if _te_summary:
            logger.info(
                "[diag:run_summary] table_extracts: total=%s with_headers=%s "
                "with_period_granularity=%s with_units_detected=%s "
                "(period_granularity and units_detected are always NULL at stage_02 — "
                "stage_08 must populate these for stage_09 to produce metric series)",
                _te_summary["total"],
                _te_summary["with_headers"],
                _te_summary["with_period_gran"],
                _te_summary["with_units"],
            )
    except Exception as _diag_err:
        logger.debug("[diag:run_summary] could not query table_extract aggregates: %s", _diag_err)

    return ok_count, failed_count, blocked_count, skipped_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stage 02: Parse")
    parser.add_argument(
        "--run-id", type=str, default="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", help="Pipeline run ID (required)"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("../../../../config/etl_config/")
    )
    parser.add_argument(
        "--source-db", type=Path, default=Path("../../../../database/preprocessed_posts.db")
    )
    parser.add_argument(
        "--working-db", type=Path, default=Path("../../../../database/processed_posts.db")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../../../output/processed/"),
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("../../../../output/processed/logs/")
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for stage 02 parse.

    :returns: Exit code (0 for success, 1 for fatal error).
    """
    args = parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")

    logger.info("Starting stage 02 parse with run_id=%s", args.run_id)

    config_path = args.config_dir / "config.yaml"
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    config_hash = get_config_version(config)

    try:
        with Stage02DatabaseInterface(args.working_db) as db:
            pipeline_run = db.get_pipeline_run(args.run_id)
            if pipeline_run is None:
                logger.error("Pipeline run not found: %s", args.run_id)
                return 1

            if pipeline_run.status != "running":
                logger.error(
                    "Pipeline run %s has status '%s', expected 'running'",
                    args.run_id,
                    pipeline_run.status,
                )
                return 1

            ok_count, failed_count, blocked_count, skipped_count = run_stage(
                db, config, args.run_id, config_hash
            )

            logger.info(
                "Stage 02 complete: ok=%d, failed=%d, blocked=%d, skipped=%d",
                ok_count,
                failed_count,
                blocked_count,
                skipped_count,
            )

            attempted = ok_count + failed_count
            if attempted > 0 and failed_count == attempted:
                logger.error(
                    "Systemic failure: all %d attempted documents failed", attempted
                )
                return 0

            return 0

    except DBError as e:
        logger.error("Database error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error in stage 02: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())