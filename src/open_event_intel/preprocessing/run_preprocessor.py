"""
Publication Preprocessor.

A clean, optimized pipeline for preprocessing scraped publications from various publishers.
Extracts actual publication text by removing links, HTML/CSS/JS artifacts, navigation elements,
and other non-content elements.

Features:
- Publisher-specific preprocessing strategies
- Efficient single-pass text processing where possible
- Compiled regex patterns for performance
- Clean configuration via dataclasses
- Comprehensive error handling
- Integration with existing PostsDatabase and Publication models
- Automatic saving of failed publications for debugging

Usage:
    from src.publications_database import PostsDatabase
    from publication_preprocessor_v2 import PublicationPreprocessor

    preprocessor = PublicationPreprocessor()
    source_db = PostsDatabase("scraped.db")
    target_db = PostsDatabase("preprocessed.db")
    preprocessor.process_table(source_db, target_db, "entsoe")
"""

import os
import re
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional, Sequence

from langid import langid

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import PostsDatabase, Publication

logger = get_logger(__name__)

# Data Models for Processing Results

@dataclass
class ProcessingResult:
    """Result of processing a publication."""

    success: bool
    text: str = ""
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


class ProcessingError(Exception):
    """Raised when publication processing fails."""

    pass

# Compiled Regex Patterns (Performance Optimization)

class Patterns:
    """Pre-compiled regex patterns for text cleaning."""

    # Link patterns
    IMAGE_LINK = re.compile(
        r"!\[.*?\]\([^)]+?\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^)]*?)?\)",
        re.IGNORECASE
    )
    DOC_LINK = re.compile(
        r"\[([^\]]+)\]\([^)]+?\.(?:html|aspx|pdf|doc|docx)(?:\?[^)]*?)?\)",
        re.IGNORECASE
    )
    EMPTY_LINK = re.compile(r"\[\]\(https?://[^)]+\)", re.IGNORECASE)
    GENERIC_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
    MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]+\)")

    # Content patterns
    BLOCK_SEPARATOR = re.compile(r"\n\s*\n")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    MULTIPLE_SPACES = re.compile(r" {2,}")
    LEADING_TRAILING_WHITESPACE = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)

    # HTML/JS artifacts
    HTML_TAGS = re.compile(r"<[^>]+>")
    JS_CODE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
    CSS_CODE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
    HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;")

    # Special characters
    ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

    @classmethod
    def strip_all_links(cls, text: str) -> str:
        """Remove all markdown links, keeping link text."""
        text = cls.IMAGE_LINK.sub("", text)
        text = cls.EMPTY_LINK.sub("", text)
        text = cls.MARKDOWN_LINK.sub(r"\1", text)
        return text

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace in text."""
        text = cls.MULTIPLE_SPACES.sub(" ", text)
        text = cls.MULTIPLE_NEWLINES.sub("\n\n", text)
        text = cls.LEADING_TRAILING_WHITESPACE.sub("", text)
        return text.strip()

    @classmethod
    def clean_html_artifacts(cls, text: str) -> str:
        """Remove HTML/CSS/JS artifacts."""
        text = cls.JS_CODE.sub("", text)
        text = cls.CSS_CODE.sub("", text)
        text = cls.HTML_TAGS.sub("", text)
        text = cls.ZERO_WIDTH.sub("", text)
        return text


# Text Processing Utilities

class TextProcessor:
    """Efficient text processing utilities."""

    @staticmethod
    def find_marker(text: str, markers: Sequence[str], find_last: bool = False) -> Optional[int]:
        """
        Find the position of the first (or last) matching marker in text.

        :param text: Text to search in
        :param markers: Sequence of marker strings to look for
        :param find_last: If True, return position of last match; otherwise first

        :returns: Position of marker, or None if not found
        """
        positions = []
        for marker in markers:
            pos = text.find(marker)
            if pos != -1:
                positions.append((pos, len(marker)))

        if not positions:
            return None

        if find_last:
            pos, length = max(positions, key=lambda x: x[0])
        else:
            pos, length = min(positions, key=lambda x: x[0])

        return pos

    @staticmethod
    def extract_between_markers(
            text: str,
            start_markers: Sequence[str],
            end_markers: Sequence[str],
            include_start: bool = False
    ) -> tuple[str, bool, bool]:
        """
        Extract text between start and end markers.

        :returns: Tuple of (extracted_text, success)
        """
        if not start_markers and not end_markers:
            return text, True, True

        start_pos = 0
        if start_markers:
            for marker in start_markers:
                pos = text.find(marker)
                if pos != -1:
                    start_pos = pos if include_start else pos + len(marker)
                    break
            else:
                return text, False, True  # No start marker found

        end_pos = len(text)
        if end_markers:
            for marker in end_markers:
                pos = text.find(marker, start_pos)
                if pos != -1:
                    end_pos = pos
                    break
            else:
                return text, True, False  # No end marker found

        if start_pos >= end_pos:
            return "", False, False

        return text[start_pos:end_pos].strip(), True, True

    @staticmethod
    def filter_lines(
            text: str,
            *,
            prefix_blacklist: Optional[Sequence[str]] = None,
            exact_blacklist: Optional[Sequence[str]] = None,
            contains_blacklist: Optional[Sequence[str]] = None,
            skip_first_n: int = 0
    ) -> str:
        """
        Filter lines based on various criteria in a single pass.

        :param text: Input text
        :param prefix_blacklist: Remove lines starting with these strings
        :param exact_blacklist: Remove lines exactly matching these strings
        :param contains_blacklist: Remove lines containing these strings
        :param skip_first_n: Skip first N lines
        :returns: Filtered text
        """
        lines = text.split("\n")

        if skip_first_n > 0:
            lines = lines[skip_first_n:]

        # Convert to sets/tuples for faster lookup
        prefix_set = tuple(prefix_blacklist) if prefix_blacklist else ()
        exact_set = frozenset(exact_blacklist) if exact_blacklist else frozenset()
        contains_list = list(contains_blacklist) if contains_blacklist else []

        filtered = []
        for line in lines:
            # Skip empty lines' checks but keep them
            stripped = line.strip()

            # Check exact match
            if stripped in exact_set:
                continue

            # Check prefix match (using startswith with tuple is optimized)
            if prefix_set and stripped.startswith(prefix_set):
                continue

            # Check contains (less common, do last)
            if contains_list and any(bl in line for bl in contains_list):
                continue

            filtered.append(line)

        return "\n".join(filtered)

    @staticmethod
    def remove_blocks_containing(text: str, blacklist: Sequence[str]) -> str:
        """
        Remove entire paragraph blocks containing blacklisted content.

        A block is defined as text separated by blank lines.
        """
        if not blacklist:
            return text

        blocks = Patterns.BLOCK_SEPARATOR.split(text)
        filtered_blocks = []

        for block in blocks:
            if not any(bl in block for bl in blacklist):
                filtered_blocks.append(block)

        return "\n\n".join(filtered_blocks)


# Publisher Strategy Protocol and Base Class
@dataclass
class PublisherConfig:
    """Configuration for a publisher's preprocessing rules."""

    name: str
    start_markers: list[str] = field(default_factory=list)
    end_markers: list[str] = field(default_factory=list)
    prefix_blacklist: list[str] = field(default_factory=list)
    exact_blacklist: list[str] = field(default_factory=list)
    block_blacklist: list[str] = field(default_factory=list)
    skip_first_lines: int = 0
    max_lines: Optional[int] = None
    strip_links: bool = True
    strip_images: bool = True
    clean_html: bool = True
    prefer_german: bool = False
    title_blacklist: list[str] = field(default_factory=list)
    # Callable for dynamic start markers (e.g., date-based)
    dynamic_start_marker: Optional[Callable[[datetime], str]] = None


class BasePublisherStrategy(ABC):  # noqa: B024
    """Base class for publisher-specific preprocessing strategies."""

    def __init__(self, config: PublisherConfig):
        """Initialize the publisher strategy."""
        self.config = config

    @property
    def name(self) -> str:
        """Return the name of the publisher strategy."""
        return self.config.name

    def process(self, text: str, publication: Publication) -> ProcessingResult:  # noqa: C901
        """Process publication text with common pipeline + publisher-specific customization."""
        warnings = []

        try:
            # Pre-processing hook
            text = self.pre_process(text, publication)

            # Build dynamic markers if configured
            start_markers = list(self.config.start_markers)
            if self.config.dynamic_start_marker:
                dynamic_marker = self.config.dynamic_start_marker(publication.published_on)
                start_markers.append(dynamic_marker)

            # Extract content between markers
            if start_markers or self.config.end_markers:
                text, success_start, success_end = TextProcessor.extract_between_markers(
                    text, start_markers, self.config.end_markers
                )
                if not success_start or not success_end:
                    if start_markers and self.config.end_markers:
                        # Both required but not found
                        return ProcessingResult(
                            success=False,
                            error=f"Could not find start/end markers (start={success_start} end={success_end}) for {publication.title}"
                        )
                    warnings.append(f"Markers not found, processing entire text (start={success_start} end={success_end}) for {publication.title}")

            # Filter lines
            text = TextProcessor.filter_lines(
                text,
                prefix_blacklist=self.config.prefix_blacklist,
                exact_blacklist=self.config.exact_blacklist,
                skip_first_n=self.config.skip_first_lines
            )

            # Remove blacklisted blocks
            if self.config.block_blacklist:
                text = TextProcessor.remove_blocks_containing(text, self.config.block_blacklist)

            # Clean links and images
            if self.config.strip_images:
                text = Patterns.IMAGE_LINK.sub("", text)

            if self.config.strip_links:
                text = Patterns.strip_all_links(text)

            # Clean HTML artifacts
            if self.config.clean_html:
                text = Patterns.clean_html_artifacts(text)

            # Normalize whitespace
            text = Patterns.normalize_whitespace(text)

            # Post-processing hook
            text = self.post_process(text, publication)

            # Validate result
            lines = text.split("\n")
            if len(lines) <= 1 and len(text) < 50:
                warnings.append("Result is very short, may be incomplete")

            if self.config.max_lines and len(lines) > self.config.max_lines:
                warnings.append(f"Result has {len(lines)} lines, exceeds max_lines={self.config.max_lines}")

            return ProcessingResult(success=True, text=text, warnings=warnings)

        except Exception as e:
            logger.exception(f"Error processing {publication.title}")
            return ProcessingResult(success=False, error=str(e))

    def pre_process(self, text: str, publication: Publication) -> str:
        """Set a hook for publisher-specific pre-processing. Override in subclasses."""
        return text

    def post_process(self, text: str, publication: Publication) -> str:
        """Set a hook for publisher-specific post-processing. Override in subclasses."""
        return text


class GenericPublisherStrategy(BasePublisherStrategy):
    """Generic strategy that uses configuration without customization."""

    pass


# Publisher-Specific Strategies
class SmardStrategy(BasePublisherStrategy):
    """Strategy for SMARD publications with extensive boilerplate."""

    def pre_process(self, text: str, publication: Publication) -> str:
        """Perform preprocessing on text."""
        # SMARD has a lot of chart-related content that should be removed
        # Remove Highcharts artifacts early
        text = re.sub(r"Created with Highcharts.*?(?=\n|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"Chart Created with Highstock.*?(?=\n|$)", "", text, flags=re.DOTALL)
        return text

    def post_process(self, text: str, publication: Publication) -> str:
        """Remove multiple '* ...' list/navigation lines from the text."""
        patterns = [
            # Lines starting with one or more "*" bullets (e.g. "* Startseite", "* * Drucken ...")
            r"(?m)^\s*\*+(?:\s+\*+)*\s+.*$",
            # Bullet-only lines (just in case)
            r"(?m)^\s*\*+\s*$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)

        # Clean up excessive blank lines created by removals
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class EexStrategy(BasePublisherStrategy):
    """Strategy for EEX press releases."""

    def pre_process(self, text: str, publication: Publication) -> str:
        """Perform preprocessing on text."""
        # EEX has specific header format: "# EEX Press Release - MM/DD/YYYY"
        # The dynamic_start_marker handles this, but we clean up any remaining artifacts
        return text


class CLEWStrategy(BasePublisherStrategy):
    """Strategy for CLEW press releases."""

    def post_process(self, text: str, publication: Publication) -> str:
        """Perform postprocessing on text."""
        patterns = [
            r"(?m)^\s*\*\s*\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4},\s+\d{2}:\d{2}\s*$",
            "(?m)^\s*\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4},\s+\d{2}:\d{2}\s*$"
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        # Clean up excessive blank lines created by removals
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text


class BnetzaStrategy(BasePublisherStrategy):
    """Strategy for EEX press releases."""

    def pre_process(self, text: str, publication: Publication) -> str:
        """Perform preprocessing on text."""
        # BNETZA has specific header format: "Erscheinungsdatum 28.04.2025"
        # The dynamic_start_marker handles this, but we clean up any remaining artifacts
        text = re.sub(
            r"\bErscheinungsdatum\s+(\d{2}\.\d{2}\.\d{4})\b",
            r"\1",
            text
        )
        return text


class EntsoeStrategy(BasePublisherStrategy):
    """Strategy for ENTSO-E publications."""

    def post_process(self, text: str, publication: Publication) -> str:
        """Perform postprocessing on text."""
        # Remove disclaimer markers that might have been partially removed
        text = re.sub(r"❗.*?❗", "", text)
        return text


class TransnetBWStrategy(BasePublisherStrategy):
    """Strategy for TransnetBW press releases."""

    def post_process(self, text: str, publication: Publication) -> str:
        """Perform postprocessing on text."""
        # Remove contact info patterns common in TransnetBW releases
        contact_patterns = [
            r"Andrea Jung.*?Unternehmenskommunikation.*?(?=\n\n|\Z)",
            r"Kathrin Egger.*?Pressesprecherin.*?(?=\n\n|\Z)",
            r"Clemens von Walzel.*?Teamleiter.*?(?=\n\n|\Z)",
        ]
        for pattern in contact_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
        return text


# Date Formatting Utilities
class DateFormats:
    """Date formatting utilities for dynamic markers."""

    @staticmethod
    def d_m_yyyy(dt: datetime) -> str:
        """Format as D.M.YYYY (no zero padding)."""
        return f"{dt.day}.{dt.month}.{dt.year}"

    @staticmethod
    def dd_mm_yyyy(dt: datetime) -> str:
        """Format as D.M.YYYY (no zero padding)."""
        return f"{dt.day:02d}.{dt.month:02d}.{dt.year}"

    @staticmethod
    def mm_dd_yyyy(dt: datetime) -> str:
        """Format as MM/DD/YYYY."""
        return f"{dt.month:02d}/{dt.day:02d}/{dt.year}"

    @staticmethod
    def yyyy_mm_dd(dt: datetime) -> str:
        """Format as YYYY-MM-DD."""
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def dd_month_YYYY_comma_HH_MM(dt: datetime):  # noqa: N802
        """Format as DD-MM-YYYY comma-HH-MM."""
        return dt.strftime("%d %b %Y, %H:%M")

# Publisher Configuration Registry


def create_default_configs() -> dict[str, PublisherConfig]:
    """Create default configurations for all supported publishers."""
    # Common blacklists
    SMARD_PREFIX_BLACKLIST = [  # noqa: N806
        "Suchbegriff eingeben", "[Direkt zum Inhalt", "![Logo der Bundesnetzagentur]",
        "[ ![Logo der Bundesnetzagentur]", "### Suchformular", "[ ![Strommarktdaten Logo]",
        "## Menü", "[ Menü Menu ]", "  * [Startseite]", "  * [Bundesnetzagentur.de]",
        "  * [Datennutzung]", "  * [Benutzerhandbuch]", "  * [ Informationen in Gebärdensprache ]",
        "  * [ Informationen in leicht verständlicher Sprache ]", "  * [ Login ]",
        "  * Link kopieren", "  * [ RSS-Feed ]", "  * [ English  ]",
        "  * [Energiemarkt aktuell]", "  * [Energiedaten kompakt]", "  * [Marktdaten visualisieren]",
        "  * [Deutschland im Überblick]", "  * [Energiemarkt erklärt]", "  * [Daten herunterladen]",
        "Hinweis: Diese Webseite", "Feedbackformular schließen", "  * [Strom]", "  * [Gas]",
        "## Strom", "# Energieträgerscharfe", "  *     * [ Drucken ]", "    * Teilen auf",
        "    * Artikel zu Favoriten", "    * Über E-Mail teilen", "  * Feedback",
        "Importe je Energieträger", "Tabelle anzeigen", "Diagramm anzeigen",
        "  * ### Grafik exportieren", "    * PDF", "    * SVG", "    * PNG", "    * JPEG",
        "  * ### Tabelle exportieren", "    * CSV", "    * XLS", "## Schlagwörteliste",
        "  * [Außenhandel]", "[Link](https://www.smard.de", "© Bundesnetzagentur",
        "  * [Tickerhistorie]", "  * [Datenschutzerklärung]", "  * [Impressum]",
        "  * [Über SMARD]", "  * Wir verwenden optionale Cookies", "Alle Cookies zulassen",
        "# Feedback mitteilen", "Weitere Informationen zur Berechnungsmethode",
        "_____________________________________", "Die Adresse dieser Seite wird beim",
        "Pflichtfelder sind mit einem", "Die Übermittlung ist fehlgeschlagen",
        "  * [--- accessibility.error.message ---]", "Name |  ", "---|---",
        "Thema |", "E-Mail |", "Text* |", "Phone |", "[--- notification.close ---]",
        "[--- dialog.name.close ---]", "# Namen eingeben", "Geben Sie der von Ihnen",
        "Default-Daten Live-Daten", "# Link kopieren", "  * [Alle Artikel]",
        "# Marktdaten visualisieren", "aktualisierte Daten verfügbar",
        "  * [Rekordwerte]", "  * [Verbindungsleitungen]", "[Marktdaten visualisieren]",
        "![](https://www.smard.de/resource", "  * [Kraftwerksabschaltung]",
        "  * [Marktdesign]", "  * [Großhandelsstrompreis]", "  * [Erneuerbare Energien]",
        "#  Lesen Sie auch", "  * [ Der Strommarkt im", "  * [Netzstabilität]",
        "  * [Netzengpassmanagement]", "  * [ Netzengpassmanagement",
        "  * [ Energiemarkt aktuell", "  * [Sturmtief]", "  * [Systemstabilität]",
        "> Quelle: smard.de", "![Strommarkt", "![Der Stromhandel", "![Auch bei",
        "![Stromerzeugung", "![Stromverbrauch", "![Solarpanel", "![Electricity trade",
        "![Ein Umspannwerk",
    ]

    SMARD_EXACT_BLACKLIST = [  # noqa: N806
        "Deutschland/Luxemburg", "Dänemark 1", "Dänemark 2", "Frankreich", "Niederlande",
        "Österreich", "Polen", "Schweden 4", "Schweiz", "Tschechien", "DE/AT/LU",
        "Italien (Nord)", "Slowenien", "Ungarn", "Biomasse", "Wasserkraft",
        "Wind Offshore", "Wind Onshore", "Photovoltaik", "Sonstige Erneuerbare",
        "Kernenergie", "Braunkohle", "Steinkohle", "Erdgas", "Pumpspeicher",
        "Sonstige Konventionelle", "Stromverbrauch - Realisierter Stromverbrauch",
        "Netzlast", "Nettoexport", "diese Artikel", "URL:", "Nach oben",
        "Auflösung ändern", "Auflösung ändernAbbrechen", "Mehr", "Mehr ",
        "Annehmen ", "Es trat ein Fehler bei der Erstellung der Exportdatei auf.",
    ]

    SMARD_BLOCK_BLACKLIST = ["Created with Highcharts", "Chart Created with Highstock"]  # noqa: N806

    ENERGY_WIRE_PREFIX_BLACKLIST = [  # noqa: N806
        "### ", "  * ", "[News](https://www.cleanenergywire.org/news",
        "[« previous news]", "[](https://www.facebook.com", "[](https://twitter.com/",
        "[](https://www.linkedin.com", "All texts created by the Clean Energy Wire",
        "[![](https://www.cleanenergywire.org", "![](https://www.cleanenergywire.org/sites",
        "If you enjoyed reading this article", "#### Support our work",
        "[Make a Donation](https://www.cleanenergywire.org/support-us)",
    ]

    # Add author links dynamically
    ENERGY_WIRE_AUTHORS = [  # noqa: N806
        "Benjamin Wehrmann", "Carolina Kyllmann", "Kira Taylor", "Sören Amelang",
        "Julian Wettengel", "Ruby Russel", "Ferdinando Cotugno", "Sam Morgan",
        "Dave Keating", "Kerstine Appunn", "Edgar Meza", "Jack McGovan",
        "Michael Phillis", "Jennifer Collins", "Franca Quecke", "Emanuela Barbiroglio",
        "Rudi Bressa", "Giorgia Colucci", "Yasmin Appelhans", "Bennet Ribbeck",
        "Joey Grostern", "Ben Cooke", "Milou Dirkx", "Rachel Waldholz",
        "Camille Lafrance", "Juliette Portala", "Isabel Sutton",
    ]
    for author in ENERGY_WIRE_AUTHORS:
        ENERGY_WIRE_PREFIX_BLACKLIST.append(f"[{author}](https://www.cleanenergywire.org/about-us-clew-team)")

    # Topic links
    ENERGY_WIRE_TOPICS = [  # noqa: N806
        "Electricity", "Business & Jobs", "Factsheet", "Dossier", "Cars", "Cost & Prices",
        "Interview", "Elections & Politics", "Renewables", "Wind", "Industry",
        "Climate & CO2", "Municipal heat planning", "Heating", "Business", "Technology",
        "Resources & Recycling", "Construction", "Gas", "Security", "Transport",
        "Adaptation", "Hydrogen", "Company climate claims", "Agriculture", "Solar",
        "Mobility", "Cities", "Grid", "Storage", "Policy", "Carbon removal", "EU",
        "Efficiency", "Society", "International",
    ]
    for topic in ENERGY_WIRE_TOPICS:
        ENERGY_WIRE_PREFIX_BLACKLIST.append(f"[{topic}](https://www.cleanenergywire.org")

    TRANSNETBW_PREFIX_BLACKLIST = [  # noqa: N806
        "  * [Impressum]", "  * [Datenschutz]", "  * [Nutzungsbedingungen]",
        "  * [AEB]", "  * [Kontakt]", "  * [Netiquette ]",
        "![](https://www.transnetbw.de/_Resources", "Andrea JungLeiterin",
        "Kathrin EggerPressesprecherin", "PDF", "Clemens von WalzelTeamleiter",
        "Matthias RuchserPressesprecher", "JPG5", "JPG1",
        "  * [www.transnetbw.de/de/", "  * [Starte Download von: ",
        "[www.stromgedacht.de]", "Copyright Bild: ", "Pressemitteilung:",
        "[www.transnetbw.de/de/newsroom", "[www.powerlaendle.de]", "[www.sonnen.de]",
        "[www.transnetbw.de/de/netzentwicklung", "![](https://www.transnetbw.de/",
        "/ / / / / / / / ", "<https://ip.ai/",
    ]

    ENTSOE_PREFIX_BLACKLIST = [  # noqa: N806
        "Share this article", "For more information", "You can register to the Public Webinar",
        "Read the complete report here ", "**_R﻿ead the full report", "Visit the ENTSO-E Technopedia",
        "Read the full report", "[Access the ERAA", "No registration is needed",
        "More about the Bidding Zone Review", "Contact:", "Read more and submit your feedback",
        "The Webinar Recording is uploaded", "The Webinar Slides are ", "Register for the VAS webinar",
        "For media enquiries", "**Read the Summer Outlook", "Media requests: ",
        "More information about the ", "_Media contacts:_", " _**ENTSO-E:**",
        "**_DSO Entity:_**", "**Read the full Roadmap", "**To access the webpage",
        "**For media inquiries", "R﻿ead more here", "_More on this", "**R﻿ead the reports",
        "F﻿or more information visit", "Press contact: ", "**Related links:**",
        "Download Consultation Package", "👉 Read more about the consultation",
        "🔗 Homepage: ", "📩 For questions", "Read more here",
        "Read ENTSO-E's full response and storyline here",
    ]

    # Add year-based tariff links
    for year in range(2019, 2028):
        ENTSOE_PREFIX_BLACKLIST.append(f"**{year}** Transmission Tariffs")

    configs = {
        "entsoe": PublisherConfig(
            name="entsoe",
            start_markers=["Button", "#  news "],
            end_markers=[
                "❗**Disclaimer** ❗", "❗Disclaimer❗ ", "#### About ENTSO-E",
                "**About ENTSO-E**", "Share this article", "Sign up for press updates",
                "Read the complete report here", "_About ENTSO-E_w",
            ],
            prefix_blacklist=ENTSOE_PREFIX_BLACKLIST,
            exact_blacklist=[
                "**GET THE MOST POWERFUL NEWSLETTER IN BRUSSELS**",
                "Read the complete report here ", "Visit the ENTSO-E Technopedia here .",
                "* * *",
            ],
            max_lines=100,
        ),

        "eex": PublisherConfig(
            name="eex",
            start_markers=["# EEX Press Release -"],
            end_markers=[
                "**CONTACT**", "**_Contacts:_**", "**Contact**", "**KONTAKT**",
                "**Pressekontakt:**", "Please find the full volume report",
                "**Kontakt:**", "Related Files",
            ],
            dynamic_start_marker=DateFormats.mm_dd_yyyy,
            max_lines=100,
        ),

        "acer": PublisherConfig(
            name="acer",
            start_markers=[],
            end_markers=["## ↓ Related News", "![acer]"],
            prefix_blacklist=["Share on: [Share]"],
            exact_blacklist=["Image", "ACER Report"],
            dynamic_start_marker=DateFormats.d_m_yyyy,
            max_lines=100,
        ),

        "ec": PublisherConfig(
            name="ec",
            start_markers=[
                "  2. News", "  * News blog", "  * News announcement",
                "  * News article", "  * Statement",
            ],
            end_markers=[
                "## Related links", "## **Related links**", "## Related Links",
                "## **Source list for the article data**", "Share this page ",
                "info(at)acer.europa.eu",
            ],
            max_lines=100,
        ),

        "icis": PublisherConfig(
            name="icis",
            start_markers=["[Home](https://www.icis.com/explore)"],
            end_markers=["## Related news"],
            prefix_blacklist=[
                "[Full story](https://www.icis.com/explore/resources/news",
                "[Related news](https://www.icis.com/explore/resources/news",
                "[Related content](https://www.icis.com/explore/resources/news",
                "[Contact us](https://www.icis.com/explor",
                "[Try ICIS](https://www.icis.com/explore/contact",
            ],
            exact_blacklist=["Jump to"],
            max_lines=100,
        ),

        "bnetza": PublisherConfig(
            name="bnetza",
            start_markers=["[Pressemitteilungen](https://www.bundesnetzagentur.de/SharedDocs"],
            end_markers=["[](javascript:void(0);) **Inhalte teilen**", "[](javascript:void\(0\);) **Inhalte teilen**"],
            skip_first_lines=1,
            dynamic_start_marker=DateFormats.dd_mm_yyyy,
            max_lines=100,
        ),

        "smard": PublisherConfig(
            name="smard",
            start_markers=[],
            end_markers=[],
            prefix_blacklist=SMARD_PREFIX_BLACKLIST,
            exact_blacklist=SMARD_EXACT_BLACKLIST,
            block_blacklist=SMARD_BLOCK_BLACKLIST,
            prefer_german=True,
            max_lines=2500, # tables can be long
        ),

        "agora": PublisherConfig(
            name="agora",
            start_markers=["  * Print"],
            end_markers=[
                "##  Stay informed", "## Impressions", "##  Event details",
                "##  Further reading",
            ],
            title_blacklist=["harvesting_policy_recipes_for_aseans_coal_to_clean_transition"],
            max_lines=100,
        ),

        "energy_wire": PublisherConfig(
            name="energy_wire",
            start_markers=[
                "Clean Energy Wire / Handelsblatt", "Tagesspiegel / Clean Energy Wire ",
                "# In brief ", "[](javascript:window.print())",
            ],
            end_markers=["#### Further Reading", "### Ask CLEW"],
            prefix_blacklist=ENERGY_WIRE_PREFIX_BLACKLIST,
            dynamic_start_marker=DateFormats.dd_month_YYYY_comma_HH_MM,
            max_lines=100,
        ),

        "transnetbw": PublisherConfig(
            name="transnetbw",
            start_markers=["Nach oben scrollen"],
            end_markers=["https://de.linkedin.com/company/transnetbw-gmbh"],
            prefix_blacklist=TRANSNETBW_PREFIX_BLACKLIST,
            exact_blacklist=["Mathias Bloch", "Pressesprecher", "m.bloch@sonnen.de", "ZurückWeiter"],
            max_lines=100,
        ),

        "tennet": PublisherConfig(
            name="tennet",
            start_markers=["Zuletzt aktualisiert"],
            end_markers=["## Downloads", "Notwendige Cookies akzeptieren"],
            prefix_blacklist=["[Cookies](https://www.tennet.eu/de/datenschutz)"],
            max_lines=100,
        ),

        "fifty_hertz": PublisherConfig(
            name="fifty_hertz",
            start_markers=["Projektmeldung", "Pressemitteilung"],
            end_markers=["Artikel teilen:"],
            prefix_blacklist=[
                "![](/DesktopModules/LotesNewsXSP",
                "[Download der Pressemitteilung als PDF-Datei]",
            ],
            max_lines=100,
        ),

        "amprion": PublisherConfig(
            name="amprion",
            start_markers=["  2. [ ](https://www.amprion.net/Presse/Pressemitteilungen"],
            end_markers=["**Kontakt:**", "Bei Fragen wenden Sie sich bitte an:", "Seite teilen:"],
            prefix_blacklist=[
                "/Presse%C3%BCbersicht_aktuell.html)",
                "  1. [ ](https://www.amprion.net/",
                "  2. [ ](https://www.amprion.net/",
                "  3. [ ](https://www.amprion.net/",
                "  4. [ ](https://www.amprion.net/",
                "  * [Presse](https://www.amprion.net",
                "    * [ ](https://www.amprion.net",
                "[](tel:+",
            ],
            dynamic_start_marker=DateFormats.dd_mm_yyyy,
            max_lines=100,
        ),
    }

    return configs


# Strategy Factory

class StrategyFactory:
    """Factory for creating publisher-specific strategies."""

    _strategy_classes: dict[str, type[BasePublisherStrategy]] = {
        "smard": SmardStrategy,
        "energy_wire": CLEWStrategy,
        "eex": EexStrategy,
        "bnetza":BnetzaStrategy,
        "entsoe": EntsoeStrategy,
        "transnetbw": TransnetBWStrategy,
    }

    @classmethod
    def create(cls, config: PublisherConfig) -> BasePublisherStrategy:
        """Create appropriate strategy for publisher."""
        strategy_class = cls._strategy_classes.get(config.name, GenericPublisherStrategy)
        return strategy_class(config)

    @classmethod
    def register(cls, name: str, strategy_class: type[BasePublisherStrategy]) -> None:
        """Register a custom strategy class for a publisher."""
        cls._strategy_classes[name] = strategy_class


# Language Detection for German Preference

class LanguageFilter:
    """Filter publications by language preference."""

    @staticmethod
    def detect_language(text: str) -> tuple[str, float]:
        """Detect language of text. Returns (language_code, confidence)."""
        try:
            return langid.classify(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return ("en", 0.0)


    @staticmethod
    def filter_prefer_german(
            publications: list[Publication]
    ) -> list[Publication]:
        """
        Filter publications preferring German versions when both EN/DE exist for same date.

        For dates with exactly 2 publications where one is German, keep only German.
        Otherwise keep English publications.
        """
        if not publications:
            return []

        # Detect languages
        lang_cache: dict[str, tuple[str, float]] = {}
        for pub in publications:
            lang_cache[pub.id] = LanguageFilter.detect_language(pub.text)

        # Group by date
        by_date: dict[str, list[Publication]] = defaultdict(list)
        for pub in publications:
            date_key = pub.published_on.date().isoformat()
            by_date[date_key].append(pub)

        selected: list[Publication] = []

        for date_key, items in by_date.items():
            items.sort(key=lambda x: (x.published_on, x.id))

            if len(items) == 2:
                german = [p for p in items if lang_cache[p.id][0] == "de"]
                if german:
                    logger.info(f"Date {date_key}: Selecting German version")
                    selected.append(german[0])
                    continue

            # Keep English items
            english = [p for p in items if lang_cache[p.id][0] == "en"]
            selected.extend(english if english else items)

        selected.sort(key=lambda x: (x.published_on, x.id))
        return selected


# Corruption detection
class CharacterCorruptionFixer:
    """
    Fixes common character encoding corruption issues.

    Loads corruption mappings from an external file to allow easy updates
    without code changes.
    """

    def __init__(self, mappings_file: str):
        """
        Initialize the fixer by loading corruption mappings from file.

        :param mappings_file: Path to file containing corruption mappings
                          Format: "corrupted → correct" (one per line)
                          Lines starting with # are comments

        :raises:
            FileNotFoundError: If mappings file doesn't exist
            ValueError: If file format is invalid
        """
        self.mappings = self._load_mappings(mappings_file)
        logger.info(f"Loaded {len(self.mappings)} corruption mappings from {mappings_file}")

    def _load_mappings(self, filepath: str) -> dict[str, str]:
        """
        Load corruption mappings from file.

        Expected format:
            # Comments start with #
            Ã¼ → ü
            Ã¶ → ö
            â€˜ → '

        Also supports tab-separated format:
            Ã¼	ü

        :returns:
            Dictionary mapping corrupted strings to correct strings
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Corruption mappings file not found: {filepath}")

        mappings = {}
        line_num = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Try arrow format first: "corrupted → correct"
                if "→" in line:
                    parts = line.split("→", 1)
                    if len(parts) == 2:
                        corrupted = parts[0].strip()
                        correct = parts[1].strip()
                        mappings[corrupted] = correct
                        continue

                # Try tab-separated format: "corrupted\tcorrect"
                if "\t" in line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        corrupted = parts[0].strip()
                        correct = parts[1].strip()
                        mappings[corrupted] = correct
                        continue

                # Invalid format
                logger.warning(f"Skipping invalid line {line_num} in {filepath}: {line[:50]}")

        if not mappings:
            raise ValueError(f"No valid mappings found in {filepath}")

        return mappings

    def fix_text(self, text: str) -> tuple[str, int]:
        """
        Fix character corruption in text.

        :param text: Text with potential character corruption
        :returns:
            Tuple of (corrected_text, total_replacements)
        """
        corrected_text = text
        total_replacements = 0

        for corrupted, correct in self.mappings.items():
            count = corrected_text.count(corrupted)
            if count > 0:
                corrected_text = corrected_text.replace(corrupted, correct)
                total_replacements += count

        return corrected_text, total_replacements

    def scan_result_for_corruption(self, result: ProcessingResult, publication: Publication) -> ProcessingResult:
        """
        Scan and fix character corruption in the processing result.

        Detects and corrects common UTF-8/Latin-1 encoding mismatches and logs
        any corrections made.

        :param result: The processing result to check
        :param publication: The publication being processed (for logging)

        :returns: Updated ProcessingResult with corrected text and additional warnings
        """
        if not result.success or not result.text:
            return result

        corrected_text, total_replacements = self.fix_text(result.text)

        if total_replacements > 0:
            date_str = publication.published_on.strftime("%Y-%m-%d_%H-%M")
            # Truncate title to 50 chars for logging
            title_short = publication.title[:50] if publication.title else "untitled"

            logger.info(f"Character corruption fixed: {publication.publisher} | {date_str} | {title_short} | {total_replacements} characters replaced")

            result.warnings.append(f"Fixed {total_replacements} corrupted characters")
            result.text = corrected_text

        return result


# Main Preprocessor Class
class PublicationPreprocessor:
    """
    Main preprocessor class that orchestrates publication cleaning.

    Uses existing PostsDatabase and Publication models from the project.
    Failed publications are automatically saved to a configurable directory.

    Usage:
        from src.publications_database import PostsDatabase

        corruption_fixer = CharacterCorruptionFixer("possible_corruptions.txt")
        preprocessor = PublicationPreprocessor(corruption_fixer=corruption_fixer)
        source_db = PostsDatabase("scraped.db")
        target_db = PostsDatabase("preprocessed.db")

        preprocessor.process_table(source_db, target_db, "entsoe")
    """

    def __init__(
        self, custom_configs: Optional[dict[str, PublisherConfig]] = None, failed_output_dir: str = "./output/failed_preprocess/", corruption_fpath: str | None = None
    ):
        """
        Initialize preprocessor with optional custom configurations.

        :param custom_configs: Custom publisher configs to override defaults
        :param failed_output_dir: Directory to save publications that fail preprocessing
        :param corruption_fixer: CharacterCorruptionFixer instance for fixing encoding issues
        """
        self.configs = create_default_configs()
        if custom_configs:
            self.configs.update(custom_configs)

        self._strategies: dict[str, BasePublisherStrategy] = {}
        self._failed_output_dir = failed_output_dir
        self.corruption_fixer = CharacterCorruptionFixer(corruption_fpath)

    def _get_strategy(self, publisher: str) -> BasePublisherStrategy:
        """Get or create strategy for publisher."""
        if publisher not in self._strategies:
            if publisher not in self.configs:
                raise ValueError(f"Unknown publisher: {publisher}. Available: {list(self.configs.keys())}")
            self._strategies[publisher] = StrategyFactory.create(self.configs[publisher])
        return self._strategies[publisher]

    def process(
            self,
            publisher: str,
            publication: Publication
    ) -> ProcessingResult:
        """
        Process a single publication.

        :param publisher: Publisher identifier
        :param publication: Publication object to process

        :returns: ProcessingResult with cleaned text or error
        """
        strategy = self._get_strategy(publisher)
        return strategy.process(publication.text, publication)

    def _save_failed_publication(
            self,
            publication: Publication,
            error_message: str
    ) -> str:
        """
        Save a failed publication's raw text as markdown for debugging.

        Uses same filename format as dump_publications_as_markdown for consistency.

        :param publication: The publication that failed processing
        :param error_message: The error message describing why it failed

        :returns: The filepath where the failed publication was saved
        """
        # Create publisher-specific subdirectory
        output_dir = os.path.join(self._failed_output_dir, publication.publisher)
        os.makedirs(output_dir, exist_ok=True)

        # Create filename (same format as dump_publications_as_markdown)
        date_str = publication.published_on.strftime("%Y-%m-%d_%H-%M")
        safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", publication.title or "untitled")[:50].strip("_")
        if not safe_title:
            safe_title = "untitled"
        safe_publisher = re.sub(r"[^A-Za-z0-9_-]", "_", publication.publisher).strip("_") or "publisher"

        filename = f"{date_str}__{safe_publisher}__{safe_title}.md"
        filepath = os.path.join(output_dir, filename)

        # Write file with error header for debugging
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("<!-- PREPROCESSING FAILED -->\n")
            f.write(f"<!-- Error: {error_message} -->\n")
            f.write(f"<!-- URL: {publication.url} -->\n")
            f.write(f"<!-- ID: {publication.id} -->\n")
            f.write(f"<!-- Published: {publication.published_on.isoformat()} -->\n")
            f.write(f"<!-- Title: {publication.title} -->\n\n")
            f.write(publication.text)

        logger.info(f"Saved failed publication to: {filepath}")
        return filepath

    def process_publications_for_the_publisher(  # noqa: C901
        self, source_db: PostsDatabase, target_db: PostsDatabase, table_name: str, *, overwrite: bool = False, allow_failures: bool = False, prefer_german: Optional[bool] = None
    ) -> dict[str, int]:
        """
        Process all publications in a source table and store in target.

        Failed publications are automatically saved to the failed output directory
        as markdown files for debugging purposes.

        :param source_db: Source PostsDatabase instance
        :param target_db: Target PostsDatabase for cleaned publications
        :param table_name: Table name (publisher)
        :param overwrite: Overwrite existing publications in target
        :param allow_failures: Continue on processing failures
        :param prefer_german: Use German version when both EN/DE exist (overrides config)

        :returns: Statistics dict with counts of processed/skipped/failed
        """
        stats = {"processed": 0, "skipped": 0, "failed": 0, "total": 0}
        processing_results = []  # Track individual results: '.' = skipped, 'o' = success, 'x' = failed
        failed_files = []  # Track failed file paths
        first_date = None
        last_date = None

        config = self.configs.get(table_name)
        if not config:
            raise ValueError(f"No configuration for publisher: {table_name}")

        # Ensure target table exists
        target_db.check_create_table(table_name)

        # Get publications using existing database interface
        publications = source_db.list_publications(table_name, sort_date=True)
        stats["total"] = len(publications)
        logger.info(f"Found {len(publications)} publications in {table_name}")

        # Determine if we should prefer German
        use_german_preference = prefer_german if prefer_german is not None else config.prefer_german

        # Apply German preference filter if needed
        if use_german_preference:
            publications = LanguageFilter.filter_prefer_german(publications)
            logger.info(f"After language filter: {len(publications)} publications")

        # Track date range
        if publications:
            first_date = publications[0].published_on
            last_date = publications[-1].published_on

        for pub in publications:
            # Check title blacklist
            if config.title_blacklist and pub.title in config.title_blacklist:
                logger.info(f"Skipping blacklisted title: {pub.title}")
                stats["skipped"] += 1
                processing_results.append(".")
                continue

            # Check if already exists in target
            pub_id = target_db.create_publication_id(pub.url)
            if not overwrite and target_db.is_publication(table_name, pub_id):
                logger.debug(f"Publication already exists, skipping: {pub.url}")
                stats["skipped"] += 1
                processing_results.append(".")
                continue

            # Validate input
            if not pub.text or len(pub.text) < 5:
                logger.warning(f"Skipping publication with invalid text: {pub.url}")
                stats["skipped"] += 1
                processing_results.append(".")
                continue

            # Process
            result = self.process(table_name, pub)

            if not result.success:
                logger.error(f"Failed to process {pub.title}: {result.error}")
                stats["failed"] += 1
                processing_results.append("x")

                # Save failed publication for debugging
                failed_file = self._save_failed_publication(pub, result.error or "Unknown error")
                if failed_file:
                    failed_files.append(failed_file)

                if not allow_failures:
                    raise ProcessingError(f"Failed to process {pub.title}: {result.error}")
                continue

            if pub.published_on > datetime.today() + timedelta(days=1):
                logger.error(f"Skipping publication {pub.title} - {pub.published_on} exceeds today {datetime.today()}")
                continue

            # Check if there are corrupted characters in the text and fix
            if self.corruption_fixer:
                result = self.corruption_fixer.scan_result_for_corruption(result, pub)

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"{pub.title}: {warning}")

            # Check if languages match
            inferred_language, language_certainty = LanguageFilter.detect_language(pub.text)
            if not inferred_language == pub.language:
                logger.error(
                    f"Expected language '{pub.language}' does not match inferred '{inferred_language}' "
                    f"(certainty {language_certainty}) for "
                    f"{pub.published_on.strftime('%Y-%m-%d_%H-%M')}_{pub.publisher}_{pub.title}"
                )

            # Store result using existing database interface
            target_db.add_publication(table_name=table_name, published_on=pub.published_on, title=pub.title, post_url=pub.url, post=result.text, language=inferred_language, overwrite=overwrite)
            stats["processed"] += 1
            processing_results.append("o")

        # Log visual representation of processing results
        if processing_results:
            result_str = " ".join(processing_results)
            date_range = ""
            if first_date and last_date:
                date_range = f" From {first_date.strftime('%d-%m-%Y')} to {last_date.strftime('%d-%m-%Y')}"
            logger.info(f"> [{result_str}]{date_range}")

            # Log failed files if any
            if failed_files:
                logger.info(f"Failed preprocessing files for {table_name}:")
                for failed_file in failed_files:
                    logger.info(f"  - {failed_file}")

        logger.info(f"Completed {table_name}: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")
        return stats

    def process_all_publications(
            self,
            source_db: PostsDatabase,
            target_db: PostsDatabase,
            tables: Optional[list[str]] = None,
            overwrite: bool = False,
            allow_failures: bool = False,
            output_base_dir: str = "./output/posts_cleaned"
    ) -> dict[str, dict[str, int]]:
        """
        Process multiple tables and export as markdown.

        :param source_db: Source database
        :param target_db: Target database
        :param tables: List of tables to process (default: all configured)
        :param overwrite: Overwrite existing publications
        :param allow_failures: Continue on failures
        :param output_base_dir: Base directory for markdown exports

        :returns: Dict mapping table names to their stats
        """
        if tables is None:
            tables = list(self.configs.keys())

        all_stats = {}

        for table_name in tables:
            if table_name not in self.configs:
                logger.warning(f"No configuration for {table_name}, skipping")
                continue

            logger.info(f"Processing {table_name}...")

            try:
                stats = self.process_publications_for_the_publisher(
                    source_db,
                    target_db,
                    table_name,
                    overwrite=overwrite,
                    allow_failures=allow_failures
                )
                all_stats[table_name] = stats

                # Export as markdown using existing database method
                output_dir = os.path.join(output_base_dir, table_name)
                target_db.dump_publications_as_markdown(table_name, output_dir)

            except ProcessingError as e:
                logger.error(f"Processing failed for {table_name}: {e}")
                if not allow_failures:
                    raise
                all_stats[table_name] = {"error": str(e)}

        return all_stats

    @property
    def available_publishers(self) -> list[str]:
        """List available publisher configurations."""
        return list(self.configs.keys())


# CLI Interface


def main():
    """Command-line interface for the preprocessor."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess scraped publications from various publishers")
    parser.add_argument("--source", nargs="?", default="all", help="Publisher to process (default: all)")
    parser.add_argument("--source-db", default="../../../database/scraped_posts.db", help="Path to source database")
    parser.add_argument("--target-db", default="../../../database/preprocessed_posts.db", help="Path to target database")
    parser.add_argument("--output-dir", default="../../../output/posts_cleaned", help="Directory for markdown exports")
    parser.add_argument("--failed-dir", default="../../../output/failed_preprocess/", help="Directory for failed preprocessing outputs")
    parser.add_argument("--corruptions-file", default="../../../config/possible_corruptions.txt", help="Path to file containing character corruption mappings")
    parser.add_argument("--overwrite", default=True, action="store_true", help="Overwrite existing publications")
    parser.add_argument("--allow-failures", default=True, action="store_true", help="Continue processing on failures")
    parser.add_argument("--list-publishers", action="store_true", help="List available publishers and exit")
    parser.add_argument("--metadata-output", default="../../../output/public_view/", help="Directory for metadata export")

    args = parser.parse_args()

    preprocessor = PublicationPreprocessor(failed_output_dir=args.failed_dir, corruption_fpath=args.corruptions_file)

    logger.info("Starting preprocessor...")
    if args.list_publishers:
        logger.info("===== Available publishers: ====== ")
        for name in sorted(preprocessor.available_publishers):
            logger.info(f"  - {name}")
        return

    # Determine which publishers to process
    if args.source == "all":
        publishers = preprocessor.available_publishers
    else:
        if args.source not in preprocessor.configs:
            logger.error(f"Unknown publisher: {args.source}")
            logger.error(f"Available: {', '.join(preprocessor.available_publishers)}")
            return 1
        publishers = [args.source]

    # Check source database exists
    if not os.path.isfile(args.source_db):
        logger.error(f"Source database not found: {args.source_db}")
        return 1

    # Open databases
    source_db = PostsDatabase(args.source_db)
    target_db = PostsDatabase(args.target_db)

    try:
        # Process and export
        all_stats = preprocessor.process_all_publications(source_db, target_db, tables=publishers, overwrite=args.overwrite, allow_failures=args.allow_failures, output_base_dir=args.output_dir)

        # Print summary
        total = {"processed": 0, "skipped": 0, "failed": 0}
        for _table_name, stats in all_stats.items():
            if isinstance(stats, dict) and "error" not in stats:
                for key in total:
                    total[key] += stats.get(key, 0)

        logger.info(f"Total: {total['processed']} processed, {total['skipped']} skipped, {total['failed']} failed")

        # Export metadata
        target_db.export_all_publications_metadata(out_dir=args.metadata_output, format="json", filename="preprocessed_publications_metadata")
        logger.info(f"Updated metadata file at {args.metadata_output}")

    finally:
        source_db.close()
        target_db.close()

    logger.info("Finished preprocessor")


if __name__ == "__main__":
    main()
