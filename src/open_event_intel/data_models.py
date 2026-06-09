"""Shared data models for the open_event_intel package.

This module is the canonical import location for cross-package data models.
``Publication`` is defined alongside the scraping database (it is what
``PostsDatabase.list_publications`` returns and what flows through the entire
scrape → preprocess → ETL → tkg chain), and is re-exported here so other
packages (notably ``src.tkg``) can depend on a stable, layer-neutral location
rather than reaching into ``scraping``.
"""
from open_event_intel.scraping.publications_database import Publication

__all__ = ["Publication"]
