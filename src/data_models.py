from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class Publication(BaseModel):
    """Data model for a news publication."""

    id: str # Unique identifier
    url:str # Original publication URL
    text: str # The full text of the post publication
    publisher: str # The name of the publication source (e.g., entsoe, acer etc)
    published_on: datetime # The date when the post was published
    added_on: datetime  # date when the post was added to the database
    title: str | None = None # title of the post

    @field_validator("published_on","added_on", mode="before")
    @classmethod
    def to_datetime(cls, d: Any) -> datetime:
        """Convert input to a datetime object."""
        if isinstance(d, datetime):
            return d
        if hasattr(d, "isoformat"):
            return datetime.fromisoformat(d.isoformat())
        return datetime.fromisoformat(str(d))