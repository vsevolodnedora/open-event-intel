import re
from datetime import datetime


def format_date_to_datetime(published_on:str):
    """Format date to datetime format."""
    # if only date, add default time 12:00
    if re.match(r"^\d{4}-\d{2}-\d{2}$", published_on):
        published_on = published_on + " 12:00:00"
    try:
        published_dt = datetime.fromisoformat(published_on)
    except ValueError as e:
        raise ValueError(
            f"Invalid published_on format: {published_on}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]"
        ) from e
    return published_dt
