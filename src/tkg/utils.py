import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import tiktoken

from src.logger import get_logger
from src.publications_database import Publication
from src.tkg.config import MODEL_PRICES, LlmOptions, TZ

logger = get_logger(__name__)

def ensure_tz(dt: datetime, name: str) -> datetime|None:
    """Ensure the given datetime is tz-aware or not."""
    if dt is None:
        return None
    if dt.tzinfo != TZ:
        logger.warning("%s (%r) not in the requested timzone; normalizing.", name, dt)
        dt = dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)
    return dt

def create_file_name(publication: Publication) -> str:
    """Generate user-friendly filename for a publication of a given publication."""
    safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", publication.title)
    date_str = publication.published_on.strftime("%Y-%m-%d_%H-%M")
    return f"{date_str}__{publication.publisher}__{safe_title}"

def count_tokens(text: str, model:str) -> int:
    """Return the number of tokens for `text` under the embedding model's tokenizer."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost_to_chatgpt_api_call(tokens:int, model_name:LlmOptions)->float:
    """Compute the estimated cost to chatgpt-api call."""
    if model_name.value not in MODEL_PRICES.keys():
        raise ValueError(f"Model {model_name} is not in the list of available models: {list(MODEL_PRICES.keys())}")
    return float(tokens) / 1.e6 * MODEL_PRICES[str(model_name.value)]

def save_publication_to_pickle(publications: List[Publication], directory_path: str = "output/posts_chunked/entsoe/" ) -> None:
    """Save each Publication object into its own pickle file."""
    os.makedirs(directory_path, exist_ok=True)

    for idx, publication in enumerate(publications, start=1):
        # Construct name
        safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", publication.title)
        date_str = publication.published_on.strftime("%Y-%m-%d_%H-%M")
        filename = f"{date_str}_{safe_title}.pkl"
        file_path = os.path.join(directory_path, filename)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(publication, f)
            logger.info(f"Saved publication to {file_path}")
        except Exception as e:
            logger.error(f"Error saving publication {idx}: {e}. Skipping.")

def load_publication_from_pickle(directory_path: str = "output/posts_chunked/entsoe/") -> list[Publication]:
    """Load all pickle files from a directory into a dictionary."""
    loaded_publications = []
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    dir_path = Path(directory_path).resolve()

    for pkl_file in sorted(dir_path.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                publication = pickle.load(f)
                # Ensure it's a Publication object
                # if not isinstance(publication, Publication):
                #     publication = Publication(**publication)
                loaded_publications.append(publication)
                logger.info(f"Loaded publication from {pkl_file.name}")
        except Exception as e:
            logger.error(f"Error loading {pkl_file.name}: {e}. Skipping.")
    logger.info(f"Loaded {len(loaded_publications)} publications.")
    return loaded_publications