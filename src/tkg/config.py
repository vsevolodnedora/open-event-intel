"""Configuration class for TKG."""
from enum import StrEnum


class LlmOptions(StrEnum):
    """LLM options for API calls."""

    gpt41mini = "gpt-4.1-mini"
    gpt41nano = "gpt-4.1-nano"
    gpt41 = "gpt-4.1"

    # embeddings
    text_embedding3mall = "text-embedding-3-small"
    text_embedding3large = "text-embedding-3-large"

    # other models
    o3mini = "o3-mini"
    o1mini = "o1-mini"
    o3pro = "o3-pro"
    o1 = "o1"

# Per 1M tokens; https://www.helicone.ai/llm-cost/provider/openai/model/text-embedding-3-large
MODEL_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "gpt-4.1-nano": 0.1,
    "gpt-4.1-mini": 0.4,
    "o3-mini": 1.1,
    "o1-mini": 1.1,
    "gpt-4.1": 2,
    "o3-pro": 20,
    "o1": 15,
}

class Config:
    """Configuration class for TKG."""

    required_prompts_and_definitions: list[str] = [
        "event_invalidation_prompt.jinja",
        "statement_extraction_prompt.jinja",
        "triplet_extraction_prompt.jinja",
        "label_definitions.yaml",
        "date_extraction_prompt.jinja",
    ]

    statement_embedding_model: str = LlmOptions.text_embedding3large
    statement_embedding_size: int = 256

    statement_extraction_model: LlmOptions = LlmOptions.gpt41mini
    temporal_range_extraction_model: LlmOptions = LlmOptions.gpt41mini
    triple_extraction_model: LlmOptions = LlmOptions.gpt41mini
    invalidation_agent_model: LlmOptions = LlmOptions.gpt41mini

    # Entity Resolution
    entity_resolution_threshold: float = 0.8
    entity_resolution_acronym_thresh: float = 98.0
    invalidation_agent_top_k: int = 10
    invalidation_agent_num_workers: int = 5

    # Invalidation
    invalidation_agent_similarity_threshold: float = 0.5

    # IO
    prompts_path: str = "src/tkg/prompts_and_definitions/"
    preprocessed_db_fpath: str = "database/preprocessed_posts.db"
    tkg_db_fpath: str = "database/tkg.db"
    eval_statements_example_path: str = "./output/tkg_eval/"  # + publisher /
    eval_results_example_path: str = "./output/tkg_eval/"  # + publisher /
    output_path_pub: str = "./output/tkg/"  # + publisher /