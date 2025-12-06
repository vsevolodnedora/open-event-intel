import asyncio
import os
import sys

from dotenv import load_dotenv

from src.logger import get_logger
from src.tkg.config import Config, LlmOptions
from src.tkg.evaluate_extraction import main_evaluate_statement_extraction_pipeline
from src.tkg.extraction_invalidation_pipeline import main_tkg_pipeline
from src.tkg.prompt_registry import PromptRegistry

load_dotenv() # Load API key for embedding model from .env into os.environ
if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OpenAI API key not set")

logger = get_logger(__name__)

def main_tkg(task:str, publisher:str):
    """Run the main loop."""
    config = Config()
    prompt_registry = PromptRegistry(prompts_path=config.prompts_path)
    prompt_registry.validate_files(filenames=config.required_prompts_and_definitions)

    if task == "eval":
        asyncio.run(main_evaluate_statement_extraction_pipeline(
            db_fpath=config.preprocessed_db_fpath, publisher=publisher,
            limit_publications=1, limit_statements=5, reference_model=LlmOptions.gpt41, production_model=LlmOptions.gpt41mini,suffix="gpt41mini")
        )
        # asyncio.run(main_evaluate_statement_extraction_pipeline(
        #     db_fpath=config.preprocessed_db_fpath, publisher=publisher,
        #     limit_publications=1, limit_statements=5, reference_model=LlmOptions.gpt41, production_model=LlmOptions.gpt41nano,suffix="gpt41nano")
        # )

    if task == "process":
        asyncio.run(main_tkg_pipeline(config=config, prompt_registry=prompt_registry, publisher=publisher,  limit_publications=1, limit_n_statements=5))


if __name__ == "__main__":

    print("launching run_tkg.py")   # noqa: T201

    if len(sys.argv) != 2:
        task = "eval"
        publisher = "entsoe"
    else:
        task = str(sys.argv[1])
        publisher = str(sys.argv[2])

    main_tkg(task=task, publisher=publisher)