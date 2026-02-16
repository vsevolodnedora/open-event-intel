"""Agentic workflows using OpenAI API."""

import asyncio
import uuid
from typing import Any

from jinja2 import Environment
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from open_event_intel.logger import get_logger
from open_event_intel.scraping.publications_database import Publication
from src.tkg.config import Config
from src.tkg.data_models import (
    Entity,
    RawExtraction,
    RawStatement,
    RawStatementList,
    RawTemporalRange,
    TemporalConfidence,
    TemporalEvent,
    TemporalType,
    TemporalValidityRange,
    Triplet,
    parse_date_str,
)
from src.tkg.prompt_registry import PromptRegistry
from src.tkg.utils import count_tokens, estimate_cost_to_chatgpt_api_call

logger = get_logger(__name__)

def get_publication_metadata(publication: Publication) -> dict:
    """Create commonly used summary for the publication."""
    return {
        "publisher": publication.publisher or None,
        "document_type": "News publication",
        "publication_date": publication.published_on,
        "url": publication.url,
        "title": publication.title,
        "document_chunk": None,
    }


class TemporalAgent:
    """
    Handles temporal-based operations for extracting and processing temporal events from text.

    Brings together the steps built up above - chunking, data models, and prompts.
    """

    def __init__(self, config:Config, prompt_registry:PromptRegistry) -> None:
        """Initialize the TemporalAgent with a client."""
        self._client = AsyncOpenAI()
        self._embedding_model = config.statement_embedding_model # embedding for statements
        self._statement_embedding_size = config.statement_embedding_size
        self._statement_extraction_model = config.statement_extraction_model
        self._temporal_range_extraction_model = config.temporal_range_extraction_model
        self._triple_extraction_model = config.triple_extraction_model

        self._prompt_registry = prompt_registry
        self._env = self._initialize_jinja_environment()

    def _initialize_jinja_environment(self) -> Environment:
        """Initialize the jinja environment."""
        # Setup jinja environment with prompt templates from registry
        env = self._prompt_registry.create_environment(
            filters={
                "split_and_capitalize": lambda x: x.replace("_", " ").title()
            }
        )
        return env


    @staticmethod
    def split_and_capitalize(value: str) -> str:
        """Split dict key string and reformat for jinja prompt."""
        return " ".join(value.split("_")).capitalize()

    # API calling methods

    async def get_statement_embedding(self, statement: str) -> list[float]:
        """Get the embedding of a statement."""
        logger.info(
            f"\t Sending statement to embedding {self._embedding_model} with {count_tokens(statement, self._embedding_model)} tokens and "
            f"est. cost = {estimate_cost_to_chatgpt_api_call(count_tokens(statement, self._embedding_model), self._embedding_model):.2f} EUR"
        )

        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=statement,
            dimensions=self._statement_embedding_size,
        )
        return response.data[0].embedding

    # Extractors

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_statements(self, publication: Publication, inputs: dict[str, Any]) -> RawStatementList:
        """
        Determine initial validity date range for a statement.

        :param publication: The publication to analyze.
        :param inputs: Additional input parameters for extraction.

        :return: Statement with updated temporal range.
        """
        # Compose the prompt
        inputs["publication"] = publication.text
        label_definitions = self._prompt_registry.load_yaml_dict("label_definitions.yaml")
        template = self._env.get_template("statement_extraction_prompt.jinja")
        prompt = template.render(
            inputs=inputs,
            definitions=label_definitions,
            json_schema=RawStatementList.model_fields,
        )

        logger.info(
            f"\t Sending extract statement prompt to {self._statement_extraction_model} with {count_tokens(prompt, self._statement_extraction_model)} tokens and "
            f"est. cost = {estimate_cost_to_chatgpt_api_call(count_tokens(prompt, self._statement_extraction_model), self._statement_extraction_model):.4f} EUR"
        )

        # extract list of statements
        response = await self._client.responses.parse(
                model=self._statement_extraction_model,
                temperature=0,
                input=prompt,
                text_format=RawStatementList,
            )

        raw_statements:RawStatementList = response.output_parsed
        logger.info(f"Extracted {len(raw_statements.statements)} raw statements from the publication")

        # Enforce unique IDs for statements after extraction
        for stmt in raw_statements.statements:
            stmt.id = uuid.uuid4()

        for i_stmt, statement in enumerate(raw_statements.statements):
            logger.info(
                f"\t\t{i_stmt}/{len(raw_statements.statements)} | {statement.statement_type} "
                f"| {statement.temporal_type} | {statement.temporal_confidence} | {statement.statement}"
            )

        statements = RawStatementList.model_validate(raw_statements)
        return statements

    async def _return_atemporal_event_temporal_range(self, statement: RawStatement):
        logger.info(f"Skipping temporal range extraction for {statement.temporal_type} and setting confidence to: LOW")
        raw_range = RawTemporalRange(
            valid_at=None,
            invalid_at=None,
            valid_at_confidence=TemporalConfidence.LOW,
            invalid_at_confidence=TemporalConfidence.LOW,
            rationale="Statement is ATEMPORAL - no temporal bounds applicable",
        )
        temp_validity = TemporalValidityRange(
            valid_at=None, invalid_at=None, valid_at_confidence=TemporalConfidence.LOW, invalid_at_confidence=TemporalConfidence.LOW, temporal_extraction_rationale=raw_range.rationale
        )
        return raw_range, temp_validity

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_temporal_range(self, statement: RawStatement, metadata: dict[str, Any]) -> tuple[RawTemporalRange, TemporalValidityRange]:
        """
        Determine initial validity date range for a statement.

        :param statement: Statement to analyze.
        :param metadata:  Publication metadata dates for the statement.

        :return: Tuple of (RawTemporalRange, TemporalValidityRange).
        """
        if statement.temporal_type == TemporalType.ATEMPORAL:
            return await self._return_atemporal_event_temporal_range(statement)

        # Generate prompt from the template and data
        template = self._env.get_template("date_extraction_prompt.jinja")
        inputs = metadata | statement.model_dump()

        # Updated to use new LABEL_DEFINITIONS structure
        label_definitions = self._prompt_registry.load_yaml_dict("label_definitions.yaml")
        prompt = template.render(
            inputs=inputs,
            temporal_guide={statement.temporal_type.value: label_definitions["temporal_type"][statement.temporal_type.value]},
            statement_guide={statement.statement_type.value: label_definitions["statement_type"][statement.statement_type.value]},
            json_schema=RawTemporalRange.model_json_schema(),
        )

        logger.info(
            f"\t Sending extract temporal range prompt to {self._temporal_range_extraction_model} with {count_tokens(prompt, self._temporal_range_extraction_model)} tokens and "
            f"est. cost = {estimate_cost_to_chatgpt_api_call(count_tokens(prompt, self._temporal_range_extraction_model), self._temporal_range_extraction_model):.4f} EUR"
        )

        response = await self._client.responses.parse(
            model=self._temporal_range_extraction_model,
            temperature=0,
            input=prompt,
            text_format=RawTemporalRange,
        )

        raw_validity: RawTemporalRange = response.output_parsed
        logger.info(
            f"\t\tExtracted temporal range valid_at={raw_validity.valid_at} (conf={raw_validity.valid_at_confidence}) "
            f"and invalid_at={raw_validity.invalid_at} (conf={raw_validity.invalid_at_confidence}) "
            f"from statement | {statement.statement_type} | {statement.temporal_type} | {statement.statement}"
        )

        # Convert RawTemporalRange to TemporalValidityRange
        temp_validity = TemporalValidityRange(
            valid_at=parse_date_str(raw_validity.valid_at) if raw_validity.valid_at else None, # -> datetime | None
            invalid_at=parse_date_str(raw_validity.invalid_at) if raw_validity.invalid_at else None, # -> datetime | None
            valid_at_confidence=raw_validity.valid_at_confidence,
            invalid_at_confidence=raw_validity.invalid_at_confidence,
            temporal_extraction_rationale=raw_validity.rationale,
        )

        if temp_validity.valid_at is None:
            pub_raw = inputs.get("publication_date")
            pub_dt = parse_date_str(pub_raw) if pub_raw is not None else None
            if pub_dt is not None:
                logger.warning(f"No valid_at found. Setting publication_date as valid_at={pub_dt} with LOW confidence")
                temp_validity.valid_at = pub_dt
                temp_validity.valid_at_confidence = TemporalConfidence.LOW
            else:
                logger.warning(f"No valid_at found and publication_date={pub_raw!r} could not be parsed; leaving valid_at=None")

        # Heuristic: if range is inverted, drop invalid_at and keep an open-ended 'valid_from'
        if temp_validity.valid_at is not None and temp_validity.invalid_at is not None and temp_validity.valid_at > temp_validity.invalid_at:
            logger.warning(
                "Date range invalid for statement: %s | %s | temporal_conf=%s | valid_at=%s | invalid_at=%s | pub_id=%s. Resetting invalid_at=None (open-ended validity).",
                statement.statement_type,
                statement.temporal_type,
                statement.temporal_confidence,
                temp_validity.valid_at,
                temp_validity.invalid_at,
                statement.publication_id,
            )
            temp_validity.invalid_at = None
            temp_validity.invalid_at_confidence = TemporalConfidence.LOW

        # # EVENT temporal type should not have invalid_at (past events remain true)
        # if statement.temporal_type == TemporalType.EVENT:
        #     temp_validity.invalid_at = None
        #     temp_validity.invalid_at_confidence = TemporalConfidence.LOW

        return raw_validity, temp_validity

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def extract_triplet(self, statement: RawStatement, max_retries: int = 3) -> RawExtraction:
        """Extract triplets and entities from a statement as a RawExtraction object."""
        template = self._env.get_template("triplet_extraction_prompt.jinja")
        predicate_definitions = self._prompt_registry.load_predicate_definitions()
        prompt = template.render(
            statement=statement.statement,
            json_schema=RawExtraction.model_fields,
            predicate_instructions=predicate_definitions,
        )

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"\tSending extract triplet prompt to {self._triple_extraction_model} with {count_tokens(prompt, self._triple_extraction_model)} tokens and "
                    f"est. cost = {estimate_cost_to_chatgpt_api_call(count_tokens(prompt, self._triple_extraction_model), self._triple_extraction_model):.4f} EUR"
                )

                response = await self._client.responses.parse(
                    model=self._triple_extraction_model,
                    temperature=0,
                    input=prompt,
                    text_format=RawExtraction,
                )
                raw_extraction:RawExtraction = response.output_parsed

                logger.info(
                    f"\t\tTriplets ({len(raw_extraction.triplets)}) and entities ({len(raw_extraction.entities)}) "
                    f"extracted successfully from statement {statement.statement_type} "
                    f"| {statement.temporal_type} | {statement.statement}"
                )
                for i_trip, triplet in enumerate(raw_extraction.triplets):
                    logger.debug(f"\t\tTriplet {i_trip}/{len(raw_extraction.triplets)} | SBO: {triplet.subject_name} - {triplet.predicate} - {triplet.object_name}")
                for i_entity, entity in enumerate(raw_extraction.entities):
                    logger.debug(f"\t\tEntity {i_entity}/{len(raw_extraction.entities)} | name: {entity.name} | type: {entity.type}")

                extraction = RawExtraction.model_validate(raw_extraction)
                return extraction
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying...")
                await asyncio.sleep(1)

        raise Exception("All retry attempts failed to extract triplets")

    # Processors

    async def _process_statement(self, publication: Publication, statement: RawStatement, doc_summary: dict[str, Any]) -> tuple[RawTemporalRange, TemporalEvent, list[Triplet], list[Entity]]:
        """Process one statement to extract events, triplets and entities."""
        logger.info(f"Processing statement {statement.statement_type} | {statement.temporal_type} | {statement.statement} | for publication {publication.published_on}__{publication.title}...")

        # Step 1: extract temporal range from the statemenet
        raw_validity, temporal_range = await self.extract_temporal_range(statement, metadata=doc_summary)

        # Step 2: extract triplets from the statement
        raw_extraction: RawExtraction = await self.extract_triplet(statement, max_retries=3)

        # Step 3: Get embeddings for the statement text
        embedding = await self.get_statement_embedding(statement.statement)

        # Step 4: Create the temporal event from the range, triplets, and embeddings
        event = TemporalEvent(
            id = statement.id, # since each statement has one temporal event -- IDs can be re-used
            publication_id=publication.id,
            statement=statement.statement,
            embedding=embedding,
            triplets=[],
            valid_at=temporal_range.valid_at,
            invalid_at=temporal_range.invalid_at,
            temporal_type=statement.temporal_type,
            statement_type=statement.statement_type,
            temporal_confidence=statement.temporal_confidence,
            valid_at_confidence=temporal_range.valid_at_confidence,
            invalid_at_confidence=temporal_range.invalid_at_confidence,
        )

        # Map raw triplets/entities to Triplet/Entity with event_id
        triplets = [Triplet.from_raw(raw_triplet, event.id) for raw_triplet in raw_extraction.triplets] # aka Triplet[SUBJECT - PREDICATE - OBJECT]
        entities = [Entity.from_raw(raw_entity, event.id) for raw_entity in raw_extraction.entities] # aka Entity[NAME, TYPE, DESCRIPTION]
        # Add triplets to the event
        event.triplets = [triplet.id for triplet in triplets]
        logger.info(
            f"Created temporal event from statement {statement.statement_type} {statement.temporal_type} is processed. "
            f"Added {len(triplets)} triplets and {len(entities)} entities."
        )
        return raw_validity, event, triplets, entities

    async def _process_statements(self, publication:Publication, doc_summary:dict, statements_list:list[RawStatement]):
        """Process statements and extract events, triplets and entities."""
        temporal_ranges = []
        events: list[TemporalEvent] = []
        chunk_triplets: list[Triplet] = []
        chunk_entities: list[Entity] = []

        if len(statements_list) == 0:
            logger.info(f"No statements to proces in the publication: {publication.published_on}__{publication.title}")
            return events, chunk_triplets, chunk_entities

        # Step 2: process each statement to extract events and triplets from each of them
        logger.info(f"Processing {len(statements_list)} extracted statements in the publication: {publication.published_on}__{publication.title}")
        for i_stmt, stmt in enumerate(statements_list):
            logger.info(f"Processing {i_stmt}/{len(statements_list)} statement...")

            raw_temporal_range, event, triplets, entities = await self._process_statement(publication, stmt, doc_summary)

            temporal_ranges.append(raw_temporal_range)
            events.append(event)
            chunk_triplets.extend(triplets)
            chunk_entities.extend(entities)

        logger.info(f"Publication is processed. Extracted {len(events)} events, {len(chunk_triplets)} triplets and {len(chunk_entities)} entities.")
        return temporal_ranges, events, chunk_triplets, chunk_entities

    async def extract_publication_events(self, publication: Publication, limit_n_statements: int|None=None) -> \
            tuple[list[RawStatement], list[RawTemporalRange], list[TemporalEvent], list[Triplet], list[Entity]]:
        """Process a publication and extract, first, statements, then, temporal events, triplets, and entities."""
        publication_id = publication.id

        logger.info(
            f"Extracting publication events from publications with ID {publication.id} "
            f"published on {publication.published_on} titled {publication.title} "
            f"with text size {len(publication.text)}."
        )
        doc_summary = get_publication_metadata(publication)

        # Extract statements from the publication
        statements_list = await self.extract_statements(publication, doc_summary)

        # Select only subsample of statements for further processing if required
        if limit_n_statements is not None:
            logger.info(f"Limiting number of statements to {limit_n_statements} statements out of {len(statements_list.statements)} total statements found in the publication.")
            statements:list[RawStatement] = statements_list.statements[:limit_n_statements]
        else:
            statements:list[RawStatement] = statements_list.statements

        # Extract temporal events, triplets and entities from all statements
        temporal_ranges, events, triplets, entities = await self._process_statements(publication, doc_summary, statements)

        return statements, temporal_ranges, events, triplets, entities