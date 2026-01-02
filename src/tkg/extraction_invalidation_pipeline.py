import asyncio
import json
import os
import pickle
import sqlite3

from src.data_models import Publication
from src.logger import get_logger
from src.publications_database import PostsDatabase
from src.tkg.config import Config
from src.tkg.data_models import Entity, RawStatement, RawTemporalRange, StatementType, TemporalEvent, Triplet
from src.tkg.entity_resolution import EntityResolution
from src.tkg.extraction_agent import TemporalAgent
from src.tkg.invalidation_agent import InvalidationAgent, batch_process_invalidation
from src.tkg.prompt_registry import PromptRegistry
from src.tkg.tkg_database import TKGDatabase
from src.tkg.utils import create_file_name, ensure_tz

logger = get_logger(__name__)


async def process_publication(publication: Publication, prompt_registry:PromptRegistry, tkg_config: Config, memory:bool, refresh_database: bool, limit_n_statements: int|None=None):
    """Process one news publication."""
    # Initialize the TKG database
    tkg_database = TKGDatabase(db_path=tkg_config.tkg_db_fpath, memory=memory, refresh=refresh_database)

    # Initialize core components
    temporal_agent = TemporalAgent(config=tkg_config, prompt_registry=prompt_registry)
    invalidation_agent = InvalidationAgent(config=tkg_config, prompt_registry=prompt_registry)

    processed_publications: list[Publication] = tkg_database.get_all_publications()
    logger.info(f"Found {len(processed_publications)} already processed publications in the database.")

    # Ensure that the datetime TZ in publication is as required
    publication.published_on = ensure_tz(publication.published_on, name="published_on")

    # === STAGE 1 Extraction Agent ===

    # Check if events, triplets and entities have already been extracted from this publication
    if not tkg_database.has_publication(publication_id=publication.id):
        # Insert new publication into the database
        tkg_database.insert_publication(publication=publication, overwrite=True)

        # Extract first statements, then triplets and entities from all statements
        extraction_results = await temporal_agent.extract_publication_events(publication=publication, limit_n_statements=limit_n_statements)
        all_statements: list[RawStatement] = extraction_results[0]
        all_temporal_ranges: list[RawTemporalRange] = extraction_results[1]
        all_events: list[TemporalEvent] = extraction_results[2]
        all_triplets: list[Triplet] = extraction_results[3]
        all_entities: list[Entity] = extraction_results[4]
        if not len(all_statements) == len(all_events):
            raise RuntimeError(f"Number of statements {len(all_statements)} and number of events {len(all_events)} does not match.")

        # Process each statement with its associated data. Assuming 1:1 mapping between statements and events
        for statement, event in zip(all_statements, all_events):
            # Insert new statement for this publication
            tkg_database.insert_statement(statement=statement, publication_id=publication.id)

            # Get triplets that belong to this event
            event_triplet_ids = event.triplets if event.triplets else []
            if len(event_triplet_ids) == 0:
                logger.warning(f"No triplet IDs found for Event: {event.id} (Type: {event.temporal_type} - {event.statement_type})")

            # Insert triplets for this statement
            for triplet in all_triplets:
                if triplet.id in event_triplet_ids:
                    tkg_database.insert_triplet(triplet=triplet, event_id=event.id, raw=True)

            # Insert entities for this statement
            for entity in all_entities:
                if entity.event_id == event.id:
                    tkg_database.insert_entity(entity=entity, event_id=event.id, raw=True)

            # Insert event with statement_id and triplet_ids
            tkg_database.insert_event(event=event, raw=True)

        # Dump the publication extraction into a human-readable format
        complete_path = tkg_config.output_path_pub + f"{publication.publisher}" + "/"
        os.makedirs(complete_path, exist_ok=True)
        fpath = complete_path + create_file_name(publication) + "_raw.txt"
        tkg_database.dump_publication(fpath=fpath, publication_id=publication.id, raw=True)

        logger.info(f"Extraction completed. Saved RAW {len(all_events)} events and {len(all_triplets)} triplets and {len(all_entities)} entities for the publication")
    else:
        # Get statements for this publication (returns iterator of (uuid, RawStatement) tuples)
        all_statements = list(tkg_database.iter_statements_for_publication(publication_id=publication.id))

        # Retrieve events, triplets and entities from the database for further processing
        all_events = tkg_database.get_all_events(raw=True)
        all_triplets = tkg_database.get_all_triplets(raw=True)
        # Note: canonical_only=False to get all entities including resolved ones
        all_entities = tkg_database.get_all_entities(raw=True, canonical_only=False)

        logger.info(f"Loaded RAW {len(all_statements)} statements, {len(all_events)} events and {len(all_triplets)} triplets and {len(all_entities)} entities for the publication")

    # === STAGE 2 Entity Resolution

    # Removing canonicals first as they might have been added incorrectly and might cause errors
    tkg_database.clear_table(table_name="entities", raw=False)
    tkg_database.clear_table(table_name="triplets", raw=False)
    tkg_database.clear_table(table_name="events", raw=False)

    # Perform entity resolution
    entity_resolver = EntityResolution(config=tkg_config, global_canonicals=tkg_database.get_all_entities(raw=False, canonical_only=True))

    entity_resolver.resolve_entities_batch(db=tkg_database, publication_entities=all_entities)
    resolved_entities = tkg_database.get_all_entities(raw=False, canonical_only=False)  # get currently resolved entities
    logger.info(f"Resolved {len(resolved_entities)} entities out of {len(all_entities)} entities for the publication")

    # Create a dict with name:resolved_id for all entities based on the newly found
    name_to_canonical = {entity.name: entity.resolved_id for entity in all_entities if entity.resolved_id}
    logger.info(f"For {len(all_entities)} entities there are {len(name_to_canonical)} canonical names")

    # Update triplets with resolved entity IDs
    for triplet in all_triplets:
        if triplet.subject_name in name_to_canonical:
            triplet.subject_id = name_to_canonical[triplet.subject_name]
        if triplet.object_name in name_to_canonical:
            triplet.object_id = name_to_canonical[triplet.object_name]

    # Invalidation processing with properly resolved triplet IDs
    events_to_update: list[TemporalEvent] = []
    if tkg_database.has_events(raw=False, statement_type=StatementType.FACT):
        all_events, events_to_update = await batch_process_invalidation(tkg_database, all_events, all_triplets, invalidation_agent)

    # Update existing events first (they're already in DB)
    if events_to_update:
        tkg_database.update_events_batch(events_to_update)
        logger.info(f"Updated {len(events_to_update)} existing events")

    # Insert new data
    for event in all_events:
        tkg_database.insert_event(event, raw=False)

    for triplet in all_triplets:
        try:
            tkg_database.insert_triplet(triplet=triplet, event_id=triplet.event_id, raw=False)
        except KeyError as e:
            logger.error(f"KeyError: {triplet.subject_name} or {triplet.object_name} not found in name_to_canonical")
            logger.warning(f"Skipping triplet: Entity '{e.args[0]}' is unresolved.")
            continue

    # Deduplicate entities by id before insert
    unique_entities = {}
    for entity in all_entities:
        unique_entities[str(entity.id)] = entity
    for entity in unique_entities.values():
        tkg_database.insert_entity(entity=entity, event_id=entity.event_id, raw=False)

    complete_path = tkg_config.output_path_pub + f"{publication.publisher}" + "/"
    os.makedirs(complete_path, exist_ok=True)
    fpath = complete_path + create_file_name(publication) + ".txt"
    tkg_database.dump_publication(publication_id=publication.id, fpath=fpath, raw=False)


async def main_tkg_pipeline(config: Config, prompt_registry:PromptRegistry, publisher: str, limit_publications: int | None, limit_n_statements:int|None, refresh_database:bool=False):
    """Execute the main tkg pipeline."""
    # Initialize source database
    publication_database = PostsDatabase(config.preprocessed_db_fpath)
    publications: list[Publication] = publication_database.list_publications(table_name=publisher, sort_date=True)
    logger.info(f"Found {len(publications)} publications in the database for the publisher: {publisher}")

    # Limit if needed
    if limit_publications:
        logger.info(f"Limiting publications to {limit_publications} publications from a total of {len(publications)} available publications")
        publications = publications[:limit_publications]

    # Process each publication (store raw extraction without validation into "raw" tables in the database)
    for publication in publications:
        await process_publication(publication=publication, prompt_registry=prompt_registry, tkg_config=config, memory=False, refresh_database=refresh_database, limit_n_statements=limit_n_statements)

    # Save/update the database overview and metadata
    tkg_database = TKGDatabase(db_path=config.tkg_db_fpath, memory=False, refresh=refresh_database)

    # Save database content in .json in a human-readable format
    complete_path = config.output_path_pub + "/" + "db_summary_raw.csv"
    tkg_database.export_to_csv(output_path=complete_path, raw=True)

    # Save the database (raw) public view as a .json
    complete_path = config.public_view_path + "/" + "tkg_raw_metadata.json"
    tkg_database.export_public_view_to_json(output_path=complete_path, raw=True)

    # Save the database public view as a .json
    complete_path = config.public_view_path + "/" + "tkg_metadata.json"
    tkg_database.export_public_view_to_json(output_path=complete_path, raw=False)

# if __name__ == "__main__":
#     asyncio.run(main_tkg_pipeline())