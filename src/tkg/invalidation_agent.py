import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from jinja2 import Environment
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine
from tenacity import retry, stop_after_attempt, wait_random_exponential

from open_event_intel.logger import get_logger
from src.tkg.config import Config
from src.tkg.data_models import StatementType, TemporalConfidence, TemporalEvent, TemporalType, Triplet
from src.tkg.prompt_registry import PromptRegistry
from src.tkg.tkg_database import TKGDatabase

logger = get_logger(__name__)

class InvalidationAgent:
    """Handles temporal-based operations for extracting and processing temporal events from text."""

    def __init__(self, config: Config, prompt_registry:PromptRegistry) -> None:
        """Initialize the TemporalAgent with a client."""
        self.max_workers = config.invalidation_agent_num_workers
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.logger = logging.getLogger(__name__)
        self._client = AsyncOpenAI()
        self._model = config.invalidation_agent_model
        self._similarity_threshold = config.invalidation_agent_similarity_threshold
        self._top_k = config.invalidation_agent_top_k

        self._prompt_registry = prompt_registry
        self._env = self._initialize_jinja_environment()

    def _initialize_jinja_environment(self)->Environment:
        return self._prompt_registry.create_environment()

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(1 - cosine(v1, v2))

    @staticmethod
    def get_incoming_temporal_bounds(event: TemporalEvent) -> dict[str, datetime] | None:
        """Get temporal bounds of all temporal events associated with a statement."""
        if (event.temporal_type == TemporalType.ATEMPORAL) or (event.valid_at is None):
            return None

        temporal_bounds = {"start": event.valid_at, "end": event.valid_at}

        if event.temporal_type == TemporalType.STATE:
            if event.invalid_at:
                temporal_bounds["end"] = event.invalid_at

        return temporal_bounds

    @staticmethod
    def _check_overlaps_dynamic(event: TemporalEvent, start: datetime, end: datetime) -> bool:
        """Check if the dynamic event overlaps with the temporal bounds of the incoming event."""
        if event.temporal_type != TemporalType.STATE:
            return False

        event_start = event.valid_at or datetime.min
        event_end = event.invalid_at

        # 1. Event contains the start
        if (event_end is not None) and (event_start <= start <= event_end):
            return True

        # 2. Ongoing event starts before the incoming start
        if (event_end is None) and (event_start <= start):
            return True

        # 3. Event starts within the incoming interval
        if start <= event_start <= end:
            return True
        return False

    def select_events_temporally(self,
        triplet_events: list[tuple[Triplet, TemporalEvent]],
        temp_bounds: dict[str, datetime],
        state: bool = False,
    ) -> list[tuple[Triplet, TemporalEvent]]:
        """
        Select temporally relevant events (static or dynamic) based on temporal bounds.

        Groups events into before, after, and overlapping categories based on their temporal bounds.

        :param triplet_events: List of (Triplet, TemporalEvent) tuples to filter
        :param temp_bounds: Dict with 'start' and 'end' datetime bounds
        :param state: If True, filter "STATE" events; if False, filter "EVENT" events
        :param n_window: Number of events to include before and after bounds

        :returns:
            Dict with keys '{type}_before', '{type}_after', '{type}_overlap' where type is 'dynamic' or 'static'
        """
        # Filter by temporal type
        target_type = TemporalType.STATE if state else TemporalType.EVENT
        filtered_events = [(triplet, event) for triplet, event in triplet_events if event.temporal_type == target_type]

        # Sort by valid_at timestamp
        sorted_events = sorted(filtered_events, key=lambda te: te[1].valid_at or datetime.min)

        start = temp_bounds["start"]
        end = temp_bounds["end"]

        if state:
            overlap: list[tuple[Triplet, TemporalEvent]] = [
                (triplet, event) for triplet, event in sorted_events if InvalidationAgent._check_overlaps_dynamic(event, start, end)
            ]
        else:
            overlap = []
            if start != end:
                overlap = [(triplet, event) for triplet, event in sorted_events if event.valid_at and start <= event.valid_at <= end]

        return overlap

    def filter_by_embedding_similarity(
        self,
        reference_event: TemporalEvent,
        candidate_pairs: list[tuple[Triplet, TemporalEvent]],
    ) -> list[tuple[Triplet, TemporalEvent]]:
        """Filter triplet-event pairs by embedding similarity."""
        pairs_with_similarity = [
            (triplet, event, self.cosine_similarity(reference_event.embedding, event.embedding)) for triplet, event in candidate_pairs
        ]

        filtered_pairs = [
            (triplet, event) for triplet, event, similarity in pairs_with_similarity if similarity >= self._similarity_threshold
        ]

        sorted_pairs = sorted(filtered_pairs, key=lambda x: self.cosine_similarity(reference_event.embedding, x[1].embedding), reverse=True)

        return sorted_pairs[: self._top_k]

    def select_temporally_relevant_events_for_invalidation(
        self,
        incoming_event: TemporalEvent,
        candidate_triplet_events: list[tuple[Triplet, TemporalEvent]],
    ) -> list[tuple[Triplet, TemporalEvent]] | None:
        """Select the temporally relevant events based on temporal range of incoming event."""
        temporal_bounds = self.get_incoming_temporal_bounds(event=incoming_event)
        if not temporal_bounds:
            return None

        # First apply temporal filtering - find overlapping events
        selected_statics = self.select_events_temporally(
            triplet_events=candidate_triplet_events,
            temp_bounds=temporal_bounds,
        )
        selected_dynamics = self.select_events_temporally(
            triplet_events=candidate_triplet_events,
            temp_bounds=temporal_bounds,
            state=True,
        )

        # Then filter by semantic similarity
        similar_static = self.filter_by_embedding_similarity(reference_event=incoming_event, candidate_pairs=selected_statics)

        similar_dynamics = self.filter_by_embedding_similarity(reference_event=incoming_event, candidate_pairs=selected_dynamics)

        return similar_static + similar_dynamics

    def should_consider_for_invalidation(
        self,
        primary_event: TemporalEvent,
        secondary_event: TemporalEvent,
    ) -> bool:
        """
        Fast pre-filter before LLM call to avoid obvious non-invalidations.

        Returns False if invalidation is impossible based on confidence + temporal logic.
        """
        # Rule: Both LOW confidence + same timestamp = likely same publication, skip
        if primary_event.valid_at_confidence == TemporalConfidence.LOW and secondary_event.valid_at_confidence == TemporalConfidence.LOW and primary_event.valid_at == secondary_event.valid_at:
            logger.info("Skipping invalidation: Both LOW confidence + same timestamp = likely same publication")
            return False

        # Rule: Secondary event has LOW confidence and occurs much later than HIGH confidence primary
        # (probably just publication date, unlikely to contain actual invalidating info)
        if (
            primary_event.valid_at_confidence == TemporalConfidence.HIGH
            and secondary_event.valid_at_confidence == TemporalConfidence.LOW
            and secondary_event.valid_at
            and primary_event.valid_at
            and (secondary_event.valid_at - primary_event.valid_at).days > 30
        ):
            logger.info("Skipping invalidation: Secondary event has LOW confidence and occurs much later than HIGH confidence primary")
            return False

        return True

    @retry(wait=wait_random_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
    async def invalidation_step_llm_call(
        self,
        primary_event: TemporalEvent,
        primary_triplet: Triplet,
        secondary_event: TemporalEvent,
        secondary_triplet: Triplet,
    ) -> TemporalEvent:
        """Check if primary event should be invalidated by secondary event."""
        template = self._env.get_template("event_invalidation_prompt.jinja")

        prompt = template.render(
            primary_event=primary_event.statement,
            primary_triplet=f"({primary_triplet.subject_name}, {primary_triplet.predicate}, {primary_triplet.object_name})",
            primary_valid_at=primary_event.valid_at,
            primary_invalid_at=primary_event.invalid_at,
            primary_valid_at_confidence=primary_event.valid_at_confidence.value,
            primary_invalid_at_confidence=primary_event.invalid_at_confidence.value,
            secondary_event=secondary_event.statement,
            secondary_triplet=f"({secondary_triplet.subject_name}, {secondary_triplet.predicate}, {secondary_triplet.object_name})",
            secondary_valid_at=secondary_event.valid_at,
            secondary_invalid_at=secondary_event.invalid_at,
            secondary_valid_at_confidence=secondary_event.valid_at_confidence.value,
            secondary_invalid_at_confidence=secondary_event.invalid_at_confidence.value,
        )

        response = await self._client.responses.parse(
            model=self._model,
            temperature=0,
            input=prompt,
        )

        response_bool = str(response).strip().lower() == "true" if response else False

        if not response_bool:
            return primary_event

        updated_event = primary_event.model_copy(
            update={
                "invalid_at": secondary_event.valid_at,
                "expired_at": datetime.now(),
                "invalidated_by": secondary_event.id,
            }
        )
        return updated_event

    async def bi_directional_event_invalidation(
        self,
        incoming_triplet: Triplet,
        incoming_event: TemporalEvent,
        existing_triplet_events: list[tuple[Triplet, TemporalEvent]],
    ) -> tuple[TemporalEvent, list[TemporalEvent]]:
        """
        Validate and update temporal information for triplet events with full bidirectional invalidation.

        :param incoming_triplet: The new triplet
        :param incoming_event: The new event associated with the triplet
        :param existing_triplet_events: List of existing (triplet, event) pairs to validate against

        :returns: tuple[TemporalEvent, list[TemporalEvent]]: (updated_incoming_event, list_of_changed_existing_events)
        """
        changed_existing_events: list[TemporalEvent] = []
        updated_incoming_event = incoming_event

        # Filter for dynamic events that can be invalidated
        dynamic_events_to_check = [
            (triplet, event)
            for triplet, event in existing_triplet_events
            if event.temporal_type == TemporalType.STATE and self.should_consider_for_invalidation(event, incoming_event)
        ]

        # 1. Check if incoming event invalidates existing dynamic events
        if dynamic_events_to_check:
            tasks = [
                self.invalidation_step_llm_call(
                    primary_event=existing_event,
                    primary_triplet=existing_triplet,
                    secondary_event=incoming_event,
                    secondary_triplet=incoming_triplet,
                )
                for existing_triplet, existing_event in dynamic_events_to_check
            ]

            updated_events = await asyncio.gather(*tasks)

            for original_pair, updated_event in zip(dynamic_events_to_check, updated_events, strict=True):
                original_event = original_pair[1]
                if (updated_event.invalid_at != original_event.invalid_at) or (
                    updated_event.invalidated_by != original_event.invalidated_by
                ):
                    changed_existing_events.append(updated_event)

        # 2. Check if existing events invalidate the incoming "state" (dynamic) event
        if incoming_event.temporal_type == TemporalType.STATE and incoming_event.invalid_at is None:
            # Only check events that occur after the incoming event
            invalidating_events = [
                (triplet, event)
                for triplet, event in existing_triplet_events
                if (incoming_event.valid_at and event.valid_at and incoming_event.valid_at < event.valid_at)
            ]

            if invalidating_events:
                tasks = [
                    self.invalidation_step_llm_call(
                        primary_event=incoming_event,
                        primary_triplet=incoming_triplet,
                        secondary_event=existing_event,
                        secondary_triplet=existing_triplet,
                    )
                    for existing_triplet, existing_event in invalidating_events
                ]

                updated_events = await asyncio.gather(*tasks)

                # Find the earliest invalidation
                valid_invalidations = [(e.invalid_at, e.invalidated_by) for e in updated_events if e.invalid_at is not None]

                if valid_invalidations:
                    earliest_invalidation = min(valid_invalidations, key=lambda x: x[0])
                    updated_incoming_event = incoming_event.model_copy(
                        update={
                            "invalid_at": earliest_invalidation[0],
                            "invalidated_by": earliest_invalidation[1],
                            "expired_at": datetime.now(),
                        }
                    )

        return updated_incoming_event, changed_existing_events

    @staticmethod
    def resolve_duplicate_invalidations(changed_events: list[TemporalEvent]) -> list[TemporalEvent]:
        """
        Resolve duplicate invalidations by selecting the most restrictive (earliest) invalidation.

        When multiple incoming events invalidate the same existing event, we should apply
        the invalidation that results in the shortest validity range (earliest invalid_at).

        :param changed_events: List of events that may contain duplicates with different invalidations

        :returns: List of deduplicated events with the most restrictive invalidation applied
        """
        if not changed_events:
            return []

        # Count occurrences of each event ID
        id_counts = Counter(str(event.id) for event in changed_events)
        resolved_events = []
        # Group events by ID only for those with duplicates
        events_by_id = defaultdict(list)
        for event in changed_events:
            event_id = str(event.id)
            if id_counts[event_id] == 1:
                resolved_events.append(event)
            else:
                events_by_id[event_id].append(event)

        # Deduplicate only those with duplicates
        for _id, event_versions in events_by_id.items():
            invalidated_versions = [e for e in event_versions if e.invalid_at is not None]
            if not invalidated_versions:
                resolved_events.append(event_versions[0])
            else:
                most_restrictive = min(invalidated_versions, key=lambda e: (e.invalid_at if e.invalid_at is not None else datetime.max))
                resolved_events.append(most_restrictive)

        return resolved_events

    async def _execute_task_pool(
        self,
        tasks: list[Coroutine[Any, Any, tuple[TemporalEvent, list[TemporalEvent]]]],
        batch_size: int = 10
    ) -> list[Any]:
        """
        Execute tasks in batches using a pool to control concurrency.

        :param tasks: List of coroutines to execute
        :param batch_size: Number of tasks to process concurrently

        :returns: List of results from all tasks
        """
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            all_results.extend(batch_results)

            # Small delay between batches to prevent overload
            if i + batch_size < len(tasks):
                await asyncio.sleep(.5)

        return all_results

    async def process_invalidations_in_parallel(
        self,
        incoming_triplets: list[Triplet],
        incoming_events: list[TemporalEvent],
        existing_triplets: list[Triplet],
        existing_events: list[TemporalEvent],
    ) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
        """
        Process invalidations for multiple triplets in parallel.

        :param incoming_triplets: List of new triplets to process
        :param incoming_events: List of events associated with incoming triplets
        :param existing_triplets: List of existing triplets from DB
        :param existing_events: List of existing events from DB

        :returns tuple[list[TemporalEvent], list[TemporalEvent]]:
            - List of updated incoming events (potentially invalidated)
            - List of existing events that were updated (deduplicated)
        """
        logger.info(f"Starting invalidations for {len(incoming_triplets)} incoming triplets and {len(incoming_events)} incoming events "
                    f"with {len(existing_triplets)} existing triplets and {len(existing_events)} existing events")

        # Create mappings for faster lookups
        event_map = {str(e.id): e for e in existing_events}
        incoming_event_map = {str(t.event_id): e for t, e in zip(incoming_triplets, incoming_events, strict=False)}

        # Prepare tasks for parallel processing
        tasks = []
        for incoming_triplet in incoming_triplets:
            incoming_event = incoming_event_map[str(incoming_triplet.event_id)]

            # Get related triplet-event pairs
            related_pairs = [
                (t, event_map[str(t.event_id)])
                for t in existing_triplets
                if (str(t.subject_id) == str(incoming_triplet.subject_id) or str(t.object_id) == str(incoming_triplet.object_id))
                and str(t.event_id) in event_map
            ]

            # Filter for temporal relevance
            all_relevant_events = self.select_temporally_relevant_events_for_invalidation(
                incoming_event=incoming_event,
                candidate_triplet_events=related_pairs,
            )

            if not all_relevant_events:
                continue

            # Add task for parallel processing
            task = self.bi_directional_event_invalidation(
                incoming_triplet=incoming_triplet,
                incoming_event=incoming_event,
                existing_triplet_events=all_relevant_events,
            )
            tasks.append(task)

        # Process all invalidations in parallel with pooling
        if not tasks:
            return [], []

        # Use pool size based on number of workers, but cap it
        pool_size = min(self.max_workers * 2, 10)  # Adjust these numbers based on your needs
        results = await self._execute_task_pool(tasks, batch_size=pool_size)

        # Collect all results (may contain duplicates)
        updated_incoming_events = []
        all_changed_existing_events = []

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with error: {str(result)}")
                continue
            updated_event, changed_events = result
            updated_incoming_events.append(updated_event)
            all_changed_existing_events.extend(changed_events)

        # Resolve duplicate invalidations for existing events
        deduplicated_existing_events = self.resolve_duplicate_invalidations(all_changed_existing_events)

        # Resolve duplicate invalidations for incoming events (in case multiple triplets from same event)
        deduplicated_incoming_events = self.resolve_duplicate_invalidations(updated_incoming_events)

        return deduplicated_incoming_events, deduplicated_existing_events

def _get_fact_triplets(
    all_events: list[TemporalEvent],
    all_triplets: list[Triplet],
) -> list[Triplet]:
    """Return only those triplets whose associated event is of statement_type FACT."""
    fact_event_ids = {event.id for event in all_events if (event.statement_type == StatementType.FACT) and (event.temporal_type != TemporalType.ATEMPORAL)}
    return [triplet for triplet in all_triplets if triplet.event_id in fact_event_ids]

async def batch_process_invalidation(
    tkg_db: TKGDatabase,
    all_events: list[TemporalEvent],
    all_triplets: list[Triplet],
    invalidation_agent: InvalidationAgent,
    prompt_registry: PromptRegistry,
) -> tuple[list[TemporalEvent], list[TemporalEvent]]:
    """
    Process invalidation for all FACT events that are temporal.

    :param tkg_db: Database
    :param all_events: List of all extracted events
    :param all_triplets: List of all extracted triplets
    :param invalidation_agent: The invalidation agent instance

    :returns:
        tuple[list[TemporalEvent], list[TemporalEvent]]:
            - final_events: All events (updated incoming events)
            - events_to_update: Existing events that need DB updates
    """
    # Prepare a list of triplets whose associated event is a FACT and not ATEMPORAL
    fact_triplets = _get_fact_triplets(all_events, all_triplets)
    if not fact_triplets:
        logger.warning("No fact triplets found among incoming events.")
        return all_events, []

    # Create event map for quick lookup
    all_events_map = {event.id: event for event in all_events}

    # Build aligned lists of valid triplets and their corresponding events
    fact_events: list[TemporalEvent] = []
    valid_fact_triplets: list[Triplet] = []
    for triplet in fact_triplets:
        # Handle potential None event_id and ensure type safety
        if triplet.event_id is not None:
            event = all_events_map.get(triplet.event_id)
            if event:
                fact_events.append(event)
                valid_fact_triplets.append(triplet)
            else:
                logger.warning(f"Could not find event for fact_triplet with event_id {triplet.event_id}")
        else:
            logger.warning(f"Fact triplet {triplet.id} has no event_id, skipping invalidation")

    if not valid_fact_triplets:
        logger.warning("No valid fact triplets (e.g., ID != None) found among incoming events.")
        return all_events, []
    logger.info(f"Found {len(fact_events)} fact_events")

    # Batch fetch all related existing triplets and events
    predicate_groups = prompt_registry.load_predicate_groups(legacy_format=True)
    existing_triplets, existing_events = tkg_db.batch_fetch_related_triplet_events(
        incoming_triplets=valid_fact_triplets, predicate_groups=predicate_groups, raw=False
    )
    logger.info(f"Found {len(existing_triplets)} existing triplets and {len(existing_events)} existing events for {len(valid_fact_triplets)} valid fact_triplets")

    # Process all invalidations in parallel
    updated_incoming_fact_events, changed_existing_events = await invalidation_agent.process_invalidations_in_parallel(
        incoming_triplets=valid_fact_triplets,
        incoming_events=fact_events,
        existing_triplets=existing_triplets,
        existing_events=existing_events,
    )

    # Create mapping for efficient updates
    updated_incoming_event_map = {event.id: event for event in updated_incoming_fact_events}

    # Reconstruct final events list with updates applied
    final_events = []
    for original_event in all_events:
        if original_event.id in updated_incoming_event_map:
            final_events.append(updated_incoming_event_map[original_event.id])
        else:
            final_events.append(original_event)

    return final_events, changed_existing_events