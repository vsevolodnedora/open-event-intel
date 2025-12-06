import string

from rapidfuzz import fuzz

from src.logger import get_logger
from src.tkg.config import Config
from src.tkg.data_models import Entity
from src.tkg.tkg_database import TKGDatabase

logger = get_logger(__name__)

class EntityResolution:
    """
    Ensure that the entity has a single, authoritative representation, eliminating duplicates and maintaining data consistency.

    Uses RapidFuzz to cluster entities based on name similarity.

    This method involves a simple, case-insensitive, punctuation-free comparison using a partial match ratio, allowing tolerance for minor typos and substring matches.

    1. Within each fuzzy-matched cluster, we select the medoid-wthe entity most representative of the cluster based on overall similarity.
    2. The medoid then serves as the initial canonical entity, providing a semantically meaningful representation of the group.
    3. Before adding a new canonical entity, we cross-check the medoid against existing canonicals, considering both fuzzy matching and acronyms.
    4. If a global match isn't found, the medoid becomes a new canonical entity, with all entities in the cluster linked to it via a resolved ID.
    5. Finally, we perform an additional safeguard check to resolve potential acronym duplication across all canonical entities, ensuring thorough cleanup.

    Suggested improvements:
    - Using embedding-based similarity on Entity.description alongside Entity.name, improving disambiguation beyond simple text similarity.
    - Employing a large language model (LLM) to intelligently group entities under their canonical forms, enhancing accuracy through semantic understanding.
    """

    def __init__(self, config:Config, global_canonicals: list[Entity]):
        """Initialize the entity resolution class."""
        self.config = config
        self.global_canonicals = global_canonicals
        self.threshold = config.entity_resolution_threshold
        self.acronym_thresh = config.entity_resolution_acronym_thresh

    def match_to_canonical_entity(self, entity: Entity, canonical_entities: list[Entity]) -> Entity | None:
        """
        Fuzzy match a single entity to a list of canonical entities.

        Returns the best matching canonical entity or None if no match above self.threshold.
        """
        def clean(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        best_score: float = 0
        best_canon = None
        for canon in canonical_entities:
            score = fuzz.partial_ratio(clean(entity.name), clean(canon.name)) / 100.
            if score > best_score:
                best_score = score
                best_canon = canon
        if best_score >= self.threshold:
            logger.info(f"Located new best canonical entity {entity.name} with score of {best_score} (partial ratio)")
            return best_canon
        return None

    def group_entities_by_fuzzy_match(self, entities: list[Entity]) -> dict[str, list[Entity]]:
        """
        Group entities by fuzzy name similarity using rapidfuzz's partial_ratio.

        Returns a mapping from canonical name to list of grouped entities.
        """
        def clean_entity_name(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        name_to_entities: dict[str, list[Entity]] = {}
        cleaned_name_map: dict[str, str] = {}
        for entity in entities:
            name_to_entities.setdefault(entity.name, []).append(entity)
            cleaned_name_map[entity.name] = clean_entity_name(entity.name)
        unique_names = list(name_to_entities.keys())

        # Collect
        clustered: dict[str, list[Entity]] = {}
        used = set()
        for name in unique_names:
            if name in used:
                continue
            clustered[name] = []
            for other_name in unique_names:
                if other_name in used:
                    continue
                score = fuzz.partial_ratio(cleaned_name_map[name], cleaned_name_map[other_name]) / 100.
                if score >= self.threshold:
                    clustered[name].extend(name_to_entities[other_name])
                    used.add(other_name)

        logger.info(
            "Resolved %d entity clusters with unique names: %s",
            len(clustered),
            {unique_name: len(entities) for unique_name, entities in clustered.items()},
        )

        return clustered

    def set_medoid_as_canonical_entity(self, entities: list[Entity]) -> Entity | None:
        """
        Select as canonical the entity in the group with the highest total similarity (sum of partial_ratio) to all others.

        Returns the medoid entity or None if the group is empty.
        """
        if not entities:
            return None

        def clean(name: str) -> str:
            return name.lower().strip().translate(str.maketrans("", "", string.punctuation))

        n = len(entities)
        scores = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    s1 = clean(entities[i].name)
                    s2 = clean(entities[j].name)
                    scores[i] += (fuzz.partial_ratio(s1, s2) / 100.)
        max_idx = max(range(n), key=lambda idx: scores[idx])

        logger.info(f"Identified medoid entity idx={max_idx} in {len(entities)} entities. Highest total similarity: {scores[max_idx]} (partial_ratio)")

        return entities[max_idx]

    def _find_acronym_match(self, acronym: str) -> Entity | None:
        """Return the first single-word canonical whose name matches the acronym above threshold."""
        for c in self.global_canonicals:
            if " " in c.name:
                continue
            if fuzz.ratio(acronym, c.name) >= self.acronym_thresh:
                logger.info(f"New acronym '{acronym}' for local canonical entity matches existing acronym '{c.name}' with score {fuzz.ratio(acronym, c.name)/100.}.")
                return c
        return None

    def resolve_entities_batch(self, db:TKGDatabase, publication_entities: list[Entity]) -> None:
        """Orchestrate the scalable entity resolution workflow for a batch of entities."""
        if len(self.global_canonicals) == 0:
            logger.warning("No canonical entities found, Are global canonicals initialized?")

        type_groups = {t: [e for e in publication_entities if e.type == t] for t in set(e.type for e in publication_entities)}  # noqa: C401
        type_groups = dict(sorted(type_groups.items(), key=lambda item: item[0])) # Sort for determinism
        logger.info(f"Identified {len(type_groups)} typed entities groups: {list(type_groups.keys())}")

        for name, entities in type_groups.items():
            logger.info(f"Processing '{name}' group entities ({len(entities)})")

            # Cluster entities based on the semantic "fuzzy" match
            clusters = self.group_entities_by_fuzzy_match(entities=entities)
            if len(list(clusters.keys())) == 0:
                logger.warning(f"No entity clusters found in {len(entities)} entities.")

            # Get canonical entity for each group
            for group in clusters.values():
                if not group:
                    logger.warning("No group of entities found.")
                    continue

                # Select as canonical the entity in the group with the highest total similarity
                local_canon: Entity | None = self.set_medoid_as_canonical_entity(group)
                if local_canon is None:
                    logger.info(f"No canonical entity is found in the group {len(group)} entities. E.g., no medoid entity found. Continuing to the next group.")
                    continue

                # Find the best matching entity if any (e.g., > threshold)
                match: Entity | None = self.match_to_canonical_entity(entity=local_canon, canonical_entities=self.global_canonicals)

                if " " in local_canon.name:  # Multi-word entity
                    logger.info(f"Local canonical name {local_canon.name} is a multi-word canonical entity. Creating acronym.")
                    acronym = "".join(word[0] for word in local_canon.name.split())
                    acronym_match = self._find_acronym_match(acronym)
                    if acronym_match:
                        match = acronym_match

                # New canonical is found if not match
                if match:
                    canonical_id = match.id # TODO: suspected issue here as it is alwaus True after the first canonical entity is found. Suspected issue in IDs
                else:
                    logger.info(f"Inserting canonical id {local_canon.id} into the database (resolved_id = None)")
                    # Insert canonical entity (its ID will be placed as a resolved_id for all entities in the group)
                    new_entity = Entity(
                        id = local_canon.id,
                        event_id = local_canon.event_id,
                        name = local_canon.name,
                        type = local_canon.type,
                        description = local_canon.description,
                        resolved_id = None
                    )
                    db.insert_entity(entity=new_entity, event_id=local_canon.event_id, raw=False)

                    # Update canonical ID and save the Entity in the local storage (extend the original one from the database)
                    canonical_id = local_canon.id
                    self.global_canonicals.append(local_canon)

                for entity in group:
                    if entity.id == canonical_id:
                        logger.info("Entity cannot resolve to itself. Given entity_id = canonical_id = {}".format(entity.id))
                        continue
                    entity.resolved_id = canonical_id
                    db.update_entity_resolved_id(canonical_id=canonical_id, entity_id=entity.id, raw=False)

        # Clean up any acronym duplicates after processing all entities
        logger.info(f"Finished resolving {len(type_groups)} typed entities. Identified {len(self.global_canonicals)} global canonical entities.")
        self.merge_acronym_canonicals(db=db)


    def merge_acronym_canonicals(self, db:TKGDatabase) -> None:
        """Merge canonical entities where one is an acronym of another."""
        if len(self.global_canonicals) == 0:
            logger.warning("No global canonical entities found, Are global canonicals initialized?")

        multi_word = [e for e in self.global_canonicals if " " in e.name]
        single_word = [e for e in self.global_canonicals if " " not in e.name]

        acronym_map = {}
        for entity in multi_word:
            acronym = "".join(word[0].upper() for word in entity.name.split())
            acronym_map[entity.id] = acronym

        for entity in multi_word:
            acronym = acronym_map[entity.id]
            for single_entity in single_word:
                score = fuzz.ratio(acronym, single_entity.name) / 100.
                if score >= self.threshold:
                    logger.info(f"Removing {entity.id} with a score {score} as there is an entity with acronym: '{acronym}'")
                    # Update all references from old_id to new_id in the database.
                    db.update_entity_references_batch({entity.id: single_entity.id}, raw=False)
                    # Remove the original entity
                    db.remove_entity(entity_id=entity.id, raw=False)
                    self.global_canonicals.remove(entity)
                    break