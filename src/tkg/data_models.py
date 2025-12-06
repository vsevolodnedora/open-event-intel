"""Contains pydantic data models."""
import json
import re
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from dateutil.parser import parse
from pydantic import BaseModel, Field, field_validator, model_validator

from src.logger import get_logger

logger = get_logger(__name__)

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

# ================ Foundation of TemporalAgent ====================

class TemporalConfidence(StrEnum):
    """
    Confidence level in the temporal validity timestamps.

    CRITICAL FOR NEWS PUBLICATIONS: Distinguishes between explicitly stated
    timestamps vs. fallback publication dates.

    - HIGH: Explicitly stated in text (e.g., "on January 15, 2023", "in Q2 2024")
    - MEDIUM: Inferred from context (e.g., "recently", "this month", "last week")
    - LOW: Defaulted to publication date when temporal info is unclear or absent

    This enables the InvalidationAgent to:
    1. Apply appropriate temporal windows (wider for LOW confidence)
    2. Require higher semantic similarity before invalidating LOW confidence events
    3. Avoid false invalidations between events published on the same day
    """

    HIGH = "HIGH"  # Precise timestamp from text
    MEDIUM = "MEDIUM"  # Contextually inferred timing
    LOW = "LOW"  # Publication date fallback


class StatementType(StrEnum):
    """
    Enumeration of statement types for statements.

    Classifies the nature of each extracted statement, capturing its epistemic characteristics.
    """

    FACT = "FACT" # A statement that asserts a verifiable claim considered true at the time it was made. However, it may later be superseded or contradicted by other facts (e.g., updated information or corrections).
    OPINION = "OPINION" # A subjective statement reflecting a belief, sentiment, or judgment. By nature, opinions are considered temporally true at the moment they are expressed.
    PREDICTION = "PREDICTION" # OR FORECAST: A forward-looking or hypothetical statement about a potential future event or outcome. Temporally, a prediction is assumed to hold true from the time of utterance until the conclusion of the inferred prediction window.


class TemporalType(StrEnum):
    """Refined temporal classification."""

    ATEMPORAL = "ATEMPORAL"  # Universal truths

    # Completed events - the FACT that they occurred is permanent
    EVENT = "EVENT"  # e.g.: STATIC for past-tense predicates
    # Examples: "ACER adopted methodology", "Pipeline commissioned"

    # Ongoing or repeatable states/processes
    STATE = "STATE"  # e.g.: DYNAMIC
    # Examples: "TenneT operates", "Germany imports gas"

    # Forward-looking (e.g. for predictions)
    FORECAST = "FORECAST"  # Optional: separate from PREDICTION statement_type


class RawStatement(BaseModel):
    """
    Data model representing a raw statement with type and temporal information.

    Individual statement extracted by an LLM, annotated with both its semantic type (StatementType) and temporal classification (TemporalType).
    Raw statements serve as intermediate representations and are intended to be transformed into TemporalEvent objects in later processing stages.
    """

    id: uuid.UUID # ID of the statement generated
    statement: str  # The textual content of the extracted statement
    temporal_type: TemporalType  # The temporal classification of the statement (Static, Dynamic, Atemporal), drawn from the TemporalType enum
    statement_type: StatementType # The type of statement (Fact, Opinion, Prediction), based on the StatementType enum
    publication_id: str  # ID of the publication from which this statement was extracted
    temporal_confidence: TemporalConfidence # How confident is temporal type extraction for the statement

    @field_validator("temporal_type", mode="before")
    @classmethod
    def _parse_temporal_label(cls, value: str | None) -> TemporalType:
        if value is None:
            return TemporalType.ATEMPORAL
        cleaned_value = value.strip().upper()
        try:
            return TemporalType(cleaned_value)
        except ValueError as e:
            raise ValueError(f"Invalid temporal type: {value}. Must be one of {[t.value for t in TemporalType]}") from e

    @field_validator("statement_type", mode="before")
    @classmethod
    def _parse_statement_label(cls, value: str | None = None) -> StatementType:
        if value is None:
            return StatementType.FACT
        cleaned_value = value.strip().upper()
        try:
            return StatementType(cleaned_value)
        except ValueError as e:
            raise ValueError(f"Invalid statement type: {value}. Must be one of {[t.value for t in StatementType]}") from e


    @field_validator("temporal_confidence", mode="before")
    @classmethod
    def _parse_temporal_confidence(
        cls,
        value: str | TemporalConfidence | None = None,
    ) -> TemporalConfidence:
        """
        Normalize temporal confidence labels.

        Defaults to MEDIUM when not provided, treating it as a neutral confidence level.
        Adjust default (e.g. to LOW) if you prefer a more conservative behavior.
        """
        if value is None:
            return TemporalConfidence.MEDIUM
        if isinstance(value, TemporalConfidence):
            return value
        cleaned_value = value.strip().upper()
        try:
            return TemporalConfidence(cleaned_value)
        except ValueError as e:
            allowed = [t.value for t in TemporalConfidence]
            raise ValueError(
                f"Invalid temporal confidence: {value}. Must be one of {allowed}"
            ) from e

class RawStatementList(BaseModel):
    """Data model representing a list of raw statements."""

    statements: list[RawStatement]

# ==================== Temporal Range ===================

class RawTemporalRange(BaseModel):
    """Data model representing the raw temporal validity range as strings. Represents the originally extracted data."""

    valid_at: str | None = Field(..., json_schema_extra={"format": "date-time"})
    invalid_at: str | None = Field(..., json_schema_extra={"format": "date-time"})

    valid_at_confidence: TemporalConfidence = TemporalConfidence.LOW
    invalid_at_confidence: TemporalConfidence = TemporalConfidence.LOW
    rationale: str = ""

    @field_validator("valid_at", "invalid_at", mode="before")
    @classmethod
    def _normalize_temporal_bound(cls, value: str | None) -> str | None:
        """
        Accept None or a string for datetime-like values coming from the LLM.

        - None -> None
        - "" (empty/whitespace-only) -> None
        - Non-empty strings are validated as ISO 8601 datetimes (with basic support for 'Z').
        """
        if value is None:
            return None

        if isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value == "":
                # Treat empty string as "no bound"
                return None

            # Allow 'Z' suffix by normalizing to +00:00 for validation.
            normalized = cleaned_value.replace("Z", "+00:00")
            try:
                # Validation only – we keep the original canonical string.
                datetime.fromisoformat(normalized)
            except ValueError as e:
                raise ValueError(
                    f"Invalid datetime value: {value!r}. "
                    "Expected an ISO 8601 datetime string, e.g. '2023-10-24T00:00:00Z'."
                ) from e

            return cleaned_value

        raise TypeError(
            f"Invalid type for datetime field: {type(value).__name__}. "
            "Expected str or None."
        )

    @field_validator("valid_at_confidence", "invalid_at_confidence", mode="before")
    @classmethod
    def _parse_confidence_label(
        cls, value: TemporalConfidence | str | None
    ) -> TemporalConfidence:
        """
        Normalize TemporalConfidence labels coming from the LLM.

        - None -> TemporalConfidence.LOW (safe default)
        - TemporalConfidence instance → returned as-is
        - String -> stripped, uppercased, and converted to the enum
        """
        if value is None:
            return TemporalConfidence.LOW

        if isinstance(value, TemporalConfidence):
            return value

        if isinstance(value, str):
            cleaned_value = value.strip().upper()
            try:
                return TemporalConfidence(cleaned_value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid temporal confidence: {value}. "
                    f"Must be one of {[c.value for c in TemporalConfidence]}"
                ) from e

        raise TypeError(
            f"Invalid type for TemporalConfidence: {type(value).__name__}. "
            "Expected TemporalConfidence, str, or None."
        )

class TemporalValidityRange(BaseModel):
    """Data model representing the parsed temporal validity range as datetimes."""

    valid_at: datetime | None = None
    invalid_at: datetime | None = None

    valid_at_confidence: TemporalConfidence = TemporalConfidence.LOW
    invalid_at_confidence: TemporalConfidence = TemporalConfidence.LOW
    temporal_extraction_rationale: str = ""  # Why these dates were chosen

    @field_validator("valid_at", "invalid_at", mode="before")
    @classmethod
    def _parse_date_string(cls, value: str | datetime | None) -> datetime | None:
        if isinstance(value, datetime) or value is None:
            return value
        return parse_date_str(value)


class Predicate(StrEnum):
    """Canonical predicates for European energy & regulation news. Oriented as SUBJECT → PREDICATE → OBJECT."""

    # --- Structure / actors
    IS_A = "IS_A"                  # ENTSO-E IS_A association
    LOCATED_IN = "LOCATED_IN"      # Nemo Link LOCATED_IN Belgium
    HOLDS_ROLE = "HOLDS_ROLE"      # Person HOLDS_ROLE Chair at ACER
    OPERATES = "OPERATES"          # TSO OPERATES Interconnector

    # --- Regulatory lifecycle & procedures
    PUBLISHES = "PUBLISHES"                    # ENTSO-E PUBLISHES ERAA
    OPENS_CONSULTATION_ON = "OPENS_CONSULTATION_ON"  # ACER OPENS_CONSULTATION_ON Balancing proposal
    SUBMITS_TO = "SUBMITS_TO"                  # TSOs SUBMITS_TO NRA
    APPROVES = "APPROVES"                      # NRA APPROVES Methodology
    ADOPTS = "ADOPTS"                          # EC ADOPTS Delegated Act
    AMENDS = "AMENDS"                          # ACER AMENDS Methodology
    ENTERS_INTO_FORCE = "ENTERS_INTO_FORCE"    # Regulation ENTERS_INTO_FORCE Date

    # --- System & infrastructure operations
    COMMISSIONS = "COMMISSIONS"                # TSO COMMISSIONS Interconnector
    DECOMMISSIONS = "DECOMMISSIONS"            # Operator DECOMMISSIONS Unit
    CURTAILS = "CURTAILS"                      # TSO CURTAILS Wind output
    EXPERIENCES_OUTAGE = "EXPERIENCES_OUTAGE"  # Plant EXPERIENCES_OUTAGE Unit 3

    # --- Markets & auctions
    RUNS_AUCTION_FOR = "RUNS_AUCTION_FOR"      # TSO RUNS_AUCTION_FOR aFRR capacity
    AWARDS_CONTRACT_TO = "AWARDS_CONTRACT_TO"  # Market Operator AWARDS_CONTRACT_TO Provider
    CLEARS_AT = "CLEARS_AT"                    # Auction CLEARS_AT <value>  (price/volume in `value`)

    # --- Metrics, planning & finance
    HAS_CAPACITY = "HAS_CAPACITY"              # Asset HAS_CAPACITY 700 MW
    FORECASTS = "FORECASTS"                    # ENTSO-E FORECASTS Winter peak demand
    SECURES = "SECURES"                        # Developer SECURES Permit/Funding


# =================== Temporal Events ==========

class RawTriplet(BaseModel):
    """
    Data model representing a subject-predicate-object triplet.

    Basic subject-predicate-object relationship that is extracted directly from textual data.
    Serves as a precursor for the more detailed triplet representation in Triplet introduced below.
    """

    subject_name: str # The textual representation of the subject entity
    subject_id: int # Numeric identifier for the subject entity
    predicate: Predicate # The relationship type, specified by the Predicate enum
    object_name: str # The textual representation of the object entity
    object_id: int # Numeric identifier for the object entity
    value: str | None = None # Numeric value associated to relationship, may be None e.g. Germany -> HAS_A -> Generation of='60 GW'


class Triplet(BaseModel):
    """
    Data model representing a subject-predicate-object triplet.

    Extends raw triplet, incorporating unique identifiers and optionally linking each triplet to a specific event.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4) # Unique ID of this triplet
    event_id: uuid.UUID | None = None # Same as statement.id
    subject_name: str
    subject_id: uuid.UUID
    predicate: Predicate
    object_name: str
    object_id: uuid.UUID
    value: str | None = None

    @classmethod
    def from_raw(cls, raw_triplet: RawTriplet, event_id: uuid.UUID | None = None) -> "Triplet":
        """Create a Triplet instance from a RawTriplet, optionally associating it with an event_id."""
        return cls(
            id=uuid.uuid4(),
            event_id=event_id,
            subject_name=raw_triplet.subject_name,
            subject_id=uuid.UUID(int=raw_triplet.subject_id),
            predicate=raw_triplet.predicate,
            object_name=raw_triplet.object_name,
            object_id=uuid.UUID(int=raw_triplet.object_id),
            value=raw_triplet.value,
        )


class RawEntity(BaseModel):
    """
    Data model representing a raw entity (for entity resolution) extracted from Statement.

    Precursor for the more detailed triplet representation in Entity (defined below).
    """

    entity_idx: int # An integer to differentiate extracted entities from the statement (links to RawTriplet)
    name: str # The name of the entity extracted e.g. E.ON
    type: str = "" # The type of entity extracted e.g. Company
    description: str = "" # The textual description of the entity


class Entity(BaseModel):
    """
    Data model representing an entity (for entity resolution).

    'id' is the canonical entity id if this is a canonical entity.
    'resolved_id' is set to the canonical id if this is an alias.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_id: uuid.UUID | None = None # Same as statement ID
    name: str
    type: str
    description: str = ""
    resolved_id: uuid.UUID | None = None # Added by the entity resolution block

    @classmethod
    def from_raw(cls, raw_entity: RawEntity, event_id: uuid.UUID | None = None) -> "Entity":
        """Create an Entity instance from a RawEntity, optionally associating it with an event_id."""
        return cls(
            id=uuid.UUID(int=raw_entity.entity_idx),
            event_id=event_id,
            name=raw_entity.name,
            type=raw_entity.type,
            description=raw_entity.description,
            resolved_id=None, # will be populated during entity resolution with the canonical entity's id to remove duplicate naming of entities in the database.
        )


# Both, Raw Triplet and Raw Entity are extracted at the same time to reduce LLM calls
class RawExtraction(BaseModel):
    """Data model representing a raw triplet extraction."""

    triplets: list[RawTriplet]
    entities: list[RawEntity]


class TemporalEvent(BaseModel):
    """
    Data model representing a temporal event with statement, triplet, and validity information.

    Brings together the Statement and all related information into one handy class.
    It's a primary output of the TemporalAgent and plays an important role within the InvalidationAgent.
    """

    id: uuid.UUID | None = None
    publication_id: str
    statement: str # full str of the statemenet extracted from the publication
    embedding: list[float] = Field(default_factory=lambda: [0.0] * 256)
    triplets: list[uuid.UUID] # list of IDs of extracted triplets
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    temporal_type: TemporalType # ATEMPORAL, EVENT, STATE, or FORECAST
    temporal_confidence: TemporalConfidence # How confident is temporal type extraction for the statement
    statement_type: StatementType # FACT, OPINION or PREDICTION
    created_at: datetime = Field(default_factory=datetime.now)
    expired_at: datetime | None = None
    invalidated_by: uuid.UUID | None = None

    valid_at_confidence: TemporalConfidence # related to valid_at (LOW, MEDIUM, HIGH)
    invalid_at_confidence: TemporalConfidence # related to invalid_ad (LOW, MEDIUM, HIGH)

    @property
    def triplets_json(self) -> str:
        """Convert triplets list to JSON string."""
        return json.dumps([str(t) for t in self.triplets]) if self.triplets else "[]"

    @classmethod
    def parse_triplets_json(cls, triplets_str: str) -> list[uuid.UUID]:
        """Parse JSON string back into list of UUIDs."""
        if not triplets_str or triplets_str == "[]":
            return []
        return [uuid.UUID(t) for t in json.loads(triplets_str)]

    @model_validator(mode="after")
    def set_expired_at(self) -> "TemporalEvent":
        """Set expired_at if invalid_at is set and temporal_type is STATE (e.g., dynamic)."""
        self.expired_at = self.created_at if (self.invalid_at is not None and self.temporal_type == TemporalType.STATE) else None
        return self


def parse_date_str(value: str | datetime | None) -> datetime | None:
    """
    Parse a date string into a datetime object.

    If the value is a 4-digit year, it returns January 1 of that year in UTC.
    Otherwise, it attempts to parse the date string using dateutil.parser.parse.
    If the resulting datetime has no timezone, it defaults to UTC.
    """
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        # Year Handling
        if re.fullmatch(r"\d{4}", value.strip()):
            year = int(value.strip())
            return datetime(year, 1, 1, tzinfo=timezone.utc)

        #  General Handing
        dt: datetime = parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    except Exception:
        return None