from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)

T = TypeVar("T")


VisibilityType = Literal[
    "public",
    "private",
    "cascading",
    "restricted",
]


@dataclass(frozen=True)
class MetadataField(Generic[T]):
    """
    Environment state value with metadata describing
    visibility and action-intent eligibility.
    """

    value: T
    visibility_type: VisibilityType

    eligibility: IntentEligibilityMetadata = field(
        default_factory=IntentEligibilityMetadata,
    )
