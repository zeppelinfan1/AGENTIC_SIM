from dataclasses import dataclass
from typing import Literal

from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)

TargetObjectType = Literal[
    "account",
    "institution",
    "market",
    "company",
]


@dataclass(frozen=True)
class TargetReference:
    """
    Stable reference to a potential action target.

    IDs are stored instead of mutable environment objects so
    the current state can always be resolved from Environment.
    """

    object_type: TargetObjectType
    object_id: int
    field_name: str


@dataclass(frozen=True)
class TargetCandidate:
    """
    A visible environment field that is eligible to participate
    in at least one intent direction.
    """

    reference: TargetReference
    current_value: object
    eligibility: IntentEligibilityMetadata
