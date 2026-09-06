from enum import StrEnum

from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)


class PrimitiveDirection(StrEnum):
    """
    Fundamental primitive transformations available to agents.

    These are world-level transformation directions, not
    human-defined economic strategies.
    """

    CREATE = "create"
    DESTROY = "destroy"
    PRODUCE = "produce"
    CONSUME = "consume"


_ELIGIBILITY_ATTRIBUTE_BY_DIRECTION: dict[
    PrimitiveDirection,
    str,
] = {
    PrimitiveDirection.CREATE: "create",
    PrimitiveDirection.DESTROY: "destroy",
    PrimitiveDirection.PRODUCE: "produce",
    PrimitiveDirection.CONSUME: "consume",
}


def is_direction_eligible(
    *,
    eligibility: IntentEligibilityMetadata,
    direction: PrimitiveDirection,
) -> bool:
    """
    Return whether a primitive direction is allowed by
    the target field's eligibility metadata.
    """

    attribute_name = _ELIGIBILITY_ATTRIBUTE_BY_DIRECTION[direction]

    return bool(
        getattr(
            eligibility,
            attribute_name,
        )
    )


def get_eligible_directions(
    *,
    eligibility: IntentEligibilityMetadata,
) -> tuple[PrimitiveDirection, ...]:
    """
    Expand eligibility booleans into concrete primitive directions.

    Ordering is deterministic and follows PrimitiveDirection
    definition order.
    """

    return tuple(
        direction
        for direction in PrimitiveDirection
        if is_direction_eligible(
            eligibility=eligibility,
            direction=direction,
        )
    )
