from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
    get_eligible_directions,
    is_direction_eligible,
)
from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)


def test_get_eligible_directions_expands_only_enabled_directions():
    eligibility = IntentEligibilityMetadata(
        create=True,
        destroy=False,
        produce=False,
        consume=True,
    )

    directions = get_eligible_directions(
        eligibility=eligibility,
    )

    assert directions == (
        PrimitiveDirection.CREATE,
        PrimitiveDirection.CONSUME,
    )


def test_is_direction_eligible_returns_expected_value():
    eligibility = IntentEligibilityMetadata(
        create=False,
        destroy=True,
        produce=False,
        consume=False,
    )

    assert is_direction_eligible(
        eligibility=eligibility,
        direction=PrimitiveDirection.DESTROY,
    )

    assert not is_direction_eligible(
        eligibility=eligibility,
        direction=PrimitiveDirection.CREATE,
    )
