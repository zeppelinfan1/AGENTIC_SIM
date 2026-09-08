import pytest

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)


def _reference() -> TargetReference:
    return TargetReference(
        object_type="account",
        object_id=1,
        field_name="balance",
    )


def test_action_candidate_accepts_positive_magnitude():
    candidate = ActionCandidate(
        target=_reference(),
        direction=PrimitiveDirection.CONSUME,
        magnitude=10.0,
    )

    assert candidate.magnitude == 10.0


@pytest.mark.parametrize(
    "magnitude",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_action_candidate_rejects_invalid_magnitude(
    magnitude: float,
):
    with pytest.raises(
        ValueError,
    ):
        ActionCandidate(
            target=_reference(),
            direction=PrimitiveDirection.CONSUME,
            magnitude=magnitude,
        )
