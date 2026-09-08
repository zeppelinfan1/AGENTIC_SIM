import pytest
import torch

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)
from agentic_sim.workflows.logic.actions.action_candidate_value import (
    ValuedActionCandidates,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)


def _candidate(
    *,
    magnitude: float,
) -> ActionCandidate:

    return ActionCandidate(
        target=TargetReference(
            object_type="account",
            object_id=1,
            field_name="balance",
        ),
        direction=(PrimitiveDirection.CONSUME),
        magnitude=magnitude,
    )


def test_candidates_and_values_are_positionally_aligned():

    candidates = (
        _candidate(magnitude=25.0),
        _candidate(magnitude=50.0),
        _candidate(magnitude=75.0),
    )

    values = torch.tensor(
        [
            -0.4,
            1.3,
            0.2,
        ]
    )

    valued = ValuedActionCandidates(
        candidates=candidates,
        predicted_values=values,
    )

    assert valued.candidates[1].magnitude == 50.0

    assert valued.predicted_values[1].item() == pytest.approx(1.3)


def test_candidate_value_count_must_match():

    candidates = (
        _candidate(magnitude=25.0),
        _candidate(magnitude=50.0),
    )

    with pytest.raises(ValueError):
        ValuedActionCandidates(
            candidates=candidates,
            predicted_values=torch.tensor(
                [
                    1.0,
                ]
            ),
        )


def test_predicted_values_must_be_one_dimensional():

    with pytest.raises(ValueError):
        ValuedActionCandidates(
            candidates=(_candidate(magnitude=25.0),),
            predicted_values=torch.tensor(
                [
                    [
                        1.0,
                    ]
                ]
            ),
        )
