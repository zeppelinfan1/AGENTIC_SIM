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
from agentic_sim.workflows.logic.actions.selection.greedy import (
    GreedyActionSelector,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)


def _candidate(
    magnitude: float,
) -> ActionCandidate:

    return ActionCandidate(
        target=TargetReference(
            object_type="account",
            object_id=1,
            field_name="balance",
        ),
        direction=PrimitiveDirection.CONSUME,
        magnitude=magnitude,
    )


def test_selects_highest_predicted_value():

    candidates = (
        _candidate(25.0),
        _candidate(50.0),
        _candidate(75.0),
    )

    valued_candidates = ValuedActionCandidates(
        candidates=candidates,
        predicted_values=torch.tensor(
            [
                0.2,
                1.5,
                0.7,
            ]
        ),
    )

    selector = GreedyActionSelector()

    selection = selector.select(
        valued_candidates=valued_candidates,
    )

    assert selection is not None

    assert selection.candidate_index == 1
    assert selection.candidate == candidates[1]
    assert selection.candidate.magnitude == 50.0

    assert selection.predicted_value == pytest.approx(1.5)


def test_returns_none_when_no_candidates_exist():

    valued_candidates = ValuedActionCandidates(
        candidates=(),
        predicted_values=torch.empty(0),
    )

    selector = GreedyActionSelector()

    selection = selector.select(
        valued_candidates=valued_candidates,
    )

    assert selection is None


def test_tie_selects_first_highest_candidate():

    candidates = (
        _candidate(25.0),
        _candidate(50.0),
        _candidate(75.0),
    )

    valued_candidates = ValuedActionCandidates(
        candidates=candidates,
        predicted_values=torch.tensor(
            [
                1.0,
                2.0,
                2.0,
            ]
        ),
    )

    selector = GreedyActionSelector()

    selection = selector.select(
        valued_candidates=valued_candidates,
    )

    assert selection is not None
    assert selection.candidate_index == 1
    assert selection.candidate.magnitude == 50.0


def test_rejects_non_finite_predictions():

    candidates = (
        _candidate(25.0),
        _candidate(50.0),
    )

    valued_candidates = ValuedActionCandidates(
        candidates=candidates,
        predicted_values=torch.tensor(
            [
                1.0,
                float("nan"),
            ]
        ),
    )

    selector = GreedyActionSelector()

    with pytest.raises(ValueError):
        selector.select(
            valued_candidates=valued_candidates,
        )
