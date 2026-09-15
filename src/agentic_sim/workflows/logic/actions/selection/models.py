from dataclasses import dataclass

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)


@dataclass(frozen=True)
class ActionSelection:
    """
    Result of selecting one action candidate.

    candidate_index preserves the candidate's position in the
    original opportunity set.

    predicted_value is the value estimate used when making
    the selection.
    """

    candidate_index: int
    candidate: ActionCandidate
    predicted_value: float
