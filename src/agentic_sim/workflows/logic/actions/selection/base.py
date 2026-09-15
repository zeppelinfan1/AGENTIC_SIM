from abc import ABC, abstractmethod

from agentic_sim.workflows.logic.actions.action_candidate_value import (
    ValuedActionCandidates,
)
from agentic_sim.workflows.logic.actions.selection.models import (
    ActionSelection,
)


class ActionSelector(ABC):
    """
    Base interface for selecting one action from an evaluated
    opportunity set.
    """

    @abstractmethod
    def select(
        self,
        *,
        valued_candidates: ValuedActionCandidates,
    ) -> ActionSelection | None:
        raise NotImplementedError
