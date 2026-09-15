import torch

from agentic_sim.workflows.logic.actions.action_candidate_value import (
    ValuedActionCandidates,
)
from agentic_sim.workflows.logic.actions.selection.base import (
    ActionSelector,
)
from agentic_sim.workflows.logic.actions.selection.models import (
    ActionSelection,
)


class GreedyActionSelector(ActionSelector):
    """
    Selects the action candidate with the highest predicted value.

    V1 uses pure greedy selection:

        selected = argmax(predicted_values)

    If multiple candidates share the same maximum value,
    torch.argmax selects the first one.

    Returns None when no action candidates are available.
    """

    def select(
        self,
        *,
        valued_candidates: ValuedActionCandidates,
    ) -> ActionSelection | None:

        if not valued_candidates.candidates:
            return None

        predicted_values = valued_candidates.predicted_values

        if not torch.isfinite(predicted_values).all():
            raise ValueError("Predicted action values must all be finite.")

        candidate_index = int(torch.argmax(predicted_values).item())

        candidate = valued_candidates.candidates[candidate_index]

        predicted_value = float(predicted_values[candidate_index].item())

        return ActionSelection(
            candidate_index=candidate_index,
            candidate=candidate,
            predicted_value=predicted_value,
        )
