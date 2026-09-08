from dataclasses import dataclass

import torch

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)


@dataclass(frozen=True)
class ValuedActionCandidates:
    """
    Action candidates paired positionally with the agent's
    current predicted long-term values.

    candidates[i] corresponds exactly to predicted_values[i].

    `predicted_values` shape:

        [num_candidates]
    """

    candidates: tuple[ActionCandidate, ...]
    predicted_values: torch.Tensor

    def __post_init__(self) -> None:

        if self.predicted_values.ndim != 1:
            raise ValueError(
                (
                    "Predicted candidate values must be a "
                    "one-dimensional tensor | "
                    f"shape={tuple(self.predicted_values.shape)}"
                )
            )

        if self.predicted_values.shape[0] != len(self.candidates):
            raise ValueError(
                (
                    "Candidate/value count mismatch | "
                    f"candidates={len(self.candidates)} | "
                    "predicted_values="
                    f"{self.predicted_values.shape[0]}"
                )
            )
