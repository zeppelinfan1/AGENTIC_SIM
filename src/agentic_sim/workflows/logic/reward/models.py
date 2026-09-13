from dataclasses import dataclass, field
import math
from typing import Mapping

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)


@dataclass(frozen=True)
class RewardSnapshot:
    """
    Objective measurements captured at one point in time
    for the purpose of reward evaluation.

    The generic metric mapping allows future reward mechanisms
    to capture completely different information without changing
    the simulation engine contract.
    """

    actor_id: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:

        for metric_name, metric_value in self.metrics.items():

            if not math.isfinite(float(metric_value)):
                raise ValueError(
                    (
                        "Reward snapshot metric must be finite | "
                        f"metric={metric_name} | "
                        f"value={metric_value}"
                    )
                )


@dataclass(frozen=True)
class RewardContext:
    """
    Information available when calculating reward for one
    completed transition.

    `signals` is deliberately generic.

    Future reward systems can supply things such as:
        - truth orientation;
        - prediction error;
        - surprise;
        - uncertainty;
        - expectation contradiction;
        - other psychological or economic signals;

    without changing this interface.
    """

    actor_id: int

    before: RewardSnapshot
    after: RewardSnapshot

    action: ActionCandidate | None = None

    signals: Mapping[str, float] = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class RewardResult:
    """
    Final subjective scalar reward produced for one transition.

    `components` preserves the quantities that contributed
    to the result for debugging and research.
    """

    value: float
    components: Mapping[str, float]

    def __post_init__(self) -> None:

        if not math.isfinite(float(self.value)):
            raise ValueError("Reward value must be finite.")

        for (
            component_name,
            component_value,
        ) in self.components.items():

            if not math.isfinite(float(component_value)):
                raise ValueError(
                    (
                        "Reward component must be finite | "
                        f"component={component_name} | "
                        f"value={component_value}"
                    )
                )
