from dataclasses import dataclass
import math

from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)


@dataclass(frozen=True)
class ActionCandidate:
    """
    One concrete primitive transformation that an agent
    could potentially request against the environment.

    A candidate consists of:

        target
        + primitive direction
        + positive magnitude

    Direction and magnitude are intentionally separate.
    Magnitude is never signed.
    """

    target: TargetReference
    direction: PrimitiveDirection
    magnitude: float

    def __post_init__(self) -> None:

        numeric_magnitude = float(
            self.magnitude,
        )

        if not math.isfinite(
            numeric_magnitude,
        ):
            raise ValueError("Action candidate magnitude must be finite.")

        if numeric_magnitude <= 0.0:
            raise ValueError(
                ("Action candidate magnitude must be " "strictly greater than zero.")
            )
