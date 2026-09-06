from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MagnitudeConstraint:
    """
    Current physical bounds for the magnitude of one
    target + primitive-direction combination.

    Magnitudes themselves are always positive. A minimum of
    zero means there is no positive lower bound imposed by
    the current V1 physics.
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:

        minimum = float(
            self.minimum,
        )

        maximum = float(
            self.maximum,
        )

        if not math.isfinite(
            minimum,
        ):
            raise ValueError("Magnitude minimum must be finite.")

        if not math.isfinite(
            maximum,
        ):
            raise ValueError("Magnitude maximum must be finite.")

        if minimum < 0.0:
            raise ValueError("Magnitude minimum cannot be negative.")

        if maximum < minimum:
            raise ValueError(
                ("Magnitude maximum cannot be smaller " "than magnitude minimum.")
            )

    @property
    def has_positive_magnitude(
        self,
    ) -> bool:
        return self.maximum > 0.0
