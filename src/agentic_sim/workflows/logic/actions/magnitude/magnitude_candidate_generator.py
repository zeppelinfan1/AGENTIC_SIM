from collections.abc import Iterable
import math

from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint import (
    MagnitudeConstraint,
)


class MagnitudeCandidateGenerator:
    """
    Converts a continuous feasible magnitude interval into
    a finite set of concrete candidate magnitudes.

    V1 uses fractions of the currently feasible interval.

    This makes exhaustive candidate evaluation possible while
    preserving the ability to replace discretisation later.
    """

    DEFAULT_FRACTIONS: tuple[
        float,
        ...,
    ] = (
        0.25,
        0.50,
        0.75,
        1.00,
    )

    def __init__(
        self,
        *,
        fractions: Iterable[float] | None = None,
    ) -> None:

        configured_fractions = tuple(
            float(fraction)
            for fraction in (
                fractions if fractions is not None else self.DEFAULT_FRACTIONS
            )
        )

        if not configured_fractions:
            raise ValueError(("At least one magnitude fraction " "must be configured."))

        for fraction in configured_fractions:

            if not math.isfinite(
                fraction,
            ):
                raise ValueError(("Magnitude fractions must " "all be finite."))

            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    ("Magnitude fractions must satisfy " "0 < fraction <= 1.")
                )

        if len(
            set(
                configured_fractions,
            )
        ) != len(
            configured_fractions,
        ):
            raise ValueError("Magnitude fractions must be unique.")

        self.fractions = tuple(
            sorted(
                configured_fractions,
            )
        )

    def generate(
        self,
        *,
        constraint: MagnitudeConstraint,
    ) -> tuple[float, ...]:

        if not constraint.has_positive_magnitude:
            return ()

        interval_size = constraint.maximum - constraint.minimum

        generated: list[float] = []

        for fraction in self.fractions:

            magnitude = constraint.minimum + interval_size * fraction

            if magnitude <= 0.0:
                continue

            if magnitude < constraint.minimum:
                continue

            if magnitude > constraint.maximum:
                continue

            generated.append(
                float(
                    magnitude,
                )
            )

        # A collapsed positive interval may generate the same
        # value for every fraction. Preserve deterministic
        # ordering while removing duplicates.
        return tuple(
            dict.fromkeys(
                generated,
            )
        )
