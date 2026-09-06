import math

from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint import (
    MagnitudeConstraint,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
    is_direction_eligible,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)
from agentic_sim.workflows.logic.actions.targets.target_reference_resolver import (
    TargetReferenceResolver,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)


class MagnitudeConstraintResolver:
    """
    Determines the currently feasible magnitude range for one
    target + primitive-direction combination.

    V1 rules:

        DESTROY / CONSUME:
            Cannot remove more of a positive quantity than
            currently exists.

        CREATE / PRODUCE:
            Use the current absolute quantity as the initial
            action scale, with a configurable positive floor.

    These are intentionally isolated V1 mechanics. More detailed
    field-specific and cross-resource constraints can replace
    them later without changing ActionCandidate.
    """

    DECREASING_DIRECTIONS: frozenset[PrimitiveDirection] = frozenset(
        {
            PrimitiveDirection.DESTROY,
            PrimitiveDirection.CONSUME,
        }
    )

    INCREASING_DIRECTIONS: frozenset[PrimitiveDirection] = frozenset(
        {
            PrimitiveDirection.CREATE,
            PrimitiveDirection.PRODUCE,
        }
    )

    def __init__(
        self,
        *,
        target_reference_resolver: TargetReferenceResolver,
        increase_scale_floor: float = 1.0,
    ) -> None:

        if not math.isfinite(
            increase_scale_floor,
        ):
            raise ValueError("increase_scale_floor must be finite.")

        if increase_scale_floor <= 0.0:
            raise ValueError(("increase_scale_floor must be " "strictly positive."))

        self.target_reference_resolver = target_reference_resolver

        self.increase_scale_floor = float(
            increase_scale_floor,
        )

    def resolve(
        self,
        *,
        actor_id: int,
        target: TargetReference,
        direction: PrimitiveDirection,
        environment: Environment,
    ) -> MagnitudeConstraint | None:
        """
        Resolve current physical magnitude bounds.

        Returns None when the requested primitive direction
        cannot currently produce a feasible positive magnitude.
        """

        # actor_id is deliberately part of the public contract.
        # Future physics can make magnitude constraints depend
        # on actor-specific resources, ownership, permissions,
        # credit, production capacity, etc.
        _ = actor_id

        metadata_field = self.target_reference_resolver.resolve_metadata_field(
            reference=target,
            environment=environment,
        )

        if not is_direction_eligible(
            eligibility=metadata_field.eligibility,
            direction=direction,
        ):
            return None

        current_value = metadata_field.value

        if isinstance(
            current_value,
            bool,
        ):
            return None

        if not isinstance(
            current_value,
            (int, float),
        ):
            return None

        current_numeric_value = float(
            current_value,
        )

        if not math.isfinite(
            current_numeric_value,
        ):
            return None

        if direction in self.DECREASING_DIRECTIONS:

            maximum = max(
                current_numeric_value,
                0.0,
            )

        elif direction in self.INCREASING_DIRECTIONS:

            maximum = max(
                abs(
                    current_numeric_value,
                ),
                self.increase_scale_floor,
            )

        else:
            raise ValueError(
                ("Unsupported primitive direction | " f"direction={direction}")
            )

        if maximum <= 0.0:
            return None

        return MagnitudeConstraint(
            minimum=0.0,
            maximum=maximum,
        )
