from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_candidate_generator import (
    MagnitudeCandidateGenerator,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint_resolver import (
    MagnitudeConstraintResolver,
)
from agentic_sim.workflows.logic.actions.primitives import (
    get_eligible_directions,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetCandidate,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)


class ActionCandidateProvider:
    """
    Expands eligible target fields into concrete primitive
    action candidates.

    Pipeline:

        target candidate
        -> eligible primitive directions
        -> feasible magnitude bounds
        -> discrete magnitudes
        -> ActionCandidate objects

    This component does not rank, value, select, or execute
    candidates.
    """

    def __init__(
        self,
        *,
        magnitude_constraint_resolver: MagnitudeConstraintResolver,
        magnitude_candidate_generator: MagnitudeCandidateGenerator,
    ) -> None:

        self.magnitude_constraint_resolver = magnitude_constraint_resolver

        self.magnitude_candidate_generator = magnitude_candidate_generator

    def get_candidates(
        self,
        *,
        actor_id: int,
        target_candidates: list[TargetCandidate],
        environment: Environment,
    ) -> list[ActionCandidate]:

        candidates: list[ActionCandidate] = []

        for target_candidate in target_candidates:

            directions = get_eligible_directions(
                eligibility=target_candidate.eligibility,
            )

            for direction in directions:

                constraint = self.magnitude_constraint_resolver.resolve(
                    actor_id=actor_id,
                    target=target_candidate.reference,
                    direction=direction,
                    environment=environment,
                )

                if constraint is None:
                    continue

                magnitudes = self.magnitude_candidate_generator.generate(
                    constraint=constraint,
                )

                for magnitude in magnitudes:

                    candidates.append(
                        ActionCandidate(
                            target=(target_candidate.reference),
                            direction=direction,
                            magnitude=magnitude,
                        )
                    )

        return candidates
