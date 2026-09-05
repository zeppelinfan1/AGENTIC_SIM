from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetCandidate,
    TargetObjectType,
    TargetReference,
)
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.perception.extractor import (
    VisibleFieldExtractor,
    VisibleTarget,
)


class TargetCandidateProvider:
    """
    Identifies environment fields that:

        1. belong to objects currently known to the agent;
        2. are visible to the agent;
        3. are eligible for at least one intent direction.

    This component does not rank or select targets.
    """

    TARGET_SECTIONS: tuple[
        tuple[str, TargetObjectType, str, str],
        ...,
    ] = (
        (
            "accounts",
            "account",
            "account_id",
            "accounts",
        ),
        (
            "institutions",
            "institution",
            "institution_id",
            "institutions",
        ),
        (
            "markets",
            "market",
            "market_id",
            "markets",
        ),
        (
            "companies",
            "company",
            "company_id",
            "companies",
        ),
    )

    def __init__(
        self,
        *,
        visible_field_extractor: VisibleFieldExtractor,
    ) -> None:

        self.visible_field_extractor = visible_field_extractor

    def _build_object_index(
        self,
        *,
        environment: Environment,
        collection_name: str,
        id_field: str,
    ) -> dict[int, VisibleTarget]:

        collection: list[VisibleTarget] = getattr(
            environment,
            collection_name,
        )

        return {
            getattr(
                target_object,
                id_field,
            ): target_object
            for target_object in collection
        }

    def get_candidates(
        self,
        *,
        observation: AgentObservation,
        environment: Environment,
    ) -> list[TargetCandidate]:

        actor_id = observation.observation_parameters["agent_id"]

        candidates: list[TargetCandidate] = []

        for (
            section_name,
            object_type,
            id_field,
            collection_name,
        ) in self.TARGET_SECTIONS:

            object_index = self._build_object_index(
                environment=environment,
                collection_name=collection_name,
                id_field=id_field,
            )

            observed_objects = observation.observation_parameters.get(
                section_name,
                [],
            )

            for observed_object in observed_objects:

                object_id = observed_object[id_field]

                if object_id not in object_index:
                    raise ValueError(
                        (
                            "Observed target does not exist "
                            "in current environment | "
                            f"type={object_type} | "
                            f"id={object_id}"
                        )
                    )

                target_object = object_index[object_id]

                visible_fields = (
                    self.visible_field_extractor.get_visible_metadata_fields(
                        actor_id=actor_id,
                        target_object=target_object,
                        environment=environment,
                    )
                )

                for (
                    field_name,
                    metadata_field,
                ) in visible_fields.items():

                    if not (metadata_field.eligibility.has_any_eligibility):
                        continue

                    candidates.append(
                        TargetCandidate(
                            reference=TargetReference(
                                object_type=object_type,
                                object_id=object_id,
                                field_name=field_name,
                            ),
                            current_value=metadata_field.value,
                            eligibility=(metadata_field.eligibility),
                        )
                    )

        return candidates
