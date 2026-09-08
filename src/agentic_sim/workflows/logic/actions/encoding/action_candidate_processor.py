from collections.abc import Iterable
from dataclasses import dataclass, fields
import math
from typing import get_origin, get_type_hints

import torch

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetObjectType,
)
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    CompanyState,
    InstitutionState,
    MarketState,
)


@dataclass(frozen=True)
class ProcessedActionCandidates:
    """
    Fixed-width numerical representation of a collection
    of concrete ActionCandidate objects.

    `values` shape:

        [num_candidates, feature_dim]

    Candidate ordering is preserved exactly so that row i
    always corresponds to candidates[i].
    """

    candidates: tuple[ActionCandidate, ...]
    values: torch.Tensor
    feature_names: tuple[str, ...]


class ActionCandidateProcessor:
    """
    Converts semantic ActionCandidate objects into fixed-width
    numerical features suitable for ActionEncoder.

    Candidate representation currently contains:

        - target object type;
        - primitive direction;
        - semantic target field identity;
        - visible current target value;
        - candidate magnitude;
        - magnitude relative to target scale;
        - zero-target indicator.

    Raw object IDs are deliberately NOT used as neural features.

    IDs identify mutable environment objects operationally, but
    the numeric ordering of those IDs has no economic meaning.
    """

    OBJECT_TYPE_ORDER: tuple[
        TargetObjectType,
        ...,
    ] = (
        "account",
        "institution",
        "market",
        "company",
    )

    DIRECTION_ORDER: tuple[
        PrimitiveDirection,
        ...,
    ] = tuple(PrimitiveDirection)

    TARGET_STATE_TYPES: dict[
        TargetObjectType,
        type,
    ] = {
        "account": AccountState,
        "institution": InstitutionState,
        "market": MarketState,
        "company": CompanyState,
    }

    OBSERVATION_SECTIONS: dict[
        TargetObjectType,
        tuple[str, str],
    ] = {
        "account": (
            "accounts",
            "account_id",
        ),
        "institution": (
            "institutions",
            "institution_id",
        ),
        "market": (
            "markets",
            "market_id",
        ),
        "company": (
            "companies",
            "company_id",
        ),
    }

    def __init__(
        self,
    ) -> None:

        self.target_field_keys = self._discover_target_field_keys()

        self.target_field_index = {
            field_key: index for index, field_key in enumerate(self.target_field_keys)
        }

        self.feature_names = self._build_feature_names()

    def _discover_target_field_keys(
        self,
    ) -> tuple[
        tuple[TargetObjectType, str],
        ...,
    ]:
        """
        Discover MetadataField-backed target fields directly
        from the environment state dataclasses.

        This provides a stable semantic vocabulary without
        depending on which particular entities happen to exist
        in one simulation run.
        """

        field_keys: list[tuple[TargetObjectType, str]] = []

        for object_type in self.OBJECT_TYPE_ORDER:

            state_type = self.TARGET_STATE_TYPES[object_type]

            type_hints = get_type_hints(state_type)

            for field_info in fields(state_type):

                annotation = type_hints.get(field_info.name)

                if annotation is None:
                    continue

                if (
                    get_origin(annotation) is not MetadataField
                    and annotation is not MetadataField
                ):
                    continue

                field_keys.append(
                    (
                        object_type,
                        field_info.name,
                    )
                )

        return tuple(field_keys)

    def _build_feature_names(
        self,
    ) -> tuple[str, ...]:

        feature_names: list[str] = []

        for object_type in self.OBJECT_TYPE_ORDER:
            feature_names.append(f"target_object_type.{object_type}")

        for direction in self.DIRECTION_ORDER:
            feature_names.append(f"direction.{direction.value}")

        for (
            object_type,
            field_name,
        ) in self.target_field_keys:

            feature_names.append(("target_field." f"{object_type}." f"{field_name}"))

        feature_names.extend(
            (
                "target_value_signed_log1p",
                "magnitude_log1p",
                "magnitude_relative_to_target_scale",
                "target_value_is_zero",
            )
        )

        return tuple(feature_names)

    def _get_visible_target_value(
        self,
        *,
        candidate: ActionCandidate,
        observation: AgentObservation,
    ) -> float:
        """
        Retrieve the target's value from the agent's observation,
        not directly from unrestricted environment state.

        This is important because candidate encoding must not
        accidentally expose information that the agent was not
        allowed to perceive.
        """

        object_type = candidate.target.object_type

        (
            section_name,
            id_field,
        ) = self.OBSERVATION_SECTIONS[object_type]

        observed_objects = observation.observation_parameters.get(
            section_name,
            [],
        )

        matching_objects = [
            observed_object
            for observed_object in observed_objects
            if observed_object.get(id_field) == candidate.target.object_id
        ]

        if not matching_objects:
            raise ValueError(
                (
                    "Action candidate target does not exist "
                    "in current agent observation | "
                    f"type={object_type} | "
                    f"id={candidate.target.object_id}"
                )
            )

        if len(matching_objects) > 1:
            raise ValueError(
                (
                    "Action candidate target appears multiple "
                    "times in current agent observation | "
                    f"type={object_type} | "
                    f"id={candidate.target.object_id}"
                )
            )

        visible_fields = matching_objects[0].get(
            "visible_fields",
            {},
        )

        field_name = candidate.target.field_name

        if field_name not in visible_fields:
            raise ValueError(
                (
                    "Action candidate target field is not "
                    "visible in current observation | "
                    f"type={object_type} | "
                    f"id={candidate.target.object_id} | "
                    f"field={field_name}"
                )
            )

        current_value = visible_fields[field_name]

        if isinstance(
            current_value,
            bool,
        ):
            raise ValueError(
                (
                    "Action candidate target value must be "
                    "numeric and non-boolean | "
                    f"field={field_name}"
                )
            )

        if not isinstance(
            current_value,
            (int, float),
        ):
            raise ValueError(
                (
                    "Action candidate target value must be "
                    "numeric | "
                    f"field={field_name} | "
                    f"value_type={type(current_value).__name__}"
                )
            )

        numeric_value = float(current_value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                (
                    "Action candidate target value must be "
                    "finite | "
                    f"field={field_name}"
                )
            )

        return numeric_value

    @staticmethod
    def _signed_log1p(
        value: float,
    ) -> float:
        """
        Scale potentially large economic values while
        preserving sign.
        """

        if value == 0.0:
            return 0.0

        return math.copysign(
            math.log1p(abs(value)),
            value,
        )

    def _build_feature_row(
        self,
        *,
        candidate: ActionCandidate,
        observation: AgentObservation,
    ) -> list[float]:

        current_target_value = self._get_visible_target_value(
            candidate=candidate,
            observation=observation,
        )

        target_field_key = (
            candidate.target.object_type,
            candidate.target.field_name,
        )

        if target_field_key not in self.target_field_index:
            raise ValueError(
                (
                    "Action candidate target field is not "
                    "present in action feature vocabulary | "
                    f"type={candidate.target.object_type} | "
                    f"field={candidate.target.field_name}"
                )
            )

        features: list[float] = []

        # Target object type one-hot
        features.extend(
            1.0 if candidate.target.object_type == object_type else 0.0
            for object_type in self.OBJECT_TYPE_ORDER
        )

        # Primitive direction one-hot
        features.extend(
            1.0 if candidate.direction == direction else 0.0
            for direction in self.DIRECTION_ORDER
        )

        # Semantic target field one-hot
        candidate_field_index = self.target_field_index[target_field_key]

        features.extend(
            1.0 if field_index == candidate_field_index else 0.0
            for field_index in range(len(self.target_field_keys))
        )

        target_scale = max(
            abs(current_target_value),
            1.0,
        )

        magnitude_relative_to_target_scale = candidate.magnitude / target_scale

        # Continuous candidate context
        features.extend(
            (
                self._signed_log1p(current_target_value),
                math.log1p(candidate.magnitude),
                magnitude_relative_to_target_scale,
                1.0 if current_target_value == 0.0 else 0.0,
            )
        )

        return features

    def process_many(
        self,
        *,
        candidates: Iterable[ActionCandidate],
        observation: AgentObservation,
    ) -> ProcessedActionCandidates:

        candidate_tuple = tuple(candidates)

        rows = [
            self._build_feature_row(
                candidate=candidate,
                observation=observation,
            )
            for candidate in candidate_tuple
        ]

        if rows:

            values = torch.tensor(
                rows,
                dtype=torch.float32,
            )

        else:

            values = torch.empty(
                (
                    0,
                    len(self.feature_names),
                ),
                dtype=torch.float32,
            )

        return ProcessedActionCandidates(
            candidates=candidate_tuple,
            values=values,
            feature_names=self.feature_names,
        )
