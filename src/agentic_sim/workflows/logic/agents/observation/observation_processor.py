from dataclasses import dataclass

import torch

from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)


@dataclass(frozen=True)
class ProcessedObservation:
    """
    Numerical representation of an agent observation.

    `values` contains the numerical information available
    for downstream model processing.

    `feature_names` preserves the semantic meaning and ordering
    of those values for traceability and debugging.
    """

    values: torch.Tensor
    feature_names: tuple[str, ...]


class ObservationProcessor:
    """
    Converts a semantic AgentObservation into numerical model input.

    V1 behavior:
        - reads all visible numerical economic fields;
        - preserves deterministic ordering;
        - ignores structural IDs and non-numerical descriptive fields.

    Future versions may add:
        - normalization;
        - derived features;
        - aggregation;
        - historical features;
        - missing-value handling;
        - categorical encoding;
        - learned preprocessing.
    """

    SECTION_ORDER: tuple[str, ...] = (
        "accounts",
        "institutions",
        "markets",
        "companies",
    )

    def _extract_numeric_fields(
        self,
        *,
        section_name: str,
        objects: list[dict],
    ) -> tuple[list[float], list[str]]:

        values: list[float] = []
        feature_names: list[str] = []

        for object_index, object_data in enumerate(objects):

            visible_fields = object_data.get(
                "visible_fields",
                {},
            )

            for field_name, field_value in visible_fields.items():

                if isinstance(field_value, bool):
                    continue

                if not isinstance(
                    field_value,
                    (int, float),
                ):
                    continue

                values.append(
                    float(field_value),
                )

                feature_names.append(f"{section_name}[{object_index}].{field_name}")

        return values, feature_names

    def process(
        self,
        observation: AgentObservation,
    ) -> ProcessedObservation:

        parameters = observation.observation_parameters

        values: list[float] = []
        feature_names: list[str] = []

        for section_name in self.SECTION_ORDER:

            section_objects = parameters.get(
                section_name,
                [],
            )

            section_values, section_feature_names = self._extract_numeric_fields(
                section_name=section_name,
                objects=section_objects,
            )

            values.extend(
                section_values,
            )

            feature_names.extend(
                section_feature_names,
            )

        tensor = torch.tensor(
            values,
            dtype=torch.float32,
        )

        return ProcessedObservation(
            values=tensor,
            feature_names=tuple(feature_names),
        )
