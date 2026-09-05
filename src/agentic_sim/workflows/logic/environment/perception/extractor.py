from dataclasses import fields
from typing import TYPE_CHECKING, TypeAlias

from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    CompanyState,
    InstitutionState,
    MarketState,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)

if TYPE_CHECKING:
    from agentic_sim.workflows.logic.environment.environment import (
        Environment,
    )


VisibleTarget: TypeAlias = AccountState | CompanyState | InstitutionState | MarketState


class VisibleFieldExtractor:

    def get_visible_metadata_fields(
        self,
        *,
        actor_id: int,
        target_object: VisibleTarget,
        environment: "Environment",
    ) -> dict[str, MetadataField]:

        visible_fields: dict[str, MetadataField] = {}

        for field_info in fields(
            target_object,
        ):

            value = getattr(
                target_object,
                field_info.name,
            )

            if not isinstance(
                value,
                MetadataField,
            ):
                continue

            if environment.visibility.is_visible(
                actor_id=actor_id,
                target_object=target_object,
                field=value,
            ):
                visible_fields[field_info.name] = value

        return visible_fields

    def get_visible_fields(
        self,
        *,
        actor_id: int,
        target_object: VisibleTarget,
        environment: "Environment",
    ) -> dict[str, object]:

        visible_metadata_fields = self.get_visible_metadata_fields(
            actor_id=actor_id,
            target_object=target_object,
            environment=environment,
        )

        return {
            field_name: metadata_field.value
            for field_name, metadata_field in visible_metadata_fields.items()
        }
