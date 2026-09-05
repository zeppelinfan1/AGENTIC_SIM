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


VisibleTarget: TypeAlias = AccountState | InstitutionState | MarketState | CompanyState


class VisibleFieldExtractor:

    def get_visible_fields(
        self,
        *,
        actor_id: int,
        target_object: VisibleTarget,
        environment: "Environment",
    ) -> dict[str, object]:

        visible_values: dict[str, object] = {}

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
                visible_values[field_info.name] = value.value

        return visible_values
