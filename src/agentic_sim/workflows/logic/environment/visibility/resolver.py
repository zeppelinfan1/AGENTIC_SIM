from typing import TYPE_CHECKING

from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    CompanyState,
    InstitutionState,
    MarketState,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
    VisibilityType,
)
from agentic_sim.workflows.logic.environment.visibility.rules import (
    CascadingVisibility,
    PrivateVisibility,
    PublicVisibility,
    RestrictedVisibility,
    VisibilityRule,
)

if TYPE_CHECKING:
    from agentic_sim.workflows.logic.environment.environment import (
        Environment,
    )


class VisibilityResolver:
    """
    Resolves visibility using field-level visibility rules
    and the topology of the current environment.
    """

    def __init__(
        self,
        environment: "Environment",
    ) -> None:
        self.environment = environment

        self.rules: dict[
            VisibilityType,
            VisibilityRule,
        ] = {
            "public": PublicVisibility(),
            "private": PrivateVisibility(),
            "cascading": CascadingVisibility(),
            "restricted": RestrictedVisibility(),
        }

    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        field: MetadataField,
    ) -> bool:
        rule = self.rules[field.visibility_type]

        return rule.is_visible(
            actor_id=actor_id,
            target_object=target_object,
            resolver=self,
        )

    def is_directly_linked(
        self,
        *,
        actor_id: int,
        target_object: object,
    ) -> bool:
        """
        Direct relationship only.

        Current V1 definition:
        Agent -> own Account
        """

        if isinstance(
            target_object,
            AccountState,
        ):
            return target_object.agent_id == actor_id

        return False

    def is_cascading_linked(
        self,
        *,
        actor_id: int,
        target_object: object,
    ) -> bool:
        """
        Resolve directional visibility through:

        Agent
        -> Account
        -> Institution
        -> Market
        -> Company
        """

        # Agent -> own Account
        if isinstance(
            target_object,
            AccountState,
        ):
            return target_object.agent_id == actor_id

        # Institutions accessible through the actor's accounts
        institution_ids = {
            account.institution_id
            for account in self.environment.accounts
            if account.agent_id == actor_id
        }

        if isinstance(
            target_object,
            InstitutionState,
        ):
            return target_object.institution_id in institution_ids

        # Markets accessible through those institutions
        market_ids = {
            institution.market_id
            for institution in self.environment.institutions
            if institution.institution_id in institution_ids
        }

        if isinstance(
            target_object,
            MarketState,
        ):
            return target_object.market_id in market_ids

        # Companies operating in accessible markets
        if isinstance(
            target_object,
            CompanyState,
        ):
            return target_object.market_id in market_ids

        return False
