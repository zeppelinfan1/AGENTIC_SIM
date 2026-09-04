from dataclasses import dataclass

from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.extractor import (
    VisibleFieldExtractor,
)


@dataclass(frozen=True)
class AgentPerceptionLinkage:
    """
    Stable environment topology accessible from one agent.

    Stores IDs only so that perception always retrieves
    fresh state from the current Environment.
    """

    agent_id: int
    account_ids: tuple[int, ...]
    institution_ids: tuple[int, ...]
    market_ids: tuple[int, ...]


class Perception:
    """
    Resolves an agent's current perception of the environment.

    build():
        Builds stable Agent -> Account -> Institution -> Market
        linkage indexes.

    perceive():
        Uses those links to retrieve current environment state,
        applies visibility rules, and returns an AgentObservation.
    """

    def __init__(
        self,
        visible_field_extractor: VisibleFieldExtractor,
    ) -> None:
        self.visible_field_extractor = visible_field_extractor

        self.linkages: dict[
            int,
            AgentPerceptionLinkage,
        ] = {}

    def build(
        self,
        *,
        agents: list[Agent],
        environment: Environment,
    ) -> None:
        """
        Build perception topology for all agents.

        This should normally run once after the environment
        topology has been created.
        """

        self.linkages = {}

        for agent in agents:
            actor_id = agent.config.agent_id

            # Agent -> Account
            account_ids = tuple(
                account.account_id
                for account in environment.accounts
                if account.agent_id == actor_id
            )

            # Account -> Institution
            institution_ids = tuple(
                sorted(
                    {
                        account.institution_id
                        for account in environment.accounts
                        if account.account_id in account_ids
                    }
                )
            )

            # Institution -> Market
            market_ids = tuple(
                sorted(
                    {
                        institution.market_id
                        for institution in environment.institutions
                        if institution.institution_id in institution_ids
                    }
                )
            )

            self.linkages[actor_id] = AgentPerceptionLinkage(
                agent_id=actor_id,
                account_ids=account_ids,
                institution_ids=institution_ids,
                market_ids=market_ids,
            )

    def perceive(
        self,
        *,
        agent: Agent,
        environment: Environment,
    ) -> AgentObservation:
        """
        Build a fresh observation for one agent from the
        current state of the environment.
        """

        actor_id = agent.config.agent_id

        if actor_id not in self.linkages:
            raise ValueError(f"No perception linkage built for agent: {actor_id}")

        linkage = self.linkages[actor_id]

        # Retrieve CURRENT objects using stable linkage IDs
        accounts = [
            account
            for account in environment.accounts
            if account.account_id in linkage.account_ids
        ]

        institutions = [
            institution
            for institution in environment.institutions
            if institution.institution_id in linkage.institution_ids
        ]

        markets = [
            market
            for market in environment.markets
            if market.market_id in linkage.market_ids
        ]

        # Apply field-level visibility
        visible_accounts = [
            {
                "account_id": account.account_id,
                "institution_id": account.institution_id,
                "visible_fields": (
                    self.visible_field_extractor.get_visible_fields(
                        actor_id=actor_id,
                        target_object=account,
                        environment=environment,
                    )
                ),
            }
            for account in accounts
        ]

        visible_institutions = [
            {
                "institution_id": institution.institution_id,
                "market_id": institution.market_id,
                "visible_fields": (
                    self.visible_field_extractor.get_visible_fields(
                        actor_id=actor_id,
                        target_object=institution,
                        environment=environment,
                    )
                ),
            }
            for institution in institutions
        ]

        visible_markets = [
            {
                "market_id": market.market_id,
                "visible_fields": (
                    self.visible_field_extractor.get_visible_fields(
                        actor_id=actor_id,
                        target_object=market,
                        environment=environment,
                    )
                ),
            }
            for market in markets
        ]

        return AgentObservation(
            agent_name=agent.config.agent_name,
            observation_type="transactional_agent",
            observation_parameters={
                "agent_id": actor_id,
                "environment_step": environment.state.step,
                "accounts": visible_accounts,
                "institutions": visible_institutions,
                "markets": visible_markets,
            },
        )
