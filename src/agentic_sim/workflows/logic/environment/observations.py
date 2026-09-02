from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import Environment
from agentic_sim.workflows.logic.environment.extractor import (
    VisibleFieldExtractor,
)


class ObservationBuilder:

    def __init__(
        self,
        visible_field_extractor: VisibleFieldExtractor,
    ) -> None:
        self.visible_field_extractor = visible_field_extractor

    def build_agent_observation(
        self,
        *,
        agent: Agent,
        environment: Environment,
    ) -> AgentObservation:

        actor_id = agent.config.agent_id

        # Agent -> Account
        accounts = [
            account for account in environment.accounts if account.agent_id == actor_id
        ]

        # Account -> Institution
        institution_ids = {account.institution_id for account in accounts}

        institutions = [
            institution
            for institution in environment.institutions
            if institution.institution_id in institution_ids
        ]

        # Institution -> Market
        market_ids = {institution.market_id for institution in institutions}

        markets = [
            market for market in environment.markets if market.market_id in market_ids
        ]

        visible_accounts = [
            self.visible_field_extractor.get_visible_fields(
                actor_id=actor_id,
                target_object=account,
                environment=environment,
            )
            for account in accounts
        ]

        visible_institutions = [
            self.visible_field_extractor.get_visible_fields(
                actor_id=actor_id,
                target_object=institution,
                environment=environment,
            )
            for institution in institutions
        ]

        visible_markets = [
            self.visible_field_extractor.get_visible_fields(
                actor_id=actor_id,
                target_object=market,
                environment=environment,
            )
            for market in markets
        ]

        return AgentObservation(
            agent_name=agent.config.agent_name,
            observation_type="transactional_agent",
            observation_parameters={
                "accounts": visible_accounts,
                "institutions": visible_institutions,
                "markets": visible_markets,
                "environment_step": environment.state.step,
            },
        )
