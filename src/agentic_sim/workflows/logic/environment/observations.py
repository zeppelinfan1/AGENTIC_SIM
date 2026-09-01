from agentic_sim.workflows.logic.agents.agent import (
    Agent,
)
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)


class ObservationBuilder:

    def build_agent_observation(
        self,
        agent: Agent,
        environment: Environment,
    ) -> AgentObservation:

        # --------------------------------------------------
        # Agent → Account
        # --------------------------------------------------

        agent_accounts = [
            account
            for account in environment.accounts
            if account.agent_id == agent.config.agent_id
        ]

        # --------------------------------------------------
        # Account → Institution
        # --------------------------------------------------

        institution_ids = {account.institution_id for account in agent_accounts}

        institutions = [
            institution
            for institution in environment.institutions
            if institution.institution_id in institution_ids
        ]

        # --------------------------------------------------
        # Institution → Market
        # --------------------------------------------------

        market_ids = {institution.market_id for institution in institutions}

        markets = [
            market for market in environment.markets if market.market_id in market_ids
        ]

        # --------------------------------------------------
        # Build semantic observation
        # --------------------------------------------------

        return AgentObservation(
            agent_name=agent.config.agent_name,
            observation_type="transactional_agent",
            observation_parameters={
                "agent_state": agent.state.agent_state,
                "accounts": agent_accounts,
                "institutions": institutions,
                "markets": markets,
                "environment_step": (environment.state.step),
            },
        )
