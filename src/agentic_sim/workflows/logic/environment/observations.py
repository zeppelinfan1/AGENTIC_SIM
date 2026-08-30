from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)


class ObservationBuilder:
    """
    Builds an actor-specific view of the environment.
    """

    def build_agent_observation(
        self,
        agent: Agent,
        environment: Environment,
    ) -> AgentObservation:

        accounts = []

        for institution in environment.institutions:
            for account in institution.accounts:

                if account.owner_agent_name == agent.config.agent_name:
                    accounts.append(account)

        observation_parameters = {
            "agent_state": agent.state.agent_state,
            "accounts": accounts,
            "markets": environment.markets,
            "step": environment.state.step,
        }

        return AgentObservation(
            agent_name=agent.config.agent_name,
            observation_type="transactional_agent",
            observation_parameters=observation_parameters,
        )
