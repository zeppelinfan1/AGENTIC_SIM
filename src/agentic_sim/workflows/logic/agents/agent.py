from dataclasses import dataclass

from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentConfig,
    AgentState,
)
from agentic_sim.workflows.logic.agents.models.agent_observation_nn import (
    AgentObservationNN,
)


@dataclass
class Agent:
    """
    Runtime representation of an economic agent.
    """

    config: AgentConfig
    state: AgentState
    brain: AgentObservationNN
