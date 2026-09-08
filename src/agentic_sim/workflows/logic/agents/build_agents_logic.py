from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentBrainConfig,
    AgentConfig,
    AgentState,
)
from agentic_sim.workflows.logic.agents.models.agent_brain import (
    AgentBrain,
)


class BuildAgents:

    def __init__(
        self,
        num_agents: int,
        spark,
        logger,
    ) -> None:

        self.num_agents = num_agents
        self.spark = spark
        self.logger = logger

        self.brain_config = AgentBrainConfig(
            embedding_dim=3,
            action_embedding_dim=16,
            value_hidden_dim=128,
            action_dim=2,
        )

        self.agent_type = "transactional"

        self.initial_agent_state = {
            "income": 100.0,
            "resources": 0.0,
            "time": 0.0,
        }

    def _build_agent(
        self,
        agent_number: int,
    ) -> Agent:

        agent_name = f"Agent {agent_number}"

        agent_config = AgentConfig(
            agent_id=agent_number,
            agent_name=agent_name,
            agent_type=self.agent_type,
            brain_config=self.brain_config,
            agent_parameters={},
        )

        agent_state = AgentState(
            agent_name=agent_name,
            agent_state=(self.initial_agent_state.copy()),
        )

        brain = AgentBrain(
            embedding_dim=(self.brain_config.embedding_dim),
            encoder_hidden_dim=(self.brain_config.encoder_hidden_dim),
            action_embedding_dim=(self.brain_config.action_embedding_dim),
            action_encoder_hidden_dim=(self.brain_config.action_encoder_hidden_dim),
            value_hidden_dim=(self.brain_config.value_hidden_dim),
            actor_hidden_dim=(self.brain_config.actor_hidden_dim),
            action_dim=(self.brain_config.action_dim),
            dropout=(self.brain_config.dropout),
        )

        return Agent(
            config=agent_config,
            state=agent_state,
            brain=brain,
        )

    def run(
        self,
    ) -> list[Agent]:

        self.logger.info(
            ("Starting agent building | " "num_agents=%s"),
            self.num_agents,
        )

        agents = [
            self._build_agent(agent_number)
            for agent_number in range(
                1,
                self.num_agents + 1,
            )
        ]

        self.logger.info(
            ("Agent building completed | " "agents_built=%s"),
            len(agents),
        )

        return agents
