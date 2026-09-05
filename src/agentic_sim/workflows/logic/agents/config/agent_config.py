from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBrainConfig:
    """
    Configuration for an agent's observation neural network.
    """

    embedding_dim: int = 3
    encoder_hidden_dim: int = 512
    actor_hidden_dim: int = 128
    action_dim: int = 2
    dropout: float = 0.1

    contrastive_margin: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuration for an agent.
    """

    agent_id: int
    agent_name: str
    agent_type: str
    brain_config: AgentBrainConfig
    agent_parameters: dict


@dataclass
class AgentState:
    """
    State of an agent.
    """

    agent_name: str
    agent_state: dict


@dataclass(frozen=True)
class AgentAction:
    """
    Action taken by an agent.
    """

    agent_name: str
    action_type: str
    action_parameters: dict


@dataclass(frozen=True)
class AgentObservation:
    """
    Observation made by an agent.
    """

    agent_name: str
    observation_type: str
    observation_parameters: dict
