from abc import ABC, abstractmethod

from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.reward.models import (
    RewardContext,
    RewardResult,
    RewardSnapshot,
)


class RewardMechanism(ABC):
    """
    Base interface for an agent reward mechanism.

    A reward mechanism has two responsibilities:

        1. capture the objective state required for reward;
        2. evaluate the transition between two captured states.

    Concrete mechanisms decide what constitutes success.

    SimulationEngine should depend on this interface rather
    than on any particular reward philosophy.
    """

    @abstractmethod
    def capture_state(
        self,
        *,
        actor_id: int,
        environment: Environment,
    ) -> RewardSnapshot:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        *,
        context: RewardContext,
    ) -> RewardResult:
        raise NotImplementedError
