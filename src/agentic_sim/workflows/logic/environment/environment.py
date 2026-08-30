from agentic_sim.workflows.logic.environment.state import (
    EnvironmentState,
    InstitutionState,
    MarketState,
)


class Environment:
    """
    Runtime representation of the simulated economic world.
    """

    def __init__(
        self,
        state: EnvironmentState,
        markets: list[MarketState],
        institutions: list[InstitutionState],
    ) -> None:
        self.state = state
        self.markets = markets
        self.institutions = institutions
