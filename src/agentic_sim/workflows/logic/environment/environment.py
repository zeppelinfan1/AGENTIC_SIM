from agentic_sim.workflows.logic.environment.config.environment_state import (
    EnvironmentState,
    InstitutionState,
    MarketState,
)


class Environment:
    """
    Runtime representation of the simulated economic environment.
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
