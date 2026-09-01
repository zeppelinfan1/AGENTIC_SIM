from agentic_sim.workflows.logic.environment.state import (
    AccountState,
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
        accounts: list[AccountState],
    ) -> None:
        self.state = state
        self.markets = markets
        self.institutions = institutions
        self.accounts = accounts
