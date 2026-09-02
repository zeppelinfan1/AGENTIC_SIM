from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
    InstitutionState,
    MarketState,
)

from agentic_sim.workflows.logic.environment.visibility.resolver import (
    VisibilityResolver,
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

        self.visibility = VisibilityResolver(
            environment=self,
        )
