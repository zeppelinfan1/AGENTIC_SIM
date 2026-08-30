from dataclasses import dataclass, field


@dataclass
class EnvironmentState:
    """
    State of the environment.
    """

    step: int = 0


@dataclass
class MarketState:
    """
    State of an economic market visible to participating actors.
    """

    market_name: str
    market_type: str

    public_state: dict[str, float] = field(default_factory=dict)


@dataclass
class AccountState:
    """
    State of an account held by an agent at an institution.
    """

    account_id: str
    owner_agent_name: str
    institution_name: str

    balance: float = 0.0
    outstanding_debt: float = 0.0
    credit_limit: float = 0.0


@dataclass
class InstitutionState:
    """
    State of a financial institution.
    """

    institution_name: str

    available_liquidity: float = 0.0
    capital: float = 0.0

    accounts: list[AccountState] = field(default_factory=list)

    @property
    def total_deposits(self) -> float:
        return sum(max(account.balance, 0.0) for account in self.accounts)
