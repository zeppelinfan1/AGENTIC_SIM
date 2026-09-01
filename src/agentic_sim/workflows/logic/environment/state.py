from dataclasses import dataclass, field


@dataclass
class EnvironmentState:
    """
    State of the environment.
    """

    step: int = 0


@dataclass
class AccountState:
    """
    State of an account held by an agent at an institution.
    """

    account_id: str
    agent_id: str
    institution_id: str

    balance: float = 0.0
    outstanding_debt: float = 0.0
    credit_limit: float = 0.0


@dataclass
class InstitutionState:
    """
    State of a financial institution.
    """

    institution_id: str
    institution_name: str
    market_id: str

    available_liquidity: float = 0.0
    capital: float = 0.0


@dataclass
class MarketState:
    """
    State of an economic market visible to participating actors.
    """

    market_id: str
    market_name: str
    market_type: str

    public_state: dict[str, float] = field(
        default_factory=dict,
    )
